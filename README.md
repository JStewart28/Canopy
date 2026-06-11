# Canopy

## Using Canopy for a Far-Field Solve

Canopy provides a parallel Fast Multipole Method (FMM) solver built on top of Kokkos and Cabana. The primary entry point is `Canopy::Solver`, defined in `src/Canopy_Solver.hpp`.

### Template Parameters

```cpp
Canopy::Solver<MemorySpace, ExecutionSpace, Scalar, P_ORDER, NComps>
```

| Parameter | Description | Default |
|---|---|---|
| `MemorySpace` | Kokkos memory space (e.g. `Kokkos::HostSpace`, `Kokkos::CudaSpace`) | — |
| `ExecutionSpace` | Kokkos execution space (e.g. `Kokkos::OpenMP`, `Kokkos::Cuda`) | — |
| `Scalar` | Floating-point type for field values | `double` |
| `P_ORDER` | Multipole expansion order (higher = more accurate, more expensive) | `8` |
| `NComps` | Number of simultaneous charge components | `1` |

### Constructor

The solver takes an `MPI_Comm` plus an `FmmConfig` struct that holds every
FMM-pipeline knob. The communicator stays a positional argument because it
is program context, not FMM behavior.

```cpp
Canopy::Solver<...> solver( MPI_Comm comm, const Canopy::FmmConfig& cfg );
```

`FmmConfig` is defined in `src/Canopy_Solver.hpp`:

| Field | Description | Default |
|---|---|---|
| `ncrit` | Max particles per leaf cell | — |
| `max_depth` | Maximum octree depth | — |
| `xmin_tol`, `xmax_tol` | Padding fractions on the low / high `x` face of the root bounding box | `0.0` |
| `ymin_tol`, `ymax_tol` | Padding fractions on the low / high `y` face | `0.0` |
| `zmin_tol`, `zmax_tol` | Padding fractions on the low / high `z` face | `0.0` |
| `ncrit_tol` | Admissibility tolerance on `ncrit` during tree adaptation | `0.1` |
| `replication_depth` | Depth to which ghost cells are replicated | `1` |
| `imbalance_tolerance` | Load-balance threshold | `0.05` |
| `mac_theta` | Multipole acceptance criterion | `0.5` |
| `softening` | Plummer softening length; `< 0` selects auto-softening from the inter-particle spacing | `-1.0` |

The six bounding-box tolerances are per-face and may be set independently,
e.g. to pad only the outflow boundary of an asymmetric domain. For an
isotropic domain set all six to the same value.

### Typical Usage Pattern

Particles are stored in a Cabana `AoSoA`. Two slices are required: one for 3D positions and one for scalar charges. The template integer arguments `PositionIdx` and `ChargeIdx` identify which AoSoA field holds each quantity.

**1. One-time setup** (call once before the first solve):

```cpp
// particles: Cabana AoSoA with positions at field 0, charges at field 1
// num_local: number of locally-owned particles before any migration
solver.setup<0, 1>(particles, num_local);
```

`setup()` builds the octree, partitions it across MPI ranks, migrates particles to their owning ranks, and prepares all FMM communication structures. After this call, `solver.num_local_particles()` reflects the post-migration local count.

**2. Solve** (call once per timestep):

```cpp
bool compute_gradient = true; // also evaluate gradient of the potential
solver.solve<0, 1>(particles, compute_gradient);

// Retrieve results (Kokkos views, indexed by local particle)
auto phi  = solver.potential(); // shape: (num_local,)
auto grad = solver.gradient();  // shape: (num_local, 3) — only valid if compute_gradient == true
```

`solve()` runs the full FMM pipeline: P2M → M2M → M2L → L2L → L2P → P2P. Output views are zeroed before each call, so results are not accumulated across timesteps.

**3. Between timesteps — maintaining the solver**

After particles move, the solver must be updated before the next `solve()`. Three options are available, in increasing cost:

| Method | When to use |
|---|---|
| `solver.migrate<0>(particles)` | Particles moved but the global octree cell structure is unchanged |
| `solver.rebalance<0>(particles)` | Tree topology changed (cells refined/coarsened) but all particles remain inside the original bounding box |
| `solver.rebuild<0, 1>(particles)` | Particles escaped the bounding box — full reconstruction required |

If you are unsure which applies, use `auto_maintain()`, which selects the cheapest valid path automatically and returns a `MaintenanceAction` enum indicating which path was taken:

```cpp
auto action = solver.auto_maintain<0, 1>(particles);
// action is one of: MaintenanceAction::Migrate, Rebalance, or Rebuild
```

### Minimal End-to-End Example

```cpp
using MemSpace = Kokkos::HostSpace;
using ExecSpace = Kokkos::Serial;

Canopy::FmmConfig cfg;
cfg.ncrit = 32;
cfg.max_depth = 10;
cfg.xmin_tol = cfg.xmax_tol = 0.4;
cfg.ymin_tol = cfg.ymax_tol = 0.4;
cfg.zmin_tol = cfg.zmax_tol = 0.4;
cfg.replication_depth = 1;

Canopy::Solver<MemSpace, ExecSpace> solver(MPI_COMM_WORLD, cfg);

// ... populate particles AoSoA ...

solver.setup<0, 1>(particles, num_local);

for (int step = 0; step < nsteps; ++step)
{
    solver.solve<0, 1>(particles, /*compute_gradient=*/false);

    auto phi = solver.potential(); // use phi in your time integrator

    // ... advance particle positions ...

    solver.auto_maintain<0, 1>(particles);
}
```

---

## Dependencies and Build Notes

### Patched Cabana required for large per-rank particle counts

Canopy migrates particles between MPI ranks with `Cabana::migrate` (a
`Cabana::Distributor` exchange) inside `TreePartitioner::migrate_particles`.
Upstream Cabana computes the per-peer MPI message size with a signed 32-bit
`int` byte count. When a single peer's particle payload exceeds **2 GiB**
(`INT_MAX` bytes) the count silently overflows: the transfer is truncated, the
receiving ranks keep the correct particle *count* but receive **garbage particle
data**, and the next tree build collapses (most particles land in one
max-depth leaf). At ~10⁸ particles/rank a 56–60 byte particle tuple crosses this
threshold (≈4.6 GB to a single peer), so the bug only appears at large scale —
small runs and the unit tests never hit it.

**You must build against a patched Cabana that uses 64-bit byte counts (or
chunks the transfer below `INT_MAX`) in the Distributor.** Use this fork:

> **https://github.com/JStewart28/Cabana**

Point your Spack environment (or CMake `Cabana_DIR`) at this fork rather than
upstream `cabana@master`. Without the patch, `gravity_solve` and any workload
that triggers a `Rebuild`/`migrate` at ≳2 GB/peer will corrupt particle
positions at scale (it will still pass the small-scale serial tests, so verify
on a large multi-rank case).

### P2P intra-leaf kernel: particle-centric, no atomics (MI300A APU)

The intra-leaf phase of P2P (interactions between particles in the same
leaf) is implemented as a flat, particle-centric kernel: each thread `pi`
iterates over the other particles in its own leaf and writes its own
`potential_out(pi, …)` / `gradient_out(pi, …)` slots directly. This costs
**2× the FLOPs** of a Newton's-3rd-law pair scheme (we compute `(i, j)`
and `(j, i)` separately) but does **no atomic writes**.

This design is a deliberate workaround for an APU-specific hang
observed during bring-up on AMD MI300A (CDNA3, unified CPU/GPU memory).

**Symptom.** With the previous TeamPolicy(num_leaves, 256) +
`Kokkos::atomic_add` pair-based kernel on MI300A:
- `Kokkos::parallel_for` returned to the host successfully.
- The trailing `Kokkos::fence()` never returned.
- I.e., the kernel was launched but some wavefronts on the device were
  stuck indefinitely, with no progress on the fence.

**Why.** Multiple wavefronts in a team racing `atomic_add` on adjacent
slots of `potential_out` / `gradient_out` — both managed-memory views
on a unified CPU/GPU coherence fabric — can fail to make forward
progress under heavy contention. AMD HSA has documented hangs in this
pattern; the failure does not reproduce on discrete-memory GPUs
(verified on an NVIDIA RTX 3500 Ti with the same code).

**Fix.** Restructure the intra-leaf kernel to mirror the inter-leaf
kernel: one thread per local particle, single-writer per output slot,
zero atomics. The cost is the 2× FLOP factor noted above; the win is
that the kernel completes deterministically on MI300A and the device
fence returns immediately after the kernel.

If you port Canopy to a hardware target where atomic contention on
unified memory is not a hazard (e.g. any discrete-memory GPU), the
prior Newton's-3rd-law pair kernel would be ~2× faster for the
intra-leaf phase and is preserved in the git history for reference.

### Known limitation: bounding box is not outlier-resistant (low priority)

`TreeBuilder::compute_global_bounding_box` takes a raw global min/max over all
particle positions. If a handful of particles escape far from the bulk (e.g.
close encounters under very small softening, or many integration steps), the
root box inflates and the finest octree cell (`width / 2^max_depth`) can become
large enough that a dense cluster collapses into a single max-depth leaf. Because
the near-field P2P kernel is O(N_leaf²) per particle, one oversized leaf makes a
solve effectively hang. This is not currently triggered at the tested parameters
(softening `0.001`), but a more robust / outlier-resistant bounding box (or
explicit handling of escaped particles) would harden the solver against it.

---

### Resources used:
1. [Fast multipole info](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/2012_fmm_encyclopedia.pdf)

2. [Lecture 2](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Lectures/lecture02.pdf)

3. [FMM for Vortical Flows](https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content)

4. [1987_greengard_dissertation](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/1987_greengard_dissertation.pdf)

5. [CSCAMM Lecture](https://home.cscamm.umd.edu/programs/fam04/dg_lecture6.pdf)

6. [1999_cheng](https://www.sciencedirect.com/science/article/pii/S0021999199963556)

7. [Rankin, WT: Efficient parallel implementations of multipole based N-body algorithms](https://www.proquest.com/dissertations-theses/efficient-parallel-implementations-multipole/docview/304504480/se-2?accountid=14613)

8. [ExaFmm] (https://github.com/exafmm/exafmm)
