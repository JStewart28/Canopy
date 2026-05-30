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

```cpp
Canopy::Solver<...> solver(
    MPI_Comm comm,           // MPI communicator
    int      ncrit,          // max particles per leaf cell
    int      max_depth,      // maximum octree depth
    double   tree_tolerance, // admissibility criterion for M2L interactions
    int      replication_depth, // depth to which ghost cells are replicated
    double   imbalance_tolerance = 0.05 // load-balance threshold
);
```

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

Canopy::Solver<MemSpace, ExecSpace> solver(MPI_COMM_WORLD,
    /*ncrit=*/32, /*max_depth=*/10,
    /*tree_tolerance=*/0.4, /*replication_depth=*/1);

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
