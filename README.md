# Canopy

## Using Canopy for a Far-Field Solve

Canopy provides a parallel Fast Multipole Method (FMM) solver built on top of Kokkos and Cabana. The primary entry point is `Canopy::Solver`, defined in `src/Canopy_Solver.hpp`.

For detailed descriptions of the algorithms used in Canopy — including the
load-balancing approach and its interface with Zoltan 2 — see the [Algorithm
and Design Documentation](#algorithm-and-design-documentation) section below.

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
| `near_softening_factor` | Near-field softening floor: pairs closer than `factor · softening` use the softened near-field (P2P) instead of the unsoftened multipole far-field (M2L). `0` disables. | `4.0` |

The multipole far-field is built from the **unsoftened** `1/r` Laplace kernel, so
it is only accurate where the Plummer `softening` is negligible (separation
`R ≫ eps`). `near_softening_factor` widens the near field to cover everything
within `factor · eps`, so any pair where softening matters is handled by the
softened P2P kernel. Without it, a clustering system whose cells shrink below the
softening length (e.g. a vortex sheet at full roll-up) gets a spurious, far too
large far-field and blows up. Larger `factor` is more accurate (far-field
relative softening error `~ 1/(2·factor²)`, ≈3% at the default `4`) but widens
the near field, putting more pairs in the (more expensive) P2P path. Set `0` to
recover the pure geometric MAC (correct only when `softening` is small relative
to all M2L separations).

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

## Building and Running Tests

The unit tests live in [`tests/`](tests/) and are driven entirely by **CTest** —
there is no need to launch individual test binaries by hand.

### Configure and build

Enable the test build at configure time:

```bash
cmake -DCanopy_ENABLE_TESTING=ON [other args] ..
make -j                      # build everything, or
make -j Canopy_Test_MultiSolve_MPI_SERIAL   # build one target
```

Each test source is a header (`tests/tst<Name>.hpp`) compiled once per enabled
Kokkos backend, producing targets named `Canopy_Test_<Name>_[MPI_]<DEVICE>`
(e.g. `Canopy_Test_MultiSolve_MPI_SERIAL`, `Canopy_Test_Helpers_OPENMP`).

System-specific configure flags (compilers, the spack environment, and on some
machines the test launcher) are documented per system under
[`systems/<system>/claude.md`](systems/). Use the matching `run_cmake_<system>.sh`
wrapper as the canonical configure command.

**Faster iteration: build fewer backend variants.** Each test header is
recompiled once per enabled Kokkos backend, so on a multi-backend build (e.g.
SERIAL + OpenMP + HIP) every test compiles three times. While iterating on a
single backend, restrict the test build with `Canopy_TEST_DEVICES`:

```bash
cmake -DCanopy_ENABLE_TESTING=ON -DCanopy_TEST_DEVICES=SERIAL [other args] ..
```

This builds only the SERIAL test variants (the minimum test set is SERIAL), a
roughly N-fold reduction in test-compile time for N enabled backends. Leave it
empty (the default) to build every enabled backend for a full pre-ship run.
Incremental rebuilds are already accelerated by `ccache` (enabled via
`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` in the `run_cmake_<system>.sh` wrappers),
which caches unchanged translation units across rebuilds.

### Run with CTest

From the build directory:

```bash
ctest -N                                          # list registered tests, run nothing
ctest --output-on-failure                         # run the whole suite
ctest --output-on-failure -L regression -R MPI_SERIAL   # the required ship gate (see below)
ctest --output-on-failure -L unit                 # the diagnostic/component suite
ctest --output-on-failure -R MultiSolve           # run tests matching a regex
ctest -j 4 --output-on-failure                    # run up to 4 tests concurrently
```

Tests carry a CTest **label** describing their tier (`ctest -L <label>`):

- **`regression`** — the full-pipeline FMM solve (`MultiSolve`). This is the
  required gate: `ctest -L regression -R MPI_SERIAL` (SERIAL backend, ranks
  1–6) must pass before any change ships. `MultiSolve` composes the entire
  pipeline end-to-end, so if it passes the pipeline is correct.
- **`unit`** — utilities, math kernels, and individual FMM-phase/component
  tests (tree build, partition, up/down sweeps, P2P, communication plan, and
  the single-tree `SingleSolve`). The diagnostic layer: run these to localize
  *which* phase a regression came from, and when changing a specific component.

Labels combine with `-R` (backend/name regex) by AND, so
`-L regression -R MPI_SERIAL` selects only the SERIAL-backend regression tests.

MPI tests are registered at several rank counts. The rank list is controlled by
the `Canopy_TEST_MPI_RANKS` cache variable (default `1;2;3;4;5;6`); ranks
exceeding `MPIEXEC_MAX_NUMPROCS` are skipped at configure time. Non-MPI tests
(`Helpers`, `Laplace`) exercise no MPI functionality and run once, serially.
CTest launches each MPI test through CMake's `MPIEXEC_EXECUTABLE` — on a
scheduler-managed machine, run `ctest` from inside an allocation (the per-system
docs provide ready-made batch wrappers, e.g.
[`scripts/tuolumne/run_ctest_minset.flux`](scripts/tuolumne/run_ctest_minset.flux)
and [`scripts/dane/run_ctest_minset.slurm`](scripts/dane/run_ctest_minset.slurm)).

The project-wide minimum test set that must pass before any change ships is
defined in [`CLAUDE.md`](CLAUDE.md).

## Algorithm and Design Documentation

The algorithms used in Canopy — for example the load-balancing approach and how
it interfaces with the Zoltan 2 partitioner — are described in detail in
[`docs/design.md`](docs/design.md). That document is the authoritative record of
the library's algorithmic design decisions; consult it when you need to
understand *why* a component works the way it does rather than just its API.

## Dependencies and Build Notes

### Particle migration is 64-bit-safe (patched Cabana no longer required)

`TreePartitioner::migrate_particles` performs its own coalesced,
registration-bounded particle exchange (a `RegisteredBufferPool`-backed
pack/`MPI_Isend`/`MPI_Irecv`/unpack, mirroring the M2L/L2L
`coalesced_view_exchange` and the P2P ghost gather) rather than
`Cabana::migrate`. The MPI element type is one whole particle tuple
(`MPI_Type_contiguous` over `sizeof(tuple_type)` bytes) and the message count
is the **tuple count**, so a single peer's payload may exceed **2 GiB**
without overflowing MPI's signed-`int` count.

This sidesteps the historical issue where upstream `Cabana::Distributor`
computed the per-peer MPI message size with a signed 32-bit `int` byte count:
at ~10⁸ particles/rank a 56–60 byte tuple crossed `INT_MAX` (≈4.6 GB to a
single peer), the count silently overflowed, and migrated particle data was
truncated to garbage (correct count, corrupt positions; the next tree build
then collapsed). Because Canopy no longer routes migration through
`Cabana::migrate`, **the patched Cabana fork is no longer required** —
upstream `cabana@master` is sufficient.

The same change bounds peak concurrent NIC memory registrations to O(1) per
direction during a `Rebalance`, fixing the GTL `dreg_evict NO_SPACE` deadlock
seen on many-way migrations at scale.

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

## Future Optimizations

Tracked optimization opportunities that are not yet implemented. None of these
are correctness issues — they are performance/scalability refinements.

### Nonblocking-consensus peer discovery in particle migration

`TreePartitioner::migrate_particles` discovers, for each rank, which sources
will send it particles via a single `MPI_Alltoall` of `comm_size` ints (each
rank's per-destination send counts). This is simple and bounded, but it is an
`O(comm_size)` collective on every `Rebalance`. At very high rank counts the
Alltoall metadata cost grows with the job size even though the actual migration
is sparse (each rank exchanges with only a handful of peers).

Replace it with a sparse nonblocking-consensus exchange (the standard
`MPI_Issend` + `MPI_Ibarrier` "NBX" dynamic-sparse-data-exchange algorithm):
each rank posts non-blocking synchronous sends to only its real destination
peers, then enters an `MPI_Ibarrier` once all its sends have completed locally,
probing for incoming count messages until the barrier completes. This makes
peer discovery cost scale with the number of *actual* peers rather than
`comm_size`. Not needed at the current target scale (≈256 ranks); revisit if
Canopy runs at many thousands of ranks.

---

## Known Issues

Tracked defects to be addressed in a later session. These are not introduced by
current feature work — they reproduce on the pre-existing baseline.

### The leaf partition is not reproducible run-to-run above two ranks

`TreePartitioner::partition_leaves` uses the Zoltan2 `multijagged` algorithm,
which `src/Canopy_TreePartitioner.hpp:417-419` already documents as
non-deterministic. Computing the partition on rank 0 and broadcasting it makes
the assignment consistent across ranks *within* a run, but it is not reproducible
*across* runs: two runs of the same binary at the same commit produce different
leaf-to-rank assignments at every rank count from 3 to 6.

Reproduce with the golden harness, which measures it directly:

```bash
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL   # run twice, diff "[laplace-solve]"
```

`n_unique_ops` for one `(nprocs, rank)` moves by tens between runs — e.g.
`(3,0)` gave 1630 / 1605 / 1605 and `(6,3)` gave 974 / 947 / 973 over three
consecutive runs. `num_cells` is identical across runs at every rank count (80,
170, 316, 431, 500, 524), so the tree build is deterministic and it is cell
*ownership* that moves; the interaction lists, the M2L operator table and
`locals()` all follow it. np=1 and np=2 are stable, the multijagged cut being
trivial for one or two parts.

This is **pre-existing** — it is a property of the partitioner, untouched by the
golden-harness work that found it. It blocks any bit-for-bit comparison above two
ranks, which is why `tasks/abstract-solver-backend.md` T1 is marked **BLOCKED**.
A plausible but unverified mechanism is Zoltan2 MJ running on
`Kokkos::DefaultExecutionSpace`, which is HIP in this build even for the SERIAL
test binaries. To be triaged in a separate session: either make the partitioner
deterministic (a deterministic algorithm, or a seeded / host-serial MJ) or cache
and reuse a committed assignment.

### `LaplaceSolve` has no committed reference data, and its frozen configuration collapses

`Canopy_Test_LaplaceSolve_MPI_SERIAL` fails at every rank count with `cannot
open reference data file .../tests/data/laplace_solve_P6.txt`. The data is
deliberately not committed: running the harness showed that the configuration
it is frozen at drives a two-body collapse, and committing a baseline taken
from that state would be worse than having none.

At 600 particles with charges uniform on `[-1, 1]`, `softening = 0.0`,
`dt = 1.0e-4` and 50 steps, the closest opposite-charge pair free-falls to
contact at about step 15. The participants are ejected, the bounding box grows
from `[0.05, 0.95]` to roughly `[-20, 11]`, and with `max_depth = 6` the tree
cannot refine into what is left. By the 50th solve there are 29 cells, no pair
is MAC-admissible, and `n_unique_ops` is **0** at every rank count from 1 to 6
— the far field the harness exists to protect is never evaluated, and the
direct-sum deviation reads 1e-15 (machine precision) instead of the ~3e-6 a
working far field gives.

The harness itself is sound. At `num_steps = 1`, where the tree is healthy (95
cells, 604 realized operators at np=1), all three gates pass at ranks 1-6:
bit-for-bit artifacts reproduce across runs at np 1-2, the cross-rank deviation
is 1e-15 on the potential and 3e-13 on the gradient at np 2-6, and the
direct-sum deviation is 3.3e-6. Choosing a viable frozen configuration —
softening, one-signed charges, fewer steps or a smaller `dt` — is a design
decision left to a later session; see `tasks/abstract-solver-backend.md` T1 and
the second `## T1` section of its progress log.

Related: `LaplaceSolve.crossRankAgreement` **hangs at np=6** on the collapsed
tree (>14 min against 8-12 s at np 1-5). At `num_steps = 1` np=6 finishes in
7.4 s, so this is a property of the degenerate tree, not of the rank count.

### Six `MultiSolve` tests fail the `1e-8` multi-step check, and np=3 hangs

`ctest --output-on-failure -L regression -R MPI_SERIAL` does not currently pass.
At np=1 and np=2, six tests fail the multi-step position/velocity comparison at
`fmm_tolerance = 1e-8` (`tests/tstMultiSolve.hpp:542,546`) with measured relative
errors of 3e-7 to 9e-6: `MultiSolve.StableTree_Migrate`,
`IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`, `AutoMaintain`,
`AutoRebalance`, `M2L_BinEdge_Fallback`. The gate then hung at
`Canopy_Test_MultiSolve_MPI_SERIAL_np_3` and was killed at a 15-minute wall —
note that the np=3 hang below was previously seen only when `SingleSolve` ran in
the same `ctest` process, whereas this run was `MultiSolve` alone.

This is **pre-existing**: checking out `src/Canopy_DownwardSweep.hpp` at the
pre-golden-harness commit `a6c90de`, rebuilding and rerunning reproduces the
identical error values to every digit (`3.485035469067542e-07`,
`6.8419528791564039e-07`, `9.1947965989306709e-06`).

It supersedes part of the entry below, which was written when these tests passed:
`SolveFusedM2L.FP32_smokeTest` is no longer the suite's only failure, and it
*passed* at np=1 in these runs. Whether the six failures and the np=3 hang share
a cause with the partitioner non-determinism above has not been investigated.

### `SolveFusedM2L.FP32_smokeTest` fails at ≥ 2 ranks

In the `Canopy_Test_MultiSolve_MPI_SERIAL` suite, `SolveFusedM2L.FP32_smokeTest`
passes at 1 rank but fails at 2–6 ranks: the FP32 max relative gradient error is
≈ 0.277, well over the test's `5e-2` budget (`tstMultiSolve.hpp:1104`). All
other tests in the suite pass at 1–6 ranks, including the migrate/rebalance
paths (`MultiSolve.StableTree_Migrate`, `AutoRebalance`,
`IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`).

This is a **pre-existing** failure, not a regression from the
registration-coalesced migration work (issue #22): checking out the parent
commit and rebuilding reproduces the identical error to FP32 noise on both
Tuolumne (base `0.27730871` vs current `0.27730911`) and Dane (base/current
`0.27730867`). It reproduces on both platforms, so it is not machine-specific.

The magnitude (≈0.277, not marginally over budget) and the rank-count
dependence point at a multi-rank FP32 accuracy problem in the fused-M2L solve
(e.g. order-dependent reductions or a genuinely too-tight FP32 budget for the
multi-rank path), independent of particle migration — which is verified
bit-exact by `TreePartitioner.testCoalescedMigrateIntegrity`. To be triaged in a
separate session: determine whether the fix is a corrected FP32 accumulation or
a re-justified error budget for the multi-rank FP32 case.

### `SingleSolve` fails at np=4 and deadlocks the suite when run with other solves

`SingleSolve` is labeled `unit`, **not** `regression`, for two reasons found
when it was re-enabled in the CTest suite:

1. **Accuracy failure at np=4.** `SingleSolve.PotentialNComps3` and
   `SingleSolve.PotentialAndGradientNComps3` fail at exactly 4 ranks with
   `max_pot_rel_err = 0.00207 vs tol 0.001` (~2× over budget,
   `tstSingleSolve.hpp:365`). Ranks 1, 2, 3, 5, 6 pass. This is *not* the FP32
   issue above. The np=4-only signature suggests a partition/decomposition edge
   case specific to that rank count.

2. **State leak that hangs a later test.** When `SingleSolve` runs in the same
   `ctest` process as `MultiSolve` (i.e. `ctest -L regression` before
   `SingleSolve` was demoted), the suite deadlocks several tests later at
   `Canopy_Test_MultiSolve_MPI_SERIAL_np_3`, hanging until the scheduler wall
   kills it. The deadlock does **not** occur when `MultiSolve` runs alone, and
   was isolated to the Canopy binaries (bare back-to-back `flux run`
   invocations, with and without `--exclusive`, do not hang — 14/14 clean). The
   working hypothesis is that `SingleSolve`'s crash (the np=4 failure, or its
   MPI/HIP teardown) leaves orphaned MPI ranks or GPU state that stalls a later
   test's `Kokkos::initialize` — the SERIAL test binaries all bring up the HIP
   backend.

Because of (2), `SingleSolve` is excluded from the regression gate so the gate
(`MultiSolve` only) runs cleanly. To be triaged in a separate session: fix the
np=4 accuracy bug, and ensure a failed solve tears down its MPI/GPU state so it
cannot poison subsequent tests in the same `ctest` run.

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
