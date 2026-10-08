# Canopy

## Using Canopy for a Far-Field Solve

Canopy provides a parallel Fast Multipole Method (FMM) solver built on top of Kokkos and Cabana. The primary entry point is `Canopy::Solver`, defined in `src/Canopy_Solver.hpp`.

For detailed descriptions of the algorithms used in Canopy — including the
load-balancing approach, a ParMETIS graph partition of the octree's cells — see
the [Algorithm
and Design Documentation](#algorithm-and-design-documentation) section below.

### Template Parameters

```cpp
Canopy::Solver<MemorySpace, ExecutionSpace, Scalar, P_ORDER, NComps, FarField>
```

| Parameter | Description | Default |
|---|---|---|
| `MemorySpace` | Kokkos memory space (e.g. `Kokkos::HostSpace`, `Kokkos::CudaSpace`) | — |
| `ExecutionSpace` | Kokkos execution space (e.g. `Kokkos::OpenMP`, `Kokkos::Cuda`) | — |
| `Scalar` | Floating-point type for field values | `double` |
| `P_ORDER` | The far field's order knob (higher = more accurate, more expensive). $P$ for a solid-harmonic basis, $p$ for Taylor, $n$ for Chebyshev — different quantities, same slot | `8` |
| `NComps` | Number of simultaneous charge components | `1` |
| `FarField` | Far-field basis, as a template taking `<Scalar, Order, NComps>` | `LaplaceKernel` |

`FarField` selects a **basis-plus-kernel composition**, not a bare kernel: the
type named there owns both the expansion the tree carries and the potential
those coefficients represent, and supplies every far-field operator (P2M, M2M,
M2L, L2L, L2P) together with the auxiliary tables they need. The default,
`Canopy::LaplaceKernel`, is the solid-harmonic $1/r$ composition; omitting the
argument is exactly today's solver. `createSolver` takes the same parameter in
the same position.

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
| `max_depth` | Maximum octree depth, at most 19 (the Morton key's limit; 20 or more throws). Each `build()` whose depth limit, rather than `ncrit`, ended a leaf prints one rank-0 `[Canopy] WARNING` line with the count of `max_depth` leaves holding more than `ncrit` particles | — |
| `xmin_tol`, `xmax_tol` | Padding fractions on the low / high `x` face of the root bounding box | `0.0` |
| `ymin_tol`, `ymax_tol` | Padding fractions on the low / high `y` face | `0.0` |
| `zmin_tol`, `zmax_tol` | Padding fractions on the low / high `z` face | `0.0` |
| `ncrit_tol` | Admissibility tolerance on `ncrit` during tree adaptation | `0.1` |
| `replication_depth` | Depth to which ghost cells are replicated | `1` |
| `imbalance_tolerance` | Load-balance threshold | `0.05` |
| `mac_theta` | Multipole acceptance criterion | `0.5` |
| `softening` | Plummer softening length; `< 0` selects auto-softening from the inter-particle spacing | `-1.0` |
| `near_softening_factor` | Near-field softening floor: pairs closer than `factor · softening` use the softened near-field (P2P) instead of the unsoftened multipole far-field (M2L). `0` disables. | `4.0` |
| `m2l_op_table_byte_budget` | Per-rank memory budget, in bytes, for the hashed M2L operator table. Pairs beyond the cap it implies fall back to the per-pair M2L translation | `2 GB` |
| `quantize_root_half_width` | Round the root cell's half-width up to the next power of two at every tree build, keeping its centre. A rebuild whose bounding box drifts within one octave then keeps the M2L operator cache of a level-keyed basis (`CartesianTaylorBasis`) instead of rebuilding it. Moves the cell set for every basis: the root cell is up to 2x wider than the bounding box, which can add one level at the top of the tree. | `false` |
| `tree_balance_max_level_delta` | Balance the tree at the end of every tree build: refine any leaf more than this many levels shallower than a leaf it touches (by a face, an edge or a corner), to a fixed point. In levels, `>= 1`; `1` is the 2:1 balance, and a value below `1` throws. Refinement creates only occupied children, so every cell stays occupied. Moves the cell set for every basis, at a cost in cells that depends on the distribution (1.10-1.28x on the two-scale test fixture at `1`). Off by default because it does not pay there: at `1` it also raises the M2L pairs refused to the per-pair fallback 1.6-3.5x (362-556 more per run), each about 12x a table pair's cost on SERIAL and 52-109x on HIP. `TreeBuilder::update()` does not balance. | `TREE_BALANCE_OFF` (off) |

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

`m2l_op_table_byte_budget` bounds the hashed M2L operator table, which holds one
dense operator column per distinct translation key. The cap that actually binds
is the smaller of this budget's worth of columns and the sweep's internal count
cap of 32768, so at the 2 GB default the count cap binds at every supported
order and lowering the budget is the only way to make it the binding constraint.
A column costs `num_coeffs_per_cell · m2l_num_src_coeffs · sizeof(coeff_type)` —
21952 B at `P_ORDER = 6` and 58320 B at `P_ORDER = 8` in double precision. Pairs
beyond the cap are refused a column and evaluated by the per-pair M2L
translation instead, which is the same mathematics evaluated pair by pair; they
are counted by `DownwardSweep::total_fallback_pair_count()`.

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

This builds only the SERIAL test variants, a roughly N-fold reduction in
test-compile time for N enabled backends. Leave it empty (the default) to build
every enabled backend.
Incremental rebuilds are already accelerated by `ccache` (enabled via
`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` in the `run_cmake_<system>.sh` wrappers),
which caches unchanged translation units across rebuilds.

### Run with CTest

From the build directory:

```bash
ctest -N                                          # list registered tests, run nothing
ctest --output-on-failure                         # run the whole suite
ctest --output-on-failure -L regression          # the full-pipeline solve (MultiSolve)
ctest --output-on-failure -L unit                 # the diagnostic/component suite
ctest --output-on-failure -R MultiSolve           # run tests matching a regex
ctest -j 4 --output-on-failure                    # run up to 4 tests concurrently
```

Tests carry a CTest **label** describing their tier (`ctest -L <label>`):

- **`regression`** — the full-pipeline FMM solve (`MultiSolve`), which
  composes the entire pipeline end-to-end.
- **`unit`** — utilities, math kernels, and individual FMM-phase/component
  tests (tree build, partition, up/down sweeps, P2P, communication plan, and
  the single-tree `SingleSolve`). The diagnostic layer: run these to localize
  *which* phase a regression came from, and when changing a specific component.

Labels combine with `-R` (backend/name regex) by AND, so
`-L regression -R MPI_SERIAL` selects only the SERIAL-backend regression tests.

MPI tests are registered at several rank counts. The rank list is controlled by
the `Canopy_TEST_MPI_RANKS` cache variable (default `1;2;3;4;5;6`); ranks
exceeding `MPIEXEC_MAX_NUMPROCS` are skipped at configure time. Two optional
per-backend overrides, unset by default, apply to one backend's MPI tests:
`Canopy_TEST_MPI_RANKS_<DEVICE>` replaces the rank list and
`Canopy_TEST_MPIEXEC_PREFLAGS_<DEVICE>` replaces `MPIEXEC_PREFLAGS`. On
Tuolumne, `run_cmake_tuolumne.sh` sets both for `HIP`, so HIP MPI tests register
at np 1-4 with one APU per rank (`--gpus-per-task=1`); a node has four APUs. Non-MPI tests
(`Helpers`, `LaplaceKernel`) exercise no MPI functionality and run once,
serially.
CTest launches each MPI test through CMake's `MPIEXEC_EXECUTABLE` — on a
scheduler-managed machine, run `ctest` from inside an allocation (the per-system
docs provide ready-made batch wrappers, e.g.
[`scripts/tuolumne/run_ctest_minset.flux`](scripts/tuolumne/run_ctest_minset.flux)
and [`scripts/dane/run_ctest_minset.slurm`](scripts/dane/run_ctest_minset.slurm)).

## Algorithm and Design Documentation

The algorithms used in Canopy — for example the load-balancing approach, a
ParMETIS graph partition of the octree's cells — are described in detail in
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
large enough that a dense cluster collapses into a single max-depth leaf (each
such build prints a `[Canopy] WARNING` line counting those leaves). Because
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

### A precomputed term table for the Cartesian-Taylor derivative ladder

`Canopy::CartesianTaylor::derivative_ladder`
(`src/Canopy_CartesianTaylorBasis.hpp`) re-derives its entire index arithmetic
on every call. For each slot it walks `inverse_slot` once to recover the
multi-index, and then calls `slot` up to seven times — once for the leading
`-r_i b_k` term and once for each surviving term of the two three-way sums — so
nothing about the traversal is reused between invocations. In the M2L that call
is per box pair, which is where the cost would actually be paid.

A table of `(m, i, k, term slots)` — the target slot, the direction the
recurrence steps in, the base multi-index, and the flat slots of the terms —
built once on host and carried in the basis's `aux_tables_type` would remove
all of it, leaving the ladder a straight-line weighted sum over a precomputed
index list.

This is not a correctness issue: the recomputed indices are exact and the
values are the same either way. **As of T3 the call site exists** —
`CartesianTaylorBasis::m2l_operator_block` evaluates one ladder per key in
`build_m2l_operators` and one per pair in `m2l_translate` — so the opportunity
is live rather than hypothetical, and it was still left unbuilt because
nothing has measured it. Note that a table here is an *indirection*, and
**R3** in
[tasks/cartesian-taylor-basis.md](tasks/cartesian-taylor-basis.md) records a
measured +18% M2L regression from a comparable one, with two candidate
micro-causes tested and excluded. So the table must be measured against the
recompute it replaces rather than assumed faster.

### The M2L operator cache empties on every build for a level-keyed basis

**Addressed, opt-in, by `FmmConfig::quantize_root_half_width`** (B2 of
[tasks/tree-opt.md](tasks/tree-opt.md)). With it on, the root half-width is a
power of two, bit-identical across every rebuild whose box stays inside one
octave, so `set_root_half_width` does nothing and the cache survives. On the
`CartesianTaylorSolve` drift trajectory below, summed over np 1-6, builds 2-4
rebuilt 5.3 % (θ 0.5) and 7.1 % (θ 0.3) of the admitted columns, only the keys
the moving tree realized for the first time, against 100 % with it off
([tasks/tree-opt-progress-log.md](tasks/tree-opt-progress-log.md) §B2). It
stays off by default because it moves the cell set for every basis. The
measurement that motivated it follows.

`DownwardSweep::set_root_half_width` clears the **entire** persistent M2L
operator cache whenever the root half-width changes, and does so only for a
basis declaring `key_needs_level = true`
(`src/Canopy_DownwardSweep.hpp:406-416`). `Solver::_push_root_half_width`
(`src/Canopy_Solver.hpp:756-770`) pushes the current box in before every
`_downward.setup()`, and the comparison is an exact double. On any moving
particle distribution the bounding box is recomputed from the particles and so
drifts at every rebuild — which means the cache empties at every rebuild, for
exactly the workload it exists to serve.

**Measured, and it is total.** T5 of
[tasks/cartesian-taylor-basis.md](tasks/cartesian-taylor-basis.md) ran a
four-solve `CartesianTaylorBasis` solve at np 1-6 and two admissibilities: at
every one of 336 builds the per-step increment in
`m2l_op_keys_built_count()` equalled `m2l_op_cache_size()` after that build
*exactly*, so **zero** cached operators survived a rebuild and each rank
constructed $3.9\times$ its cached key count over the run (per-rank figures in
[tasks/cartesian-taylor-basis-progress-log.md](tasks/cartesian-taylor-basis-progress-log.md)
§T5). The same counters on `LaplaceKernel`, which is level-blind, measure zero
keys rebuilt across a topology change (§T9 of
[tasks/abstract-solver-backend-progress-log.md](tasks/abstract-solver-backend-progress-log.md)).

Two candidate fixes, and one of them is already ruled out by measurement:

- **Key the cache on the physical operator scale** rather than on the level —
  i.e. on the width the operator was actually built at, so a box that returns
  to a previous width hits the cache. Untested.
- **Tolerate a root-width change that is an exact power of two**, which would
  leave the level-indexed widths unchanged. **Measured not to fire:** the
  realized drift is a smooth monotone contraction of 0.18% to 0.40% per build,
  which is what recomputing a bounding box from moving particles produces, and
  nowhere near a power of two.

This is not a correctness issue, and the clearing is not gratuitous — it is
what stops a level-keyed basis from evaluating operators built at a width it no
longer has. Any fix has to preserve that (risk **R5** in the same document) and
has $3.9\times$ to beat, which is to say it must turn four operator
constructions into one. Nothing has **timed** the rebuild: the counters above
are the whole of what is measured. Whoever prices it should read
`TIMER_ILIST_S4_OP_TABLE_BUILD` in `build-tuolumne-prof/` — the host operator
build lies outside `run_m2l_all`'s scope, so the `M2L kernel (all depths)` row
is the wrong number for this.

### Keep the M2L operator cache across an octave crossing of the root width

With `quantize_root_half_width` on, a box that crosses an octave changes the
quantized root half-width by a factor of two and the cache of a level-keyed
basis is emptied exactly as before. It need not be: the column for depth $d$
at root width $W$ is the column for depth $d + 1$ at root width $2W$, so shifting
every cached key's `max_d` by the change in the quantized exponent, instead of
clearing, would keep the whole cache across the crossing too. Not done in B2
of [tasks/tree-opt.md](tasks/tree-opt.md) because a wrong shift reuses every
column at the wrong width and presents as a plausible but wrong field (risk
**R4** there). A key shifted past depth 0 or `max_depth` has no counterpart and
would have to be dropped. Whoever implements it should rerun B2's failure
direction, which is the check that catches a column used at the wrong width.

### Partition cost on small trees

`TreePartitioner::partition_cells` sorts the full vertex list on every rank
and builds its ParMETIS graph with `unordered_map` lookups (27 per local
vertex), then runs a distributed ParMETIS solve. On `MultiSolve`'s trees of a
few hundred cells the mean `TIMER_PARTITION` is 3-20 ms at SERIAL np 2-6,
against 0.5-1.0 ms for the multijagged partitioner it replaced; on HIP it is
within 2x of multijagged-on-HIP (flux jobs `f3cZDMUWgsYj`, `f3cZDMd8JhWw`;
`tasks/fix-hang-rebalance-progress-log.md`, H2 partitioner arm). Candidates: a
size threshold below which one rank partitions serially with METIS (the vertex
list is already replicated, so no data moves), and a sorted-key lookup in
place of the per-edge hash map. Not measured at production scale, where the
solve rather than the setup is expected to dominate.

### Particle balance on near-degenerate trees

About 40% of `MultiSolve`'s partitions miss the particle-balance tolerance,
with max/mean up to 5.8 at np 6 and up to four ranks owning no leaf
(`[Canopy diag] partition` lines, flux job `f3cZDMUWgsYj`). Every such
partition is an 11-29-vertex graph with no band constraint; max/mean of exactly
2.0 at np 2 and 4.0 at np 4 puts every particle on one rank, which fits one
leaf holding nearly all of them (the ejected-particle boxes of
`LargeMotion_Rebuild` and `AutoRebalance`). No partition of cells can split
such a leaf, so the remedy is in the tree (e.g. bounding the box against an
outlier, or a deeper `max_depth` where one leaf is crowded), not in the
partitioner. Measure the largest leaf's share of the particles first: that
has not been done, so the single-leaf cause is unverified.

---

### Device-side neighbour search in the tree-balancing pass

With `tree_balance_max_level_delta` set, `TreeBuilder::balance` finds each
leaf's touching neighbours on the host, every pass. For each leaf it looks at
the 26 cells beside it at the leaf's depth. Each lookup walks up to `depth`
ancestors through `_cell_lookup`, so a pass costs `O(leaves · 26 · max_depth)`
hash lookups. This is the same on every rank, because the cell list is
replicated. It is negligible on the test fixtures (about 300 cells, 2-3
passes), but it grows with the global leaf count and runs serially on every
rank. Options: move the search to the device (a sorted key array with a binary
search per neighbour), or restrict later passes to the neighbourhoods of the
leaves the previous pass refined. Only a refined leaf's surroundings can
become newly unbalanced. Not needed while balancing is off by default;
revisit if A3 turns it on for large trees.

### Batch the per-pair M2L fallback on the device

Pairs refused an operator column (range guard or column cap) go through
`DownwardSweep::run_m2l_fallback_at_depth`, which launches one team per pair
running `m2l_translate`. Per pair, `CartesianTaylorBasis<double, 3>` at θ 0.3
measured (tree-opt C1, `[c1-m2l]`):

- HIP: 52-109x the fused GEMM path (about 6.3e-7 s against 5.8e-9 to 1.2e-8 s).
  The ratio is largest at np 1, where the GEMM batch is biggest.
- SERIAL: 10.4-12.2x.

So every refused pair costs about 55 GEMM pairs at HIP np 4. Two options:

- Group fallback pairs that share a canonical key, or a depth difference, and
  build their operators into a transient table evaluated by the fused kernel.
- Have one team process many pairs of a target, accumulating into the
  target's local once.

Either would shrink the cost of every refusal that chain A of
`tasks/tree-opt.md` does not remove. A2 measured that balancing raises the
number of refusals.

## Known Issues

Tracked defects to be addressed in a later session. These are not introduced by
current feature work — they reproduce on the pre-existing baseline.

### HIP solves are not bit-reproducible run to run; the cell partition is

`TreePartitioner` partitions cells with ParMETIS on the host
(`docs/design.md`, "Load Balancing"), and that partition reproduces: two
passes of `MultiSolve` at SERIAL np 2-6 and HIP np 2-4 give the identical
ownership-map hash on every partition and every refresh (the profiling-build
`[Canopy diag] partition` and `refresh_ownership` lines; flux jobs
`f3cZDMUWgsYj` and `f3cZDMd8JhWw`, `fix-hang-rebalance` H2). On SERIAL the
`[multisolve-dev]` lines are identical between the passes at every np. On a HIP
`ExecutionSpace` they are not, even at np 1 where nothing is partitioned: at
np 2-4 no case is identical between two passes, differing by relative 6e-12 to
2e-4, because device reductions accumulate in a run-dependent order (H0c
measured the same at np 1-4 under the previous partitioner).
`LaplaceSolve.bitForBitArtifacts` therefore stays on `Kokkos::Serial`, and at
np 1-2 because the committed reference holds only the (1,0), (2,0) and (2,1)
records (`tests/tstLaplaceSolve.hpp:1214-1222`); `crossRankAgreement` and
`matchesDirectSum` carry ranks 3-6.

Reproduce with `scripts/tuolumne/run_ctest_h2.flux prepro hip` and compare the
two passes' `[multisolve-dev]` lines.

### `DownwardSweep.testIdempotentExecution` compares `locals()` with itself on SERIAL

`tests/tstDownwardSweep.hpp:365-379` snapshots `downward.locals()` after each
of two `execute()` calls with `Kokkos::create_mirror_view_and_copy`. On a host
memory space that returns the view itself, so `h_L1` and `h_L2` alias one
buffer and the locals half of the check cannot fail on SERIAL (or OPENMP). The
potential half is sound: it compares two distinct views. Found by
`01_fix-tests` F2, where `UpwardSweep`'s twin test had the same defect: with
the zeroing `deep_copy` removed from `UpwardSweep::execute()`, its SERIAL
entry still passed. Fixed there with `create_mirror` + `deep_copy`; not yet
applied to `DownwardSweep`. Found by reading, not by a run.

### `SolveFusedM2L.FP32_smokeTest` is disabled: it fails at ≥ 2 ranks

**The case is commented out** (`tests/tstMultiSolve.hpp:1603-1656`), not filtered,
so it is absent from the `Canopy_Test_MultiSolve_MPI_SERIAL` binary, pending
the investigation below. Re-enable it as written once the defect is fixed — do
**not** re-enable it by widening its `5e-2` budget, which would retire the only
signal this defect has.

It passes at 1 rank but fails at 2–6 ranks: the FP32 max relative gradient error
is ≈ 0.277 at np=2, rising to ≈ 0.339 at np=3, well over the test's `5e-2`
budget.

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

### Two `unit` test targets do not compile

A `make -k` over the whole tree fails exactly two targets, at every backend:

- **`Canopy_Test_LaplaceKernel_*`** — 35 errors, all "no matching function" for
  `p2m_contribution`, `m2m_translate`, `m2l_translate`, `l2l_translate` and
  `l2p_evaluate`. `tests/tstLaplaceKernel.hpp` calls these operators with
  argument lists that no longer match their declarations in
  `src/Canopy_LaplaceKernel.hpp`.
- **`Canopy_Test_P2P_*`** — 3 errors; `tests/tstP2P.hpp:449` constructs a
  `TreeBuilder` with a `std::array<double,3>` bounding-box tolerance where the
  constructor (`src/Canopy_TreeBuilder.hpp:164-166`) takes
  `std::array<double,6>`. The signature drift traces to commit `8b0298e`
  "Refactor Solver constructor".

Both are **pre-existing** — verified by rebuilding
`Canopy_Test_LaplaceKernel_SERIAL` at `64d1648` with unrelated in-flight changes
stashed, which produces the same errors. Both are test-side drift behind a
`src/` signature change, not a defect in the library.

The consequence is that `ctest -L unit` cannot be run as the diagnostic layer
described under [Run with CTest](#run-with-ctest) until they are fixed. Build
and run the individually-compiling component tests by name in the meantime. To
be fixed in a separate session: update both test files to the current
signatures.

### `SingleSolve` fails at np=4 and deadlocks the suite when run with other solves

`SingleSolve` is labeled `unit`, **not** `regression`, for two reasons found
when it was re-enabled in the CTest suite:

1. **Accuracy failure at np=4.** `SingleSolve.PotentialNComps3` and
   `SingleSolve.PotentialAndGradientNComps3` fail at exactly 4 ranks with
   `max_pot_rel_err = 0.00196 vs tol 0.001` (~2× over budget; `0.00207`
   before `mac_satisfied` rejected exact ties,
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

Because of (2), `SingleSolve` is kept out of the `regression` label so that
`ctest -L regression` (`MultiSolve` only) does not deadlock. To be triaged in a separate session: fix the
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
