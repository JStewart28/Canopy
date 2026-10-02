# Tree balance and M2L operator-table economy on non-uniform trees — progress log

Session record for tree-opt. Companion to `tree-opt.md`, which holds the design,
the task sequence and the risks; this file holds what actually happened, in
order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `tree-opt.md` can cite it by
ID. No dates: the order of the sections is the chronology. If a session covers
more than one task, name them all; if it belongs to no task, name the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

Three things this topic in particular depends on your recording:

- **Every measurement, per `(nprocs, rank)` and from two runs.** The tree and
  partition path is run-to-run nondeterministic at np $\ge$ 3 (risk **R6**), so
  a single draw is not the number and a later before/after comparison needs
  your spread to be readable against.
- **The cell-count multiplier and the fallback-to-GEMM time ratio.** A3's
  default is arithmetic over those two measured inputs and nothing else. If
  either is missing from this log, A3 cannot be done.
- **Whether a measured factor was large enough to justify the task that
  follows it.** B0 and A1 both exist to produce a number that may say "do not
  bother". Recording a null result is the task succeeding.

(No entries yet.)

## T1

### Step 1 — which guard the existing clustered fixture's refusals come from

**Every refusal is a RANGE-GUARD refusal, and specifically an OFFSET refusal.**
Measured on `MultiSolve.M2L_BinEdge_Fallback` unchanged, at its own
`ncrit = 8`, `max_depth = 8`, `mac_theta = 0.3`, 300 particles/rank, 2 solves,
`build-tuolumne` configured `Canopy_ENABLE_PROFILING=ON`, flux job
`f3bkTMSV6Pfd`:

| nprocs | rank | step | range_guard | count_cap | depth_dropped | total | unique_ops | cells_at_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 10 | 0 | 0 | 10 | 350 | 1,4,29,30,35,28 |
| 1 | 0 | 1 | 10 | 0 | 0 | 10 | 350 | 1,4,29,30,35,28 |
| 2 | 0 | 0 | 132 | 0 | 0 | 132 | 1640 | 1,4,19,30,52,36 |
| 2 | 0 | 1 | 120 | 0 | 0 | 120 | 1629 | 1,4,19,30,53,34 |
| 2 | 1 | 0 | 46 | 0 | 0 | 46 | 1485 | 1,4,28,28,46,34 |
| 2 | 1 | 1 | 52 | 0 | 0 | 52 | 1478 | 1,4,28,28,45,37 |
| 3 | 0 | 0 | 285 | 0 | 0 | 285 | 1954 | 1,4,21,14,39,61,23 |
| 3 | 1 | 0 | 206 | 0 | 0 | 206 | 2350 | 1,4,22,25,46,52,6 |
| 3 | 2 | 0 | 155 | 0 | 0 | 155 | 2060 | 1,4,31,34,50,13,0 |
| 4 | 0 | 0 | 383 | 0 | 0 | 383 | 2520 | 1,4,22,15,45,55,29 |
| 4 | 1 | 0 | 334 | 0 | 0 | 334 | 2933 | 1,4,23,27,60,37,3 |
| 4 | 2 | 0 | 300 | 0 | 0 | 300 | 2862 | 1,4,23,23,40,43,24 |
| 4 | 3 | 0 | 103 | 0 | 0 | 103 | 2577 | 1,4,30,16,30,36,8 |
| 5 | 0 | 0 | 321 | 0 | 0 | 321 | 1982 | 1,4,24,4,20,62,39 |
| 5 | 4 | 0 | 292 | 0 | 0 | 292 | 3583 | 1,4,31,19,49,28,14 |
| 6 | 0 | 0 | 401 | 0 | 0 | 401 | 2852 | 1,4,25,5,20,55,47 |
| 6 | 3 | 0 | 419 | 0 | 0 | 419 | 2803 | 1,4,25,15,38,38,22 |
| 6 | 5 | 0 | 275 | 0 | 0 | 275 | 3663 | 1,4,32,16,35,41,13 |

(Truncated to one step per (nprocs, rank) above np=2; the full 42-line set is in
`canopy-t1-step1.f3bkTMSV6Pfd.log` (flux writes `--output` relative to the
submit directory, the repo root). `count_cap` and
`depth_dropped` are 0 on **every** line, and `range_guard == total` on every
line, so the sum identity holds exactly at all 42.)

Two conclusions, and the second is what decides step 2:

1. **It is the range guard, not the count cap.** The test has always run at the
   default column cap, so nothing it refuses is a budget refusal. Any later
   assertion on this fixture is an assertion about representability.
2. **Within the range guard it is the OFFSET bound, not the `dd` bound.** The
   per-depth occupancy never reaches past depth 6, so the largest depth
   difference any pair can carry is 6 — exactly `m2l_key_dd_max` for
   `LaplaceKernel` at `double` (`src/Canopy_LaplaceKernel.hpp:701`), which the
   guard admits (`std::abs( dd ) <= M2L_KEY_DD_MAX`). `|dd| > 6` is therefore
   geometrically unreachable on this draw, and every refusal must be a pair
   with an offset component beyond `M2L_KEY_OFFSET_MAX = 32` half-widths at
   the deeper cell's depth.

So the stale `M2L_BIN_RANGE = 3` comments were wrong about the number but right
about the *kind* of refusal: it is an offset refusal. The bound is 32, not 3,
and the four comment sites plus the `EXPECT_GT` failure message now name
`M2L_KEY_OFFSET_MAX` and `KernelType::m2l_key_dd_max` instead.

**Consequence for step 2:** this is the "only offset-guard refusals" branch of
T1's decision rule, so step 2 is a **new two-scale distribution** in
`tests/tstDownwardSweep.hpp`, not a relocation of the clustered draw. The
clustered draw's depth span (0-6, with the bulk of leaves at 2-5) is not a large
depth difference, and a fixture that B0/A1/C1 read level-difference numbers out
of needs one by construction rather than by luck of the draw.

### Step 2 — a new two-scale distribution, not a shared helper

Step 1's answer took T1 down the branch its own decision rule names: the
clustered draw produces only offset-guard refusals and no large depth
difference, so the geometry was built fresh in `tests/tstDownwardSweep.hpp`
rather than lifted out of `tstMultiSolve.hpp`. Two further reasons, both found
while doing it:

- The clustered draw is **entangled with a multi-step solve**. It lives inside
  `testMultiStepGravity`, which integrates particles, migrates/rebalances and
  compares against a brute-force reference. B0, A1 and C1 want a tree and one
  solve, not a trajectory. Extracting just the draw would have left a helper
  used at two different scales by two files with different `Position`/`Charge`
  field layouts.
- The two-scale geometry is **a different shape**, not the same shape moved.
  The clustered draw is one Gaussian of width 0.05 — a density gradient, which
  gives a depth span but no abrupt level difference. What the later tasks need
  is a step change in density: a cube of half-width 0.01 holding 87.5 % of the
  particles beside a uniform halo holding the rest.

`TwoScaleFixture<MemorySpace, ExecutionSpace, FarField = Kernel>` is the result
(`tests/tstDownwardSweep.hpp`), shaped like the `CachingFixture` beside it:
build, partition, rebuild on the local particles, comm plan, upward sweep,
`downward.setup()`, and a `solve()` that runs one `execute()` — which is what
populates every counter, since the per-reason tallies are set by
`build_interaction_list_device()`.

**Signature introduced, for B0 to use:**

```cpp
template <class TEST_MS, class TEST_ES, class FarField = Kernel>
struct DownwardSweepTest::TwoScaleFixture;
```

Templated on the far-field type and not fixed to this file's `Kernel`, because
B0 reads numbers off this same fixture for `CartesianTaylorBasis` *and*
`LaplaceKernel` and the two must be the same tree. The particle AoSoA is shared
with the rest of the file, so the fixture carries
`static_assert( FarField::num_components == Kernel::num_components )` — a
far-field type with a different component count would otherwise silently
mis-size the charge slice. Only `LaplaceKernel` is instantiated today; B0 adds
the second instantiation and needs no change here.

Knobs, all public members with their units on the declaration:
`num_particles_global = 1200`, `ncrit = 8`, `max_depth = 8`, `tolerance = 0.1`,
`replication_depth = 2`, `mac_theta = 0.3`. The column cap is **left at its
default** (measured in force: 32768), which is what makes `count_cap == 0` a
real check rather than a restatement of the configuration.

Two cases, deliberately separate so a failure localizes:
`DownwardSweepTwoScale.treeHasShallowAndDeepLeaves` (the geometry contract) and
`DownwardSweepTwoScale.refusalsAreRangeGuard` (the refusal claim and the sum
identity). Both `unit` tier, by the stem's existing label; nothing was
relabelled and no CMake change was needed.

### Bugs only running revealed

**1. A per-rank particle count made the fixture super-linear in rank count.**
The first version drew `num_particles = 1200` *per rank*, so the global set —
and with it the blob's refinement depth and the depth-8 interaction lists at
`theta = 0.3` — grew with every added rank. Measured: the two cases ran in
8.1 s at np 4 and had **not finished at 300 s** at np 5 and np 6 (flux job
`f3bkdURkfmR1`, both rank counts `***Timeout`). Fixed by making the count
**global** and splitting it across ranks (remainder to the low ranks). Runtime
at np 6 is now 9.1 s for the whole stem. This is also the better measurement:
with a fixed global set the rank counts are comparable to each other, which is
the entire point of reading per-`(nprocs, rank)` numbers off one fixture.

**2. "Shallowest occupied depth" asserts nothing.** The first version of the
geometry contract asserted `shallowest_occupied_depth <= 4`. Depth 0 is the
root and is occupied on every tree ever built, uniform ones included, so that
reading is always 0 and the assertion always passes — precisely the vacuous
pass **R7** is about, introduced into the test written to prevent it. Replaced
with the **shallowest depth at which refinement stops**: the smallest `d >= 1`
with `cells_at_depth[d] > cells_at_depth[d+1]`, so at least one occupied cell
at `d` has no children and is a leaf. On this draw it reads 1 or 2, never 0,
and a uniformly refined tree makes it `-1` and fails the `ASSERT_GE`.

**3. `build-tuolumne` was configured `Canopy_ENABLE_PROFILING=OFF`**, contrary
to `run_cmake_tuolumne.sh`, which sets it `ON`. The first step-1 run therefore
returned `-1` for all three counters and measured nothing (flux job
`f3bkPjbFSXhq`). Re-ran `run_cmake_tuolumne.sh` in that directory to restore
the documented configuration; the profiling-OFF build now lives in its own
`build-tuolumne-noprof/`. Worth knowing for any later session: the sentinel
reading is indistinguishable from "this build has profiling off", so check the
cache before concluding anything from a `-1`.

### Measured: the two-scale fixture, two separate runs (R6)

Both runs are `ctest -V` over the same binary back to back in one allocation,
flux job `f3bkmwQncaF9`, `build-tuolumne` with `CANOPY_ENABLE_PROFILING`.
`count_cap` and `depth_dropped` are **0 on every line of both runs**, and
`range_guard == total_fallback` on every line, so the sum identity holds
exactly at all 42 readings.

| nprocs | rank | range_guard r1 | r2 | unique_ops r1 | r2 | shallow leaf | deepest | cells_at_depth (r1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 184 | 184 | 162 | 162 | 2 | 8 | 1,8,59,2,2,1,8,34,116 |
| 2 | 0 | 74 | 74 | 47 | 47 | 2 | 8 | 1,8,13,1,1,1,5,14,64 |
| 2 | 1 | 114 | 114 | 45 | 45 | 2 | 8 | 1,8,45,0,0,1,3,13,57 |
| 3 | 0 | 55 | 55 | 41 | 41 | 1 | 8 | 1,7,5,1,1,1,3,6,41 |
| 3 | 1 | 42 | 42 | 45 | 45 | 2 | 8 | 1,7,11,0,1,1,3,9,42 |
| 3 | 2 | 131 | 131 | 82 | 82 | 2 | 8 | 1,8,39,16,0,0,2,11,33 |
| 4 | 0 | **33** | **43** | 31 | 31 | 1 | 8 | 1,8,5,1,1,1,3,9,33 |
| 4 | 1 | 36 | 36 | 16 | 16 | 2 | 8 | 1,8,14,0,0,0,2,5,24 |
| 4 | 2 | **46** | **36** | **20** | **17** | 2 | 8 | 1,8,12,0,0,0,3,7,35 |
| 4 | 3 | 51 | 51 | 27 | 27 | 2 | 8 | 1,8,35,5,0,0,0,3,21 |
| 5 | 0 | 38 | 38 | 33 | 33 | 1 | 8 | 1,8,1,1,1,1,2,5,23 |
| 5 | 1 | 34 | 34 | 20 | 20 | 1 | 8 | 1,8,4,0,0,0,2,6,23 |
| 5 | 2 | 50 | 50 | 14 | 14 | 2 | 8 | 1,8,10,0,0,0,3,8,20 |
| 5 | 3 | 40 | 40 | 35 | 35 | 2 | 8 | 1,8,13,0,1,1,0,6,31 |
| 5 | 4 | 148 | 148 | 40 | 40 | 2 | 8 | 1,8,32,0,0,0,1,4,21 |
| 6 | 0 | 27 | 27 | 47 | 47 | 1 | 8 | 1,8,2,2,1,1,2,4,23 |
| 6 | 1 | 12 | 12 | 35 | 35 | 1 | 8 | 1,8,3,0,0,0,1,3,24 |
| 6 | 2 | 28 | 28 | 53 | 53 | 2 | 8 | 1,8,13,0,0,0,1,4,28 |
| 6 | 3 | 36 | 36 | 36 | 36 | 1 | 8 | 1,8,5,0,0,0,3,8,20 |
| 6 | 4 | 28 | 28 | 31 | 31 | 2 | 8 | 1,8,9,0,0,0,1,4,23 |
| 6 | 5 | 67 | 67 | 64 | 64 | 2 | 8 | 1,8,35,6,0,0,0,4,17 |

`m2l_n_demanded_ops()` equals `m2l_n_unique_ops()` on every line of both runs,
and `m2l_realized_keys().size()` equals both — the column cap never bound, as
intended.

**Do the two runs agree? At np 1, 2, 3, 5 and 6, exactly — every field of every
line.** At **np 4 they do not**: rank 0's `range_guard` moved 33 -> 43, rank 2's
moved 46 -> 36 and its `unique_ops` 20 -> 17, and `num_local` moved on three of
the four ranks (309/309/295/287 -> 314/304/308/274) while the global total
stayed 1200. That is exactly **R6**, and it is worth noting that np 3, 5 and 6
reproducing here is not evidence R6 is narrower than documented — it is one
pair of draws. **The usable spread for a later before/after comparison on this
fixture is therefore roughly ±25 % on a per-rank `range_guard` at np >= 3**, and
no comparison at those rank counts should be read as a single pair of numbers.
np 1 and 2 are stable, consistent with the README entry.

The geometry is stable where it matters and is the reason the fixture exists:
**`deepest == 8` on every rank of every rank count in both runs**, the
shallowest leaf is at depth 1 or 2 throughout, and the occupied-depth span is
therefore 6 or 7 levels — against the clustered fixture's span of 5 with its
leaves bunched at 2-5. The `cells_at_depth` profile shows the two scales
directly: occupancy falls to 0-2 cells through the middle depths and rises
again to 17-116 at depth 8.

### Failure direction, verified

A separate build directory `build-tuolumne-noprof/`, configured identically to
`run_cmake_tuolumne.sh` except `-DCanopy_ENABLE_PROFILING=OFF`, with only
`Canopy_Test_DownwardSweep_MPI_SERIAL` built in it. Flux job `f3bkpiGKH3qR`:
all six rank counts **pass**, all three counters read **-1** on every rank, and
the `sum identity SKIPPED` notice is printed once per rank count (6 of 6). The
`#else` branch asserts `-1` and never `0`, and does not evaluate the sum
identity — `-1 + -1` against a real total is a claim about nothing.

Note the asymmetry the log makes visible: `total_fallback_pair_count()` is
**not** profiling-gated and still reads its real value (184 at np 1, matching
the profiling-on run exactly) while the three per-reason counters are
sentinels. That is why the sum identity has to be skipped rather than computed
from whatever is available.

### Runtime

`Canopy_Test_DownwardSweep_MPI_SERIAL` costs 3.7-9.1 s per rank count with the
two new cases in it, rising monotonically with rank count; the stem was 5-8 s
before. Nothing in it needs trimming. The expensive thing in T1's exit
criterion is `MultiSolve`, at 4.8-15.4 s per rank count plus an intermittent
np-3 hang that ran past 800 s once in this session (flux job `f3bkWYfR6Ao9`,
cancelled) — the hang README "Known Issues" already records. Both `ctest`
invocations in `scripts/tuolumne/run_ctest_t1.flux` therefore carry
`--timeout 300`, and `DownwardSweep` runs first, so a `MultiSolve` hang cannot
take the measurement with it.

### The exit criterion's `MultiSolve` arm does not pass, and did not before

Seven tests fail at every rank count: the six named in README "Known Issues"
(`StableTree_Migrate`, `IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`,
`AutoMaintain`, `AutoRebalance`, `M2L_BinEdge_Fallback`) on the `1e-8`
multi-step position/velocity check with errors of 3e-7 to 9e-6, plus
`SolveFusedM2L.FP32_smokeTest` at np 2-6, also recorded there. The error values
reproduce the README's to every digit (`6.8419528791564039e-07`,
`9.1947965989306709e-06`). T1 changed comments and added printing in that file
and nothing else, and **did not touch the tolerance** — sharpening those bounds
is V1's job, and loosening one to make a gate green would change what DONE
means invisibly. `M2L_BinEdge_Fallback`'s own `EXPECT_GT( max_fallback, 0 )`
**passes**; it fails only on the shared multi-step accuracy check inside
`testMultiStepGravity`.

**Affects:**
- **V1** — step 3 turns step 1's answer into an assertion, and the answer is
  **range guard**, with `count_cap == 0` and `depth_dropped == 0` alongside it.
  V1 can assert all three on the clustered fixture. Note the trap: it must
  assert on the *reason*, not on a total, since `total_fallback_pair_count()`
  is ungated and reads a real number even where the per-reason counters are
  sentinels. V1 also inherits the six pre-existing `MultiSolve` failures in the
  file it edits — it cannot verify its own change by that stem going green.
- **B0** — the fixture it reads from is
  `DownwardSweepTest::TwoScaleFixture<MS, ES, FarField>`, already templated for
  its two far-field types; it adds the `CartesianTaylorBasis` instantiation and
  nothing else. Its baseline numbers are the `unique_ops`/`demanded_ops`
  columns above, which are equal everywhere, so the column cap is not binding
  and any key-count reduction B0 measures is a real reduction.
- **A1** — the level differences it measures are available on this fixture: the
  occupied span is 6-7 levels with the shallowest leaf at depth 1-2 and the
  deepest cell at 8 on every rank of every rank count. A1's failure direction
  wants a neighbour level difference of at least 2, and the span here is well
  past that; its measurement is of *neighbouring* leaves, which this entry does
  not measure, so A1 still has to do it.
- **C1** — same fixture, same `solve()` entry point; drive it at a column cap
  of 0 by calling `set_m2l_op_count_cap(0)` before `solve()`. Note the default
  cap in force here is 32768 and the demanded counts are 14-162, so C1 must set
  the cap explicitly; it will never bind on its own.
- **A3** — unblocked on nothing by this entry; it still waits on C1 for the
  time ratio.
- **Everything measuring on this fixture at np >= 3** — report two runs. np 4
  disagreed between two back-to-back runs of the same binary by up to 25 % on a
  per-rank `range_guard`.
