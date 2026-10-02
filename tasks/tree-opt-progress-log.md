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

## V1

**Outcome: the stop-and-report branch.** Step 1's derivation check fired, so
**no `fmm_tolerance` bound was moved** and V1 is not done. The six sites still
pass `1.0e-8`. The independent steps (2, 4, 5, the doc corrections) are in the
working tree; step 3 was measured but not applied, for the reason given under it.

Provenance for every figure below: commit `08653ef` plus this section's
uncommitted test edits, `build-tuolumne` (cache read directly:
`Canopy_ENABLE_PROFILING:BOOL=ON`, `Canopy_PROFILING_LEVEL:STRING=2`),
Cray clang 20.0.0, env `tuolumne_trilinos`, SERIAL backend. Each job's log
echoes the same provenance at its head.

### Decisions recorded (made before the session, not reopened)

- **Bounds are per call site, not one shared value.** The parameter is already
  per-site, the six configurations differ (`nsteps` 2/4/5/8, `drift_multiplier`
  1/2/5/50, uniform against clustered), and the measured errors span far more
  than 30x (table below). A single bound at the worst observed would pass a
  regression at the best-behaved site.
- **`SolveFusedM2L.FP32_smokeTest` is commented out and stays commented out**,
  and `README.md` records it as disabled pending investigation. Its budget was
  not re-justified.
- **The `theta_canopy` bound is confirmed, not redone** (step 4, below).
- **The `theta_ref` arm's `1e-3` bar is untouched.** It is an accuracy claim,
  not a regression bound.

### Measurement machinery added

`testMultiStepGravity` gained a `case_label` parameter (second, after `mode`)
and prints `[multisolve-dev] case <label> nprocs N nsteps N drift D
max_pos_rel X max_vel_rel Y tol T` on rank 0 **unconditionally**, before its two
`EXPECT_LT`s. `SolveFusedM2L.matchesPriorReference` prints
`[fusedm2l-dev] ... pot_err X grad_err Y` the same way. Both are readable on a
pass, which is what R10, A3 and B2 need. `CartesianTaylorSolve` already printed
`[ct-solve] ... direct_softened_sum max_pot_dev ... max_grad_dev` unconditionally.

### Step 1 — the six `fmm_tolerance` sites, three passes

Flux job `f3bmo4JYikKh` (`scripts/tuolumne/run_ctest_v1.flux`): three passes of
both stems in succession, `ctest -V --timeout 300`. No hang in any pass.
Max over the three passes, `max_pos_rel / max_vel_rel`, dimensionless:

| site | np 1 | np 2 | np 3 | np 4 | np 5 | np 6 |
| --- | --- | --- | --- | --- | --- | --- |
| `StableTree_Migrate` (Migrate, nsteps 5, dt 1e-4, drift 1) | 3.14e-11 / 3.49e-07 | 1.20e-10 / 5.45e-07 | 4.97e-10 / 2.39e-06 | 1.01e-09 / 2.97e-06 | 3.21e-09 / 8.39e-06 | 3.39e-09 / 6.18e-06 |
| `IntermediateMotion_Rebalance` (Rebalance, nsteps 5, dt 1e-3, drift 5) | 6.84e-07 / 1.41e-06 | 2.11e-06 / 2.99e-06 | 1.56e-05 / 4.36e-05 | 1.80e-04 / 2.77e-04 | 2.96e-04 / 8.28e-04 | 9.66e-05 / 3.38e-04 |
| `LargeMotion_Rebuild` (Rebuild, nsteps 4, dt 1e-3, drift 50) | 2.86e-06 / 2.91e-06 | 7.69e-05 / 1.34e-04 | 1.09e-04 / 1.49e-04 | 1.73e-03 / 1.60e-03 | 5.95e-04 / 9.23e-04 | 6.16e-05 / 8.50e-05 |
| `AutoMaintain` (Auto, nsteps 5, dt 1e-3, drift 5) | 6.84e-07 / 1.41e-06 | 2.11e-06 / 2.87e-06 | 1.56e-05 / 4.36e-05 | 1.80e-04 / 2.77e-04 | 2.96e-04 / 8.28e-04 | 9.66e-05 / 3.38e-04 |
| `AutoRebalance` (Auto, tree_tol 0.3, nsteps 8, dt 1e-3, drift 2) | 9.19e-06 / 8.55e-06 | 9.12e-05 / 1.36e-04 | 1.82e-04 / 3.00e-04 | 5.11e-05 / 5.06e-04 | 7.15e-04 / **6.57e-03** | **3.15e-03** / **1.71e-02** |
| `M2L_BinEdge_Fallback` (Migrate, clustered, theta 0.3, nsteps 2, dt 1e-4, drift 1) | 2.51e-11 / 4.75e-07 | 9.28e-11 / 7.22e-07 | 4.13e-10 / 1.66e-06 | 3.18e-10 / 1.13e-06 | 3.65e-10 / 1.04e-06 | 1.71e-09 / 9.16e-07 |

**Run-to-run spread: at most 2.5 %** (AutoRebalance np 5 velocity,
6.41e-3 / 6.41e-3 / 6.57e-3); every other reading agrees across the three
passes to 5 or more significant figures, np 3-6 included. R6 moves the counters
by up to ±25 % but barely moves these accuracy figures, so **stability is not
what blocks tightening here**. The np 1 figures reproduce the README's to every
digit (`3.485035e-07`, `6.841953e-07`, `9.194797e-06`).

**Read against the derivation.** The floor at `P_ORDER = 8`, `theta = 0.5` is
$\theta^{P+1} = 1.95 \times 10^{-3}$, a bound on the relative gradient error.
The integrator can only shrink that in velocity: `v` changes by `dt * g` per
step, so its relative error is at most the gradient's, and equals it only when
the field dominates `v`. **`AutoRebalance` exceeds the *undamped* floor** at
np 5 (velocity 6.41e-3 to 6.57e-3, 3.3x) and np 6 (velocity 1.71e-2, 8.8x;
position 3.15e-3, 1.6x). Every other reading sits below the raw floor. The
dt = 1e-3 sites also grow steeply with rank count (AutoRebalance velocity
~2000x from np 1 to np 6, global N 200 to 1200), while the directly measured
per-solve far-field gradient error does not
(`matchesPriorReference`, step 3: 7.3e-5 to 3.1e-4 over np 1-6, flat).

### The theta sweep: the excess is far-field-driven

Flux job `f3bn8EK66YaK` (`scripts/tuolumne/run_ctest_v1_theta_gain.flux`):
`CANOPY_MAC_THETA` 0.4 and 0.7, no rebuild. The floor ratio against 0.5 is
0.134x at 0.4 and 20.7x at 0.7. Velocity deviation ratio against theta 0.5:

| site | theta 0.4 (np 1-3) | theta 0.7 (np 1-6) |
| --- | --- | --- |
| `StableTree_Migrate` | 0.039, 0.058, 0.15 | 312, 60, 73, 90, 83, 39 |
| `IntermediateMotion_Rebalance` | 0.16, 0.058, 0.14 | 146, 61, 80, 40, 43, 119 |
| `LargeMotion_Rebuild` | 0.050, 0.0073, — | 2990, 135, 63, 211, 24, 79 |
| `AutoMaintain` | 0.16, 0.060, — | 146, 63, 80, 40, 43, 119 |
| `AutoRebalance` | 0.0032, 0.029, — | 23, 25, 47, 169, 54, 24 |
| `M2L_BinEdge_Fallback` (pins theta 0.3) | 1, 1, 1 | 1 at every np |

At theta 0.7, AutoRebalance's velocity reached 0.41 at np 6, and LargeMotion's
0.34 at np 4.

Every site that reads the env var moves in the floor's direction, and the
control site does not move at all. So the deviation **enters through the far
field**, and it does not come from migration, maintenance or the integrator on
their own. But the response is steeper than the floor (up to ~150x the floor
ratio at 0.7), so the deviation is not "floor times a fixed gain below 1" either.
Unmeasured candidates, recorded here and **not** pursued:
(a) per-particle normalization, since all-positive charges make $|g|$ cancel at
interior particles, which inflates a relative gradient error well past the floor
the same way `matchesPriorReference`'s potential figure is inflated (step 3);
(b) trajectory amplification through the tree changes of the dt = 1e-3 cases.
Telling them apart needs a per-particle, per-step gradient error printed beside
$|g|$, and that is its own task.

**Per the task, stopped here.** A bound set over the AutoRebalance figures would
be indistinguishable in the diff from a correct one. Setting bounds at the five
sites that fit was also not done: they share the driver, and the defect has
not been located.

### Step 2 — `FP32_smokeTest` disabled

Commented out in place (`tests/tstMultiSolve.hpp:1210`) under a block naming
the ~0.277 (np 2) / 0.339 (np 3) gradient error, the `5.0e-2` budget it
fails, and the README entry. README "Known Issues" retitled to
"`SolveFusedM2L.FP32_smokeTest` is disabled: it fails at ≥ 2 ranks", with the
re-enable condition. The rebuilt binary compiles; whether the case is absent from
`--gtest_list_tests` was not checked, because the session stopped before the
next job.

### Step 3 — `matchesPriorReference`: measured, not applied

From `f3bmo4JYikKh`, three passes, bit-identical at np 1, 2, 4, 6 and to 10+
digits at 3, 5:

| np | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| `pot_err` | 8.93e-04 | 2.43e-03 | 2.26e-02 | 6.77e-04 | **4.33e-02** | 1.77e-02 |
| `grad_err` | 7.26e-05 | 1.29e-04 | **3.07e-04** | 2.38e-04 | 1.61e-04 | 2.56e-04 |

- **The `5.0e-2` potential bound is not loose**, contrary to **R10**. The worst
  reading, 4.33e-2 at np 5, uses 0.87 of it. The figure is per-particle
  relative with mixed-sign charges (`q ~ U(-1, 1)`), so $|\phi| \to 0$ inflates
  it, and the 64x spread across rank counts is that cancellation, not the far
  field. Step 3's own recipe (2x the worst) would *loosen* it to 8.7e-2.
  Leave it as it is (step 7).
- **The `1.0e-1` gradient bound is 325x loose.** 2x the worst would be
  `6.2e-4`, and the figure is stable enough to carry it. **Not applied**,
  because the task stopped at step 1. It is the first bound to move when V1
  resumes, since it does not depend on the step-1 finding: this test is the
  per-solve far-field check that came out healthy.

### Step 4 — `theta_canopy`: confirmed, satisfied by prior work

Three passes × np 1-6: `theta_canopy` gradient `1.8651556395e-02` and potential
`9.9666798509e-04`, identical at all 18 readings and matching the rationale
block to every printed digit. `3.74e-02` is 2.005x the gradient. Constant
and block unchanged.

Found while there: the block's **`theta_ref` figures are stale**. It records
gradient `7.0132e-04` / potential `1.8963e-05` (job `f3YfvsefN86T`), and this
session measures `7.0717918545e-04` / `1.9263340835e-05`, identical at all 18
readings. The arm still clears its untouched `1e-3` bar, at 1.41x rather than
1.43x. The prose was left alone, since the decision covers the bar and the
re-measure was not asked for. A later edit to that block should update it.

### Step 5 — per-reason fallback assertion: implemented, compiled, not run

In the probe block of `testMultiStepGravity` (only `M2L_BinEdge_Fallback`
enables it), per rank and per solve: under `CANOPY_ENABLE_PROFILING`,
`count_cap == 0`, `depth_dropped == 0`, and `range_guard + count_cap ==
total_fallback_pair_count()`. Together with the existing `max_fallback > 0`,
this pins every refusal to the range guard, the reason T1 identified, without
asserting which rank holds the deep subtree. The `#else` branch asserts all
three at `-1`, prints `sum identity SKIPPED` once, and never evaluates the sum,
because `total_fallback_pair_count()` is ungated. **Both branches compile**
(`build-tuolumne` and `build-tuolumne-noprof`, target
`Canopy_Test_MultiSolve_MPI_SERIAL`). **Neither has been run.** T1's 42
readings all satisfy the ON-branch assertions, but that is not a run of this
code.

### Doc corrections

`tasks/tree-opt.md`: the `operatorCacheAcrossDrift*` citation
`tstCartesianTaylorSolve.hpp:1040-1060` corrected to `:1040-1062` in
**Current state**, B2 step 6 and **R9**. `:684`'s stale `fmm_tol=2e-2` prose
is **not** corrected. It belongs with the bound change that did not happen.

### Bugs only running revealed

**`ctest --timeout 300` does not contain the np-3 hang under flux, it spreads
it.** In `f3bn8EK66YaK` at theta 0.4, np 3 hung. ctest killed its `flux run`
client at 300 s, but the flux sub-job kept running `--exclusive`, so np 4, 5
and 6 sat in state `S` and each timed out at 300 s without starting
(`flux proxy f3bn8EK66YaK flux jobs -a`: np 3 at 19.33 min `R`, the rest `S`).
`flux cancel` on the sub-job freed the node, and np 4 then completed in 11 s
into a client that was already gone. Net loss: np 3-6 at theta 0.4. The
assumption in T1's log and in `run_ctest_t1.flux`, that 300 s bounds a hang
without costing the other rank counts, is false on this system.
`run_ctest_v1.flux` now runs a background watchdog that `flux cancel`s any
sub-job running longer than 300 s. **The watchdog itself has not been run yet.**
README's hang entry records this.

Also: pdebug rejects `--time-limit` above 1 h at submit. Three passes of both
stems take 424 s with no hang.

**Affects:**
- **V1** — resumes only once the AutoRebalance excess is attributed. When it
  does, apply step 3's gradient bound (`6.2e-4`), keep the potential at
  `5.0e-2`, run step 5's assertions and the watchdog script, and correct the
  `:684` prose with the bound change.
- **New task, before V1 resumes** — attribute AutoRebalance's np 5-6 deviation:
  per-particle, per-step gradient error beside $|g|$ on that configuration, to
  separate normalization from amplification from a far-field defect on the
  multi-rank multi-step path.
- **B1, A2, B2** — the six `fmm_tolerance` sites are still at `1.0e-8` and fail,
  so they cannot gate these tasks. Use the `[multisolve-dev]` and
  `[fusedm2l-dev]` before/after figures (deterministic to <=2.5 %) instead of
  pass/fail.
- **A3** — R10's before/after comparison can read `[multisolve-dev]` directly.
  The spread to read against is <=2.5 %, not R6's ±25 %.
- **R10** — `matchesPriorReference`'s potential bound is not loose (0.87
  used). Only its gradient bound is.
- **Every task whose script runs `MultiSolve` at np >= 3** — copy the watchdog
  from `run_ctest_v1.flux`. `--timeout` alone loses every later rank count to
  one hang.
