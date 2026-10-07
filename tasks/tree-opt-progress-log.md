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

## E2 (fix-hang-rebalance)

Doc-only; no code, bound or job. Recorded here because it changes V1.

**Attribution.** `fix-hang-rebalance.md` E1 attributed `MultiSolve.AutoRebalance`'s
np 5-6 excess (section V1 above) to **trajectory amplification at unsoftened
close encounters**, V1's candidate (b). V1's candidate (a), per-particle
normalization, is excluded: the `max_vel_rel` particle's $\lvert v\rvert$ is at or
above the median at np 6 and at N = 1200. The far field is healthy per solve:
per-step field-scale error at most `9.5e-7` for AutoRebalance and `1.52e-6` for
any case, at SERIAL np 1-6 (`f3cZoFBJNNc7`) and HIP np 1-4 (`f3cZoFK9Wa7y`).
SERIAL np 1 at N = 1200 reaches `max_vel_rel = 2.67e-3` with no partition,
about 7000x the solve error its trajectory saw. The theta sweep above moved the
trajectory deviation because θ = 0.7 raises the per-step error about 140x, and the
dynamics amplify whatever goes in. Figures: `fix-hang-rebalance-progress-log.md`
section E1.

**Changed in `tree-opt.md` V1.**
- Status `**BLOCKED**` → `**NOT STARTED**`; the blocked paragraph is replaced
  by a "Resume from step 1" paragraph with the attribution.
- Step 1's derivation: the floor bounds the per-solve error; the trajectory
  check does not damp it at unsoftened close encounters, with E1's figures. The
  old text said a gradient error reaches `max_pos_rel` reduced by the step size
  twice over, which holds only while trajectories stay close.
- Step 1's stop clause now triggers on the per-step probe
  (`CANOPY_MULTISOLVE_PROBE=1`) exceeding $\theta^{P+1}$, not on the trajectory
  deviation. A trajectory excess whose probe stays under the floor and sits on
  close encounters is recorded as dynamics.
- Line citations updated after E1's edit to `tests/tstMultiSolve.hpp`: call
  sites `:955, :972, :990, :1008, :1045, :1094`, `EXPECT`s `:929-933`, the
  stale `2e-2` prose `:1032`, `P_ORDER` `:88`, `get_test_mac_theta()` `:42-47`.

**Affects:**
- **V1** — resumes from step 1 and re-measures (the partition changed in
  `fix-hang-rebalance.md` H2). Its stop clause reads the per-step probe. Whether
  an amplification site (AutoRebalance np 5-6) gates on the trajectory or on the
  probe is V1's decision; E2 made none. The other resume items in section V1's
  **Affects** still stand.

## T1 (HIP arm)

Post-H2 re-record of the step-8 lines on both backends, and the HIP arm of the
exit criterion. Provenance for every figure here: commit `3bb7fb5`, Cray clang
20.0.0, env `tuolumne_trilinos`; `build-tuolumne` with
`Canopy_ENABLE_PROFILING:BOOL=ON` and `build-tuolumne-noprof` with `OFF`, both
read from the cache and echoed at each job's head. Every run went through
`canopy_ctest` after a passing watchdog self-test, with the HIP environment set
in a subshell around the HIP calls only. Script:
`scripts/tuolumne/run_ctest_t1_hip.flux measure|noprof serial|hip`, written
from `run_ctest_e1.flux` because `run_ctest_t1.flux` predates `canopy_ctest`
and HIP registration. No entry went over budget and the watchdog cancelled
nothing.

| job | build | what |
| --- | --- | --- |
| `f3cacJQf92r3` | profiling ON | SERIAL np 1-6: two `DownwardSweep` passes, one `MultiSolve` pass |
| `f3cacJZDntFZ` | profiling ON | HIP np 1-4: the same |
| `f3cacJgXoLrB` | profiling OFF | SERIAL np 1-6: one `DownwardSweep` pass |
| `f3cacJp59gyu` | profiling OFF | HIP np 1-4: one `DownwardSweep` pass |

### `build-tuolumne-noprof/` reconfigured

It was configured before `fix-hang-rebalance.md` H0a, so its cache had no
`Canopy_TEST_MPI_RANKS_HIP` / `Canopy_TEST_MPIEXEC_PREFLAGS_HIP` and its HIP
entries would have registered at np 1-6 with no GPU binding. Re-ran
`run_cmake_tuolumne.sh`'s exact arguments with only
`-DCanopy_ENABLE_PROFILING=OFF` changed. After: cache reads
`Canopy_ENABLE_PROFILING:BOOL=OFF`, the two HIP variables are set, and
`ctest -N -R '_MPI_HIP_np_'` lists np 1-4 only (11 stems each).

### The step-8 lines, post-H2

**Reproducible on both backends, and backend-independent.** The two
`DownwardSweep` passes print identical `[two-scale]` lines, every field of
every `(nprocs, rank)`, at SERIAL np 1-6 and HIP np 1-4; and at np 1-4 the HIP
lines equal the SERIAL lines field for field. **The np-4 disagreement the
SERIAL arm recorded (±25 % on a per-rank `range_guard`) no longer occurs.** It
came from the MJ partition; the ParMETIS partition H2 put in reproduces, as H2
measured. So **R6**'s spread does not apply to this fixture on the current
partitioner: a before/after comparison on it can be read line for line at
every np, on either backend.

`count_cap == 0`, `depth_dropped == 0` and `range_guard == total_fallback` on
every line; `unique_ops == demanded_ops == realized_keys` on every line, so the
column cap (32768) never bound. np 1 is unchanged from the SERIAL arm
(184 / 162), as it must be with no partition; every np >= 2 line moved.

| nprocs | rank | num_local | range_guard | unique_ops | occupied | shallow leaf | deepest | cells_at_depth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1200 | 184 | 162 | 9 | 2 | 8 | 1,8,59,2,2,1,8,34,116 |
| 2 | 0 | 606 | 141 | 57 | 9 | 2 | 8 | 1,8,57,1,1,1,4,11,61 |
| 2 | 1 | 594 | 47 | 37 | 7 | 1 | 8 | 1,8,1,0,0,1,4,16,60 |
| 3 | 0 | 374 | 145 | 80 | 9 | 2 | 8 | 1,7,47,6,1,1,2,6,40 |
| 3 | 1 | 437 | 35 | 35 | 7 | 1 | 8 | 1,7,4,5,0,0,2,8,42 |
| 3 | 2 | 389 | 48 | 53 | 9 | 1 | 8 | 1,8,4,6,1,1,4,12,34 |
| 4 | 0 | 304 | 10 | 9 | 6 | 1 | 8 | 1,8,2,0,0,0,1,6,28 |
| 4 | 1 | 297 | 20 | 28 | 6 | 1 | 8 | 1,8,2,0,0,0,2,7,26 |
| 4 | 2 | 304 | 10 | 18 | 7 | 1 | 8 | 1,8,2,5,0,0,1,4,31 |
| 4 | 3 | 295 | 126 | 50 | 9 | 2 | 8 | 1,8,60,1,1,1,4,7,28 |
| 5 | 0 | 238 | 46 | 49 | 9 | 2 | 8 | 1,8,8,1,2,1,3,8,22 |
| 5 | 1 | 238 | 46 | 18 | 7 | 1 | 8 | 1,8,1,0,0,1,2,5,24 |
| 5 | 2 | 244 | 25 | 11 | 6 | 1 | 8 | 1,8,1,0,0,0,1,5,24 |
| 5 | 3 | 233 | 21 | 37 | 6 | 1 | 8 | 1,8,1,0,0,0,1,5,25 |
| 5 | 4 | 247 | 172 | 33 | 6 | 2 | 8 | 1,8,49,0,0,0,1,6,23 |
| 6 | 0 | 203 | 12 | 42 | 6 | 1 | 8 | 1,8,2,0,0,0,1,5,22 |
| 6 | 1 | 206 | 44 | 52 | 6 | 2 | 8 | 1,8,24,0,0,0,1,4,23 |
| 6 | 2 | 202 | 12 | 34 | 6 | 1 | 8 | 1,8,2,0,0,0,1,3,24 |
| 6 | 3 | 201 | **0** | 22 | 5 | 1 | 8 | 1,8,2,0,0,0,0,3,24 |
| 6 | 4 | 195 | 44 | 37 | 7 | 2 | 8 | 1,8,12,6,0,0,1,5,22 |
| 6 | 5 | 193 | 86 | 109 | 9 | 2 | 8 | 1,8,25,2,1,1,4,7,20 |

np 6 rank 3 refuses nothing: its partition holds no pair the guard refuses.
That is why the assertion is on the all-rank sum, not per rank.

### Failure direction

Profiling OFF, both backends: all three counters read **-1** on every
`[two-scale]` and `[two-scale-refusals]` line (21 of 21 SERIAL, 10 of 10 HIP),
`total_fallback` still reads its real value (184 at np 1), and
`sum identity SKIPPED` prints once per rank count (6 SERIAL, 4 HIP). Every
entry passes.

### `MultiSolve`: what fails, and what was carried

SERIAL np 1-6 fails exactly the six `1e-8` cases, and only at the two
trajectory `EXPECT`s (`tests/tstMultiSolve.hpp:929`, `:933` in this binary).
**HIP np 1-4 fails the same six plus `SolveFusedM2L.multipleSolvesIdempotent`**,
whose three back-to-back solves differ in about the 12th digit
(`76.698048540534216` against `76.698048540863965`). That case is HIP-only,
not exercised by anything T1 added, and **pre-existing**: H0c's baseline log
`canopy-h0.f3cM4ghTjtiT.log` already shows it failing, as does every H2 and E1
HIP `MultiSolve` log. Their written summaries ("fails only the six") omitted
it, so it was never in README. It is carried, and README "Known Issues" now
has an entry for it.

`M2L_BinEdge_Fallback`'s per-reason assertions (`count_cap == 0`,
`depth_dropped == 0`, sum identity; V1 step 5) ran here for the first time,
in the profiling-ON branch, on both backends, and never fired.

**Affects:**
- **V1** — its exit criterion requires `MultiSolve` HIP with no failure
  carried, and `multipleSolvesIdempotent` fails on HIP for a reason no bound
  can move (bit-identity under a run-dependent device accumulation order). V1
  cannot be **DONE** on HIP until that case is fixed or its claim changed,
  which is outside V1. Step 5's ON branch is now verified; its OFF branch still
  needs a run.
- **B0, A1, C1** — read this table, not the SERIAL arm's: np >= 2 moved with
  H2. The fixture now reproduces at every np on both backends, so one run per
  backend is a usable baseline and a second run is a check, not a spread.
- **B1, A2, B2, A3** — the before/after spread on this fixture is zero at every
  np; any difference in a step-8 line is the change's.

## V1 (resume)

**Outcome: every step done; the exit criterion is met on SERIAL and blocked on
HIP by `SolveFusedM2L.multipleSolvesIdempotent` alone**, a pre-existing
HIP-only failure V1 does not own (see `## T1 (HIP arm)` and README "Known
Issues"). V1 is marked **BLOCKED**, not **DONE**.

Provenance: commit `1e6185d` plus this section's test edits, Cray clang 20.0.0,
env `tuolumne_trilinos`, `build-tuolumne` (`Canopy_ENABLE_PROFILING:BOOL=ON`)
and `build-tuolumne-noprof` (`OFF`), read from the cache and echoed at each
job's head. Every run went through `canopy_ctest` after a passing watchdog
self-test; no entry went over budget, none was refused, and the watchdog
cancelled nothing. Script: `scripts/tuolumne/run_ctest_v1_bounds.flux
measure|exit [serial|hip]`, `noprof`, `fail`.

### Decisions recorded (made before the session, not reopened)

- **Each `MultiSolve` trajectory bound is the tighter of a derived figure and a
  measured one**: $\theta^{P+1}$ carried through the site's integration, and
  the worst deviation over three runs per backend (SERIAL np 1-6, HIP np 1-4)
  times a margin no smaller than the run-to-run spread. The "leave it as it
  is" rule applies only to tightening `matchesPriorReference`'s passing
  gradient bound.
- **`AutoRebalance` gates on both** an unconditional per-step probe (field-scale
  error on every step) and a trajectory bound commented as a dynamics catch.
  Any other site whose trajectory exceeds its derived bound with a clean probe
  would get the same treatment — none did (below). `MultiSolve`'s `default`
  rows were re-calibrated for the probe's runtime.
- **`matchesPriorReference`'s potential bound stays at `5.0e-2`.**

### Signature changed, and why

`testMultiStepGravity` (`tests/tstMultiSolve.hpp:230`): `double fmm_tolerance`
became **`double pos_tolerance, double vel_tolerance`**, and a trailing
**`double probe_field_tol = 0.0`** was added. Position and velocity deviations
differ by up to four orders of magnitude at one site (StableTree: pos 3.4e-9,
vel 9.4e-6), so one shared tolerance would have been 1000x loose on position.
`probe_field_tol > 0` turns the probe on regardless of
`CANOPY_MULTISOLVE_PROBE` and `EXPECT`s every step's field-scale error under
it. The six call sites in the same file are the only callers.

### The derived figure, as implemented

The shadow loop on rank 0 accumulates, per particle, a first-order error
budget: `budget_dv += dt * A_i` before each kick and
`budget_dr += dt * drift * budget_dv` after it, where
$A_i = \sum_{j\ne i} |q_j| / r_{ij}^2$ (new `absolute_field()`). If every
far-field interaction is in error by at most $\varepsilon$ relative, then
$|\delta v_i| \le \varepsilon\,\texttt{budget\_dv}_i$ and
$|\delta r_i| \le \varepsilon\,\texttt{budget\_dr}_i$ while the trajectories
stay close (no feedback). $A$ rather than $|g|$ because $|g|$ cancels and $A$
does not. The `[multisolve-dev]` line now prints
`derived_pos` $= \theta^{P+1} \max_i \texttt{budget\_dr}_i / |r_i|$ and
`derived_vel` $= \theta^{P+1} \max_i \texttt{budget\_dv}_i / |v_i|$ (same
$<10^{-10}$ fallback as the deviation), plus `pos_tol` / `vel_tol` in place of
`tol`. The site constant is the worst derived figure over np.

### Step 1 — measured: flux jobs `f3cajPhDd7dZ` (SERIAL) and `f3cajPqbu44f` (HIP)

Three passes each, `MultiSolve` under `CANOPY_MULTISOLVE_PROBE=1` (config
`probe`; the probe is read-only and inert on `[multisolve-dev]`, E1) and
`CartesianTaylorSolve` at default. **SERIAL repeats bit for bit across the
three passes at every np; HIP moves by at most 1.0004x** (IntermediateMotion
np 2 position). HIP figures equal SERIAL's to about four digits. np 1 matches
the first pass's table to every printed digit; np >= 2 moved with H2.

**Stop clause: not triggered.** The per-step field-scale error is at most
`1.52e-6` (LargeMotion np 5) on any step, site, np or backend, against
`1.95e-3`; `M2L_BinEdge_Fallback` peaks at `2.33e-8` against its own floor
$0.3^9 = 1.97 \times 10^{-5}$. AutoRebalance's probe `end` line: `n_excess` 2
(1 close) at np 5 and 11 (10 close) at np 6, zero elsewhere.

**Every measured deviation is under its derived figure at every np** —
AutoRebalance np 6 velocity `1.713e-2` against `4.01e-1`, LargeMotion np 4
position `1.742e-3` against `6.23e-2` — so measured x 2 is the tighter figure
at every site, and no site beyond AutoRebalance needed the dynamics
treatment. (The first-order budget is generous where particles pass close,
because $A_i$ is large exactly there; that is why E1's close-encounter excess
still sits inside it.)

| site | derived pos / vel (worst np) | measured worst pos / vel | old | new pos / vel |
| --- | --- | --- | --- | --- |
| `StableTree_Migrate` | 9.64e-6 / 1.97e-1 | 3.367e-9 (np 6) / 9.364e-6 (np 5) | 1.0e-8 | **6.8e-9 / 1.9e-5** |
| `IntermediateMotion_Rebalance` | 1.76e-2 / 9.73e-2 | 2.203e-4 (np 5) / 6.182e-4 (np 5) | 1.0e-8 | **4.5e-4 / 1.3e-3** |
| `LargeMotion_Rebuild` | 9.33e-2 / 2.24e-1 | 1.742e-3 (np 4) / 1.612e-3 (np 4, HIP) | 1.0e-8 | **3.5e-3 / 3.3e-3** |
| `AutoMaintain` | 1.76e-2 / 9.73e-2 | 2.203e-4 (np 5) / 6.182e-4 (np 5) | 1.0e-8 | **4.5e-4 / 1.3e-3** |
| `AutoRebalance` (trajectory) | 3.26e-2 / 4.01e-1 | 3.160e-3 (np 6) / 1.713e-2 (np 6) | 1.0e-8 | **6.4e-3 / 3.5e-2** |
| `AutoRebalance` (probe, per step) | 1.95e-3 | 9.509e-7 (np 3, both backends) | none | **1.9e-6** |
| `M2L_BinEdge_Fallback` (θ 0.3) | 2.28e-5 / 4.92e-4 | 1.715e-9 (np 6) / 1.661e-6 (np 3) | 1.0e-8 | **3.5e-9 / 3.4e-6** |

`AutoMaintain` and `IntermediateMotion_Rebalance` print identical figures at
every np: with `drift 5` every Auto step takes Rebuild or Rebalance along the
same trajectory. Two position bounds (StableTree, BinEdge) land below the old
`1e-8`; both deviations are bit-stable on SERIAL and within 1.0004x on HIP,
so the tightening is explained (step 7).

### Step 3 — `matchesPriorReference`, applied

Identical over three runs on each backend: `grad_err` 7.256e-5, 1.421e-4,
**2.847e-4**, 2.286e-4, 1.764e-4, 2.463e-4 at np 1-6; `pot_err` 8.93e-4,
2.82e-3, 2.21e-2, 8.29e-4, **4.32e-2**, 1.80e-2. **Gradient bound
`1.0e-1` → `5.7e-4`** (worst x 2; floor at P = 6 is $0.5^7 = 7.8\times10^{-3}$).
Potential stays `5.0e-2` (0.87 used, cancellation-inflated). The comment's old
"complete-regression bug" prose was replaced by these figures.

### Step 4 — `theta_canopy` confirmed, rationale figures refreshed

Gradient `1.8651551291e-02`, potential `9.9666786091e-04`, identical at all 30
readings (SERIAL np 1-6, HIP np 1-4, three passes). These differ from the
block's `1.8651556395e-02` / `9.9666798509e-04` in the seventh digit, so the
block was updated; `3.74e-02` is 2.005x the new gradient, so the constant
stands. `theta_ref`: `7.0717918545e-04` / `1.9263340835e-05`, identical at all
30, 1.41x under the untouched `1e-3` bar; the stale `7.0132e-04` /
`1.8963e-05` / "1.43x" prose was replaced.

### Step 5 — both branches run

Profiling ON: the per-reason assertions ran in T1's HIP-arm jobs and in every
job here and never fired. Profiling OFF (`f3cax1mLU4mu`,
`build-tuolumne-noprof`, SERIAL np 1-6): all 42 `[m2l-fallback-reason]` lines
read -1 on all three counters, `sum identity SKIPPED` prints at all six rank
counts, and `M2L_BinEdge_Fallback` passes.

### Budget rows re-calibrated

`run_ctest_h0b.flux calibrate`, `CANOPY_CAL_REGEX` = `MultiSolve` SERIAL
np 1-6: `default` (`f3cajB3pSQ1m`), `probe` (`f3camEXYMR7u`), and a new
`theta0.7` config for the failure direction (`f3caoGecUY71`). The default
rows barely moved at np >= 2 (6.63-16.25 s against 6.81-15.8 s); np 1 rose
to 8.56 s, the cold first entry of the job.

### Failure direction — `f3carGPn6qH9`

Temporarily made `matchesPriorReference` read `get_test_mac_theta()`, rebuilt,
ran `MultiSolve` SERIAL np 1-6 under `CANOPY_MAC_THETA=0.7` (config
`theta0.7`), reverted, rebuilt. Demonstrated:
- **`matchesPriorReference` gradient**: 2.31e-3 to 1.23e-2 at np 1-6, failing
  `5.7e-4` at every np; the old `1.0e-1` passes all six.
- **AutoRebalance probe gate**: per-step field error up to 2.04e-5, failing
  `1.9e-6` (8 `EXPECT`s over np 1-6); a gate at the floor `1.95e-3` passes.
- The six trajectory sites fail too (StableTree position at np 1, 3.5e-9,
  stays under `6.8e-9`); the old `1e-8` already failed them, so they
  demonstrate nothing new. `M2L_BinEdge_Fallback` pins θ = 0.3 and is
  unchanged — the control.

### Exit criterion — `f3cax1EfQvc7` (SERIAL), `f3cax1ePDS87` (HIP)

Three successive passes of `CartesianTaylorSolve` then `MultiSolve`, default
config. **SERIAL: 36 of 36 entries pass**, every figure identical to the
measurement runs; the largest fraction of any bound used is 0.498
(LargeMotion np 4 position). **HIP: all 12 `CartesianTaylorSolve` entries
pass; 11 of 12 `MultiSolve` entries fail, every failure in
`SolveFusedM2L.multipleSolvesIdempotent`** (58 failure lines, all in that
case; it passed once at np 1). Every case V1 changed passes on HIP, at most
0.498 of its bound, and AutoRebalance's probe peaks at 9.51e-7 of `1.9e-6`.

### Doc corrections made

`tree-opt.md`: R10 rewritten for the new bounds; stale `tstMultiSolve.hpp`
citations in Test naming, Current state, T1, C1 and R7; the drift-case
citation (`:1041-1061`) and the direct-sum arms (`:977-1007`). README: the
"Six `MultiSolve` tests fail the `1e-8`" entry removed, the FP32 entry's
citation and its cross-reference fixed, the idempotence entry updated.

**Affects:**
- **V1** — **DONE** once `multipleSolvesIdempotent` passes on HIP; rerun
  `run_ctest_v1_bounds.flux exit hip` and nothing else. It needs either
  deterministic device accumulation (a `src/` change) or a decision that the
  case asserts agreement to a tolerance — neither is V1's.
- **B1, A2, B2, A3** — the `MultiSolve` trajectory sites now gate, and
  `[multisolve-dev]` prints `derived_pos` / `derived_vel` beside each
  deviation. SERIAL figures repeat bit for bit, so a before/after comparison is
  exact on SERIAL and within 1.0004x on HIP. Bounds are 2x the worst over
  np 1-6, so record figures, not pass/fail (R10).
- **A2, A3** — `matchesPriorReference`'s gradient bound is now `5.7e-4`, about
  2x its worst; a balanced tree that moves it by more than that fails.
- **Any task that adds work to `MultiSolve`** — re-calibrate its `default`,
  `probe` and `theta0.7` rows.

## V1 (close)

**`SolveFusedM2L.multipleSolvesIdempotent` now asserts agreement to a
tolerance, not bit-identity**, which closes V1's HIP arm. V1 is **DONE**.

**Decision.** The case exists to catch residual state in `_locals` between
solves or a broken zero-init in `execute()`. Either presents as an O(1)
change, so agreement to a round-off tolerance keeps the test's purpose. Bit
identity is not a property the HIP backend has: its device reductions
accumulate in a run-dependent order (README "Known Issues", HIP
bit-reproducibility). The comparison is field-scale, the `field_scales` rule
of `tstLaplaceSolve.hpp`: `max_i |x_k - x_1| / max_i |x_1|` over all ranks,
for the potential and for the gradient vector, solves 2 and 3 against solve 1.
Rank 0 prints `[fusedm2l-idem] nprocs N pot_drift X grad_drift Y` on every run.

**Measured** (provisional tolerance `1e-10`; flux jobs `f3cb6Hpf9s35` SERIAL,
`f3cb6HwmJRNw` HIP; three passes each):

| backend | np | pot_drift | grad_drift |
| --- | --- | --- | --- |
| SERIAL | 1-6 | 0 | 0 |
| HIP | 1 | 5.0e-17 | 6.8e-14 – 1.67e-13 |
| HIP | 2 | 9.3e-17 – 1.86e-16 | 1.90e-13 – 3.80e-13 |
| HIP | 3 | 7.0e-17 – 1.39e-16 | 3.82e-13 – 5.41e-13 |
| HIP | 4 | 5.2e-17 – 7.9e-17 | 6.4e-14 – 9.1e-14 |

**`IDEM_TOL = 1e-11`**, about 20x the worst gradient drift. It is set as a
round-off budget rather than measured x 2: the drift moves up to 2.8x between
passes at one np, and three samples per np do not pin its tail. SERIAL drift is
exactly 0, so on SERIAL the check is as strong as bit-identity was.

**Exit criterion — `f3cbCkKDsirK` (SERIAL), `f3cbCkSwbyxT` (HIP)**, on the final
binary: three successive passes of `CartesianTaylorSolve` and `MultiSolve`.
SERIAL 36 of 36 entries pass and HIP 24 of 24, with no failure carried, no
entry over budget and no watchdog cancellation. Worst drift: HIP
`grad_drift 5.41e-13` at np 3. README's "`multipleSolvesIdempotent` fails on
HIP" entry is removed.

**Affects:**
- **B1, A2, B2, A3** — `MultiSolve` and `CartesianTaylorSolve` now pass with
  nothing carried on both backends, so they gate outright. A HIP failure in
  either is the change's (R11).

## B0

**Outcome: the duplicate factor is exactly 1.0 for both bases, at every
`(nprocs, rank)`, on both backends.** On T1's two-scale fixture no two admitted
keys share $(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$, so a
`dd`-free key would remove nothing here. That is the null result the design
says to record before B1 starts.

Provenance: commit `b516a17` plus this section's edits to
`tests/tstDownwardSweep.hpp`, Cray clang 20.0.0, env `tuolumne_trilinos`,
`build-tuolumne` (`Canopy_ENABLE_PROFILING:BOOL=ON`) and `build-tuolumne-noprof`
(`OFF`). Both were read from the cache and echoed at each job's head. Every run
went through `canopy_ctest` after a passing watchdog self-test. No entry went
over budget, and the watchdog cancelled nothing but the self-test. Script:
`scripts/tuolumne/run_ctest_b0.flux measure|noprof serial|hip`, copied from
`run_ctest_t1_hip.flux` without the `MultiSolve` pass.

| job | build | what |
| --- | --- | --- |
| `f3cbxZUkYRWf` | ON | SERIAL np 1-6, two passes: the step-0 check, before the B0 case existed |
| `f3cc36Kv7B4j` | ON | `run_ctest_h0b.flux calibrate`, `DownwardSweep` SERIAL np 1-6 |
| `f3cc4m4y111Z` | ON | SERIAL np 1-6, two passes |
| `f3cc4mDrvh5Z` | ON | HIP np 1-4, two passes |
| `f3cc4mNnLNRu` | OFF | SERIAL np 1-6, one pass |
| `f3cc4mXSvAxo` | OFF | HIP np 1-4, one pass |

### Decisions recorded (made before the session, not reopened)

- **`CartesianTaylorBasis<double, 3, 1>`**, matching the downstream
  configuration's 3200 B per column. `LaplaceKernel` stays at the file's
  `P_ORDER = 6`, which is 21 952 B per column. Both byte figures come from the
  basis's own `bytes_per_key`.
- **One fixed positive softening for both bases**, `1.0e-3` domain units. It
  exists only to satisfy `CartesianTaylorBasis`'s guards. It moves no key and no
  interaction-list entry, and nothing here asserts accuracy.

### Step 0 — the fixture change

`TwoScaleFixture` gained a `softening` member. Its constructor now hands
`M2LKernelParams{softening}` to both sweeps, the upward one before `setup()`,
and sets the downward root half-width from `builder.root_box()` exactly as
`Solver::_push_root_half_width` does. Without this, a `CartesianTaylorBasis`
solve aborts in `build_m2l_operators`. Before the B0 case was written,
`f3cbxZUkYRWf` printed `[two-scale]` lines identical to the `## T1 (HIP arm)`
table, field for field at all 21 `(nprocs, rank)`, and identical across its two
passes. The change moved nothing T1 measured.

### The case

`DownwardSweepTwoScale.ddDuplicateColumns` (`testTwoScaleDdDuplicates`) builds
the fixture once per basis and solves. It counts `m2l_realized_keys()` and the
distinct `(max_d, ii, jj, kk)` tuples, and prints one `[b0-dd]` line per basis
per rank. It asserts `distinct <= admitted` and nothing about the ratio. It also
asserts the two fixtures' `m2l_cells_at_depth()` are equal, because the control
means something only on the same tree.

### Measured

Both passes print identical `[b0-dd]` lines on each backend, and HIP equals
SERIAL line for line at np 1-4. The factor is 1.0000 on every line, so only the
admitted count is listed. `distinct == admitted` and
`demanded_ops == admitted` everywhere, so the column cap (32768) never bound.

| nprocs | rank | CT admitted | CT bytes | Laplace admitted | Laplace bytes |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 162 | 518 400 | 162 | 3 556 224 |
| 2 | 0 | 57 | 182 400 | 57 | 1 251 264 |
| 2 | 1 | 37 | 118 400 | 37 | 812 224 |
| 3 | 0 | **85** | 272 000 | **80** | 1 756 160 |
| 3 | 1 | 35 | 112 000 | 35 | 768 320 |
| 3 | 2 | **57** | 182 400 | **53** | 1 163 456 |
| 4 | 0 | 9 | 28 800 | 9 | 197 568 |
| 4 | 1 | 28 | 89 600 | 28 | 614 656 |
| 4 | 2 | **19** | 60 800 | **18** | 395 136 |
| 4 | 3 | **51** | 163 200 | **50** | 1 097 600 |
| 5 | 0 | 49 | 156 800 | 49 | 1 075 648 |
| 5 | 1 | 18 | 57 600 | 18 | 395 136 |
| 5 | 2 | 11 | 35 200 | 11 | 241 472 |
| 5 | 3 | 37 | 118 400 | 37 | 812 224 |
| 5 | 4 | 33 | 105 600 | 33 | 724 416 |
| 6 | 0 | 42 | 134 400 | 42 | 921 984 |
| 6 | 1 | 52 | 166 400 | 52 | 1 141 504 |
| 6 | 2 | 34 | 108 800 | 34 | 746 368 |
| 6 | 3 | 22 | 70 400 | 22 | 482 944 |
| 6 | 4 | **43** | 137 600 | **37** | 812 224 |
| 6 | 5 | **115** | 368 000 | **109** | 2 392 768 |

`LaplaceKernel` equals T1's `unique_ops` at every rank. Where the bases differ
(bold), `CartesianTaylorBasis` has more columns. That is its retained `max_d`:
`LaplaceKernel` zeroes it, so same-offset keys at different levels collapse
there. It has nothing to do with `dd`.

### Why 1.0 — partly structural, partly this draw

The classify pass (`src/Canopy_DownwardSweep.hpp:1671-1697`) computes each
offset component as `(2i_s+1-2^{d_s})·2^{max_d-d_s} - (2i_t+1-2^{d_t})·2^{max_d-d_t}`.
At `dd == 0` both terms are odd, so every component is **even**. At `dd != 0`
the deeper cell's term is odd and the coarser one's is even, so every component
is **odd**. A `dd == 0` key therefore can never collide with a `dd != 0` key on
any tree. The proof stops there: keys with different non-zero `dd` (including
`+k` against `-k`) are not separated by parity. They did not collide on this
fixture. One plausible reason, **not measured**, is that the MAC puts each
`|dd|` in its own offset-magnitude shell. A pair is emitted only after its
parent failed, and the range guard's 32 half-widths admits only small `|dd|`.
So a tree with many admitted `|dd| >= 1` keys is the only kind on which B1
could pay.

### Failure direction

Profiling OFF, both backends: the case passes. `admitted`, `distinct` and the
byte figures equal the profiling-ON lines exactly. Only `demanded_ops` reads
`-1`, because it is profiling-gated. `m2l_realized_keys()` is ungated, as the
exit criterion states.

### Budget rows re-calibrated

`DownwardSweep` SERIAL `default` rows from `f3cc36Kv7B4j`: 7.45, 5.41, 6.68,
7.49, 8.45, 9.61 s at np 1-6 (previously 3.84-9.16 s). np 1 is the cold first
entry of the job. The exit-criterion runs used these budgets (np 1 timeout
25 s, np 6 22 s) and took at most 9.52 s.

**Affects:**
- **B1** — B0 measured no saving on T1's fixture, so B1's payoff is unproven.
  Its exit criterion's "column count reduced by B0's measured factor"
  degenerates to "unchanged" here, so it cannot tell a working B1 from a
  no-op. Only `expectKeyTraitsAgree`'s failure direction would. Before B1
  starts, decide whether to measure the factor on a tree with many admitted
  `|dd| >= 1` keys (the downstream configuration, or a variant of this
  fixture) or to drop B1. `LaplaceKernel`'s counts above are the "unchanged"
  baseline B1 must reproduce (R3).
- **B2** — `TwoScaleFixture` now sets a root half-width, so a
  `key_needs_level` basis can be driven on it. B2's cache-retention cases can
  reuse it.
- **A1, C1** — `TwoScaleFixture` takes a `FarField` and now runs
  `CartesianTaylorBasis`. Their T1 baseline is unchanged.

## B0b

**Outcome: the duplicate factor exceeds 1.0.** On the graded draw,
`CartesianTaylorBasis` measures 1.0000-1.0403 per rank at θ 0.3 (above 1.0 on
17 of 21 ranks) and 1.0896-1.3163 at θ 0.5 (all 21). T1's two-scale draw stays
at exactly 1.0 at θ 0.3, which reproduces B0. At θ 0.5 the same draw reaches
1.1449 (14 of 21 ranks above 1.0). B0's null result was a property of that
draw **at that angle**, not of the key.

Provenance: commit `cac7036` plus this section's edits to
`tests/tstDownwardSweep.hpp` and `scripts/tuolumne/serial_runtimes.tsv`, Cray
clang 20.0.0, env `tuolumne_trilinos`, `build-tuolumne`
(`Canopy_ENABLE_PROFILING:BOOL=ON`, read from the cache and echoed at each
job's head; HIP registered at np 1-4 with
`--gpus-per-task=1 --cores-per-task=8`). Every run went through `canopy_ctest`
after a passing watchdog self-test, with the HIP environment set in a subshell
around the HIP calls only. The watchdog cancelled nothing but the self-test, and
every entry's `canopy_ctest` outcome is `completed`. Script:
`scripts/tuolumne/run_ctest_b0.flux` unchanged, submitted with `-t 8m`, because
each job takes about 3 minutes against the preamble's 15.

| job | what |
| --- | --- |
| `f3ccEbJQDASb` | `run_ctest_h0b.flux calibrate`, `DownwardSweep` SERIAL np 1-6, three passes |
| `f3ccGbJHwxPy` | `measure serial`: SERIAL np 1-6, two passes, all `rc=0` |
| `f3ccGbSC48UX` | `measure hip`: HIP np 1-4, two passes, all `rc=0` |
| `f3ccKujp6xQP` | failure direction: `measure serial` with the graded case pointed at the two-scale draw |
| `f3ccPJ3uuANT` | `measure serial` again after the revert and rebuild, all `rc=0` |

### Decisions

- **The graded draw.** $r = r_{\min}(r_{\max}/r_{\min})^u$ about the centre
  $(0.5, 0.5, 0.5)$, with $r_{\max} = 0.45$ and $r_{\min} = r_{\max}/64$, so
  six octaves, in domain units. The constants were chosen with a host-side
  Python octree model of the build (ncrit 8, max_depth 8, padding 0.1), which
  put leaves at depths 2-8. The real tree agrees: `[b0b-tree]` reports 7 leaf
  depths (2-8) at every np. T1's draw has 3-7 at θ 0.3. A position outside
  $[0, 1)^3$ is **rejected and redrawn**, not clamped, because clamping would
  pile particles onto the faces. The sphere lies inside the box at these
  constants, so the rejection never fires. The draw keeps the global count of
  1200 and per-rank seeding `42 + rank * 7919`.
- **One case, both draws, both angles.** `testGradedDdDuplicates` builds eight
  fixtures: {two-scale, graded} × θ {0.3, 0.5} × {CT order 3, Laplace}. The
  contract compares the two draws within the same binary, the same np and the
  same angle, so it reads its baseline live rather than from a constant.
- **The contract is asserted for both bases.** Both bases see the same tree,
  and the cross-level sums differ between them only because Laplace's
  canonicalization collapses `max_d`.
- **B0's `[b0-dd]` line is untouched.** `reportTwoScaleDdDuplicates` gained
  `tag` and `context` parameters, defaulting to `"b0-dd"` and `""`, and now
  returns the rank's cross-level count. B0b prints
  `[b0b-dd] draw <d> theta <t> basis ...` with the same fields. Both cases
  print the new `[dd-hist]` line, so T1's fixture at θ 0.3 gets its histogram
  twice per rank, identical both times (step 1).
- **No leaf-depth assertion.** The task's step 5 lists the case's assertions,
  and a depth floor is not among them. The leaf depths are printed on
  `[b0b-tree]` and not asserted.

### Constructor change and call sites

`TwoScaleFixture()` became
`explicit TwoScaleFixture( TwoScaleDraw draw_in = TwoScaleDraw::TwoScale, double mac_theta_in = 0.3 )`.
`mac_theta` lost its in-class `= 0.3` and is set in the member-initializer
list, ahead of `comm_plan( MPI_COMM_WORLD, mac_theta )`, so the angle reaches
the comm plan. The new `TwoScaleDraw draw` member is declared before `builder`
for the same reason. `enum class TwoScaleDraw { TwoScale, Graded }` and
`to_string( TwoScaleDraw )` sit just before the fixture. The existing sites
(`testTwoScaleTreeHasShallowAndDeepLeaves`, `testTwoScaleRefusalsAreRangeGuard`
and both of `testTwoScaleDdDuplicates`'s) compile unchanged on the defaults.
Their lines are **byte-identical** to B0's jobs: every `[two-scale]`,
`[two-scale-refusals]` and `[b0-dd]` line in `f3ccGbJHwxPy` matches
`f3cc4m4y111Z` (84 distinct lines), and every one in `f3ccGbSC48UX` matches
`f3cc4mDrvh5Z` (40).

### What only running showed

No bug surfaced. Two results were not expected beforehand:

- **θ 0.5 admits far more keys than θ 0.3.** At np 1 it admits 1312 against 162
  on T1's draw and 26 702 against 4162 on the graded one, and it also admits
  more `|dd| = 3` keys. One plausible reason, **not measured**: a tighter angle
  admits pairs at larger offsets, and more of those exceed the 32-half-width
  guard.
- **T1's draw duplicates at θ 0.5.** B0 measured it at θ 0.3 only.

### Reproducibility

Both passes print identical lines on each backend: 570 sorted record lines on
SERIAL and 276 on HIP. At np 1-4, HIP equals SERIAL line for line. The
post-revert run `f3ccPJ3uuANT` equals `f3ccGbJHwxPy` line for line in both
passes. So **R6**'s spread is zero here too, and the tables below come from a
single pass.

### Contract: summed cross-level (`dd != 0`) admitted keys

| θ | basis | np 1 | np 2 | np 3 | np 4 | np 5 | np 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.30 | CartesianTaylor | 48 → 3698 | 18 → 3022 | 70 → 3019 | 28 → 4003 | 34 → 4132 | 28 → 3729 |
| 0.30 | Laplace | 48 → 3188 | 18 → 2884 | 69 → 2723 | 28 → 3843 | 34 → 3840 | 27 → 3212 |
| 0.50 | CartesianTaylor | 614 → 23990 | 283 → 24254 | 650 → 25908 | 311 → 29178 | 518 → 30211 | 375 → 25344 |
| 0.50 | Laplace | 598 → 9688 | 282 → 11798 | 627 → 15273 | 310 → 18225 | 518 → 16632 | 373 → 14902 |

Each cell reads two-scale → graded. The graded draw admits 16-168 times as many
cross-level keys.

### Measured

Each row is one `(nprocs, rank)`. *adm* is `m2l_realized_keys().size()`, and
*dist* is the number of distinct $(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$
tuples. The factor is adm / dist, bold where it exceeds 1. Bytes use each
basis's `bytes_per_key`: 3200 for CT order 3 and 21 952 for Laplace P 6.
*hist* is the admitted-key count at signed `dd` = −3/−2/−1/0/1/2/3. Every line
reads 0 at `|dd|` 4-6, and `outside` reads 0. `demanded_ops == adm` on every
line, so the column cap never bound. The Laplace columns are the control. That
operator depends on `dd`, so its factor is **not** a saving. Its `max_d` is
canonicalized to 0, so its *dist* also merges levels.

**two-scale, θ 0.30**

| np | rank | CT adm | CT dist | CT factor | CT bytes adm → dist | CT hist dd −3..3 | L adm | L dist | L factor | L bytes adm → dist | L hist dd −3..3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 162 | 162 | 1.0000 | 518 400 → 518 400 | 0/19/5/114/5/19/0 | 162 | 162 | 1.0000 | 3 556 224 → 3 556 224 | 0/19/5/114/5/19/0 |
| 2 | 0 | 57 | 57 | 1.0000 | 182 400 → 182 400 | 0/8/1/39/1/8/0 | 57 | 57 | 1.0000 | 1 251 264 → 1 251 264 | 0/8/1/39/1/8/0 |
| 2 | 1 | 37 | 37 | 1.0000 | 118 400 → 118 400 | 0/0/0/37/0/0/0 | 37 | 37 | 1.0000 | 812 224 → 812 224 | 0/0/0/37/0/0/0 |
| 3 | 0 | 85 | 85 | 1.0000 | 272 000 → 272 000 | 0/11/4/46/3/21/0 | 80 | 80 | 1.0000 | 1 756 160 → 1 756 160 | 0/11/3/42/3/21/0 |
| 3 | 1 | 35 | 35 | 1.0000 | 112 000 → 112 000 | 0/0/0/26/5/4/0 | 35 | 35 | 1.0000 | 768 320 → 768 320 | 0/0/0/26/5/4/0 |
| 3 | 2 | 57 | 57 | 1.0000 | 182 400 → 182 400 | 0/14/6/35/2/0/0 | 53 | 53 | 1.0000 | 1 163 456 → 1 163 456 | 0/14/6/31/2/0/0 |
| 4 | 0 | 9 | 9 | 1.0000 | 28 800 → 28 800 | 0/0/0/9/0/0/0 | 9 | 9 | 1.0000 | 197 568 → 197 568 | 0/0/0/9/0/0/0 |
| 4 | 1 | 28 | 28 | 1.0000 | 89 600 → 89 600 | 0/0/0/28/0/0/0 | 28 | 28 | 1.0000 | 614 656 → 614 656 | 0/0/0/28/0/0/0 |
| 4 | 2 | 19 | 19 | 1.0000 | 60 800 → 60 800 | 0/0/0/15/3/1/0 | 18 | 18 | 1.0000 | 395 136 → 395 136 | 0/0/0/14/3/1/0 |
| 4 | 3 | 51 | 51 | 1.0000 | 163 200 → 163 200 | 0/10/4/27/1/9/0 | 50 | 50 | 1.0000 | 1 097 600 → 1 097 600 | 0/10/4/26/1/9/0 |
| 5 | 0 | 49 | 49 | 1.0000 | 156 800 → 156 800 | 0/16/1/24/1/7/0 | 49 | 49 | 1.0000 | 1 075 648 → 1 075 648 | 0/16/1/24/1/7/0 |
| 5 | 1 | 18 | 18 | 1.0000 | 57 600 → 57 600 | 0/0/0/18/0/0/0 | 18 | 18 | 1.0000 | 395 136 → 395 136 | 0/0/0/18/0/0/0 |
| 5 | 2 | 11 | 11 | 1.0000 | 35 200 → 35 200 | 0/0/0/11/0/0/0 | 11 | 11 | 1.0000 | 241 472 → 241 472 | 0/0/0/11/0/0/0 |
| 5 | 3 | 37 | 37 | 1.0000 | 118 400 → 118 400 | 0/0/0/37/0/0/0 | 37 | 37 | 1.0000 | 812 224 → 812 224 | 0/0/0/37/0/0/0 |
| 5 | 4 | 33 | 33 | 1.0000 | 105 600 → 105 600 | 0/0/0/24/0/9/0 | 33 | 33 | 1.0000 | 724 416 → 724 416 | 0/0/0/24/0/9/0 |
| 6 | 0 | 42 | 42 | 1.0000 | 134 400 → 134 400 | 0/0/0/42/0/0/0 | 42 | 42 | 1.0000 | 921 984 → 921 984 | 0/0/0/42/0/0/0 |
| 6 | 1 | 52 | 52 | 1.0000 | 166 400 → 166 400 | 0/0/0/50/0/2/0 | 52 | 52 | 1.0000 | 1 141 504 → 1 141 504 | 0/0/0/50/0/2/0 |
| 6 | 2 | 34 | 34 | 1.0000 | 108 800 → 108 800 | 0/0/0/34/0/0/0 | 34 | 34 | 1.0000 | 746 368 → 746 368 | 0/0/0/34/0/0/0 |
| 6 | 3 | 22 | 22 | 1.0000 | 70 400 → 70 400 | 0/0/0/22/0/0/0 | 22 | 22 | 1.0000 | 482 944 → 482 944 | 0/0/0/22/0/0/0 |
| 6 | 4 | 43 | 43 | 1.0000 | 137 600 → 137 600 | 0/0/0/40/1/2/0 | 37 | 37 | 1.0000 | 812 224 → 812 224 | 0/0/0/34/1/2/0 |
| 6 | 5 | 115 | 115 | 1.0000 | 368 000 → 368 000 | 0/8/6/92/5/4/0 | 109 | 109 | 1.0000 | 2 392 768 → 2 392 768 | 0/8/5/87/5/4/0 |

**two-scale, θ 0.50**

| np | rank | CT adm | CT dist | CT factor | CT bytes adm → dist | CT hist dd −3..3 | L adm | L dist | L factor | L bytes adm → dist | L hist dd −3..3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1312 | 1146 | **1.1449** | 4 198 400 → 3 667 200 | 0/12/295/698/295/12/0 | 1114 | 944 | **1.1801** | 24 454 528 → 20 722 688 | 0/12/287/516/287/12/0 |
| 2 | 0 | 652 | 651 | **1.0015** | 2 086 400 → 2 083 200 | 0/6/113/506/21/6/0 | 519 | 518 | **1.0019** | 11 393 088 → 11 371 136 | 0/6/112/374/21/6/0 |
| 2 | 1 | 445 | 443 | **1.0045** | 1 424 000 → 1 417 600 | 0/0/25/308/112/0/0 | 445 | 443 | **1.0045** | 9 768 640 → 9 724 736 | 0/0/25/308/112/0/0 |
| 3 | 0 | 730 | 720 | **1.0139** | 2 336 000 → 2 304 000 | 0/5/168/398/149/10/0 | 604 | 539 | **1.1206** | 13 259 008 → 11 832 128 | 0/5/152/288/149/10/0 |
| 3 | 1 | 423 | 422 | **1.0024** | 1 353 600 → 1 350 400 | 0/0/101/277/45/0/0 | 403 | 377 | **1.0690** | 8 846 656 → 8 275 904 | 0/0/94/264/45/0/0 |
| 3 | 2 | 408 | 408 | 1.0000 | 1 305 600 → 1 305 600 | 0/6/58/236/107/1/0 | 392 | 359 | **1.0919** | 8 605 184 → 7 880 768 | 0/6/58/220/107/1/0 |
| 4 | 0 | 232 | 232 | 1.0000 | 742 400 → 742 400 | 0/0/0/211/21/0/0 | 209 | 209 | 1.0000 | 4 587 968 → 4 587 968 | 0/0/0/188/21/0/0 |
| 4 | 1 | 364 | 364 | 1.0000 | 1 164 800 → 1 164 800 | 0/0/18/295/51/0/0 | 364 | 364 | 1.0000 | 7 990 528 → 7 990 528 | 0/0/18/295/51/0/0 |
| 4 | 2 | 331 | 331 | 1.0000 | 1 059 200 → 1 059 200 | 0/4/79/245/3/0/0 | 325 | 324 | **1.0031** | 7 134 400 → 7 112 448 | 0/4/78/240/3/0/0 |
| 4 | 3 | 531 | 531 | 1.0000 | 1 699 200 → 1 699 200 | 0/5/45/396/76/9/0 | 426 | 415 | **1.0265** | 9 351 552 → 9 110 080 | 0/5/45/291/76/9/0 |
| 5 | 0 | 439 | 430 | **1.0209** | 1 404 800 → 1 376 000 | 5/12/42/307/73/0/0 | 408 | 399 | **1.0226** | 8 956 416 → 8 758 848 | 5/12/42/276/73/0/0 |
| 5 | 1 | 261 | 256 | **1.0195** | 835 200 → 819 200 | 0/0/47/177/37/0/0 | 258 | 253 | **1.0198** | 5 663 616 → 5 553 856 | 0/0/47/174/37/0/0 |
| 5 | 2 | 233 | 219 | **1.0639** | 745 600 → 700 800 | 0/0/28/163/42/0/0 | 231 | 217 | **1.0645** | 5 070 912 → 4 763 584 | 0/0/28/161/42/0/0 |
| 5 | 3 | 365 | 345 | **1.0580** | 1 168 000 → 1 104 000 | 0/0/72/251/42/0/0 | 360 | 340 | **1.0588** | 7 902 720 → 7 463 680 | 0/0/72/246/42/0/0 |
| 5 | 4 | 486 | 483 | **1.0062** | 1 555 200 → 1 545 600 | 0/0/37/368/64/12/5 | 393 | 386 | **1.0181** | 8 627 136 → 8 473 472 | 0/0/37/275/64/12/5 |
| 6 | 0 | 297 | 296 | **1.0034** | 950 400 → 947 200 | 0/0/29/261/7/0/0 | 273 | 272 | **1.0037** | 5 992 896 → 5 970 944 | 0/0/29/237/7/0/0 |
| 6 | 1 | 401 | 399 | **1.0050** | 1 283 200 → 1 276 800 | 0/0/28/316/55/2/0 | 366 | 362 | **1.0110** | 8 034 432 → 7 946 624 | 0/0/28/281/55/2/0 |
| 6 | 2 | 269 | 269 | 1.0000 | 860 800 → 860 800 | 0/0/3/254/12/0/0 | 269 | 269 | 1.0000 | 5 905 088 → 5 905 088 | 0/0/3/254/12/0/0 |
| 6 | 3 | 284 | 284 | 1.0000 | 908 800 → 908 800 | 0/0/6/271/7/0/0 | 284 | 284 | 1.0000 | 6 234 368 → 6 234 368 | 0/0/6/271/7/0/0 |
| 6 | 4 | 373 | 371 | **1.0054** | 1 193 600 → 1 187 200 | 0/0/60/257/54/2/0 | 354 | 330 | **1.0727** | 7 771 008 → 7 244 160 | 0/0/60/238/54/2/0 |
| 6 | 5 | 513 | 511 | **1.0039** | 1 641 600 → 1 635 200 | 0/6/52/403/50/2/0 | 446 | 437 | **1.0206** | 9 790 592 → 9 593 024 | 0/6/51/338/49/2/0 |

**graded, θ 0.30**

| np | rank | CT adm | CT dist | CT factor | CT bytes adm → dist | CT hist dd −3..3 | L adm | L dist | L factor | L bytes adm → dist | L hist dd −3..3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 4162 | 4068 | **1.0231** | 13 318 400 → 13 017 600 | 0/1269/580/464/580/1269/0 | 3480 | 3208 | **1.0848** | 76 392 960 → 70 422 016 | 0/1132/462/292/462/1132/0 |
| 2 | 0 | 1461 | 1441 | **1.0139** | 4 675 200 → 4 611 200 | 0/432/244/177/239/367/2 | 1346 | 1257 | **1.0708** | 29 547 392 → 27 593 664 | 0/429/224/128/196/367/2 |
| 2 | 1 | 2005 | 1978 | **1.0137** | 6 416 000 → 6 329 600 | 2/559/299/267/273/605/0 | 1854 | 1724 | **1.0754** | 40 699 008 → 37 845 248 | 2/555/250/188/258/601/0 |
| 3 | 0 | 1143 | 1140 | **1.0026** | 3 657 600 → 3 648 000 | 0/308/283/160/108/284/0 | 1034 | 1005 | **1.0289** | 22 698 368 → 22 061 760 | 0/293/238/124/107/272/0 |
| 3 | 1 | 1266 | 1260 | **1.0048** | 4 051 200 → 4 032 000 | 0/241/108/138/257/522/0 | 1094 | 1062 | **1.0301** | 24 015 488 → 23 313 024 | 0/234/103/107/185/465/0 |
| 3 | 2 | 1025 | 1022 | **1.0029** | 3 280 000 → 3 270 400 | 0/407/191/117/190/120/0 | 918 | 882 | **1.0408** | 20 151 936 → 19 361 664 | 0/378/170/92/169/109/0 |
| 4 | 0 | 1355 | 1344 | **1.0082** | 4 336 000 → 4 300 800 | 0/289/249/176/203/435/3 | 1276 | 1190 | **1.0723** | 28 010 752 → 26 122 880 | 0/287/214/138/199/435/3 |
| 4 | 1 | 1110 | 1083 | **1.0249** | 3 552 000 → 3 465 600 | 1/227/141/146/249/346/0 | 1054 | 1003 | **1.0508** | 23 137 408 → 22 017 856 | 1/227/140/122/224/340/0 |
| 4 | 2 | 997 | 986 | **1.0112** | 3 190 400 → 3 155 200 | 2/366/188/139/116/186/0 | 919 | 888 | **1.0349** | 20 173 888 → 19 493 376 | 2/348/163/106/114/186/0 |
| 4 | 3 | 1140 | 1123 | **1.0151** | 3 648 000 → 3 593 600 | 0/353/241/138/187/221/0 | 1078 | 1014 | **1.0631** | 23 664 256 → 22 259 328 | 0/345/227/118/167/221/0 |
| 5 | 0 | 1318 | 1267 | **1.0403** | 4 217 600 → 4 054 400 | 0/266/83/133/250/586/0 | 1220 | 1149 | **1.0618** | 26 781 440 → 25 222 848 | 0/260/82/118/203/557/0 |
| 5 | 1 | 1044 | 1040 | **1.0038** | 3 340 800 → 3 328 000 | 0/240/131/119/147/407/0 | 952 | 917 | **1.0382** | 20 898 304 → 20 129 984 | 0/237/127/97/127/364/0 |
| 5 | 2 | 774 | 774 | 1.0000 | 2 476 800 → 2 476 800 | 0/188/131/77/105/273/0 | 706 | 690 | **1.0232** | 15 498 112 → 15 146 880 | 0/184/124/52/81/265/0 |
| 5 | 3 | 713 | 713 | 1.0000 | 2 281 600 → 2 281 600 | 0/347/143/54/55/114/0 | 652 | 599 | **1.0885** | 14 312 704 → 13 149 248 | 0/331/123/40/49/109/0 |
| 5 | 4 | 728 | 727 | **1.0014** | 2 329 600 → 2 326 400 | 0/432/188/62/34/12/0 | 662 | 642 | **1.0312** | 14 532 224 → 14 093 184 | 0/412/159/45/34/12/0 |
| 6 | 0 | 522 | 522 | 1.0000 | 1 670 400 → 1 670 400 | 0/180/95/62/36/149/0 | 503 | 458 | **1.0983** | 11 041 856 → 10 054 016 | 0/171/91/56/36/149/0 |
| 6 | 1 | 861 | 854 | **1.0082** | 2 755 200 → 2 732 800 | 0/195/138/82/102/344/0 | 742 | 728 | **1.0192** | 16 288 384 → 15 981 056 | 0/171/130/69/87/285/0 |
| 6 | 2 | 657 | 656 | **1.0015** | 2 102 400 → 2 099 200 | 0/251/99/56/78/173/0 | 510 | 506 | **1.0079** | 11 195 520 → 11 107 712 | 0/214/68/45/56/127/0 |
| 6 | 3 | 952 | 924 | **1.0303** | 3 046 400 → 2 956 800 | 0/214/101/99/165/373/0 | 741 | 708 | **1.0466** | 16 266 432 → 15 542 016 | 0/191/81/80/124/265/0 |
| 6 | 4 | 519 | 519 | 1.0000 | 1 660 800 → 1 660 800 | 0/266/82/31/45/95/0 | 473 | 438 | **1.0799** | 10 383 296 → 9 614 976 | 0/228/78/27/45/95/0 |
| 6 | 5 | 609 | 603 | **1.0100** | 1 948 800 → 1 929 600 | 0/207/83/61/118/140/0 | 573 | 527 | **1.0873** | 12 578 496 → 11 568 704 | 0/191/71/53/118/140/0 |

**graded, θ 0.50**

| np | rank | CT adm | CT dist | CT factor | CT bytes adm → dist | CT hist dd −3..3 | L adm | L dist | L factor | L bytes adm → dist | L hist dd −3..3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 26702 | 20286 | **1.3163** | 85 446 400 → 64 915 200 | 3165/5385/3445/2712/3445/5385/3165 | 10352 | 6790 | **1.5246** | 227 247 104 → 149 054 080 | 1956/2080/808/664/808/2080/1956 |
| 2 | 0 | 14072 | 11556 | **1.2177** | 45 030 400 → 36 979 200 | 1377/2586/2363/1582/1942/2706/1516 | 6505 | 4793 | **1.3572** | 142 797 760 → 105 215 936 | 989/1155/750/481/637/1319/1174 |
| 2 | 1 | 13343 | 10900 | **1.2241** | 42 697 600 → 34 880 000 | 1443/2653/2256/1579/1754/2354/1304 | 6311 | 4596 | **1.3732** | 138 539 072 → 100 891 392 | 924/1275/729/537/651/1246/949 |
| 3 | 0 | 10220 | 9132 | **1.1191** | 32 704 000 → 29 222 400 | 814/1725/1612/1256/1177/2062/1574 | 6166 | 4468 | **1.3800** | 135 356 032 → 98 081 536 | 750/1176/650/511/563/1207/1309 |
| 3 | 1 | 9450 | 8296 | **1.1391** | 30 240 000 → 26 547 200 | 1026/2031/1869/1200/1205/1509/610 | 5275 | 4110 | **1.2835** | 115 796 800 → 90 222 720 | 870/1211/657/422/534/1049/532 |
| 3 | 2 | 9898 | 8476 | **1.1678** | 31 673 600 → 27 123 200 | 1070/1947/1715/1204/1593/1643/726 | 5242 | 3934 | **1.3325** | 115 072 384 → 86 359 168 | 832/1088/626/477/631/1006/582 |
| 4 | 0 | 8491 | 7596 | **1.1178** | 27 171 200 → 24 307 200 | 691/1588/1697/1144/1068/1444/859 | 5108 | 4042 | **1.2637** | 112 130 816 → 88 729 984 | 609/1141/680/478/536/955/709 |
| 4 | 1 | 8619 | 7607 | **1.1330** | 27 580 800 → 24 342 400 | 799/1696/1698/1021/1098/1434/873 | 5143 | 4115 | **1.2498** | 112 899 136 → 90 332 480 | 704/1221/693/371/472/922/760 |
| 4 | 2 | 8036 | 7006 | **1.1470** | 25 715 200 → 22 419 200 | 701/1598/1708/990/1069/1335/635 | 4671 | 3676 | **1.2707** | 102 537 792 → 80 695 552 | 560/1081/716/408/494/836/576 |
| 4 | 3 | 8220 | 7371 | **1.1152** | 26 304 000 → 23 587 200 | 846/1500/1522/1033/1376/1273/670 | 5020 | 3883 | **1.2928** | 110 199 040 → 85 239 616 | 667/1079/722/460/622/879/591 |
| 5 | 0 | 7260 | 6498 | **1.1173** | 23 232 000 → 20 793 600 | 582/1541/1663/972/965/1036/501 | 4084 | 3214 | **1.2707** | 89 651 968 → 70 553 728 | 372/922/680/426/501/743/440 |
| 5 | 1 | 7072 | 6320 | **1.1190** | 22 630 400 → 20 224 000 | 583/1250/1701/1078/919/1055/486 | 3762 | 3114 | **1.2081** | 82 583 424 → 68 358 528 | 476/818/699/443/384/581/361 |
| 5 | 2 | 6879 | 6057 | **1.1357** | 22 012 800 → 19 382 400 | 567/1374/1418/826/898/1125/671 | 3420 | 2887 | **1.1846** | 75 075 840 → 63 375 424 | 377/823/573/315/319/569/444 |
| 5 | 3 | 6528 | 5813 | **1.1230** | 20 889 600 → 18 601 600 | 554/1152/1440/847/891/1096/548 | 3742 | 2959 | **1.2646** | 82 144 384 → 64 955 968 | 414/850/597/339/451/674/417 |
| 5 | 4 | 7108 | 6073 | **1.1704** | 22 745 600 → 19 433 600 | 595/1231/1316/913/1050/1328/675 | 3485 | 2772 | **1.2572** | 76 502 720 → 60 850 944 | 436/677/533/338/383/609/509 |
| 6 | 0 | 5079 | 4588 | **1.1070** | 16 252 800 → 14 681 600 | 386/841/1045/668/695/904/540 | 2744 | 2375 | **1.1554** | 60 236 288 → 52 136 000 | 326/524/509/294/253/467/371 |
| 6 | 1 | 5058 | 4642 | **1.0896** | 16 185 600 → 14 854 400 | 435/869/1025/727/654/895/453 | 3092 | 2521 | **1.2265** | 67 875 584 → 55 340 992 | 312/566/578/384/365/553/334 |
| 6 | 2 | 4931 | 4421 | **1.1154** | 15 779 200 → 14 147 200 | 418/891/998/621/756/837/410 | 2417 | 2031 | **1.1901** | 53 057 984 → 44 584 512 | 248/504/471/249/276/425/244 |
| 6 | 3 | 4561 | 4103 | **1.1116** | 14 595 200 → 13 129 600 | 441/929/1055/660/602/587/287 | 2522 | 2142 | **1.1774** | 55 362 944 → 47 021 184 | 321/617/498/281/266/343/196 |
| 6 | 4 | 4846 | 4397 | **1.1021** | 15 507 200 → 14 070 400 | 442/808/1032/646/663/793/462 | 2874 | 2365 | **1.2152** | 63 090 048 → 51 916 480 | 356/570/508/268/341/482/349 |
| 6 | 5 | 4870 | 4462 | **1.0914** | 15 584 000 → 14 278 400 | 403/909/1181/679/645/680/373 | 3072 | 2542 | **1.2085** | 67 436 544 → 55 801 984 | 315/621/599/343/399/467/328 |

### Why the factor exceeds 1.0 here

B0's parity argument still holds: a `dd == 0` key never collides with a
`dd != 0` one, so every duplicate comes from the cross-level part of each
histogram. On the graded draw that part is most of the table. At np 1 it is
3698 of 4162 keys at θ 0.3 and 23 990 of 26 702 at θ 0.5. On T1's draw at
θ 0.3 it is 48 of 162, and none of those 48 collide.

The per-`dd` collision pairs (`+k` against `−k`, or `k` against `k'`) were not
broken down. The histograms are symmetric at np 1, which fits `+k`/`−k`
mirrors, but that is **not measured**.

### Failure direction

With `testGradedDdDuplicates`'s `graded` constant set to
`TwoScaleDraw::TwoScale` (`f3ccKujp6xQP`), every SERIAL entry at np 1-6 failed
in both passes. That is 168 failures: 2 passes × 21 ranks × 2 bases × 2 angles.
All of them are the contract assertion, and no other case failed. At np 1:

```
tests/tstDownwardSweep.hpp:1997: Failure
Expected: (sum[1]) > (sum[0]), actual: 48 vs 48
CartesianTaylor at theta 0.29999999999999999: the graded draw admits no more cross-level (dd != 0) keys, summed over ranks, than T1's two-scale draw, so it tests nothing B0 did not
```

The change was reverted and both targets rebuilt. `f3ccPJ3uuANT` then passed
with lines identical to `f3ccGbJHwxPy`.

### Budget rows re-calibrated

`DownwardSweep` SERIAL `default` rows from `f3ccEbJQDASb`: 9.1, 6.12, 7.26,
8.04, 8.92 and 9.9 s at np 1-6 (previously 7.45-9.61 s). np 1 is the cold first
entry. The exit-criterion runs took at most 10.04 s, on HIP at np 1, against a
22 s budget.

**Affects:**
- **B1**: it now has a measured payoff, on the graded draw. Its step 6 and exit
  criterion must reproduce `CartesianTaylorBasis`'s admitted count falling to
  *dist*, which is a factor of **1.0000-1.0403** per rank at θ 0.3 and
  **1.0896-1.3163** at θ 0.5. At np 1 that is 4162 → 4068 and 26 702 → 20 286,
  from the `graded` tables above. Read them per `(nprocs, rank)`; they
  reproduce exactly. `LaplaceKernel`'s *adm* must stay unchanged (**R3**).
  Unlike on T1's draw at θ 0.3, its *dist* is now below *adm*, so a B1 that
  wrongly drops `dd` for Laplace would show up as a measurable drop. At the
  downstream configuration's θ 0.3, the payoff is small: about 2 % of columns
  at np 1, and at most 4 % on any rank. Whether that justifies B1 is for this
  document to decide, per B0b's "Additional information needed". B1 is not
  edited here.
- **B0's record**: "1.0 on T1's fixture" holds at θ 0.3 only. The same draw
  reaches 1.1449 at θ 0.5.
- **A1, C1, B2**: `TwoScaleFixture` now takes a draw and an angle. The graded
  draw reproduces at every np on both backends, with 7 leaf depths, and is
  available to them.

## B1

**Outcome: `CartesianTaylorBasis`'s key no longer carries `dd`, and the column
counts fall exactly as B0b predicted.** On every line its admitted count now
equals B0b's *distinct* count. Every `LaplaceKernel` line is unchanged.

Provenance: commit `a57ec2f` plus this section's edits, Cray clang 20.0.0, env
`tuolumne_trilinos`, `build-tuolumne` (`Canopy_ENABLE_PROFILING:BOOL=ON`, read
from the cache and echoed at each job's head; HIP registered at np 1-4 with
`--gpus-per-task=1 --cores-per-task=8`). Every run went through `canopy_ctest`
after a passing watchdog self-test, with the HIP environment set in a subshell
around the HIP calls only. The watchdog cancelled nothing but the self-test.
Script: `scripts/tuolumne/run_ctest_b1.flux measure|noprof|contract
serial|hip`, copied from `run_ctest_b0.flux`. `measure` runs the five MPI stems
once, then the non-MPI `CartesianTaylor` stem on the same backend, then a
second `DownwardSweep` pass. `contract` runs `FarFieldContract` and
`DownwardSweep` only. The SERIAL `measure` job took 7.5 min and the HIP one
5 min, against `-t 20m`. The `contract` jobs used `-t 10m`.

| job | what |
| --- | --- |
| `f3chRgEQqzd5` | `measure serial`: 37 entries, all `completed`, all `rc=0` |
| `f3chRgNMv9GK` | `measure hip`: 25 entries, all `completed`, all `rc=0` |
| `f3chWq7XkJbH` | failure direction: CT declares `key_needs_dd = true` and still zeroes `dd` |
| `f3chZbiYKp7h` | failure direction: CT declares `false` and keeps `dd` |
| `f3chc3fq5LqM` | failure direction: `LaplaceKernel`'s `canonicalize_key` also zeroes `dd` (R3) |
| `f3cheVoEuy9Z` | `contract serial` after the revert and rebuild, all `rc=0` |

### Decisions recorded (made before the session, not reopened)

- **B1 proceeds.** B0b measured a payoff of a few percent of columns at θ 0.3
  and 9-32 % at θ 0.5. The change is exact, and B2 depends on it.
- **`key_needs_dd` values:** `false` for `CartesianTaylorBasis`; `true` for
  `LaplaceKernel` and `MonopoleBasis`. `LevelBlindBasis` inherits
  `MonopoleBasis`'s value.
- **`m2l_key_dd_max` and the sweep's `|dd|` range guard stay exactly as they
  are**, at 6 for CT. Only the comment changed: the value now only keeps the
  fallback population comparable across bases.

### What changed

- `key_needs_dd` is a `static constexpr bool` declared beside
  `key_needs_level` on all three bases, with no default.
  `CartesianTaylorBasis::canonicalize_key` now zeroes `dd`, so its canonical
  key is $(\texttt{max\_d}, 0, \texttt{ii}, \texttt{jj}, \texttt{kk})$.
  `m2l_operator_block`, `build_m2l_operators`, the guard and
  `M2L_KEY_OFFSET_MAX` are untouched. `build_m2l_operators` never read `dd`.
- `expectKeyTraitsAgree` lost its unconditional `EXPECT_EQ( a.dd, ca.dd )`. It
  now branches on `key_needs_dd` exactly as it does on `key_needs_level`, with a
  third probe key that differs from the first in `dd` only. Its call site in
  `testLevelReachesTheKey` now also covers `LaplaceKernel<double, 6>` and
  `CartesianTaylorBasis<double, 3>`. The test file now includes
  `Canopy_CartesianTaylorBasis.hpp`. The calls run before that test's np != 1
  skip, so they gate at every np.
- `testGradedDdDuplicates` reads each basis's trait. For a `false` basis it
  asserts a cross-level count of 0 on every rank, for both draws. For a `true`
  basis it keeps B0b's graded-greater-than-two-scale contract. `[b0b-cross]`
  still prints for both bases, so CT's line now reads `two-scale 0 graded 0`.
- Comments: CT's `m2l_key_dd_max`, `canonicalize_key`, `m2l_operator_block`
  and `build_m2l_operators` blocks (including the `dd`-is-read-only-by-guards
  paragraph), and the `max_d` abort message, no longer say the key is the
  identity or that `dd` duplicates are expected. `LaplaceKernel`'s contract
  header names four members. Two of these comments were rewrapped to 80
  columns after the runs above. That was a comment-only edit, so no binary
  changed and nothing was rebuilt.

The `key_needs_dd` declaration has no reader in `src/`, unlike
`key_needs_level`. Nothing in the sweep needs it: canonicalization alone does
the work. The conformance test is the trait's only reader, and a basis that
omits the trait fails to compile only once it is passed to
`expectKeyTraitsAgree`.

### `canonicalize_key` callers, by search

`grep -rn "canonicalize_key\s*[<(]" src tests examples benchmarks`, excluding
the definitions, finds:

- `src/Canopy_DownwardSweep.hpp:1715`, the classify pass's single hash site;
- `tests/tstFarFieldContract.hpp:932-934`, `expectKeyTraitsAgree`'s three probe
  calls (two before B1).

There are no others. The definitions are on `LaplaceKernel`,
`CartesianTaylorBasis`, `MonopoleBasis` and `LevelBlindBasis`.

### Measured

Each `[b0b-dd]`, `[b0-dd]` and `[dd-hist]` line was compared by script with
B0b's `f3ccGbJHwxPy`, keyed by tag, draw, angle, basis, np and rank. That is
378 lines on SERIAL. The 180 HIP lines were compared against both
`f3ccGbSC48UX` and `f3ccGbJHwxPy`. There were no mismatches and no missing
lines:

- **CT**: admitted, `demanded_ops` and admitted bytes equal B0b's
  `distinct_no_dd` and distinct bytes, and the factor is 1.0000. Every
  `[dd-hist]` line has `cross_level 0`, `outside 0` and all keys at `dd = 0`.
- **Laplace**: every field of every line is identical to B0b, histograms
  included. Its `[b0b-cross]` sums are identical too.
- **B0's `[b0-dd]` lines** (two-scale, θ 0.3) are unchanged for both bases,
  because CT's distinct count already equalled its admitted count there.
- **Reproducibility (R6)**: both `DownwardSweep` passes in each `measure` job
  print identical lines, and HIP equals SERIAL at np 1-4. The post-revert
  `f3cheVoEuy9Z` equals `f3chRgEQqzd5` on all five count tags.

CT's columns per `(nprocs, rank)`, B0b → B1. The two-scale θ 0.30 draw is
unchanged on all 21 ranks (162 → 162 at np 1), and is omitted. Bytes are at
3200 B per column. Laplace is unchanged everywhere, and its values are in
`## B0b`.

**two-scale, θ 0.50**

| np | rank | CT adm B0b → B1 | CT bytes B0b → B1 | saved |
| --- | --- | --- | --- | --- |
| 1 | 0 | 1 312 → 1 146 | 4 198 400 → 3 667 200 | 12.7 % |
| 2 | 0 | 652 → 651 | 2 086 400 → 2 083 200 | 0.2 % |
| 2 | 1 | 445 → 443 | 1 424 000 → 1 417 600 | 0.4 % |
| 3 | 0 | 730 → 720 | 2 336 000 → 2 304 000 | 1.4 % |
| 3 | 1 | 423 → 422 | 1 353 600 → 1 350 400 | 0.2 % |
| 3 | 2 | 408 → 408 | 1 305 600 → 1 305 600 | 0 |
| 4 | 0 | 232 → 232 | 742 400 → 742 400 | 0 |
| 4 | 1 | 364 → 364 | 1 164 800 → 1 164 800 | 0 |
| 4 | 2 | 331 → 331 | 1 059 200 → 1 059 200 | 0 |
| 4 | 3 | 531 → 531 | 1 699 200 → 1 699 200 | 0 |
| 5 | 0 | 439 → 430 | 1 404 800 → 1 376 000 | 2.1 % |
| 5 | 1 | 261 → 256 | 835 200 → 819 200 | 1.9 % |
| 5 | 2 | 233 → 219 | 745 600 → 700 800 | 6.0 % |
| 5 | 3 | 365 → 345 | 1 168 000 → 1 104 000 | 5.5 % |
| 5 | 4 | 486 → 483 | 1 555 200 → 1 545 600 | 0.6 % |
| 6 | 0 | 297 → 296 | 950 400 → 947 200 | 0.3 % |
| 6 | 1 | 401 → 399 | 1 283 200 → 1 276 800 | 0.5 % |
| 6 | 2 | 269 → 269 | 860 800 → 860 800 | 0 |
| 6 | 3 | 284 → 284 | 908 800 → 908 800 | 0 |
| 6 | 4 | 373 → 371 | 1 193 600 → 1 187 200 | 0.5 % |
| 6 | 5 | 513 → 511 | 1 641 600 → 1 635 200 | 0.4 % |

**graded, θ 0.30**

| np | rank | CT adm B0b → B1 | CT bytes B0b → B1 | saved |
| --- | --- | --- | --- | --- |
| 1 | 0 | 4 162 → 4 068 | 13 318 400 → 13 017 600 | 2.3 % |
| 2 | 0 | 1 461 → 1 441 | 4 675 200 → 4 611 200 | 1.4 % |
| 2 | 1 | 2 005 → 1 978 | 6 416 000 → 6 329 600 | 1.3 % |
| 3 | 0 | 1 143 → 1 140 | 3 657 600 → 3 648 000 | 0.3 % |
| 3 | 1 | 1 266 → 1 260 | 4 051 200 → 4 032 000 | 0.5 % |
| 3 | 2 | 1 025 → 1 022 | 3 280 000 → 3 270 400 | 0.3 % |
| 4 | 0 | 1 355 → 1 344 | 4 336 000 → 4 300 800 | 0.8 % |
| 4 | 1 | 1 110 → 1 083 | 3 552 000 → 3 465 600 | 2.4 % |
| 4 | 2 | 997 → 986 | 3 190 400 → 3 155 200 | 1.1 % |
| 4 | 3 | 1 140 → 1 123 | 3 648 000 → 3 593 600 | 1.5 % |
| 5 | 0 | 1 318 → 1 267 | 4 217 600 → 4 054 400 | 3.9 % |
| 5 | 1 | 1 044 → 1 040 | 3 340 800 → 3 328 000 | 0.4 % |
| 5 | 2 | 774 → 774 | 2 476 800 → 2 476 800 | 0 |
| 5 | 3 | 713 → 713 | 2 281 600 → 2 281 600 | 0 |
| 5 | 4 | 728 → 727 | 2 329 600 → 2 326 400 | 0.1 % |
| 6 | 0 | 522 → 522 | 1 670 400 → 1 670 400 | 0 |
| 6 | 1 | 861 → 854 | 2 755 200 → 2 732 800 | 0.8 % |
| 6 | 2 | 657 → 656 | 2 102 400 → 2 099 200 | 0.2 % |
| 6 | 3 | 952 → 924 | 3 046 400 → 2 956 800 | 2.9 % |
| 6 | 4 | 519 → 519 | 1 660 800 → 1 660 800 | 0 |
| 6 | 5 | 609 → 603 | 1 948 800 → 1 929 600 | 1.0 % |

**graded, θ 0.50**

| np | rank | CT adm B0b → B1 | CT bytes B0b → B1 | saved |
| --- | --- | --- | --- | --- |
| 1 | 0 | 26 702 → 20 286 | 85 446 400 → 64 915 200 | 24.0 % |
| 2 | 0 | 14 072 → 11 556 | 45 030 400 → 36 979 200 | 17.9 % |
| 2 | 1 | 13 343 → 10 900 | 42 697 600 → 34 880 000 | 18.3 % |
| 3 | 0 | 10 220 → 9 132 | 32 704 000 → 29 222 400 | 10.6 % |
| 3 | 1 | 9 450 → 8 296 | 30 240 000 → 26 547 200 | 12.2 % |
| 3 | 2 | 9 898 → 8 476 | 31 673 600 → 27 123 200 | 14.4 % |
| 4 | 0 | 8 491 → 7 596 | 27 171 200 → 24 307 200 | 10.5 % |
| 4 | 1 | 8 619 → 7 607 | 27 580 800 → 24 342 400 | 11.7 % |
| 4 | 2 | 8 036 → 7 006 | 25 715 200 → 22 419 200 | 12.8 % |
| 4 | 3 | 8 220 → 7 371 | 26 304 000 → 23 587 200 | 10.3 % |
| 5 | 0 | 7 260 → 6 498 | 23 232 000 → 20 793 600 | 10.5 % |
| 5 | 1 | 7 072 → 6 320 | 22 630 400 → 20 224 000 | 10.6 % |
| 5 | 2 | 6 879 → 6 057 | 22 012 800 → 19 382 400 | 11.9 % |
| 5 | 3 | 6 528 → 5 813 | 20 889 600 → 18 601 600 | 11.0 % |
| 5 | 4 | 7 108 → 6 073 | 22 745 600 → 19 433 600 | 14.6 % |
| 6 | 0 | 5 079 → 4 588 | 16 252 800 → 14 681 600 | 9.7 % |
| 6 | 1 | 5 058 → 4 642 | 16 185 600 → 14 854 400 | 8.2 % |
| 6 | 2 | 4 931 → 4 421 | 15 779 200 → 14 147 200 | 10.3 % |
| 6 | 3 | 4 561 → 4 103 | 14 595 200 → 13 129 600 | 10.0 % |
| 6 | 4 | 4 846 → 4 397 | 15 507 200 → 14 070 400 | 9.3 % |
| 6 | 5 | 4 870 → 4 462 | 15 584 000 → 14 278 400 | 8.4 % |

**`CartesianTaylorSolve`'s own fixture** (`[ct-solve]`, SERIAL, against V1
(close)'s `f3cbCkKDsirK`) drops much more than the two-scale fixture.
`n_unique_ops` falls by 1.376-1.457x per rank at θ 0.3 (np 1: 26 468 → 18 588)
and by 1.331-1.441x at θ 0.5 (np 1: 7756 → 5726). `fallback_pairs` is
unchanged on every rank. HIP's `n_unique_ops` equals SERIAL's at np 1-4.

**Accuracy did not move (R10).** On SERIAL, every `[ct-solve]` deviation line
is bit-identical to `f3cbCkKDsirK`: both arms, every np. So are `MultiSolve`'s
`[multisolve-dev]`, `[multisolve-probe]`, `[fusedm2l-dev]` and
`[fusedm2l-idem]` lines. The columns are the same values, so only the table's
size changed. HIP passed every case. It is not bit-reproducible run to run
(README "Known Issues"), so its lines were not compared. Its worst CT figure is
`theta_canopy` `max_grad_dev 1.865e-02` against `3.74e-02`.

**Runtime.** No entry approached its budget. The slowest relative to budget was
`CartesianTaylorSolve` np 1, at 19.98 s against 40 s. `FarFieldContract` took
4.22-7.64 s and `DownwardSweep` 5.64-10.13 s, both within their existing rows.
No row was re-calibrated.

### Failure direction

Each change was made to the source, the two SERIAL targets were rebuilt, and
`contract serial` was run. In each run all 12 entries failed, six per stem.
Only the cases named here failed, apart from the R3 run's two L2P cases.

1. **Declared `true`, zeroes `dd`** (CT, `f3chWq7XkJbH`).
   `FarFieldContract.levelReachesTheKey` failed at every np:
   ```
   tests/tstFarFieldContract.hpp:965: Failure
   Expected equality of these values:
     a.dd
       Which is: -1
   ...
   CartesianTaylorBasis: key_needs_dd is true but canonicalize_key did not preserve dd
   tests/tstFarFieldContract.hpp:969: Failure
   Expected: (ca.dd) != (cc.dd), actual: 0 vs 0
   CartesianTaylorBasis: key_needs_dd is true but canonicalize_key maps two different dd onto the same key, so two different operators would share one column
   ```
   `ddDuplicateColumnsGraded` also failed, because CT then took the `true`
   branch: `Expected: (sum[1]) > (sum[0]), actual: 0 vs 0` for CT at both
   angles.
2. **Declared `false`, keeps `dd`** (CT, `f3chZbiYKp7h`, with `k.dd = 0`
   commented out):
   ```
   tests/tstFarFieldContract.hpp:977: Failure
   CartesianTaylorBasis: key_needs_dd is false but canonicalize_key lets two different dd through as distinct keys, so the operator table would hold one identical column per dd
   ```
   `ddDuplicateColumnsGraded` failed with `CartesianTaylor at theta <t>, draw
   <d>: key_needs_dd is false, yet a realized key carries dd != 0, so
   canonicalize_key did not reach the table`. That fired on 21 of 21 ranks for
   graded at both angles and two-scale at θ 0.5, and on 12 of 21 for two-scale
   at θ 0.3, the ranks whose B0b histogram has any `dd != 0` key.
3. **`dd` collapse visible in the counts** (R3: `LaplaceKernel` also zeroes
   `dd`, trait left `true`, `f3chc3fq5LqM`). Laplace's admitted count fell to
   exactly B0b's Laplace *distinct* count on every line. It **dropped on all 21
   ranks of both graded angles**: at np 1, 3480 → 3208 at θ 0.3 and
   10 352 → 6790 at θ 0.5. It also dropped on 17 of 21 ranks of two-scale
   θ 0.5, and stayed unchanged at two-scale θ 0.3. `expectKeyTraitsAgree`
   failed naming `LaplaceKernel` with both `true`-branch messages, and the
   contract failed `0 vs 0` for Laplace. The aliased operator also failed
   `DownwardSweep.testL2PApproximatesDirectSumAdaptiveBasic` and `...Small` at
   np 1. That is R3's "wrong velocity, not a slow one" presentation.

Each change was reverted and both targets rebuilt. `f3cheVoEuy9Z` then passed
with lines identical to `f3chRgEQqzd5`. The HIP binaries and the other SERIAL
binaries were built before the failure-direction edits, from the final source.

**Affects:**
- **B2**: `CartesianTaylorBasis`'s canonical key is now
  $(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$, with `dd` held
  at 0. B2's relabelling rewrites `max_d` on that key. It no longer has to carry
  or remap `dd`, and two cached columns cannot differ by `dd` alone. The CT
  column counts B2 starts from are this section's, not B0b's *adm*. On
  `CartesianTaylorSolve`'s drift trajectory, the per-build cache size that
  `keys_built` is read against is about 1.4x smaller than in V1's logs.
- **A2, A3**: CT's realized key count on any tree is now B0b's *dist* figure.
  A before/after comparison of CT column counts must start from this section.
- **R3**: the conformance check now runs the `false` branch on a real basis.
  The failure direction shows that an aliased `dd` on Laplace is visible both
  in the counts and in two L2P accuracy cases.

## B2

**Outcome: with `FmmConfig::quantize_root_half_width` on, the root half-width
is a power of two, and a rebuild inside one octave keeps the
`CartesianTaylorBasis` operator cache.** With the knob off, nothing moved. On
the drift trajectory, builds 2-4 rebuild 0.4-11.6 % of admitted columns per np
knob-on, against 100 % knob-off. A re-derived width fails the knob-on
direct-sum bound by 50-100x.

Provenance: commit `6de3d19` plus this section's edits, Cray clang 20.0.0, env
`tuolumne_trilinos`, `build-tuolumne` (`Canopy_ENABLE_PROFILING:BOOL=ON`, read
from the cache and echoed at each job's head; HIP registered at np 1-4 with
`--gpus-per-task=1 --cores-per-task=8`). Every run went through `canopy_ctest`
after a passing watchdog self-test, with the HIP environment in a subshell. The
watchdog cancelled nothing but the self-test. Script:
`scripts/tuolumne/run_ctest_b2.flux measure|failure serial|hip`, copied from
`run_ctest_b1.flux` with the stem list changed to the five B2 stems; `failure`
runs `CartesianTaylorSolve` only. `-t 20m` (`15m` for `failure`).

| job | what |
| --- | --- |
| `f3civx1CZ2qD` | `measure serial`, drift bounds provisional (1.0): 29 of 30 pass; `DownwardSweep` np 5 failed (below) |
| `f3civx9mCtEj` | `measure hip`, same binary: 20 of 20 pass |
| `f3civxJ6Wr79` | `run_ctest_h0b.flux calibrate`, `TreeBuilder`, `DownwardSweep`, `CartesianTaylorSolve` SERIAL np 1-6 |
| `f3cj2wdB5vSb` | failure direction: `_push_root_half_width` re-derives from `root_box()` |
| `f3cj6QNSmYfy` | exit criterion, SERIAL np 1-6: 30 of 30 `completed`, every stem `rc=0` |
| `f3cj6QWEwmd9` | exit criterion, HIP np 1-4: 20 of 20 `completed`, every stem `rc=0` |

`f3civxJ6Wr79` ran on its own node, concurrently with the two `measure` jobs,
and rewrote `serial_runtimes.tsv` near their end. Only the three calibrated
stems' rows changed, and the exit runs used the new rows.

### Decisions recorded (made before the session, not reopened)

- The knob is `FmmConfig::quantize_root_half_width`, `bool`, default `false`.
  It reaches `TreeBuilder` as a new last constructor argument (default
  `false`), so every existing `TreeBuilder` construction is unchanged.
- The root half-width has one source, `TreeBuilder::root_half_width()`: the
  value `build()` stamped on the root cell. `_root_box` stays the expanded,
  non-cubic bounding box.
- No relabelling of cached keys. `set_root_half_width`'s logic is unchanged
  (only its comment now points at `root_half_width()`). Cross-octave
  relabelling is in README "Future Optimizations", not implemented.
- No environment-variable override. Only the new cases set the knob. Only the
  `default` rows of the stems with added cases were re-calibrated.
- The drift harness is parameterized on the knob. The two existing
  `operatorCacheAcrossDrift*` cases stay knob-off, two new
  `operatorCacheAcrossDriftQuantized*` cases run knob-on, and all four assert a
  direct-sum bound measured on their own trajectory.

### What changed

- `TreeBuilder`: `_quantize_root_hw`, `_root_half_width` (domain units; its
  declaration states the up-to-2x width and the extra-level cost, R5),
  `root_half_width()`, and `static quantized_half_width(hw)`: `frexp`
  mantissa 0.5 returns `hw`, otherwise `ldexp(1.0, e)`. It returns
  non-positive and non-finite input unchanged. `build()` quantizes after the
  tolerance expansion and before stamping the root cell. The centre is the
  box centre, as before.
- `Solver`: the `FmmConfig` field, passed to the builder.
  `_push_root_half_width` is now one line,
  `_downward.set_root_half_width( _builder.root_half_width() )`.
- `TwoScaleFixture` takes a third constructor argument (the knob). Its build
  sequence moved into `rebuild_with( TreeBuilder& )`, which the constructor
  calls with `builder`. The sweeps' kernel parameters are now set before the
  first build rather than after `comm_plan.build`. They do not change between
  the two points, and every pre-existing `DownwardSweep` case passed unchanged.
- New cases: `TreeBuilder.quantizedRootHalfWidth` (step 6),
  `DownwardSweepTwoScale.rootWidthQuantizationRetainsCache` (step 5),
  `CartesianTaylorSolve.operatorCacheAcrossDriftQuantizedThetaCanopy` /
  `...ThetaRef` (step 8). The two existing drift cases now go through
  `runArm` with their own bounds.
- `with_cartesian_taylor_solve` prints `[ct-cache-inc]` after every solve in
  every arm and asserts the cache rule per build. On a changed sweep width the
  increment must equal `m2l_n_unique_ops()`. On an unchanged width it must
  equal the number of realized keys not seen since the last change. The test
  computes that count itself from `m2l_realized_keys()`. Pre-existing print
  formats are unchanged.

### Readers of the root width, by search

`grep -rn "root_box()" src tests examples`:

| site | reads | action |
| --- | --- | --- |
| `src/Canopy_Solver.hpp` `_push_root_half_width` | root width | **switched** |
| `src/Canopy_Solver.hpp` `_init_auto_softening` | box volume | kept: a box quantity |
| `src/Canopy_DownwardSweep.hpp` `set_root_half_width` comment | — | comment points at `root_half_width()` |
| `tests/tstDownwardSweep.hpp` `TwoScaleFixture` | root width | **switched** |
| `tests/tstCartesianTaylorSolve.hpp` `[ct-cache]` print | root width | **switched** |
| `tests/tstCartesianTaylorSolve.hpp` `[ct-solve]` print | root width | **switched** |
| `tests/tstRebalanceDiag.hpp:410` | longest box edge | kept: a box diagnostic |
| `tests/tstUpwardSweep.hpp:209`, `:715` | root centre only | kept: the centre does not move. `:715` reads the width from the root cell already |

`TreeBuilder::needs_rebuild` reads `_root_box` directly and is unchanged.
Every other width in `src/` is a cell's own `half_width`, which descends from
the stamped root.

### Knob off: nothing moved

A script compared the SERIAL `[ct-solve]`, `[multisolve-dev]`,
`[multisolve-probe]`, `[fusedm2l-dev]` and `[fusedm2l-idem]` lines
of `f3cj6QNSmYfy` (and of `f3civx1CZ2qD`) with `f3chRgEQqzd5`, as multisets per
stem and np. It excluded only the new lines: `arm=drift*` deviation lines and the knob-on cases'
configuration lines (root half-width a power of two). That is 198 lines
(96 / 36 / 54 / 6 / 6). There were **0 mismatches**. The pre-existing drift
cases' `[ct-cache]` increments are B1's, exactly (np 1 θ 0.5: 5388, 5464,
5560, 5726). `LaplaceSolve.bitForBitArtifacts` ran on 3 SERIAL entries and
skipped on the other 36 and on HIP, as in B1. `tests/data` was not touched.

### Retention case (`[b2-retain]`)

Two-scale draw, θ 0.3, `CartesianTaylorBasis<double, 3, 1>`, one downward
sweep. The second build's padding is 0.2 (in-octave) or 0.75 (cross-octave),
against the fixture's 0.1. Same on both backends, at every `(nprocs, rank)`:

| arm | knob | width 1 → 2 | increment |
| --- | --- | --- | --- |
| in-octave | on | 1 → 1 | **0** on 31 of 31 ranks; `n_unique_ops` unchanged |
| in-octave | off | 0.5972 → 0.6967 (np 1) | `== m2l_n_unique_ops()` (np 1: 164) |
| cross-octave | on | 1 → 2 | `== m2l_n_unique_ops()` (np 1: 54) |

### Drift cases: bounds and increments

Bounds are 2x the worst over SERIAL np 1-6 and HIP np 1-4, rounded up at the
third figure. The gradient is the worst field in all four. Every figure agrees
across np and backend to 12 significant figures, and the exit runs reproduced
the measure runs.

| case | knob | worst `max_grad_dev` | worst `max_pot_dev` | bound | ratio |
| --- | --- | --- | --- | --- | --- |
| `operatorCacheAcrossDriftThetaCanopy` | off | 1.8799053015e-02 | 1.0217956014e-03 | 3.76e-02 | 2.000 |
| `operatorCacheAcrossDriftThetaRef` | off | 9.5675103599e-03 | 3.3292179793e-04 | 1.92e-02 | 2.007 |
| `operatorCacheAcrossDriftQuantizedThetaCanopy` | on | 1.8211525767e-02 | 1.0861234562e-03 | 3.65e-02 | 2.004 |
| `operatorCacheAcrossDriftQuantizedThetaRef` | on | 7.6584280953e-03 | 2.1453247968e-04 | 1.54e-02 | 2.011 |

Both drift arms run at p = 2, so the θ 0.3 figures are not comparable to the
p = 3 gating arm's.

`keys_built` increments per build, summed over ranks, SERIAL (`f3cj6QNSmYfy`).
HIP's np 1-4 sums are identical. Builds 1-4 are the four solves:

| np | θ | knob off | knob on | knob-on share of builds 2-4 |
| --- | --- | --- | --- | --- |
| 1 | 0.5 | 5388, 5464, 5560, 5726 | 4108, 12, 14, 52 | 0.6 % |
| 2 | 0.5 | 8951, 9045, 9176, 9386 | 7492, 17, 17, 55 | 0.4 % |
| 3 | 0.5 | 12281, 12385, 12536, 12757 | 10725, 716, 154, 186 | 3.2 % |
| 4 | 0.5 | 15359, 15453, 15614, 15839 | 13341, 798, 1054, 86 | 4.7 % |
| 5 | 0.5 | 18368, 18438, 18498, 18737 | 15915, 1898, 827, 590 | 6.6 % |
| 6 | 0.5 | 21225, 21329, 21530, 21700 | 17188, 2059, 2306, 749 | 8.6 % |
| 1 | 0.3 | 17824, 17926, 18128, 18588 | 12662, 138, 154, 290 | 1.5 % |
| 2 | 0.3 | 28859, 28971, 29254, 29890 | 21926, 414, 335, 354 | 1.6 % |
| 3 | 0.3 | 39189, 39338, 39662, 40440 | 30989, 1255, 837, 1124 | 3.3 % |
| 4 | 0.3 | 48096, 48222, 48595, 49394 | 37061, 5167, 2058, 1271 | 7.1 % |
| 5 | 0.3 | 56613, 56557, 57021, 56610 | 43832, 7687, 3402, 1583 | 8.9 % |
| 6 | 0.3 | 63518, 63629, 64074, 66080 | 46448, 8150, 6408, 4203 | 11.6 % |

Knob off, every build's increment equals its admitted count. Knob on, the sweep
width reads 0.25 on every build, so no build after the first changes it.
Summed over np 1-6, builds 2-4 rebuild 5.3 % (θ 0.5) and 7.1 % (θ 0.3) of
admitted columns.

### What only running revealed

1. **The knob-on drift increment is not 0.** Step 8 says it "must read 0 on
   every step whose quantized root half-width did not change". The width never
   changed, yet every build made some columns. They were exactly the keys the
   moved tree realized for the first time on that rank. The root centre drifts
   with the box, and the rebalance and migrate flows move cells between ranks.
   The share grows with np (0.4-0.6 % at np 1-2, 8.6-11.6 % at np 6), which
   points at ownership moves more than geometry. The harness asserts that
   sharper rule: the increment equals the first-seen count. It held on 336 of
   336 builds per backend. The design's literal 0 holds in the retention case,
   where the particles are identical.
2. **Quantizing changes the tree a lot, not just by one level (R5).** On
   `CartesianTaylorSolve`'s fixture the root goes from 0.1386 to 0.25 (1.80x).
   `max_depth = 6` then caps the deepest cells at 1.80x their knob-off width,
   and the first build admits 4108 columns against 5388 at np 1 θ 0.5 (12 662
   against 17 824 at θ 0.3). On the two-scale draw the root goes from 0.597 to
   1.0 and np 1 admits 54 columns against 162. R5 predicted a stable key count
   with shifted values. Under a binding `max_depth` the count falls instead, and
   the accuracy figures move (drift θ 0.5 gradient 1.880e-2 → 1.821e-2). The
   knob-on figures belong to a different tree. They are not the knob-off
   figures with the cache kept.
3. **A per-rank vacuity guard is wrong on the knob-on two-scale tree.** At np 5
   rank 2 owns no admitted pair at all, in either build. The first `measure`
   job failed `DownwardSweep` np 5 on `EXPECT_GT( unique1, 0 )` alone. Every
   cache assertion passed. The guard is now on the all-rank sum, as T1's
   `range_guard` assertion is.
4. **In the failure direction the cache counters look healthy.** With the width
   re-derived, the sweep is handed a changed width every build. So the cache
   clears and rebuilds in full, and `[ct-cache-inc]` satisfies the cache rule.
   Only the direct-sum bound catches it, which is R4's and R9's point.

### Failure direction — `f3cj2wdB5vSb`

`_push_root_half_width` was made to re-derive the width from `root_box()`
again. Only `Canopy_Test_CartesianTaylorSolve_MPI_SERIAL` was rebuilt, and it
was run alone. All six entries failed, each on exactly the two knob-on drift
cases. The knob-off drift cases and both gating arms passed. Every np printed
the same figures:

```
[ct-solve] theta=0.5 ... arm=drift_q_theta_canopy direct_softened_sum max_pot_dev=0.71170570742428541 max_grad_dev=1.9328138902251875 tol=0.036499999999999998
tests/tstCartesianTaylorSolve.hpp:1025: Failure
Expected: (max_pot_dev) < (tol), actual: 0.71170570742428541 vs 0.0365
tests/tstCartesianTaylorSolve.hpp:1033: Failure
Expected: (max_grad_dev) < (tol), actual: 1.9328138902251875 vs 0.0365
[ct-solve] theta=0.3 ... arm=drift_q_theta_ref ... max_pot_dev=0.57436289851158484 max_grad_dev=1.5977382783903646 tol=0.0154
```

The sweep was told 0.1386-0.1356 while the tree was built at 0.25. The
octave-crossing half of the failure direction is the retention case's
cross-octave arm, which passes in every run.
`src/Canopy_Solver.hpp` was then restored from a copy with its mtime, so no
other target went stale. The CT SERIAL object was deleted and rebuilt. A
`make` of all ten targets then rebuilt only that object. The exit runs used
those binaries.

### Budget rows re-calibrated

`f3civxJ6Wr79`, `default` rows, max of three passes (s, np 1-6):

| stem | before | after |
| --- | --- | --- |
| `TreeBuilder` | 3.53, 4.17, 5.2, 6.12, 6.9, 7.86 | 3.26, 3.98, 5.01, 5.89, 6.73, 7.62 |
| `DownwardSweep` | 9.1, 6.12, 7.26, 8.04, 8.92, 9.9 | 5.58, 6.15, 7.57, 8.33, 9.27, 10.2 |
| `CartesianTaylorSolve` | 18.86, 12.4, 11.65, 11.2, 11.53, 11.88 | 25.34, 16.14, 14.72, 14.24, 14.47, 14.95 |

`DownwardSweep` np 1's old 9.1 s was a cold first entry. In the calibration
it is not first. The exit runs' tightest entry was `TreeBuilder` np 1, the cold
first entry of each job: 9.68 s of 12 s on SERIAL and 10.18 s of 12 s on HIP.

**Affects:**
- **A2**: balancing must read the root cell's width from
  `TreeBuilder::root_half_width()` (or the cells' `half_width`), never from
  `root_box()`. That is the expanded, non-cubic box. The root cell is the cube
  of half-width `root_half_width()` about its centre: it equals the box's
  largest half extent with the knob off, and is up to 2x wider with it on.
  `needs_rebuild` deliberately stays on the box.
- **A1, C1**: their occupied-depth and fallback figures are knob-off. With the
  knob on, every per-depth width scales by the quantized/unquantized ratio
  (1.67x on the two-scale draw, 1.80x on `CartesianTaylorSolve`'s fixture). The
  deepest occupied depth can then rise by one on an uncapped tree, or
  the leaves can fill up at `max_depth` on a capped one, as measured here. Their
  numbers must be re-measured knob-on, not translated by one level.
- **R5**: under a binding `max_depth` the realized key count is not stable
  across the snap (np 1: 162 → 54, 5388 → 4108). Read a count change on a
  knob change as a tree change, not as a cache defect.
- **A3 / the default**: the knob stays off. Before anything turns it on by
  default, `MultiSolve`'s and both gating arms' figures must be re-pinned on the
  knob-on tree. Its accuracy is different, not worse by construction.
