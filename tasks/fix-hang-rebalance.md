# The np-3 `MultiSolve` hang, and `AutoRebalance`'s excess deviation

**Status:** IN PROGRESS

## Problem

Two defects in the `MultiSolve` stem block `tree-opt.md` task V1, which
re-derives the bounds the tree/key optimization chains are verified against.
V1 resumes once both are resolved.

**1. `Canopy_Test_MultiSolve_MPI_SERIAL_np_3` hangs intermittently.** Over the
14 np-3 runs with surviving logs, 5 hung and 9 completed. No other rank count
has hung. The three hangs run with `ctest -V` (`f3bkWYfR6Ao9`, `f3bn8EK66YaK`,
and H1's `f3bnfasqQDAo`) all stopped in `MultiSolve.LargeMotion_Rebuild`, the
only case that calls `Solver::rebuild` on every step. The other two hangs,
`f3XHShznrAEs` and `f3XUPJqSuqdh`, ran without `-V`, so their stall point is
unrecorded.

The hang is also **not contained**. `ctest --timeout` kills the `flux run`
client, but the flux sub-job it launched keeps running and keeps the node
`--exclusive`. Every later rank count then waits in state `S` and times out
without starting. In `f3bn8EK66YaK` the np-3 sub-job was still `R` at 19.5 min
while np 4-6 waited, and cancelling it let np 4 finish in 11 s. One hang
therefore costs the rest of the pass, which is how three measurement rank counts
were lost.

**2. `MultiSolve.AutoRebalance` exceeds the far-field truncation floor.** The
floor at `MultiSolveTest::P_ORDER = 8` (`tests/tstMultiSolve.hpp:54`) and
`theta = 0.5` is $\theta^{P+1} = 1.95 \times 10^{-3}$, a bound on the relative
gradient error. The integrator can only shrink it in velocity. Its max relative
velocity deviation from the brute-force trajectory is `6.41e-3` to `6.57e-3` at
np 5 and `1.71e-2` at np 6 (3.3x and 8.8x the floor), and its position deviation
is `3.15e-3` at np 6, over three passes. It grows ~2000x from np 1 to np 6. By
contrast, the direct per-solve gradient error that
`SolveFusedM2L.matchesPriorReference` measures (P = 6, uniform, no time
stepping) is flat at 7e-5 to 3e-4 over np 1-6. Sweeping `CANOPY_MAC_THETA`
shows the deviation is far-field-driven: it shrinks at 0.4 and grows at 0.7,
more steeply than the floor ratio. The per-site figures are in
`tree-opt-progress-log.md` section V1.

**End state.**
- np 3 completes reliably.
- A hang anywhere in a Canopy flux script is cancelled and captured, never
  propagated to later rank counts.
- The AutoRebalance excess is attributed to one named mechanism and resolved:
  either the far field is fixed, or the derivation V1 reads against is corrected
  with the measurement that justifies it.

**Out of scope.**
- `SolveFusedM2L.FP32_smokeTest` (disabled; its own README entry).
- The `SingleSolve`-coupled np-3 deadlock. README "Known Issues" records it as
  occurring only when `SingleSolve` shares the ctest process. Every task here
  runs `MultiSolve` alone.
- Making Zoltan2's partition deterministic. H2 records whether the host-node
  partition reproduces, and goes no further.
- Moving any `MultiSolve` bound. That is V1's work.

## Approach

Each defect gets **measure first, then fix**. Nothing found so far names a
mechanism for either, and a fix written without one cannot show it fixed the
right thing.

**The hang.** H1 makes the hang survivable and observable. A shared watchdog
cancels any flux sub-job older than 300 s, and **before cancelling it takes
`gstack` of every rank's process**. H1 then reproduces the hang under the
watchdog. At ~1 in 3 per np-3 run, 20 runs miss it with probability
$(2/3)^{20} \approx 3 \times 10^{-4}$. H2 reads the stacks, names the mechanism,
fixes it, and turns the precondition that was violated into a loud check.

**The deviation.** E1 adds a per-step probe that compares the FMM gradient to a
brute-force gradient **at the FMM's own current positions**. That error is the
far field's alone, with no trajectory in it. The probe runs beside the existing
end-of-run trajectory comparison. The two quantities separate the candidate
mechanisms:

| mechanism | per-solve error (E1 probe) | trajectory deviation | np 1 at global N = 1200 |
| --- | --- | --- | --- |
| **(a) trajectory amplification**: a close encounter turns a small force difference into a large velocity difference | at or below the floor on every step | concentrated on particles with a close encounter | shows the excess too (N-driven) |
| **(b) multi-rank maintenance defect**: a rebuild/rebalance at np >= 2 leaves the far field wrong | above the floor on steps following a Rebuild/Rebalance, np >= 2 only | follows the bad steps | clean |
| **(c) far field at this geometry**: e.g. a box inflated by an ejected particle with refinement capped at `max_depth` | above the floor | follows | shows the excess too |

E2 acts on E1's classification. Under (a) the code is correct and the
derivation is not: E2 records that in V1 and changes no `src/`. Under (b) or
(c), E2 fixes `src/`, with the per-solve probe as the gauge.

**A lead E1 must read, not assume.** AutoRebalance's global tree shrinks over
the run once it has taken a Rebuild step. At np 6 the cell count goes
207 → 130 → 60 → … → 36 with 1200 particles and `ncrit = 16`, while at np 1
(200 particles) it stays at 53-79 (`[Canopy diag] auto_maintain` lines, job
`f3bmo4JYikKh`). The tree is replicated identically on every rank, built from
all-reduced counts (`src/Canopy_TreeBuilder.hpp:740-770`) inside a bounding box
that is itself all-reduced (`:345-346`). So the shrinking count means the
particles' extent grew, not that cells were lost. That is consistent with an
ejected particle (mechanisms a and c) and says nothing about (b).

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Diagnostic switches | environment variables read in `tests/tstMultiSolve.hpp`, **default off**, one helper each beside `get_test_mac_theta()` (`:38-43`) | `CANOPY_MAC_THETA` is the existing precedent. The suite has no `DISABLED_` tests. An env switch adds no ctest entry and no gtest case, so the `regression` gate (`tests/CMakeLists.txt:65-67`) stays unchanged. |
| Switch names | `CANOPY_MULTISOLVE_PROBE` (`1` enables the per-step probe), `CANOPY_MULTISOLVE_NPP` (positive integer; overrides `num_particles_per_rank` at every site) | One name per quantity, prefixed by the stem it affects. |
| Probe output | one line per step on rank 0, tag `[multisolve-probe]`, printed with `%.17g` | Matches `[multisolve-dev]` (`:644`). A line must be diffable across runs. |
| Normalization | report **both** the per-particle max relative error and the field-scale error $\max_i \lvert\Delta g_i\rvert / \max_i \lvert g_i\rvert$ | Per-particle relative error is inflated wherever $\lvert g_i\rvert$ cancels, the way `matchesPriorReference`'s potential figure is (V1 log). The field-scale rule is the suite's own: `field_scales` (`tests/tstLaplaceSolve.hpp:1162-1184`), whose comment makes the same cancellation argument. Reading one figure without the other is how a normalization artifact gets reported as a defect. |
| Floor | $\theta^{P+1}$, `1.95e-3` at `theta = 0.5`, `P_ORDER = 8` | The figure every task here reads against, stated once. |
| Watchdog | `scripts/tuolumne/flux_watchdog.sh`, **sourced** by a batch script after setting `WATCHDOG_S`; starts a background loop; `watchdog_stop` ends it, and `watchdog_wait_idle` blocks until no sub-job is running | A copy per script drifts. Call `watchdog_wait_idle` after any ctest that may have timed out: ctest returns before the watchdog has stacked and cancelled the hung sub-job. |
| Watchdog threshold | `WATCHDOG_S=300`, equal to ctest `--timeout` | ~20x the slowest observed np runtime (15.4 s), so a slow pass is never cancelled. |
| Batch preamble | copied verbatim from `scripts/tuolumne/run_ctest_v1.flux:40-65`, plus `flux_watchdog.sh` | Includes the static-TLS workaround, without which every binary aborts before `main`. Walltime option is `--time-limit` (≤ 60 min on pdebug); `--flags=waitable` is rejected; wait with `flux job status <id>`. |
| Provenance | every job echoes `spack env status`, `CC --version`, commit SHA, `git status --porcelain`, submit command and the build's `Canopy_ENABLE_PROFILING` from `CMakeCache.txt` | A number with no provenance cannot be re-derived. |
| Build directory | `build-tuolumne/` (profiling ON); check the cache, not the script | A profiling-OFF build reads `-1` from every per-reason counter, indistinguishable from a reading. |
| Determinism | report per `(nprocs, rank)` from **two** runs, and state whether they agree | The partition is nondeterministic at np >= 3 (README "Known Issues"). V1 measured the accuracy figures as reproducible to ≤ 2.5 %. |
| Failure behavior | a violated precondition throws or aborts with a message naming it; never a silent repair, a retry, or a longer timeout | A hang "fixed" by retrying is still present. |
| Isolation from E1 | H2 works in its own git worktree at `../Canopy-h2`, branch `fix-hang-h2` cut from `investigate-m2l-cap`, with its own `build-tuolumne-h2/` configured by `run_cmake_tuolumne.sh`. Its job scripts set `CANOPY_SRC` and `CANOPY_BUILD` to that checkout. It pushes `fix-hang-h2` and does not merge it into `investigate-m2l-cap`. | E1 edits `tests/tstMultiSolve.hpp` and rebuilds `build-tuolumne/tests/Canopy_Test_MultiSolve_MPI_SERIAL` in the main checkout, and may do so while H2 runs. A shared tree would compile each task's uncommitted edits into the other's binary. Every existing flux script hardcodes the main checkout's paths. |
| Formatting | never run clang-format | `CLAUDE.md`. |
| Comments | units, signs and ranges on every declaration added | Probe quantities are relative or absolute depending on normalization. Say which. |

### Deliberate deviations

- **The probe compares gradients at the FMM's positions, not at the brute-force
  shadow's.** The existing check (`:484-498` integrates the shadow, `:591-660`
  compares) diverges from the FMM trajectory by construction. Evaluating brute
  force where the FMM particles actually are removes the trajectory from the
  error, and that separation is the purpose of the probe. The shadow comparison
  is kept unchanged beside it.
- **The hang is captured by stack sampling, not reproduced in a debugger
  session.** At ~1 in 3 it takes tens of ctest runs to see one. An interactive
  allocation is the wrong tool, and a sampling watchdog also contains the
  hang.

## Current state

- `scripts/tuolumne/flux_watchdog.sh` stacks, then cancels, any sub-job older
  than `WATCHDOG_S`. `run_ctest_v1.flux` and `run_ctest_h1.flux` source it;
  every other script has no watchdog.
- ctest launches each MPI test as `flux run --ntasks N --nodes=1 --exclusive
  --cores-per-task=1` through the `MPIEXEC_*` overrides in
  `run_cmake_tuolumne.sh:10-12`. The batch script runs on the same single node
  as its sub-jobs, so `gstack` there sees every rank. `kernel.yama.ptrace_scope`
  is `0` on the compute nodes, and `gstack` attaches to the test ranks
  (`fix-hang-rebalance-progress-log.md` section H1).
- `Solver::rebuild` (`src/Canopy_Solver.hpp:399-404`) is `_full_setup`
  (`:549-635`). It rebuilds the tree, re-runs the Zoltan2 partition
  (`TreePartitioner::partition_leaves`, `src/Canopy_TreePartitioner.hpp:314-440`),
  then sorts, refreshes ownership, rebuilds the comm plan and sets up all three
  sweeps. `partition_leaves` solves on rank 0 only and broadcasts a
  `num_leaves`-long assignment (`:431`). It **assumes, without checking**, that
  every rank holds the identical leaf set (`:347-361`).
- `auto_maintain` (`src/Canopy_Solver.hpp:422-520`) takes `rebuild` when a
  particle escaped the box, `rebalance` when the cell-key set changed, and
  `migrate` otherwise. AutoRebalance takes all three over its 8 steps at every
  rank count. At np 5-6 the sequence is Rebalance, Rebuild, Rebuild, then
  Rebalance for the remaining five steps (`[Canopy diag] auto_maintain`
  lines on stderr, profiling builds only, job `f3bmo4JYikKh`).
- `testMultiStepGravity` (`tests/tstMultiSolve.hpp:154`) draws particles per
  rank with seed `42 + rank * 7919` (`:208`), so global N is
  `200 * nprocs`. N and the rank count move together at every site.
- The `[multisolve-dev]` report (`:644`) prints each site's end-of-run
  `max_pos_rel` and `max_vel_rel` on every run. No per-step or per-solve
  gradient error is printed anywhere in the stem.
- `MultiSolve` sets `cfg.softening = 0.0` (`:344`), matching the unsoftened
  brute-force reference, so a close pair is unsoftened in both.

## Progress log

`fix-hang-rebalance-progress-log.md` holds what each session measured, decided
and found by running. Consult it before implementing a task, changing a
signature, or reopening a question this document treats as settled.

## Task sequence

### H1 — Contain the np-3 hang and capture its stacks — **DONE**

**Depends on:** none.
**Fill in:** new `scripts/tuolumne/flux_watchdog.sh`;
`scripts/tuolumne/run_ctest_v1.flux` (replace its inline loop at `:69-90` and
`:107` with the sourced helper); new `scripts/tuolumne/run_ctest_h1.flux`;
README "Known Issues", the np-3 hang entry.
**Reference:** the inline loop at `scripts/tuolumne/run_ctest_v1.flux:69-90`.
**Do:**
1. Write `flux_watchdog.sh`. Every 15 s it lists running sub-jobs of the
   enclosing instance (`flux jobs --filter=running --no-header -o '{id}
   {runtime}'`). For each one older than `WATCHDOG_S`, it first runs `gstack`
   on that sub-job's own processes, then `flux cancel`s it. "Its own" means
   descendants of that job's `flux-shell` on this node whose process name
   matches `CANOPY_WATCHDOG_PGREP` (default `Canopy_Test_`), matched on the
   name and not with `pgrep -f`. A node-wide name match also catches the
   watchdog's own `sleep 15`, and a full-command-line match also catches
   `ctest` (its `-R` regex) and the `flux run` client. Each capture goes to stdout between
   `### watchdog stacks <subjob> ###` markers, with PID, runtime and a
   timestamp. It exports `watchdog_stop` to kill the loop.
2. `run_ctest_h1.flux` does two things in one allocation. **Self-test:** set
   `CANOPY_WATCHDOG_PGREP=sleep`, launch `flux run --ntasks=3 --nodes=1
   --exclusive sleep 900` in the background, and wait for it. It must be
   cancelled between 300 and 330 s with three stacks captured, and an
   immediately following `flux run --ntasks=4 ... hostname` must start. Then,
   with the default pattern, loop up to 20 times over
   `ctest -V --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'`, and
   stop at the first run the watchdog cancels.
3. Record the stack of every rank of the captured hang in the log, plus the
   last gtest case and the last `[multisolve-dev]` line before it.

**Exit criterion:** one `run_ctest_h1.flux` job whose log shows:
- the self-test sub-job cancelled between 300 and 330 s, with exactly three
  `sleep` stacks, each a child of the self-test sub-job's shell, and the np-4
  follow-on started;
- at least one real np-3 hang cancelled, with a non-empty `gstack` for each of
  its three ranks.

Failure direction: the self-test proves the watchdog cancels and captures. If
20 np-3 runs complete with no hang, record that as the finding, with its
$(2/3)^{20}$ odds, and stop; do not run more to force one.

**Met.** Job `f3bnfasqQDAo`. The self-test sub-job was cancelled at a runtime
of 303.6 s with exactly three `sleep` stacks, each a direct child of its
`flux-shell` (`matched=3 children=3 nonempty=3`), and the np-4 `hostname`
follow-on then ran. In the np-3 loop, run 1 completed and run 2 hung in
`MultiSolve.LargeMotion_Rebuild`. The watchdog cancelled it at 302.2 s with
non-empty `gstack` captures of all three ranks. The stacks are verbatim in the
progress log, section H1.

### H2 — Name the hang's mechanism and fix it — **NOT STARTED**

**Depends on:** H1 **DONE**.
**Fill in:** `src/Canopy_TreePartitioner.hpp`: the Zoltan2 adapter type at
`:367`. A copy of `scripts/tuolumne/run_ctest_h1.flux` with `CANOPY_SRC` and
`CANOPY_BUILD` pointing at the H2 checkout (Conventions, "Isolation from E1").
README "Known Issues": remove the np-3 hang entry when fixed, and add an entry
for Zoltan2 still running on HIP under a HIP `ExecutionSpace`.
**Reference:** the collectives on the rebuild path. These are the bounding-box
`MPI_Allreduce`s (`src/Canopy_TreeBuilder.hpp:345-346`), the per-depth count
`MPI_Allreduce` (`:743`), `partition_leaves`'s `MPI_Bcast`
(`src/Canopy_TreePartitioner.hpp:431`) and the migration `MPI_Alltoall`
(`:613`).
**Do:** H1's stacks (progress log, section H1) show one rank inside
`Zoltan2::PartitioningProblem::solve`, stuck in `hipDeviceSynchronize` called
from a `Kokkos::deep_copy` in `AlgMJ::mj_get_new_cut_coordinates`. The other two
ranks wait in `partition_leaves`'s `MPI_Bcast` (`:431`). Every Zoltan2 frame is
on `KokkosDeviceWrapperNode<Kokkos::HIP>`. That is because `:367` builds the
adapter on `Tpetra::Map<int, int64_t>`, whose default node is HIP, even in
`TreePartitioner<Kokkos::HostSpace, Kokkos::Serial>`. This Trilinos instantiates
Tpetra for Serial and OpenMP nodes too (`HAVE_TPETRA_INST_SERIAL`,
`HAVE_TPETRA_INST_OPENMP`, `TpetraCore_config.h:136,138` in the spack view).
1. **Leading candidate: Zoltan2 on the wrong execution space.** Template the
   adapter's `Tpetra::Map` on
   `Tpetra::KokkosCompat::KokkosDeviceWrapperNode<ExecutionSpace>`, so Zoltan2
   runs where the partitioner runs. `static_assert` that the node's execution
   space equals `ExecutionSpace`. A HIP `ExecutionSpace` still runs Zoltan2 on
   HIP. That case is out of scope and goes in README "Known Issues".
2. The stacks did not show these alternatives. Fall back to them only if the
   leading candidate fails its exit criterion:
   - **Ranks in different collectives, or one collective with different
     counts**, e.g. disagreement on `num_leaves` at `:431`. The fix is a loud
     check that `num_leaves` agrees across ranks before the broadcast.
   - **All ranks inside Kokkos/HIP**, not MPI. This shares a signature with the
     `SingleSolve` deadlock (README). Record it and decide with the user before
     changing anything: that entry places the cause outside `MultiSolve`.
3. Run the solve twice at np 3-6 on the host node and state in the log whether
   the leaf partition reproduces. Do not pursue determinism: that stays out of
   scope.

The fix must remove the cause. A retry, a longer timeout or a skipped case is
not a fix.

The node change moves the partition at np >= 3 (R3). H2's log section states
that the partition path changed, and that any E1 or V1 figure at np >= 3
measured before `fix-hang-h2` merges is stale.

**Exit criterion:** both directions.
- **Fixed:** 15 consecutive
  `ctest --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'` runs
  complete under the watchdog with no cancellation. Unfixed at ~1 in 3, that
  passes by chance with probability $(2/3)^{15} \approx 2 \times 10^{-3}$.
- **Checked:** with the node change temporarily reverted, the
  `run_ctest_h1.flux` 20-run np-3 loop reproduces at least one hang that the
  watchdog cancels. At ~1 in 3 it misses one with probability
  $(2/3)^{20} \approx 3 \times 10^{-4}$. If the reverted build does not hang in
  20 runs, the result is inconclusive: stop and report.

### E1 — Classify the AutoRebalance excess — **NOT STARTED**

**Depends on:** H1 **DONE**. Its watchdog keeps an np-3 hang from costing
np 4-6 of a measurement pass.
**Fill in:** `tests/tstMultiSolve.hpp`: two helpers beside `get_test_mac_theta`
(`:38-43`); a per-step probe in the time loop of `testMultiStepGravity`
(`:354` onward, after `solve()` and before the integrate kernel); the
`CANOPY_MULTISOLVE_NPP` override where `num_particles_per_rank` is consumed.
New `scripts/tuolumne/run_ctest_e1.flux`.
**Reference:** `brute_force_gradient` (`:75-107`); the end-of-run GlobalId
gather and alignment (`:524-595`), which the probe repeats per step;
`field_scales` (`tests/tstLaplaceSolve.hpp:1162-1184`).
**Do:**
1. With `CANOPY_MULTISOLVE_PROBE=1`, gather positions, charges, FMM gradient
   and GlobalId to rank 0 after each `solve()`. Compute the brute-force gradient
   **at those positions**, then print one `[multisolve-probe]` line per step:
   - `case`, `nprocs`, `step`, and the maintenance action the previous step took;
   - global cell count and root half-width (`solver.builder().cells().size()`,
     `root_box()`);
   - per-particle max relative gradient error; field-scale error
     $\max_i \lvert\Delta g_i\rvert / \max_i \lvert g_i\rvert$;
   - the argmax particle's GlobalId, $\lvert g\rvert$, and its minimum
     separation to any other particle.

   At end of run, print the GlobalId of the `max_vel_rel` particle and the
   minimum separation it reached over the run. Default off: with the variable
   unset, nothing new prints and the run does the same work as before.
2. Run AutoRebalance's configuration at np 1-6, twice. Then run np 1 with
   `CANOPY_MULTISOLVE_NPP=1200`, the same global N as np 6 though not the same
   particles, and np 1 with `CANOPY_MAC_THETA=0.7`.
3. Classify against the Approach table, as (a), (b) or (c), or none of them,
   with the figures that decide it. Define "close encounter" as a minimum
   separation below one tenth of the mean spacing $(0.8^3/N)^{1/3}$, and record
   the number.

**Exit criterion:** the `run_ctest_e1.flux` log has `[multisolve-probe]` lines
for every step at np 1-6 in two runs, plus the two np-1 variants, and the log
records one classification with its deciding figures. Both directions:
- **Measuring:** at np 1, `CANOPY_MAC_THETA=0.7` raises the per-step
  field-scale error above its theta-0.5 value. This shows the probe measures
  the far field.
- **Inert when off:** with the probe unset, the np 1-2 `[multisolve-dev]`
  figures of all six sites match the V1 table (`tree-opt-progress-log.md`
  section V1) to every printed digit. np 1-2 are deterministic.

### E2 — Resolve the excess per E1's classification — **NOT STARTED**

**Depends on:** E1 **DONE**.
**Fill in:** under (a), `tasks/tree-opt.md` V1 step 1, the derivation
paragraph, and `tree-opt-progress-log.md`, with no `src/` and no bound change.
Under (b) or (c), the `src/` path E1 names. Both branches update README
"Known Issues", the `1e-8` entry.
**Reference:** E1's log section.
**Do:**
- **(a) amplification.** The far field is correct. Rewrite V1's derivation so
  it covers trajectory amplification at close encounters, citing E1's probe
  figures. A trajectory bound at such a site measures dynamics, not the far
  field; record that. Whether such a site should gate on the per-solve probe
  instead is V1's decision. Do not make it here.
- **(b) or (c) defect.** Fix the far field on the path E1 named, without
  touching any bound.

**Additional information needed:** E1's classification. The (b)/(c) fix
cannot be designed before it.

**Exit criterion:**
- **Under (a):** V1's derivation states the amplification mechanism with E1's
  figures, and every probe step at np 1-6 has field-scale error ≤ `1.95e-3`.
- **Under (b) or (c):** in two runs at np 1-6, AutoRebalance's per-step
  field-scale error is ≤ `1.95e-3` on every step, and its `max_vel_rel` at
  np 5-6 is below `1.95e-3`. The failure direction is the specific condition E1
  identified: reintroduced in a temporary revert, it makes the probe exceed the
  floor again.

## Known risks

**R2 — Observation changes the hang rate.** The watchdog's polling, or `-V`
output, shifts timing, and 20 runs see no hang. Presentation: H1's loop
completes clean. That is indistinguishable from a hang that went away on its
own, so record it as "not reproduced at 20" and never as "fixed".
Distinguishing measurement: the rate in surviving logs, 4 hangs in 12 runs,
spans runs with and without `-V`.

**R3 — H2's fix moves every np >= 3 accuracy figure.** Moving Zoltan2 onto the
host node changes the partition. Every `[multisolve-dev]` figure V1 recorded at
np >= 3 is then stale, and so is any E1 figure at np >= 3 measured in the main
checkout before `fix-hang-h2` merges. Presentation: V1 or E1 resumes and its
np >= 3 figures no longer reproduce. Response: H2's log section states that the
partition path changed. E1 and V1 re-measure np >= 3 on the merged tree before
classifying or pinning anything.

**R4 — Amplification and a defect present at once.** Mechanism (a) is real at
some particles while (b) or (c) inflates others. Presentation: E1 finds a close
encounter at the `max_vel_rel` particle and stops there. Distinguishing
measurement: the per-step field-scale error must sit at or below the floor on
**every** step before (a) alone is the classification. A close encounter does
not excuse a bad solve.

**R5 — A diagnostic switch leaks into a gate.** `CANOPY_MULTISOLVE_NPP` set in
a gate script changes what all six sites test while still printing plausible
figures. Presentation: V1's figures drift with no code change. Response: E1's
inert-when-off criterion, and the `[multisolve-dev]` line carries `nsteps` and
the case label but not N. When V1 resumes, compare its np-1 figures with the
V1 table before trusting a run.
