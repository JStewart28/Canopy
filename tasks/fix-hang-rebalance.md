# The np-3 `MultiSolve` hang, and `AutoRebalance`'s excess deviation

**Status:** IN PROGRESS

## Problem

Two defects in the `MultiSolve` stem block `tree-opt.md` task V1, which
re-derives the bounds the tree/key optimization chains are verified against.
V1 resumes once both are resolved.

Production runs Canopy on the HIP backend, so every defect here is resolved on
both backends the test suite builds on Tuolumne: SERIAL at np 1-6 and HIP at
np 1-4, one APU per rank. A result measured on one backend says nothing about
the other.

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
- `MultiSolve` completes reliably at SERIAL np 1-6 and at HIP np 1-4.
- HIP MPI tests register at np 1-4 and launch with one APU per rank; SERIAL
  registration and launch are unchanged.
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
- The OPENMP test variants, and HIP above np 4. A node has four APUs, so HIP
  np 5-6 cannot run one APU per rank.

## Approach

Each defect gets **measure first, then fix**. Nothing found so far names a
mechanism for either, and a fix written without one cannot show it fixed the
right thing.

**The HIP baseline.** No HIP test has run in any of this work. H0a makes the
HIP MPI tests runnable under ctest with one APU per rank. H0c builds every HIP
target named by an exit criterion in this document or in `tree-opt.md`, and
records what each one does at np 1-4 before anything is changed. Every later
task reads its HIP figures against H0c.

**Time budgets.** Waiting on test runs is the bottleneck. A default-configuration
SERIAL entry completes in 4.5-24 s (Current state), yet a hang was waited out
for 300 s. H0b sets each ctest entry's timeout and watchdog threshold at
1.75x the maximum completed SERIAL runtime of the same `(stem, np)`. That
applies to every backend. An entry that runs past its budget is hung or
otherwise wrong, and it is stacked and cancelled like a hang. The calibration
in H0b is the last 300 s wait.

**The hang.** H1 makes the hang survivable and observable. A shared watchdog
cancels any flux sub-job older than its threshold (300 s in H1, the per-entry
budget after H0b), and **before cancelling it takes
`gstack` of every rank's process**. H1 then reproduces the hang under the
watchdog. At ~1 in 3 per np-3 run, 20 runs miss it with probability
$(2/3)^{20} \approx 3 \times 10^{-4}$. H2 reads the stacks, names the mechanism,
fixes it, and turns the precondition that was violated into a loud check.
H1 and H2 each carry a SERIAL arm, met, and a HIP arm. H2's SERIAL fix leaves a
HIP solver running Zoltan2 on HIP, the device H1's SERIAL stacks stalled on.

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
| Backends | every exit criterion runs SERIAL at np 1-6 **and** HIP at np 1-4: `Canopy_Test_<Stem>_MPI_SERIAL_np_[1-6]` and `Canopy_Test_<Stem>_MPI_HIP_np_[1-4]`; a non-MPI stem runs `Canopy_Test_<Stem>_SERIAL` and `Canopy_Test_<Stem>_HIP` | Production runs on HIP. A node has four APUs, so HIP at one APU per rank stops at np 4. |
| HIP launch | through ctest only, after H0a: the HIP entries carry `--gpus-per-task=1 --cores-per-task=8` | The same binding as `scripts/tuolumne/run_treepartitioner_hip.flux`. A HIP test launched by hand bypasses what H0a registers. |
| HIP environment | the GPU variables of `systems/tuolumne/claude.md` section 4 (`MPICH_GPU_SUPPORT_ENABLED=1`, `GTL_HSA_VSMSG_CUTOFF_SIZE=4096`, `FI_CXI_ATS=0`, `HSA_XNACK=1`, `MPICH_SMP_SINGLE_COPY_MODE=NONE`) set with `env` on the HIP `ctest` command only, never exported for the whole script | Device buffers reach MPI only with GPU-aware MPICH. Keeping them off the SERIAL commands keeps SERIAL runs in the environment their baselines (`canopy-v1.f3bmo4JYikKh.log`, H2's jobs) were measured in. |
| Diagnostic switches | environment variables read in `tests/tstMultiSolve.hpp`, **default off**, one helper each beside `get_test_mac_theta()` (`:38-43`) | `CANOPY_MAC_THETA` is the existing precedent. The suite has no `DISABLED_` tests. An env switch adds no ctest entry and no gtest case, so the `regression` label's registration (`tests/CMakeLists.txt:61-63`) stays unchanged. |
| Switch names | `CANOPY_MULTISOLVE_PROBE` (`1` enables the per-step probe), `CANOPY_MULTISOLVE_NPP` (positive integer; overrides `num_particles_per_rank` at every site) | One name per quantity, prefixed by the stem it affects. |
| Probe output | one line per step on rank 0, tag `[multisolve-probe]`, printed with `%.17g` | Matches `[multisolve-dev]` (`:644`). A line must be diffable across runs. |
| Normalization | report **both** the per-particle max relative error and the field-scale error $\max_i \lvert\Delta g_i\rvert / \max_i \lvert g_i\rvert$ | Per-particle relative error is inflated wherever $\lvert g_i\rvert$ cancels, the way `matchesPriorReference`'s potential figure is (V1 log). The field-scale rule is the suite's own: `field_scales` (`tests/tstLaplaceSolve.hpp:1162-1184`), whose comment makes the same cancellation argument. Reading one figure without the other is how a normalization artifact gets reported as a defect. |
| Floor | $\theta^{P+1}$, `1.95e-3` at `theta = 0.5`, `P_ORDER = 8` | The figure every task here reads against, stated once. |
| Watchdog | `scripts/tuolumne/flux_watchdog.sh`, **sourced** by a batch script after setting `WATCHDOG_S`; starts a background loop; `watchdog_stop` ends it, and `watchdog_wait_idle` blocks until no sub-job is running | A copy per script drifts. Call `watchdog_wait_idle` after any ctest that may have timed out: ctest returns before the watchdog has stacked and cancelled the hung sub-job. |
| Time budget | per ctest entry, `ceil(1.75 * t_ref_s)` s, where `t_ref_s` is the `(stem, np, config)` row of `scripts/tuolumne/serial_runtimes.tsv`: the maximum completed SERIAL runtime over three calibration passes. The same budget is the ctest `--timeout` and the watchdog threshold, applied by `canopy_ctest` (H0b), for every backend. | An entry past 1.75x its SERIAL time is hung or wrong, and finding out at 1.75x saves the rest of 300 s. The maximum rather than the mean is used because the first pass in a job can run ~4 s slow at np 1. |
| Budget rows | a run under a diagnostic switch (`CANOPY_MULTISOLVE_PROBE`, `CANOPY_MULTISOLVE_NPP`, `CANOPY_MAC_THETA`) sets `CANOPY_BUDGET_CONFIG` to a name for that configuration, and needs rows calibrated under it, as in H0b step 1. A task that adds a case to a stem, or changes its SERIAL runtime, re-calibrates that stem's rows in the same change. No row means no run. | Switch runs do different work: the θ = 0.7 sweep ran `MultiSolve` np 1 in 17.7 s against 8.84 s at default. A default row would cancel them. A stale row either cancels a correct run or waits on a hang. |
| Batch preamble | copied verbatim from `scripts/tuolumne/run_ctest_v1.flux:40-65`, plus `flux_watchdog.sh` | Includes the static-TLS workaround, without which every binary aborts before `main`. Walltime option is `--time-limit` (≤ 60 min on pdebug); `--flags=waitable` is rejected; wait with `flux job status <id>`. |
| Provenance | every job echoes `spack env status`, `CC --version`, commit SHA, `git status --porcelain`, submit command and the build's `Canopy_ENABLE_PROFILING` from `CMakeCache.txt` | A number with no provenance cannot be re-derived. |
| Build directory | `build-tuolumne/` (profiling ON) for both backends; check the cache, not the script | A profiling-OFF build reads `-1` from every per-reason counter, indistinguishable from a reading. |
| Determinism | report per `(nprocs, rank)` and per backend from **two** runs, and state whether they agree | The partition is nondeterministic at np >= 3 on a HIP node (README "Known Issues"); on the host node it reproduces (H2). Device reductions can also reorder floating-point sums run to run, so a HIP figure is never assumed bit-stable until H0c measures it. |
| Failure behavior | a violated precondition throws or aborts with a message naming it; never a silent repair, a retry, or a longer timeout | A hang "fixed" by retrying is still present. |
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
  than `WATCHDOG_S`. `run_ctest_v1.flux`, `run_ctest_h1.flux` and
  `run_ctest_h2.flux` source it; every other script has no watchdog. Its
  default process pattern, `Canopy_Test_`, matches the HIP binaries as well.
- ctest launches each MPI test as `flux run --ntasks N --nodes=1 --exclusive
  --cores-per-task=1` through the `MPIEXEC_*` overrides in
  `run_cmake_tuolumne.sh:10-12`. The batch script runs on the same single node
  as its sub-jobs, so `gstack` there sees every rank. `kernel.yama.ptrace_scope`
  is `0` on the compute nodes, and `gstack` attaches to the test ranks
  (`fix-hang-rebalance-progress-log.md` section H1).
- **Those flags and the rank list are shared by every backend.**
  `Canopy_add_tests` registers every device's MPI tests at the one rank list
  `CANOPY_TEST_MPI_RANKS_EFFECTIVE` (`cmake/test_harness/test_harness.cmake:73-80`,
  `:89`) with the one `${MPIEXEC_PREFLAGS}` (`:122`). So
  `Canopy_Test_<Stem>_MPI_HIP_np_[1-6]` are registered today and launch with no
  GPU binding; at np 5-6 a node's four APUs cannot give one per rank. The only
  HIP script that has run, `scripts/tuolumne/run_treepartitioner_hip.flux`,
  bypasses ctest with `--gpus-per-task=1 --cores-per-task=8` at np 1 and 4.
- **Every timeout is 300 s, and the watchdog has one threshold.** The scripts
  pass `ctest --timeout 300` (e.g. `run_ctest_h2.flux`) and set
  `WATCHDOG_S=300`. `flux_watchdog.sh` applies that one threshold to every
  sub-job and polls every `WATCHDOG_POLL_S=15` s. Completed SERIAL runtimes in
  the default configuration are far shorter. The table below gives the maximum
  per `(stem, np)` over the repo-root job logs, excluding the T4-diagnostic,
  LaplaceSolve-trace and θ-gain jobs, which ran non-default work. Sample counts
  are in parentheses:

  | stem | np 1 | np 2 | np 3 | np 4 | np 5 | np 6 |
  | --- | --- | --- | --- | --- | --- | --- |
  | `MultiSolve` | 8.84 (12) | 7.01 (12) | 12.96 (25) | 12.21 (9) | 15.65 (9) | 16.26 (9) |
  | `CartesianTaylorSolve` | 24.38 (10) | 13.84 (10) | 11.31 (9) | 11.15 (9) | 11.31 (9) | 11.69 (9) |
  | `LaplaceSolve` | 11.94 (44) | 9.04 (34) | 7.67 (32) | 8.01 (31) | 8.72 (31) | 8.65 (30) |
  | `DownwardSweep` | 7.83 (10) | 5.41 (10) | 6.61 (10) | 8.08 (10) | 8.57 (9) | 9.65 (9) |
  | `FarFieldContract` | 7.36 (13) | 4.48 (12) | 5.63 (12) | 6.75 (12) | 7.82 (12) | 8.66 (12) |

  These are seconds per ctest entry, launch and `Kokkos::initialize` included.
  `TreeBuilder`, `TreePartitioner`, `CommunicationPlan` and `CartesianTaylor`
  have no SERIAL ctest timing in any log. The figures span builds and commits,
  so they size the budgets; H0b's calibration sets them.
- **No HIP test has run for this work.** No log in the repo root contains an
  `MPI_HIP` test. The HIP binaries in `build-tuolumne/tests` predate H2, and
  `Canopy_Test_CartesianTaylorSolve_MPI_HIP` and
  `Canopy_Test_FarFieldContract_MPI_HIP` have never been built, so whether they
  compile is unknown.
- **A HIP solver still runs Zoltan2 on HIP.** `partition_leaves` builds its
  adapter on `KokkosDeviceWrapperNode<ExecutionSpace>`
  (`src/Canopy_TreePartitioner.hpp:374-383`), so for
  `TEST_EXECSPACE = Kokkos::Experimental::HIP`
  (`cmake/test_harness/TestHIP_Category.hpp`) MJ runs on the device H1's SERIAL
  stacks stalled on. Its inputs are host `std::vector`s (`:319-333`), and it
  solves on rank 0 only over a `Teuchos::SerialComm` (`:405-406`, `:437-447`).
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

### H0a — Register HIP MPI tests at np 1-4, one APU per rank — **NOT STARTED**

**Depends on:** none.
**Fill in:** `cmake/test_harness/test_harness.cmake` (`Canopy_add_tests`, the
rank list at `:73-80` and `:89`, the `add_test` launch at `:121-123`);
`CMakeLists.txt` (document the two per-device variables beside
`Canopy_TEST_MPI_RANKS`, `:205-208`); `run_cmake_tuolumne.sh`;
`systems/tuolumne/claude.md` section 5, "Preferred: drive the suite with
CTest"; README "Run with CTest", the rank-count paragraph.
**Reference:** the HIP binding in
`scripts/tuolumne/run_treepartitioner_hip.flux` (`--gpus-per-task=1
--cores-per-task=8`); the existing `MPIEXEC_MAX_NUMPROCS` filter
(`test_harness.cmake:73-85`), which a per-device list goes through unchanged.
**Do:**
1. Add two optional per-device overrides, read inside the `_device` loop:
   `Canopy_TEST_MPI_RANKS_<DEVICE>` replaces `Canopy_TEST_MPI_RANKS` for that
   device's MPI tests, and `Canopy_TEST_MPIEXEC_PREFLAGS_<DEVICE>` replaces
   `MPIEXEC_PREFLAGS`. Unset or empty means the shared value, so every other
   system and every other device registers exactly as before. A per-device rank
   list passes through the same `MPIEXEC_MAX_NUMPROCS` filter and warning.
2. In `run_cmake_tuolumne.sh`, set `-DCanopy_TEST_MPI_RANKS_HIP="1;2;3;4"` and
   `"-DCanopy_TEST_MPIEXEC_PREFLAGS_HIP=--nodes=1;--exclusive;--gpus-per-task=1;--cores-per-task=8"`,
   with a comment saying why HIP stops at 4.
3. Reconfigure `build-tuolumne/` with the wrapper. Document the variables in
   README and the systems doc.

**Exit criterion:** in `build-tuolumne/` after reconfiguring:
- `ctest -N -R '_MPI_HIP_np_'` lists only `_np_1` to `_np_4` entries, and
  `ctest --show-only=json-v1` shows `--gpus-per-task=1` on every
  `_MPI_HIP_np_` command;
- **SERIAL unchanged:** the `_MPI_SERIAL_` entries and their full command lines
  in `ctest --show-only=json-v1` are identical to the same dump taken before
  the change (save both to the log's directory and `diff` them);
- inside a one-node allocation, `flux run --ntasks=4 --nodes=1 --exclusive
  --gpus-per-task=1 printenv ROCR_VISIBLE_DEVICES` prints four distinct
  devices, and the same command at `--ntasks=5` is refused by flux as
  unsatisfiable. That refusal is why HIP stops at np 4.

### H0b — Per-test time budgets from measured SERIAL runtimes — **NOT STARTED**

**Depends on:** none.
**Fill in:** new `scripts/tuolumne/serial_runtimes.tsv`; new
`scripts/tuolumne/ctest_budget.sh`; `scripts/tuolumne/flux_watchdog.sh` (the
per-sub-job threshold and the poll period); new
`scripts/tuolumne/run_ctest_h0b.flux`.
**Reference:** the default-configuration SERIAL runtimes under **Current
state**; `flux_watchdog.sh`'s cancel test (`"${rt%.*}" -gt "${WATCHDOG_S}"` in
`watchdog_start`) and its inputs block.
**Do:**
1. **Calibrate.** In one allocation on `build-tuolumne/`, run every SERIAL
   entry of every stem named by H0c step 1 three times, without `-V`, at
   `--timeout 300` under the watchdog. These are the only 300 s runs the
   budgets allow. Write one row per `(stem, np)` to `serial_runtimes.tsv`:
   `stem`, `np` (1 for a non-MPI stem), `config` (`default`), `t_ref_s`, and the
   job id. `t_ref_s` is the **maximum** completed runtime over the three
   passes, not the mean. The first pass of a job can be ~4 s slower at np 1
   (H2 repro, 8.84 s then 4.86 s), and a mean would put that outlier over
   budget. A calibration run that does not complete is a hang. Record it and
   stop; never write a row for it.
2. **`ctest_budget.sh`**, sourced after `flux_watchdog.sh`, defines
   `canopy_ctest <anchored-regex> [ctest args…]`. It lists the matching entries
   with `ctest -N -R`. For each entry it parses `(stem, backend, np)` from the
   name and looks up the row for `(stem, np, ${CANOPY_BUDGET_CONFIG:-default})`.
   It then runs that entry alone as `ctest --timeout <budget> -R '^<entry>$'
   [ctest args…]`, followed by `watchdog_wait_idle`. The budget is
   `ceil(1.75 * t_ref_s)` seconds, for every backend including SERIAL. The
   function prints one line per entry: `entry`, runtime, budget, and outcome
   `completed`, `failed` or `over-budget`. **No row means no run**: the
   function exits non-zero naming the missing `(stem, np, config)` before
   launching anything. There is no default budget.
3. **Watchdog.** `canopy_ctest` exports the current entry's budget to the
   watchdog through a file in `${WATCHDOG_DIR}`, and the loop cancels a sub-job
   older than that budget. `WATCHDOG_S` remains the threshold only for
   sub-jobs that `canopy_ctest` did not launch (the self-test). Lower
   `WATCHDOG_POLL_S` to 2, so a cancel lands within 2 s of the budget instead of
   15. Stacks are still taken before the cancel.
4. Every exit-criterion `ctest` line in this document and in `tree-opt.md` is
   run through `canopy_ctest` with the same regex. A SERIAL entry that goes over
   budget is itself an outcome to report: either the stem got slower, or it hung.

**Exit criterion:** the `run_ctest_h0b.flux` log shows:
- `serial_runtimes.tsv` holds one `default` row per `(stem, np)` named by H0c
  step 1: np 1-6 for each MPI stem, np 1 for `CartesianTaylor`. Each row comes
  from three completed calibration passes;
- the self-test, with `WATCHDOG_S=20` and a 3-rank `sleep 900`, cancelled
  between 20 and 25 s with three stacks;
- **no false positive:** two
  `canopy_ctest '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'` passes finish
  with every entry `completed` or `failed` and none `over-budget`;
- **refusal:** `canopy_ctest '^Canopy_Test_UpwardSweep_MPI_SERIAL_np_1$'`, a stem
  with no row, exits non-zero naming `(UpwardSweep, 1, default)` and launches
  no sub-job.

### H0c — Build every named HIP target and record its baseline — **NOT STARTED**

**Depends on:** H0a **DONE**, H0b **DONE**.
**Fill in:** new `scripts/tuolumne/run_ctest_h0.flux`, which sources the
watchdog; README "Known Issues", one entry per HIP target that does not compile
or fails a case its SERIAL twin passes.
**Reference:** `scripts/tuolumne/run_ctest_h1.flux` for the watchdog self-test
and the batch preamble.
**Do:**
1. In `build-tuolumne/`, build the HIP target of every stem named by an exit
   criterion in this document or in `tree-opt.md`:
   `Canopy_Test_{MultiSolve,DownwardSweep,TreeBuilder,TreePartitioner,CommunicationPlan,LaplaceSolve,CartesianTaylorSolve,FarFieldContract}_MPI_HIP`
   and `Canopy_Test_CartesianTaylor_HIP`. A target that does not compile is
   recorded with its first error and skipped. Do not fix it here.
2. In one allocation: run H0b's watchdog self-test, then each built MPI stem
   once at `_MPI_HIP_np_[1-4]` and the non-MPI stem once, all through
   `canopy_ctest`. Run `MultiSolve` with `-V`
   **twice**, so its `[multisolve-dev]` lines at np 1-4 are recorded from two
   passes. Split the stems across jobs if one exceeds the 60-min `pdebug`
   limit.
3. Record in the log a table per `(stem, np)`: compiled, passed, failed (which
   cases), or over budget (cancelled, with its stacks and the budget). For each failed case, give the
   SERIAL outcome of the same case: SERIAL failures already in README "Known
   Issues" are carried, not new. State whether `MultiSolve`'s HIP
   `[multisolve-dev]` lines agree between the two passes at each np.

**Exit criterion:** the `run_ctest_h0.flux` log(s) show H0b's self-test
passing and an outcome for every `(stem, np)` named in step 1, with every
over-budget entry stacked on all its ranks. The log has the table and the `MultiSolve` two-pass comparison. README
"Known Issues" has an entry for each HIP-only compile failure, case failure or
over-budget entry.
The log's **Affects** line names every task whose HIP arm a recorded failure
blocks.

### H1 — Contain the np-3 hang and capture its stacks — **REOPENED**

SERIAL arm **met**; HIP arm **not started**. **Fill in** through the SERIAL
exit criterion below are the SERIAL arm's; the HIP arm follows them.

**Depends on:** none for the SERIAL arm. HIP arm: H0c **DONE**.
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

**SERIAL exit criterion:** one `run_ctest_h1.flux` job whose log shows:
- the self-test sub-job cancelled between 300 and 330 s, with exactly three
  `sleep` stacks, each a child of the self-test sub-job's shell, and the np-4
  follow-on started;
- at least one real np-3 hang cancelled, with a non-empty `gstack` for each of
  its three ranks.

Failure direction: the self-test proves the watchdog cancels and captures. If
20 np-3 runs complete with no hang, record that as the finding, with its
$(2/3)^{20}$ odds, and stop; do not run more to force one.

**HIP arm.**
1. Add a backend argument to `run_ctest_h1.flux` (`serial`, the default, or
   `hip`). Under `hip` the loop runs
   `canopy_ctest '^Canopy_Test_MultiSolve_MPI_HIP_np_3$'` with the HIP
   environment, and also every HIP rank count at which H0c recorded a
   `MultiSolve` entry over budget. The self-test is H0b's.
2. Record the stacks of every rank of each captured HIP hang, plus the last
   gtest case and last `[multisolve-dev]` line before it, as in step 3.

**HIP exit criterion:** one `run_ctest_h1.flux hip` job whose log shows H0b's
self-test passing and either:
- at least one HIP `MultiSolve` run cancelled over budget, with a non-empty
  `gstack` for every rank; or
- 20 clean runs at each looped rank count, recorded as "not reproduced at 20"
  and never as "no hang" (**R2**).

**Met (SERIAL).** Job `f3bnfasqQDAo`. The self-test sub-job was cancelled at a runtime
of 303.6 s with exactly three `sleep` stacks, each a direct child of its
`flux-shell` (`matched=3 children=3 nonempty=3`), and the np-4 `hostname`
follow-on then ran. In the np-3 loop, run 1 completed and run 2 hung in
`MultiSolve.LargeMotion_Rebuild`. The watchdog cancelled it at 302.2 s with
non-empty `gstack` captures of all three ranks. The stacks are verbatim in the
progress log, section H1.

### H2 — Name the hang's mechanism and fix it — **REOPENED**

SERIAL arm **met**; HIP arm **not started**. **Fill in** through the SERIAL
exit criterion below are the SERIAL arm's; the HIP arm follows them.

**Depends on:** H1's SERIAL arm for the SERIAL arm. HIP arm: H1's HIP arm
**met**.
**Fill in:** `src/Canopy_TreePartitioner.hpp`: the Zoltan2 adapter type at
`:367`. New `scripts/tuolumne/run_ctest_h2.flux`, which sources the
watchdog and runs against `build-tuolumne/`.
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
that the partition path changed. V1's recorded np >= 3 figures are then stale,
and E1 measures on the post-H2 partition.

**SERIAL exit criterion:** both directions.
- **Fixed:** 15 consecutive
  `ctest --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'` runs
  complete under the watchdog with no cancellation. Unfixed at ~1 in 3, that
  passes by chance with probability $(2/3)^{15} \approx 2 \times 10^{-3}$.
- **Checked:** with the node change temporarily reverted, the
  `run_ctest_h1.flux` 20-run np-3 loop reproduces at least one hang that the
  watchdog cancels. At ~1 in 3 it misses one with probability
  $(2/3)^{20} \approx 3 \times 10^{-4}$. If the reverted build does not hang in
  20 runs, the result is inconclusive: stop and report.

**HIP arm.**
**Fill in:** the `src/` path H1's HIP stacks name; `scripts/tuolumne/run_ctest_h2.flux`
(a backend argument, as for H1); README "Known Issues", the entry "Zoltan2
runs on HIP when the solver's `ExecutionSpace` is HIP": remove it when fixed,
or rewrite it to state what the HIP stacks showed.
**Reference:** H1's HIP stacks (progress log, section H1); the SERIAL mechanism
and fix in the progress log, section H2; the collectives on the rebuild path
listed under **Reference** above, at their current lines
(`src/Canopy_TreePartitioner.hpp:447` for the `MPI_Bcast`).
**Do:**
1. Read H1's HIP stacks and name the mechanism, as the SERIAL arm did. The
   design proposes no fix in advance. If the stacks match a fallback in step 2
   above, that fallback's rule applies. That includes the "all ranks inside
   Kokkos/HIP" case, which needs a decision with the user before any change.
2. Fix the cause. The SERIAL fix stays: whatever changes, `nm -C` on
   `Canopy_Test_MultiSolve_MPI_SERIAL` still shows no `AlgMJ` symbol on a HIP
   node.
3. Run the HIP solve twice at np 3-4 and state in the log whether the leaf
   partition reproduces.

If H1's HIP arm reproduced no hang, there is nothing to name. Skip steps 1-2,
record that in the log, and run only the **Fixed** direction below.

**HIP exit criterion:** both directions.
- **Fixed:** 15 consecutive
  `canopy_ctest '^Canopy_Test_MultiSolve_MPI_HIP_np_3$'` runs, and 15 at each
  other rank count H1's HIP arm hung at, finish with no entry over budget. So do
  two `canopy_ctest '^Canopy_Test_MultiSolve_MPI_HIP_np_[1-4]$'` passes. In the
  same job, 15 `canopy_ctest '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'` runs
  stay within budget, so the HIP fix has not reopened the SERIAL hang.
- **Checked:** with the HIP fix temporarily reverted, the `run_ctest_h1.flux hip`
  loop reproduces at least one hang that the watchdog cancels. If 20 reverted
  runs do not hang, the result is inconclusive: stop and report.

**Met (SERIAL).** Zoltan2 MJ was running on Tpetra's default HIP node. It now runs on the
partitioner's `ExecutionSpace`, through `Zoltan2::BasicUserTypes`; templating
`Tpetra::Map` alone does not reach Zoltan2's `node_t` (progress log, section
H2). **Fixed:** job `f3bnvGYaSqWP`, 15 of 15 np-3 runs completed with no
watchdog cancellation. **Checked:** job `f3bnyu3peHWw`, on the reverted build,
hung on run 1 and was cancelled at 302.3 s, with the same stack signature as H1.
Job `f3bnvGg3MDo5`: two np 1-6 passes print identical `[multisolve-dev]` lines,
and the np 1-2 lines match `canopy-v1.f3bmo4JYikKh.log`.

### E1 — Classify the AutoRebalance excess — **NOT STARTED**

**Depends on:** H1 **DONE**, both arms: its watchdog keeps an np-3 hang from costing
np 4-6 of a measurement pass. H2 **DONE**, both arms: H2 changes the np >= 3
partition that E1 measures, and a HIP pass that hangs cannot be measured. H0c
**DONE**: its HIP `[multisolve-dev]` lines are E1's HIP baseline.
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
2. First calibrate budget rows for each switch configuration this step runs
   (`probe`, `probe-npp1200`, `probe-theta0.7`), per H0b step 1, at the rank
   counts it runs them. Then run AutoRebalance's configuration at SERIAL np 1-6
   and at HIP np 1-4, twice each. Then run SERIAL np 1 with `CANOPY_MULTISOLVE_NPP=1200`, the
   same global N as np 6 though not the same particles, and SERIAL np 1 with
   `CANOPY_MAC_THETA=0.7`.
3. Classify against the Approach table, as (a), (b) or (c), or none of them,
   with the figures that decide it. Compare the HIP per-step field-scale error
   with SERIAL's at the same np and step. If it differs by more than the
   two-run spread on either backend, that is a HIP-specific far-field defect
   and is recorded as a second classification beside the first. Define "close encounter" as a minimum
   separation below one tenth of the mean spacing $(0.8^3/N)^{1/3}$, and record
   the number.

**Exit criterion:** the `run_ctest_e1.flux` log has `[multisolve-probe]` lines
for every step at SERIAL np 1-6 and HIP np 1-4 in two runs each, plus the two
np-1 variants. The log records the classification with its deciding figures,
and the HIP-against-SERIAL comparison. Both directions:
- **Measuring:** at np 1, `CANOPY_MAC_THETA=0.7` raises the per-step
  field-scale error above its theta-0.5 value. This shows the probe measures
  the far field.
- **Inert when off:** with the probe unset, the 12 np 1-2 `[multisolve-dev]`
  lines (six sites, two rank counts) match the corresponding lines of
  `canopy-v1.f3bmo4JYikKh.log` (repo root, untracked; V1's step-1 job) character
  for character. np 1-2 are deterministic: all three passes of that job print
  each of the 12 identically. The V1 table in `tree-opt-progress-log.md` is
  rounded to three figures and cannot decide this. On HIP, with the probe
  unset, the np 1-2 `[multisolve-dev]` lines match H0c's character for
  character if H0c found them identical between its two passes. Otherwise they
  fall within H0c's measured spread, stated in the log.

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
- **A HIP-specific defect** (E1 step 3's second classification) is fixed on
  the path E1 named, under the same rules as (b) or (c).

**Additional information needed:** E1's classification. The (b)/(c) fix
cannot be designed before it.

**Exit criterion:**
- **Under (a):** V1's derivation states the amplification mechanism with E1's
  figures, and every probe step at SERIAL np 1-6 and HIP np 1-4 has
  field-scale error ≤ `1.95e-3`.
- **Under (b) or (c):** in two runs at SERIAL np 1-6 and HIP np 1-4,
  AutoRebalance's per-step field-scale error is ≤ `1.95e-3` on every step. Its
  `max_vel_rel` is below `1.95e-3` at SERIAL np 5-6 and at every HIP np. The failure direction is the specific condition E1
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
np >= 3 is then stale. Presentation: V1 resumes and its np >= 3 table no
longer reproduces. Response: H2's log section states that the partition path
changed. E1 runs after H2 and measures on the new partition. V1 re-measures
np >= 3 before pinning anything.

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

**R6 — A HIP failure is read as the change's when it predates it.** The HIP
binaries have never run for this work. A case may already fail on HIP because
its reference was generated on SERIAL: `LaplaceSolve.bitForBitArtifacts`
compares hashes against `tests/data`, and device reductions sum in a different
order. Or it may fail through a HIP-only defect. Presentation: a task's HIP arm
fails on a case its change does not touch. Distinguishing measurement: H0c's
table, taken before any change. A failure that H0c recorded is carried and named
in README "Known Issues", not attributed to the task that ran into it. A failure
that H0c did not record belongs to the change.
