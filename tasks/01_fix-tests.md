# Make `UpwardSweep` and `LaplaceSolve` pass on SERIAL and HIP

**Status:** IN PROGRESS — F1, F2, F3 done

## Problem

`fix-hang-rebalance.md` H2's partitioner arm gates on `UpwardSweep` and
`LaplaceSolve` passing through `canopy_ctest` at SERIAL np 1-6 and HIP np 1-4.
Neither does today, for reasons unrelated to the partitioner. A gate that
fails before any change is made cannot tell a regression from the baseline,
so these failures are resolved first. H2's partitioner arm starts once every
task here is **DONE**.

The failures, measured in `fix-hang-rebalance` H0b and H0c (flux jobs
`f3cLieUe4u2F`, `f3cM4ghTjtiT`, `f3cMCHLDMNu5`; per-case detail in
`fix-hang-rebalance-progress-log.md`, section H0c):

| test case | backends, np | symptom |
| --- | --- | --- |
| `UpwardSweep.testRootMultipoleMatchesDirectP2M{Basic,Small}` | SERIAL and HIP, np 1 | max relative error 34.8-35.8 against `1e-10` |
| `UpwardSweep.testRootMultipoleMatchesDirectP2MMultiRank{Basic,Small}` | SERIAL and HIP, np 2-6 | same |
| `UpwardSweep.testIdempotentExecution{Basic,Small}` | HIP np 1-3 (np 4's output was cut off by ctest's timeout) | last-bit differences between two `execute()` calls; 3 000-17 000 mismatch lines per run |
| `LaplaceSolve.bitForBitArtifacts` | HIP np 1-2 | `locals()` hash differs from the SERIAL-generated reference |
| `LaplaceSolve.crossRankAgreement` | HIP np 3-4 | field deviates `4.23e-7` (potential), `3.89e-8` (gradient) from the np-1 reference, against `LS_CROSS_RANK_TOL = 5.6e-10` |

**End state.**
- `canopy_ctest '^Canopy_Test_(UpwardSweep|LaplaceSolve)_MPI_SERIAL_np_[1-6]$'`
  and the HIP np 1-4 equivalent report every entry `completed`, except
  `LaplaceSolve.crossRankAgreement` at HIP np 3-4 if F4 attributes it to the
  Zoltan2-on-HIP partition path. In that case it is recorded as carried, and
  H2's partitioner arm checks it.
- `UpwardSweep`'s `extra_s` budget allowance is gone from
  `scripts/tuolumne/serial_runtimes.tsv`.
- README "Known Issues" no longer lists the `UpwardSweep` and `LaplaceSolve`
  entries this document resolves.

**Out of scope.**
- Making HIP results bit-reproducible run to run. P2M accumulates with
  `Kokkos::atomic_add` (`src/Canopy_LaplaceKernel.hpp:426-429`), so HIP
  summation order varies between runs. The tests are made to tolerate that;
  `src/` is not changed to remove it.
- The OPENMP test variants. They also accumulate with atomics, and nothing
  here builds or runs them.
- `MultiSolve`'s failures and its HIP stall (`fix-hang-rebalance` H1, H2, E1).
- Any change to `LS_CROSS_RANK_TOL`, `LS_DIRECT_SUM_TOL`, `LS_NUM_STEPS` or
  the committed `tests/data/laplace_solve_P6.txt` records.

## Approach

**The root-multipole failure is a test defect.** P2M stores scale-normalized
multipoles $\bar M_{n,m} = M_{n,m} / w^{n+1}$, with $w$ the cell's half-width
(`src/Canopy_LaplaceKernel.hpp:399-433`), and M2M carries that normalization
up to the root. The test's reference, `direct_p2m_to_center`
(`tests/tstUpwardSweep.hpp:83-117`), computes unnormalized $M_{n,m}$. The root
half-width is the largest half-extent of the tolerance-inflated bounding box,
about 0.55-0.6 for these fixtures (`src/Canopy_TreeBuilder.hpp:606-636`), and
$0.6^{-7} \approx 36$ matches the measured error of 34.8-35.8. F1 scales the
reference by $w_{root}^{-(n+1)}$.

**HIP is allowed to reassociate.** Two tests demand bits that HIP's atomic
accumulation cannot deliver.
- `testIdempotentExecution` stays exact on `Kokkos::Serial`, the only
  execution space whose order is fixed. Elsewhere it compares at a field-scale
  relative tolerance (F2).
- `bitForBitArtifacts` exists to attribute a bitwise change across a refactor
  that should change no bits. SERIAL serves that purpose, and its reference
  data is SERIAL-generated (`tests/data/laplace_solve_P6.txt` header). So it
  runs on `Kokkos::Serial` only and skips elsewhere with the reason (F3).
  `crossRankAgreement` and `matchesDirectSum` still cover HIP.

**`crossRankAgreement` on HIP is measured before it is touched.** HIP np 2
passes and HIP np 3-4 fail by about 750x the tolerance, above the test's own
R8 threshold (`LS_CROSS_RANK_R8_THRESHOLD = 1.0e-9`,
`tests/tstLaplaceSolve.hpp:190-193`), beyond which a deviation is not
attributable to summation order. HIP np ≥ 3 also partitions with Zoltan2 MJ on
the device (`fix-hang-rebalance.md`, Current state), a path no other backend
takes and that H2's partitioner arm deletes. F4 measures which of three
mechanisms it is, and only the defect branch changes code here:

| mechanism | one-step HIP deviation (np 3-4) | partition forced onto the host node |
| --- | --- | --- |
| (a) device reassociation amplified over 12 steps | below `1e-9`, growing with steps | deviation unchanged |
| (b) defect in the MJ-on-HIP partition path | — | deviation falls to SERIAL's level |
| (c) other HIP defect at np ≥ 3 | at or above `1e-9` | deviation unchanged |

The responses: (a) stop and report, with no tolerance chosen; (b) record it as
carried, for H2's partitioner arm to check; (c) find and fix.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Gate | every exit criterion runs through `canopy_ctest` (`scripts/tuolumne/ctest_budget.sh`) at SERIAL np 1-6 and HIP np 1-4, with anchored regexes | The `fix-hang-rebalance` conventions "Backends" and "Time budget". HIP stops at np 4: a node has four APUs. |
| HIP environment | `( export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE; canopy_ctest '<regex>' … )` | `canopy_ctest` is a shell function; the subshell keeps the variables off SERIAL commands (`scripts/tuolumne/run_ctest_h0.flux`, `hip_ctest`). |
| Batch script | one script, `scripts/tuolumne/run_ctest_fix_tests.flux`, with the preamble, provenance block, watchdog self-test and `hip_ctest` of `scripts/tuolumne/run_ctest_h0.flux`; `--time-limit` (≤ 60 min on `pdebug`); wait with `flux job status <id>` | `--flags=waitable` is rejected on this machine; bare `--time` is rejected. |
| Build | `make -j Canopy_Test_UpwardSweep_MPI_SERIAL Canopy_Test_UpwardSweep_MPI_HIP Canopy_Test_LaplaceSolve_MPI_SERIAL Canopy_Test_LaplaceSolve_MPI_HIP` in `build-tuolumne/`, naming only the targets a task's criterion runs | `CLAUDE.md`: never a full build. |
| Exact backend | `std::is_same_v<ExecutionSpace, Kokkos::Serial>` (in a test, `TEST_EXECSPACE`) | The only execution space with a fixed accumulation order. OPENMP is out of scope and takes the tolerance branch. |
| Skips | `GTEST_SKIP()` with a message that names the reason and the backend | Matches `testBitForBitArtifacts`' existing np ≥ 3 skip (`tests/tstLaplaceSolve.hpp:1196-1205`). A skip is visible in the log; a silent `return` is not. |
| Tolerances | never widened to make a case pass; a new tolerance is measured first, and the measurement goes in the log with the margin it leaves | The `tstLaplaceSolve.hpp` rule (`:166-172`). |
| `src/` | unchanged, except by F4 under mechanism (c) | Every other failure is in a test. |
| Temporary instrumentation | F4's diagnostics live in the working tree only. The log records their diff, and `git status` shows them gone before the commit. | The precedent in `abstract-solver-backend-progress-log.md:505-508`. |
| README | each task removes or rewrites the README "Known Issues" entry it resolves, in the same commit | `CLAUDE.md`. |
| Formatting and comments | never clang-format; comments state units and the normalization (absolute, relative, field-scale) of every tolerance | `CLAUDE.md`. |

### Deliberate deviations

- **`testIdempotentExecution` is not exact on HIP.** Its doc comment
  (`tests/tstUpwardSweep.hpp:323-334`) promises bit-identical results. The
  property it guards is that `execute()` zeroes its storage
  (`src/Canopy_UpwardSweep.hpp:701`) and keeps no state across calls. A leaked
  partial sum is O(1) relative to the coefficients, and reassociation is O(N
  ε). A tolerance between the two still catches the defect the test exists
  for. The comment is rewritten to say so.
- **`bitForBitArtifacts` skips on HIP rather than holding HIP reference
  records.** HIP is not run-to-run reproducible: `MultiSolve`'s HIP
  `[multisolve-dev]` lines differ between passes even at np 1
  (`fix-hang-rebalance-progress-log.md`, section H0c). So a committed HIP
  record would fail at random.

## Current state

- The `UpwardSweep` root-multipole tests (`tests/tstUpwardSweep.hpp:138-227`,
  multi-rank variant `:512-689`) compare the sweep's normalized root
  multipole against an unnormalized reference and fail at every rank count on
  both backends.
- Both callers of `direct_p2m_to_center` take the expansion center from
  `builder.root_box()`. That matches the root cell's center
  (`src/Canopy_TreeBuilder.hpp:624-627`). Neither has the half-width. The root
  cell, key `ROOT_KEY`, is in `builder.cells()` (`std::vector<CellInfo>`,
  `half_width` at `src/Canopy_TreeBuilder.hpp:89`).
- `testIdempotentExecution` (`tests/tstUpwardSweep.hpp:335-397`) runs two
  `EXPECT_EQ`s per coefficient per cell, each printing a message on
  mismatch. On HIP that is up to ~17 000 failure messages per entry, and
  ctest needs ~10 s after the ranks exit to process them.
- `serial_runtimes.tsv` carries `extra_s` of 2/7/18/23 s on the `UpwardSweep`
  np 1-4 rows to cover that processing time (`fix-hang-rebalance-progress-log.md`,
  "Budget allowances"). The allowance applies on both backends because rows
  have no backend column.
- `testBitForBitArtifacts` (`tests/tstLaplaceSolve.hpp:1192-1409`) runs on
  every backend at np 1-2 and compares against `tests/data/laplace_solve_P6.txt`,
  whose header states it was solved on the SERIAL backend. The file is located
  through the compile definition `CANOPY_TEST_DATA_DIR`
  (`tests/CMakeLists.txt:80-81`, `tests/tstLaplaceSolve.hpp:275-276`).
- `testCrossRankAgreement` (`tests/tstLaplaceSolve.hpp:1423-1549`) compares
  the np=k field after `LS_NUM_STEPS = 12` steps with the committed np=1 field
  record, at `LS_CROSS_RANK_TOL = 5.6e-10`. That tolerance is 100x the worst
  SERIAL deviation measured, `5.6e-12` (`:166-187`). Above
  `LS_CROSS_RANK_R8_THRESHOLD = 1.0e-9` it also prints an R8 message
  (`:1527-1538`).
- On a HIP solver, `TreePartitioner::partition_leaves` runs Zoltan2 MJ on
  `KokkosDeviceWrapperNode<Kokkos::HIP>`. The node is chosen by
  `zoltan_node_t` and asserted by the `static_assert` at
  `src/Canopy_TreePartitioner.hpp:374-382`.

## Progress log

`01_fix-tests-progress-log.md` holds what each session measured, decided and
found by running. Consult it before implementing a task, changing a
signature, or reopening a question this document treats as settled.

## Task sequence

### F1 — Normalize the root-multipole reference — **DONE**

**Depends on:** none.

**Fill in:** `tests/tstUpwardSweep.hpp`.
- `direct_p2m_to_center` (`:83-117`): add the root half-width as a parameter
  and multiply each degree-$n$ term by $w^{-(n+1)}$.
- Its two callers: `testRootMultipoleMatchesDirectP2M` (`:195`) and
  `testRootMultipoleMatchesDirectP2MMultiRank` (`:657`). Each reads the
  half-width of the `ROOT_KEY` entry of `builder.cells()` and fails the test
  with `ASSERT` if there is none.
- The doc comment at `:122-137`, so it names the normalization.
- README "Known Issues", the entry "`UpwardSweep`'s root-multipole checks fail
  at every rank count": remove it.

**Reference:** `p2m_contribution` (`src/Canopy_LaplaceKernel.hpp:399-433`),
whose `term` starts at $1/w$ and multiplies by $\rho / w$ per degree. The root
cell's construction (`src/Canopy_TreeBuilder.hpp:606-656`).

**Do:**
1. Scale the reference to the kernel's convention. Do not change the
   tolerance `1.0e-10`.
2. Record in the log the max relative error per case and np after the fix.

**Exit criterion:** with the four `UpwardSweep` targets rebuilt:
- **Passes:** in `run_ctest_fix_tests.flux`, every
  `UpwardSweep.testRootMultipoleMatchesDirectP2M*` case passes at SERIAL np 1-6
  and HIP np 1-4. The log has the max relative errors.
- **Fails for the intended reason:** with the exponent temporarily changed to
  $w^{-n}$ in a scratch build, the SERIAL np-1 case fails with an error of
  order $1/w - 1$ (≈ 0.7-0.8). Revert before committing.

**Met.** In flux job `f3cMpQqPjuiw`, every `testRootMultipoleMatchesDirectP2M*`
case passed at SERIAL np 1-6 and HIP np 1-4, with a worst max relative error
of `7.7e-14` (HIP np 3). Every SERIAL entry reported `completed`. The HIP
entries still report `failed`, from `testIdempotentExecution` alone (F2). In
perturbed job `f3cMiZgKVDDy` ($w^{-n}$), SERIAL np 1 failed at `0.668` and
`0.674`, which is $1/w - 1$ for $w_{root} \approx 0.60$. Per-case figures are
in the log, section F1.

### F2 — Make `testIdempotentExecution` backend-aware and bounded — **DONE**

**Depends on:** F1 (both edit `tests/tstUpwardSweep.hpp`, and F2's criterion
runs the whole stem).

**Fill in:**
- `tests/tstUpwardSweep.hpp`: `testIdempotentExecution` (`:335-397`) and its
  doc comment (`:323-334`).
- `scripts/tuolumne/serial_runtimes.tsv`: clear `extra_s` and `extra_reason` on
  the `UpwardSweep` np 1-4 rows.
- README "Known Issues", the entry "`UpwardSweep`'s idempotence checks fail on
  HIP, and their output overruns the budget": remove it.

**Reference:** the field-scale rule in `field_scales`
(`tests/tstLaplaceSolve.hpp:1162-1184`), which normalizes by the largest
magnitude rather than per element, so cancelling coefficients do not inflate
the error.

**Do:**
1. Compute one scalar, $\max_{c,i} \lvert M^{(1)}_{c,i} - M^{(2)}_{c,i}\rvert / \max_{c,i} \lvert M^{(1)}_{c,i}\rvert$,
   over all cells and coefficients, plus the location of its maximum.
2. Assert it with one `EXPECT`, which prints the scalar and the location once:
   - on `Kokkos::Serial`, equal to `0.0`;
   - elsewhere, below `LS_IDEMPOTENT_TOL = 1.0e-12`, a named constant with a
     comment giving its normalization and the measurement it rests on.
3. Before fixing the tolerance, measure the scalar on HIP np 1-4 in two runs and
   record it. The constant needs at least 100x margin over the worst
   measurement. If the worst is above `1.0e-14`, stop and report rather than
   raise the constant: that is more than reassociation.

**Exit criterion:** in `run_ctest_fix_tests.flux`:
- **Passes:** with the allowance removed,
  `canopy_ctest '^Canopy_Test_UpwardSweep_MPI_SERIAL_np_[1-6]$'` and
  `'^Canopy_Test_UpwardSweep_MPI_HIP_np_[1-4]$'` report every entry
  `completed`. Each HIP entry's ctest output is under 500 lines, and the log
  has the measured scalar per np.
- **Fails for the intended reason:** with the `Kokkos::deep_copy` that zeroes
  `_multipoles` (`src/Canopy_UpwardSweep.hpp:701`) temporarily removed in a
  scratch build, `testIdempotentExecution*` fails on SERIAL np 1 and HIP np 1.
  Each failure is one message with an O(1) scalar. Revert before committing.

**Met.** With the `UpwardSweep` np 1-4 allowance cleared, flux jobs
`f3cN2kx53B2o` and `f3cN3nXho9rP` (`-V`) and `f3cN5VTtsPWP` (default
`--output-on-failure`) each reported every SERIAL np 1-6 and HIP np 1-4 entry
`completed`. SERIAL's scalar was exactly `0`. HIP's worst scalar was `3.7e-16`,
about 2700x under `LS_IDEMPOTENT_TOL = 1.0e-12`. The HIP entries printed 10-40
lines in `f3cN5VTtsPWP`. In perturbed job `f3cMxBbPLv9d` (zeroing `deep_copy`
removed), SERIAL np 1 and HIP np 1 each failed once per case, with scalars
`1.54` and `1.50`. That required a test fix first: on SERIAL, both snapshots
had aliased `sweep.multipoles()` (log, section F2).

### F3 — Run `bitForBitArtifacts` on `Kokkos::Serial` only — **DONE**

**Depends on:** none.

**Fill in:**
- `tests/tstLaplaceSolve.hpp`: the top of `testBitForBitArtifacts`
  (`:1192-1205`), plus a `GTEST_SKIP` when `ExecutionSpace` is not
  `Kokkos::Serial`, naming the backend and that the reference data is
  SERIAL-generated.
- The file header's gate table (`:20-26`).
- README "Known Issues", the `bitForBitArtifacts` bullet of "`LaplaceSolve`
  fails two bit-level checks on HIP": remove it, and keep the
  `crossRankAgreement` bullet for F4.

**Reference:** the existing np ≥ 3 skip (`:1196-1205`).

**Exit criterion:** in `run_ctest_fix_tests.flux`:
- **Passes:** `LaplaceSolve.bitForBitArtifacts` is reported `SKIPPED` with the
  new message at HIP np 1-2 and still passes at SERIAL np 1-2.
- **Fails for the intended reason:** with one hex digit of the `(1,0)` `locals`
  hash in `tests/data/laplace_solve_P6.txt` temporarily altered, SERIAL np 1
  fails on `locals_hash`. Restore it (`git checkout -- tests/data/laplace_solve_P6.txt`)
  before committing.

**Met.** In flux job `f3cNB48LsWNB` (`-V`), `LaplaceSolve.bitForBitArtifacts`
was `OK` at SERIAL np 1-2 and `SKIPPED` with the new message at HIP np 1-2.
In perturbed job `f3cNANZK5J1V` (the `(1,0)` `locals` hash ending `6dd6`
changed to `6dd7`), SERIAL np 1 failed on `locals_hash` alone. The data file
was restored with `git checkout`. In the same pass run, HIP np 2
`crossRankAgreement` failed at `1.6e-7` (log, section F3; F4's business).

### F4 — Classify and resolve `crossRankAgreement` on HIP np 3-4 — **NOT STARTED**

**Depends on:** F3 (its criterion runs the whole `LaplaceSolve` stem).

**Fill in:** depends on the classification.
- Every branch: README "Known Issues", the remaining `crossRankAgreement`
  bullet.
- Mechanism (b): the `fix-hang-rebalance-progress-log.md` H2 note below.
- Mechanism (c): the `src/` path the measurement names.

**Reference:** the Approach table; `testCrossRankAgreement`
(`tests/tstLaplaceSolve.hpp:1423-1549`); the R8 instruction at `:1527-1538`
("re-measure at `LS_NUM_STEPS = 1`").

**Do:**
1. **Spread.** Run `LaplaceSolve` HIP np 2-4 twice through `canopy_ctest`.
   Record `crossRankAgreement`'s `max_pot_dev` and `max_grad_dev` per np and
   run.
2. **One step.** With temporary instrumentation:
   - set `LS_NUM_STEPS` to 1;
   - regenerate the np-1 SERIAL record into a scratch directory
     (`CANOPY_LAPLACE_SOLVE_REGENERATE`, then concatenate as the comment at
     `tstLaplaceSolve.hpp:55-65` describes);
   - point `LS_DATA_FILE` at the scratch file;
   - run `crossRankAgreement` at SERIAL np 2-4 and HIP np 2-4.

   Record the one-step deviations.
3. **Partition attribution.** With the 12-step configuration and temporary
   instrumentation that sets `zoltan_node_t` to
   `KokkosDeviceWrapperNode<Kokkos::Serial>` for every `ExecutionSpace`
   (`src/Canopy_TreePartitioner.hpp:374-382`; `static_assert` relaxed with
   it), run HIP np 3-4 twice. Record the deviations.
4. Classify against the Approach table, with the deciding figures, and remove
   all instrumentation.
   - **(a):** stop. Record the figures and report; no tolerance is chosen in
     this task.
   - **(b):** record that HIP np 3-4 `crossRankAgreement` is carried until H2's
     partitioner arm. Append to `fix-hang-rebalance-progress-log.md` a section
     whose **Affects:** line names H2: its partitioner-arm exit criterion
     includes `LaplaceSolve.crossRankAgreement` passing at HIP np 3-4 at the
     unchanged tolerance. Rewrite the README bullet to state the attribution.
   - **(c):** find the defect on the path the measurements point to, and fix it
     without touching any tolerance.

**Additional information needed:** the classification. The fix under (c)
cannot be designed before the measurement names a path.

**Exit criterion:** the log has the step-1 spread, the step-2 one-step
deviations and the step-3 host-partition deviations, with the classification
they decide. `git status` shows no instrumentation left. Then, per branch:
- **(a):** this task stops at the report. It is not marked **DONE** until the
  response is decided and added here.
- **(b):** `canopy_ctest '^Canopy_Test_LaplaceSolve_MPI_HIP_np_[1-4]$'` fails
  only `crossRankAgreement` at np 3-4. The H2 note and README bullet exist.
- **(c):** `canopy_ctest '^Canopy_Test_LaplaceSolve_MPI_SERIAL_np_[1-6]$'` and
  the HIP np 1-4 equivalent report every entry `completed` at the unchanged
  `LS_CROSS_RANK_TOL`. Temporarily reintroducing the defect makes HIP np 3
  fail again.

## Known risks

**R1 — F1's normalization is not the whole error.** After the scaling, a
residual above `1e-10` remains. Presentation: F1's pass direction fails with
a small error rather than ~35. Distinguishing measurement: the residual's
size. A residual near $10^{-15}$-$10^{-13}$ is round-off and would mean the
tolerance is wrong, which is not this task's call: stop and report. A residual
of $10^{-6}$ or more points at the expansion center or at M2M, so compare the
per-degree errors. A center mismatch grows with $n$.

**R2 — The HIP idempotence difference is not reassociation.** A race that
leaks partial sums between calls also shows up as a non-zero difference.
Presentation: F2's measured scalar is large, or varies widely between runs.
Distinguishing measurement: reassociation is bounded by about $N\varepsilon$
relative, $10^{-13}$ for $10^3$ particles. F2 stops above `1.0e-14` rather
than absorb a larger value.

**R3 — Skipping `bitForBitArtifacts` on HIP hides a HIP-only regression.** A
refactor that changes only HIP arithmetic passes the bit gate. Presentation:
none, which is the risk. Mitigation: `crossRankAgreement` and
`matchesDirectSum` still run on HIP. F4 leaves `crossRankAgreement` meaningful
on HIP.

**R4 — The MJ-on-HIP stall interrupts F4's HIP runs.** HIP np ≥ 3 partitions
with Zoltan2 MJ on the device, which stalls intermittently
(`fix-hang-rebalance-progress-log.md`, H0c). Presentation: a `LaplaceSolve`
HIP entry goes over budget with rank 0 in `hipDeviceSynchronize`. Response:
that run has no deviation to read. Rerun it, and record the stall with its
stacks as an occurrence of the known issue, not as a result. Step 3's
host-node runs do not have this exposure.

**R5 — Temporary instrumentation is committed.** F1, F2, F3 and F4 all
perturb code or data temporarily. Presentation: a later run shows the
perturbed behavior with no source diff to explain it. Mitigation: each task
checks `git status` and `git diff` for its scratch changes before
committing, and its log section records which perturbation was applied and
reverted.
