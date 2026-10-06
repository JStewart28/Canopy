# Make `UpwardSweep` and `LaplaceSolve` pass on SERIAL and HIP — progress log

Session record for 01_fix-tests. Companion to `01_fix-tests.md`, which holds the
design, the task sequence and the risks; this file holds what actually
happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `01_fix-tests.md` can cite it
by ID. No dates: the order of the sections is the chronology. If a session
covers more than one task, name them all; if it belongs to no task, name the
topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that
invalidates a later task is worthless if the session starting that task has to
read the whole log to notice it; this line is the index that makes it findable.
Name `fix-hang-rebalance.md` task H2 there whenever a finding changes what its
partitioner arm will gate on.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could
not stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation
departed from the task's stated **Do** steps, and why — a task marked `**DONE**`
that was done differently than it was written is the quietest way for a design
to stop describing the code.

(No entries yet.)

## F1

**Change.** `direct_p2m_to_center` (`tests/tstUpwardSweep.hpp`) takes a new
`double w` argument after `cz`, and accumulates
$q \rho^n Y_n^{-m} / w^{n+1}$, the `p2m_contribution` convention. A new helper,
`UpwardSweepTest::root_half_width(builder.cells())`, returns the `ROOT_KEY`
cell's `half_width`, or `-1` when the cell is missing. Both callers
`ASSERT_GT(w_root, 0.0)` on it. Both tests now print one line,
`[root-multipole] np=<k> max_rel_err=<e> w_root=<w>`, so a passing run still
records its margin. The tolerance `1.0e-10` is unchanged.

**New script.** `scripts/tuolumne/run_ctest_fix_tests.flux` runs each
argument regex through `canopy_ctest`, using `hip_ctest` for regexes that
contain `_HIP`. It carries `run_ctest_h0.flux`'s preamble, provenance block,
np 5-6 abort and watchdog self-test, and adds `git diff --stat` and each
binary's mtime to the provenance. The default `--time-limit` is 20. The ctest
output flag comes from `FIX_TESTS_CTEST_ARGS` (default `--output-on-failure`).
The F1 pass run used `-V`, because `--output-on-failure` hides a passing
entry's errors. `--time-limit` can be overridden on the `flux batch` command
line.

**Measured** (job `f3cMpQqPjuiw`, the committed test code). Pairs are
(`Basic`, `Small`); np 1 is `testRootMultipoleMatchesDirectP2M`, np ≥ 2 the
`MultiRank` variant.

| np | SERIAL | HIP | $w_{root}$ |
| --- | --- | --- | --- |
| 1 | 9.9e-15, 4.7e-15 | 8.6e-15, 4.4e-15 | 0.5996, 0.5974 |
| 2 | 8.2e-15, 2.1e-14 | 1.0e-14, 2.1e-14 | 0.5996, 0.5977 |
| 3 | 2.2e-14, 1.4e-14 | 7.7e-14, 8.6e-15 | 0.5996, 0.5980 |
| 4 | 2.7e-14, 6.8e-15 | 3.7e-14, 5.2e-15 | 0.5999, 0.5983 |
| 5 | 7.9e-15, 7.1e-15 | — | 0.5999, 0.5995 |
| 6 | 1.6e-14, 2.3e-14 | — | 0.5999, 0.5995 |

The residual is round-off, at least 1300x below the tolerance. R1 did not
fire. $w_{root}$ varies with np because each rank count generates a different
global particle set. SERIAL np 1-6 report `completed`. HIP np 1-4 report
`failed` from `testIdempotentExecution{Basic,Small}` only; every
root-multipole case is `OK`.

**Fails for the intended reason.** Perturbation: `term` initialized to `1.0`
instead of `1.0 / w`, i.e. the reference scaled by $w^{-n}$. Built into
`build-tuolumne/` and run in job `f3cMiZgKVDDy`, SERIAL np 1: `Basic` failed at
`0.66782`, `Small` at `0.67381`. That is $1/w - 1$ for $w = 0.5996, 0.5974$,
slightly under the task text's "≈ 0.7-0.8" estimate, which assumed
$w \approx 0.55$-$0.6$. Reverted; `grep SCRATCH` is empty and the binaries were
rebuilt before both pass runs (`f3cMm2wURzM5` without the print,
`f3cMpQqPjuiw` with it).

**Affects:** none.

## F2

**Change.** `testIdempotentExecution` (`tests/tstUpwardSweep.hpp`) now reduces
the comparison to one scalar,
$d = \max_{c,i,k} \lvert M^{(1)} - M^{(2)}\rvert / \max_{c,i,k} \lvert M^{(1)}\rvert$
(complex modulus, every cell, coefficient and component). Both maxima are
global: `MPI_MAXLOC` finds the owning rank of the largest difference, and that
rank broadcasts its (cell, coeff, comp). One `EXPECT_TRUE` checks `d == 0` when
`TEST_EXECSPACE` is `Kokkos::Serial` and `d < LS_IDEMPOTENT_TOL` otherwise. Its
message names the backend, `d`, the limit and the location. Every rank asserts,
so a failing np-k entry prints k copies of one line. Rank 0 also prints
`[idempotent] np=<k> rel_diff=<d> max_abs_diff=<num> max_mag=<den>`.
`LS_IDEMPOTENT_TOL = 1.0e-12` sits in `UpwardSweepTest`. The `LS_` prefix
follows the task text, although the constant is not `LaplaceSolve`'s.
`serial_runtimes.tsv`: `extra_s`/`extra_reason` cleared on `UpwardSweep` np 1-4.

**Bug only running revealed: the SERIAL check was vacuous.** The first
perturbed run (`f3cMuUxoun7q`) failed HIP np 1 as intended but passed SERIAL
np 1. `create_mirror_view_and_copy(HostSpace(), v)` returns `v` itself when
`v` is already host-resident. So on SERIAL, `h_M_first` and `h_M_second` were
the same buffer, and the old exact `EXPECT_EQ`s compared each value with
itself. The first snapshot now uses `Kokkos::create_mirror` (which always
allocates) plus `deep_copy`. `DownwardSweep.testIdempotentExecution` has the
same pattern for `locals()` (`tests/tstDownwardSweep.hpp:365-378`); that is
out of scope here and recorded in README "Known Issues". Its potential
comparison uses two distinct views and is sound.

**Measured** (HIP; SERIAL was exactly `0` at np 1-6 in both runs). Values are
$d$ for (`Basic`, `Small`):

| np | run 1 `f3cN2kx53B2o` | run 2 `f3cN3nXho9rP` |
| --- | --- | --- |
| 1 | 1.8e-16, 9.1e-17 | 2.0e-16, 1.2e-16 |
| 2 | 1.5e-16, 3.7e-16 | 2.7e-16, 1.9e-16 |
| 3 | 1.8e-16, 3.1e-16 | 2.5e-16, 1.8e-16 |
| 4 | 2.4e-16, 8.7e-17 | 2.1e-16, 1.1e-16 |

The worst is `3.75e-16`, which is about 1.7 ulp of the largest coefficient
(`max_mag` 21.8-153.9). That is reassociation, so R2 did not fire, and
`1.0e-12` leaves ~2700x margin. Both runs used `-V`, which is how passing
entries' scalars get printed. Under `-V`, the HIP entries' full output was
154/340/558/808 lines at np 1-4, nearly all of it the gtest listing of every
rank. Under the default `--output-on-failure` (`f3cN5VTtsPWP`, every entry
`completed`), they printed 10/20/30/40 lines. The constant's comment was
written after these runs and is the only difference between the committed file
and the measured binaries.

**Fails for the intended reason.** Perturbation: the
`Kokkos::deep_copy( _multipoles, coeff_type() )` at
`src/Canopy_UpwardSweep.hpp:701` commented out. Job `f3cMxBbPLv9d`, with the
snapshot fix: SERIAL np 1 and HIP np 1 each failed once per case, `d = 1.541`
(`Basic`, cell 6) and `1.504` (`Small`, cell 0), identical on both backends.
Reverted with `git checkout -- src/Canopy_UpwardSweep.hpp` and rebuilt before
the pass runs.

**Budget observation.** The first perturbed submission (`f3cMtaAWZFkT`, on
binaries just relinked) went over budget on both entries before any test ran.
SERIAL np 1 took 18.0 s against 13. HIP np 1 took 12.0 s against 7, and its
rank 0 stack at 7.7 s was still in `Kokkos::HIP::impl_initialize` →
`hipMemcpyToSymbol` → code-object load. An unchanged resubmit
(`f3cMuUxoun7q`) ran in 6.9 s and 4.1 s. A HIP np 1 entry that runs after a
SERIAL entry gets no cold-start allowance, and with `extra_s` gone its budget
is 7 s against a ~3.8 s warm runtime. A slow first launch of a freshly linked
HIP binary can still exceed that. It happened once in seven submissions.

**Affects:** none in this document. The DownwardSweep finding is a README
Known Issue, not a task here.

## F3

**Change.** `testBitForBitArtifacts` (`tests/tstLaplaceSolve.hpp`) now opens
with a `GTEST_SKIP` when `!std::is_same_v<ExecutionSpace, Kokkos::Serial>`,
ahead of the existing np ≥ 3 skip. The message, as printed at HIP np 1:

> bit-for-bit artifacts are gated on Kokkos::Serial only, and this backend is
> HIP: the reference data in <LS_DATA_FILE> was generated on SERIAL, and device
> backends accumulate with atomics, so their bit patterns vary run to run.
> crossRankAgreement and matchesDirectSum cover this backend. See
> tasks/01_fix-tests.md F3.

The file-header gate table and the comment above the function say the same.
README: the `bitForBitArtifacts` bullet is gone, and the entry is retitled to
`crossRankAgreement` alone. The np ≥ 3 skip citation at `README.md:466` now
reads `tests/tstLaplaceSolve.hpp:1211-1219`; it used to read `:1065`.

**Measured** (job `f3cNB48LsWNB`, `-V`, SERIAL and HIP np 1-2):
`bitForBitArtifacts` was `OK` at SERIAL np 1 (532 ms) and np 2 (both ranks),
and `SKIPPED` at HIP np 1 and np 2 (both ranks).

**Fails for the intended reason.** Perturbation: line 92 of
`tests/data/laplace_solve_P6.txt`, the `set 1 0` `locals` hash, changed from
`0xfb2cddef75e26dd6` to `…6dd7`. Job `f3cNANZK5J1V`, SERIAL np 1:
`bitForBitArtifacts` failed with "locals() hash differs (nprocs=1 rank=0):
measured 0xfb2cddef75e26dd6, reference 0xfb2cddef75e26dd7", and it was the
entry's only failure. The file was restored with
`git checkout -- tests/data/laplace_solve_P6.txt` after that job finished and
before the pass run was submitted. `git status` showed it clean.

**Found, not investigated: HIP np 2 `crossRankAgreement` fails.** In
`f3cNB48LsWNB`, `Canopy_Test_LaplaceSolve_MPI_HIP_np_2` reported `failed`.
The only failure was `crossRankAgreement`, with `max_pot_dev = 1.59e-7` and
`max_grad_dev = 2.03e-8` against `LS_CROSS_RANK_TOL = 5.6e-10`, and the R8
message fired (above `1e-9`). Both ranks recorded 397/386 unique operators.
`fix-hang-rebalance` H0c saw HIP np 2 pass; one run here cannot say whether it
is intermittent. `01_fix-tests.md`'s Approach says HIP np 2 passes and that
only np 3-4 fail. It also attributes HIP np ≥ 3 to the MJ-on-device partition,
whose cut is trivial at two parts. A ~2.8e2x-tolerance deviation at np 2 is
evidence against mechanism (b) being the whole story. The README bullet notes
the np 2 failure.

**Affects:** F4. Its step 1 (spread, HIP np 2-4 twice) should establish
whether HIP np 2 fails reproducibly. Its Approach premise "HIP np 2 passes"
and the End state's "except `crossRankAgreement` at HIP np 3-4" may need np 2
added. If np 2 fails reliably, the (b) row of the classification (deviation
falls to SERIAL's level when the partition is forced onto the host node)
should also be run at np 2. `fix-hang-rebalance.md` H2: if F4 carries
`crossRankAgreement` on HIP, the carried set may include np 2.

## F4

**Decisions carried in.** F4 is a classification task. Under (a) and (b) it
changes no code in `src/` or the tests; only (c) changes `src/`. H2's
partitioner arm replaces Zoltan2 MJ, so nothing here is made to pass on the
current partitioner and MJ-on-HIP is not made reproducible. The failing set is
the HIP rank counts in np 2-4 that fail `crossRankAgreement` in either step-1
run, and step 3 runs at np 2-4 regardless.

**Departure from Do: more runs than written.** Every entry fails
intermittently, at roughly 1 in 4, so two passes per np cannot tell
"unchanged" from "fixed". Step 3 ran six passes, and a control job on the
committed binaries (`f3cNLgnHcxib`) added four HIP and four SERIAL passes at
np 2-4. All runs used `FIX_TESTS_CTEST_ARGS=-V`. Values below are
(`max_pot_dev`, `max_grad_dev`); F = the entry failed.

**Step 1: spread** (job `f3cNHmnYRF1y`, committed code, HIP np 2-4 twice).

| np | run 1 | run 2 |
| --- | --- | --- |
| 2 | 9.7e-13, 4.9e-12 | 3.6e-13, 1.9e-12 |
| 3 | 8.1e-13, 4.1e-12 | **4.2271443817898591e-07, 3.89e-08 F** |
| 4 | 2.2e-12, 1.1e-11 | 4.0e-13, 2.0e-12 |

Failing set by the task's definition: {np 3}.

**Control** (job `f3cNLgnHcxib`, committed code, four passes). SERIAL
np 2/3/4 was bit-identical every pass: `4.1994e-13`/`2.1570e-12`,
`8.0211e-13`/`4.0354e-12`, `1.1143e-12`/`5.5987e-12`, with per-rank
`n_unique_ops` 368/386 at np 2. HIP:

| np | pass 1 | pass 2 | pass 3 | pass 4 |
| --- | --- | --- | --- | --- |
| 2 | 2.8e-13 | 1.2e-13 | **1.5865136617e-07, 2.0292e-08 F** | 7.2e-13 |
| 3 | 3.3e-13 | **1.5865136542e-07, 2.0292e-08 F** | 4.8e-14 | 4.0e-13 |
| 4 | **1.9545431357e-07, 2.3481e-08 F** | 1.3e-12 | 1.2e-12 | 6.1e-13 |

So np 2 and np 4 fail too. The failing set as defined ({3}) understates it.
On committed code, HIP failed 4 of 18 entries across steps 1 and control, and
np 2 also failed in F3 (`f3cNB48LsWNB`).

**Step 2: one step** (job `f3cNSNzuQpKR`). Instrumentation, reverted with
`git checkout -- tests/tstLaplaceSolve.hpp`:

```diff
-static constexpr int LS_NUM_STEPS = 12;
+static constexpr int LS_NUM_STEPS = 1; // F4 SCRATCH
-    CANOPY_TEST_DATA_DIR "/laplace_solve_P6.txt";
+    "/usr/workspace/stewartj/canopy-f4-scratch/laplace_solve_P6_1step.txt"; // F4 SCRATCH
```

`scripts/tuolumne/run_f4_step2.flux` regenerates the one-step SERIAL record at
np 1-2 (`CANOPY_LAPLACE_SOLVE_REGENERATE` pointed at
`/usr/workspace/stewartj/canopy-f4-scratch/regen`). It then concatenates the
three parts under the committed header, and runs each regex with
`GTEST_FILTER=*crossRankAgreement*`. The scratch record's initial hash is
`0xb6ad437608ad69b7`, the committed one.

| np | SERIAL | HIP (4 passes) |
| --- | --- | --- |
| 2 | 2.0e-15, 4.90e-12 | pot 1.9-2.0e-15; grad 4.90-6.93e-12 |
| 3 | 1.7e-15, 4.90e-12 | pot 1.6-1.7e-15; grad 4.90e-12 |
| 4 | 1.7e-15, 4.90e-12 | pot 1.7-1.9e-15; grad 4.90e-12 |

All 12 HIP entries passed. At one step HIP is at SERIAL's reassociation level,
about six orders under `LS_CROSS_RANK_R8_THRESHOLD = 1e-9`.

**Step 3: host partition** (job `f3cNWTvVDx7h`, 12 steps, committed data, six
passes). Instrumentation, reverted with
`git checkout -- src/Canopy_TreePartitioner.hpp`:

```diff
-        Tpetra::KokkosCompat::KokkosDeviceWrapperNode<ExecutionSpace>;
+        Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::Serial>; // F4 SCRATCH
-    static_assert(
+    static_assert( // F4 SCRATCH: relaxed with zoltan_node_t
         std::is_same_v<typename adapter_t::node_t::execution_space,
-                       ExecutionSpace>,
+                       Kokkos::Serial>,
```

| np | failures | failing values |
| --- | --- | --- |
| 2 | 0 of 6 | — (passing 3.3e-13 to 1.5e-12) |
| 3 | 2 of 6 | 7.7193e-08 / 9.14e-09; 1.5865472965e-07 / 2.0292e-08 |
| 4 | 3 of 6 | 1.9551736394e-07 / 2.3479e-08 (twice); 1.5865136702e-07 / 2.0292e-08 |

That is 5 of 18 against 4 of 18 on committed code, at the same discrete
values. The deviation is unchanged with Zoltan2 on the host node.

**Steps 1-3 alone pointed at (a), and that reading was wrong.** Step 2's
one-step check had little power: a 12-step run fails about 1 time in 4, so a
per-step trigger would fail only about 2% of one-step runs. Two further
experiments located the cause.

**Experiment 1: HIP np 1 against the SERIAL np-1 reference** (job
`f3cNfrMYm1GB`, 20 passes, `GTEST_FILTER=*crossRankAgreement*`).
Instrumentation, reverted:

```diff
-    if ( nprocs == 1 )
+    if ( nprocs == 1 && std::is_same_v<ExecutionSpace, Kokkos::Serial> ) // F4 SCRATCH
```

11 of 20 failed, at the same discrete values as np 2-4: `7.7193e-08`,
`1.5865136e-07` (four times), `1.5865473e-07`, `4.2271444e-07` (twice) and
`4.2273204e-07` (three times). Passing passes were `1.7e-13` to `5.3e-13`.
**The failure does not need more than one rank, so it is not in the
distributed path.**

**Experiment 2: first divergent step** (jobs `f3cNoqiVDvFZ`, `f3cNt51qKjWw`;
script `scripts/tuolumne/run_f4_trace.flux`). Temporary instrumentation in
`with_laplace_solve`, after each `solve()` at np 1, since reverted:
- **Trace:** gather positions, potential and gradient by `GlobalId`. Hash
  `solver.builder().cells()` (key, `global_count`, `is_leaf`), and record
  `root_box()` and `downward().m2l_n_unique_ops()`. With `F4_TRACE_WRITE`
  (SERIAL) it writes a per-step reference under
  `/usr/workspace/stewartj/canopy-f4-scratch/`. With `F4_TRACE_READ` (HIP) it
  prints per-step deviations against that reference.
- **Tie count:** over every pair of cells, apply `mac_satisfied`'s test as
  written (`src/Canopy_CommunicationPlan.hpp:337-346`). Count the pairs with
  $\lvert R^2\theta^2 - r_{sum}^2\rvert \le 10^{-9} r_{sum}^2$ and how many of
  those the floating-point test accepts.
- **Skip:** the np-1 skip also stays off for SERIAL while tracing.

Findings:
- **The root box is recomputed from the particles at every step.** Its last
  bits differ between HIP and SERIAL from step 2 on. So every cell center and
  half-width carries that last-bit noise.
- **Every step's tree has 32 pairs exactly on the MAC threshold** in exact
  arithmetic. For θ = 0.5 and two same-depth cells offset by (2,2,2) cells,
  the two sides are equal because $|n|^2 = 12$.
  Rounding decides how many are accepted. Within one deterministic SERIAL run
  that count goes 0, 16, 0, 24, 32, 32, 24, 0, 16, 24, 24, 16 over steps
  0-11.
- **Both failing HIP runs (of 8) diverge at step 5, and only there.** Steps
  0-4 match SERIAL to `≤1e-14`, with positions within `4e-15` and identical
  tree topology (103 cells, 90 leaves, same key/count hash). At step 5 HIP
  accepted 16 of the 32 ties where SERIAL accepted 32. `m2l_n_unique_ops` went
  from 686 to 716, and the potential jumped from `~4e-15` to `2.150e-07`
  (gradient `9.9e-07`). Positions then diverge by `1e-10` and up, and the
  deviation persists to step 11 (`1.587e-07`).
- **All six passing runs accepted 32 at step 5, as SERIAL did.** Tie decisions
  that differ from SERIAL at other steps (steps 2-4, 7-11) left the field at
  `≤1e-12`. Those pairs are presumably never reached by the dual-tree
  traversal, because their parents are already accepted or rejected.

**Classification: none of (a), (b), (c) as written. It is a defect in `src/`,
not in HIP and not in the distributed path.** `mac_satisfied` decides exact
geometric ties by floating-point rounding. Its inputs, the cell centers, carry
last-bit noise from a root box rebuilt from particle positions every step.
Which side of the MAC a tie lands on changes the far field by truncation size
(~2e-7 here). HIP exposes this only because its positions are not
bit-reproducible run to run. SERIAL's decisions are just as arbitrary but
repeatable. The committed reference in `tests/data/laplace_solve_P6.txt` was
generated with arbitrary tie decisions too.

`is_well_separated`, next to it (`:306-318`), already guards its own tie with
`eps = 1.0e-10`. `mac_satisfied` has no such guard.

**Not done here:**
- Any fix. A tie-robust MAC, for example rejecting pairs within a relative
  band of the threshold, or comparing in integer lattice units derived from
  Morton keys, changes which pairs are accepted. That moves SERIAL's results
  and so `tests/data/laplace_solve_P6.txt`, which this document puts out of
  scope.
- Any tolerance.

**Stalls (R4):** none. Every HIP entry finished within budget in all jobs, and
every watchdog cancel record held only the self-test.

**Affects:**
- F4: stopped. The cause is a `src/` defect whose fix changes the committed
  `LaplaceSolve` reference data, which needs a decision. The failing set is
  HIP np 1-4, since np 1 fails too once compared.
- `fix-hang-rebalance.md` H2: replacing MJ does not remove this failure. The
  partitioner arm cannot gate on HIP `crossRankAgreement` at the unchanged
  tolerance until the MAC tie is fixed or the gate changes. The same tie
  sensitivity can shift any MultiSolve or LaplaceSolve comparison whose
  inputs differ in the last bit.

## F4 — fix

**Decision.** F4 is widened to fix the MAC tie in `src/` and regenerate
`tests/data/laplace_solve_P6.txt`, which the design had put out of scope. No
tolerance and no `LS_NUM_STEPS` changed.

**Change.** `mac_satisfied` (`src/Canopy_CommunicationPlan.hpp:338-356`)
accepts only when $R^2\theta^2 > r_{sum}^2 (1 + 10^{-10})$, through a local
constant `MAC_TIE_REL = 1.0e-10`. An exact geometric tie is now always
rejected (the pair is refined or goes to P2P), whatever the last bits of the
cell centers. A pair that is not tied clears the threshold by O(1) relative,
so no other decision moves. The relative band matches `is_well_separated`'s
absolute `eps = 1.0e-10` next to it. A lattice-exact test from Morton keys was
not chosen: it needs depth-aware integer offsets for cross-depth pairs, for
no gain at this tie margin. `mac_satisfied` has one caller, the dual-tree
traversal (`:624` before the change).

**Regeneration** (job `f3cP3HeeQ4oZ`, `run_laplace_solve_regenerate.flux`,
SERIAL np 1-6 all passed). The three parts were merged under the existing
header, and the `initial` hash is unchanged (`0xb6ad437608ad69b7`).
Final-solve `n_unique_ops` at np 1 rose from 686 to 724: the ties that SERIAL
used to accept by rounding are now refined into more, smaller M2L pairs.
Per rank, np 2 is 390/400 (was 368/386), and np 2-6 span 121-400. The header,
the tolerance comment in `tests/tstLaplaceSolve.hpp` and README's partition
paragraph carry the new counts.

**Pass** (job `f3cP6CubiRBM`, `-V`): every entry `completed`.
- `(UpwardSweep|LaplaceSolve)` SERIAL np 1-6 once, `UpwardSweep` HIP np 1-4
  once.
- `LaplaceSolve` HIP np 1-4 five times: 20 of 20 entries, against 4 of 18
  failing before the fix.
- `crossRankAgreement` SERIAL np 2-6:

  | np | pot | grad |
  | --- | --- | --- |
  | 2 | 4.32e-13 | 2.20e-12 |
  | 3 | 2.84e-13 | 1.46e-12 |
  | 4 | 9.79e-13 | 4.93e-12 |
  | 5 | 1.23e-12 | 6.1748535032248251e-12 |
  | 6 | 2.83e-13 | 1.47e-12 |

- HIP np 2-4 over the five passes: potential `1.8e-13` to `1.4e-12`, gradient
  `9.7e-13` to `7.2e-12`.
- Direct sum: potential `3.3063e-07` (worst `3.3063265504587434e-07`, np 2)
  and gradient `4.33e-08` at every np, on both backends.

**Tolerance margins moved, the tolerances did not.**
- `LS_CROSS_RANK_TOL = 5.6e-10` was 100x the old worst SERIAL deviation. It
  is now 91x the new worst (`6.17e-12`, np 5 gradient).
- `LS_DIRECT_SUM_TOL = 9.63e-07` was 3x; it is now 2.9x. The truncation
  error rose slightly because more pairs went from M2L to finer M2L/P2P at
  different cells, not less accurate ones.

The comment above the tolerances records both. Neither tolerance needs
re-pinning, and the design keeps them out of scope.

**Fails for the intended reason** (job `f3cW2MtcJkto`). Perturbation:
`MAC_TIE_REL = 0.0`, which is the old test. Every entry failed: SERIAL np 1
on `bitForBitArtifacts`, SERIAL np 2-4 and HIP np 2-4 (three passes) on
`crossRankAgreement`. The deviations were the tie-flip values from before,
`4.2274511e-07` and `1.9545431e-07`. SERIAL fails deterministically, because
it accepts some ties that the new reference rejects; HIP lands on one of the
recurring values. Reverted from a saved copy and both targets rebuilt. `grep
"F4 SCRATCH"` over `src/` and `tests/` is empty.

**Not run, and why.** `mac_satisfied` sits on every solve's path. Every
solver test therefore moves by up to truncation size wherever its trees have
exact ties: `MultiSolve`, `SolveFusedM2L`, `CartesianTaylorSolve`,
`FarFieldContract` and the examples. Their tolerances are not this tight, and
`tests/data/` holds no other committed record. The `CLAUDE.md` build rule
limits this task to its two stems, so none were built or run.

**Affects:**
- `fix-hang-rebalance.md` H2: the partitioner arm starts with `UpwardSweep`
  and `LaplaceSolve` passing at SERIAL np 1-6 and HIP np 1-4. Nothing is
  carried.
- V1's and E1's recorded M2L operator counts and accuracy figures predate the
  tie guard, so a comparison against them must account for it.
