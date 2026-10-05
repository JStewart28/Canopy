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
