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
