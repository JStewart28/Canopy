# An abstract far-field backend for Canopy — progress log

Session record for abstract-solver-backend. Companion to
[abstract-solver-backend.md](abstract-solver-backend.md), which holds the
design, the task sequence and the risks; this file holds what actually
happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `abstract-solver-backend.md`
can cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task, name
the topic.

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

Three things this project in particular will want back later, so record them
where they arise:

- **The commit the golden reference data was generated at** (T1), and every
  later regeneration with the reason for it. Reference data whose provenance is
  unknown cannot be trusted to mean anything.
- **The profiling breakdown before and after T3 and T4** (R3). A `constexpr`
  that quietly became a runtime value has no correctness signal at all, and
  these two numbers are the only record that it did not happen.
- **The measured `n_unique_ops` and `n_unique_ops * bytes_per_key`** from T8's
  instrumentation (R7). The current figure is a claim in a tuning comment, not a
  measurement, and it is the number most likely to decide whether a
  compressed-operator basis is worth building at all.

## T1 — the golden bit-for-bit harness

The harness is built, committed and demonstrably sensitive. Its exit criterion
is **not met**, and cannot be met as written, because running it revealed that
the pipeline is not reproducible run-to-run at any rank count above two. T1 is
left **BLOCKED**, not `DONE`. Everything below is what was decided, what was
measured, and what the blocker actually is.

### Decisions

- **Reference data is keyed by `(nprocs, rank)` — 21 sets in one committed
  file.** The particle seed is `1234 + rank * 31 + P`
  (`tests/tstMultiSolve.hpp:757`), so the particle set and every artifact
  derived from it is a function of both the rank count and the rank. A
  serial-only harness was rejected: T4's MPI packing change and T10's Allreduce
  change would then have no bit-for-bit gate, which is most of why the harness
  exists.
- **`locals()` and the operator table are compared by 64-bit hash plus extents;
  the $A_{n,m}$ table and `n_unique_ops` by full bit patterns.** The operator
  table is 21.4 KB per key at $P=6$, so committing it in full for 21 sets is not
  affordable. Full arrays are dumped to the build directory on mismatch, and the
  failure message names the path, so R2's direct key-list comparison stays
  possible.
- **A non-zero `total_fallback_pair_count()` stops the task.** It is asserted to
  be 0 rather than pinned to a measurement: R4's discriminator and T8's exit
  criterion both rest on it being zero at $P=6$, and pinning a non-zero value
  would retire that discriminator silently. It measured 0 at every rank count
  and every rank, so this did not fire.
- **The commit the reference data was generated at is recorded** — in the log
  here, in the file's own provenance header, and in the commit message. Data
  whose provenance is unknown cannot be trusted to mean anything.

Two smaller choices not dictated by the task:

- **`DownwardSweep::A_table()` rather than `Solver::upward()`.** The task
  allowed either. `_A_table` is borrowed from the upward sweep at
  `setup()` (`src/Canopy_DownwardSweep.hpp:529`), so it is the same view; the
  accessor keeps the whole diagnostic surface on one class and leaves
  `src/Canopy_Solver.hpp` untouched.
- **The regeneration gate is an environment variable, not a CMake option.**
  `CANOPY_GOLDEN_REGENERATE=<dir>` makes each rank write
  `<dir>/golden_np<N>_rank<R>.part` and skip the comparison. Unset — the default,
  and what CTest runs — the test always compares and never writes, so a later
  task cannot silently re-baseline itself.

### Accessor signatures added

All in `class DownwardSweep`, in the public block that begins at
`src/Canopy_DownwardSweep.hpp:408`, and all additive and read-only:

```cpp
using m2l_op_table_view_type =
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, memory_space>;
using m2l_key_type = M2LKey;   // public alias for the private nested struct

const m2l_op_table_view_type&      m2l_op_table()      const;
const std::vector<m2l_key_type>&   m2l_realized_keys() const;
int                                m2l_n_unique_ops()  const;
const a_view_type&                 A_table()           const;
```

plus one private member, `std::vector<M2LKey> _m2l_realized_keys`, filled by a
single added statement `_m2l_realized_keys = ops;` immediately after
`n_unique_ops` is computed in `build_interaction_list_device`. `ops` was a
function-local discarded on return; it is copied, not moved, because the
stage-4 table build still consumes it.

The accessor is `m2l_n_unique_ops()`, not `n_unique_ops()`, so it cannot be
shadowed by the function-local `n_unique_ops` a few lines below it.

`clang-format` was run only over the added line ranges. Running it over the
whole file rewrites 431 lines: `src/Canopy_DownwardSweep.hpp` is not currently
`.clang-format`-clean, and reformatting it would have buried exactly the diff
T1 exists to keep attributable.

### Reference data provenance

Generated at commit **`6f7907515394e7c3987d193448703662171fa077`**, on tuolumne,
Cray clang 20.0.0, spack env `tuolumne_trilinos`, `RelWithDebInfo`, Kokkos
SERIAL backend, flux job `f3XHNLmod1Wo`, via
`scripts/tuolumne/run_golden_regenerate.flux`. Committed as
`tests/data/golden_solid_harmonic_P6.txt` (21 records, 80 KB).

`.gitignore` had a bare `data` rule that excluded `tests/data` entirely; it
gained a `!tests/data/` negation, since this is a committed test input rather
than generated output.

### Measurements

`total_fallback_pair_count()` is **0** at every rank count and every rank, in
every run performed (four independent runs). The count cap is 32768 and the
realized key counts are two orders of magnitude below it, so nothing came close
to the overflow valve.

`n_unique_ops` as committed (the values from the generating run):

| nprocs | num_cells | n_unique_ops by rank |
| --- | --- | --- |
| 1 | 80 | 340 |
| 2 | 170 | 1126, 1159 |
| 3 | 316 | 1605, 1259, 1339 |
| 4 | 431 | 1557, 1184, 1116, 1077 |
| 5 | 500 | 1497, 979, 805, 948, 890 |
| 6 | 524 | 1329, 964, 781, 973, 1142, 894 |

The largest value seen across all runs is **1630**. At 21.4 KB per key that is
about 34 MB of operator table per rank, not the terabytes R7 fears — but this is
a 400-particle-per-rank tree at `max_depth = 6`, so it bounds nothing at
production scale. It is a floor for T8's instrumentation, not an answer to R7.

### The blocker: the pipeline is not reproducible run-to-run above two ranks

The harness passed at every rank count in the run that generated the data, then
failed at np=3 and np=6 when run again against that data minutes later, same
binary, same commit. Running it three more times (flux job `f3XHRJBSnxYf`) gives
the shape of it — `n_unique_ops` for one `(nprocs, rank)` across three
consecutive runs:

| np,rank | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| 1,0 | 340 | 340 | 340 |
| 2,0 | 1126 | 1126 | 1126 |
| 2,1 | 1159 | 1159 | 1159 |
| 3,0 | **1630** | **1605** | 1605 |
| 3,1 | **1261** | **1259** | 1259 |
| 4,2 | **1116** | **1112** | 1116 |
| 4,3 | **1077** | **1063** | 1077 |
| 5,0 | **1496** | **1497** | **1492** |
| 6,1 | **925** | **964** | 964 |
| 6,3 | **974** | **947** | **973** |

**np=1 and np=2 are stable across every run; every rank count from 3 to 6
varies.** `num_cells` is identical across all runs at every rank count (80, 170,
316, 431, 500, 524), so the tree build is deterministic and it is the
*ownership* of those cells that moves. A different leaf-to-rank assignment
changes each rank's interaction lists, hence its realized key set, hence
`n_unique_ops`, the operator table and `locals()`.

The cause is stated in the source itself, at
`src/Canopy_TreePartitioner.hpp:417-419`:

> Solve on rank 0 only, then broadcast. We cannot use ther deterministic "rcb"
> algorithm because it breaks on Tuolumne. The "multijagged" algorithm is
> non-deterministic, so only rank 0 computes, then broadcasts.

Computing on rank 0 and broadcasting makes the partition *consistent across
ranks within one run*. It does nothing about reproducibility *across* runs, and
that is what a bit-for-bit gate needs. With one or two parts the multijagged cut
is trivial and comes out the same every time, which is exactly the observed
np≤2 / np≥3 split. A plausible but unverified mechanism for the remaining
non-determinism is that Zoltan2's MJ runs on `Kokkos::DefaultExecutionSpace`,
which is HIP in this build even for the SERIAL test binaries; that was not
tested and should not be treated as established.

This contradicts the design's own claim in
[The bit-for-bit gate](abstract-solver-backend.md#the-bit-for-bit-gate) that
"the CSR — and therefore the summation order — is deterministic at fixed rank
count". That argument is sound as far as it goes — it is about
`M2LPlan::interaction_lists` and the `std::sort` over `(depth, target_idx)` —
but it only ever examined the M2L CSR. It never examined the partitioner, which
decides the *input* to that CSR. Determinism of the sweep given a partition does
not give determinism of the solve.

Nothing was worked around: no rank count was dropped, no comparison was
loosened, and the reference data was not regenerated to paper over the drift.

### What the harness does demonstrate

- It **passes at ranks 1 and 2** on unmodified code, against committed data
  generated in an earlier run — so the harness, the file format, the hash and
  the data plumbing are all correct.
- With the `m` loop at `src/Canopy_DownwardSweep.hpp:1559` reversed by hand to
  `for ( int m = n; m >= -n; m-- )` — mathematically identical, bitwise
  different — it **fails on the `locals()` comparison and on nothing else** at
  both np=1 and np=2 (flux job `f3XHf4KgoWFZ`): no operator-table, key-list,
  `n_unique_ops` or $A_{n,m}$ failure accompanies it. That is the precise
  discrimination the exit criterion asks for, and it is the half of the
  criterion that tests the harness rather than the code. The perturbation was
  reverted and the target rebuilt.
- The line is `:1559` rather than the `:1504` the task names, because the
  accessors added 55 lines above it. It is the only
  `for ( int m = -n; m <= n; m++ )` in the file.

### Ship gate

`ctest --output-on-failure -L regression -R MPI_SERIAL` (flux job
`f3XHShznrAEs`) **fails, and the failures predate T1.** Six `MultiSolve` tests
fail at np=1 and np=2 on a position/velocity check with `fmm_tolerance = 1e-8`
against measured errors of 3e-7 to 9e-6 (`tests/tstMultiSolve.hpp:542,546`):
`StableTree_Migrate`, `IntermediateMotion_Rebalance`, `LargeMotion_Rebuild`,
`AutoMaintain`, `AutoRebalance`, `M2L_BinEdge_Fallback`. The job then hung at
np=3 and was killed at the 15-minute wall.

These are not T1's doing. Checking out the pre-T1
`src/Canopy_DownwardSweep.hpp` (commit `a6c90de`), rebuilding and rerunning
reproduces **the identical error values to every digit** — e.g.
`3.485035469067542e-07`, `6.8419528791564039e-07`, `9.1947965989306709e-06`
(flux job `f3XHbvZ9xeVV`). The identity of those digits is also independent
evidence that the accessors change no arithmetic.

`README.md` "Known Issues" currently asserts these same tests "pass at 1–6
ranks" and that `SolveFusedM2L.FP32_smokeTest` is the only regression-suite
failure; that is now stale — `FP32_smokeTest` passed at np=1 in these runs while
the six above failed. README has been updated.

### Operational notes for later sessions on tuolumne

- `# flux: --time=N` is rejected by this flux ("unrecognized arguments"); the
  option is `--time-limit`. `scripts/tuolumne/run_ctest_minset.flux` still has
  `--time=15` and therefore will not submit as written. The new scripts use
  `--time-limit`.
- `flux batch --flags=waitable` is refused: "only the instance owner can submit
  with FLUX_JOB_WAITABLE". Submit plainly and wait with `flux job status
  <jobid>`, which blocks and returns the job's exit code.
- The regression gate needs more than 15 minutes of wall on this checkout when
  np=3 hangs. Budget accordingly, or expect a TIMEOUT that looks like a failure.

**Affects:** **T3** — its exit criterion is a bit-for-bit comparison of the M2L
move, and above two ranks there is no stable baseline to compare against; as the
document's own gate for the whole design, T3 cannot be run as written until the
partitioner is deterministic. **T4** and **T10** — both were the reason the
harness is multi-rank at all, and both now have a bit-for-bit gate only at np=2.
**T7** — R2's presentation ("golden test failing on the sorted-key-list and
`n_unique_ops` artifacts") is indistinguishable from partitioner drift above two
ranks, so R2 is only diagnosable at np≤2 until this is fixed. **T8** — R4's
discriminator is intact: `total_fallback_pair_count()` is 0 everywhere measured.
**T9** — its re-solve-after-`invalidate_interaction_list()` check must not
re-partition, or R5's drift and this non-determinism will be
indistinguishable. **R7** — the measured realized key count at $P=6$ on this
small configuration peaks at 1630 per rank, not thousands-scaling-to-TB; T8's
instrumentation should measure at production scale before R7 is judged. **A new
task is needed before T3**: make `TreePartitioner::partition_leaves`
reproducible run-to-run (a deterministic partitioner, a seeded/serial MJ, or
caching-and-committing the assignment), since the entire bit-for-bit strategy
rests on it.

## T1 — the Laplace-solve gate (50-step harness)

`tests/tstGolden.hpp` became `tests/tstLaplaceSolve.hpp`, the single solve
became a 50-timestep loop over a fixed *global* particle set, and the one
bit-for-bit test became three gates split by rank count. All of that is built,
formatted, committed and demonstrably working. **The exit criterion is not
met** and T1 stays **IN PROGRESS**, because running it revealed that the frozen
configuration the design specifies destroys the very thing the harness exists
to protect: at the 50th solve the far field is not evaluated at all.
Everything below is what was decided, what was measured, and what the blocker
is.

### Decisions

- **The particle set is a global 600 from seed `1234 + P`, sliced
  contiguously.** Confirmed working: `initial_hash` measured
  `0xd9e3c66ef5718015` at every rank and every rank count from 1 to 6, so the
  set really is rank-count-independent and the np=$k$-versus-np=1 comparison is
  well defined.
- **Velocities start at zero.** The task's generator specifies positions and
  charges only, and the time loop needs a velocity. Zero is the only choice
  that adds no new frozen constant. It is not a free choice for the next
  session to revisit casually: it is part of what the initial-set hash and
  every committed record would mean.
- **The trailing update and `migrate` after the last solve are omitted.** The
  loop is 50 solves with 49 intervening maintenance steps, not 50 of each.
  `migrate()` re-runs `downward.setup()`, which would overwrite `locals()` and
  the operator table before the gate reads them, and the update would move the
  particles away from the positions the field was evaluated at. Reading "the
  50th solve's artifacts and field" and running a maintenance step after that
  solve are mutually exclusive.
- **All three tests parse the reference file on every rank**, so the
  initial-particle-set hash is checked everywhere before anything else, but
  only rank 0 performs the field comparisons. There are no collectives after
  the gather, so a rank-0-only `ASSERT_` cannot desynchronize the ranks.
- **`crossRankAgreement` and `matchesDirectSum` skip outright in regeneration
  mode.** Only `bitForBitArtifacts` writes, and only it can: it runs at np 1-2,
  which is exactly the set of records the file holds. The np=1 rank-0 part
  additionally carries the `initial` hash and the np=1 `field` record. One
  `ctest` pass over ranks 1-6 therefore produces exactly three parts.
- **`total_fallback_pair_count()` is asserted to be 0**, in both
  `bitForBitArtifacts` and `crossRankAgreement`, never pinned to a measurement.
  It measured **0 at every rank, every rank count, and both step counts**, so
  R4's discriminator is intact.

### File format

The committed file gained two record kinds beside the `set`/`end` bit-for-bit
records, which are unchanged: `initial <hash>`, and a `field <n>` block of
`f <pot> <gx> <gy> <gz>` lines in canonical `GlobalId` order terminated by
`endfield`. The parser dispatches on the leading token, so the three kinds may
appear in any order.

### The blocker: the frozen configuration collapses, and the far field stops being evaluated

At `num_steps = 50`, `dt = 1.0e-4`, `drift_multiplier = 1.0`, `softening = 0.0`
and charges uniform on $[-1, 1]$, the closest opposite-charge pair in the
600-particle set free-falls to contact well inside the simulated interval. A
per-step trace (flux job `f3XK5b4VcCNT`, temporary instrumentation since
reverted) shows it precisely:

| step | cells | n_unique_ops | position range | max abs gradient |
| --- | --- | --- | --- | --- |
| 0 | 95 | 604 | [0.0502, 0.9498] | 4.53e+03 |
| 9 | 95 | 604 | [0.0502, 0.9497] | 8.13e+03 |
| 13 | 95 | 604 | [0.0498, 0.9497] | 2.91e+04 |
| 14 | 95 | 604 | [0.0497, 0.9498] | 7.69e+04 |
| **15** | 95 | 604 | [0.0496, 0.9498] | **6.28e+07** |
| 16 | 155 | 1416 | [0.0495, 1.147] | 1.67e+03 |
| 19 | 147 | 1176 | [-1.659, 2.049] | 1.41e+03 |
| 29 | 105 | 322 | [-7.923, 5.057] | 1.93e+03 |
| 39 | 45 | 4 | [-14.19, 8.064] | 4.13e+03 |
| **49** | **29** | **0** | **[-20.45, 11.07]** | 1.74e+06 |

The pair collides at step 15, the participants are ejected at high velocity,
the bounding box grows about thirtyfold, and with `max_depth = 6` the tree
cannot refine into the residual cloud. By step 50 there are 29 cells, **no pair
is MAC-admissible, and `n_unique_ops` is 0 at every rank and every rank count**
(np 1-6, jobs `f3XK1SHwZ5NF` and `f3XK879Eqj4T`).

The estimate agrees: for 600 points uniform on $[0.05, 0.95]^3$ the expected
closest-pair distance is $\approx 0.0099$, giving an acceleration of
$\approx 1.0\times10^{4}$ and a free-fall time of $\approx 1.4\times10^{-3}$,
against a simulated interval of $50 \times 10^{-4} = 5\times10^{-3}$. The
collapse is not marginal; there is roughly a factor of 3.5 of headroom.

This is not a defect in the sweep, in the harness, or in the accessors. It is
the specified configuration. `tests/tstMultiSolve.hpp:562-567` runs the same
`dt` and `drift_multiplier` but only **5** steps, and draws charges from
$[0.5, 1.5]$ — all one sign. T1 pairs a 50-step integration with the golden
harness's $\pm 1$ charges, and the two had never been run together.

**Why this defeats the gate rather than merely degrading it.** The direct-sum
deviation at step 50 measures **1e-15 to 2e-15** — machine precision, not the
$\approx 8\times10^{-3}$ truncation a working far field carries. The solve has
silently become pure P2P. Consequently:

- `bitForBitArtifacts` would compare an operator table with zero realized
  columns. Both its `optab` and `keys` hashes come back as
  `0x14650fb0739d0383`, which is simply the FNV-1a seed over an empty input.
- `matchesDirectSum` passes trivially, because P2P is exact.
- The exit criterion's **first sensitivity perturbation cannot fire.** The `m`
  loop at `src/Canopy_DownwardSweep.hpp:1559` sits inside the fused M2L apply,
  reached only through a CSR entry with a valid `op_idx`. With zero realized
  operators the loop body never executes, so reversing it changes no bits and
  `bitForBitArtifacts` would *pass* under a perturbation it is required to
  fail. The perturbations were therefore **not run**: at this configuration the
  criterion is not merely unmet, it is unmeetable.

No workaround was applied. The particle count, the step count, `dt`,
`drift_multiplier`, the tolerances and the rank counts are all exactly as the
design specifies, and no reference data was committed from the degenerate
state.

### Measurements

Both step counts, on unmodified code, SERIAL backend. `fallback_pairs` is 0
throughout. Deviations are normalized by a global scale — $\max|\varphi|$ and
$\max|\nabla\varphi|$ over the reference set — never per-particle.

**`num_steps = 1` (job `f3XKLVSa9diw`) — the tree is healthy and M2L is fully
exercised: 95 cells, `n_unique_ops` 604 at np=1 and 348/316 at np=2.**

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 3.2507385349279139e-06 | 2.3644684347284734e-06 |
| 2 | 1.1775774170600731e-15 | 2.8346336198383183e-13 | 3.2507385351802518e-06 | 2.3644685309655667e-06 |
| 3 | 1.3458027623543694e-15 | 2.8343581811170314e-13 | 3.2507385350961393e-06 | 2.3644685309467664e-06 |
| 4 | 1.0303802399275642e-15 | 2.8347559461330696e-13 | 3.2507385350120264e-06 | 2.3644684347373647e-06 |
| 5 | 9.2523939911862892e-16 | 4.0083877814098415e-13 | 3.2507385349279139e-06 | 2.3644684347321203e-06 |
| 6 | 1.5140281076486656e-15 | 2.8347559690854843e-13 | 3.2507385345073502e-06 | 2.3644685309769457e-06 |

Worst cross-rank deviation **4.0083877814098415e-13** (np=5, gradient); 100x is
**4.0e-11**. Worst direct-sum deviation **3.2507385351802518e-06** (np=2,
potential); 3x is **9.8e-06**. These are the values the tolerances would be
pinned at *if one step were the frozen configuration*. It is not, so they are
recorded here and `LS_CROSS_RANK_TOL` / `LS_DIRECT_SUM_TOL` are left
deliberately unpinned, with a comment in the header saying so.

**`num_steps = 50` (job `f3XK879Eqj4T`) — the degenerate state described
above; `n_unique_ops` is 0 everywhere.**

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 5.9019702405243413e-16 | 6.8994946976919231e-16 |
| 2 | 3.1119400878717255e-07 | 6.3084034822540923e-07 | 3.934648051455127e-16 | 5.8736844457938883e-16 |
| 3 | 7.9851024544389753e-07 | 1.5986976962365076e-06 | 1.5738599875498137e-15 | 5.6118219870778517e-16 |
| 4 | 5.6146327358761271e-07 | 9.6987566395904367e-07 | 1.5738594988117639e-15 | 4.9355545064264574e-16 |
| 5 | 9.1329590373755531e-08 | 1.5322878490201487e-07 | 1.9673232624470761e-15 | 5.5633213998306189e-16 |
| 6 | hung (see below) | | | |

`n_unique_ops` at `num_steps = 1`: np=1 → 604; np=2 → 348, 316. At
`num_steps = 50`: 0 at every rank and rank count. Compare the earlier
single-solve harness, whose np=1 value was 340 at 400 particles; 604 at 600
particles on the same tree parameters is the expected scaling.

### R8 does not fire

The 50-step cross-rank deviation reaches 1.6e-06, three orders of magnitude
above the 1e-9 threshold, which read alone looks exactly like R8 — an FMM
answer that depends on the partition. **It is not.** The design's own
discriminator settles it: at `num_steps = 1`, on the same configuration and the
same code, the cross-rank deviation is **1e-15 on the potential and 3e-13 on
the gradient at every rank count from 2 to 6**. A single np=$k$ solve
reproduces a single np=1 solve to floating-point reassociation, precisely as
[the bit-for-bit gate](abstract-solver-backend.md#the-bit-for-bit-gate) argues
it must. What the 50-step number measures is the integrator amplifying that
reassociation through a near-collision, where the trajectory is chaotic and
Lyapunov growth is unbounded — not a partition-dependent answer.

So the cross-rank half of the gate is **sound in principle and validated in
practice at ranks 2-6**. Only the 50-step configuration is unusable.

### A second finding: np=6 hangs on the degenerate tree

In job `f3XK879Eqj4T`, `LaplaceSolve.crossRankAgreement` at np=6 produced no
output for over 14 minutes and the job was killed at its 20-minute wall. np 1-5
each completed the same test in 8-12 seconds. At `num_steps = 1` (job
`f3XKLVSa9diw`) np=6 completes in **7.4 seconds** and all six rank counts pass,
so this is not an inherent np=6 problem and not a defect in the harness: it is
the post-collapse tree — 29 cells spread over a box roughly thirty times the
original, at six ranks — that hangs. It is plausibly related to the np=3 hang
this log already records against the regression suite, but that was not
established and should not be assumed.

### What the harness does demonstrate

Run against provisional reference data generated from the same binary in an
*earlier job*, the machinery is correct end to end:

- **`bitForBitArtifacts` passes at np=1 and at both ranks of np=2**, comparing
  bit patterns written by job `f3XK1SHwZ5NF` against a solve in job
  `f3XK879Eqj4T`. Bit-for-bit reproducibility across runs at np 1-2 is
  confirmed for the 50-step loop, so `migrate` really does leave the partition
  alone, as the design's reason for choosing it over `rebalance` predicted.
- **`bitForBitArtifacts` skips at np 3-6** with the partitioner message, so
  `ctest` output shows the gate skipped rather than passed.
- **The `initial` hash check, the `GlobalId` pairing, the gather, the direct
  sum, the mismatch dumps, the regeneration gate and the parser all work.** At
  `num_steps = 1` the whole suite reports `100% tests passed, 0 tests failed
  out of 6`.

The one thing not exercised is the pair of sensitivity perturbations, for the
reason given above.

### Repository state left behind

- `tests/data/laplace_solve_P6.txt` is **not committed.** Generating it from
  the degenerate state would commit a baseline whose operator table is empty,
  and every later task would compare against nothing. `ctest -R
  Canopy_Test_LaplaceSolve_MPI_SERIAL` therefore fails at every rank count with
  `cannot open reference data file`, naming the regeneration script. That is
  the honest IN-PROGRESS state; `LaplaceSolve` is in the `unit` tier and does
  not gate ships.
- `tests/data/golden_solid_harmonic_P6.txt` is **left in place but orphaned** —
  `tstGolden.hpp` no longer exists and its own header now names a deleted file.
  It is keyed to the old per-rank generator (`1234 + rank * 31 + P`, 400 per
  rank) and cannot serve the new harness. Delete it when valid replacement data
  is committed.
- The reference data that *was* generated, at commit
  `0d51d790713b09b9ee26ec999b9414bbf584f0b3` (tuolumne, Cray clang 20.0.0,
  spack env `tuolumne_trilinos`, RelWithDebInfo, Kokkos SERIAL, flux job
  `f3XK1SHwZ5NF`), is recorded here for provenance only and was discarded.
- Temporary artifacts — the per-step trace instrumentation, `t1_trace.flux`,
  `t1_onestep.flux` and the provisional data file — were all removed, and
  `LS_NUM_STEPS` is back at 50. Nothing under `src/` was modified at any point.

### What the next session has to decide

The blocker is a configuration question, and it is **not** one this session had
the authority to answer: `num_steps = 50`, `dt = 1.0e-4`,
`drift_multiplier = 1.0`, `softening = 0.0` and charges on $[-1, 1]$ are frozen
by the design, and changing any of them silently redefines what T1's **DONE**
means. The options, in the order they preserve the design's intent:

1. **Give the frozen configuration a softening floor.** `FmmConfig::softening`
   is already a knob and is currently 0.0. A non-zero value removes the
   two-body singularity without touching the integrator. It changes what the
   MAC softening floor does, though — the design notes `near_softening_factor`
   "only has an effect when softening > 0" — so R4's `total_fallback_pair_count
   == 0` would have to be re-verified.
2. **Make the charges one-signed**, as `tstMultiSolve` does. Gravity still
   collapses, but far more slowly than an opposite-charge pair at contact.
3. **Shorten the interval** to a step count that stays inside the collapse
   time. The trace puts the first collision at step 15, so anything up to about
   10 steps is safe at this `dt`; 5 is what `MultiSolve.StableTree_Migrate`
   uses. This is the smallest change but it weakens the "state at step 50 is
   cumulative" argument the design leans on.
4. **Reduce `dt`.** Scaling `dt` down by 10 moves the collision out past step
   150 and leaves 50 steps comfortably inside it, at the cost of the
   trajectory barely evolving.

Whichever is chosen, the harness itself needs no structural change: only the
constants in the frozen-configuration block, a regeneration run, and the
tolerance pinning that the 1-step numbers above already show is achievable
(4.0e-11 cross-rank, 9.8e-06 direct-sum).

**Affects:** **T1 itself** — remains IN PROGRESS; the file, the renames, the
CMake wiring, the scripts and all three gates are done, and what is outstanding
is a frozen-configuration decision plus one regeneration and one measurement
run. **T3** — its bit-for-bit gate at np 1-2 is now demonstrated to work
across runs for a 50-step `migrate` loop, which is stronger than the previous
session's single-solve result; but it has no committed baseline until T1's
configuration is settled. **T4** and **T10** — their tight multi-rank gate is
in better shape than the previous log entry implied: `crossRankAgreement`
measures 1e-15/3e-13 at np 2-6 on a healthy tree, so the cross-rank check is a
real gate for the MPI packing and the shared-cell Allreduce, provided the
configuration evaluates a far field at all. **R8** — **retired as a live risk
at this configuration**: the FMM answer is partition-independent to
reassociation at every rank count from 2 to 6, measured directly. **R4** —
discriminator still intact, `total_fallback_pair_count()` is 0 at every rank,
rank count and step count measured. **T8** — `n_unique_ops` at 600 particles on
a healthy tree is 604 at np=1 and 348/316 at np=2, well under the 32768 count
cap, consistent with the earlier 400-particle figures. **A new investigation
may be warranted**: `crossRankAgreement` hangs at np=6 on a tree degenerated by
particle ejection; whether that shares a cause with the np=3 regression-suite
hang recorded above is unknown.

## T1 — the Laplace-solve gate (12-step harness, completed)

T1 is **DONE**. The blocker the previous session left — a frozen configuration
whose last solve evaluated no far field at all — is resolved, the reference data
is committed, both tolerances are pinned at measured values, and both
sensitivity perturbations have been run. One of the two does not behave the way
the design document predicts, for a measured reason recorded below; nothing was
worked around to make it look otherwise.

### Decisions taken (given, not chosen here)

- **Charges are uniform on $[0.5, 1.5]$**, replacing $[-1, 1]$ — the
  distribution `tests/tstMultiSolve.hpp:200` uses for its gravity tests. Seed
  stays `1234 + P`, positions stay uniform on `[0.05, 0.95]`, velocities stay
  zero, and the draw order stays one position triple then one charge per
  particle. The `initial` hash moved from `0xd9e3c66ef5718015` to
  **`0xb6ad437608ad69b7`** as a result, and the committed value comes from the
  regeneration run rather than being carried over. It measures identically at
  every rank and every rank count from 1 to 6, so the set is still
  rank-count-independent and the np=$k$-versus-np=1 comparison is still well
  defined.
- **`num_steps` chosen by measurement, then frozen** — see the trace below.
- **The global-scale normalization stays** — $\max|\varphi|$ and
  $\max|\nabla\varphi|$ over the gathered set, for both `crossRankAgreement`
  and `matchesDirectSum`. The justifying comment changed with the charges: the
  one-signed set does keep per-particle $|\varphi|$ away from zero, but the
  gradient components still pass through zero by cancellation wherever a
  particle's neighbours pull against each other, so a per-particle ratio there
  would measure cancellation rather than accuracy.
- **`dt = 1.0e-4`, `drift_multiplier = 1.0`, `softening = 0.0` and `migrate`
  unchanged.** Softening was not reached for; the step count was the knob.

### The per-step trace, and the chosen step count

Temporary instrumentation in the time loop, `num_steps = 50`, one-signed
charges, np=1 and np=2 (flux job `f3XKeDE9os3u`; instrumentation since
removed). np=2 reproduces np=1's cell count, position range and maximum
gradient to every printed digit through step 19, so the physics does not depend
on the rank count. Step 0's range is the initial bounding box, because the
trace prints after the solve and before the update.

| step | cells | n_unique_ops | position range (x / y / z) | max abs gradient |
| --- | --- | --- | --- | --- |
| 0 | 95 | 604 | [0.0528, 0.9497] / [0.0502, 0.9498] / [0.0503, 0.9494] | 8.60e+03 |
| 4 | 95 | 604 | [0.0529, 0.9496] / [0.0503, 0.9496] / [0.0504, 0.9493] | 1.17e+04 |
| 5 | 103 | 686 | [0.0530, 0.9495] / [0.0504, 0.9495] / [0.0504, 0.9493] | 1.41e+04 |
| 8 | 103 | 686 | [0.0533, 0.9492] / [0.0506, 0.9492] / [0.0506, 0.9490] | 5.94e+04 |
| 9 | 103 | 686 | [0.0535, 0.9491] / [0.0507, 0.9490] / [0.0506, 0.9489] | 5.99e+05 |
| 10 | 103 | 686 | [0.0536, 0.9490] / [0.0508, 0.9489] / [0.0507, 0.9488] | 1.02e+05 |
| **11** | **103** | **686** | **[0.0538, 0.9489] / [0.0510, 0.9487] / [0.0508, 0.9487]** | **6.88e+05** |
| 12 | 103 | 686 | [0.0540, 0.9488] / [0.0511, 0.9485] / [0.0509, 0.9486] | 3.58e+04 |
| 16 | 103 | 686 | [0.0548, 0.9481] / [0.0517, 0.9475] / [0.0513, 0.9480] | 8.54e+05 |
| 17 | 103 | 686 | [0.0551, 0.9479] / [0.0519, 0.9472] / [0.0515, 0.9478] | 7.47e+04 |
| **18** | 103 | 686 | [0.0554, 0.9478] / [0.0521, 0.9469] / [0.0516, 0.9476] | **3.81e+06** |
| 21 | 115 | 1008 | [0.0560, 0.9811] / [0.0528, 0.9459] / [0.0521, 0.9469] | 1.09e+06 |
| 27 | 152 | 1762 | [0.0576, 1.088] / [-0.2380, 1.396] / [-0.0786, 1.232] | 5.37e+04 |
| 34 | 193 | 1992 | [0.0490, 1.212] / [-3.197, 4.156] / [-2.070, 3.216] | 1.55e+05 |
| 40 | 133 | 616 | [-0.7934, 1.548] / [-5.734, 6.522] / [-3.894, 4.916] | 2.65e+06 |
| 44 | 92 | 148 | [-1.355, 2.028] / [-7.425, 8.099] / [-5.110, 6.049] | 1.62e+05 |
| 49 | 71 | 72 | [-2.057, 2.627] / [-9.539, 10.07] / [-6.630, 7.466] | 1.14e+06 |

Applying the three degeneracy tests against step 0 (`n_unique_ops` below 302;
any per-dimension range wider than 1.5x the initial width of 0.897 / 0.900 /
0.899; max gradient above 8.60e+05):

- `n_unique_ops` never falls below 604 before step 40, so it does not fire
  first.
- The position range stays inside the initial box until step 20 and does not
  exceed 1.5x any initial width until well past step 27, so it does not fire
  first.
- **The maximum gradient fires at step 18**, at 3.81e+06 against a 8.60e+05
  threshold. Steps 9 (5.99e+05), 11 (6.88e+05) and 16 (8.54e+05) all approach
  it and none crosses it — step 16 misses by 0.7%.

So the first degenerate step is **18**, and `LS_NUM_STEPS` = $\lfloor 2/3
\times 18 \rfloor$ = **12**. The last solve is therefore step index 11: 103
cells, 686 realized operators at np=1, bounding box still the initial one.

**On the shape of the gradient trace.** The spikes from step 9 onward are a
close same-sign pair accelerating together, passing, and scattering hard with
`softening = 0.0` — the collapse the design predicted, arriving as a cloud
rather than as a two-body singularity. It is slower than the $[-1, 1]$ case's
collision at step 15, but not by a large factor; one-signed charges alone were
not sufficient and the measured step count was load-bearing. Note also that the
`num_steps = 50` end state is *less* degenerate than the $[-1, 1]$ run's was —
71 cells and 72 operators at step 49 rather than 29 and 0 — which is why the
degeneracy test had to be applied to the trace rather than to the end state.

### Reference data provenance

`tests/data/laplace_solve_P6.txt` (846 lines, 62 KB) generated at commit
**`fedf400e10a409f14a6cb06cc71891eda1742dc4`** on tuolumne, Cray clang 20.0.0,
spack env `tuolumne_trilinos`, `RelWithDebInfo`, Kokkos SERIAL backend, flux
job **`f3XKhLPnyd9H`**, via
`scripts/tuolumne/run_laplace_solve_regenerate.flux`. The three `.part` files
were merged in the header's stated order — (1,0), (2,0), (2,1) — under a new
provenance header modelled on the deleted golden file's, describing the new
configuration and both new record kinds. Committed in `371686a`, which also
deletes `tests/data/golden_solid_harmonic_P6.txt`: it was keyed to the removed
per-rank generator and to `tests/tstGolden.hpp`, and nothing read it.

The regeneration pass reported `100% tests passed, 0 tests failed out of 6` and
wrote exactly the three expected parts. Bit-for-bit records:

| np, rank | locals hash | optab hash | keys hash | n_unique_ops |
| --- | --- | --- | --- | --- |
| 1,0 | `0xfb2cddef75e26dd6` | `0x2d3ed2544cc860cb` | `0x18e62ec1353e24e3` | 686 |
| 2,0 | `0xff96d2ad9682a340` | (in file) | (in file) | 368 |
| 2,1 | `0xc25b035a7d6a1fcd` | (in file) | (in file) | 386 |

`locals_ext` is `(103, 28, 1)` and the $A_{n,m}$ extent 169 at every record.

### Pinned tolerance measurements

Measured on unmodified code at every rank count against the committed data
(flux job `f3XKiNmyT48o`), then reproduced to every digit by the verification
run with the pinned values (flux job `f3XKkSvw54pF`) and again by the final
post-revert run (flux job `f3XTw8Jdt7eo`). `fallback_pairs` is 0 throughout.
Deviations are normalized by a global scale, never per-particle.

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 3.2093610331931809e-07 | 4.2399302231264458e-08 |
| 2 | 4.1994107222659022e-13 | 2.1570013757642702e-12 | 3.2093610363952985e-07 | 4.239936667274564e-08 |
| 3 | 8.0211305411572796e-13 | 4.0353546216363242e-12 | 3.2093610299898352e-07 | 4.2399380705248681e-08 |
| 4 | 1.114294857300434e-12 | **5.5987399483706545e-12** | 3.2093610299888352e-07 | 4.2399368624331468e-08 |
| 5 | 9.5809726336249496e-14 | 6.4438389009577268e-13 | 3.209361028925181e-07 | 4.2399357908696204e-08 |
| 6 | 5.688835866646797e-13 | 2.8631357246763719e-12 | 3.2093610299905848e-07 | 4.2399381662523099e-08 |

- Worst cross-rank deviation **5.5987399483706545e-12** (np=4, gradient). 100x
  is 5.6e-10, and `LS_CROSS_RANK_TOL` is pinned there.
- Worst direct-sum deviation **3.2093610363952985e-07** (np=2, potential). 3x
  is 9.63e-07, and `LS_DIRECT_SUM_TOL` is pinned there.
- **The stop-and-report clause did not fire.** No cross-rank deviation came
  within two orders of magnitude of the $10^{-9}$ threshold, so no
  `num_steps = 1` re-measurement was required and **R8 stays retired** at this
  configuration. For comparison, the previous session's `num_steps = 1` numbers
  were 1e-15 on the potential and 3e-13 on the gradient; twelve steps of the
  integrator amplify reassociation by roughly one order of magnitude on the
  potential and a factor of about 20 on the gradient, which is exactly the
  amplification the design says the tolerance must be measured rather than
  derived because of.
- The direct-sum potential deviations agree to ten digits across all six rank
  counts and the gradient deviations are 4.24e-08 everywhere, as they should
  be: that error is truncation, not partitioning.
- `n_unique_ops` per rank at the last solve: np=1 → 686; np=2 → 368, 386;
  np=3 → 273, 204, 329; np=4 → 264, 128, 234, 194; np=5 → 180, 168, 147, 217,
  187; np=6 → 174, 156, 111, 160, 116, 175. All well under the 32768 count cap.

### Perturbation 1 — the `m` loop: behaves exactly as required

With `src/Canopy_DownwardSweep.hpp:1559` reversed by hand to
`for ( int m = n; m >= -n; m-- )` (flux job `f3XTnWDziu35`):

- **`bitForBitArtifacts` fails on the `locals()` comparison and on nothing
  else**, at np=1 rank 0, np=2 rank 0 and np=2 rank 1. The only failing
  assertions in the whole job are the `locals()` hash `EXPECT_EQ` and the
  `ADD_FAILURE` that names the dump path. No operator-table, key-list,
  `n_unique_ops`, $A_{n,m}$, extent or fallback assertion failed. Measured
  hashes `0x80fb5305cf818963` / `0x0927c45d161bbb1f` / `0xa5e1cf8234fea508`
  against reference `0xfb2cddef75e26dd6` / `0xff96d2ad9682a340` /
  `0xc25b035a7d6a1fcd`.
- **`crossRankAgreement` passes at np 2-6 and `matchesDirectSum` passes at
  np 1-6**, which is the required distinction: a reassociation-level difference
  is below their tolerances by construction, and only the bitwise gate sees it.
- Unlike the previous session, the perturbation *could* fire, because the tree
  now realizes operators. That was the whole point of re-freezing the
  configuration.

### Perturbation 2 — the shared-cell Allreduce: does not behave as the design states

The design's second perturbation is "offset the running counter by one slot at
`src/Canopy_DownwardSweep.hpp:1825-1835` and `:1847-1857`", expecting
`crossRankAgreement` to fail at np ≥ 2 "while np=1 is unaffected". Neither half
of that outcome is reachable with a slot offset at this configuration, and the
reason is a measured property of `allreduce_shared_locals_at_depth`.

**Applying the offset to both sites is a provable no-op.** The two loops are
the pack (`:1832`) and unpack (`:1854`) of one elementwise `MPI_Allreduce`. Let
$\mathrm{rot}(j) = (j+1) \bmod M$ with $M$ = `per_cell_complex`, $L_r$ the
rank's locals, $S$ the pre-M2L snapshot and $P$ the rank count. Rotating both
sides gives

$$L_{\rm new}[j] = \sum_r L_r[j] - (P-1)\,S[\mathrm{rot}(j)]$$

against a correct $\sum_r L_r[j] - (P-1)\,S[j]$, so the entire error is
$(P-1)\big(S[j] - S[\mathrm{rot}(j)]\big)$ — it vanishes identically when $S$
is constant across slots, and in particular when $S = 0$.

**And $S$ is identically zero.** Temporary diagnostic instrumentation in
`allreduce_shared_locals_at_depth` (flux job `f3XTsRvGuimM`, since removed)
reports `max_abs_snapshot = 0` at **every** depth, every rank and both rank
counts measured:

| np | depth | nshared | per_cell_complex | max abs snapshot | max abs summed delta |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 28 | 0 | 0 |
| 1 | 1 | 8 | 28 | 0 | 0 |
| 1 | 2 | 3-4 | 28 | 0 | 42.5 - 51.9 |
| 1 | ≥3 | 0 | — | (early return) | — |
| 2 | 0 | 1 | 28 | 0 | 0 |
| 2 | 1 | 8 | 28 | 0 | 0 |
| 2 | 2 | 3-4 | 28 | 0 | 42.5 - 51.9 |
| 2 | ≥3 | 0 | — | (early return) | — |

No pair is MAC-admissible at depth 0 or 1 under $\theta = 0.5$, so the locals
inherited by L2L into depths 1 and 2 are zero and the snapshot taken before
M2L at each shared depth is zero. Confirmed empirically as well as
analytically: with the offset applied to both sites (flux job `f3XTprsFKehD`)
**every cross-rank and direct-sum deviation reproduces the unperturbed run to
all 17 printed digits, `bitForBitArtifacts` still passes at np 1-2, and the
suite reports 6/6 passed.** The perturbation as literally specified changes no
bits.

**Offsetting the pack side alone does corrupt the dataflow, and
`crossRankAgreement` catches it — but np=1 is not insulated.** With `:1832`
rotated and `:1854` left canonical (flux job `f3XTuDYZg9G3`), the summed deltas
land in the wrong coefficient slot:

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (skipped) | (skipped) | 2.5639944119095502e-02 | 4.23915390367346e-04 |
| 2 | 2.5532001106757757e-02 | 4.2286661057078494e-04 | 2.5639934677803095e-02 | 4.2391539849971827e-04 |
| 3 | 2.5531189613041513e-02 | 4.2288722529969122e-04 | 2.563994411906759e-02 | 4.239153903221673e-04 |
| 4 | 2.553118961302327e-02 | 4.2288722533406428e-04 | 2.5639944119083106e-02 | 4.2391539035808074e-04 |
| 5 | 2.5531189613432648e-02 | 4.2288722548828976e-04 | 2.5639934677788839e-02 | 4.2391539849631122e-04 |
| 6 | 3.1093538235035616e-02 | 5.1938393056206815e-04 | 3.1209314334698746e-02 | 5.2013335501954001e-04 |

- **`crossRankAgreement` fails at every rank count from 2 to 6**, by seven
  orders of magnitude — 2.6e-02 against a 5.6e-10 tolerance. So the substantive
  claim the perturbation exists to establish — that this test is a real gate on
  the shared-cell dataflow that **T4** and **T10** rewrite — is verified.
- **np=1 is not unaffected.** `bitForBitArtifacts` and `matchesDirectSum` both
  fail at np=1, the latter with a direct-sum potential deviation of 2.56e-02
  against a 9.63e-07 tolerance. The cause is in the table above: the
  shared-cell Allreduce path **runs at np=1**, over 1 shared cell at depth 0, 8
  at depth 1 and 3-4 at depth 2. At one rank `MPI_Allreduce` copies send to
  recv elementwise, so a pack/unpack mismatch corrupts np=1 exactly as it
  corrupts np=2.
- The two cases are exhaustive for a slot offset: a *consistent* offset cancels
  through the elementwise reduction at every rank count, and an *inconsistent*
  one corrupts every rank count including np=1. There is no slot offset whose
  effect is confined to np ≥ 2, because the only $P$-dependent term is the
  snapshot and the snapshot is zero.
- **This contradicts R6's premise** that "shared cells only exist above one
  rank" (and the reading of `src/Canopy_DownwardSweep.hpp:701-708` that the
  np=1 Allreduce is inert). R6's *conclusion* — that a `sets_per_component`
  aliasing bug in these two loops presents as correct at rank 1 and wrong at
  ranks 2-6 — should not be relied on until this is rechecked: on this
  evidence such a bug would present as wrong at rank 1 as well.

Nothing was worked around. No tolerance was raised, no rank count dropped, no
softening enabled, and no reference data was taken from a perturbed or
degenerate state. Both perturbations were reverted with `git checkout` and the
target rebuilt; the final verification run (flux job `f3XTw8Jdt7eo`) confirms
`src/` is byte-identical to `HEAD` and the suite passes 6/6.

### Repository state left behind

- `tests/tstLaplaceSolve.hpp` — charges on $[0.5, 1.5]$, `LS_NUM_STEPS = 12`,
  `LS_CROSS_RANK_TOL = 5.6e-10`, `LS_DIRECT_SUM_TOL = 9.63e-07`,
  `EXPECT_GT( r.n_unique_ops, 0 )` in `bitForBitArtifacts` and
  `crossRankAgreement`, the "NOT YET PINNED" block replaced with the measured
  values and the configuration that produced them, and the normalization
  comment corrected. `clang-format`-clean.
- `tests/data/laplace_solve_P6.txt` committed; `golden_solid_harmonic_P6.txt`
  deleted.
- `scripts/tuolumne/run_ctest_laplace_solve.flux` header says 12-timestep. Both
  flux wrappers were used as written and needed no change beyond that comment.
- **Nothing under `src/` is modified.** All temporary instrumentation — the
  per-step trace, the shared-cell diagnostic — and both perturbations were
  removed. Three commits: `fedf400` (configuration), `371686a` (data),
  `4a351e1` (tolerances).
- Out of scope and untouched, as directed: the partitioner's run-to-run
  non-determinism, the six pre-existing `MultiSolve` regression failures,
  `tasks/todo_0.md`. The np=6 hang the previous session saw on a degenerate
  tree **did not reappear** — np=6 completes in 8.0 s — which is consistent
  with it having been a symptom of the collapse rather than a defect.

### Operational note

`make -j` with no job limit was killed on the login node once (SIGKILL, exit
137) while rebuilding this target; `make -j 4` succeeded immediately
afterwards and for every subsequent rebuild. Worth preferring a bounded `-j`
for by-hand builds on the tuolumne login node.

**Affects:** **T3** — the committed baseline it has been waiting on now exists:
`tests/data/laplace_solve_P6.txt` at `371686a`, and `bitForBitArtifacts` is
demonstrated to attribute a reassociation-level change to `locals()` alone at
np 1-2 on a tree that realizes 686 operators. T3 inherits `num_steps = 12` and
both tolerances. **T4** — inherits the same, and its multi-rank gate is now
directly validated: a pack/unpack corruption in the shared-cell Allreduce is
caught by `crossRankAgreement` at np 2-6 by seven orders of magnitude. **T7**,
**T8**, **T10** — all inherit `num_steps = 12`, `LS_CROSS_RANK_TOL = 5.6e-10`
and `LS_DIRECT_SUM_TOL = 9.63e-07`; regenerating the reference data is required
if any of them changes the configuration, and the file's provenance header says
how. **T8** — `n_unique_ops` on a healthy 600-particle tree is 686 at np=1 and
111-386 per rank at np 2-6, and `total_fallback_pair_count()` is 0 at every
rank and rank count, so **R4**'s discriminator is intact. **R8** — stays
retired at this configuration: the worst measured cross-rank deviation is
5.6e-12, two orders of magnitude below the threshold. **R6 needs correcting** —
its premise that shared cells exist only above one rank is false as measured
(np=1 has 12-13 shared cells across depths 0-2), which weakens its stated
presentation for a `sets_per_component` aliasing bug; **T10 should not rely on
"correct at rank 1, wrong at ranks 2-6" as its discriminator** without
re-establishing it. **A note for whoever revisits the second perturbation:** it
cannot be made to fire at np ≥ 2 only, for the reason measured above; if the
design wants a perturbation that discriminates np=1 from np ≥ 2 in this path,
it needs one that touches the summation rather than the slot indexing.

## T2 — dead solid-harmonic scaffolding deleted

T2 is **DONE**. Six members removed, 142 lines, no arithmetic change
demonstrated by the Laplace-solve gate reproducing T1's entire measured table to
every digit. The regression comparison was completed at np 1-2 and then stopped
at the user's direction because the run was hanging at np=3; that clause of the
exit criterion is therefore partially met, and the gap is np=3-6.

### The searches, run before deleting and again after

Each of the six was searched across `src/`, `tests/` and `examples/` before it
was touched. The document's callers table was correct — no caller had landed
since:

| Symbol | Hits before | What they were |
| --- | --- | --- |
| `apply_p2m_normalization_bridge` | 2 | declaration `Canopy_UpwardSweep.hpp:212`, definition `:418` |
| `apply_l2p_normalization_bridge` | 2 | declaration `Canopy_DownwardSweep.hpp:524`, definition `:2008` |
| `scale_locals_at_depth` | 3 | declaration `:531`, definition `:1961`, and its own `parallel_for` label string `:1978` |
| `M2L_NUM_SRC` | 1 | definition `:306` |
| `has_mplus_symmetry` | 1 | definition `Canopy_LaplaceKernel.hpp:159` |
| `DownwardSweep::P` | 1 code + 4 comments | definition `:110`, read only by `M2L_NUM_SRC`; `:97`, `:296`, `:299`, `:493`, `:1520` are prose |

The third `scale_locals_at_depth` hit is worth naming because a callers table
cannot show it: the Kokkos kernel-label string inside its own body. It is not a
caller, and it disappeared with the body.

`M2L_NUM_SRC` was deleted first and `P` second, as required. After deletion all
five named symbols return **zero hits** across the three directories, and the
only surviving whole-word `P` in `Canopy_DownwardSweep.hpp` is in five prose
comments (`:97`, `:295`, `:298`, `:490`, `:1505`) that refer to the expansion
order as a concept, not to the deleted member. `get_coeff_3d` keeps all seven
references and `Canopy_SphericalCoefficients.hpp:70-91` is untouched.

### Measured line-number offsets after the deletion

Deletions: `Canopy_DownwardSweep.hpp` −100 net (1 at `:110`, 2 at `:304`, 12 at
`:518`, 84 at `:1941`, and −2/+1 on `execute()`'s reworded comment);
`Canopy_UpwardSweep.hpp` −40 (5 at `:208`, 35 at `:411`);
`Canopy_LaplaceKernel.hpp` −1. File lengths went 2190 → 2090 and 683 → 643.

The document's "low by roughly 55 after `:408`" was measured, not guessed, and
it was *approximately* right for the wrong reason: relative to pre-T1 commit
`a6c90de`, T1's three insertions (+6 at `:343`, +44 at `:445`, +5 at `:1071`)
made the tail exactly +55 and the band `:446`-`:1071` +50, so the single figure
hid a two-band structure. After T2 the offset is genuinely multi-band and the
tail **changes sign**. Measured by `git diff -U0 a6c90de -- <file>`, added to a
cited line to reach the current one:

- `Canopy_DownwardSweep.hpp`: `0` ≤`:109`, `-1` ≤`:305`, `-3` ≤`:343`,
  `+3` ≤`:445`, `+47` ≤`:470`, `+35` ≤`:1071`, `+40` ≤`:1903`, `-45` >`:1903`.
- `Canopy_UpwardSweep.hpp`: T1 changed nothing, so citations were exact;
  now `0` ≤`:207`, `-5` ≤`:416`, `-40` >`:416`.

The Current-state paragraph carries both, plus the post-T1-numbering variant for
T1's and T2's own citations. Individual citations elsewhere were deliberately
not renumbered.

### What only running revealed

- **The ship-gate baseline does not match what Current state describes.** Flux
  job **`f3XUAWAy6WFR`**, `ctest --output-on-failure -L regression -R MPI_SERIAL`
  at commit `64d1648` before any deletion: **all six** rank counts fail, not
  np=1-2 as the document says, and a **seventh** test fails —
  `SolveFusedM2L.FP32_smokeTest`, at np 2-6, passing only at np=1 (`max_grad_rel`
  0.277 at np=2 rising to 0.339 at np=3 against a 5e-2 budget). The six
  `MultiSolve` failures are the documented ones and np=1 reproduces the
  document's digits exactly (`3.485035469067542e-07`,
  `6.8419528791564039e-07`, `9.1947965989306709e-06`). The document's
  "six pre-existing failures" phrasing is what an exit criterion is compared
  against, so **a later task using it verbatim will mis-attribute
  `FP32_smokeTest` to itself.** It is pre-existing: it is in the baseline.
- **The np=3 hang is intermittent, and that cost this task its np=3-6
  comparison.** The baseline ran the entire gate at all six rank counts in 62 s
  with no hang at all — the first time this has been observed. The
  post-deletion run (**`f3XUPJqSuqdh`**, same script, same wall, same binary
  path) hung at np=3 for over 10 minutes and was cancelled. Same command, same
  checkout, opposite behaviour, which points at the partitioner
  non-determinism rather than at anything T2 touched. Anyone budgeting wall for
  this gate should assume the hang, not the 62 s.
- **The np 1-2 overlap is identical, which is the real evidence.** Baseline and
  post-deletion agree on the failure *set* at np=1 and np=2 and on **every
  reported error value to the last digit**, including the np=2 values that
  depend on the partition and the FP32 gradient `0.27730911646389911`. A
  deletion that changed arithmetic could not do that.
- **Two `unit` test targets do not compile, and neither is T2's doing.** A
  `make -j 4 -k` over the whole tree fails exactly two targets at every backend:
  `Canopy_Test_LaplaceKernel_*` (35 errors, "no matching function" for
  `p2m_contribution`, `m2m_translate`, `m2l_translate`, `l2l_translate`,
  `l2p_evaluate`) and `Canopy_Test_P2P_*` (3 errors, `tstP2P.hpp:449` passes
  `std::array<double,3>` where `TreeBuilder`'s constructor
  (`Canopy_TreeBuilder.hpp:164-166`) takes `std::array<double,6>`). Both files
  are unmodified by T2, the `TreeBuilder` signature drift traces to commit
  `8b0298e` "Refactor Solver constructor", and stashing T2's diff and rebuilding
  `Canopy_Test_LaplaceKernel_SERIAL` at `64d1648` produces the same 14 errors.
  Neither is in the `regression` gate, so neither blocks the ship gate — but
  `ctest -L unit` cannot be run as the document's diagnostic layer until they
  are fixed, and **`tstLaplaceKernel.hpp` is where T3 has to add its
  per-operator tests.**
- **Formatting.** `Canopy_DownwardSweep.hpp` is 365 lines from
  `.clang-format`-clean after the deletion (431 before, the drop being deleted
  non-clean lines), `Canopy_UpwardSweep.hpp` 31 and `Canopy_LaplaceKernel.hpp`
  63. `clangformat.sh` was not run, per the task's instruction; the one line T2
  inserted is format-clean on its own.

### Repository state left behind

- `src/Canopy_UpwardSweep.hpp`, `src/Canopy_DownwardSweep.hpp`,
  `src/Canopy_LaplaceKernel.hpp` — six members deleted, one comment reworded.
  Pure deletion otherwise; no reformatting, no renaming, no behaviour change.
- `tasks/abstract-solver-backend.md` — T2 marked **DONE** with a `Met.`
  paragraph; the Current-state opener now states what T1 and T2 actually built;
  the line-number paragraph replaced with the measured multi-band offsets; the
  group (a) heading marked done so a later session does not re-run T2.
- Logs kept: `canopy-ctest-minset-dev.f3XUAWAy6WFR.log` (baseline),
  `canopy-laplace-solve.f3XUMSqyY4qV.log` (gate),
  `canopy-ctest-minset-dev.f3XUPJqSuqdh.log` (partial post-deletion).
- Out of scope and untouched: the pre-existing regression failures, the np=3
  hang, the partitioner non-determinism, the two broken `unit` targets, and
  `README.md` "Known Issues".

**Affects:** **T3** — three things. Its exit criterion must not be written
against "the six pre-existing `MultiSolve` failures": the real baseline is seven
tests at the rank counts measured above, and `FP32_smokeTest` will otherwise be
charged to T3. Its per-operator tests belong in `tests/tstLaplaceKernel.hpp`,
**which does not currently compile** for reasons predating T2, so T3 must budget
for repairing that target before it can add anything to it. And
`m2l_apply_operator`, which T3 grows into `m2l_core`, was left alone by T2 as
the document's [Deliberate deviations](#deliberate-deviations) directs. **T4**,
**T7**, **T8**, **T10** — same baseline correction applies to every exit
criterion phrased as "exactly the six pre-existing failures". **Every task whose
gate is the regression suite** — budget for the np=3 hang; it is intermittent,
it consumed this task's np=3-6 comparison, and 15 minutes of wall is not enough
when it fires. The Laplace-solve gate, by contrast, completed in 41 s and is the
cheaper and sharper instrument: it caught nothing here precisely because it
reproduces T1's numbers exactly. **Whoever fixes the partitioner** (the task the
first T1 section called for before T3) — the baseline's clean 62 s run at all
six rank counts is a data point that the hang is not deterministic in the tree
shape alone.

## T3 — M2L is three kernel-owned stages

T3 is **DONE**, and the load-bearing result is negative in the way the document
wanted: **R1 did not fire.** The solid-harmonic M2L moved out of the sweep and
into the basis with identical bit patterns on all four artifacts at np 1-2, and
every cross-rank and direct-sum deviation at np 1-6 reproduces T1's pinned table
to all 17 digits. The design's three-stage decision survives, and the fallback to
the narrow abstraction is not needed. Commit `49a88de`; gate run flux job
**`f3XWVoKhkc5u`**.

### The four contract members as actually written

All in `src/Canopy_LaplaceKernel.hpp`, all `KOKKOS_INLINE_FUNCTION static`,
plus two typedefs. The document specified the parameter *lists*; the types below
are what they had to become.

```cpp
template <class MemorySpace>
using m2l_operators_type =
    Kokkos::View<complex_type***, Kokkos::LayoutLeft, MemorySpace>;

template <class ScratchSpace>
using m2l_accumulator_type =
    Kokkos::View<scalar_type*, ScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

static constexpr std::size_t m2l_scratch_bytes( int n_comps );

template <class TeamMember, class MView, class OpsType, class ScratchView>
static void m2l_pre_cell( const TeamMember&, const MView& M_full,
                          int source_cell, const OpsType& ops,
                          const ScratchView& scratch );

template <class TeamMember, class MView, class OpsType, class ScratchView>
static void m2l_core( const TeamMember&, const MView& M_full,
                      int source_cell, const OpsType& ops, int op_idx,
                      const ScratchView& scratch );

template <class TeamMember, class ScratchView, class LView, class OpsType>
static void m2l_post_cell( const TeamMember&, const ScratchView& scratch,
                           const LView& L_out, int target_cell,
                           const OpsType& ops );
```

**`m2l_operators_type` had to become an alias *template*.** The document names it
as a plain typedef, but `LaplaceKernel<Scalar, P, NComps>` carries no memory
space — every one of its methods is templated on the view type instead — while
`_m2l_op_table` is declared in `memory_space`. The sweep therefore spells it
`typename KernelType::template m2l_operators_type<memory_space>` and re-exports
that as its own `m2l_operators_type`; the existing public
`m2l_op_table_view_type` (which `tests/tstLaplaceSolve.hpp:689` `static_assert`s
on for LayoutLeft) is now an alias of it, so the test compiles unchanged and
still asserts the layout the committed hashes assume. **T9 should keep the alias
template, not flatten it**: a basis that builds its operators in a different
memory space than the sweep's would be the only reason to change it, and nothing
here needs that.

**`m2l_scratch_bytes` is `constexpr`, and that is load-bearing for R3.** The
sweep calls it as `KernelType::m2l_scratch_bytes( NComps )` and assigns the
result to a `constexpr size_t`, so the scratch size, the `TeamVectorRange` bound
and every extent inside the stages stay compile-time constants. `n_comps` is
formally a parameter but is only ever passed the sweep's `NComps`, which is
`KernelType::num_components`; inside the stages the extents come from the basis's
own `num_coeffs_per_cell * NComps` rather than from the argument, so nothing in
the moved arithmetic can be turned into a runtime value by a caller.

### How the scratch bytes are sized and viewed

`m2l_scratch_bytes( n ) = 2 * num_coeffs_per_cell * n * sizeof( scalar_type )` —
896 bytes at $P=6$, `NComps = 1`, `double`.

The sweep allocates it as **raw bytes**:

```cpp
using ScratchBytes = Kokkos::View<char*, scratch_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
policy.set_scratch_size( 0, Kokkos::PerTeam(
    ScratchBytes::shmem_size( scratch_bytes ) ) );
...
ScratchBytes scratch( team.team_scratch( 0 ), scratch_bytes );
```

and the basis re-views the same storage as the two `scalar_type` arrays the old
sweep held directly, at offsets 0 and `n_acc`:

```cpp
scalar_type* acc_base = reinterpret_cast<scalar_type*>( scratch.data() );
acc_type team_acc_re( acc_base, n_acc );
acc_type team_acc_im( acc_base + n_acc, n_acc );
```

This is the part the task flagged as most likely to break the gate, and it did
not: the real/imag **split** is preserved, so each accumulator update touches 8
bytes rather than 16, and the summation is the same sequence of `double`
additions it was before. The rationale comment moved with it onto
`m2l_scratch_bytes`, which now also says explicitly that collapsing the two
arrays into one `complex_type` array is mathematically identical and *bitwise
different* and so is not an available simplification. A single
`Kokkos::View<char*>` allocation replaces the two `ScratchReal::shmem_size` ones;
`ScratchMemorySpace::get_shmem` returns an 8-byte-aligned pointer, which is what
makes the `reinterpret_cast` to `scalar_type*` well-defined.

**The zero-fill stayed in the sweep, as the task directed, and therefore became a
byte fill** — the sweep no longer knows the layout, so it fills raw storage and
relies on IEEE-754 giving every floating-point accumulator an all-zero-bytes
representation. The contract is documented on `run_m2l_fused`: a basis whose
accumulator identity element is not all-zero bytes must establish it itself.

### Where `m2l_pre_cell` is called, and what that placement is worth

Once per source cell in the target team's CSR slice, immediately before
`m2l_core` for that pair. Within a team that is genuinely once per source cell —
a target's CSR slice holds distinct sources — but *across* teams the same source
cell is re-visited by every target that sees it. So the hook exists and is
correctly scoped for a basis that needs per-source state in team scratch, but it
does **not** yet deliver the flop advantage the design's
$U_\ell(\sum C_{\rm key}(V_\ell^\top M^B))$ form is after, which needs
$V_\ell^\top M^B$ computed once per source cell *globally* and stored somewhere
that outlives the team. That storage does not exist: the signature the document
specifies hands `m2l_pre_cell` the team scratch, which is per-target by
construction. **This is the genuinely new structural element the design predicted
would "fit nowhere", and it is only half-placed.** Whoever builds the first basis
that needs it (T6, and T8 if it assumes the compressed form's cost model) must
either add a per-source-cell buffer outside the team loop or accept
recomputation per target.

### Where the fused loop's arithmetic and `m2l_apply_operator`'s differed

Two differences, and per the task the fused loop won both:

- **Source access.** `m2l_apply_operator` read the multipole through
  `get_coeff_3d( M_full, source_cell, n, m, c )`; the fused loop expanded the
  conjugate symmetry inline as `storage_idx = n*(n+1)/2 + abs_m` followed by
  `complex_type( stored.real(), -stored.imag() )` for $m<0$. These are
  arithmetically the same here — `get_coeff_3d`'s two out-of-range guards cannot
  fire for $0 \le n \le P$, $|m| \le n$, `coeff_index(n, abs_m)` *is*
  `n*(n+1)/2 + abs_m` (`Canopy_SphericalCoefficients.hpp:59`), and its $m<0$
  branch is the same conjugation. Kept the inline form anyway: identical
  arithmetic is not the same claim as identical code generation, and the gate is
  bitwise.
- **Loop nesting.** `m2l_apply_operator` nested $(n, m, c)$ with a
  `complex_type accum[NComps]` array; the fused loop nested $(c, n, m)$ with one
  `complex_type acc(0,0)` scalar. The per-component addition *sequence* is the
  same in both, so this was a free choice on paper — but only the fused nesting
  had been compiled and measured against the reference data, so it is what
  `m2l_core` carries.

The function was grown, not deleted and rewritten: it keeps its identity, its
`KOKKOS_INLINE_FUNCTION static` form and its documentation lineage, and the
comment now records both deviations above so the next reader does not "restore"
`get_coeff_3d` and silently break the gate.

### R3 — measured, and it fired

Not a correctness gate and no exit criterion depends on it, but the answer is not
"no change". Method: a **second** build tree, `build-tuolumne-prof/`, configured
from the same `run_cmake_tuolumne.sh` with `Canopy_ENABLE_PROFILING=ON` and
`Canopy_PROFILING_LEVEL=2` — **the committed `build-tuolumne/` has profiling
`OFF`**, contrary to what the task prompt assumed, and reconfiguring it would
have put the bitwise gate and the timing measurement in different build
configurations across the before/after pair. `scripts/tuolumne/run_laplace_solve_profile.flux`
runs `Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1` there under `ctest -V`. The
figure is `M2L kernel (all depths)` from the `DownwardSweep::execute()` table,
summed over the 24 solves one np=1 invocation performs.

| Build | flux job | M2L kernel (24 solves) | Downward sweep total | Total solve |
| --- | --- | --- | --- | --- |
| before (unmodified) | `f3XW3XTKcr8o` | **0.053** | 1.169 | 1.348 |
| before (repeat) | `f3XW5NgdA1y9` | **0.055** | 1.185 | 1.365 |
| after (T3 as committed) | `f3XWBFwkZb6f` | **0.065** | 1.198 | 1.380 |
| after (repeat) | `f3XWDkueaXqq` | **0.063** | 1.192 | 1.367 |
| after (final binary) | `f3XWWdYC7twR` | **0.068** | 1.210 | 1.390 |
| variant: word-granularity zero-fill | `f3XWNdSA5rpf` | 0.062 | 1.196 | 1.374 |
| variant: raw pointers, no accumulator views | `f3XWQMddcbRH` | 0.067 | 1.211 | 1.390 |

**The two clusters do not overlap**: 0.053-0.055 before, 0.062-0.068 after. That
is roughly **+18% on the M2L kernel**, which is +1.5% on the downward sweep and
+0.5% on `solve()`. The per-sample timer prints only three decimals on values of
0.002-0.004 s, so no single sample is meaningful; the signal is the summed total
and the histogram shift (before: 20x 2 ms / 3x 3 ms / 1x 4 ms; after: 10 / 8 / 6).

**Two candidate causes were tested and both were excluded.** Making the sweep's
zero-fill word-granular instead of byte-granular (0.062) and replacing the
basis's two unmanaged accumulator views with raw `scalar_type*` (0.067) both land
inside the post-move cluster. Both experiments were reverted; the committed code
is the byte fill and the unmanaged views, which is also the form the design
document describes. **The cost is therefore intrinsic to putting the contraction
behind the basis interface**, not to either micro-detail, and finding it would
need a sharper instrument than a 1 ms timer — a Kokkos kernel-level profile or a
raw loop-count harness. Recorded, not fixed: it is 0.5% of a solve, and chasing
it inside T3 would have meant expanding the one diff this task exists to keep
attributable.

### What only running revealed

- **`build-tuolumne/` is configured with `Canopy_ENABLE_PROFILING=OFF`.** The
  task prompt states it is ON. It is not, and no prior gate log contains a single
  `[Canopy Diagnostics]` line. Anyone else asked to measure R3 or R4 needs the
  second build tree (or to reconfigure and accept the confound); the script and
  the reasoning are committed.
- **The partitioner's non-determinism now shows up *within a single job* at
  np=5, and it is not T3's.** At np=5 the three test bodies run three separate
  solves. In every T1/T2-era log all three drew the same cut
  (`n_unique_ops` 180/168/147/217/187). In this session's runs they split into
  two cuts — one solve draws 170 on rank 0 and 177 on rank 1, the others draw the
  old set — which moves the np=5 direct-sum deviation in its 9th significant
  digit (3.209361028925181e-07 vs 3.2093610310582619e-07), still 3x under
  tolerance. **Attributed by a control run, not by argument**: the change was
  stashed, `build-tuolumne/` rebuilt from unmodified `HEAD`, and the gate re-run
  (flux job **`f3XWSVHd6FVh`**) — unmodified code reproduces the same 170/177
  split. So this is `TreePartitioner::partition_leaves`' documented multijagged
  non-determinism (`src/Canopy_TreePartitioner.hpp:417-419`) presenting between
  invocations in one process, and something about this session's nodes rather
  than the source changed how it lands. The final gate run happened to draw the
  T1 cut in all three solves, which is why the **Met.** paragraph can claim all
  17 digits at all six rank counts. **A later task must not read a np=5
  direct-sum difference in the 9th digit as its own doing**, and the np 1-2
  bitwise half of the gate is unaffected because the cut over one or two parts is
  reproducible.
- **The gate ran clean six times in a row.** No np=3 hang appeared in any of the
  four full-gate jobs this session (41-46 s each), consistent with T2's note that
  it is intermittent.
- **Formatting.** `clang-format` was run with explicit `--lines=` ranges covering
  only the touched hunks, per the task's instruction not to run `clangformat.sh`
  over these headers. It changed exactly one line — reflowing the `ScratchBytes`
  alias onto two lines.

### Repository state left behind

- `src/Canopy_LaplaceKernel.hpp` — `m2l_operators_type`,
  `m2l_accumulator_type`, `m2l_scratch_bytes`, `m2l_pre_cell`, `m2l_core`
  (grown from `m2l_apply_operator`), `m2l_post_cell`; `#include <cstddef>`
  added for `std::size_t`.
- `src/Canopy_DownwardSweep.hpp` — `run_m2l_fused`'s body is now traversal,
  zero-fill and three stage calls; `_m2l_op_table` and the two other spellings of
  its type route through `m2l_operators_type`; `m2l_op_table_view_type` is an
  alias of it. No `complex_type` remains anywhere in the fused kernel's body.
- `scripts/tuolumne/run_laplace_solve_profile.flux` — new, the R3 instrument.
  `build-tuolumne-prof/` is a build tree, not committed.
- Logs kept: `canopy-laplace-solve.f3XWVoKhkc5u.log` (the gate),
  `canopy-laplace-solve.f3XWSVHd6FVh.log` (the unmodified control),
  and the seven `canopy-laplace-solve-prof.*.log` R3 runs.
- **No test was added**, per the task's scope: `tests/tstLaplaceKernel.hpp` still
  does not compile and was not touched. `ctest -L regression`,
  `Canopy_Test_MultiSolve_*`, the np=3 hang, the partitioner, R6's premise and
  `README.md` were all left alone as directed.

**Affects:** **T4** — `coeff_type`'s shape is now constrained from two sides.
The operator table's element type is fixed by
`m2l_operators_type<MemorySpace> = View<complex_type***, LayoutLeft>` and the
`LayoutLeft` `static_assert` in `tests/tstLaplaceSolve.hpp:689` is now written
against that alias, so a T4 that generalizes the coefficient element type must
carry the operator table's element type with it or the two will silently
disagree. The scratch, by contrast, imposes **nothing**: it is raw bytes sized by
a basis-supplied `constexpr`, so a basis whose coefficients are real, or blocked,
or of a different width needs no sweep change at all. T4 should also re-run the
R3 measurement with `scripts/tuolumne/run_laplace_solve_profile.flux` against
this section's table rather than against unmodified code, since T3 has already
moved the baseline. **T7** and **T8** — the `op_idx` boundary is now the *only*
thing the sweep says about an operator: `csr_op_idx` carries an `int`, `-1` still
means "fall back", and `m2l_core` is the sole reader. T7's depth-carrying key
changes what `op_idx` *indexes* and needs no change to the sweep's fused kernel;
T8's overflow policy changes which pairs get `-1` and likewise touches only the
builder. Both are now genuinely local edits. **T9** — the operator builder it
replaces is the block at `src/Canopy_DownwardSweep.hpp:1121` that allocates
`m2l_operators_type op_table( ..., Nt, Ns, n_unique_ops )` and fills it with
`KernelType::m2l_build_operator`; that is the last place in the sweep that
commits to the operator table's *shape*, and the typedef it now uses is the
handle T9 should move behind a basis-owned `build_m2l_operators`. **T6** — see
the `m2l_pre_cell` placement note above: the hook exists, but a basis needing
once-per-source-cell work that outlives the team has nowhere to put it yet.

## T4 — coefficient storage and MPI packing are basis-agnostic

T4 is **DONE** and, like T3, the load-bearing result is negative: the
solid-harmonic path came through with **identical bit patterns** on all four
artifacts at np 1-2, and all 22 cross-rank and direct-sum deviations at np 1-6
reproduce T1's pinned table to all 17 printed digits. Not one figure moved — not
even the 9th-digit np 3-6 wobble the task warned about — so no control run from
unmodified `HEAD` was needed. Gate run flux job **`f3XfAA99uRAw`**; R3 instrument
flux job **`f3XfAAGBc1fm`**. Worked at `HEAD` = `ff737a2`.

### Decisions taken (given, not chosen here)

- **The operator table's element type follows `coeff_type`.** The basis alias is
  now `Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>`. For this
  basis `coeff_type` *is* `Kokkos::complex<Scalar>`, so no bits moved; the point
  is that the operator-table element type and the coefficient element type can no
  longer be set independently and silently disagree, since `m2l_core` contracts
  one against the other.
- **`CoalescedExchangeBuffers`' `ComplexType` template parameter is renamed
  `CoeffType`**, and nothing else in `Canopy_MpiCoalescedExchange.hpp` beyond
  what T4's `Do` list names. In particular the local variable names
  `per_cell_complex`, `total_complex`, `umcplx_view` and the function-local
  `scalar_type` alias were **left alone** in all three files, deliberately: they
  are local and renaming them would have buried the diff the gate has to
  attribute. A later task that wants them accurate can rename them freely —
  nothing reads them across a boundary.
- **`component_scalar_type` is `Scalar`, not `double`.** `mpi_scalar` is chosen
  from `sizeof(component_scalar_type)` at three sites; hardcoding `double` would
  have sent `MPI_DOUBLE` for the basis's live `float` payloads. The Conventions
  table's `Scalar = double` rule binds the *new* bases only.

### The three traits as actually written

On `LaplaceKernel`, `src/Canopy_LaplaceKernel.hpp`:

```cpp
using coeff_type = Kokkos::complex<Scalar>;
using component_scalar_type = Scalar;
static constexpr int scalars_per_coeff = 2;

static_assert( sizeof( coeff_type ) ==
                   scalars_per_coeff * sizeof( component_scalar_type ), ... );

// basis-private spelling, retained
using complex_type = coeff_type;
```

**`complex_type` was kept as a basis-private alias of `coeff_type` rather than
deleted.** It has 50 uses inside `Canopy_LaplaceKernel.hpp` and most of them are
not coefficients at all — the `Ynm` tables, the $i^k$ tables, the conjugation
under $m \to -m$, the `i_power` return type. Renaming those to `coeff_type` would
have made them read as storage elements, which they are not. The leak the task
exists to close is shared code naming `Kokkos::complex`, and that is closed:
`grep -n complex_type` returns **zero** hits in `Canopy_UpwardSweep.hpp`,
`Canopy_DownwardSweep.hpp` and `Canopy_MpiCoalescedExchange.hpp`. A basis author
copying `LaplaceKernel` as a template will see `coeff_type` in the contract block
at the top and `complex_type` only inside the solid-harmonic operators, with a
comment saying which is which.

**The zero element is `coeff_type()`**, value-initialization, replacing
`complex_type( 0.0, 0.0 )` at three sites (`_multipoles` deep_copy, `_locals`
deep_copy, `_shared_snapshot_buf.assign`). This is bit-identical here and not
by luck: Kokkos declares `RealType re_{}; RealType im_{};` as NSDMIs
(`Kokkos_Complex.hpp:40-41`), so `Kokkos::complex<double>()` is `(+0.0, +0.0)`,
the same bits `complex_type( 0.0, 0.0 )` produced. It is documented on the
contract block as the identity element a basis must supply, which is the same
convention T3 established for the raw-byte scratch (all-zero bytes).

### The one thing the design did not anticipate: `coalesced_view_exchange` has no basis

T4's `Do` item 3 says to replace `typename complex_type::value_type`
(`Canopy_MpiCoalescedExchange.hpp:72`) "with the trait". **There is no trait
reachable from there.** `coalesced_view_exchange` is a free function template
whose only type inputs are `CoeffView` and `ExchBuffers`; it is handed a
`Kokkos::View` and never a `KernelType`, and the design also requires it keep its
signature. `ExchBuffers` is `CoalescedExchangeBuffers<coeff_type, memory_space>`
and carries no more information than the view does.

Resolved with a small traits class in the same header, `Canopy::detail`:

```cpp
template <class CoeffType>
struct coeff_traits            // primary: a real-coefficient basis
{
    static_assert( std::is_floating_point<CoeffType>::value, ... );
    using component_scalar_type = CoeffType;
    static constexpr int scalars_per_coeff = 1;
};

template <class RealType>
struct coeff_traits<Kokkos::complex<RealType>>
{
    using component_scalar_type = RealType;
    static constexpr int scalars_per_coeff = 2;
};
```

and the function body reads `scalar_type` and `scalars_per_coeff` from it. The
primary template is what makes the (c) failure mode go away for real coefficients
— the design's note that "a real `double` coefficient has no `::value_type`, so
this function template does not compile" is now false by construction, and a
`coeff_type` that is neither a real scalar nor `Kokkos::complex` gets a named
`static_assert` telling the author to specialize rather than a template error
inside MPI argument deduction.

**This is a second source of truth, and it is closed with a `static_assert` in
each sweep.** The sweeps are the only place a basis and `coalesced_view_exchange`
meet, so each of `UpwardSweep` and `DownwardSweep` now asserts, next to its
`coeff_type` typedef, that
`detail::coeff_traits<coeff_type>::component_scalar_type` and
`::scalars_per_coeff` equal the basis's own. A basis that declares
`scalars_per_coeff = 1` for a `Kokkos::complex` coefficient — the exact way this
duplication could rot — fails to compile with a message naming both sources.
Each sweep also repeats the `sizeof` assert from the exit criterion, so a
padding-bearing `coeff_type` is caught at the sweep that would mis-size the
Allreduce as well as at the basis.

### Signatures and declarations changed

- `src/Canopy_LaplaceKernel.hpp` — `coeff_type`, `component_scalar_type`,
  `scalars_per_coeff` and the `sizeof` `static_assert` added;
  `m2l_operators_type<MemorySpace>`'s element type is `coeff_type`;
  `complex_type` demoted to an alias of `coeff_type` with a comment scoping it
  to the basis.
- `src/Canopy_MpiCoalescedExchange.hpp` — `detail::coeff_traits` added (+
  `#include <type_traits>`); `CoalescedExchangeBuffers<ComplexType, …>` →
  `<CoeffType, …>`; the `::value_type` chain replaced by the traits;
  `per_cell_real = scalars_per_coeff * per_cell_complex`; `umcplx_view`'s element
  type is `coeff_type`. `coalesced_view_exchange`'s signature is unchanged, as
  the design requires.
- `src/Canopy_UpwardSweep.hpp` — `complex_type` typedef replaced by the three
  traits plus two `static_assert`s (+ `#include <type_traits>`);
  `coeff_view_type`, `_m2m_exch_bufs`, the two M2M Allreduce staging views and
  the `_multipoles` zero-fill route through `coeff_type`; the M2M Allreduce casts
  to `component_scalar_type*` with count `scalars_per_coeff * total_complex` and
  picks `mpi_scalar` from `sizeof(component_scalar_type)`.
- `src/Canopy_DownwardSweep.hpp` — the same typedef and `static_assert` block;
  `coeff_view_type`, `_exch_bufs`, `_shared_snapshot_buf`, both shared-cell
  Allreduce host buffers and the `_locals` zero-fill route through `coeff_type`;
  the shared-cell Allreduce casts to `component_scalar_type*` with count
  `scalars_per_coeff * total_complex`.
- `tests/tstLaplaceSolve.hpp` — **not touched**, and it compiled unchanged. That
  is part of what proves the rename changed nothing: its two `static_assert`s
  still reach `LayoutRight` through `DownwardSweep::coeff_view_type` and
  `LayoutLeft` through `m2l_op_table_view_type`, which is an alias of
  `m2l_operators_type`, whose element type this task changed.

### Gate measurements

Flux job **`f3XfAA99uRAw`**, tuolumne2149, Cray clang 20.0.0, spack env
`tuolumne_trilinos`, `RelWithDebInfo`, Kokkos SERIAL, `build-tuolumne/`
(`Canopy_ENABLE_PROFILING=OFF`), 40.63 s of CTest wall time. `100% tests passed, 0 tests failed out of
6`.

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 3.2093610331931809e-07 | 4.2399302231264458e-08 |
| 2 | 4.1994107222659022e-13 | 2.1570013757642702e-12 | 3.2093610363952985e-07 | 4.239936667274564e-08 |
| 3 | 8.0211305411572796e-13 | 4.0353546216363242e-12 | 3.2093610299898352e-07 | 4.2399380705248681e-08 |
| 4 | 1.114294857300434e-12 | **5.5987399483706545e-12** | 3.2093610299888352e-07 | 4.2399368624331468e-08 |
| 5 | 9.5809726336249496e-14 | 6.4438389009577268e-13 | 3.209361028925181e-07 | 4.2399357908696204e-08 |
| 6 | 5.688835866646797e-13 | 2.8631357246763719e-12 | 3.2093610299905848e-07 | 4.2399381662523099e-08 |

**Every cell is character-for-character T1's table.** `fallback_pairs = 0` and
`locals_ext = (103,28,1)`, `a_extent = 169`, `initial_hash =
0xb6ad437608ad69b7` at every rank and rank count. `bitForBitArtifacts` ran and
passed at np=1 rank 0 and np=2 ranks 0 and 1 — so `locals()`, the operator table,
the $A_{n,m}$ table and the realized key list are byte-identical to
`tests/data/laplace_solve_P6.txt`, and no hashes needed recording — and reports
`SKIPPED` at np 3-6 by design.

`n_unique_ops` per rank at the last solve: np=1 → 686; np=2 → 368, 386;
np=3 → 273, 204, 329; np=4 → 264, 128, 234, 194; np=6 → 174, 156, 111, 160, 116,
175 — all identical to T1. **At np=5 the multijagged split T3 recorded reappeared
and cost nothing measurable:** the `crossRankAgreement` solve drew 170 on rank 0
and 177 on rank 1 while the `matchesDirectSum` solve drew T1's 180/168, yet both
printed deviations still match T1 to all 17 digits. So the two-cut split is
confirmed as still live on this session's nodes and is confirmed as *not*
sufficient to move any printed figure at this configuration — which is a slightly
stronger statement than T3 could make, since T3's final gate run happened to draw
the T1 cut in all three solves.

### R3 — measured against T3's baseline, and it did not fire again

Flux job **`f3XfAAGBc1fm`**, `build-tuolumne-prof/`
(`Canopy_ENABLE_PROFILING=ON`, `Canopy_PROFILING_LEVEL=2`), np=1 only,
`M2L kernel (all depths)` from the `DownwardSweep::execute()` table summed over
the 24 solves of one invocation. Neither build tree was reconfigured.

| Build | flux job | M2L kernel (24 solves) | Downward sweep total | Total solve |
| --- | --- | --- | --- | --- |
| before T3 (unmodified) | `f3XW3XTKcr8o` / `f3XW5NgdA1y9` | 0.053 / 0.055 | 1.169 / 1.185 | 1.348 / 1.365 |
| after T3 (the T4 baseline) | `f3XWBFwkZb6f` … `f3XWWdYC7twR` | **0.062 - 0.068** | 1.192 - 1.210 | 1.367 - 1.390 |
| after T4 | `f3XfAA99uRAw`-era binary, `f3XfAAGBc1fm` | **0.065** | 1.205 | 1.385 |

0.065 sits inside T3's post-move cluster on all three figures. The per-sample
histogram is 8x 2 ms / 15x 3 ms / 1x 4 ms, which is a different shape from T3's
post-move 10/8/6 at the same total — fewer 4 ms samples, more 3 ms ones — and
clearly distinct from the pre-move 20/3/1. Since the timer prints three decimals
on values of 0.002-0.004 s, only the summed total carries signal and the shape
difference at a fixed sum should not be read as anything. **T4 adds nothing
measurable**, which is
what a compile-time trait indirection should cost: `scalars_per_coeff` is
`static constexpr int`, `coeff_type` is a typedef, and nothing in the fused M2L
kernel changed at all. One sample only — no repeat was run, because R3 is not a
correctness gate and the figure is already inside a cluster established by five
prior samples.

### What only running revealed

- **Nothing failed on the first build.** No compile error, no gate failure, no
  bitwise difference. The `static_assert` cross-check between the basis traits
  and `detail::coeff_traits` was written before the first build and passed on it.
- **`clang-format` changed nothing.** Run with explicit `--lines=` ranges
  covering only the touched hunks in all four headers, per the task's
  instruction not to run `clangformat.sh` over them. The diff was already in the
  repo style — the `std::is_same<...>` line breaks in the two sweep
  `static_assert`s were the only thing at risk and it had already chosen the
  format `clang-format` wanted.
- **The four other SERIAL targets and both examples still compile**, verified
  because these are shared headers: `Canopy_Test_UpwardSweep_MPI_SERIAL`,
  `Canopy_Test_DownwardSweep_MPI_SERIAL`, `Canopy_Test_MultiSolve_MPI_SERIAL`,
  `Canopy_Test_SingleSolve_MPI_SERIAL`, `example_fmm`, `gravity_solve` — all
  build and link clean. A repository-wide grep had already shown no user of
  `KernelType::complex_type` outside the four headers, and the builds confirm it.
  Not run: `Canopy_Test_LaplaceKernel_*` and `Canopy_Test_P2P_*`, which do not
  compile at `HEAD` and are out of scope.
- **`make -j 4` again, per T1's operational note.** No SIGKILL.
- **`run_cmake_tuolumne.sh` shows as modified in `git status` and it is not
  T4's.** The diff is whole-file line-ending churn plus a mode change to 755,
  present before this session started. Left alone; it is not committed with T4.
  `setup-repo.txt` is likewise a pre-existing untracked file.

### Repository state left behind

- Four headers modified, listed under "Signatures and declarations changed"
  above. No test added, per scope. `tests/tstLaplaceSolve.hpp`,
  `tests/data/laplace_solve_P6.txt` and `README.md` untouched.
- Logs kept: `canopy-laplace-solve.f3XfAA99uRAw.log` (the gate),
  `canopy-laplace-solve-prof.f3XfAAGBc1fm.log` (R3).
- Out of scope and untouched, as directed: `sets_per_component` and the
  shared-cell Allreduce shape (T10), the `m2l_pre_cell` per-source-cell storage
  gap (T6), `M2L_KEY_DD_MAX`'s `float` branch and the $A_{n,m}$ table's existence
  (group (d), T5/T7), R6's premise, `ctest -L regression` and its intermittent
  np=3 hang, and the partitioner's non-determinism. No `regression`-suite run was
  attempted.

**Affects:** **T6** — the conformance basis's traits are now fully determined and
need no new machinery: `coeff_type = double`, `component_scalar_type = double`,
`scalars_per_coeff = 1` matches `detail::coeff_traits`' primary template exactly,
so `coalesced_view_exchange`, both shared-cell Allreduces, both coefficient
views and the operator table all work for it with **zero** further sweep changes.
The only sweep-side obstacle left for T6 is the `m2l_pre_cell` placement gap T3
recorded and group (d)'s two harmonic-derived values (T5). If T6 instead wants a
blocked or mixed-width `coeff_type`, it must specialize
`Canopy::detail::coeff_traits` for it — the primary template's `static_assert`
says so by name. **T5** — `a_view_type` is still
`Kokkos::View<scalar_type*, memory_space>`, i.e. the $A_{n,m}$ table is keyed on
`KernelType::scalar_type` and **not** on `component_scalar_type`. T4 deliberately
did not touch it: it is group (d), and for this basis the two types are the same
`Scalar`. When T5 moves the table into the basis, it should decide which of the
two it is a table *of* rather than inheriting the ambiguity. **T10** — the two
loops it rewrites (`allreduce_shared_locals_at_depth`'s pack at
`src/Canopy_DownwardSweep.hpp:1825` and unpack at `:1848`, and the snapshot
assign at `:1773`) now index `std::vector<coeff_type>` with a `per_cell_complex`
stride that T4 left at `coeffs_per_cell * NComps`. The MPI count is
`scalars_per_coeff * total_complex`, so a `sets_per_component` change has exactly
one arithmetic site to update per buffer and the byte count follows the stride
automatically — but the *variable names* there still say `complex`, which will
read wrong after T10 and are free to rename. T10 also inherits T1's finding that
`crossRankAgreement` is a seven-orders-of-magnitude gate on precisely this
dataflow, now re-exercised clean at np 2-6. **T7**, **T8**, **T9** — unaffected;
T9 should note only that the operator table it moves behind
`build_m2l_operators` now has element type `coeff_type`, so the basis-owned
builder and the basis-owned coefficient type are already consistent by
construction.

## T5 — auxiliary tables are owned by the basis

T5 is **DONE** and, like T3 and T4, the load-bearing result is negative: the
$A_{n,m}$ table came out of the sweeps and into `LaplaceKernel` with **identical
bit patterns**, all 169 of them, at np 1-2. Gate run flux job
**`f3XfiYHXy4b1`**; R3 instrument flux job **`f3Xfk8H1YPNF`**. Worked at `HEAD` =
`d72c3c4` (T4 was already committed there — the handoff prompt's claim that T4
sat uncommitted in the working tree was stale, and nothing depended on it).

### The decision T5 owned: `scalar_type`, not `component_scalar_type`

T4 handed this choice forward explicitly, having left `a_view_type` on
`scalar_type` because for this basis the two are the same `Scalar` and the
question was group (d)'s, not (b)'s. **The table is of `scalar_type`.**

The reasoning, which is written out on the `aux_tables_type` declaration in
`src/Canopy_LaplaceKernel.hpp` and is the part worth inheriting:

- **`component_scalar_type` is a storage trait, and $A_{n,m}$ is not storage.**
  Its contract, as T4 wrote it, is "the real scalar MPI sees": it names what
  `coeff_type` decomposes into so the sweeps can `reinterpret_cast` a coefficient
  buffer and hand MPI a count. $A_{n,m}$ is never packed, never crosses a rank
  boundary, and is not a coefficient — it is a real constant multiplied into the
  translation arithmetic. Nothing about the MPI datatype selection has any claim
  on it.
- **`scalar_type` is what the four operators already read it into.** Every use
  site in the basis is `const Scalar A_jk = A_table( a_index( j, k ) );` and its
  siblings. Typing the table as anything else would introduce a conversion at
  every read that the code does not currently perform.
- **The two types can come apart, and when they do, `component_scalar_type`
  would be the wrong one.** A basis with a blocked or mixed-precision
  `coeff_type` — the case T4's `detail::coeff_traits` primary template invites a
  specialization for — would have a packed component narrower than its
  arithmetic type. Keying this table on that would silently demote a table of
  normalization constants along with the storage, which is a decision about
  bandwidth leaking into a decision about accuracy.

No bits moved either way here, which is precisely why it was cheap to settle
now: `scalar_type` and `component_scalar_type` are both `Scalar` on
`LaplaceKernel`, so the gate cannot distinguish the two choices and the argument
had to be made on meaning rather than on measurement.

### The contract as actually written

On `LaplaceKernel`, `src/Canopy_LaplaceKernel.hpp`, immediately after
`m2l_operators_type` and in the same memory-space-parameterized shape:

```cpp
template <class MemorySpace>
struct aux_tables_type
{
    Kokkos::View<scalar_type*, MemorySpace> A_table;
};

template <class MemorySpace>
static aux_tables_type<MemorySpace> build_aux_tables( int order )
{
    aux_tables_type<MemorySpace> aux;
    aux.A_table =
        build_A_coefficients<scalar_type, MemorySpace>( 2 * order );
    return aux;
}
```

`aux_tables_type` is a **struct template, not an alias template**, unlike
`m2l_operators_type`. It has to be: a basis with two tables adds a second member
without any caller changing, and T6's `MonopoleBasis` supplies an empty struct.
The spelling at the sweeps is the same either way —
`typename KernelType::template aux_tables_type<memory_space>`.

**`build_aux_tables` takes the order and multiplies by two itself.** The `2 *`
and the sentence explaining it — M2L reaches degree $n+j$ with both $n$ and $j$
up to $P$ — moved out of `src/Canopy_UpwardSweep.hpp` and onto the basis
function, which is the substance of this task rather than a side effect of it:
the factor of two is a fact about the solid-harmonic translation theorems and
shared code had no business asserting it. The comment on `build_aux_tables` also
records why a short table is dangerous rather than loud — `m2l_build_operator`
`continue`s on a zero $A$ (`src/Canopy_LaplaceKernel.hpp:688-689` pre-T5), so
one degree short is a quietly wrong operator, not a fault.

**`build_aux_tables` is a plain static member, not `KOKKOS_INLINE_FUNCTION`**,
matching the Conventions table's rule for host-side construction: it allocates
and fills a `View`.

### The two memory spaces, and why the host one is built rather than mirrored

The device aux is built once in `UpwardSweep::setup()` and borrowed by
`DownwardSweep::setup()`, exactly as the bare view was. The host aux is built
fresh at the operator-table build:

```cpp
const auto h_aux =
    KernelType::template build_aux_tables<Kokkos::HostSpace>(
        KernelType::max_order );
```

replacing `create_mirror_view_and_copy( Kokkos::HostSpace{}, _A_table )`. The
sweep **cannot** mirror the device aux, because mirroring means naming members,
and the whole point of `aux_tables_type` is that shared code does not know what
is in it. Building is bit-identical rather than merely equal, for the reason the
design gave and which held: `build_A_coefficients` fills a host mirror from
`A_coeff<Scalar>(n, m)` — a pure function of $(n,m)$ — before deep-copying, so
the host build and a mirror of the device build are the same bytes by
construction, and on the SERIAL backend `memory_space` *is* `Kokkos::HostSpace`
so they are the same code path besides. The gate confirms it: the M2L operator
table's hash is unchanged at np 1-2, and that table is built entirely from
`h_aux`.

One cost worth naming: the host table is now rebuilt on every operator-table
build rather than mirrored. At $P=6$ that is 169 doubles and 169 calls to
`A_coeff`, inside a block that then fills a $28\times49\times n_{\rm ops}$
operator table — unmeasurable, and the profile run confirms the operator-table
build did not move. It is the right trade for not having shared code reach
inside the struct.

### Signatures and declarations changed

- `src/Canopy_LaplaceKernel.hpp` — `aux_tables_type<MemorySpace>` and
  `build_aux_tables<MemorySpace>(int order)` added. Four operators take
  `const AuxType& aux` where they took `const AType& A_table`:
  `m2m_translate`, `m2l_translate`, `m2l_build_operator`, `l2l_translate`. Each
  body opens with `const auto& A_table = aux.A_table;` so the translation
  arithmetic below is untouched — that is what makes the diff readable and what
  makes "no bits moved" checkable by eye as well as by the gate.
  `p2m_contribution`, `l2p_evaluate`, `m2l_pre_cell`, `m2l_core` and
  `m2l_post_cell` never took the table and are unchanged.
- `src/Canopy_UpwardSweep.hpp` — `a_view_type` replaced by `aux_tables_type`;
  `_A_table` → `_aux`; `A_table()` → `aux()`; the `2*P`
  `build_A_coefficients` call in `setup()` → `build_aux_tables<memory_space>(P)`;
  the M2M lambda capture `auto A_table = _A_table;` → `auto aux = _aux;` and the
  `m2m_translate` call updated.
- `src/Canopy_DownwardSweep.hpp` — the same alias, member and accessor changes;
  `_aux = upward_sweep.aux();` at `setup()`; the HostSpace aux at the
  operator-table build feeding `m2l_build_operator`; and both device lambda
  captures (`run_m2l_fallback_at_depth`, `run_l2l_at_depth`) with their
  `m2l_translate` / `l2l_translate` calls.
- `tests/tstLaplaceSolve.hpp` — **one line**, `collect_a_table`'s
  `ds.A_table()` → `ds.aux().A_table`. `tests/data/laplace_solve_P6.txt` was not
  regenerated; the four `"A_table()"` strings elsewhere in the file are failure
  messages and were left as they are.

`grep -n "_A_table" src/Canopy_UpwardSweep.hpp src/Canopy_DownwardSweep.hpp`
returns nothing, and neither sweep mentions $A_{n,m}$ at all now.

### Gate measurements

Flux job **`f3XfiYHXy4b1`**, tuolumne1020, Cray clang 20.0.0, spack env
`tuolumne_trilinos`, `RelWithDebInfo`, Kokkos SERIAL, `build-tuolumne/`
(`Canopy_ENABLE_PROFILING=OFF`), 39.99 s of CTest wall time.
`100% tests passed, 0 tests failed out of 6`.

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 3.2093610331931809e-07 | 4.2399302231264458e-08 |
| 2 | 4.1994107222659022e-13 | 2.1570013757642702e-12 | 3.2093610363952985e-07 | 4.239936667274564e-08 |
| 3 | 8.0211305411572796e-13 | 4.0353546216363242e-12 | 3.2093610299898352e-07 | **4.2399380705233977e-08** |
| 4 | 1.114294857300434e-12 | 5.5987399483706545e-12 | 3.2093610299888352e-07 | 4.2399368624331468e-08 |
| 5 | 9.5809726336249496e-14 | 6.4438389009577268e-13 | 3.209361028925181e-07 | 4.2399357908696204e-08 |
| 6 | 5.688835866646797e-13 | 2.8631357246763719e-12 | 3.2093610299905848e-07 | 4.2399381662523099e-08 |

**Twenty-one of the 22 cells are character-for-character T1's.** The bolded one
is not: T1 and T4 both printed 4.23993807052**48681**e-08 and this run printed
4.23993807052**33977**e-08 — twelve significant figures of agreement, a move in
the thirteenth, 22x under `LS_DIRECT_SUM_TOL`.

`fallback_pairs = 0`, `locals_ext = (103,28,1)`, `optab_ext = (28,49,n_ops)`,
`a_extent = 169` and `initial_hash = 0xb6ad437608ad69b7` at every rank and rank
count. `bitForBitArtifacts` ran and passed at np=1 rank 0 and np=2 ranks 0 and 1
— all four artifacts byte-identical to `tests/data/laplace_solve_P6.txt`, so no
hashes needed recording — and `SKIPPED` at np 3-6 by design. Per-test tallies
from the log: `bitForBitArtifacts` 3 OK / 36 SKIPPED, `crossRankAgreement` 20 OK
/ 2 SKIPPED, `matchesDirectSum` 21 OK.

`n_unique_ops` per rank at the last solve: np=1 → 686; np=2 → 368, 386;
np=4 → 264, 128, 234, 194; np=5 → 180, 168, 147, 217, 187; np=6 → 174, 156, 111,
160, 116, 175 — all identical to T1. **np=3 is where the multijagged split
landed this time**, and it is the whole explanation of the one moved figure.

### The np=3 wobble, attributed from the gate log alone

The design's procedure for a np 3-6 deviation that moves in its 9th or later
significant figure is to confirm it against a control run from unmodified `HEAD`
before recording it as a finding. **No control run was needed, because this run's
own log is the control.** `ctest -V` prints one `[laplace-solve] nprocs=3 rank=R`
line per test body, and np=3's three solves did not draw the same cut:

| solve | rank 0 | rank 1 | rank 2 | sum |
| --- | --- | --- | --- | --- |
| two of the three | 273 | 204 | 329 | 806 |
| the other one | 285 | 189 | 329 | 803 |

`(273, 204, 329)` is T1's np=3 set exactly. The second cut is a different
partition of the same tree, which is the same two-cut split T3 first saw and T4
recorded at np=5 — reproduced there by unmodified `HEAD` in flux job
`f3XWSVHd6FVh`. It has now been observed at np=3 as well, so it is not a property
of any particular rank count. This run it moved one direct-sum gradient in its
thirteenth significant figure; at np=5, where T4 saw it, it moved nothing
printable. Both observations are consistent with the documented statement and
neither is a finding of T5.

Two things worth keeping from this. First, **the split is visible directly in
the `ctest -V` output and does not need a separate run to diagnose** — comparing
the per-rank `n_unique_ops` lines across the three test bodies at one rank count
is enough, and is cheaper than a stash-and-rebuild control. Later tasks should
look there first. Second, **T4's "the split does not move any printed figure" is
now known to be configuration-dependent rather than general**: it did not move
one at np=5 and it does move one, in the thirteenth figure, at np=3.

### R3 — sampled, and it did not fire

Flux job **`f3Xfk8H1YPNF`**, `build-tuolumne-prof/`
(`Canopy_ENABLE_PROFILING=ON`, `Canopy_PROFILING_LEVEL=2`), np=1 only,
`M2L kernel (all depths)` from the `DownwardSweep::execute()` table summed over
the 24 solves of one invocation. Neither build tree was reconfigured.

| Build | flux job | M2L kernel (24 solves) | Downward sweep total | Total solve |
| --- | --- | --- | --- | --- |
| before T3 (unmodified) | `f3XW3XTKcr8o` / `f3XW5NgdA1y9` | 0.053 / 0.055 | 1.169 / 1.185 | 1.348 / 1.365 |
| after T3 | `f3XWBFwkZb6f` … `f3XWWdYC7twR` | 0.062 - 0.068 | 1.192 - 1.210 | 1.367 - 1.390 |
| after T4 | `f3XfAAGBc1fm` | 0.065 | 1.205 | 1.385 |
| after T5 | `f3Xfk8H1YPNF` | **0.070** | **1.209** | **1.390** |

0.070 is 2 ms above the top of T3's five-sample post-move cluster, over 24
samples that the timer prints to three decimals on values of 0.002-0.004 s — one
tick per sample of quantization. The per-sample histogram is 6x 2 ms / 14x 3 ms /
4x 4 ms, against T4's 8/15/1 and T3's post-move 10/8/6. The two figures that are
~20x larger and therefore far better resolved — downward sweep 1.209 and total
solve 1.390 — both sit inside T3's post-move ranges, at their tops.

**The structural argument says T5 cannot have cost anything here, and it is
stronger than the sample.** `TIMER_M2L_KERNEL` scopes `run_m2l_all` and
`run_m2l_at_depth`. The fused kernel inside them goes through `m2l_pre_cell` /
`m2l_core` / `m2l_post_cell` against the operator table and **never takes
`aux`**; the only operator in that scope that does is `m2l_translate`, reached
through `run_m2l_fallback_at_depth`, which returns immediately on `n_fb == 0` —
and `fallback_pairs` is 0 at np=1. So no line T5 touched executes inside this
timer at this configuration. The three operators that did change signature are
M2M, L2L and the host-side operator build, none of which this timer covers. One
sample, no repeat, nothing tuned — R3 is not a correctness gate.

### What only running revealed

- **Nothing failed on the first build, and nothing failed on the first run.** No
  compile error, no gate failure, no bitwise difference.
- **`clang-format` reflowed six call sites and three signatures, and nothing
  else.** Run with explicit `--lines=` ranges derived from `git diff -U0` over
  each touched hunk, per the instruction not to run `clangformat.sh` over these
  headers. Every reflow was confined to a line this task had already changed:
  three operator parameter lists in `Canopy_LaplaceKernel.hpp` repacking
  `const AuxType& aux` onto the previous line, and the `m2m_translate`,
  `m2l_build_operator`, `m2l_translate` and `l2l_translate` call sites in the two
  sweeps repacking now that `aux` is shorter than `A_table`. The
  `aux_tables_type` / `build_aux_tables` block was already in the repo style.
- **The four other SERIAL targets and both examples compile**, verified because
  these are shared headers: `Canopy_Test_UpwardSweep_MPI_SERIAL`,
  `Canopy_Test_DownwardSweep_MPI_SERIAL`, `Canopy_Test_MultiSolve_MPI_SERIAL`,
  `Canopy_Test_SingleSolve_MPI_SERIAL`, `example_fmm`, `gravity_solve` — all
  built and linked clean. **They were built before the `clang-format` pass and
  not rebuilt after it**, at the user's instruction to skip the full rebuild for
  time; the only post-build change to those headers was the whitespace reflow
  described above, and `Canopy_Test_LaplaceSolve_MPI_SERIAL` *was* rebuilt after
  it and is the binary the gate ran. Not built: `Canopy_Test_LaplaceKernel_*`
  and `Canopy_Test_P2P_*`, out of scope — see the next item.
- **`tests/tstLaplaceKernel.hpp` is now broken in one more way than it was, as
  the task anticipated.** It calls `build_A_coefficients` directly at `:397`,
  `:559`, `:682` and `:773` and passes the resulting bare `View` to
  `m2m_translate`, `m2l_translate`, `m2l_build_operator` and `l2l_translate`,
  which now want an aux struct. That target already did not compile at `HEAD`
  (35 errors, "no matching function" on exactly those five operators), so this
  deepens a pre-existing breakage rather than creating one. **The repair is
  mechanical when someone takes it on**, but the four sites do not all map the
  same way: `:559`, `:682` and `:773` build to `2 * P_ORDER` and become
  `Kernel::template build_aux_tables<TEST_MEMSPACE>( P_ORDER )` exactly, while
  `:397` builds only to `P_ORDER` and feeds `m2m_translate`, which reads $A$ at
  degree $j \le P$ and so does not need the doubled table. `:397` can either
  call `build_aux_tables<TEST_MEMSPACE>( P_ORDER )` and over-build harmlessly, or
  keep `build_A_coefficients` and wrap the view in an
  `aux_tables_type<TEST_MEMSPACE>` by hand. `build_A_coefficients` itself is
  unchanged and still public, so neither route needs a new function.
- **Both sweeps still `#include "Canopy_SphericalCoefficients.hpp"` and no
  longer use anything from it.** Verified by grep for every symbol the header
  defines — `num_coeffs_symmetric`, `coeff_index`, `get_coeff`, `A_coeff`,
  `build_A_coefficients`, `a_index` — zero hits in either sweep after this
  change. The includes were **left in place**: they are not among T5's fifteen
  named sites, and removing them risks breaking a consumer that was relying on
  the transitive include, which is a build cycle this task did not need to spend.
  It is a one-line cleanup for whoever is next in these headers.
- **`make -j 4` again, per T1's operational note.** No SIGKILL.
- **`run_cmake_tuolumne.sh` still shows as modified and is still not this task's.**
  Whole-file line-ending churn plus a mode change to 755, predating the session.
  `setup-repo.txt` is likewise a pre-existing untracked file. Both left alone.

### Repository state left behind

- Three headers plus one test line modified: `src/Canopy_LaplaceKernel.hpp`,
  `src/Canopy_UpwardSweep.hpp`, `src/Canopy_DownwardSweep.hpp`,
  `tests/tstLaplaceSolve.hpp`. No test added, per scope.
  `tests/data/laplace_solve_P6.txt` and `README.md` untouched.
- `tasks/abstract-solver-backend.md`: T5 marked **DONE** with a **Met.**
  paragraph; the "Current state" opening updated from four built tasks to five;
  group (d)'s first bullet renumbered off the stale pre-T1 citations
  (`UpwardSweep:233-235`, `:501-504`; `DownwardSweep:529`, `:1099-1100`,
  `:1636-1639`, `:1688-1691`) onto the verified pre-T5 ones (`:258-259`, `:493`;
  `:597`, `:1164`, `:1686`, `:1738`) and marked **Done in T5**.
- Logs kept: `canopy-laplace-solve.f3XfiYHXy4b1.log` (the gate),
  `canopy-laplace-solve-prof.f3Xfk8H1YPNF.log` (R3).
- Out of scope and untouched, as directed: T6's `MonopoleBasis`;
  `M2L_KEY_DD_MAX`'s `float` branch, group (d)'s second bullet (T7);
  `sets_per_component` and the shared-cell Allreduce shape (T10);
  `kernel_params` (T9); the `per_cell_complex` / `total_complex` /
  `umcplx_view` / function-local `scalar_type` names T4 deliberately left (T10);
  `tests/tstLaplaceKernel.hpp` and `Canopy_Test_P2P_*`; `ctest -L regression`;
  and the partitioner's non-determinism. No `regression`-suite run was attempted.

**Affects:** **T6** — the last sweep-side obstacle group (d) posed to a
non-harmonic basis is gone on the aux half: `MonopoleBasis` supplies
`template <class MS> struct aux_tables_type {};` and
`build_aux_tables(int) { return {}; }`, and both sweeps compile against it with
no further change, because neither names a member of the struct. The remaining
group (d) obstacle is `M2L_KEY_DD_MAX`'s `float` branch, which is T7's. Note
also that **`aux()` returns the struct and not the table**, so T6 must not write
a conformance test that calls `ds.A_table()` — there is no such accessor on
either sweep now; `tests/tstLaplaceSolve.hpp:569` shows the solid-harmonic
spelling, `ds.aux().A_table`, and it is correct only for this basis. **T9** —
`build_aux_tables` takes exactly one argument today and the declaration says so;
when T9 introduces `kernel_params` and plumbs `FmmConfig::softening` into
`_downward`, the second argument goes here, and there are exactly two call sites
to update (`UpwardSweep::setup()` and the operator-table build in
`DownwardSweep`). **T7** — if the `M2L_KEY_DD_MAX` work wants a basis-supplied
value keyed on order, `aux_tables_type` is not the place: it is a *table* type
built at setup, and a scalar cap should be a `static constexpr` trait alongside
`m2l_num_src_coeffs`. **T10 and any later task using the gate** — the np 3-6
partitioner wobble is now known to reach np=3, not just np=5, and to be
diagnosable from the `ctest -V` log alone by comparing the three test bodies'
per-rank `n_unique_ops` lines; T4's observation that the split moves no printed
figure holds at np=5 but not at np=3, where it moved a direct-sum gradient in
its thirteenth significant figure. Do not read a 9th-or-later-figure move at any
rank count 3-6 as a finding without checking those lines first.

## T6 — a non-harmonic conformance basis drives the full pipeline

T6 is **DONE**, and unlike T3, T4 and T5 the load-bearing result is *positive*.
A basis that is not a spherical harmonic — coefficient a bare `double` rather
than `Kokkos::complex`, operator table $1\times1$ rather than $28\times49$,
auxiliary tables an empty struct, one coefficient per cell instead of 28 —
drives `UpwardSweep` and `DownwardSweep` end to end with **zero** changes to any
file under `src/`. The whole diff is two new files in `tests/`, one line in
`tests/CMakeLists.txt`, one new flux script, and the documentation. The
far-field contract is a real interface, not a rename of the solid-harmonic one.

Gate run flux job **`f3XgCb6ZSSwy`**. Worked at `HEAD` = `ac77ad3`.

### Decisions taken (given, not chosen here)

- **The host reference reads `comm_plan.m2l_plan().interaction_lists`**
  (`src/Canopy_CommunicationPlan.hpp:96`, `:219`), not a new `DownwardSweep`
  accessor. The eight CSR views the fused kernel actually walks
  (`_m2l_ns_csr_*`, `_m2l_sh_csr_*`, `src/Canopy_DownwardSweep.hpp:402-411`) are
  private with no accessor and stayed that way;
  `tests/tstDownwardSweep.hpp:831` already establishes the interaction-list
  route. **No diagnostic surface was added to `DownwardSweep`.** The two
  accessors the test does use, `total_m2l_pair_count()` and
  `total_fallback_pair_count()`, were already public (`:465`, `:478`).
- **The host reference sums in the sweep's order, not merely over the same
  set.** `EXPECT_DOUBLE_EQ` is 4 ULP and a reassociated sum of a few hundred
  same-magnitude terms drifts past it. The CSR sorts target entries by
  `(depth, target_idx)` (`src/Canopy_DownwardSweep.hpp:781-787`,
  `:1245-1252`) and keeps the traversal's push order within an entry, and that
  push order *is* the vector order of `interaction_lists[target_key]`
  (`src/Canopy_CommunicationPlan.hpp:481`) — so the reference iterates that
  vector as-is.
- **The M2L operator is the document's scale-normalized dimensionless one**,
  and the negative test is the document's permanent
  `#ifdef CANOPY_TEST_EXPECT_COMPILE_FAILURE` block. Neither was reopened.
- **`m2l_pre_cell` is a no-op, so T3's half-placed-hook finding does not bite
  and was not fixed.** T3 recorded that team scratch is per-target, so there is
  nowhere to put once-per-source-cell work that outlives the team.
  `MonopoleBasis` contracts the source monopole directly and has no per-source
  work to hoist, so it never needed the storage. The gap is unchanged and still
  belongs to whoever builds the first basis that needs it.

### The `dd` convention chosen, and why

The document requires a basis to state how it handles `dd` and does not pick
one. **`MonopoleBasis` uses**

$$
T(dd, i_x, i_y, i_z) = \frac{F(dd)}{\lVert (i_x,i_y,i_z) \rVert},
\qquad F(dd) = 2^{\max(0,\,-dd)}
$$

which is **`LaplaceKernel`'s own $F(dd, n, j)$ residual factor
(`src/Canopy_LaplaceKernel.hpp:656-661`) evaluated at the monopole term
$n = j = 0$**:

- $dd \ge 0 \Rightarrow F = 2^{j\,dd} = 2^0 = 1$
- $dd < 0 \Rightarrow F = 2^{-(n+1)\,dd} = 2^{-dd}$

and $F(0,\cdot,\cdot) = 1$, so same-depth operators carry no extra scaling.

Three reasons this was chosen over the two obvious alternatives:

- **It is derived, not invented.** Read directly: the physical monopole M2L is
  $L = M/r$. With this basis's multipole (a bare charge) and local (a charge
  over a separation measured in *deeper-cell half-widths*), the conversion
  between "separation in units of $w_{\rm unit}$" and "separation in units of
  $w_{\rm source}$" is $w_s/w_{\rm unit}$, which is $1$ when the source is the
  deeper cell ($dd \ge 0$) and $2^{-dd}$ when the target is ($dd < 0$). So
  $F(dd)$ is exactly what makes the operator depth-independent given the key,
  which is the premise hashing pairs onto keys rests on.
- **It makes `dd` load-bearing.** The obvious alternative — ignore `dd`
  entirely, $T = 1/\lVert\cdot\rVert$ — compiles and passes, but then distinct
  `dd` values produce keys whose operators are bit-identical, and a sweep bug
  that passed the wrong `dd` would be invisible to this fixture. Under the
  chosen convention a wrong `dd` changes the answer by a factor of two.
- **It is exact, not merely accurate.** $|dd| \le$ `M2L_KEY_DD_MAX` $= 6$ for
  `Scalar = double`, so $F \le 64$, every $F$ is an exact power of two, and the
  repeated doubling that computes it is exact in binary64. That matters because
  the whole gate is an exactness claim.

### Contract members `MonopoleBasis` had to supply that T6's `Do` list did not name

`Do` step 1 names six traits and step 2 names the nine operators in prose. The
complete set the two sweeps actually reach for, from
`grep -o 'KernelType::[a-zA-Z_0-9]*'` over both, is larger. The ones **not** in
the `Do` list, and what each had to become:

| Member | Form | Why the `Do` list could not have known |
| --- | --- | --- |
| `scalar_type` | `using scalar_type = Scalar` | Both sweeps re-export it (`UpwardSweep:63`, `DownwardSweep:107`) and `potential_view_type` / `gradient_view_type` are built on it. |
| `component_scalar_type` | `= Scalar` | T4's trait. The MPI datatype is selected from `sizeof` of it at three sites. |
| `max_order` | `= Order` | `UpwardSweep:94` aliases it as `P` and passes it to `build_aux_tables`; `DownwardSweep:1177` does the same on host. Nothing else reads it, so `Order = 0` is safe and is what the test instantiates. |
| `num_components` | `= NComps` | `NComps` in both sweeps, and the extent of `potential_view_type`. |
| `m2l_accumulator_type<ScratchSpace>` | unmanaged `View<scalar_type*>` | Not named anywhere in the sweeps — it is basis-private — but it is the shape the raw-byte scratch is re-viewed as, and omitting it would have meant a bare `reinterpret_cast` in two stages. |
| `m2l_scratch_bytes(int)` | `KOKKOS_INLINE_FUNCTION static constexpr` | `= num_coeffs_per_cell * n_comps * sizeof(scalar_type)`, i.e. 16 bytes at `NComps = 2`. **Kept `constexpr`**, per the task: `src/Canopy_DownwardSweep.hpp:1562` assigns it to a `constexpr size_t`. |
| `m2l_operators_type<MemorySpace>` | alias template, `View<coeff_type***, LayoutLeft, MS>` | T3 made it an alias *template*; a plain typedef does not compile against `typename KernelType::template m2l_operators_type<memory_space>`. |
| `m2l_translate` | full implementation, not a stub | The `Do` list's nine operators include it, but the list does not say it must agree *bit for bit* with the fused path. See the next section. |

Plus two members that exist only to keep the test exact and are not part of the
contract at all: **`m2l_operator_entry(dd, ix, iy, iz)`** and
**`m2l_accumulate(acc, T, M)`**. See "What only running revealed".

### `aux_tables_type` is a struct template, and nothing names a member of it

Exactly as T5 handed forward:

```cpp
template <class MemorySpace>
struct aux_tables_type
{
};

template <class MemorySpace>
static aux_tables_type<MemorySpace> build_aux_tables( int order )
{
    (void)order;
    return {};
}
```

Both sweeps compiled against this with no change, because neither names a
member of the struct — which is the property T5 built and this is the first
independent confirmation of it. T5's warning that **`aux()` returns the struct
and not the table** was heeded: no conformance test here calls `A_table()` or
`ds.aux().A_table`, and there is no such accessor on either sweep.

### `sets_per_component` is declared and inert

`grep -rn sets_per_component src/ tests/` finds it in exactly one place after
this change: the declaration on `MonopoleBasis`. No sweep reads it, because
**T10** is what raises it to 2 and teaches the sweeps to read it. It is
declared anyway so the trait list a basis author sees is complete and T10's
diff is a change of value rather than an addition, and the declaration says in
its own comment that nothing consumes it until T10 — so a reader does not hunt
for the consumer. T6's `Do` step 1 has been annotated to the same effect.

### Signatures changed

**None.** Not one file under `src/` was modified. The complete repository diff
is:

- `tests/CanopyTest_MonopoleBasis.hpp` — new, 663 lines.
- `tests/tstFarFieldContract.hpp` — new, 798 lines.
- `tests/CMakeLists.txt` — one line: `FarFieldContract` added to
  `UNIT_MPI_TESTS` (`:48-57`), not to `REGRESSION_MPI_TESTS`. That yields
  target `Canopy_Test_FarFieldContract_MPI_SERIAL` (and the OPENMP/HIP variants
  the harness always generates, neither of which was built) and tests
  `..._np_1` through `..._np_6` with CTest label `unit`.
- `scripts/tuolumne/run_ctest_far_field_contract.flux` — new. Runs both suites
  in one allocation deliberately, so the claim that the Laplace gate did not
  move is checkable from one log.
- `tasks/abstract-solver-backend.md` — T6 marked **DONE** with a **Met.**
  paragraph; the two stale bits in T6 corrected (see below).
- `tasks/abstract-solver-backend-progress-log.md` — this section.

### The gate: what it compares, and why the comparison is exact

`tests/tstFarFieldContract.hpp` drives both sweeps on `MonopoleBasis`, then
compares `downward.locals()` against a host reference. Three tests, all at 1-6
ranks: `localsMatchHostReferenceBasic` (1000 particles/rank, ncrit 32,
max_depth 6, replication_depth 2), `localsMatchHostReferenceSmall` (200, 16, 4,
1) and `l2pReturnsTheLocal`.

For `MonopoleBasis` the whole downward pipeline collapses to a telescoping sum.
Writing $D(a) = \sum_{s \in \mathrm{ilist}(a)} T(\mathrm{key}(a,s))\,M(s)$ for
the M2L delta, and noting that L2L copies a parent's local to each child,
`locals()` must hold $L(t) = D(t) + L(\mathrm{parent}(t))$ with
$L(\mathrm{root}) = D(\mathrm{root})$. The reference evaluates that
root-downward.

Four things make it exact rather than close, and each is load-bearing:

1. **$D(a)$ is summed in the sweep's order.** The reference iterates
   `interaction_lists[target_key]` as-is, per the decision above.
2. **The operator value and the multiply-accumulate step are not duplicated.**
   Both go through `MonopoleBasis::m2l_operator_entry` and
   `MonopoleBasis::m2l_accumulate`, the same functions the device kernel calls.
3. **The reference takes the upward sweep's output as given.** It mirrors
   `upward.multipoles()` rather than recomputing P2M and M2M, so no assumption
   about particle or child iteration order enters. This gates the *far field* —
   M2L, L2L, L2P — which is what T6 is for. The mirror is taken **after**
   `downward.execute()`, because `exchange_multipoles_for_m2l` writes remote
   source multipoles into the same view (non-accumulating,
   `src/Canopy_DownwardSweep.hpp:1497`) and only then does it hold every source
   the interaction lists name.
4. **Shared cells get their own arithmetic branch.** See the next section; this
   is the one place the reference had to reproduce a mechanism rather than a
   result, and getting it wrong would have made the gate fail.

$D(a)$ is defined globally — the plan puts `interaction_lists[a]` on exactly
one rank (`src/Canopy_CommunicationPlan.hpp:477-486`: $a$'s owner, or rank 0
when $a$ is shared) — so the reference computes the local part and closes it
with one `MPI_Allreduce(MPI_SUM)` over a dense per-cell array in which every
other rank contributes an exact `+0.0`. That is bit-exact whatever order the
reduction takes, which is why the reference is the same expression at every
rank count.

The rank-local cell index is used as a global index, which is only valid
because `TreeBuilder` builds the same tree on every rank (refinement driven by
`MPI_Allreduce`'d per-cell global counts, `src/Canopy_TreeBuilder.hpp:743`).
**That is asserted, not assumed**: the test `MPI_Allreduce`s the cell count
(MIN vs MAX) and an in-repo FNV-1a hash of the key sequence (MIN vs MAX) and
`ASSERT_EQ`s both, so if it ever stops holding the test says so rather than
silently combining different cells.

The comparison is restricted to cells this rank owns outright plus the shared
ones. A cell a rank neither owns nor shares carries a partial value in
`_locals` — L2L for a shared parent writes into every child's slot on every
rank — and no correctness claim rests on it.

### The one thing the design did not anticipate: the shared-cell round trip is not the identity

R6 states that at np=1 `allreduce_shared_locals_at_depth` is "an exact
identity, structurally", because the unpack computes
`Snap[k] + (L(j) - Snap[k])` which "is `L(j)`". **That is true of the slot
algebra and false of the floating-point arithmetic**, and a reference written on
it would have failed the gate.

Concretely, for a shared target with $a = L(\mathrm{parent})$ and $b = D(t)$ the
sweep computes

```
a     = L(parent)      snapshot, taken after L2L(depth-1)
c     = a + b          m2l_post_cell's `+=`
delta = c - a          the Allreduce send buffer
L(t)  = a + delta      snapshot + summed delta
```

and `a + fl(fl(a+b) - a)` is not `fl(a+b)` in general, because the subtraction
re-rounds. A non-shared target, by contrast, really does get `D(t) + L(parent)`
— `run_m2l_all` writes $D$ into a freshly zeroed local before the depth loop and
L2L then adds the parent, directly or through the accumulating L2L exchange.

So the reference carries **two branches**, selected by
`owner == OWNER_SHARED`. That predicate is safe to use because
`TreePartitioner` marks a cell `OWNER_SHARED` exactly when
`depth <= replication_depth && !is_leaf`
(`src/Canopy_TreePartitioner.hpp:493-502`), which is character-for-character
`CommunicationPlan`'s shared-cell predicate (`:698`) — so the ownership map, the
shared-target CSR filter and the snapshot's cell list agree by construction, and
one lookup decides which arithmetic applies.

R6's operational conclusions are unaffected: the four classes it tabulates, and
the finding that a pack-side-only perturbation corrupts np=1 while a both-sides
one does not, are all statements about the slot map $\sigma$ and all still hold.
What does not hold is the sentence that `_locals` is *unchanged* by the
function at np=1. It is changed, in the last bit, on any shared cell where
`c - a` re-rounds. **T10 should not build a check on `_locals` being unchanged
at np=1.**

### The shared-cell path is exercised at np=1, as required

Measured, from the gate log, `shared_cells` per rank:

| config | np=1 | np=2 | np=3 | np=4 | np=5 | np=6 |
| --- | --- | --- | --- | --- | --- | --- |
| Basic (replication_depth 2) | **11** | 34 | 60 | 68 | 72 | 73 |
| Small (replication_depth 1) | **9** | 9 | 9 | 9 | 9 | 9 |

np=1 has 11 shared cells at the Basic configuration — 1 at depth 0, 8 at depth
1 and 2 at depth 2 — and 9 at Small (1 + 8, all depth-1 cells non-leaf). The
Basic counts grow with rank count only because the particle count is per-rank,
so more ranks means a bigger tree and more non-leaf cells at depths $\le 2$;
at np=6 all 64 depth-2 cells are non-leaf and the count saturates at
$1+8+64=73$. **Nothing in the test is designed on the premise that np=1 is
insulated from this path**, and the two-branch reference above is what makes
that concrete rather than a claim.

### Gate measurements — FarFieldContract

Flux job **`f3XgCb6ZSSwy`**, tuolumne1041, Cray clang 20.0.0, spack env
`tuolumne_trilinos`, `RelWithDebInfo`, Kokkos SERIAL, `build-tuolumne/`
(`Canopy_ENABLE_PROFILING=OFF`), 35.09 s of CTest wall time.
`100% tests passed, 0 tests failed out of 6`. Three test bodies x six rank
counts, all `OK`, no `SKIPPED`.

**Every rank at every rank count reported `not_bit_identical=0`.** The verdict
the test asserts is `EXPECT_DOUBLE_EQ` (4 ULP), as T6 specifies, but the
reference is in fact bit-exact on all 42 (rank, configuration) pairs and every
one of the 9190 (cell, component) slots checked across them — so the exactness
argument above is not merely within tolerance, it is tight, and a future run
that reports a non-zero `not_bit_identical` while still passing is a signal
that something in it has developed a hole.

`fallback_pairs=0` everywhere, so no pair tripped a range guard or the count
cap, and `m2l_pairs` equals `ilist_pairs` on every rank — the fused path
carried every pair.

| np | Basic: targets / m2l_pairs / checked (per rank) | Small: targets / m2l_pairs / checked |
| --- | --- | --- |
| 1 | 80 / 1200 / 178 | 54 / 610 / 138 |
| 2 | 141,116 / 11215,10975 / 300,300 | 32,32 / 408,408 / 82,82 |
| 3 | 197,130,129 / 26121,19861,18770 / 412,380,378 | 23,40,35 / 474,696,808 / 64,98,88 |
| 4 | 179,118,113,109 / 24668,19081,18781,17858 / 376,372,362,354 | 35,37,41,40 / 1546,1397,1680,1537 / 88,92,100,98 |
| 5 | 160,92,108,98,100 / 23087,16070,17681,16309,15961 / 338,328,360,340,344 | 49,47,40,34,47 / 3116,2983,2686,2354,3123 / 116,112,98,86,112 |
| 6 | 152,80,87,79,80,91 / 21080,13940,13985,14541,13546,14120 / 322,306,320,304,306,328 | 63,53,39,49,41,62 / 6246,5096,4435,5109,4273,5957 / 146,124,98,118,100,142 |

The largest single sum the reference reproduces bit-for-bit is a 197-target
rank at np=3 carrying 26 121 pairs, i.e. individual $D(a)$ sums of a few hundred
terms of like magnitude — which is exactly the regime the "sum in the sweep's
order" decision exists for.

### Gate measurements — the Laplace-solve gate, unchanged

Same flux job **`f3XgCb6ZSSwy`**, second `ctest` invocation, 36.02 s of CTest
wall time, `100% tests passed, 0 tests failed out of 6`.

| np | cross-rank pot | cross-rank grad | direct-sum pot | direct-sum grad |
| --- | --- | --- | --- | --- |
| 1 | (reference) | (reference) | 3.2093610331931809e-07 | 4.2399302231264458e-08 |
| 2 | 4.1994107222659022e-13 | 2.1570013757642702e-12 | 3.2093610363952985e-07 | 4.239936667274564e-08 |
| 3 | 8.0211305411572796e-13 | 4.0353546216363242e-12 | 3.2093610299898352e-07 | 4.2399380705248681e-08 |
| 4 | 1.114294857300434e-12 | 5.5987399483706545e-12 | 3.2093610299888352e-07 | 4.2399368624331468e-08 |
| 5 | 9.5809726336249496e-14 | 6.4438389009577268e-13 | 3.209361028925181e-07 | 4.2399357908696204e-08 |
| 6 | 5.688835866646797e-13 | 2.8631357246763719e-12 | 3.2093610299905848e-07 | 4.2399381662523099e-08 |

**All 22 cells are character-for-character T1's table**, including the np=3
direct-sum gradient that T5's run moved in its thirteenth figure
(4.23993807052**48681**e-08 here, T1's value; T5 printed
4.23993807052**33977**e-08). That is the expected outcome — T6 adds a basis and
a test and touches no shared code — and it also **retroactively confirms T5's
attribution** of that move to the partitioner rather than to T5's own diff.

`fallback_pairs = 0`, `locals_ext = (103,28,1)`, `optab_ext = (28,49,n_ops)`,
`a_extent = 169` and `initial_hash = 0xb6ad437608ad69b7` at every rank and rank
count. `bitForBitArtifacts` ran and passed at np=1 rank 0 and np=2 ranks 0 and
1 — `locals()`, the operator table, the $A_{n,m}$ table and the realized key
list byte-identical to `tests/data/laplace_solve_P6.txt` — and `SKIPPED` at
np 3-6 by design.

`n_unique_ops` per rank: np=1 → 686; np=2 → 368, 386; np=3 → 273, 204, 329;
np=4 → 264, 128, 234, 194; np=5 → 180, 168, 147, 217, 187 — all identical to T1.

**The multijagged two-cut split landed at np=6 this run, which is a rank count
it had not previously been seen at**, and it moved nothing printable.
Attributed from this run's own log with no control run, per T5's procedure:

| np=6 solve | rank 0 | 1 | 2 | 3 | 4 | 5 | sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `matchesDirectSum` (T1's cut) | 174 | 156 | 111 | 160 | 116 | 175 | 892 |
| `crossRankAgreement` | 174 | 160 | 111 | 153 | 126 | 175 | 899 |

Both printed figures still match T1 to all 17 digits, so np=6 joins np=5 (T3,
T4) as a rank count where the split moves nothing, against np=3 (T5) where it
moved a gradient in the thirteenth figure. The split has now been observed at
np=3, np=5 and np=6, i.e. it is not a property of any particular rank count,
and the np 1-2 bitwise half of the gate is unaffected because the cut over one
or two parts is reproducible.

### R3 — not measured, and why that is the right call

R3 is a trait indirection deoptimizing the fused M2L kernel. **T6 changes no
line the fused M2L kernel executes on the solid-harmonic path.** Not one file
under `src/` was modified, so the `Canopy_Test_LaplaceSolve_MPI_SERIAL` binary's
M2L code path is the same instruction stream T5 measured, and a
`build-tuolumne-prof/` sample would be re-measuring T5's number with a
different node assignment. The structural argument is total here rather than
partial, which is stronger than any single sample of a timer that prints three
decimals on values of 0.002-0.004 s. T5's `f3Xfk8H1YPNF` figure (0.070 s over
24 solves) stands as the current baseline for T7.

### What only running revealed

- **`clang::-ffp-contract` is a real hazard between a device kernel and a host
  reference, and it is why `m2l_accumulate` exists.** `acc += T * M` as one
  statement is contractible to an FMA under the compiler's default
  `-ffp-contract=on`. Nothing guarantees the same decision in `m2l_core`
  (inside a `Kokkos::parallel_for` lambda, compiled for `gfx942` as well as
  host) and in a plain host loop, and a single differing FMA over a few hundred
  terms would drift past 4 ULP. `MonopoleBasis::m2l_accumulate` splits the
  product into a named local, putting a statement boundary that
  `-ffp-contract=on` may not cross, and both callers go through it. This was
  written before the first build and `not_bit_identical=0` everywhere is the
  evidence it was needed — or at least that it is sufficient.
- **Clang reports only the FIRST failing class-scope `static_assert` per class
  instantiation, so one inconsistent basis exercises only two of the four
  guards.** The negative block originally declared one basis
  (`coeff_type = double`, `scalars_per_coeff = 2`) and the build produced
  exactly two diagnostics — `Canopy_UpwardSweep.hpp:73` and
  `Canopy_DownwardSweep.hpp:117`, the two `sizeof` asserts — and never reached
  the two `detail::coeff_traits` cross-checks, even though both would have
  failed. That means a one-basis block would **not** have noticed if the two
  traits asserts had been deleted, which is precisely the property the design
  wants the block to have. **A second basis was added**,
  `InconsistentComponentBasis`, with `component_scalar_type = float` and
  `scalars_per_coeff = 2`: `sizeof(double) == 8 == 2 * sizeof(float)` so the
  `sizeof` assert passes and the traits cross-check is the first to fail. With
  both cases the build emits all four messages. **Any later task adding a
  class-scope `static_assert` to a sweep must add its own case here, or the
  block will silently stop covering it.**
- **Both inconsistent bases are `struct X : public MonopoleBasis<...>` with one
  or two traits redeclared, and that is deliberate rather than a shortcut.**
  Because the *only* thing wrong with each is its changed traits, deleting the
  four asserts makes the block compile — which is the failure signal. A
  hand-rolled minimal bad basis would keep failing on missing members after the
  asserts were gone and would therefore detect nothing.
- **`m2l_translate` had to be bit-consistent with the fused path, which the
  `Do` list does not say.** A basis could satisfy the contract with an
  arbitrarily different fallback, but then the host reference would have to know
  which path each pair took. `MonopoleBasis::m2l_translate` instead
  reconstructs the same integer key from the physical geometry it is handed —
  `w_unit = min(w_source, w_target)` is the sweep's own definition of the
  deeper half-width; `w_target / w_source` is exactly $2^{dd}$ because every
  half-width is the root half-width scaled by a power of two — and calls the
  same `m2l_operator_entry`. `fallback_pairs = 0` at every rank count, so this
  path did not in fact execute, but the test does not depend on that and the
  consistency is what makes `fallback_pairs` a printed diagnostic rather than
  an assertion.
- **`NComps = 2`, not 1, and it is not cosmetic.** With
  `num_coeffs_per_cell = 1`, `NComps = 1` would make
  `per_cell_complex = coeffs_per_cell * NComps = 1` in
  `allreduce_shared_locals_at_depth`, its slot expression degenerate, and any
  slot-indexing error there invisible. `NComps = 2` gives stride 2 and puts
  those pack/unpack loops under a real index. This matters for **T10**, which
  rewrites exactly those loops.
- **`Order = 0` is safe.** `grep -n '\bP\b' src/Canopy_UpwardSweep.hpp` shows
  `max_order` reaches nothing but `build_aux_tables`, in both sweeps, so a
  basis with no order at all can declare `max_order = 0`. The test instantiates
  `MonopoleBasis<double, 0, 2>`.
- **`M2L_KEY_DD_MAX`'s `float` branch was never reached**, as the task said it
  would not be: `Scalar = double` takes the non-`float` branch
  (`src/Canopy_DownwardSweep.hpp:333-334`) and it compiles. Left alone; it is
  T7's.
- **Nothing failed on the first build, and nothing failed on the first run.**
  The only two source edits after the first successful compile were the
  shared-cell arithmetic branch (found by reading `R6` against the code, not by
  a failing run) and the second negative-test basis.
- **Adding a name to `tests/CMakeLists.txt` needs `make
  cmake_check_build_system` first.** `make -j 4 <new target>` in
  `build-tuolumne/` fails with "No rule to make target" — make errors out
  before it regenerates, because the target does not exist in the current
  Makefile. `make cmake_check_build_system` regenerates against the existing
  cache (which it preserved: `Canopy_ENABLE_PROFILING` stayed `OFF`, the
  `MPIEXEC_*` overrides stayed put) and the target then builds.
- **`make -j 4`, per T1's operational note.** No SIGKILL. The `tstLaplaceSolve`
  and `tstFarFieldContract` translation units each take roughly three minutes
  on the login node.
- **No `clang-format` pass was run**, per `CLAUDE.md`'s "Do not clang format"
  and commit `82b052c`.
- **`run_cmake_tuolumne.sh` still shows as modified and is still not this
  task's** — whole-file line-ending churn plus a mode change, predating the
  session. `setup-repo.txt` is likewise a pre-existing untracked file. Both
  left alone. **`tests/tstLaplaceSolve.hpp` also shows as modified, and that is
  T5's uncommitted one-liner** (`ds.A_table()` → `ds.aux().A_table`), without
  which `Canopy_Test_LaplaceSolve_MPI_SERIAL` does not compile at `HEAD`. It is
  in the working tree, was needed to build the gate, and is committed with T6
  rather than left dangling.

### How to re-run the negative test

Committed in this file's header comment as well, so it does not live only here:

```bash
b=build-tuolumne/tests
d=$b/CMakeFiles/Canopy_Test_FarFieldContract_MPI_SERIAL.dir
touch $b/SERIAL/tstFarFieldContract_SERIAL.cpp
make -C $b Canopy_Test_FarFieldContract_MPI_SERIAL \
  CXX_DEFINES="$(sed -n 's/^CXX_DEFINES = //p' $d/flags.make) \
               -DCANOPY_TEST_EXPECT_COMPILE_FAILURE"
```

`make`'s command-line assignment overrides the one `flags.make` makes, so the
`sed` re-supplies the defines the build needs and appends ours; the `touch` is
there because changing a `-D` changes no file timestamp. This route was chosen
over adding a CMake option because the alternative would have written a new
cache entry into `build-tuolumne/`, which is the bitwise gate's configuration
and which R3 rests on not moving. **Remember to rebuild the positive target
afterwards** — the failed compile leaves the object stale.

### Doc corrections made as part of T6

- T6's **Reference** cited `tests/tstDownwardSweep.hpp:57` for how a test
  instantiates a basis. That line is blank. Repointed at `:56` (the basis
  alias, `using Kernel = LaplaceKernel<double, P_ORDER>;`) and at `:156-163`
  (the pattern actually worth copying — construct builder, partitioner and comm
  plan, then both sweeps and `setup()`).
- T6's **Do** step 1 listed `sets_per_component = 1` among the traits with no
  note that nothing reads it. Kept, with a paragraph added saying it is inert
  and forward-looking until T10 and that the declaration must say so.

### Repository state left behind

- `tests/CanopyTest_MonopoleBasis.hpp` (663 lines) and
  `tests/tstFarFieldContract.hpp` (798 lines) — new.
- `tests/CMakeLists.txt` — one line.
- `scripts/tuolumne/run_ctest_far_field_contract.flux` — new.
- `tests/tstLaplaceSolve.hpp` — T5's uncommitted one-liner, carried in.
- **No file under `src/` was modified.**
- Log kept: `canopy-far-field-contract.f3XgCb6ZSSwy.log` (both suites).
- Out of scope and untouched, as directed: `M2L_KEY_DD_MAX`'s `float` branch,
  `canonicalize_key`, `key_needs_level` (T7); `unit_w` and `kernel_params` in
  the operator builder (T9); `sets_per_component` actually doing something and
  the shared-cell slot expression (T10); the `m2l_pre_cell` per-source-cell
  storage gap (T3/T6 — not needed, see above); `tests/tstLaplaceKernel.hpp` and
  `Canopy_Test_P2P_*`, which do not compile at `HEAD`; `ctest -L regression`;
  the partitioner's non-determinism; `run_cmake_tuolumne.sh`; and
  `setup-repo.txt`. `README.md` was not touched: no public API and no example's
  arguments changed, and no new known issue was found.

**Affects:** **T7** and **T10** — both edit
`tests/CanopyTest_MonopoleBasis.hpp`, and three things about its shape matter.
First, **the operator value lives in exactly one function**,
`m2l_operator_entry(dd, ix, iy, iz)`, called by both `m2l_build_operator` and
the host reference; T7's depth-carrying key changes what `dd` *means* to the
sweep but not this function's signature, and if T7 does change the signature it
must change both callers or the gate will compare an operator against a
different operator and fail with a message that looks like a sweep bug. Second,
**`m2l_accumulate` exists to defeat FP contraction** and must not be inlined
back into `m2l_core`; see "What only running revealed". Third, **the negative
`#ifdef` block covers four guards with two bases because clang stops at the
first failing class-scope assert per instantiation** — any new sweep-side
`static_assert` needs its own case added there. **T7** additionally: this basis
takes the non-`float` branch of `M2L_KEY_DD_MAX` and the `float` branch is
still unexercised; and `MonopoleBasis`'s `dd` convention is
$F(dd) = 2^{\max(0,-dd)}$, which is `LaplaceKernel`'s $F(dd,n,j)$ at
$n=j=0$ — so if T7 makes the key carry depth, this basis's operator is already
depth-independent given the key and needs no change beyond whatever
`canonicalize_key` does to the key itself. **T10** additionally, and this is
the substantive finding: **R6's claim that `allreduce_shared_locals_at_depth`
leaves `_locals` unchanged at np=1 is true of the slot algebra and false of the
floating-point arithmetic** — `a + fl(fl(a+b) - a)` is not `fl(a+b)`, so the
function does perturb `_locals` in the last bit at np=1 on any shared cell where
the subtraction re-rounds. Do not build a T10 check on np=1 invariance of
`_locals`. R6's four-class table and its pack-side-versus-both-sides finding are
unaffected, since both are statements about the slot map. T10 also inherits a
ready-made multi-component exercise of exactly the loops it rewrites:
`FarFieldContract` runs at `NComps = 2` with `num_coeffs_per_cell = 1`, so
`per_cell_complex` is 2 and an aliasing or truncating slot expression is
visible; and `owner == OWNER_SHARED` is a safe predicate for "this cell goes
through the snapshot/Allreduce path", because
`src/Canopy_TreePartitioner.hpp:493-502` and
`src/Canopy_CommunicationPlan.hpp:698` carry the same predicate.
**T8** — `total_fallback_pair_count()` is 0 on all 42 (rank, configuration)
pairs measured here, so R4's discriminator is intact for `MonopoleBasis` too,
and its $1\times1$ operator makes it the cheapest basis to test an overflow
policy against: at 16 bytes per key the count cap binds long before any byte
budget. **T11** — `MonopoleBasis` is now a second `FarField` type that
instantiates both sweeps cleanly, so T11 can use it to check that its `Solver`
parameter really is a parameter without needing T12's Cartesian-Taylor basis to
exist first. **T12** — the shape of `MonopoleBasis` is the template to copy;
the two things it should keep are the single-source-of-truth operator function
and the units-and-conventions block on the declaration.
