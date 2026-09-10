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
