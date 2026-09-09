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
