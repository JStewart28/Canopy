# Tree balance and M2L operator-table economy on non-uniform trees

**Status:** IN PROGRESS

## Problem

Canopy's M2L fast path works by **tabulation**: the classify pass reduces every
(target, source) pair to a five-integer operator key
$(\texttt{max\_d}, \texttt{dd}, \texttt{ii}, \texttt{jj}, \texttt{kk})$, pairs
sharing a canonical key share one dense operator column, and the fused kernel
contracts source moments against that column. Tabulation pays only when many
pairs collide on one key.

On a **deeply non-uniform tree it stops paying**, in three separate ways that
all trace to the same cause. The tree is built purely by density — a cell
becomes a leaf as soon as it holds at most `ncrit` particles
(`src/Canopy_TreeBuilder.hpp:769`), with no reference to its neighbours' depths
— so a sparse region produces a **shallow leaf** that can sit adjacent to a
deeply refined region. The dual-tree traversal cannot split a leaf, so when one
side is a shallow leaf it splits the other side instead
(`src/Canopy_CommunicationPlan.hpp:639-641`):

```cpp
const bool split_t = S->is_leaf || ( !T->is_leaf && T->half_width >= S->half_width );
```

It therefore descends the fine side repeatedly while the coarse leaf stays put,
and emits pairs whose depth difference is as large as the tree's depth span.
Three consequences:

1. **Those pairs are refused a column and take the per-pair fallback.** The
   classify pass's range guard (`src/Canopy_DownwardSweep.hpp:1704-1709`) refuses
   any pair with $|\texttt{dd}| > \texttt{m2l\_key\_dd\_max}$ or an offset
   component over `M2L_KEY_OFFSET_MAX` (`:570`). Both fire for this geometry:
   the key's unit length is the half-width at `max_d`, the **finer** cell, so a
   separation that is modest in coarse-cell units is $2^{|\texttt{dd}|}$ times
   larger in the key's units.
2. **The table holds provably duplicate columns.**
   `CartesianTaylorBasis`'s operator is a function of the physical
   $(R, b)$ alone and has no `dd` dependence at all
   (`src/Canopy_CartesianTaylorBasis.hpp:972-979`), yet `dd` is in its key, so
   keys differing only in `dd` build identical columns.
3. **The operator cache retains nothing across rebuilds.** The key carries
   `max_d`, which indexes a per-level physical half-width, and the root
   half-width is the global particle bounding box recomputed on every full
   rebuild (`src/Canopy_TreeBuilder.hpp:608-635`). So `set_root_half_width`
   clears the entire cache whenever `key_needs_level` is true
   (`src/Canopy_DownwardSweep.hpp:450-460`), and a moving particle distribution
   rebuilds the whole table at every evaluation.

The end state is that all three are bounded independently of tree depth: the
range guard stops firing because no pair has a large depth difference
(**chain A**), the table stops holding duplicate columns and stops being keyed
on a quantity that drifts (**chain B**), and the depth a large problem actually
needs is measured rather than assumed (**chain C**).

### What this is not

- **Not a widening of `M2L_KEY_DD_MAX` or `M2L_KEY_OFFSET_MAX`.** Those bounds
  are correctly sized; see [The bounds are sized for a balanced
  tree](#the-bounds-are-sized-for-a-balanced-tree). Widening them converts a
  correct fast path into an untabulatable table — for a large depth difference
  the fine cell can sit anywhere within $\sim 11 W_{\text{coarse}}$ at
  resolution $W_{\text{fine}}$, so those pairs share no keys with each other and
  each admitted column is rebuilt every evaluation for a single pair's benefit.
  The per-pair fallback is the correct algorithm for them.
- **Not a change to any kernel's mathematics**, to `m2l_operator_block`, to the
  derivative ladder, or to the softening.
- **Not a change to the MAC**, to `mac_satisfied`
  (`src/Canopy_CommunicationPlan.hpp:338-350`), or to `mac_theta`.
- **Not scale-normalization of `CartesianTaylorBasis`.** It is unreachable for
  that basis; see [Why `CartesianTaylorBasis` cannot be
  level-blind](#why-cartesiantaylorbasis-cannot-be-level-blind).
- **Not a relabeling of any existing test.** Every task here adds `unit`-tier
  coverage, and each task's exit criterion below names the stems and rank counts
  it must pass — that list is the task's gate, and nothing wider is required of
  it. The `regression` label keeps its single member, `MultiSolve`
  (`tests/CMakeLists.txt:61-63`). `CartesianTaylorBasis`'s coverage stays in the
  `unit` tier, and B1's and B2's exit criteria are what hold it: both name
  `CartesianTaylorSolve` as the authority on the basis they change, precisely
  because `MultiSolve` would pass with that basis wholly broken.

## Approach

**Measure, then sharpen the checks, then change the mechanism.** In that order,
and the middle step is not optional: the payoff of every mechanism change here
is currently unknown and countable, and the tests that would have to catch a
mistake are in several places too loose to do it.

The work is three mechanism chains, preceded by two tasks that serve all of
them. **T1** establishes the fixture the rest measure on. **V1** sharpens what
the mechanism changes will be verified against — without it a chain-A or
chain-B regression of a few percent passes every existing check (**R9**,
**R10**). The chains are independent of each other and may land in any order:

| | removes | tasks |
| --- | --- | --- |
| **A** | the refusals, by balancing the tree | A1, A2, A3 |
| **B** | the duplicate and drifting columns, by fixing the key | B0, B1, B2 |
| **C** | the unknowns about scale and about what the fallback costs | C1 |

The instrumentation needed to measure all of this already exists and is listed
under [Current state](#current-state). No new accessor is required by any task
here.

### The bounds are sized for a balanced tree

`mac_satisfied` accepts a pair iff $R\theta > r_A + r_B$ with
$r = \sqrt{3}\,w$ the circumradius of a cell of half-width $w$
(`src/Canopy_CommunicationPlan.hpp:338-350`). A pair is only *emitted* after its
parent pair failed that test, so for two cells at the same depth with
half-width $w$, whose parents have half-width $2w$,

$$
R \le \frac{\sqrt{3}\,(2w) + \sqrt{3}\,(2w)}{\theta}
  = \frac{4\sqrt{3}\,w}{\theta},
\qquad\text{so}\qquad
\frac{R}{w} \le \frac{4\sqrt{3}}{\theta} \approx 23 \ \text{at}\ \theta = 0.3 .
$$

The key's offset is measured in half-widths at `max_d`, so for a near-balanced
tree $|\texttt{ii}|,|\texttt{jj}|,|\texttt{kk}| \lesssim 23$, just inside
`M2L_KEY_OFFSET_MAX = 32`. **The bound is a deliberate fit to a balanced tree,
and the guard fires precisely when the tree is unbalanced.** That is the reason
chain A attacks the balance rather than the bound.

### Why `CartesianTaylorBasis` cannot be level-blind

Making a basis level-blind — `key_needs_level = false` and `max_d` zeroed in
`canonicalize_key`, as `LaplaceKernel` does
(`src/Canopy_LaplaceKernel.hpp:721`, `:742-746`) — requires the operator to be
homogeneous in length, so that normalizing by powers of the cell width removes
every dependence on absolute scale. **`CartesianTaylorBasis` is not
homogeneous, and cannot be made so.** Its kernel is Plummer-softened,

$$
\varphi(r) = \left(r^2 + b\right)^{-1/2}, \qquad b > 0,
$$

and $b$ is a physical length squared carried as one scalar for the whole solve
(`src/Canopy_CartesianTaylorBasis.hpp:536-560`). A positive $b$ is **mandatory**,
not optional: `m2l_translate` aborts when it is not positive, because the
unsoftened kernel has no expansion in this basis
(`src/Canopy_CartesianTaylorBasis.hpp:1072-1080`). Writing $R = n W$ for an
integer offset vector $n$ at width $W$, an order-$m$ derivative of $\varphi$
scales as

$$
W^{-(1+m)}\, g\!\left(n,\ \frac{b}{W^{2}}\right),
$$

so width normalization removes the prefactor but leaves a residual dependence on
the dimensionless $b/W^{2}$, which differs at every level. Two pairs with the
same integer offset at different depths therefore have genuinely different
operators, and `max_d` must stay in the key. Chain B pursues the two aims a
level-blind key would have served by other means that are exact: B1 removes
`dd`, which the operator provably does not depend on, and B2 replaces the
drifting level index with a stable one.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| New basis trait name | `key_needs_dd` | Mirrors `key_needs_level` exactly in name, placement (beside it in the basis's M2L key-contract block), type (`static constexpr bool`) and role: a basis declaring which of the key's five integers its operator depends on. A reader who knows one knows the other. |
| Trait default | none — every basis states it explicitly | `key_needs_level` has no default either. A silent default on a correctness-critical trait is how a new basis gets the wrong one; the conformance test in B1 fails a basis that omits it. |
| Trait/`canonicalize_key` agreement | asserted, never trusted | `expectKeyTraitsAgree` (`tests/tstFarFieldContract.hpp:925-958`) already enforces this for `key_needs_level`; B1 extends the same function rather than adding a second one. |
| Balancing knob name | `FmmConfig::tree_balance_max_level_delta` | `FmmConfig` is where every other tree and table knob lives (`src/Canopy_Solver.hpp`), and the name states the invariant as a number rather than as a mode, so "2:1" is the value 1 and "off" is a large value rather than a second boolean. |
| Balancing knob default | the off value, until A3 | A2 must not move any existing result. A knob whose default is the current behavior is a change with no runtime surface until something sets it, which is what makes A2's exit criterion checkable against the existing stems, unmodified. |
| Root-width quantization | `std::ldexp`/`std::frexp`, never `pow(2, round(log2(w)))` | Exact in binary floating point. A rounded `pow` reintroduces the drift the quantization exists to remove, and would do it only on some inputs. |
| Where a refused pair goes | unchanged — `m2l_overflow_policy` | `CartesianTaylorBasis` selects `PerPairTranslate` (`src/Canopy_CartesianTaylorBasis.hpp:516`) and that stays. Nothing here changes what happens to a refused pair, only how many pairs are refused. |
| New test tier | `unit`, always | Every task here adds a component-level claim, and none of them relabels an existing test. `regression` has one member, `MultiSolve` (`tests/CMakeLists.txt:61-63`). Each task's exit criterion names the stems it must pass, so the label does not decide what a task is held to. |
| Backends | every exit criterion passes on SERIAL at np 1-6 **and** on HIP at np 1-4 | Production runs on HIP. A node has four APUs, and HIP runs one APU per rank (`fix-hang-rebalance.md` H0a). A figure a task *records* is taken from the SERIAL binary unless the task says otherwise; the HIP arm is pass/fail, read against `fix-hang-rebalance.md` H0c's baseline (**R11**). |
| New test registration | append the stem to `UNIT_MPI_TESTS` (`tests/CMakeLists.txt:46-57`) | The macro call below it applies the label and the 1-6 rank sweep, so a stem added to the list is registered everywhere for free (HIP at np 1-4, per `fix-hang-rebalance.md` H0a). |
| Fixture determinism | report per `(nprocs, rank)`, never a mean, and state reproducibility across two runs | The tree/partition path is run-to-run nondeterministic at np $\ge$ 3; see **R6**. A single draw is not the number. |
| Formatting | never run clang-format | `CLAUDE.md`: "Do not clang format." Write in the style of the surrounding code. |
| Comments | state units, signs and ranges on the declaration | Every bound and offset here is in units of a half-width at some depth, and which depth is exactly what goes wrong silently. |

### Test naming, and how to run only what a task needs

Stated once here; every task's exit criterion names **stems** and relies on this
section for the rest. Building and running the whole suite is never required by
any task below.

`Canopy_add_tests` (`cmake/test_harness/test_harness.cmake:87-171`) generates one
executable per (stem, backend) and, for an MPI stem, one ctest entry per rank
count. Every exit criterion below runs two backends:

| | build target | ctest entry |
| --- | --- | --- |
| MPI stem, SERIAL | `Canopy_Test_<Stem>_MPI_SERIAL` | `Canopy_Test_<Stem>_MPI_SERIAL_np_<N>`, `N` in 1-6 |
| MPI stem, HIP | `Canopy_Test_<Stem>_MPI_HIP` | `Canopy_Test_<Stem>_MPI_HIP_np_<N>`, `N` in 1-4 |
| non-MPI stem | `Canopy_Test_<Stem>_SERIAL`, `Canopy_Test_<Stem>_HIP` | the same names |

HIP registers at np 1-4 with one APU per rank only once
`fix-hang-rebalance.md` H0a is **DONE**; every task below
reach H0c through T1, whose HIP arm depends on it. Run each HIP `ctest` line with the HIP environment
set on that command only (`fix-hang-rebalance.md`, Conventions, "HIP
environment"). Run every `ctest` line through `canopy_ctest` with the same
regex. That gives each entry a timeout of 1.75x its measured SERIAL runtime
instead of 300 s (`fix-hang-rebalance.md` H0b and Conventions, "Time
budget"). The code blocks below show the bare `ctest`. A task that adds a case
to a stem, or changes its runtime, re-calibrates that stem's rows in
`scripts/tuolumne/serial_runtimes.tsv` in the same change. That includes a
knob turned on in an exit criterion (A2, B2), which runs under its own budget
config. So a task that names
stems `DownwardSweep` and `MultiSolve` is run by building exactly four targets
and matching exactly twenty ctest entries:

```bash
make -j Canopy_Test_DownwardSweep_MPI_SERIAL Canopy_Test_MultiSolve_MPI_SERIAL \
       Canopy_Test_DownwardSweep_MPI_HIP Canopy_Test_MultiSolve_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_(DownwardSweep|MultiSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(DownwardSweep|MultiSolve)_MPI_HIP_np_[1-4]$'
```

The anchored regex matters: unanchored, `-R Canopy_Test_CartesianTaylor` also
matches `CartesianTaylorSolve`, and `-R MultiSolve` also matches the
`SolveFusedM2L` cases, which live in the `MultiSolve` stem but are a separate
gtest suite.

**Tighter still, while iterating:** the generated binaries take gtest flags, so a
single case can be run directly without ctest —

```bash
mpirun -n 4 ./tests/Canopy_Test_CartesianTaylorSolve_MPI_SERIAL \
    --gtest_filter=CartesianTaylorSolve.matchesDirectSumThetaRef
```

Use this for the edit-compile-check loop and the ctest lines for the exit
criterion; a gtest filter silently matching nothing exits 0, so it is not a
criterion.

**Which gtest suites live in which stem**, where it is not the stem's own name:
the `MultiSolve` stem carries both `MultiSolve.*` and `SolveFusedM2L.*`
(`tests/tstMultiSolve.hpp:560-1079`); the `LaplaceSolve` stem carries
`LaplaceSolve.*` only; the `CartesianTaylorSolve` stem carries
`CartesianTaylorSolve.*` only.

### Deliberate deviations

- **A2 balances the tree rather than restricting the traversal.** The traversal
  is where the large depth difference becomes visible, so restricting it there
  looks cheaper. It is not available: the traversal splits cells by looking up
  child keys in `_cell_map` (`src/Canopy_CommunicationPlan.hpp:643-657`), and a
  leaf has no children to look up. Splitting one virtually would mean
  synthesizing cells, their multipoles and their partition ownership inside the
  traversal, duplicating the tree builder. The alternative — accepting the pair
  at a coarser level — changes which approximation is used and so changes
  accuracy. Balancing the tree is the one option that leaves both the MAC and
  the traversal untouched.
- **B1 removes `dd` from the key rather than reducing `m2l_key_dd_max`.**
  Lowering the bound would refuse *more* pairs; removing the field refuses none
  and deletes duplicate columns. The two are opposite directions, and the field
  is removable only because the basis's operator provably ignores it
  (`src/Canopy_CartesianTaylorBasis.hpp:972-979`).
- **B1 keeps `m2l_key_dd_max` as a trait and keeps the sweep's `dd` guard.**
  Once `dd` is out of `CartesianTaylorBasis`'s key, that basis has no
  representability reason to bound `dd` — but `LaplaceKernel` does: its own
  bound is a precision claim about a $2^{\,j|\texttt{dd}|}$ residual factor in
  its scale-normalized operator, and is tightened to 4 for `float`
  (`src/Canopy_LaplaceKernel.hpp:689-702`). The guard is therefore still needed
  and still basis-driven; what changes is only that one basis can now set the
  bound by precision alone rather than inheriting a key-space limit.
- **B2 quantizes the root half-width rather than making the cache key physical.**
  Keying the cache on the floating-point width would make cache identity depend
  on bit-exact equality of a derived quantity, which is the kind of thing that
  works until a reduction order changes. Quantizing the root width to a power of
  two makes the *set* of per-depth widths recur exactly across rebuilds, so an
  integer exponent is a faithful key.
- **The three mechanism chains are independent and may land in any order.**
  They share T1's fixture and V1's sharpened bounds, and nothing else — no task
  in one chain depends on a task in another, except that A3 waits on C1 for the
  cost ratio its decision is arithmetic over. A reader looking for a single
  combined remedy will not find one here, deliberately: chain A removes
  refusals, chain B removes duplicate and drifting columns, and either is worth
  having without the other.

## Current state

**The instrumentation needed to measure all of this already exists**, is
profiling-gated where it costs anything, and returns $-1$ when not compiled in
(`CANOPY_ENABLE_PROFILING` undefined) — never $0$, since $0$ is a legal count
for every one of these:

| accessor | file:line | what it reports |
| --- | --- | --- |
| `m2l_n_fallback_pairs_range_guard()` | `src/Canopy_DownwardSweep.hpp:1113-1116` | pairs refused by the classify pass's range guard, **in pairs** |
| `m2l_n_fallback_pairs_count_cap()` | `:1120-1123` | pairs refused a column by the merge's count cap |
| `m2l_n_fallback_pairs_depth_dropped()` | `:1130-1133` | refused pairs placed in neither an operator column nor the fallback table. **Must be 0** |
| `total_fallback_pair_count()` | `:1012-1018` | all fallback pairs; the first two sum to it exactly |
| `m2l_n_unique_ops()` | `:1068` | columns admitted |
| `m2l_n_demanded_ops()` | `:1055` | distinct canonical keys the merge saw, admitted or not |
| `m2l_realized_keys()` | `:1060-1063` | the admitted key list itself, by const reference |
| `m2l_cells_at_depth()` | `:1151-1158` | occupied cell count per depth, **ungated**, by value |
| `m2l_op_keys_built_count()` | `:435` | cumulative columns ever built; its per-build increment is the cache-retention measure |
| `m2l_effective_op_cap()` | `:368-374` | the column cap in force |

So no new accessor is needed to measure refusals, key duplication, cache
retention or occupied depth.

**A non-uniform fixture already exists, and it already produces refusals.**
`testMultiStepGravity` takes a `clustered` flag — 80 % of particles from a tight
Gaussian blob in one corner, 20 % uniform (`tests/tstMultiSolve.hpp:159-171`,
`:211-233`) — and `MultiSolve.M2L_BinEdge_Fallback`
(`tests/tstMultiSolve.hpp:734-748`) drives it at `ncrit = 8`, `max_depth = 8`
and `mac_theta = 0.3`, asserting that the fallback population is non-zero
(`:751-762`). That test is the `regression` label's only member
(`tests/CMakeLists.txt:61-63`).

**Every refusal on that fixture is an offset refusal.** The range guard carries
all of them — the count-cap and dropped counters read 0 on every reading — and
within the guard it is the offset bound that fires rather than the `dd` bound:
that tree's occupancy never passes depth 6, so `|dd|` cannot exceed
`LaplaceKernel`'s `m2l_key_dd_max` of 6 and that half of the guard is
geometrically unreachable. The bound in force is `M2L_KEY_OFFSET_MAX = 32`
(`src/Canopy_DownwardSweep.hpp:570`), in half-widths at the deeper cell's depth,
and the test's comments and its failure message name it. The uniform fixtures
remain the control: on those the range-guard counter reads **0**, and the only
refusals are count-cap refusals driven by a deliberately tiny cap. The fixture
that carries a large *depth* difference is a separate two-scale distribution in
`tests/tstDownwardSweep.hpp`; see T1.

Also true now:

- **No balancing of any kind exists.** `TreeBuilder` refines a cell iff its
  global particle count exceeds `ncrit` and its depth is below `max_depth`
  (`src/Canopy_TreeBuilder.hpp:769-776`), and nothing anywhere consults a
  neighbour's depth. There is no post-pass, no flag and no partial form of it.
- **`max_depth` is capped at 19 by Morton-key storage**, which `TreeBuilder`'s
  constructor enforces by throwing (`src/Canopy_TreeBuilder.hpp:177-182`). Any
  smaller limit a caller runs at is that caller's choice, not a library limit.
- **`CartesianTaylorBasis` declares `key_needs_level = true`** (`:485`) and its
  `canonicalize_key` is the identity (`:497-501`). `LaplaceKernel` declares
  `false` (`:721`) and zeroes `max_d` (`:742-746`).
- **`m2l_key_dd_max` is 6 for `CartesianTaylorBasis`** (`:469`) and its comment
  states that for this basis the number "merely bounds the key space" and "is
  not a precision claim about this basis" — chosen so the refused set matches
  what other bases see. For `LaplaceKernel` it is 6 for `double` and 4 for
  `float`, and there it *is* a precision bound (`:689-702`).
- **`expectKeyTraitsAgree` asserts that `canonicalize_key` does NOT alter `dd`**
  for any basis (`tests/tstFarFieldContract.hpp:934-937`). B1 must change that
  function; it is not merely extended by it.
- **`set_root_half_width` clears the whole operator cache** when
  `key_needs_level` is true, and deliberately does not when it is false
  (`src/Canopy_DownwardSweep.hpp:435-460`). The root half-width is the largest
  half-extent of the global particle bounding box
  (`src/Canopy_TreeBuilder.hpp:608-635`) and is not quantized.
- **There is no `FmmConfig` knob for tree balance.**
- **The `regression`-labeled stem exercises `LaplaceKernel` only.**
  `REGRESSION_MPI_TESTS` is the single stem `MultiSolve`
  (`tests/CMakeLists.txt:61-63`), and `tstMultiSolve.hpp` instantiates
  `Canopy::Solver<..., Scalar, P, 1>` (`:744-745`) without a far-field
  argument, so it takes the default `FarField = LaplaceKernel`
  (`src/Canopy_Solver.hpp:146-148`). **No `regression`-labeled test
  instantiates `CartesianTaylorBasis.`** Its only direct-sum coverage is
  `CartesianTaylorSolve.matchesDirectSumThetaRef` and `...ThetaCanopy`
  (`tests/tstCartesianTaylorSolve.hpp:976-1006`), both in the `unit` tier, and
  that is where its coverage stays. So a chain-B change is
  checked by the `unit` tier alone: B1's and B2's exit criteria are the only
  thing standing between a broken `CartesianTaylorBasis` and a green run of
  whatever stems some other task happens to name.
- **The two operator-cache drift cases assert no accuracy at all.**
  `CartesianTaylorSolve.operatorCacheAcrossDriftThetaCanopy` and
  `...ThetaRef` (`tests/tstCartesianTaylorSolve.hpp:1040-1062`) state in place
  that they "MAKE NO ACCURACY CLAIM AND ASSERT NO DEVIATION"; their only
  assertions are that the far field was live and that at least one operator was
  built. They also run a **longer** trajectory than the gating arms, precisely
  so the bounding box drifts. So the configuration in which cache reuse across
  drift actually happens is the one with no correctness check on it — which is
  **R9**, and is why B2 adds one.

### Measured on a downstream application's configuration

These are the figures the chains are sized from. The configuration:
`CartesianTaylorBasis` at order 3 (3200 B per column), `ncrit = 8`,
`mac_theta = 0.3`, `max_depth = 10`, softening positive, 2562 particles on a
closed 2-D manifold embedded in 3D, sampled at 81 states of a moving-particle
trajectory, at a column cap of 65536 — high enough that **no key was ever
refused for a count-cap reason**, so every refusal below is a range-guard
refusal.

- **The range guard accounts for 100 % of fallback**, at 1 and at 4 ranks:
  215 302 pairs of 215 302 at np1 and 215 742 of 215 742 at np4, with the
  count-cap counter reading 0 at every one of 405 sampled (state, rank) rows and
  the dropped counter reading 0 everywhere.
- **Fallback is 1.58 % of all M2L pairs** over the series and 4.2 % at the worst
  single state, against a per-state total of 129 152 – 197 902 M2L pairs.
- **It tracks occupied depth, not particle count.** Per state, at np1:

  | occupied depths | states | states with non-zero fallback | mean fallback pairs | max |
  | --- | --- | --- | --- | --- |
  | 5 | 7 | **0** | 0 | 0 |
  | 6 | 23 | 20 | 859 | 1 920 |
  | 7 | 50 | 50 | 3 852 | 6 884 |
  | 8 | 1 | 1 | 2 926 | 2 926 |

  A shallower configuration of the same application — 642 particles, occupied
  depths 4-6 — produced **zero** fallback in all 243 rows measured.
- **It is essentially rank-count-independent**: the peak is 6 884 pairs at np1
  against 6 832 at np4, at the same state, across a 4x different partition. A
  per-rank budget cannot behave that way; a geometric limit on the tree must.
- **The cache retained nothing**: the per-build increment of
  `m2l_op_keys_built_count()` equalled `m2l_n_unique_ops()` in **810 of 810**
  rows, so the whole admitted table — up to 37 678 columns, 115 MiB at 3200 B
  per column — was rebuilt at every evaluation.
- **Raising the column cap did not help, and bounded how much it could.**
  Doubling it from 32768 to 65536 removed 44.3 % of fallback *pairs* at np1 and
  moved the count of states with non-zero fallback by **zero** — 71 of 81 both
  before and after, at both rank counts.

### Not read

- **The partitioner.** `fix-hang-rebalance.md` H2 replaces it: a distributed
  ParMETIS graph partition over every non-shared cell, with one balance
  constraint per band of depths. A2 changes the cell set that partition
  consumes, so **A2 is the task here that first opens it**, and A1 must not.
  A1's cost model is a cell count, which `m2l_cells_at_depth()` and
  `TreeBuilder`'s own cell list answer without it. A balanced tree also changes
  the band populations H2's constraints are drawn from, so A2 records the
  per-band imbalance before and after.
- **The upward sweep's coefficient formation.**
  `src/Canopy_UpwardSweep.hpp` was read only far enough to confirm where
  `build_aux_tables` is called. No task here changes coefficient scaling, so it
  is not on any path; B2 changes only how a key names a width, not what any
  coefficient means.
- **`m2l_translate`'s performance.** The fallback's per-pair cost relative to
  the GEMM path is not measured anywhere, so "1.58 % of pairs" is a pair count
  and not a time share. **C1 measures it**, because whether chain A is worth its
  cell-count cost depends on it.

## Progress log

`tasks/tree-opt-progress-log.md` holds the reasoning, the measured numbers and
the bugs that only running revealed, as one section per task ID. **Read it
before implementing a task, changing a signature, or reopening a question this
document states flatly** — this file says what is true, the log says how it got
that way and what was already tried.

## Task sequence

### T1 — A fixture whose tree has shallow leaves beside deep subtrees — **REOPENED**

SERIAL arm **met**; HIP arm **not started**. The fixture and its assertions are
in place. What remains is running the exit criterion below on HIP, and fixing
whatever fails there that H0c did not already record.

**Depends on:** none for the SERIAL arm. HIP arm: `fix-hang-rebalance.md` H0c
**DONE** (HIP registration and baseline) and H2 **DONE**, both arms, since a
`MultiSolve` HIP pass that hangs at np 3 cannot be read.
**Fill in:** `tests/tstMultiSolve.hpp` (the per-reason report on the existing
clustered fixture); `tests/tstDownwardSweep.hpp` (the reusable fixture and the
new case). Both stems are already registered
(`tests/CMakeLists.txt:46-57`, `:61-63`), so no CMake change is needed.
**Reference:** the existing clustered distribution and the test that drives it
(`tests/tstMultiSolve.hpp:158-167`, `:205-220`, `:686-708`) — **the starting
point, not a model to copy**; `with_laplace_solve`
(`tests/tstLaplaceSolve.hpp:793-800`) for the shape of a fixture that configures
a solve and hands an outcome plus the sweep to a callback; the refusal accessors
listed under [Current state](#current-state).
**Do:**
1. **First, report the per-reason breakdown on the existing clustered fixture**,
   unchanged, at its own `ncrit = 8`, `max_depth = 8`, `mac_theta = 0.3`. This
   answers which guard its non-zero fallback comes from and costs no new
   fixture. Record the answer in the log before writing anything else — it
   decides whether the rest of this task is a new fixture or a relocation of
   this one. Correct that test's stale `M2L_BIN_RANGE` rationale in the same
   change, naming the bound that is actually in force.
2. Then make the geometry reachable from the sweep-level tests, which is where
   B0 and A1 read their numbers: either lift the clustered draw into a shared
   helper both files use, or reproduce it in `tstDownwardSweep.hpp`. If step 1
   showed the clustered draw produces **only** offset-guard refusals and no
   large depth difference, a **two-scale** distribution is needed instead — a
   dense cluster forcing refinement several levels below `ncrit`, plus a sparse
   halo whose cells reach `ncrit` immediately and become shallow leaves,
   positioned so the two are spatial neighbours. A single Gaussian does not
   guarantee it; the requirement is a **large density ratio over a short
   distance**.
3. Assert the tree actually has the geometry, from `m2l_cells_at_depth()`:
   at least three occupied depths, and occupancy at both a shallow and a deep
   depth. **This assertion is the fixture's contract** — without it a later
   change to `ncrit` or to the distribution could flatten the tree and every
   measurement built on it would quietly become a measurement of nothing.
4. Assert `m2l_n_fallback_pairs_range_guard() > 0` under
   `CANOPY_ENABLE_PROFILING`, at a column cap left at its default so no refusal
   can be a count-cap refusal. Assert `m2l_n_fallback_pairs_count_cap() == 0`
   in the same case, which is what makes the first assertion unambiguous.
5. Assert `m2l_n_fallback_pairs_depth_dropped() == 0`.
6. Assert the sum identity
   `range_guard + count_cap == total_fallback_pair_count()`.
7. In the `#else` branch, assert all three accessors read $-1$ and **skip** the
   sum identity rather than evaluating it — $-1 + -1$ against a real total is a
   claim about nothing, and a check that "passed" on sentinels would be a
   vacuous pass.
8. Print one line per `(nprocs, rank)` carrying the three counters, the
   fallback total, the per-depth occupancy and the realized key count, so later
   tasks read their numbers out of this fixture's log rather than re-deriving
   them.

**HIP arm.** `fix-hang-rebalance.md` H2 changes the partition at every
np ≥ 2, so the per-`(nprocs, rank)` figures the SERIAL arm recorded above np 1
are stale. This arm records post-H2 figures for both backends. Build and run
the exit criterion's HIP lines, and the failure
direction in `build-tuolumne-noprof/` on HIP as well. Record in the log, per
`(nprocs, rank)` at np 1-4, the line step 8 prints, from two runs, and whether
they agree with each other and with SERIAL at the same np. A HIP case that
fails where SERIAL passes, and that H0c did not record, is a defect in T1's
fixture or in what it exercises: fix it here. A failure H0c did record is
carried under the same rule as the `MultiSolve` carve-out below.

**Met (SERIAL).** Step 1 settled the open question first, on the clustered fixture
unchanged at its own `ncrit = 8`, `max_depth = 8`, `mac_theta = 0.3`: **every
refusal there is a range-guard refusal**, with `count_cap == 0` and
`depth_dropped == 0` on all 42 `(nprocs, rank, step)` readings and
`range_guard == total` on every one of them. Within the range guard they are
**offset** refusals specifically — that tree's occupancy never passes depth 6,
so `|dd|` cannot exceed `LaplaceKernel`'s `m2l_key_dd_max` of 6 and that half
of the guard is geometrically unreachable. The four stale `M2L_BIN_RANGE = 3`
comment sites and the `EXPECT_GT` failure message now name
`M2L_KEY_OFFSET_MAX = 32` (half-widths at the deeper cell's depth) and
`KernelType::m2l_key_dd_max`, and the message points at the per-reason lines
instead of anticipating a condition that cannot occur.

That answer put step 2 on the "new two-scale distribution" branch of this
task's own decision rule, so `DownwardSweepTest::TwoScaleFixture<MS, ES,
FarField = Kernel>` was built in `tests/tstDownwardSweep.hpp` — a cube of
half-width 0.01 holding 87.5 % of a **global** 1200-particle set beside a
uniform halo, templated on the far-field type so B0 can read the same tree for
both `CartesianTaylorBasis` and `LaplaceKernel`. Verified on it, at ranks 1-6
over **two separate runs** of the same binary: `deepest == 8` and the
shallowest *leaf* depth 1 or 2 on every rank of every rank count, an
occupied-depth span of 6-7 levels; `range_guard > 0` with `count_cap == 0`,
`depth_dropped == 0` and `range_guard + count_cap == total_fallback_pair_count()`
exactly, at a column cap left at its default (32768, measured) so no refusal
can be a budget refusal. The geometry contract is asserted on the depth at
which refinement *stops*, not on the shallowest occupied depth — depth 0 is
occupied on every tree, uniform ones included, so the latter is a vacuous pass
of exactly the kind **R7** describes. The failure direction was run in a
separate `build-tuolumne-noprof/`: all six rank counts pass with all three
counters reading **-1** and the sum identity reported SKIPPED, six of six.

**Two qualifications, both recorded in the log.** First, the two runs agree
field-for-field at np 1, 2, 3, 5 and 6 but **disagree at np 4** — a per-rank
`range_guard` moved by up to 25 % — which is **R6** and sets the spread any
later before/after comparison on this fixture must be read against. Second,
**the `MultiSolve` arm of the criterion below does not pass, and did not before
T1**: seven tests fail the `1e-8` multi-step check at every rank count, all
seven already recorded in README "Known Issues" with error values this run
reproduces to every digit. T1 did not touch that tolerance — sharpening those
bounds is V1's task, and loosening one to green a gate would change what
**DONE** means invisibly. `M2L_BinEdge_Fallback`'s own fallback assertion
passes; it fails only on the shared accuracy check inside
`testMultiStepGravity`. The `DownwardSweep` arm passes at ranks 1-6 in both
runs. See `tree-opt-progress-log.md` `## T1`.

**Exit criterion:** stems `MultiSolve`, `DownwardSweep` pass on SERIAL at ranks
1-6 and on HIP at ranks 1-4 —

```bash
make -j Canopy_Test_MultiSolve_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_SERIAL \
       Canopy_Test_MultiSolve_MPI_HIP Canopy_Test_DownwardSweep_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_(MultiSolve|DownwardSweep)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(MultiSolve|DownwardSweep)_MPI_HIP_np_[1-4]$'
```

— the log records which guard the existing clustered fixture's refusals come
from, and the new sweep-level case prints `range_guard > 0` with
`count_cap == 0` on at least one rank at every rank count.
Failure direction: a build configured `-DCanopy_ENABLE_PROFILING=OFF`
(`build-tuolumne-noprof/`) passes the same case on both backends with all three
counters reading $-1$, not $0$, and the sum identity reported as skipped.

---

### V1 — Sharpen the checks these chains will be verified against — **NOT STARTED**

**Resume from step 1.** A first pass stopped at step 1 because
`MultiSolve.AutoRebalance`'s trajectory deviation exceeds $\theta^{P+1}$ at
np 5-6 (`tree-opt-progress-log.md` section V1). `fix-hang-rebalance.md` E1
attributed that excess to trajectory amplification at unsoftened close
encounters: the per-step far-field error is at most `9.5e-7` there
(`fix-hang-rebalance-progress-log.md` section E1). Step 1's derivation and stop
clause below now say so. The partition changed since the first pass
(`fix-hang-rebalance.md` H2), so re-measure every figure before pinning a bound.
Steps 2, 4 and 5 are in place from the first pass; step 3 was measured but not
applied.

**Depends on:** T1 **DONE**, both arms; `fix-hang-rebalance.md` E2 **DONE**.
**Fill in:** `tests/tstMultiSolve.hpp` (the six `fmm_tolerance` call sites,
`SolveFusedM2L.matchesPriorReference`'s bounds, `SolveFusedM2L.FP32_smokeTest`,
and the clustered test's per-reason assertion);
`tests/tstCartesianTaylorSolve.hpp` (the two direct-sum arms' bounds, and the
two drift cases); `README.md` (the disabled FP32 case, and the Known Issues
entry "Six `MultiSolve` tests fail the `1e-8` multi-step check", removed or
rewritten to what still fails); `scripts/tuolumne/serial_runtimes.tsv`
(`MultiSolve`'s `default` rows, re-calibrated because AutoRebalance's
unconditional probe changes the stem's runtime — `fix-hang-rebalance.md`
Conventions, "Budget rows"; `run_ctest_h0b.flux calibrate` with
`CANOPY_CAL_REGEX`).
**Reference:** the measured deviations and bounds recorded in place — the six
`fmm_tolerance` call sites at `1.0e-8` (`tests/tstMultiSolve.hpp:955`, `:972`,
`:990`, `:1008`, `:1045`, `:1094`), applied to `max_pos_rel` and `max_vel_rel` at
`:929-933`; `SolveFusedM2L.matchesPriorReference` (`:1320-1350`) at `5.0e-2` /
`1.0e-1` (`:1349-1350`); `CTS_DEV_TOL_THETA_CANOPY = 3.74e-02`
(`tests/tstCartesianTaylorSolve.hpp:365`), which the rationale block above it
(`:334-363`) records as **2x the worst measured figure** — gradient
`1.8651556395e-02`, potential `9.9666798509e-04`, the six rank counts agreeing
to 15 significant figures; `CTS_DEV_TOL_THETA_REF`, the `1e-3` bar itself
(`:364`); the two arms that consume them (`:976`, `:998`); the drift cases'
explicit absence of any accuracy claim (`:1040-1062`).
**Do:**
1. **Re-derive the six `fmm_tolerance` bounds, one per call site.** `1.0e-8` has
   no derivation behind it, and the prose at `tests/tstMultiSolve.hpp:1032` still
   documents a prior value of `2e-2` that no call site passes; correct it in the
   same change. Each bound is the **tighter of two figures**, both written in
   the site's comment:
   - **derived:** the per-solve floor $\theta^{P+1}$ carried through that
     site's integration (`dt`, `nsteps`, `drift_multiplier`) per the
     derivation below. This sets the bound's form and order of magnitude;
   - **measured:** the worst deviation over **at least three runs** on each
     backend — SERIAL np 1-6 **and** HIP np 1-4 — times a margin no smaller
     than the observed run-to-run spread. HIP `[multisolve-dev]` figures are
     not bit-stable even at np 1, and **R6** moves the partition at
     np $\ge$ 3, so a margin below the spread is a bound drawn on noise.

   Per call site and not one shared value: the parameter is already per-site,
   the configurations differ materially — `nsteps` 2/4/5/8,
   `drift_multiplier` 1/2/5/50, uniform against clustered — and the measured
   errors span orders of magnitude, so a single bound at the worst observed
   would pass a regression at the best-behaved site.

   **The derivation each bound is read against.** The far-field truncation floor
   at `MultiSolveTest::P_ORDER = 8` (`:88`) and `get_test_mac_theta() = 0.5`
   (`:42-47`) is $\theta^{P+1} = 1.95 \times 10^{-3}$, a bound on the
   **per-solve** relative gradient error. `1.0e-8` is below what a
   finite-order FMM can deliver at that order and admissibility, which is why
   these sites fail — not because the method is wrong. The trajectory check
   compares the end state of two integrations (`v += dt * g`, then
   `r += dt * drift_multiplier * v`) against a brute-force shadow. While the
   trajectories stay close, a gradient error reaches `max_vel_rel` no larger
   than it entered. It is **not** damped when particles approach unsoftened
   (`cfg.softening = 0.0`, `:402`): a close encounter amplifies a small force
   difference into a large velocity difference. E1 of `fix-hang-rebalance.md`
   measured this on `AutoRebalance`
   (`fix-hang-rebalance-progress-log.md` section E1):
   - the per-step field-scale error, FMM against brute force at the FMM's own
     positions, is at most `9.5e-7` for AutoRebalance and `1.52e-6` for any
     case, at SERIAL np 1-6 and HIP np 1-4;
   - SERIAL np 1 at N = 1200 (`CANOPY_MULTISOLVE_NPP=1200`, no partition)
     reaches `max_vel_rel = 2.67e-3`, about 7000x the `3.8e-7` and `1.1e-9`
     solve errors of steps 0-1, the only far-field error that run's trajectory
     sees;
   - at np 6, 10 of the 11 particles above the floor had a close encounter
     (separation below 0.1 of the mean spacing), against a 48% base rate;
   - θ = 0.7 raises the per-step error about 140x at np 1, which is why the
     first pass's `CANOPY_MAC_THETA` sweep moved the trajectory deviation: the
     dynamics amplify whatever far-field error goes in.

   So the excess is N-driven, not rank-driven. A trajectory bound at a site with
   unsoftened close encounters measures the dynamics, not the far field.
   `AutoRebalance` at np 5-6 is such a site, as is np 1 once N reaches 1200.

   **`AutoRebalance` gates on both the per-step probe and the trajectory.** The
   probe runs unconditionally at that site, not behind
   `CANOPY_MULTISOLVE_PROBE`, and an `EXPECT` holds every step's field-scale
   error under the tighter of $\theta^{P+1}$ and its measured worst times a
   margin, as above — that is the far-field gate. The trajectory `EXPECT`s keep a
   bound at the worst measured deviation times a margin, commented as a
   dynamics and complete-regression catch rather than a far-field claim. Any
   other site whose measured trajectory deviation exceeds its derived bound
   while its probe stays under the floor takes the same treatment, recorded in
   the log as dynamics with the probe's `n_excess` / `n_excess_close` figures.

   **Stop and report if the per-step probe exceeds the floor.** With
   `CANOPY_MULTISOLVE_PROBE=1`, `testMultiStepGravity` prints the per-step
   field-scale error on each `[multisolve-probe]` line. If it exceeds
   $\theta^{P+1}$ on any step, at any site, np or backend, the far field is
   wrong and the bound is not the defect. Record the finding in the log, report
   it, and stop — fixing it is its own sequence of tasks and must not be folded
   in here. A trajectory deviation above the floor does **not** stop this task
   when the site's probe stays under the floor on every step and its excess sits
   on close encounters (the probe's `end` line: `n_excess` against
   `n_excess_close`). Record it as dynamics, with those figures. The clause
   exists because a bound widened to cover a real far-field defect is
   indistinguishable in the diff from one derived correctly, which is the
   failure mode this whole task exists to close; the probe is what tells the
   two apart.
2. **Disable `SolveFusedM2L.FP32_smokeTest`, and leave it disabled.** It is the
   stem's only FP32 case, commented out at `tests/tstMultiSolve.hpp:1488-1540`,
   and fails at
   np 2-6 with a max relative gradient error of $\approx 0.277$ against its own
   `5.0e-2` budget — 5.5x over, not marginal, and reproducing on two platforms.
   Nothing in this task can move it: it is not an `fmm_tolerance` test, and its
   magnitude together with its rank-count dependence points at a multi-rank FP32
   accumulation defect whose triage is a separate sequence of tasks. Comment the
   case out, with a block naming the measured figure, the budget it is against,
   and the `README.md` entry that carries it; record in `README.md` that the
   FP32 case is disabled pending that investigation. **Do not re-justify the
   FP32 budget here** — a widened budget would retire the only signal that
   defect has.
3. **Bound each remaining loose check at its measured deviation plus a stated
   margin**, rather than at a round number chosen to be safe.
   `SolveFusedM2L.matchesPriorReference` is the clearest case: its own comment
   says `5.0e-2` exists to catch "a complete-regression bug", which is a
   different job from noticing a tree change that costs a few percent. Measure
   the actual deviation over **at least three runs** on each backend and set
   the bound at the worst observed times a margin you state in the comment,
   with the measured figures beside it.
4. **Confirm the `theta_canopy` bound rather than redoing it.**
   `CTS_DEV_TOL_THETA_CANOPY` is already a measured deviation times a stated
   margin, in exactly the form step 3 prescribes, at 2x the worst of the two
   fields. Verify that is still what the constant and its rationale block say,
   record that the step was satisfied by prior work, and leave the value alone
   unless you can state why a tighter margin is warranted — step 7 governs it.
   Leave the `theta_ref` arm's $10^{-3}$ bar alone: it is an accuracy *claim*
   about the method, not a regression bound, and tightening it would change what
   the test asserts.
5. **Assert the per-reason fallback breakdown where the fixture already
   produces it.** T1 reports it on the clustered test in the `MultiSolve`
   stem; turn that report into an assertion on the reason T1 identified, so a
   later change that
   silently moves refusals from one reason to the other fails rather than
   prints. Keep the sum identity and the `depth_dropped == 0` assertion, and
   keep both skipped under the $-1$ sentinel.
6. Record every bound this task moves in the log, old and new, with the runs
   the new one came from. A later session that finds one of these tests failing
   needs to know whether it is reading a real regression or a bound that was
   drawn too tightly here.
7. **Do not tighten a bound you cannot explain.** If a measured deviation is
   much smaller than the bound and you cannot say why the bound was set where
   it was, say so in the log and leave it; a bound tightened onto noise fails
   on an unrelated change and gets widened again by someone with less context.

**Additional information needed:** whether `matchesPriorReference`'s gradient
deviation is stable enough at np $\ge$ 3 to carry a tightened bound. Step 3's
three runs per backend answer it. If it moves between runs by more than the
margin, the bound is not tightened and stays as it is, with that recorded. This
rule governs only the tightening of a bound that already passes. It does not
apply to the six `fmm_tolerance` sites: they fail at `1.0e-8`, so their bounds
only move up, and step 1's margin absorbs the spread. The `theta_canopy` arm
needs no such measurement: its six rank counts agree to 15 significant figures,
recorded on the constant.

**Exit criterion:** stems `CartesianTaylorSolve`, `MultiSolve` pass on SERIAL at
ranks 1-6 and on HIP at ranks 1-4
with the re-derived bounds, three times in succession, with
`SolveFusedM2L.FP32_smokeTest` commented out per step 2 so that carve-out is
visible in the diff rather than hidden behind a gtest filter —

```bash
make -j Canopy_Test_CartesianTaylorSolve_MPI_SERIAL Canopy_Test_MultiSolve_MPI_SERIAL \
       Canopy_Test_CartesianTaylorSolve_MPI_HIP Canopy_Test_MultiSolve_MPI_HIP
for i in 1 2 3; do
  ctest --output-on-failure \
    -R '^Canopy_Test_(CartesianTaylorSolve|MultiSolve)_MPI_SERIAL_np_[1-6]$' || break
  ctest --output-on-failure \
    -R '^Canopy_Test_(CartesianTaylorSolve|MultiSolve)_MPI_HIP_np_[1-4]$' || break
done
```

— each `ctest` line run through `canopy_ctest` with the same regex
([Test naming](#test-naming-and-how-to-run-only-what-a-task-needs)). Every case
in both stems passes with no failure carried. That covers the six
`fmm_tolerance` sites against their re-derived bounds, `AutoRebalance`'s
per-step probe gate on every step, and the FP32 case absent from the binary.
The log records each moved bound with its derived and measured figures.
`README.md` records the disabled FP32 case and no longer lists the six sites as
failing. Failure
direction: inflate one measured deviation artificially (widen `mac_theta` on one
arm, say) and confirm the tightened bound **fails**, where the old bound would
have passed; revert, and record which bound was demonstrated this way. A
tightened bound that was never shown to fail is a bound nobody has tested.

---

### B0 — Measure how many admitted columns are `dd` duplicates — **NOT STARTED**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstDownwardSweep.hpp` (a case beside T1's, reusing its
fixture).
**Reference:** `m2l_realized_keys()`
(`src/Canopy_DownwardSweep.hpp:1060-1063`) returns the admitted key list by
const reference; `CartesianTaylorBasis`'s no-`dd`-dependence statement
(`src/Canopy_CartesianTaylorBasis.hpp:972-979`).
**Do:**
1. On T1's fixture, with `CartesianTaylorBasis`, count the admitted keys and
   count the **distinct** key tuples ignoring `dd`, i.e. distinct
   $(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$.
   The ratio is the duplicate factor B1 would remove.
2. Report it per `(nprocs, rank)`, with the absolute column counts and the bytes
   they cost at the basis's own `bytes_per_key`
   (`src/Canopy_CartesianTaylorBasis.hpp:506-509`) — never a literal.
3. Do the same count for `LaplaceKernel` on the same fixture and report it
   beside, as a control: that basis's operator **does** depend on `dd`, so its
   duplicate factor is the figure B1 must **not** claim for it.
4. Assert only that the distinct-tuple count is **at most** the key count, which
   is true by construction and is a self-check on the counting. **Assert no
   threshold on the ratio** — it is the measurement, and a case that asserted a
   value would need retuning every time the fixture moved.

**Additional information needed:** whether the duplicate factor is large enough
to justify B1 at all. If it is $1.0$ — every admitted key already has a distinct
$(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$ — then B1 removes
nothing on this fixture and its value rests entirely on a different tree shape,
which must be said in the log before B1 is started.

**Exit criterion:** stem `DownwardSweep` passes on SERIAL at ranks 1-6 and on
HIP at ranks 1-4 —

```bash
make -j Canopy_Test_DownwardSweep_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_HIP_np_[1-4]$'
```

— and the log records the duplicate factor per `(nprocs, rank)` for both bases,
with the absolute column counts and byte figures. Failure direction: the `-DCanopy_ENABLE_PROFILING=OFF`
build passes the same case, since `m2l_realized_keys()` is ungated — a $-1$
anywhere in this case's output would be this case reading the wrong accessor.

---

### A1 — Measure the depth imbalance and the cost of removing it — **NOT STARTED**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstTreeBuilder.hpp` (a measurement case). No `src/` change.
**Reference:** `TreeBuilder`'s cell list and its refinement decision
(`src/Canopy_TreeBuilder.hpp:769-776`); `m2l_cells_at_depth()` for the
per-depth occupancy.
**Do:**
1. On T1's distribution, build the tree and compute, over all leaf pairs that
   are spatial neighbours, the distribution of the **level difference** —
   minimum, maximum, and the count exceeding 1. This is the imbalance A2 removes
   and is currently unquantified.
2. Compute the **cost model without implementing balancing**: how many cells a
   2:1 balance would add, by counting the leaves that would have to be refined
   and the cells their refinement implies, transitively. State it as a
   multiplier on the current cell count.
3. Report both per `(nprocs, rank)`, and state whether two separate runs
   reproduce them (**R6**).
4. Record the maximum level difference actually observed, because that is the
   number that says whether the range guard's $|\texttt{dd}| \le 6$ is exceeded
   by a little or by a lot — and therefore whether a `max_level_delta` of 1 is
   needed or whether a looser bound suffices.

**Additional information needed:** the cell-count multiplier. A 2:1 balance on a
2-D manifold embedded in 3D can be cheap or can be a large constant factor
depending on how the surface folds, and **A2 must not be started until this
number is in the log** — it is the input to A2's decision about whether to
balance fully or only against the deepest neighbour.

**Exit criterion:** stem `TreeBuilder` passes on SERIAL at ranks 1-6 and on HIP
at ranks 1-4 —

```bash
make -j Canopy_Test_TreeBuilder_MPI_SERIAL Canopy_Test_TreeBuilder_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_TreeBuilder_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_TreeBuilder_MPI_HIP_np_[1-4]$'
```

— and the log records, per `(nprocs, rank)`, the neighbour
level-difference distribution, the count exceeding 1, and the implied cell-count
multiplier, from two separate runs. Failure direction: the case asserts the
level-difference maximum is **at least 2** on T1's distribution — if it is 1,
the fixture is already balanced and is the wrong fixture for chain A, which is a
finding about T1 and must fail here rather than silently making A2 untestable.

---

### C1 — Depth and fallback-cost headroom against problem size — **NOT STARTED**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstTreeBuilder.hpp` (the depth-scaling case);
`tests/tstCartesianTaylorSolve.hpp` (the path-equivalence and timing case, which
needs a solve and so belongs beside `with_cartesian_taylor_solve` rather than in
the tree builder's tests). No `src/` change.
**Reference:** the Morton depth limit (`src/Canopy_TreeBuilder.hpp:177-182`);
`m2l_cells_at_depth()`.
**Do:**
1. Build T1's two-scale distribution at a geometric sweep of particle counts and
   record, at each: the deepest occupied depth, the **count of occupied depths**,
   and whether any cell hit `max_depth` and so was made a leaf by the depth
   limit rather than by `ncrit` (`src/Canopy_TreeBuilder.hpp:769`). The second
   case is a silent accuracy change, not an error, and nothing currently reports
   it.
2. Fit the growth of occupied-depth count against particle count, and state the
   particle count at which the deepest occupied depth reaches **19**, the Morton
   limit. For points distributed on a 2-D manifold the leaf count at depth $d$
   grows as $4^{d}$ rather than $8^{d}$, so the required depth is
   $\approx \log_4(N/\texttt{ncrit})$ — state the measured exponent rather than
   assuming that one.
3. Measure the **per-pair cost of the fallback path relative to the GEMM path**
   on T1's fixture, as a time ratio: drive one solve at a column cap of 0 so
   every pair takes `m2l_translate`, and one at the default cap, and report the
   M2L phase time of each. Without this ratio a pair-count fraction cannot be
   turned into a cost and chain A's value cannot be judged.
4. **Assert that the two M2L paths agree.** The pair of solves in step 3 —
   one at a column cap of 0 so every pair takes `m2l_translate`, one at the
   default cap so almost none does — differ only in which path evaluates the
   far field, so comparing their fields costs one extra comparison and closes
   the gap that chains A and B both rest on. Nothing in the suite asserts this
   today: `MultiSolve.M2L_BinEdge_Fallback` asserts the fallback is
   *exercised* (`tests/tstMultiSolve.hpp:751-762`), not that it produces the
   same answer as the tabulated path. **Chain A's entire purpose is to move
   pairs between those two paths**, so without this assertion a chain-A change
   that silently broke one of them would present as an accuracy shift with no
   indication of which path was at fault. Bound the deviation at the level
   measured here rather than at a guessed constant; the two paths evaluate the
   same mathematics and should agree far more tightly than either agrees with
   the direct sum.
5. Add a loud report — not an assertion — when a cell is made a leaf by
   `max_depth` rather than by `ncrit`.

**Exit criterion:** stems `TreeBuilder`, `CartesianTaylorSolve` pass on SERIAL
at ranks 1-6 and on HIP at ranks 1-4 —

```bash
make -j Canopy_Test_TreeBuilder_MPI_SERIAL Canopy_Test_CartesianTaylorSolve_MPI_SERIAL \
       Canopy_Test_TreeBuilder_MPI_HIP Canopy_Test_CartesianTaylorSolve_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|CartesianTaylorSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|CartesianTaylorSolve)_MPI_HIP_np_[1-4]$'
```

— the new case asserts the cap-0 and default-cap fields agree at the deviation
measured here, and the log records the occupied-depth growth against particle
count, the extrapolated particle count at which depth 19 is reached, and the
fallback-to-GEMM M2L phase-time ratio.
Failure direction: the case asserts that a tree built at `max_depth = 19` throws
nothing and that a tree requested at `max_depth = 20` **does** throw
`std::runtime_error` from `TreeBuilder`'s constructor, so the limit is pinned by
a test rather than by a comment.

---

### B1 — `key_needs_dd`, and a `CartesianTaylorBasis` key without `dd` — **NOT STARTED**

**Depends on:** B0 **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` (the new trait,
`canonicalize_key`); `src/Canopy_LaplaceKernel.hpp` (the new trait);
`tests/CanopyTest_MonopoleBasis.hpp` (the new trait, both bases it declares —
see `:131` and `:150`); `tests/tstFarFieldContract.hpp`
(`expectKeyTraitsAgree`).
**Reference:** `key_needs_level`'s declaration and its two readers
(`src/Canopy_LaplaceKernel.hpp:704-721`) as the exact pattern to mirror;
`CartesianTaylorBasis`'s no-`dd`-dependence statement
(`src/Canopy_CartesianTaylorBasis.hpp:972-979`); the current conformance
assertions (`tests/tstFarFieldContract.hpp:934-937`).
**Do:**
1. Add `static constexpr bool key_needs_dd` to every basis, beside
   `key_needs_level`, documented in the same style: what it declares, and that a
   basis whose operator ignores `dd` must zero it in `canonicalize_key` so the
   table does not hold one column per `dd` value.
2. Set it **`false` for `CartesianTaylorBasis`** and have its `canonicalize_key`
   zero `dd`. Set it **`true` for `LaplaceKernel`**, whose operator does depend
   on `dd`, and `true` for the test bases unless their own operator provably
   does not.
3. **Enumerate the callers of `canonicalize_key` before changing it.** There is
   one call site in `src/` — the classify pass's single hash site
   (`src/Canopy_DownwardSweep.hpp:1715-1716`) — plus the conformance test's two
   calls. Verify by search rather than by this list.
4. Extend `expectKeyTraitsAgree` to branch on `key_needs_dd` exactly as it
   branches on `key_needs_level`: when the trait is true, `dd` must survive
   canonicalization and two keys differing only in `dd` must stay distinct; when
   false, they must collapse. **The existing unconditional `EXPECT_EQ(a.dd,
   ca.dd)` must go** — it currently forbids what this task does.
5. Leave `m2l_key_dd_max` and the sweep's `dd` range guard in place and
   unchanged. `LaplaceKernel` still needs both.
6. Re-run B0's duplicate count and confirm the admitted column count fell by the
   factor B0 predicted.

**Exit criterion:** six stems, and **`CartesianTaylorSolve` is the authority
here while `MultiSolve` is not** — the `MultiSolve` stem instantiates
`LaplaceKernel` only, so it would pass with `CartesianTaylorBasis` wholly
broken, and `CartesianTaylorSolve`'s two direct-sum arms are the only checks
this change can fail on accuracy. `MultiSolve` and `LaplaceSolve` are here to
prove the other basis did **not** move.

```bash
make -j Canopy_Test_CartesianTaylorSolve_MPI_SERIAL Canopy_Test_FarFieldContract_MPI_SERIAL \
       Canopy_Test_DownwardSweep_MPI_SERIAL Canopy_Test_LaplaceSolve_MPI_SERIAL \
       Canopy_Test_MultiSolve_MPI_SERIAL Canopy_Test_CartesianTaylor_SERIAL \
       Canopy_Test_CartesianTaylorSolve_MPI_HIP Canopy_Test_FarFieldContract_MPI_HIP \
       Canopy_Test_DownwardSweep_MPI_HIP Canopy_Test_LaplaceSolve_MPI_HIP \
       Canopy_Test_MultiSolve_MPI_HIP Canopy_Test_CartesianTaylor_HIP
ctest --output-on-failure -R '^Canopy_Test_(CartesianTaylorSolve|FarFieldContract|DownwardSweep|LaplaceSolve|MultiSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(CartesianTaylorSolve|FarFieldContract|DownwardSweep|LaplaceSolve|MultiSolve)_MPI_HIP_np_[1-4]$'
ctest --output-on-failure -R '^Canopy_Test_CartesianTaylor_(SERIAL|HIP)$'
```

All pass on SERIAL at ranks 1-6 and on HIP at ranks 1-4, and B0's case reports
a duplicate factor of exactly **1.0** for
`CartesianTaylorBasis` — every admitted key now has a distinct
$(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$ — with its absolute
column count reduced by B0's measured factor and `LaplaceKernel`'s count
**unchanged**. Failure direction: a basis declaring `key_needs_dd = true` whose
`canonicalize_key` zeroes `dd`, and the converse, both fail
`expectKeyTraitsAgree` with a message naming the basis; verify by temporarily
mis-declaring one and seeing the named failure, then reverting.

---

### A2 — 2:1 tree balancing, configurable, default off — **NOT STARTED**

**Depends on:** A1 **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_TreeBuilder.hpp` (the balancing pass);
`src/Canopy_Solver.hpp` (`FmmConfig` member and its route to the builder);
`tests/tstTreeBuilder.hpp` (the cases). **This task first opens
`src/Canopy_TreePartitioner.hpp`** — see [Not read](#not-read).
**Reference:** the refinement loop and leaf decision
(`src/Canopy_TreeBuilder.hpp:657-790`); the existing post-pass that refines
leaves exceeding `ncrit` after a rebuild (`:1113-1130`) as the model for a pass
that runs over the finished cell set; `FmmConfig`'s existing knobs and their
route to the sweep (`src/Canopy_Solver.hpp`) for how a new one is threaded.
**Do:**
1. Add `FmmConfig::tree_balance_max_level_delta`, defaulting to the **off**
   value, and route it to `TreeBuilder`. Reject a value below 1 by throwing, as
   the sweep's other integer knobs do — a 0 would mean "every leaf at one
   depth", which is a request to abandon adaptivity and is far more likely a
   typo.
2. Implement the balancing pass as a **post-pass over the finished cell set**,
   not as a change to the refinement loop's leaf test. Repeatedly refine any
   leaf whose level is more than `max_level_delta` below a spatial neighbour's,
   until no such leaf remains; the refinement is transitive, so it iterates.
   Bound the iteration count and throw if the bound is hit rather than looping —
   a non-terminating balance presents as a hang, which is the most expensive
   failure to diagnose.
3. Do not let the pass refine past `max_depth`. A leaf at `max_depth` that is
   still unbalanced stays unbalanced; **report it loudly** rather than silently
   leaving the invariant broken, because every downstream claim about bounded
   `dd` rests on it.
4. Enumerate and update the consumers of the cell set the pass changes. At
   minimum the partitioner and the per-depth occupancy the sweep reads; verify
   by searching for consumers of `TreeBuilder`'s cell accessors rather than
   assuming this list is complete.
5. Add cases asserting: with the knob off, the cell set is **bit-for-bit what it
   is today** on T1's distribution; with it at 1, no neighbouring leaves differ
   by more than one level; and the cell-count multiplier matches A1's prediction.
6. **Do not regenerate the bit-for-bit reference data to make a test pass.**
   `LaplaceSolve.bitForBitArtifacts` compares hashes of internal artifacts
   against `tests/data/laplace_solve_P6.txt`, and
   `CANOPY_LAPLACE_SOLVE_REGENERATE` rewrites them in one step
   (`tests/tstLaplaceSolve.hpp:55-60`). Those hashes are regression-to-self and
   carry no accuracy claim, so a regeneration absorbs a real error exactly as
   readily as a legitimate tree change. With the knob at its default they must
   not move at all; if they move, that is this task failing its own first
   assertion. Regenerate only under a non-default knob, only after both
   `CartesianTaylorSolve` direct-sum arms and
   `SolveFusedM2L.matchesPriorReference` have passed, and record the measured
   deviations in the log beside the regeneration.

**Exit criterion:** **two lists, because the knob has two states.** Balancing
changes the tree for every basis, so unlike B1 this needs both bases and the
partition path.

*Knob at its default* — nothing may move, including the bit-for-bit hashes:

```bash
make -j Canopy_Test_TreeBuilder_MPI_SERIAL Canopy_Test_TreePartitioner_MPI_SERIAL \
       Canopy_Test_CommunicationPlan_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_SERIAL \
       Canopy_Test_LaplaceSolve_MPI_SERIAL Canopy_Test_CartesianTaylorSolve_MPI_SERIAL \
       Canopy_Test_MultiSolve_MPI_SERIAL \
       Canopy_Test_TreeBuilder_MPI_HIP Canopy_Test_TreePartitioner_MPI_HIP \
       Canopy_Test_CommunicationPlan_MPI_HIP Canopy_Test_DownwardSweep_MPI_HIP \
       Canopy_Test_LaplaceSolve_MPI_HIP Canopy_Test_CartesianTaylorSolve_MPI_HIP \
       Canopy_Test_MultiSolve_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|TreePartitioner|CommunicationPlan|DownwardSweep|LaplaceSolve|CartesianTaylorSolve|MultiSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|TreePartitioner|CommunicationPlan|DownwardSweep|LaplaceSolve|CartesianTaylorSolve|MultiSolve)_MPI_HIP_np_[1-4]$'
```

*Knob at 1* — the new cases, run by the same seven stems, assert the balance
invariant and the predicted cell-count multiplier.

T1's case reports the identical cell set
and identical refusal counters as before this task — the change has no runtime
surface until something sets the knob. With the knob at 1 on T1's distribution, a
new case asserts the maximum neighbouring-leaf level difference is 1 and that
`m2l_n_fallback_pairs_range_guard()` is **0**. Failure direction:
`tree_balance_max_level_delta = 0` throws from `FmmConfig`'s route, and a
distribution that cannot be balanced within `max_depth` produces the loud report
rather than a silently unbalanced tree.

---

### A3 — Make balancing the default — **NOT STARTED**

**Depends on:** A2 **DONE**, C1 **DONE**.
**Fill in:** `src/Canopy_Solver.hpp` (the default value); `README.md`
(the knob and its default).
**Reference:** A2's measured cell-count multiplier and C1's
fallback-to-GEMM time ratio, both in the log.
**Do:**
1. Decide the default from the two measured numbers: balancing is worth its
   cell-count cost iff the M2L time it saves on refused pairs exceeds the cost
   of the cells it adds. State the arithmetic in the log with both measured
   inputs — this is the one task whose correct answer is a number from an
   earlier task and not a design choice.
2. If the arithmetic favours balancing, change the default to 1 and re-run the
   seven stems below. If it does not, **leave the default off and say so**: a
   knob that is measured not to pay is a successful outcome for this task, not
   a failure, and the measurement belongs in the log either way.
3. Update `README.md` per `CLAUDE.md`'s keep-in-sync rule, since this changes a
   public configuration default.

**Exit criterion:** A2's seven stems pass on SERIAL at ranks 1-6 and on HIP at
ranks 1-4 at whatever default this task sets —

```bash
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|TreePartitioner|CommunicationPlan|DownwardSweep|LaplaceSolve|CartesianTaylorSolve|MultiSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|TreePartitioner|CommunicationPlan|DownwardSweep|LaplaceSolve|CartesianTaylorSolve|MultiSolve)_MPI_HIP_np_[1-4]$'
```

— the log records the arithmetic with both measured inputs, and `README.md`
documents the knob and its default. Failure
direction: if the default changes to 1, the **measured deviations** of both
`CartesianTaylorSolve` direct-sum arms and of
`SolveFusedM2L.matchesPriorReference` must be recorded before and after and any
change reported — a balanced tree is a different tree, and all three of those
bounds are pinned constants loose enough to absorb a real degradation silently
(**R10**). An unchanged pass is a claim to verify, not evidence.

---

### B2 — Key the table on a stable width exponent — **NOT STARTED**

**Depends on:** B1 **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_TreeBuilder.hpp` (the root-width quantization);
`src/Canopy_Solver.hpp` (the knob, see step 1);
`src/Canopy_DownwardSweep.hpp` (`set_root_half_width`'s invalidation);
`tests/tstDownwardSweep.hpp` and `tests/tstCartesianTaylorSolve.hpp` (the
cases).
**Reference:** the cache-invalidation rule and the reasoning already recorded on
it (`src/Canopy_DownwardSweep.hpp:435-460`); the root half-width's derivation
from the particle bounding box (`src/Canopy_TreeBuilder.hpp:608-635`); the
per-level `unit_w` array the operator builder indexes by `max_d`
(`src/Canopy_CartesianTaylorBasis.hpp:1179-1222`).
**Do:**
1. **Put the quantization behind a knob, default off**, exactly as A2 does for
   balancing, and for the same reason: the root half-width is the tree's own
   geometry, so snapping it moves the cell set **for every basis**, not only
   for the one whose key this chain is about. Left unconditional, this task
   would silently shift every existing result — the bit-for-bit hashes, both
   direct-sum arms and `MultiSolve`'s trajectory alike — and the exit criterion
   below could not distinguish that from a relabelling bug. Name it beside
   A2's knob in `FmmConfig`.
2. Quantize the root half-width: round it **up** to the next power of two with
   `std::frexp`/`std::ldexp`. Up, never down — down would shrink the root box
   and can place a particle outside it, which is the escape condition that
   forces a full rebuild. Document on the declaration that the box is
   deliberately up to 2x larger than the particle extent, and what that costs:
   one extra potentially-empty level at the top of the tree.
3. With the widths quantized, the set of per-depth half-widths recurs exactly
   across rebuilds. Change `set_root_half_width`'s invalidation from "clear the
   whole cache when `key_needs_level`" to **relabelling**: a column cached for
   depth $d$ at half-width $W$ is reusable at the depth that now carries $W$.
   Prefer clearing to a wrong reuse — if the exponents cannot be matched
   exactly, clear, and say so in the log.
4. Assert the retention directly: drive two builds whose root box differs by a
   factor inside one power of two, and assert the per-build increment of
   `m2l_op_keys_built_count()` is **0** on the second — the cache was reused —
   where today it equals `m2l_n_unique_ops()`.
5. Assert the arithmetic is exact: a quantized width round-tripped through
   `frexp`/`ldexp` is bit-identical, and the quantized width is never smaller
   than the input.
6. **Add a direct-sum deviation assertion to the drift trajectory.** The two
   `operatorCacheAcrossDrift*` cases assert no accuracy today
   (`tests/tstCartesianTaylorSolve.hpp:1040-1062`), and they run the trajectory
   on which this task's relabelling operates — so a column reused at the wrong
   width would change only the `keys_built` figure those cases print and would
   corrupt the field silently (**R4**, **R9**). Give them the same
   direct-softened-sum comparison the gating arms use, at a tolerance **measured
   on that trajectory** rather than inherited from the gating arms, whose
   constants were not measured there.

**Exit criterion:** five stems, in two states as A2 has.

```bash
make -j Canopy_Test_TreeBuilder_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_SERIAL \
       Canopy_Test_CartesianTaylorSolve_MPI_SERIAL Canopy_Test_LaplaceSolve_MPI_SERIAL \
       Canopy_Test_MultiSolve_MPI_SERIAL \
       Canopy_Test_TreeBuilder_MPI_HIP Canopy_Test_DownwardSweep_MPI_HIP \
       Canopy_Test_CartesianTaylorSolve_MPI_HIP Canopy_Test_LaplaceSolve_MPI_HIP \
       Canopy_Test_MultiSolve_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|DownwardSweep|CartesianTaylorSolve|LaplaceSolve|MultiSolve)_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_(TreeBuilder|DownwardSweep|CartesianTaylorSolve|LaplaceSolve|MultiSolve)_MPI_HIP_np_[1-4]$'
```

*Knob off*: all five pass unchanged, `LaplaceSolve`'s bit-for-bit hashes
included — the task has no runtime surface until something sets the knob.
*Knob on*: all five pass, the drift cases now carry a direct-sum deviation
bound, and a new case shows the second of two builds across a sub-octave box
change rebuilding **0** columns while realizing the same count.

Failure direction: a box
change spanning more than one power of two clears the cache rather than reusing
it wrongly, asserted by the increment equalling `m2l_n_unique_ops()` in that
case; and the quantized root half-width is asserted to be $\ge$ the unquantized
one on a sweep of inputs, so a rounding that shrank the box fails here rather
than as a particle escape later.

## Known risks

**R1 — Balancing explodes the cell count on a folded surface.** A 2:1 balance on
points distributed over a 2-D manifold can add cells in proportion to the
surface's local curvature and folding, and the multiplier is unknown until A1
measures it. Presentation: not a failure, a **slowdown and a memory rise** in
every phase — upward sweep, partitioner, P2P and M2L alike — with correct
answers throughout, which is the hardest kind of regression to attribute.
Distinguishing measurement: A1's cell-count multiplier, taken *before* A2
implements anything, and A2's own assertion that the multiplier matches the
prediction. If they disagree, the cost model is wrong and A3's arithmetic cannot
be trusted.

**R2 — Balancing changes accuracy, and `MultiSolve` does not notice.** A
balanced tree is a different tree: cells that were leaves become internal, so
pairs move between the P2P and M2L paths and the far-field approximation
applies at different scales. Presentation: a shifted accuracy figure in
`MultiSolve`, or no visible change at all if its tolerance is loose enough to
absorb it. The second is worse. Distinguishing measurement: A3 compares
`MultiSolve`'s own accuracy figures before and after the default changes,
rather than reading a pass as evidence of no change.

**R3 — `key_needs_dd = false` on a basis whose operator does depend on `dd`.**
That aliases two different operators onto one column, and the symptom is a
**wrong velocity, not a slow one**: the table returns a plausible column built
for a different depth ratio. Presentation: an accuracy failure somewhere
unrelated to the table, with no diagnostic pointing at the key. Distinguishing
measurement: `expectKeyTraitsAgree`'s branch on the trait, which B1 makes
symmetric, plus B0's per-basis duplicate counts — `LaplaceKernel`'s column count
must be **unchanged** by B1, and a drop in it is this risk firing.

**R4 — B2's relabelling reuses a column built at a different physical width.**
Arithmetically identical to R3 and presents identically: a cached column used
for a key whose width is not the one it was built at. Distinguishing measurement:
B2 step 4 asserts a rebuild count of 0 on a reuse that *should* happen, and its
failure direction asserts a full rebuild on a box change that spans more than one
octave. The two together pin both directions; either alone would be satisfied by
a cache that always reuses or always clears.

**R5 — The power-of-two root snap enlarges the box enough to matter.** Rounding
up can double the root half-width, which adds a level at the top of the tree and
shifts every per-depth width. Presentation: `occupied_depths` rising by one and
every realized key changing, so the whole table turns over once on the change —
which looks exactly like R4 in the logs. Distinguishing measurement: the
realized key *count* should be stable across the change while the key *values*
shift by one level; a change in the count is a tree change, not a relabelling.

**R6 — The fixture is not reproducible, so a single draw is not the number.**
The tree and partition path is **run-to-run nondeterministic at np $\ge$ 3**:
two identical passes of the same binary on the same input have been measured to
disagree on the admitted column count, the per-depth occupancy and the fallback
count, while the particle set itself is bit-identical. Nothing asserts those
quantities today, so every test passes through it. Presentation: a measurement
that does not reproduce, or worse, a before/after comparison that attributes the
nondeterminism to the change. Distinguishing measurement: A1 and B0 each report
from **two separate runs** and state whether they agree; a before/after
comparison at np $\ge$ 3 must be read against that spread and never as a single
pair of numbers. np 1 and 2 have been measured stable.

**R7 — A measurement taken on a uniform distribution reads as "already fixed".**
The range-guard counter is **0** on every uniform fixture in the suite, so any
of these cases run against the wrong distribution reports zero refusals and
looks like success. Presentation: a clean pass that measures nothing.
Distinguishing measurement: T1 step 3 asserts the tree's occupied-depth
structure and step 4 asserts `range_guard > 0`, so a flattened fixture fails
loudly at the source rather than silently downstream. A1's failure direction
asserts a neighbour level difference of at least 2 for the same reason. The
clustered fixture's own guard against this
(`tests/tstMultiSolve.hpp:751-762`) names the bounds actually in force,
`M2L_KEY_OFFSET_MAX` and `KernelType::m2l_key_dd_max`, and points at the
per-reason lines rather than at a condition that cannot occur — a guard written
against one of those is not a guard, which is what V1 step 5 turns from a report
into an assertion.

**R8 — The fallback's cost is assumed rather than measured, and chain A is sized
from the wrong number.** "1.58 % of pairs" is a pair count; the time share
depends on the per-pair cost of `m2l_translate` relative to the GEMM path, which
is measured nowhere today. Presentation: A3 computes its arithmetic from a pair
fraction and reaches a confident wrong default. Distinguishing measurement: C1
step 3 measures the ratio directly by driving one solve at a column cap of 0, and
A3 is blocked on C1 for exactly that reason.

**R9 — The one trajectory that exercises cache reuse has no correctness check.**
`CartesianTaylorSolve.operatorCacheAcrossDriftThetaCanopy` and `...ThetaRef`
assert only that the far field was live and that an operator was built
(`tests/tstCartesianTaylorSolve.hpp:1040-1062`), and they deliberately run a
longer trajectory than the gating arms so the bounding box drifts — which is
exactly the condition B2 changes the handling of. Presentation: B2 lands, the
`keys_built` figure improves, every test passes, and the field is wrong wherever
a relabelled column was reused. Nothing in the suite would say so.
Distinguishing measurement: B2 step 6 gives those two cases a direct-sum
deviation bound measured on that trajectory. **Until that exists, a `keys_built`
improvement from B2 is not evidence of correctness**, and the gating arms do not
cover it — they run a shorter trajectory on which the box barely drifts.

**R10 — A pinned tolerance absorbs a real degradation.** Three of the bounds
these chains are checked against are loose relative to the method's measured
accuracy: `SolveFusedM2L.matchesPriorReference` runs at `5.0e-2` potential and
`1.0e-1` gradient and says in place that it exists to catch "a
complete-regression bug" rather than a degradation
(`tests/tstMultiSolve.hpp:962-983`); `CartesianTaylorSolve`'s
`theta_canopy` arm is pinned at `3.74e-02`, 2x its own worst measured figure;
and the `MultiSolve` tests compare **positions and
velocities after a short integration** at `1.0e-8`
(`tests/tstMultiSolve.hpp:585-589`), which at `dt = 1.0e-4` over five steps
bounds a force error only very weakly, since the particles barely move.
Presentation: a tree or key change degrades the far field by a few percent and
every bound still passes. Distinguishing measurement: record the **measured
deviations**, not the pass/fail, before and after any task that changes the tree
or the key — A3's failure direction requires exactly that, and it is the only
reason an unchanged pass there means anything.

**R11 — A HIP failure is attributed to the task that ran into it.** No HIP
test ran before this design's tasks, so a HIP arm may fail for a reason that
predates them. Examples: `LaplaceSolve`'s bit-for-bit hashes in `tests/data`
were generated by the SERIAL binary, and device reductions can sum in a
different order. Presentation: a task's HIP arm fails on a case the task does
not touch. Distinguishing measurement: `fix-hang-rebalance.md` H0c's per-stem
HIP table, taken before any change. A failure recorded there is carried and
stays in README "Known Issues". A failure not recorded there belongs to the
task. Never widen a bound to green a HIP arm. The same rule applies as for
SERIAL (V1 step 1).
