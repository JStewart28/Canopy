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
(`src/Canopy_TreeBuilder.hpp:918`), with no reference to its neighbours' depths
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
   (`src/Canopy_CartesianTaylorBasis.hpp:980-986`), yet `dd` was in its key, so
   keys differing only in `dd` built identical columns. B1 removed `dd` from
   that basis's key.
3. **The operator cache retains nothing across rebuilds.** The key carries
   `max_d`, which indexes a per-level physical half-width, and the root
   half-width is the global particle bounding box recomputed on every full
   rebuild (`src/Canopy_TreeBuilder.hpp:608-635`). So `set_root_half_width`
   clears the entire cache whenever `key_needs_level` is true
   (`src/Canopy_DownwardSweep.hpp:450-460`), and a moving particle distribution
   rebuilds the whole table at every evaluation.

The end state is that all three are bounded independently of tree depth: the
range guard stops firing because no pair has a large depth difference
(**chain A**), the table stops holding duplicate columns and stops being rebuilt
whenever the bounding box drifts (**chain B**), and the depth a large problem actually
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
| **B** | the duplicate columns, by fixing the key; the drifting ones, by quantizing the root width | B0, B0b, B1, B2 |
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
`dd`, which the operator provably does not depend on, and B2 quantizes the
root half-width to a power of two, so the per-depth widths `max_d` indexes stay
fixed while the box drifts within an octave and the cache survives the rebuild.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| New basis trait name | `key_needs_dd` | Mirrors `key_needs_level` exactly in name, placement (beside it in the basis's M2L key-contract block), type (`static constexpr bool`) and role: a basis declaring which of the key's five integers its operator depends on. A reader who knows one knows the other. |
| Trait default | none — every basis states it explicitly | `key_needs_level` has no default either. A silent default on a correctness-critical trait is how a new basis gets the wrong one; the conformance test in B1 fails a basis that omits it. |
| Trait/`canonicalize_key` agreement | asserted, never trusted | `expectKeyTraitsAgree` (`tests/tstFarFieldContract.hpp:927-983`) enforces this for `key_needs_level` and, since B1, for `key_needs_dd`, in the same function. |
| Balancing knob name | `FmmConfig::tree_balance_max_level_delta` | `FmmConfig` is where every other tree and table knob lives (`src/Canopy_Solver.hpp`), and the name states the invariant as a number rather than as a mode, so "2:1" is the value 1 and "off" is a large value rather than a second boolean. |
| Balancing knob default | the off value, until A3 | A2 must not move any existing result. A knob whose default is the current behavior is a change with no runtime surface until something sets it, which is what makes A2's exit criterion checkable against the existing stems, unmodified. |
| Root-width quantization knob | `FmmConfig::quantize_root_half_width`, `bool`, default `false` | Quantizing moves the cell set for every basis, so it has no runtime surface until something sets it — the same reason the balancing knob defaults off. A `bool` because it is a mode with no magnitude. |
| Root half-width source | `TreeBuilder::root_half_width()`, the value `build()` stamped on the root cell | `_root_box` is the expanded, non-cubic bounding box and stays so: `needs_rebuild` (`src/Canopy_TreeBuilder.hpp:860-890`) and the auto-softening volume (`src/Canopy_Solver.hpp:797-815`) read it as a box. Every reader that needs the root cell's half-width reads the accessor, never a re-derivation from `root_box()`, so the sweep is told exactly the width the tree was built at. |
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

HIP registers at np 1-4 with one APU per rank because
`fix-hang-rebalance.md` H0a is **DONE**. Run each HIP `ctest` line with the HIP environment
set on that command only (`fix-hang-rebalance.md`, Conventions, "HIP
environment"). Run every `ctest` line through `canopy_ctest` with the same
regex. That gives each entry a timeout of 1.75x its measured SERIAL runtime
instead of 300 s (`fix-hang-rebalance.md` H0b and Conventions, "Time
budget"). The code blocks below show the bare `ctest`. A task that adds a case
to a stem, or changes its runtime, re-calibrates that stem's rows in
`scripts/tuolumne/serial_runtimes.tsv` in the same change. B2's and A2's
knobs are set by the cases that need them, so each re-calibrates only the
`default` rows of the stems it adds cases to. So a task that names
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
(`tests/tstMultiSolve.hpp:1041-1600`); the `LaplaceSolve` stem carries
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
  (`src/Canopy_CartesianTaylorBasis.hpp:980-986`).
- **B1 keeps `m2l_key_dd_max` as a trait and keeps the sweep's `dd` guard.**
  Once `dd` is out of `CartesianTaylorBasis`'s key, that basis has no
  representability reason to bound `dd` — but `LaplaceKernel` does: its own
  bound is a precision claim about a $2^{\,j|\texttt{dd}|}$ residual factor in
  its scale-normalized operator, and is tightened to 4 for `float`
  (`src/Canopy_LaplaceKernel.hpp:689-702`). The guard is therefore still needed
  and still basis-driven; what changes is only that one basis can now set the
  bound by precision alone rather than inheriting a key-space limit.
- **B2 quantizes the root half-width and keeps `max_d` as the cache key.**
  Keying the cache on the floating-point width would make cache identity depend
  on bit-exact equality of a derived quantity, which is the kind of thing that
  works until a reduction order changes. Quantized to a power of two, the root
  half-width is bit-identical across every rebuild whose box stays inside one
  octave, so `set_root_half_width` sees an unchanged value, does nothing, and
  the cache survives with its keys untouched. A box that crosses an octave
  changes the value and clears the cache exactly as an unquantized change does
  today. Shifting cached keys' `max_d` by the exponent change instead of
  clearing would keep the cache across octaves too; it is recorded in
  `README.md` "Future Optimizations" and not done here, because every column it
  reuses is a column a wrong shift would corrupt (**R4**).
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
| `m2l_op_keys_built_count()` | `:479` | cumulative columns ever built; its per-build increment is the cache-retention measure |
| `m2l_effective_op_cap()` | `:368-374` | the column cap in force |

So no new accessor is needed to measure refusals, key duplication, cache
retention or occupied depth.

**A non-uniform fixture already exists, and it already produces refusals.**
`testMultiStepGravity` takes a `clustered` flag — 80 % of particles from a tight
Gaussian blob in one corner, 20 % uniform (`tests/tstMultiSolve.hpp:261`,
`:211-233`) — and `MultiSolve.M2L_BinEdge_Fallback`
(`tests/tstMultiSolve.hpp:1197-1213`) drives it at `ncrit = 8`, `max_depth = 8`
and `mac_theta = 0.3`, asserting that the fallback population is non-zero
(`:1214-1229`). That test is the `regression` label's only member
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
  (`src/Canopy_TreeBuilder.hpp:918`), and nothing anywhere consults a
  neighbour's depth. There is no post-pass, no flag and no partial form of it.
- **`max_depth` is capped at 19 by Morton-key storage**, which `TreeBuilder`'s
  constructor enforces by throwing (`src/Canopy_TreeBuilder.hpp:252`). Any
  smaller limit a caller runs at is that caller's choice, not a library limit.
- **`CartesianTaylorBasis` declares `key_needs_level = true` and
  `key_needs_dd = false`** (`:486`, `:492`). Its `canonicalize_key` keeps
  `max_d` and zeroes `dd` (`:503-508`), so its canonical key is
  $(\texttt{max\_d}, 0, \texttt{ii}, \texttt{jj}, \texttt{kk})$. `LaplaceKernel`
  declares `key_needs_level = false` and `key_needs_dd = true` (`:721`,
  `:730`) and zeroes `max_d` only (`:752-756`). `MonopoleBasis` declares both `true`.
- **`m2l_key_dd_max` is 6 for `CartesianTaylorBasis`** (`:469`) and its comment
  states that for this basis the number "merely bounds the key space" and "is
  not a precision claim about this basis" — chosen so the refused set matches
  what other bases see. For `LaplaceKernel` it is 6 for `double` and 4 for
  `float`, and there it *is* a precision bound (`:689-702`).
- **`expectKeyTraitsAgree` branches on `key_needs_dd` as it does on
  `key_needs_level`** (`tests/tstFarFieldContract.hpp:927-983`): a `true` basis
  must preserve `dd` and keep two keys differing only in `dd` distinct, a
  `false` one must collapse them. Only the offset `(ii, jj, kk)` is asserted
  unaltered for every basis. It runs on `MonopoleBasis`, `LevelBlindBasis`,
  `LaplaceKernel` and `CartesianTaylorBasis` (`:994-998`).
- **`set_root_half_width` clears the whole operator cache** when
  `key_needs_level` is true, and deliberately does not when it is false
  (`src/Canopy_DownwardSweep.hpp:450-460`, rationale block from `:414`), and
  is a no-op when handed the value it already holds. The root half-width is the largest
  half-extent of the global particle bounding box
  (`src/Canopy_TreeBuilder.hpp:608-635`) and is not quantized.
- **There is no `FmmConfig` knob for tree balance.**
- **The `regression`-labeled stem exercises `LaplaceKernel` only.**
  `REGRESSION_MPI_TESTS` is the single stem `MultiSolve`
  (`tests/CMakeLists.txt:61-63`), and `tstMultiSolve.hpp` instantiates
  `Canopy::Solver<..., Scalar, P, 1>` (`:274-275`) without a far-field
  argument, so it takes the default `FarField = LaplaceKernel`
  (`src/Canopy_Solver.hpp:146-148`). **No `regression`-labeled test
  instantiates `CartesianTaylorBasis.`** Its only direct-sum coverage is
  `CartesianTaylorSolve.matchesDirectSumThetaRef` and `...ThetaCanopy`
  (`tests/tstCartesianTaylorSolve.hpp:977-1007`), both in the `unit` tier, and
  that is where its coverage stays. So a chain-B change is
  checked by the `unit` tier alone: B1's and B2's exit criteria are the only
  thing standing between a broken `CartesianTaylorBasis` and a green run of
  whatever stems some other task happens to name.
- **The two operator-cache drift cases assert no accuracy at all.**
  `CartesianTaylorSolve.operatorCacheAcrossDriftThetaCanopy` and
  `...ThetaRef` (`tests/tstCartesianTaylorSolve.hpp:1041-1061`) state in place
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
  A1's cost model is a cell count, which `TreeBuilder`'s own cell list answers
  without it. A balanced tree also changes
  the band populations H2's constraints are drawn from, so A2 records the
  per-band imbalance before and after.
- **The upward sweep's coefficient formation.**
  `src/Canopy_UpwardSweep.hpp` was read only far enough to confirm where
  `build_aux_tables` is called. No task here changes coefficient scaling, so it
  is not on any path; B2 changes only the root cell's width, not what any
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

### T1 — A fixture whose tree has shallow leaves beside deep subtrees — **DONE**

**Depends on:** `fix-hang-rebalance.md` H0c **DONE** (HIP registration and
baseline) and H2 **DONE**, both arms, since a `MultiSolve` HIP pass that hangs
at np 3 cannot be read.
**Fill in:** `tests/tstMultiSolve.hpp` (the per-reason report on the existing
clustered fixture); `tests/tstDownwardSweep.hpp` (the reusable fixture and the
new case). Both stems are already registered
(`tests/CMakeLists.txt:46-57`, `:61-63`), so no CMake change is needed.
**Reference:** the existing clustered distribution and the test that drives it
(`tests/tstMultiSolve.hpp:309-322`, `:230-262`, `:1197-1229`) — **the starting
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

**Met.** Step 1 settled the open question first, on the clustered fixture
unchanged at its own `ncrit = 8`, `max_depth = 8`, `mac_theta = 0.3`: **every
refusal there is a range-guard refusal**, with `count_cap == 0` and
`depth_dropped == 0` on every `(nprocs, rank, step)` reading and
`range_guard == total`. Within the range guard they are **offset** refusals
specifically — that tree's occupancy never passes depth 6, so `|dd|` cannot
exceed `LaplaceKernel`'s `m2l_key_dd_max` of 6 and that half of the guard is
geometrically unreachable. The stale `M2L_BIN_RANGE = 3` comment sites and the
`EXPECT_GT` failure message name `M2L_KEY_OFFSET_MAX = 32` (half-widths at the
deeper cell's depth) and `KernelType::m2l_key_dd_max`.

That answer put step 2 on the "new two-scale distribution" branch, so
`DownwardSweepTest::TwoScaleFixture<MS, ES, FarField = Kernel>` was built in
`tests/tstDownwardSweep.hpp` — a cube of half-width 0.01 holding 87.5 % of a
**global** 1200-particle set beside a uniform halo (since A1, with a 5 % skirt
of half-width 0.04 about the blob; the baseline step-8 lines are A1's, see
`tree-opt-progress-log.md` `## A1`), templated on the far-field
type so B0 can read the same tree for both `CartesianTaylorBasis` and
`LaplaceKernel`. On the post-H2 partition, two separate runs on each backend
(SERIAL np 1-6, HIP np 1-4) print **identical** step-8 lines, field for field,
and HIP matches SERIAL field for field at every np 1-4: `deepest == 8` and the
shallowest leaf at depth 1 or 2 on every rank, `count_cap == 0`,
`depth_dropped == 0`, `range_guard == total_fallback_pair_count()` on every
line, and `range_guard > 0` on at least one rank at every rank count, at the
default column cap (32768). The geometry contract is asserted on the depth at
which refinement *stops*, not on the shallowest occupied depth — depth 0 is
occupied on every tree, so the latter is a vacuous pass of exactly the kind
**R7** describes. The failure direction passes in `build-tuolumne-noprof/` on
both backends: every counter reads **-1** and the sum identity is reported
SKIPPED once per rank count.

The `MultiSolve` arm carries the failures README "Known Issues" records: on
both backends the six `1e-8` multi-step cases, which are V1's to resolve, and
on HIP alone `SolveFusedM2L.multipleSolvesIdempotent`, which H0c's own log
already shows failing. `M2L_BinEdge_Fallback`'s fallback and per-reason
assertions pass; it fails only on the shared `1e-8` check. See
`tree-opt-progress-log.md` `## T1` and `## T1 (HIP arm)`.

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

### V1 — Sharpen the checks these chains will be verified against — **DONE**

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
**Reference:** the bounds as they now stand, each with its derivation in
place — the six call sites' `pos_tol` / `vel_tol` (`tests/tstMultiSolve.hpp:1051`,
`:1070`, `:1090`, `:1111`, `:1154`, `:1209`) under the shared derivation block
above `:1041`, applied to `max_pos_rel` and `max_vel_rel` at `:1006-1013`;
AutoRebalance's `probe_field_tol` (`:1157`), checked per step in the probe
block; `SolveFusedM2L.matchesPriorReference` (`:1435-1466`) at `5.0e-2` /
`5.7e-4`; `CTS_DEV_TOL_THETA_CANOPY = 3.74e-02`
(`tests/tstCartesianTaylorSolve.hpp`), which the rationale block above it
records as **2x the worst measured figure** — gradient `1.8651551291e-02`,
potential `9.9666786091e-04`, identical at every rank count on both backends;
`CTS_DEV_TOL_THETA_REF`, the `1e-3` bar itself; the drift cases' explicit
absence of any accuracy claim (`:1041-1061`).
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

**Met.** Three successive passes of both stems pass with no failure carried, on
SERIAL at np 1-6 (36 of 36 entries) and on HIP at np 1-4 (24 of 24), flux
jobs `f3cbCkKDsirK` and `f3cbCkSwbyxT`. Step 1's stop clause never fired: the
per-step field-scale error is at most `1.52e-6` at any site, np or backend.
Every measured trajectory deviation sits under its derived first-order figure,
so each of the six sites is bounded at its worst measured deviation x 2
(separate `pos_tol` / `vel_tol`), and `AutoRebalance` also gates every step's
field error at `1.9e-6`. `matchesPriorReference`'s gradient bound is `5.7e-4`
(from `1.0e-1`); its potential stays `5.0e-2`. The `theta_canopy` bound is
confirmed at 2.005x its worst figure. Step 5's assertions ran in both
branches. Failure direction (θ = 0.7 with `matchesPriorReference` reading it,
temporarily): its gradient fails `5.7e-4` at every np where `1.0e-1` passes,
and `AutoRebalance`'s probe gate fails where a gate at the floor passes.
`SolveFusedM2L.multipleSolvesIdempotent` asserts agreement to `1e-11` at
field scale rather than bit-identity, which the HIP backend's accumulation
order cannot give. `FP32_smokeTest` is commented out. See
`tree-opt-progress-log.md` `## V1 (resume)` and `## V1 (close)`.

---

### B0 — Measure how many admitted columns are `dd` duplicates — **DONE**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstDownwardSweep.hpp` — `TwoScaleFixture`'s constructor
(`:1457-1545`), and a case beside T1's that instantiates the fixture for both
bases.
**Reference:** `m2l_realized_keys()`
(`src/Canopy_DownwardSweep.hpp:1060-1063`) returns the admitted key list by
const reference; `CartesianTaylorBasis`'s no-`dd`-dependence statement
(`src/Canopy_CartesianTaylorBasis.hpp:972-979`); `Solver::_push_root_half_width`
(`src/Canopy_Solver.hpp:780-791`) for the root half-width a solve hands the
sweep; the `[two-scale]` table in `tree-opt-progress-log.md` `## T1 (HIP arm)`,
the baseline every per-`(nprocs, rank)` figure here is read beside — it
reproduces line for line on both backends at every np on the current
partitioner.
**Do:**
0. **Make the fixture able to run `CartesianTaylorBasis`.** It drives the
   sweeps directly, so it runs at `M2LKernelParams::softening = 0.0` and a root
   half-width of 0 — and `CartesianTaylorBasis` aborts on both:
   `build_m2l_operators` requires a positive softening
   (`src/Canopy_CartesianTaylorBasis.hpp:1187`) and a positive `unit_w` entry
   (`:1220`), and `m2l_translate` reads `b` from the aux tables
   `UpwardSweep::setup` builds and `DownwardSweep` borrows
   (`src/Canopy_UpwardSweep.hpp:174-198`). Give the fixture a positive
   `softening` member (a LENGTH $\varepsilon$; document on it that it moves
   neither the integer keys nor the interaction list), pass it through
   `set_m2l_kernel_params` to the upward sweep **before** `upward.setup()` and
   to the downward sweep, and set the downward sweep's root half-width from
   `builder.root_box()` exactly as `_push_root_half_width` does. Neither setting
   reaches `LaplaceKernel`'s keys — they are integer, and that basis declares
   `key_needs_level = false` — so T1's two cases must print `[two-scale]` lines
   identical to the baseline table. That is the check that this step moved
   nothing.
1. On T1's fixture, with `CartesianTaylorBasis` **at order 3** — the order of
   the downstream configuration the chains are sized from (3200 B per column,
   [Measured on a downstream application's
   configuration](#measured-on-a-downstream-applications-configuration)), so
   the byte figures compare directly; the key set itself does not depend on
   the order — count the admitted keys and
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
with the absolute column counts and byte figures. The stem's rows in
`scripts/tuolumne/serial_runtimes.tsv` are re-calibrated for the added case
([Test naming](#test-naming-and-how-to-run-only-what-a-task-needs)). Failure direction: the `-DCanopy_ENABLE_PROFILING=OFF`
build passes the same case, since `m2l_realized_keys()` is ungated — a $-1$
anywhere in this case's output would be this case reading the wrong accessor.

**Met.** The duplicate factor is **exactly 1.0** for both bases at every
`(nprocs, rank)`: every admitted key already has a distinct
$(\texttt{max\_d}, \texttt{ii}, \texttt{jj}, \texttt{kk})$. So B1 removes no
column on this fixture. `CartesianTaylorBasis` at order 3 admits 9-162 columns
per rank (28.8 KB-518 KB at 3200 B), and `LaplaceKernel` 9-162 (21 952 B each).
`DownwardSweepTwoScale.ddDuplicateColumns` passes on SERIAL np 1-6 and HIP
np 1-4 in two passes each, flux jobs `f3cc4m4y111Z` and `f3cc4mDrvh5Z`. The two
passes print identical lines, and HIP matches SERIAL line for line. Step 0's
fixture change left T1's `[two-scale]` lines identical to the baseline. The
failure direction passes in `build-tuolumne-noprof/` on both backends
(`f3cc4mNnLNRu`, `f3cc4mXSvAxo`), with the same admitted and distinct counts;
only the gated `demanded_ops` reads $-1$. See `tree-opt-progress-log.md`
`## B0`.

---

### B0b — The `dd`-duplicate factor on a tree dense in cross-level keys — **DONE**

**Depends on:** B0 **DONE**.
**Fill in:** `tests/tstDownwardSweep.hpp` — a second draw beside
`generate_two_scale_particles` (`:1416`), a draw selector on `TwoScaleFixture`
(`:1529-1644` after this task), a per-`dd` histogram in
`reportTwoScaleDdDuplicates` (`:1757-1784` before this task), and a case beside
`ddDuplicateColumns` (`:1822` before this task).
**Reference:** B0's case (`testTwoScaleDdDuplicates`, `:1787-1805`), which
this task reuses unchanged in what it counts; the classify pass's offset
computation (`src/Canopy_DownwardSweep.hpp:1671-1697`).

B0 measured a factor of exactly 1.0 on T1's fixture, and part of that is
structural: at `dd == 0` every offset component is even, and at `dd != 0`
every component is odd (`src/Canopy_DownwardSweep.hpp:1671-1697`), so a
same-level key never collides with a cross-level one. Only keys with
**different non-zero** `dd` — including $+k$ against $-k$ — can collide. A
duplicate factor above 1.0 therefore needs a tree that admits many keys at
$|\texttt{dd}| \ge 1$, and T1's fixture was built for a step change in depth
(most cross-level pairs exceed the range guard) rather than for that.

**Do:**
1. **Histogram the admitted keys by signed `dd`** in
   `reportTwoScaleDdDuplicates`, on one extra line per basis per rank, and
   report it for T1's fixture first. This is the number B0 did not record: how
   many cross-level keys that fixture admitted at all.
2. **Add a graded draw**: particles at a log-uniform radius
   $r = r_{\min} (r_{\max}/r_{\min})^{u}$, $u \sim U(0,1)$, isotropic in
   direction, about a centre inside the unit box. Density then falls as
   $r^{-3}$, the leaf width needed to hold `ncrit` particles grows as $r$, and
   the leaf depth drops by one level per octave of radius — every shell
   boundary is a one-level step, which is the geometry that admits the most
   $|\texttt{dd}| = 1$ pairs inside the range guard. Choose
   $r_{\min}, r_{\max}$ so the tree spans at least five occupied depths at
   `max_depth = 8`; record the values on their declarations, in domain units.
3. **Select the draw with an enum**, not a bool: a `TwoScaleDraw` (or
   similarly named) enumerator passed to `TwoScaleFixture`'s constructor,
   defaulting to the existing two-scale draw so T1's and B0's cases are
   unchanged. Enumerate the fixture's construction sites before changing the
   constructor; they are all in this file.
4. Run B0's count — admitted, distinct $(\texttt{max\_d}, \texttt{ii},
   \texttt{jj}, \texttt{kk})$, factor, bytes, and the step-1 histogram — on the
   graded draw for both bases, at `mac_theta` **0.3** (the downstream
   configuration's value and the fixture's) and **0.5** (the solver default).
   Admissibility moves which `|dd|` shells are populated, so one angle is not
   the measurement.
5. **Assert the graded tree's contract**, as T1 does for its own: the sum over
   ranks of admitted keys with `dd != 0` is **greater than** the same sum on T1's
   fixture at the same rank count and angle. A graded draw that admits no more
   cross-level keys than T1's has not tested anything B0 did not. Keep
   `distinct <= admitted` per rank. **Assert no threshold on the factor.**

**Additional information needed:** whether the factor exceeds 1.0 anywhere.
If it is 1.0 on the graded draw at both angles as well, B1 has no measured
payoff on any tree in this suite. Record that in the log with the histograms
that show why, and stop: whether B1 is still worth doing is then a change to
this document, not a decision for the session that runs B0b.

**Exit criterion:** stem `DownwardSweep` passes on SERIAL at ranks 1-6 and on
HIP at ranks 1-4 —

```bash
make -j Canopy_Test_DownwardSweep_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_HIP_np_[1-4]$'
```

— T1's and B0's lines unchanged against `## B0`'s table, and the log records,
per `(nprocs, rank)`, basis and angle, the admitted and distinct counts, the
factor, the bytes and the signed-`dd` histogram, for both draws, from two runs
per backend. The stem's rows in `scripts/tuolumne/serial_runtimes.tsv` are
re-calibrated for the added cases. Failure direction: the graded case's
contract assertion fails when the case is temporarily pointed at the
two-scale draw — the cross-level sum is then equal, not greater — demonstrated
once and reverted, with the failure message recorded.

**Met.** The factor **exceeds 1.0**. On the graded draw,
`CartesianTaylorBasis` measures 1.0000-1.0403 per rank at θ 0.3 and
1.0896-1.3163 at θ 0.5. At np 1 that is 4162 → 4068 columns at θ 0.3 and
26 702 → 20 286 at θ 0.5 (85.4 MB → 64.9 MB at 3200 B). T1's draw stays at
exactly 1.0 at θ 0.3, but reaches 1.1449 at θ 0.5. `LaplaceKernel`'s control
reads 1.0079-1.5246 on the graded draw. That figure is not a saving, because
its operator depends on `dd`. By B0's parity argument, every collision is
between two non-zero `dd` values. No admitted key has `|dd|` above 3.
`DownwardSweepTwoScale.ddDuplicateColumnsGraded` (`testGradedDdDuplicates`,
`tests/tstDownwardSweep.hpp:1925-2004`) passes on SERIAL np 1-6 and HIP np 1-4,
two passes each (`f3ccGbJHwxPy`, `f3ccGbSC48UX`). The two passes are identical,
HIP equals SERIAL line for line, and T1's and B0's lines are byte-identical to
B0's jobs. The contract assertion fails `48 vs 48` at np 1 when the graded case
is pointed at the two-scale draw (`f3ccKujp6xQP`). See
`tree-opt-progress-log.md` `## B0b`.

---

### A1 — Measure the depth imbalance and the cost of removing it — **DONE**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstDownwardSweep.hpp` — a measurement case beside T1's,
reading `DownwardSweepTest::TwoScaleFixture`'s `builder.cells()` directly. The
two-scale draw (`generate_two_scale_particles`, `:1416`), the graded draw
(`generate_graded_particles`, `:1459`) and the fixture all live in that file,
on its `Position` + `Charge` AoSoA. No `src/` change.
**Reference:** `TreeBuilder::cells()` (`src/Canopy_TreeBuilder.hpp:202`) and
the refinement loop (`:692-812`), whose leaf decision is at `:804`. A candidate
with no particles is skipped (`:789`), so the cell list holds occupied cells
only. Counts are `MPI_Allreduce`d (`:778`), so the cell list is the **global**
tree, identical on every rank: per-rank figures at one np must agree. The tree
differs between rank counts, because the draw is seeded per rank.
`m2l_cells_at_depth()` (`src/Canopy_DownwardSweep.hpp:1152`) is a sweep
accessor and serves only as a cross-check: at np 1 the per-depth count of
`builder.cells()` equals T1's `cells_at_depth` line.
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

**Exit criterion:** stem `DownwardSweep` passes on SERIAL at ranks 1-6 and on
HIP at ranks 1-4 —

```bash
make -j Canopy_Test_DownwardSweep_MPI_SERIAL Canopy_Test_DownwardSweep_MPI_HIP
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_SERIAL_np_[1-6]$'
ctest --output-on-failure -R '^Canopy_Test_DownwardSweep_MPI_HIP_np_[1-4]$'
```

— and the log records, per `(nprocs, rank)`, the neighbour
level-difference distribution, the count exceeding 1, and the implied cell-count
multiplier, from two separate runs. T1's, B0's, B0b's and B1's existing lines
in the stem stay byte-identical, and the stem's `default` rows in
`scripts/tuolumne/serial_runtimes.tsv` are re-calibrated for the added case. Failure direction: the case asserts the
level-difference maximum is **at least 2** on T1's distribution — if it is 1,
the fixture is already balanced and is the wrong fixture for chain A, which is a
finding about T1 and must fail here rather than silently making A2 untestable.

**Met.** `DownwardSweepTwoScale.balanceCost` prints one `[a1-balance]` line per
`(draw, nprocs, rank)` and passes on SERIAL np 1-6 (`f3cmHLxa4PFM`) and HIP
np 1-4 (`f3cmHMF2pQjR`), two passes each, every `canopy_ctest` outcome
`completed`. Both passes are identical on each backend, and HIP equals SERIAL
at np 1-4. Asserted: the lines are identical across ranks; the particle walk
reproduces `builder.cells()`' occupancy; each simulated balance ends within
its delta; at np 1 the per-depth cell count equals T1's `[two-scale]` line;
and the two-scale draw's largest touching-leaf level difference is at least 2
(it is 3-4). **Deviation:** T1's draw as it stood failed that last assertion
at np 2, 5 and 6 (largest difference 1; the blob bordered only empty cells),
so `generate_two_scale_particles` gained a skirt. Every line that reads the
two-scale draw moved and was re-baselined in the log. The graded-draw lines
are byte-identical to B2's `f3cj6QNSmYfy` / `f3cj6QWEwmd9` (210 SERIAL and
100 HIP lines per pass). Multipliers at delta 1 / 2 / 3: two-scale
1.096-1.280 / 1.031-1.135 / 1.000-1.022, graded 1.316-1.465 / 1 / 1. The
`default` rows are re-calibrated (`f3cmFUQpEFXd`). No `src/` change.

---

### C1 — Depth and fallback-cost headroom against problem size — **NOT STARTED**

**Depends on:** T1 **DONE**.
**Fill in:** `tests/tstTreeBuilder.hpp` (the depth-scaling case, on that
file's own copy of T1's skirted draw, `TreeBuilderTest::generate_two_scale_positions`
(`:126`), which is fixed at 1200 global particles and gains a count
parameter); `tests/tstCartesianTaylorSolve.hpp` (the path-equivalence and
timing case, which needs a solve and so belongs beside
`with_cartesian_taylor_solve` (`:448`) rather than in the tree builder's
tests); `src/Canopy_TreeBuilder.hpp` (step 5's warning, at the leaf decision).
**Reference:** the Morton depth limit (`src/Canopy_TreeBuilder.hpp:252`);
the leaf decision (`:918`); A2's balance warning (`:1083-1084`), the style
step 5 follows; `m2l_cells_at_depth()`; `set_m2l_op_count_cap`
(`src/Canopy_DownwardSweep.hpp:350-358`), where 0 is legal and means no column
is built, so every pair takes the overflow path; the profiling timer registry
(`src/Canopy_Profiling.hpp:134-148`, `timer_registry()` / `reset_timers()`).
**Do:**
1. Build T1's two-scale distribution at a geometric sweep of particle counts and
   record, at each: the deepest occupied depth, the **count of occupied depths**,
   and whether any cell hit `max_depth` and so was made a leaf by the depth
   limit rather than by `ncrit` (`src/Canopy_TreeBuilder.hpp:918`). The second
   case is a silent accuracy change, not an error, and nothing currently reports
   it. Build the sweep at `max_depth = 19`: T1's tree at its own `max_depth = 8`
   already reaches depth 8 on every rank (`tree-opt-progress-log.md` `## A1`),
   so a capped sweep measures the cap, not the depth the draw needs. Report the
   `max_depth`-leaf count at the fixture's own `max_depth = 8` as well. Count
   such leaves from `builder.cells()` — a leaf at depth `max_depth` with
   `global_count > ncrit` — with no new accessor.
2. Fit the growth of occupied-depth count against particle count, and state the
   particle count at which the deepest occupied depth reaches **19**, the Morton
   limit. For points distributed on a 2-D manifold the leaf count at depth $d$
   grows as $4^{d}$ rather than $8^{d}$, so the required depth is
   $\approx \log_4(N/\texttt{ncrit})$ — state the measured exponent rather than
   assuming that one.
3. Measure the **per-pair cost of the fallback path relative to the GEMM path**,
   as a time ratio, on `with_cartesian_taylor_solve`'s fixture with
   `CartesianTaylorBasis` at order 3, the downstream configuration's order. The
   ratio is a property of the basis, order and backend, not of the draw. Drive
   the solve once at `FmmConfig::m2l_op_count_cap = 0`, so every pair takes
   `m2l_translate`, and once at the default cap. The time is the profiling
   registry's `m2l_kernel` (level 1), which covers both the fused GEMM kernel
   (`run_m2l_all`, `src/Canopy_DownwardSweep.hpp:2590-2602`) and the per-pair
   fallback (`run_m2l_fallback_at_depth`, called inside `run_m2l_at_depth`,
   `:2611-2627`). The ratio is per pair: `m2l_kernel` divided by the pairs that
   path evaluated, at cap 0 over at the default cap, taken on a warm second
   solve of an unchanged tree with the timers reset before it. Report beside it
   `ilist_s4_op_table_build` (level 2) from the cold first solve, the GEMM
   path's table cost. Record SERIAL and HIP side by side: production runs on
   HIP, and one team per pair against a fused GEMM can cost very differently on
   the two backends. Both timers are profiling-gated, so with
   `CANOPY_ENABLE_PROFILING` undefined the case prints the $-1$ sentinel and
   skips the ratio, as T1 skips its sum identity. Without this ratio a
   pair-count fraction cannot be turned into a cost and chain A's value cannot
   be judged.
4. **Assert that the two M2L paths agree.** The pair of solves in step 3 —
   one at a column cap of 0 so every pair takes `m2l_translate`, one at the
   default cap so almost none does — differ only in which path evaluates the
   far field, so comparing their fields costs one extra comparison and closes
   the gap that chains A and B both rest on. Nothing in the suite asserts this
   today: `MultiSolve.M2L_BinEdge_Fallback` asserts the fallback is
   *exercised* (`tests/tstMultiSolve.hpp:1214-1229`), not that it produces the
   same answer as the tabulated path. **Chain A's entire purpose is to move
   pairs between those two paths**, so without this assertion a chain-A change
   that silently broke one of them would present as an accuracy shift with no
   indication of which path was at fault. Bound the deviation at the level
   measured here rather than at a guessed constant; the two paths evaluate the
   same mathematics and should agree far more tightly than either agrees with
   the direct sum.
5. Add a loud report — not an assertion — when a cell is made a leaf by
   `max_depth` rather than by `ncrit`. It lives in `TreeBuilder::build()` at
   the leaf decision (`src/Canopy_TreeBuilder.hpp:918`), so downstream runs see
   it: one rank-0 `[Canopy] WARNING` line per build, in the style of A2's
   balance warning (`:1083-1084`), stating how many leaves at `max_depth` hold
   more than `ncrit` particles. No accessor is added; tests count these leaves
   from `builder.cells()`.

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
count, the extrapolated particle count at which depth 19 is reached, the
per-pair fallback-to-GEMM `m2l_kernel` ratio on SERIAL and on HIP, and the
cold-solve `ilist_s4_op_table_build` time beside it. The `default` rows of
`TreeBuilder` and `CartesianTaylorSolve` in
`scripts/tuolumne/serial_runtimes.tsv` are re-calibrated for the added cases.
Failure direction: the case asserts that a tree built at `max_depth = 19` throws
nothing and that a tree requested at `max_depth = 20` **does** throw
`std::runtime_error` from `TreeBuilder`'s constructor, so the limit is pinned by
a test rather than by a comment.

---

### B1 — `key_needs_dd`, and a `CartesianTaylorBasis` key without `dd` — **DONE**

**Depends on:** B0b **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` (the new trait,
`canonicalize_key`); `src/Canopy_LaplaceKernel.hpp` (the new trait);
`tests/CanopyTest_MonopoleBasis.hpp` (the new trait on `MonopoleBasis`, its one
basis, in the key-contract block `:131-177`; `LevelBlindBasis`
(`tests/tstFarFieldContract.hpp:897`) derives from it and inherits the trait);
`tests/tstFarFieldContract.hpp` (`expectKeyTraitsAgree` and its call sites);
`tests/tstDownwardSweep.hpp` (`testGradedDdDuplicates`'s contract).
**Reference:** `key_needs_level`'s declaration
(`src/Canopy_LaplaceKernel.hpp:704-721`) and its two readers,
`set_root_half_width` (`src/Canopy_DownwardSweep.hpp:455`) and the diagnostic
print (`:2007-2018`), as the exact pattern to mirror;
`CartesianTaylorBasis`'s no-`dd`-dependence statement
(`src/Canopy_CartesianTaylorBasis.hpp:972-979`); the current conformance
assertions (`tests/tstFarFieldContract.hpp:934-937`) and their only call sites
(`:970-971`); B0b's cross-level count (`tests/tstDownwardSweep.hpp:1854-1864`)
and contract (`:1987-1999`).

**Payoff.** B0b measured it on the graded draw: `CartesianTaylorBasis`'s
duplicate factor is 1.0000-1.0403 per rank at θ 0.3 (np 1: 4162 → 4068
columns) and 1.0896-1.3163 at θ 0.5 (np 1: 26 702 → 20 286, 85.4 MB →
64.9 MB). That is small at the downstream configuration's θ 0.3 and
substantial at θ 0.5. The change is exact, since the operator ignores `dd`, and
B2 depends on it, so B1 proceeds.

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
   ca.dd)` must go** — it currently forbids what this task does. Call it on
   `CartesianTaylorBasis` and `LaplaceKernel` as well (include
   `src/Canopy_CartesianTaylorBasis.hpp`). Today it runs only on
   `MonopoleBasis` and `LevelBlindBasis` (`:970-971`), and both keep `dd`:
   `MonopoleBasis`'s operator carries $F(\texttt{dd}) = 2^{\max(0,-\texttt{dd})}$
   (`tests/CanopyTest_MonopoleBasis.hpp:382-425`), so its trait is `true`.
   Without the two new calls, the `false` branch runs on no basis.
5. Leave `m2l_key_dd_max` and the sweep's `dd` range guard in place and
   unchanged. `LaplaceKernel` still needs both.
6. **Change B0b's contract to match.** `reportTwoScaleDdDuplicates` counts
   cross-level keys from `m2l_realized_keys()`, which holds canonical keys
   (`src/Canopy_DownwardSweep.hpp:1715-1716`). Once `CartesianTaylorBasis`
   zeroes `dd`, its cross-level count is 0 on both draws, and
   `EXPECT_GT( sum[1], sum[0] )` (`tests/tstDownwardSweep.hpp:1997`) fails
   `0 vs 0`. For a basis declaring `key_needs_dd = false`, assert instead that
   the cross-level count is **0 on every rank**, which checks that the trait
   reaches the table. Keep the graded-greater-than-two-scale contract on
   `LaplaceKernel` only. It sees the same tree, so the geometry claim is
   unchanged.
7. Re-run B0b's case and compare it line by line with the tables in
   `tree-opt-progress-log.md` `## B0b` (see the exit criterion).

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

All pass on SERIAL at ranks 1-6 and on HIP at ranks 1-4. Each line is compared
with the tables in `tree-opt-progress-log.md` `## B0b`. Those tables reproduce
exactly on both backends, so any difference comes from this change:

- On every `[b0b-dd]` line (both draws, θ 0.3 and 0.5) and every `[b0-dd]`
  line, `CartesianTaylorBasis`'s admitted count equals B0b's recorded CT
  *distinct* count, and its factor reads 1.0000.
- `CartesianTaylorBasis`'s `[dd-hist]` lines put every admitted key at
  `dd = 0`.
- `LaplaceKernel`'s admitted count, distinct count, factor, bytes and histogram
  are **unchanged** on every line. A drop in any of them is **R3** firing.
- B0's `[b0-dd]` lines (two-scale, θ 0.3) keep their admitted and distinct
  counts and factor for both bases, because CT's distinct count already equals
  its admitted count there.

Failure direction: a basis declaring `key_needs_dd = true` whose
`canonicalize_key` zeroes `dd`, and the converse, both fail
`expectKeyTraitsAgree` with a message naming the basis; verify by temporarily
mis-declaring one and seeing the named failure, then reverting.

**Met.** All six stems pass through `canopy_ctest` on SERIAL np 1-6
(`f3chRgEQqzd5`) and HIP np 1-4 (`f3chRgNMv9GK`): 37 and 25 entries, none over
budget, nothing cancelled. All 378 SERIAL count lines (`[b0b-dd]`, `[b0-dd]`,
`[dd-hist]`) were compared by script with B0b's job `f3ccGbJHwxPy`, and all 180
HIP lines with both B0b jobs. On every line `CartesianTaylorBasis`'s admitted
count, `demanded_ops` and bytes equal B0b's CT *distinct* figures, its factor
reads 1.0000, and its histogram is entirely at `dd = 0`. Every `LaplaceKernel`
line is identical to B0b's, and so are its `[b0b-cross]` sums. At np 1 CT goes
from 4162 to 4068 columns at θ 0.3 and from 26 702 to 20 286 at θ 0.5 on the
graded draw. `CartesianTaylorSolve`'s own fixture drops by 1.38-1.46x, with
fallback counts unchanged. Its SERIAL `[ct-solve]` deviation lines, and
`MultiSolve`'s `[multisolve-dev]`, `[multisolve-probe]` and `[fusedm2l-*]` lines,
are bit-identical to V1 (close)'s `f3cbCkKDsirK`. Both trait mismatches fail
`FarFieldContract.levelReachesTheKey` with a message naming
`CartesianTaylorBasis` (`f3chWq7XkJbH`, `f3chZbiYKp7h`). Zeroing `dd` in
`LaplaceKernel` drops its admitted count to B0b's *distinct* count on all 42
graded lines (`f3chc3fq5LqM`). After the revert, `f3cheVoEuy9Z` matched the
exit-criterion run line for line. See `tree-opt-progress-log.md` `## B1`.

---

### A2 — 2:1 tree balancing, configurable, default off — **DONE**

**Depends on:** A1 **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_TreeBuilder.hpp` (the balancing pass, at the end of
`build()`, and the knob's route in); `src/Canopy_Solver.hpp` (`FmmConfig`
member and its route to the builder); `tests/tstDownwardSweep.hpp` (the cases
on T1's draw, beside `testTwoScaleBalanceCost`: they need
`DownwardSweepTest::TwoScaleFixture`, its downward sweep, and the
touching-leaf helpers `balance_coarser_neighbours`, `balance_level_hist` and
`balance_simulate`, all in that file); `tests/tstTreeBuilder.hpp`
(builder-only cases: a value below 1 rejected, the pass bound, the
`max_depth` report); `scripts/tuolumne/serial_runtimes.tsv` (the `default`
rows of `TreeBuilder` and `DownwardSweep`). **This task first opens
`src/Canopy_TreePartitioner.hpp`** — see [Not read](#not-read).
**Reference:** the refinement loop (`src/Canopy_TreeBuilder.hpp:692-812`),
its leaf decision (`:804`), and its rule that a candidate with no particles is
not a cell (`:789`), with counts `MPI_Allreduce`d over ranks (`:778`);
`update()`'s Step 6 (`:1147-1241`) for the shape of a pass that iterates over
the finished cell set to a fixed point — **only the loop shape**: it refines
through `refine_leaf` (`:592-630`), which creates all eight children,
empty ones included, and the balancing pass must not; A1's model of this pass,
`balance_simulate` (`tests/tstDownwardSweep.hpp:2264`), and its neighbour
search, `balance_coarser_neighbours` (`:2198`); `FmmConfig`'s existing knobs
and their route to the builder (`src/Canopy_Solver.hpp:126`, `:204-208`).
**Do:**
1. Add `FmmConfig::tree_balance_max_level_delta`, defaulting to the **off**
   value, and route it to `TreeBuilder` as a new last constructor argument, so
   every existing construction is unchanged. Reject a value below 1 by
   throwing, as the sweep's other integer knobs do — a 0 would mean "every leaf
   at one depth", which is a request to abandon adaptivity and is far more
   likely a typo.
2. Implement the balancing pass as a **post-pass at the end of `build()`**,
   not as a change to the refinement loop's leaf test. Neighbours are leaves
   that **touch**, by a face, an edge or a corner (26 directions) — the
   definition A1 measured with. Repeatedly refine any leaf more than
   `max_level_delta` levels shallower than a touching leaf, until none is; the
   refinement is transitive, so it iterates. Refining a leaf creates **only
   its occupied children**, from particle counts reduced over all ranks as
   `build()` reduces them, so the cell list keeps holding occupied cells only
   and the result is the tree A1's model predicts. Bound the iteration count
   and throw if the bound is hit rather than looping — a non-terminating
   balance presents as a hang, which is the most expensive failure to diagnose.
   `Solver` changes the tree only through `build()`
   (`src/Canopy_Solver.hpp:340-701`); `update()` does not balance, and its
   declaration says so.
3. Do not let the pass refine past `max_depth`. A leaf at `max_depth` that is
   still unbalanced stays unbalanced; **report it loudly** rather than silently
   leaving the invariant broken, because every downstream claim about bounded
   `dd` rests on it.
4. Enumerate and update the consumers of the cell set the pass changes. At
   minimum the partitioner and the per-depth occupancy the sweep reads; verify
   by searching for consumers of `TreeBuilder`'s cell accessors rather than
   assuming this list is complete.
5. Add the cases. On T1's draw, at every np: with the knob off, the cell set is
   **bit-for-bit what it is today**; with it at 1, no two touching leaves differ
   by more than one level (`balance_level_hist`), and the balanced cell count
   equals `balance_simulate` at delta 1 on the same unbalanced tree — A1's
   recorded multipliers, 1.2735, 1.2798, 1.2353, 1.2489, 1.1843 and 1.0957 at
   np 1-6. Only these cases set the knob; there is no environment override.
6. **Record, not assert,** `m2l_n_fallback_pairs_range_guard()` knob-off against
   knob 1 per `(nprocs, rank)` on T1's draw, and state in the log whether
   balancing removed the refusals. A balance of touching leaves does not bound
   the depth difference of a pair across **empty space** — the geometry of
   T1's refusals (`tree-opt-progress-log.md` `## A1`) — so 0 is not a
   guaranteed outcome, and A3 reads the figure.
7. Record the partition's per-band imbalance (the depth bands of
   `fix-hang-rebalance.md` H2's balance constraints) knob-off against knob 1,
   per `(nprocs, rank)`.
8. **Do not regenerate the bit-for-bit reference data to make a test pass.**
   `LaplaceSolve.bitForBitArtifacts` compares hashes of internal artifacts
   against `tests/data/laplace_solve_P6.txt`, and
   `CANOPY_LAPLACE_SOLVE_REGENERATE` rewrites them in one step
   (`tests/tstLaplaceSolve.hpp:56-65`). Those hashes are regression-to-self and
   carry no accuracy claim, so a regeneration absorbs a real error exactly as
   readily as a legitimate tree change. With the knob at its default they must
   not move at all; if they move, that is this task failing its own first
   assertion.
9. Re-calibrate the `default` rows of `scripts/tuolumne/serial_runtimes.tsv`
   for `TreeBuilder` and `DownwardSweep`, the stems this task adds cases to.

The asserted cases run at `max_level_delta = 1`. A1 measured the cost at 1, 2
and 3 and recommends 1: at θ 0.3 the worst-case centre offset of a pair whose
depths differ by $k$, $\sqrt{3}\,(2^k + 2)/\theta$ half-widths at the finer
depth, is 23.1 at $k = 1$ and 34.6 at $k = 2$, against `M2L_KEY_OFFSET_MAX =
32`.

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

Every pre-existing line in the seven stems is unchanged against the jobs of
the task that last recorded it — on T1's draw, `tree-opt-progress-log.md`
`## A1` — and `LaplaceSolve.bitForBitArtifacts` passes with `tests/data`
untouched.

*Knob at 1* — the new cases, run by the same seven stems: on T1's draw at every
np, the largest touching-leaf level difference is at most 1 and the balanced
cell count equals `balance_simulate`'s at delta 1. The log records the
knob-off and knob-1 `range_guard` and per-band imbalance per `(nprocs, rank)`.
Failure direction: `tree_balance_max_level_delta = 0` throws from `FmmConfig`'s
route; a pass whose iteration bound is set below what a balance needs throws
rather than returning; and a distribution that cannot be balanced within
`max_depth` produces the loud report rather than a silently unbalanced tree.

**Met.** Flux jobs `f3cmue2op151` (SERIAL np 1-6) and `f3cmueBubbPm` (HIP
np 1-4) ran the seven stems through `canopy_ctest`, with a second
`DownwardSweep` pass. Every entry finished `completed` (48 of 48 and 32 of 32),
every `rc=0`, and the watchdog cancelled nothing. **Knob off:** all 2622 SERIAL
tagged lines are byte-identical to the baseline `f3cmTeajNv8o`, taken on the
unmodified `e06996d`. On HIP the structural lines are identical (1207 of 1284).
The 77 that differ are the accuracy-figure lines, and they differ from the HIP
baseline `f3cmTeofYejH` by as much as two unmodified HIP runs differ from each
other (R11). `LaplaceSolve.bitForBitArtifacts` passes with `tests/data`
untouched. **Knob 1** (`DownwardSweepTwoScale.balancePass`): at every np the
balanced tree equals `balance_simulate`'s, run in the same binary on the
knob-off tree. That holds key for key and leaf flag for leaf flag, with the
same cell count and passes. So the multipliers are A1's 1.2735, 1.2798,
1.2353, 1.2489, 1.1843 and 1.0957. The largest touching-leaf difference is 1,
nothing is stuck, and the figures are identical across ranks, across both
passes, and between HIP and SERIAL. The failure directions fire
(`TreeBuilder.balanceKnobRejectsBelowOne`,
`TreeBuilder.balancePassBoundAndDepthReport`). **Deviation:** a leaf at
`max_depth` can never be the shallower side of an unbalanced pair, so through
`build()` the stuck report is unreachable. It is exercised through a
test-only depth limit (`set_balance_depth_limit`). **Finding:** balancing does
not remove T1's refusals. It raises `range_guard` 1.6-3.5x per np. Tables and
messages are under `## A2` in the log.

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

### B2 — Quantize the root half-width so the operator cache survives a drifting box — **DONE**

**Depends on:** B1 **DONE**, V1 **DONE**.
**Fill in:** `src/Canopy_TreeBuilder.hpp` (the quantization in `build()`, a
stored root half-width and its `root_half_width()` accessor, and the knob's
route in); `src/Canopy_Solver.hpp` (`FmmConfig::quantize_root_half_width`, its
route to the builder, and `_push_root_half_width`); `tests/tstDownwardSweep.hpp`
(`TwoScaleFixture`'s root half-width, `:1617-1621`, and the retention case);
`tests/tstCartesianTaylorSolve.hpp` (the `[ct-cache]` and `[ct-solve]` prints'
root half-width, `:559-563` and `:642-646`, and the drift cases);
`tests/tstTreeBuilder.hpp` (the quantization arithmetic case); `README.md`
(the knob, and the "Future Optimizations" entry of step 7).
**Reference:** the cache-invalidation rule and the reasoning recorded on it
(`src/Canopy_DownwardSweep.hpp:450-460`, rationale block from `:414`); the root
half-width's derivation from the expanded bounding box
(`src/Canopy_TreeBuilder.hpp:606-635`); its re-derivation in
`Solver::_push_root_half_width` (`src/Canopy_Solver.hpp:780-791`); the
per-level `unit_w` array the operator builder indexes by `max_d`
(`src/Canopy_CartesianTaylorBasis.hpp:1187-1231`).
**Do:**
1. **Put the quantization behind `FmmConfig::quantize_root_half_width`,
   default `false`.** The root half-width is the tree's own geometry, so
   snapping it moves the cell set **for every basis**, not only for the one
   this chain is about. Left unconditional, this task would silently shift
   every existing result — the bit-for-bit hashes, both direct-sum arms and
   `MultiSolve`'s trajectory alike.
2. **Give the root half-width one source.** Today `build()` computes it as a
   local from the expanded `_root_box` (`src/Canopy_TreeBuilder.hpp:630-635`),
   and four readers re-derive it from `root_box()` the same way:
   `Solver::_push_root_half_width` (`src/Canopy_Solver.hpp:780-791`),
   `TwoScaleFixture` (`tests/tstDownwardSweep.hpp:1617-1621`) and the two
   `CartesianTaylorSolve` prints (`tests/tstCartesianTaylorSolve.hpp:559-563`,
   `:642-646`). Store the value `build()` stamps on the root cell and expose it
   as `TreeBuilder::root_half_width()`; switch all four readers to it. Leave
   `_root_box` the expanded bounding box: `needs_rebuild`
   (`src/Canopy_TreeBuilder.hpp:860-890`) and `_init_auto_softening`
   (`src/Canopy_Solver.hpp:797-815`) read it as a box and must not change.
   Search for other readers of `root_box()` rather than trusting this list.
   With the knob off the stored value equals today's local bit for bit, so
   this step moves nothing on its own.
3. **Quantize the root half-width**: round it **up** to the next power of two
   with `std::frexp`/`std::ldexp`, after the tolerance expansion. Up, never
   down — down would shrink the root cell below the particle extent. The root
   cell keeps its centre. Document on the declaration that the root cell is
   deliberately up to 2x wider than the expanded box, and what that costs: one
   extra potentially-empty level at the top of the tree (**R5**).
4. **Leave `set_root_half_width`'s invalidation unchanged.** It already returns
   without clearing when handed the value it holds
   (`src/Canopy_DownwardSweep.hpp:452-453`), so with quantization a rebuild
   whose box stays inside one octave keeps the whole cache, and one that
   crosses an octave clears it exactly as today.
5. **Assert the retention directly**: on one downward sweep, as `Solver`
   reuses its own across rebuilds, drive two builds of the **same particles**
   whose expanded box differs only through a larger symmetric `bb_tolerance_factor`
   (`TreeBuilder`'s constructor, `src/Canopy_TreeBuilder.hpp:165`; the
   expansion at `:610-622` keeps the centre), chosen so the expanded half-width
   stays inside one octave. Knob on, the quantized root half-width and so the
   tree and its key set are identical, and the per-build increment of
   `m2l_op_keys_built_count()` must be **0** on the second build while
   `m2l_n_unique_ops()` is unchanged. Knob off, the same pair changes the width
   the sweep is told, and the increment must equal `m2l_n_unique_ops()` —
   today's behavior. Change only the box factor, not `ncrit_tolerance_factor`.
6. **Assert the arithmetic is exact**: on a sweep of inputs, the quantized
   width is a power of two, is $\ge$ the input, is $< 2\times$ the input, and
   is returned unchanged when the input is already a power of two.
7. **Record cross-octave reuse in `README.md` "Future Optimizations"**:
   shifting every cached key's `max_d` by the change in the quantized exponent
   would keep the cache across an octave crossing as well, since the column for
   depth $d$ at width $W$ is the column for whichever depth now carries $W$. It
   is not done here, because a wrong shift reuses a column at the wrong width
   (**R4**).
8. **Give the drift trajectory an accuracy check, in both knob states.** The
   two `operatorCacheAcrossDrift*` cases assert no accuracy today
   (`tests/tstCartesianTaylorSolve.hpp:1041-1061`), and they run the trajectory
   on which the cache now survives rebuilds — so a column used at the wrong
   width would change only the `keys_built` figure those cases print and would
   corrupt the field silently (**R4**, **R9**). Parameterize the drift harness
   on the knob: the existing two cases stay knob-off, and two new cases,
   `operatorCacheAcrossDriftQuantizedThetaCanopy` and `...ThetaRef`, run it
   knob-on. Give all four the same direct-softened-sum comparison the gating
   arms use, each at a tolerance **measured on its own trajectory** rather than
   inherited from the gating arms, whose constants were not measured there.
   The knob-on pair also prints its per-build `keys_built` increment, which
   must read 0 on every step whose quantized root half-width did not change.
9. Re-calibrate the `default` rows of `scripts/tuolumne/serial_runtimes.tsv`
   for `TreeBuilder`, `DownwardSweep` and `CartesianTaylorSolve`, the stems
   this task adds cases to.

**Exit criterion:** five stems.

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

All five pass on SERIAL at ranks 1-6 and on HIP at ranks 1-4. The knob is set
only by the cases that need it, so every pre-existing case runs knob-off and
is unchanged: `LaplaceSolve`'s bit-for-bit hashes pass untouched, and on SERIAL
every `[ct-solve]`, `[multisolve-dev]`, `[multisolve-probe]` and
`[fusedm2l-*]` line is bit-identical to `tree-opt-progress-log.md` `## B1`'s
`f3chRgEQqzd5`. The retention case reads an increment of **0** knob-on and
`m2l_n_unique_ops()` knob-off; all four drift cases pass their measured
direct-sum bound; and the knob-on drift pair's `keys_built` increments are
recorded in the log beside the knob-off pair's.

Failure direction: a box change crossing an octave clears the cache rather
than reusing it, asserted by the increment equalling `m2l_n_unique_ops()` in
that case, knob on. Temporarily make `_push_root_half_width` re-derive the
width from `root_box()` again, so the sweep is told the unquantized width while
the tree is built at the quantized one, and confirm a knob-on drift case fails
its direct-sum bound; revert, and record the failure message.

**Met.** Flux jobs `f3cj6QNSmYfy` (SERIAL) and `f3cj6QWEwmd9` (HIP), on the
final binaries, profiling ON: all five stems pass, 30 of 30 entries at SERIAL
np 1-6 and 20 of 20 at HIP np 1-4. No entry went over budget and the watchdog
cancelled nothing. `LaplaceSolve.bitForBitArtifacts` passed with
`tests/data` untouched. A script compared the SERIAL `[ct-solve]`,
`[multisolve-dev]`, `[multisolve-probe]` and `[fusedm2l-*]` lines with
`f3chRgEQqzd5`. All 198 are identical, excluding only the new drift-arm
deviation lines and the knob-on configuration lines. `rootWidthQuantizationRetainsCache` read an
increment of **0** knob-on on all 31 `(nprocs, rank)` across both backends,
`m2l_n_unique_ops()` knob-off, and `m2l_n_unique_ops()` across an octave
(width 1 → 2) knob-on. The four drift cases pass bounds set at 2x their own
worst over both backends (3.76e-2, 1.92e-2, 3.65e-2, 1.54e-2). Knob on, builds
2-4 of the drift trajectory rebuilt 5.3 % (θ 0.5) and 7.1 % (θ 0.3) of
admitted columns, summed over SERIAL np 1-6; knob off, 100 %. **Not 0**, as
step 8 predicted: each build realizes some keys for the first time. The harness
instead asserts that the increment equals that first-seen count on every
unchanged-width build, and it held on all 336 builds per backend. Failure
direction `f3cj2wdB5vSb`: with `_push_root_half_width` re-deriving from
`root_box()`, both knob-on drift cases failed at every np (gradient 1.93 and
1.60 against 3.65e-2 and 1.54e-2) and nothing else failed. That was reverted and
rebuilt before the exit runs. See `tree-opt-progress-log.md` `## B2`.

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

**R4 — The sweep is told a width other than the one the tree was built at.**
`build_m2l_operators` builds every column from `unit_w[max_d]`, which comes from
the root half-width `set_root_half_width` is handed, while the cells come from
the half-width `TreeBuilder::build` stamped. If the two differ — a reader that
re-derives the width from the expanded `root_box()` after B2 quantizes it, or a
cached column kept across a width change — every column is built or reused at
the wrong scale. Arithmetically identical to R3 and presents identically: a
plausible field that is wrong, with no diagnostic pointing at the table.
Distinguishing measurement: B2 step 2 gives the width one source, its failure
direction shows a re-derived width failing the knob-on drift cases' direct-sum
bound, and its retention case pins both cache directions — 0 rebuilt on a reuse
that *should* happen, a full rebuild across an octave. Either cache assertion
alone would be satisfied by a cache that always reuses or always clears.

**R5 — The power-of-two root snap enlarges the box enough to matter.** Rounding
up can double the root half-width, which adds a level at the top of the tree and
shifts every per-depth width. Presentation: `occupied_depths` rising by one and
every realized key changing, so the whole table turns over once on the change —
which in the logs looks like a cache that stopped retaining. Distinguishing
measurement: the realized key *count* should be stable across the change while
the key *values* shift by one level; a change in the count is a tree change,
not a cache defect.

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
(`tests/tstMultiSolve.hpp:1214-1229`) names the bounds actually in force,
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
(`tests/tstCartesianTaylorSolve.hpp:1041-1061`), and they deliberately run a
longer trajectory than the gating arms so the bounding box drifts — which is
exactly the condition B2 changes the handling of. Presentation: B2 lands, the
`keys_built` figure improves, every test passes, and the field is wrong wherever
a column was used at the wrong width. Nothing in the suite would say so.
Distinguishing measurement: B2 step 8 gives those cases, and their knob-on
pair, a direct-sum deviation bound measured on that trajectory. **Until that exists, a `keys_built`
improvement from B2 is not evidence of correctness**, and the gating arms do not
cover it — they run a shorter trajectory on which the box barely drifts.

**R10 — A pinned tolerance absorbs a real degradation.** Every bound these
chains are checked against is a constant, and a constant set at a measured
deviation times 2 still passes a regression smaller than that factor.
`SolveFusedM2L.matchesPriorReference`'s potential bound (`5.0e-2`,
`tests/tstMultiSolve.hpp:1464`) is already 0.87 used by cancellation at
`|phi| -> 0`, so it can catch only a complete regression; its gradient bound
(`5.7e-4`, `:1465`) is 2x the worst measured. `CartesianTaylorSolve`'s
`theta_canopy` arm is pinned at `3.74e-02`, 2x its own worst figure. The six
`MultiSolve` trajectory sites (`pos_tol`/`vel_tol`, checked at `:1006-1013`)
are each 2x their worst over np 1-6, so at np 1, where the deviation is up to
1000x smaller, they bound little. Presentation: a tree or key change degrades
the far field by a few percent and every bound still passes. Distinguishing
measurement: record the **measured deviations**, not the pass/fail, before
and after any task that changes the tree or the key — the `[multisolve-dev]`,
`[multisolve-probe]`, `[fusedm2l-dev]` and `[ct-solve]` lines print on every
run and repeat bit for bit on SERIAL. A3's failure direction requires exactly
that, and it is the only reason an unchanged pass there means anything.

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
