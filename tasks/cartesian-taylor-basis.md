# A Cartesian-Taylor far-field basis for Canopy

**Status:** NOT STARTED

## Problem

Canopy's far field is a solid-harmonic expansion of $1/r$. The downstream
solver needs a far field for a **Plummer-softened** kernel,

$$
\varphi(r) = \left(r^2 + b\right)^{-1/2}, \qquad
K_l = -\partial_l \varphi = \frac{r_l}{(r^2 + b)^{3/2}}, \qquad b > 0,
$$

and no finite solid-harmonic expansion can represent it: that expansion rests on
the addition theorems, which require harmonicity, and the only isotropic
harmonic functions in 3D are $\mathrm{const}$ and $1/r$.

Canopy works around this today with a **near-field softening floor**. Any pair
closer than `near_softening_factor * eps` is refused by the MAC and forced into
the softened direct sum (`mac_satisfied`,
`src/Canopy_CommunicationPlan.hpp:347-359`; the knob is
`FmmConfig::near_softening_factor`, default `4.0`, `src/Canopy_Solver.hpp:78`).
Against the bare kernel the far field expands, dropping the blob costs a
relative error of $\sim\frac{3}{2}(\delta/r)^2$, so holding $10^{-2}$ needs every
far interaction at $r \gtrsim 12\delta$. At the downstream solver's
configuration — $\varepsilon = 0.025$ on a bubble of radius $0.25$ — that is
larger than the bubble, so nearly the whole domain falls into the near field and
the $O(N)$ advantage of the FMM is gone. The floor is not a MAC that can be
tuned around; a blob-unaware far-field operator forces it.

**What is being built.** `src/Canopy_CartesianTaylorBasis.hpp`: a real-coefficient
Cartesian-Taylor basis in which the softening rides inside $w = r^2 + b$ at every
derivative order, so the far field is regularized automatically and
`near_softening_factor = 0` becomes a viable configuration. It plugs into the
`FarField` template slot that `Solver` already carries
(`src/Canopy_Solver.hpp:125-134`) and changes no shared code.

**The end state.** A `Solver<Mem, Exec, double, p, NComps, CartesianTaylorBasis>`
runs a full pipeline at `near_softening_factor = 0` and reproduces a direct
softened sum to the accuracy the reference treecode documents, at the same
expansion order; today no such configuration exists at any accuracy, because
`LaplaceKernel` at `near_softening_factor = 0` evaluates an unsoftened far field
against a softened near field and produces a discontinuous, wrong answer.

### The target is the regularization unblock, not $10^{-10}$

The blob correction dies as $(\delta/R)^{2(k+1)}$ once $k$ correction orders are
kept, so a **low order beats the regularization problem** at standard
admissibility: one order pulls the required separation from $\sim 12\delta$ to
$\sim 4\delta$, two orders to $\sim 2.5\delta$, back inside ordinary FMM
well-separatedness. That is what makes `near_softening_factor = 0` viable, and it
is the whole goal here.

A far-field relative error of $10^{-10}$ is **out of reach for this basis** and
is not a goal of any task below. A Cartesian Taylor expansion truncated at order
$p$ has relative error $\sim (c\,w/R)^{p+1}$ with $c$ between $1$ and $\sqrt3$;
at standard admissibility $R/w \approx 3$ each additional order buys between
0.24 and 0.48 decades while the DOF count grows as $\binom{p+3}{3}\sim p^3/6$, so
$10^{-6}$ alone wants $p \approx 11$–$24$, i.e. 364 to 2925 coefficients per cell
([canopy-kernel-rec.md](canopy-kernel-rec.md), "Convergence per DOF"). Reaching
$10^{-10}$ is black-box FMM's job and has its own design
([canopy-bbFMM.md](canopy-bbFMM.md)). The basis that admits softening trivially
is the basis that does not scale to high accuracy, and both statements are true
at once.

$p$ occupies the existing `P_ORDER` template slot on `Solver`, which is already
documented as the basis's order knob — $P$ for solid-harmonic, $p$ for Taylor,
$n$ for Chebyshev (`src/Canopy_Solver.hpp:115-123`, `README.md:11-33`). No new
template parameter and no rename. The slot carries no units, so this basis states
on its own declaration that it reads `P_ORDER` as the Taylor order.

### The far field is three scalar passes, not a vector kernel

The downstream velocity is $[K \times \gamma]_i = \varepsilon_{ilm} K_l \gamma_m$
with $K_l = -\partial_l\varphi$. The cross product is bilinear, so it peels off:
the far field is **three independent scalar-$\varphi$ FMM passes**, one per
strength component $\gamma_m$, recombined by $\varepsilon_{ilm}$ at the end.
Canopy's `NComps` already provides exactly those three independent passes, and
`l2p_evaluate` already returns a potential and a gradient per component. **The
$\varepsilon$ recombination is the downstream solver's and is out of scope
here**; this basis delivers $\varphi$ and $\nabla\varphi$ per component and
nothing recombines them inside Canopy.

### Out of scope

- The $\varepsilon_{ilm}$ recombination above.
- The black-box-FMM basis, single-precision support for this basis, and any
  order-adaptive or compressed-operator variant.
- `src/Canopy_P2P.hpp`. It never calls the far field and already runs the
  softened near kernel — `set_softening` takes a length and squares it into
  `_softening2` (`:103-106`), used as `r^2 + eps2` at `:799-803` and `:1082`.
  Nothing here changes it.
- `src/Canopy_DownwardSweep.hpp`. No task below edits it. In particular the
  cache-clearing behaviour described under **R6** is measured, not fixed.
- Repairing `tests/tstLaplaceKernel.hpp` or `Canopy_Test_P2P_*`, which do not
  compile; the `regression` suite's pre-existing failures and its intermittent
  np=3 hang; the partitioner's run-to-run non-determinism. All four are recorded
  in `README.md` "Known Issues".
- The per-source-cell storage gap in the M2L three-stage split: `m2l_pre_cell` is
  handed team scratch, which is per-target, so once-per-source-cell work that
  outlives the team has nowhere to live. **This basis has no per-source work to
  hoist** — it contracts the source multipole directly — so the gap does not bite
  and no task widens it.

## Approach

### What a Cartesian-Taylor far field is, completely

Everything below is derived from one scalar and one identity. Provenance:
[canopy-questions.md](canopy-questions.md) §§1-4, which is the reference author's
statement of the recurrences the reference treecode implements. Cite that section
on every routine that transcribes a formula from it.

**The ladder.** With $w = r^2 + b$ and $P_m = w^{-(2m+1)/2}$, so that
$P_0 = \varphi$,

$$
\partial_a P_m = -(2m{+}1)\, r_a\, P_{m+1}.
$$

Because $\partial_a w = 2 r_a$, differentiating never leaves the ladder: it
advances $m$ and drops an $r_a$. **$b$ is inert under it** — it appears only
inside $w$ — which is the entire reason this basis is blob-aware and the
solid-harmonic one cannot be.

**The tensors.** Writing $b_k = \partial^k \varphi$ for a multi-index $k$, the
ladder gives, in closed form through $|k| = 3$:

$$
\begin{aligned}
b_\emptyset &= P_0 \\
\partial_a \varphi &= -r_a P_1 \\
\partial_a \partial_b \varphi &= -\delta_{ab} P_1 + 3 r_a r_b P_2 \\
\partial_a \partial_b \partial_c \varphi &=
  3(\delta_{ab} r_c + \delta_{ac} r_b + \delta_{bc} r_a) P_2
  - 15\, r_a r_b r_c P_3
\end{aligned}
$$

These four are the **order-2 oracle**: they are what `_expansion_batch` in the
reference treecode implements, up to the sign and index shift $K = -\nabla\varphi$
introduces, and they are hand-coded in-repo as the per-operator test's reference.

**The recurrence**, for arbitrary order, solving $w\,\partial_i \varphi = -r_i \varphi$
order by order:

$$
w\, b_{k+e_i} = -r_i b_k - k_i b_{k-e_i}
  - 2 \sum_j k_j r_j\, b_{k+e_i-e_j}
  - \sum_j k_j (k_j - 1)\, b_{k+e_i-2e_j}
$$

with $e_i$ the unit multi-index in direction $i$ and any $b$ carrying a negative
component identically zero. Every coefficient at order $|k|+1$ comes from orders
$|k|$ and $|k|-1$; $b$ enters only through the leading $w$. The reference author
states that this reproduces $b_\emptyset$, $b_{e_i}$ and $b_{2e_i}$ above, which
is the cross-check T1 makes into an assertion rather than a claim.

**The five operators.** With $d = y - c_B$ for a source particle $y$ in box $B$,
$a = x - c_A$ for a target particle $x$ in box $A$, $s$ a center offset, and
$R = c_A - c_B$:

| Operator | Formula | Sees the kernel? |
| --- | --- | --- |
| P2M | $M_q^B = \sum_{j \in B} \dfrac{d_j^q}{q!}\, s_j$ | no |
| M2M | $M_q^{\rm parent} = \sum_{q' \le q} \dfrac{s^{q-q'}}{(q-q')!}\, M_{q'}^{\rm child}$ | no |
| M2L | $\ell_p^A = \sum_q (-1)^{|q|}\, b_{p+q}(R)\, M_q^B$ | **yes, and only here** |
| L2L | $\ell_p^{\rm child} = \sum_{p' \ge p} \dfrac{s^{p'-p}}{(p'-p)!}\, \ell_{p'}^{\rm parent}$ | no |
| L2P | $u(x) = \sum_p \dfrac{a^p}{p!}\, \ell_p^A$ | no |

M2M and L2L are plain binomial Taylor shifts and are identical to any Cartesian
FMM. **M2L is the sole kernel-touching operator**, and $b$ rides inside
$w = |R|^2 + b$, so the translation is regularized automatically.

The $1/q!$ lives in the moment and the $1/p!$ in the L2P evaluation; the $b_k$
themselves are **raw derivatives with no factorial**. That split is what makes
the $(-1)^{|q|}$ multiplier above correct, and it is the one convention the
reference author flags as needing care — getting it subtly wrong produces a
plausible-looking field rather than an obvious failure. **T3 pins it against the
reference rather than asserting it.**

**L2P's gradient is analytic**, by shifting the multi-index:

$$
\partial_a u(x) = \sum_p \frac{a^{p - e_a}}{(p - e_a)!}\, \ell_p^A
$$

which deletes the central finite difference at
`src/Canopy_LaplaceKernel.hpp:1378-1407` **for this basis only**. `LaplaceKernel`
keeps its FD path; nothing in this document touches that file.

**The M2L needs $b_k$ to order $2p$.** $|p| \le p_{\rm order}$ and
$|q| \le p_{\rm order}$, so $b_{p+q}$ runs to $|p+q| = 2 p_{\rm order}$. The
ladder is evaluated to $2p$, not $p$, everywhere an operator is built.

### The sign of $R$, which the key does not give directly

This is the single most likely place to lose a session. The sweep's M2L key
carries an integer offset

$$
(ii, jj, kk)\cdot w_{\rm unit}[\texttt{max\_d}] \;=\; c_{\rm source} - c_{\rm target}
$$

— **source minus target** — built at `src/Canopy_DownwardSweep.hpp:1464-1481`,
with `w_unit[d] = w_root / 2^d` the half-width at the deeper of the two depths.
The M2L formula above wants $R = c_A - c_B = c_{\rm target} - c_{\rm source}$, so

$$
R = -\,(ii, jj, kk)\cdot w_{\rm unit}[\texttt{max\_d}].
$$

The negation is not optional and is not absorbed anywhere else. A useful
independent check: $\varphi$ is even, so $b_n(-R) = (-1)^{|n|} b_n(R)$, and the
two spellings

$$
\ell_p = \sum_q (-1)^{|q|} b_{p+q}(R)\,M_q
\qquad\text{and}\qquad
\ell_p = (-1)^{|p|} \sum_q b_{p+q}(S)\,M_q,\quad S = -R
$$

must agree numerically. T3 asserts that they do; disagreement means the parity or
the sign has been applied twice.

`m2l_translate` is handed the same quantity in physical form — the fallback call
site computes `dx = src_ci.center[0] - target_ci.center[0]`
(`src/Canopy_DownwardSweep.hpp:2338-2340`), again source minus target — so the
same negation applies there, and the two paths must produce the same operator or
`total_fallback_pair_count()` becoming non-zero silently moves the answer (**R4**).

### `key_needs_level = true`, necessarily

The operator is **physical**: $R$ carries a length, and $b$ carries the absolute
length $\sqrt{b}$. Softening destroys the scale invariance the solid-harmonic
operators exploit, so there is no normalization that makes the operator a
function of $(dd, ii, jj, kk)$ alone. The basis therefore declares
`key_needs_level = true` and `canonicalize_key` is the identity, which is what
makes `unit_w[k.max_d]` meaningful in `build_m2l_operators`. A basis that zeroed
`max_d` would index `unit_w[0]` and get $R$ wrong by a power of two on every
pair below the root.

This is the branch of the key contract that `MonopoleBasis` already exercises;
`LaplaceKernel` takes the other one. It has one consequence worth naming
separately — see **R6**.

### The memory question is settled for this basis

`bytes_per_key` is $\binom{p+3}{3}^2 \times 8$: **800 B at $p=2$**, 9.8 KB at
$p=4$. Against the 2 GB default budget the operator count cap
(`M2L_OP_COUNT_CAP = 32768`) binds first by orders of magnitude, so the byte
budget is not a constraint here and no task tunes it. For scale, T7 measured
**4628** level-carrying keys on a 694-cell ncrit-4 tree and 686 level-blind keys
on the 103-cell Laplace-solve tree, with a level-carrying key costing **1.8x**
the key count of a level-blind one on the same tree
([abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md),
§T7). At $p=2$, 4628 keys is 3.7 MB per rank. The concern that a per-key operator
table is unaffordable belongs to the black-box basis at $n \ge 6$, not to this one.

### Verification strategy

Two instruments, and **neither is `ctest -L regression`**.

- **The Laplace-solve gate**, `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL`.
  Three checks: bitwise identity of four artifacts at np 1-2, cross-rank
  agreement at np 2-6, and a direct-sum match at np 1-6. This basis adds a header
  and changes no shared arithmetic, so the gate must stay green **unchanged, with
  no reference-data regeneration**. It is the guard that a new basis has not
  perturbed the existing one.
- **A new conformance suite** for this basis, in the `unit` tier. Two test files;
  see [Conventions](#conventions).

The `regression` suite is not run by any task here. It does not pass on unmodified
code, it hangs intermittently at np=3, and the Laplace-solve gate is the sharper
and far cheaper instrument; the measured baseline and the reasoning are in
[abstract-solver-backend.md](abstract-solver-backend.md), **Deliberate
deviations**. Promoting any new test into `regression` requires confirming with
the user first, per `CLAUDE.md`.

**The np 3-6 partitioner wobble.** `TreePartitioner::partition_leaves` uses
Zoltan2 `multijagged`, documented non-deterministic at
`src/Canopy_TreePartitioner.hpp:417-419`, and the cut can move between
invocations inside a single process. A np 3-6 direct-sum or cross-rank deviation
that moves only in its 9th or later significant figure and stays under its pinned
tolerance is this, and **not** a finding. Attribute it from the job's own
`ctest -V` log in three steps, stopping at the first that answers:

1. Compare the three test bodies' per-rank `n_unique_ops` lines at that rank
   count. Two bodies of one job over one binary drawing two different cuts is the
   non-determinism measured directly.
2. If the three bodies agree with each other, compare the cut against the cuts
   [abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md)
   records. A cut already paired with this same moved figure in an earlier
   section settles it.
3. If the cut agrees too, check whether the figure that moved is a *cross-rank*
   one while every direct-sum figure holds to all 17 digits. That combination is
   reassociation under an unchanged key count and is not attributable to a source
   change at all.

Only if the direct-sum figures move as well is a stash-and-rebuild control run
from unmodified `HEAD` worth the wall time. The np 1-2 bitwise half of the gate
is unaffected throughout.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Header name | `src/Canopy_CartesianTaylorBasis.hpp` | `Canopy_<Name>Basis.hpp` distinguishes a basis from `Canopy_LaplaceKernel.hpp`, which keeps its name. Add it to `HEADERS_PUBLIC`, `src/CMakeLists.txt:3-18`. |
| `Scalar` | `double` only, by `static_assert` | A softened kernel has no scale invariance, so this basis carries **physical** operators keyed by level; the FP32 conditioning argument the solid-harmonic width normalizations exist for does not transfer. Spell it exactly as `MonopoleBasis` does (`tests/CanopyTest_MonopoleBasis.hpp:97-99`). |
| Trait naming | `snake_case`, `static constexpr` or typedef | Matches `num_coeffs_per_cell`, `m2l_num_src_coeffs`. |
| Operator naming | `snake_case`, `KOKKOS_INLINE_FUNCTION static` | Matches `p2m_contribution`, `m2m_translate`, `l2l_translate`, `l2p_evaluate`. |
| Host-side operator construction | plain `static`, **not** `KOKKOS_INLINE_FUNCTION` | `build_m2l_operators` and `build_aux_tables` run once on host and may allocate. Marking them device-callable would forbid it. |
| Units and conventions on declarations | **required** | Every width states half-width vs full width; every offset states its sign (`source − target` or the reverse); every operator states what it consumes and produces. None of it is recoverable from the code, and the $R$ sign above is the reason this rule is load-bearing here. |
| Provenance comments | **required** on anything transcribed | Name [canopy-questions.md](canopy-questions.md) and the exact § on every routine carrying one of its formulas, as `src/Canopy_LaplaceKernel.hpp:265` and `:676` already do for Greengard. |
| Single source of truth for an operator value | required | Both the device path and any host reference must reach an operator value through **one** function, as `MonopoleBasis::m2l_operator_entry` is for its table build and its host reference. This is what makes a conformance comparison exact rather than merely close. |
| Guarding against `-ffp-contract` | required where a host reference must match a device kernel bitwise | `acc += T * M` as one statement is contractible to an FMA under the default `-ffp-contract=on`, and nothing guarantees the same decision in a `Kokkos::parallel_for` lambda and a plain host loop. Split the product into a named local in one function both callers reach, as `MonopoleBasis::m2l_accumulate` does. |
| Per-operator test | `tests/tstCartesianTaylor.hpp`, name `CartesianTaylor` in `UNIT_SERIAL_TESTS` (`tests/CMakeLists.txt:36-39`) | Host-only math over no MPI. Target becomes `Canopy_Test_CartesianTaylor_SERIAL` (`cmake/test_harness/test_harness.cmake:104-112`). |
| Solve test | `tests/tstCartesianTaylorSolve.hpp`, name `CartesianTaylorSolve` in `UNIT_MPI_TESTS` (`tests/CMakeLists.txt:48-58`) | Target becomes `Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`, tests `..._np_<N>` for N in 1-6, CTest label `unit`. |
| Adding a name to `tests/CMakeLists.txt` | run `make cmake_check_build_system` **first** | `make -j 4 <new target>` otherwise fails with "No rule to make target": make errors out before it regenerates, because the target is not in the current Makefile. Regenerating preserves the existing cache. |
| Build command | `make -j 4 <target>` for exactly the targets a task's exit criterion names | A bare `make -j` has been SIGKILLed on the login node, and a whole-tree build cannot exit 0 in any case — `Canopy_Test_LaplaceKernel_*` and `Canopy_Test_P2P_*` do not compile and are out of scope. |
| Build tree | `build-tuolumne/`, **never reconfigured** | It is the Laplace-solve gate's configuration. `Canopy_ENABLE_PROFILING` is `OFF` there, so any profiling figure comes from `build-tuolumne-prof/` instead. |
| Formatting | do not run `clang-format` | Per `CLAUDE.md`. |
| Checkpoint commits | named per task below | Per `CLAUDE.md`: a later failure rolls back to the nearest checkpoint. |

### Deliberate deviations

- **`m2l_post_cell` is not a no-op.** `m2l_pre_cell` is — there is no per-source
  work to hoist — but `m2l_post_cell` is **the only stage that writes to the
  locals view**, and it carries the accumulator flush. Both existing bases write
  it that way (`src/Canopy_LaplaceKernel.hpp:1155-1194`,
  `tests/CanopyTest_MonopoleBasis.hpp:785-819`). A no-op `m2l_post_cell` compiles
  and leaves every local zero.
- **`m2l_key_dd_max = 6`, for a different reason than `LaplaceKernel`'s 6.**
  There, 6 is a precision bound: the scale-normalized operator carries a residual
  factor reaching $2^{j|dd|}$, and a basis carrying physical operators "inherits
  neither" bound (`src/Canopy_LaplaceKernel.hpp:689-702`). Here the guard merely
  bounds the key space; 6 is chosen so that the set of pairs routed to the
  fallback path is the same one every other basis in the repository sees, which
  keeps `total_fallback_pair_count()` comparable across bases. It is not a
  precision claim.
- **The tolerance is measured and pinned, not asserted from theory.** The
  accuracy exit criterion pins the achieved deviation as a named constant beside
  the bound, the way `LS_DIRECT_SUM_TOL` is pinned with its provenance comment
  (`tests/tstLaplaceSolve.hpp:155-187`). A loose bound that a regression could
  drift inside of without failing is not a gate.
- **The conformance test does not reuse `tests/tstFarFieldContract.hpp`.** That
  file's fixture is built around `MonopoleBasis` at `BASIS_NCOMPS = 2` with two
  sets, and its host reference is an exact telescoping sum that exists because a
  monopole local is constant over its cell. Neither property holds here. It stays
  the conformance gate for the *contract*; this basis gets its own accuracy gate.
- **`sets_per_component = 1`.** A Taylor local is one set of $\binom{p+3}{3}$
  coefficients per component. At 1 the `(component, set)` flattening collapses to
  `c` exactly and the locals view's third extent is `NComps`, as it was before
  that trait existed.

## Current state

- **The `FarField` slot exists and is the whole contract surface.** `Solver` and
  `createSolver` take `template <class, int, int> class FarField = LaplaceKernel`
  (`src/Canopy_Solver.hpp:125-127`, `:816-825`) and `kernel_type` is
  `FarField<Scalar, P_ORDER, NComps>` (`:134`). A basis is admissible iff that
  instantiation completes.
- **`Solver`'s member bodies have never been instantiated on a
  non-`LaplaceKernel` basis.** The existing compile-only test forces the *class*
  to be complete via `sizeof`, which instantiates the data members and therefore
  the sweeps and their class-scope guards — but **not** `solve()`, which is
  compiled only for instantiations something actually calls, and nothing calls a
  `MonopoleBasis`-backed solver. See **R7**.
- **`M2LKernelParams` reaches the basis**, carrying the softening as a **LENGTH**
  $\varepsilon$; the kernel's $b$ is $\varepsilon^2$
  (`src/Canopy_FarFieldContract.hpp:71-129`). It is the *effective* softening:
  `FmmConfig::softening < 0` selects distribution-based auto-softening, and what
  arrives is the length actually in force. `Solver::_push_m2l_kernel_params`
  (`:733-745`) sets both sweeps in one call.
- **`unit_w` reaches the operator builder.** `Solver::_push_root_half_width`
  (`:756-770`) hands `DownwardSweep` the largest of the three half extents of
  `root_box()`, and the sweep turns it into `unit_w[d] = w_root / 2^d`.
- **The operator cache persists across topology changes.** Measured at T9: after
  one solve, `invalidate_interaction_list()`, and a second solve,
  `interaction_list_build_count() == 2` while `m2l_op_keys_built_count()` equals
  the cache size — **zero keys rebuilt** — at every rank of every rank count.
- **No Cartesian-Taylor code exists.** `src/Canopy_CartesianTaylorBasis.hpp` does
  not exist, and no file in the repository evaluates $\partial^k\varphi$ at any
  order. The order-2 tensors exist **only** in the reference treecode, outside
  this repository; the in-repo oracle is the closed form in
  [canopy-questions.md](canopy-questions.md) §2, hand-coded by T1.
- **The only complete worked non-harmonic basis is
  `tests/CanopyTest_MonopoleBasis.hpp`** (910 lines). It is the shape to copy:
  real `coeff_type`, `scalars_per_coeff = 1`, a struct-template
  `aux_tables_type`, `key_needs_level = true`, and every contract member present
  with its units stated. `LaplaceKernel` is the only single-set example.
  `MonopoleBasis` is a *fixture*, not a method — its truncation error is $O(1)$ —
  so copy its structure, never its mathematics.

### The reference implementation, and what it fixes

`treecode.py` in the downstream solver's repository
(`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`) is a
Barnes-Hut treecode whose far field is the blob-aware expansion this basis
generalizes into an FMM. It is **not in this repository** and no task depends on
having it; the in-repo oracle is [canopy-questions.md](canopy-questions.md) §2.
Four facts from it are used below and are stated here so no task has to go
looking:

| Fact | Value | Source |
| --- | --- | --- |
| Expansion order | `order = 2` (moments G/D/Q, i.e. orders 0/1/2) | `treecode.py:103`, `_expansion_batch` at `:56-79` |
| MAC | `node.radius < theta * s`, `theta = 0.3` | `treecode.py:99`, `:121` |
| Leaf size | `ncrit = 64` | `treecode.py:104` |
| Documented accuracy | **~1e-3 relative velocity** at `theta = 0.3` | `README.md:71`, `PHYSICS.md:137`; measured 5.9e-4 / 1.1e-3 / 9.4e-4 at N = 642 / 2562 / 10242 in `PARALLELIZATION.md:24-30` |

`_expansion_batch(R, G, D, Q, blob, order)` **is** the $\ell_0$ local coefficient
— the value at the box center — for a single source box. Extending it to a real
FMM means keeping that as $\ell_0$ and adding the $p \ge 1$ coefficients, which
are just higher $b_{p+q}(R)$ off the same ladder.

**Its `blob` is not $\varepsilon$.** `blob` is the quantity added to $r^2$, and at
the solver's default `use_matlab_blob = True` it equals `eps`, not `eps**2`
(`zmodel3d/mesh_solver.py:394`). Canopy's `M2LKernelParams::softening` is a
length whose square is $b$. **Anything comparing a Canopy number against a
reference number must match $b$, not $\varepsilon$.**

**Its admissibility is not Canopy's.** The reference uses a Barnes-Hut radius MAC
at $\theta = 0.3$; Canopy uses $R^2\theta^2 > 3(w_a + w_b)^2$ at $\theta = 0.5$
(`src/Canopy_CommunicationPlan.hpp:340-346`), which admits pairs the reference
would refuse. Taylor truncation goes as $(c\,w/R)^{p+1}$, so the two
configurations do not deliver the same accuracy at the same order, and the 1e-3
figure transfers only at matched admissibility. T4 measures both.

## Progress log

[cartesian-taylor-basis-progress-log.md](cartesian-taylor-basis-progress-log.md)
holds what each session actually did: decisions and what forced them, signature
changes, measured numbers, and what only running revealed. **Read it before
implementing any task, before changing a signature this document names, and
before reopening anything this document states flatly** — a task marked
`**DONE**` may have been done differently than it is written here, and the log is
where that is recorded. Append a `## <task ID>` section at the end of every task,
ending with an `**Affects:**` line naming the later task IDs the entry changes.

## The complete contract a `FarField` type must supply

This is the full set, enumerated rather than cited, because it is larger than any
single existing document states and a missing member is a compile error deep
inside a sweep. Every member below is reached by `UpwardSweep`, `DownwardSweep`,
`P2P` or `Solver`, or is required to keep one of them well-formed. Signatures are
as actually built, not as earlier designs specified them.

**Types and aliases**

```cpp
using scalar_type           = Scalar;   // both sweeps re-export it
using coeff_type            = Scalar;   // real for this basis
using component_scalar_type = Scalar;   // the real scalar MPI sees
static constexpr int scalars_per_coeff = 1;

template <class MemorySpace>            // ALIAS TEMPLATE, not a typedef
using m2l_operators_type =
    Kokkos::View<coeff_type***, Kokkos::LayoutLeft, MemorySpace>;

template <class MemorySpace>            // STRUCT TEMPLATE; no sweep names a member
struct aux_tables_type { /* ... */ };

template <class ScratchSpace>
using m2l_accumulator_type =
    Kokkos::View<scalar_type*, ScratchSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
```

`m2l_operators_type` **must** be an alias template: the sweep spells it
`typename KernelType::template m2l_operators_type<memory_space>`, and
`DownwardSweep` instantiates it at two spaces at once — `memory_space` for the
device table and `Kokkos::HostSpace` for the persistent operator cache. A plain
typedef does not compile. `aux_tables_type` is likewise a struct template for the
same spelling reason.

**Compile-time constants**

```cpp
static constexpr int  max_order            = P_ORDER;        // the Taylor order p
static constexpr int  num_coeffs_per_cell  = binom(p+3, 3);
static constexpr int  num_components       = NComps;
static constexpr int  sets_per_component   = 1;
static constexpr int  m2l_num_src_coeffs   = binom(p+3, 3);
static constexpr int  m2l_key_dd_max       = 6;
static constexpr bool key_needs_level      = true;
static constexpr std::size_t bytes_per_key =
    std::size_t(num_coeffs_per_cell) * std::size_t(m2l_num_src_coeffs)
        * sizeof(coeff_type);
static constexpr Canopy::M2LOverflow m2l_overflow_policy =
    Canopy::M2LOverflow::PerPairTranslate;

KOKKOS_INLINE_FUNCTION
static constexpr std::size_t m2l_scratch_bytes( int n_comps );
```

**`m2l_scratch_bytes` must stay `constexpr`.** The sweep assigns it to a
`constexpr size_t` and derives the `TeamVectorRange` zero-fill bound from it, so
every extent inside the M2L stages stays a compile-time constant. Making it — or
`num_coeffs_per_cell`, or `sets_per_component` — a runtime value deoptimizes the
fused kernel with no correctness signal at all (**R3**). `n_comps` is a parameter
only so the sweep can size scratch without reaching into the basis's template
arguments; it is only ever passed `num_components`.

The sweep hands the stages **raw, zero-filled bytes** and relies on all-zero
bytes being the accumulator identity. That holds for IEEE-754 binary64, whose
all-zero representation is $+0.0$.

**Required `static_assert`s on the basis itself**

```cpp
static_assert( std::is_same<Scalar, double>::value, "...must be double" );
static_assert( sizeof( coeff_type ) ==
                   scalars_per_coeff * sizeof( component_scalar_type ), "..." );
```

**Host functions**

```cpp
template <class Key>
static Key canonicalize_key( Key k );          // identity here; host-only

template <class MemorySpace>
static aux_tables_type<MemorySpace>
build_aux_tables( int order, const Canopy::M2LKernelParams& kernel_params );

template <class KeyType, class AuxType, class OpsView>
static void build_m2l_operators( const KeyType* keys, int n_keys,
                                 const double* unit_w, int n_levels,
                                 const Canopy::M2LKernelParams& kernel_params,
                                 const AuxType& aux, const OpsView& ops );
```

`canonicalize_key` is a **function template on the key type**, deliberately:
`M2LKey` is a nested type of `DownwardSweep<…, KernelType>`, so a basis cannot
name it without a circular dependency. The sweep passes its own `M2LKey` and
`Key` is deduced.

`build_m2l_operators` **replaced** the older `m2l_build_operator`, which took one
key's four integers and no kernel parameters. A basis written against that name
will not compile. It is a **plain static member**, not `KOKKOS_INLINE_FUNCTION`,
so that a basis needing LAPACK here may call it. `ops` is
`(Nt, Ns, n_keys)` with column $j$ belonging to `keys[j]`, arriving as a
`LayoutLeft` contiguous range subview, and it is allocated
**`WithoutInitializing`** — every entry of every column must be written.

**Device operators**, all `KOKKOS_INLINE_FUNCTION static`:

```cpp
template <class MSliceType>
static void p2m_contribution( const Scalar (&charges)[NComps],
                              Scalar dx, Scalar dy, Scalar dz, Scalar w_self,
                              const MSliceType& M_out );

template <class TeamMember, class MView, class AuxType, class MParentType>
static void m2m_translate( const TeamMember&, const MView& M_full,
                           int child_cell, Scalar dx, Scalar dy, Scalar dz,
                           Scalar w_child, Scalar w_parent, const AuxType& aux,
                           const MParentType& M_parent_out );

template <class TeamMember, class MView, class OpsType, class ScratchView>
static void m2l_pre_cell( const TeamMember&, const MView& M_full,
                          int source_cell, const OpsType& ops,
                          const ScratchView& scratch );

template <class TeamMember, class MView, class OpsType, class ScratchView>
static void m2l_core( const TeamMember&, const MView& M_full, int source_cell,
                      const OpsType& ops, int op_idx,
                      const ScratchView& scratch );

template <class TeamMember, class ScratchView, class LView, class OpsType>
static void m2l_post_cell( const TeamMember&, const ScratchView& scratch,
                           const LView& L_out, int target_cell,
                           const OpsType& ops );

template <class TeamMember, class MView, class AuxType, class LTargetType>
static void m2l_translate( const TeamMember&, const MView& M_full,
                           int source_cell, Scalar dx, Scalar dy, Scalar dz,
                           Scalar w_source, Scalar w_target, const AuxType& aux,
                           const LTargetType& L_target_out );

template <class TeamMember, class LView, class AuxType, class LChildType>
static void l2l_translate( const TeamMember&, const LView& L_full,
                           int parent_cell, Scalar dx, Scalar dy, Scalar dz,
                           Scalar w_child, Scalar w_parent, const AuxType& aux,
                           const LChildType& L_child_out );

template <class LView, class GradAccess>
static void l2p_evaluate( const LView& L_full, int leaf_cell,
                          Scalar dx, Scalar dy, Scalar dz, Scalar w_self,
                          Scalar (&phi_out)[NComps], const GradAccess& grad_out,
                          bool compute_gradient );
```

**`m2l_translate` must be a real implementation, not a stub.** The basis selects
`M2LOverflow::PerPairTranslate`, which is the enumerator that *requires* it; the
other enumerator fails a class-scope `static_assert` in `DownwardSweep`. It must
also agree with the table path, or which pairs overflow changes the answer
(**R4**).

**Offset and width conventions at every call site**, read from the sweeps:

| Operator | `dx, dy, dz` | Widths | Call site |
| --- | --- | --- | --- |
| `p2m_contribution` | particle − cell center ($d$) | `w_self` = leaf half-width | `src/Canopy_UpwardSweep.hpp:470-481` |
| `m2m_translate` | child center − parent center ($s$) | both half-widths | `src/Canopy_UpwardSweep.hpp:533-539` |
| `m2l_translate` | source center − target center ($-R$) | both half-widths | `src/Canopy_DownwardSweep.hpp:2338-2345` |
| `l2l_translate` | child center − parent center ($s$) | both half-widths | `src/Canopy_DownwardSweep.hpp:2389-2397` |
| `l2p_evaluate` | particle − cell center ($a$) | `w_self` = leaf half-width | `src/Canopy_DownwardSweep.hpp:2658-2668` |

Every width is a **half-width** — half the side length of the cell's cube,
matching `Canopy::CellInfo::half_width`. This basis carries physical operators
and ignores all of them; they are in the signatures because the sweeps hand them
to every basis.

**The six sweep guards a basis must satisfy**, all class-scope:

| Guard | Location |
| --- | --- |
| `sizeof(coeff_type) == scalars_per_coeff * sizeof(component_scalar_type)` | `src/Canopy_UpwardSweep.hpp:74-79` |
| agreement with `detail::coeff_traits` | `src/Canopy_UpwardSweep.hpp:85-93` |
| the `sizeof` relation again | `src/Canopy_DownwardSweep.hpp:124-129` |
| the `coeff_traits` agreement again | `src/Canopy_DownwardSweep.hpp:135-143` |
| `sets_per_component >= 1` | `src/Canopy_DownwardSweep.hpp:154-158` |
| `m2l_overflow_policy == PerPairTranslate` | `src/Canopy_DownwardSweep.hpp:572-581` |

A real `coeff_type = double` with `scalars_per_coeff = 1` is the case
`detail::coeff_traits`' **primary** template covers, so `coalesced_view_exchange`
and both shared-cell Allreduces work with no specialization.

**No task below adds a class-scope `static_assert` to a sweep.** If one ever
does, `tests/tstFarFieldContract.hpp`'s permanent
`#ifdef CANOPY_TEST_EXPECT_COMPILE_FAILURE` block needs its own new basis for it:
clang reports only the **first** failing class-scope assert per class
instantiation, so a case folded into one of the four bases already there would
never be reached. The block currently carries four bases covering the six guards
above, and the by-hand build command is in that file's header comment
(`tests/tstFarFieldContract.hpp:130-151`).

## Running anything on tuolumne

Every executable goes in a batch script submitted to flux. Copy the preamble
verbatim from `scripts/tuolumne/run_ctest_t11.flux` and keep all of it: the spack
env activation, `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000` (the Cray
static-TLS workaround, **required for every Canopy binary**), the `OMP_*`
exports, the provenance echo, and `ctest -V` so per-rank measurement lines reach
the log. Save new scripts under `scripts/tuolumne/`.

- `# flux: --time-limit=8` and `-q pdebug`. `--time-limit=8` backfills
  immediately; a `--time-limit=20` submission has been observed sitting in
  `SCHED` for over twenty minutes.
- `# flux: --time=N` is **rejected** by this flux — the option is `--time-limit`.
- `flux batch --flags=waitable` is refused to non-owners, so the wait is
  `jobid=$(flux batch <script>)` then `flux job status "$jobid"`, which blocks and
  returns the job's exit code. **Never `flux job attach`.**
- Build on the login node with `make -j 4`, and only the targets a task's exit
  criterion names.

## Task sequence

### T1 — The index map and the derivative ladder — **NOT STARTED**

**Depends on:** none.

**Fill in:** new `src/Canopy_CartesianTaylorBasis.hpp` — the multi-index ↔ flat
slot map and the $b_k$ evaluator, nothing else; new
`tests/tstCartesianTaylor.hpp`; one line in `src/CMakeLists.txt` `HEADERS_PUBLIC`
(`:3-18`); one line in `tests/CMakeLists.txt` `UNIT_SERIAL_TESTS` (`:36-39`).

**Reference:** [canopy-questions.md](canopy-questions.md) §1 for the ladder
$\partial_a P_m = -(2m{+}1) r_a P_{m+1}$, §2 for the closed-form tensors through
$\partial_a\partial_b\partial_c$, §3 for the multi-index recurrence.

**Do:**

1. Fix the index map first and test it before writing any operator. Hand-derived
   Cartesian FMMs habitually go wrong here, and the map is a design decision, not
   an implementation detail. Choose a total order on multi-indices $k$ with
   $|k| \le p$, expose `slot(kx, ky, kz)` and its inverse, and state the chosen
   order on the declaration. Assert `slot` is a bijection onto
   $[0, \binom{p+3}{3})$ at each order 0 through at least 6, and that
   `inverse(slot(k)) == k` for every $k$ in range.
2. Implement $b_k(r; b)$ for all $|k| \le 2p$ by the §3 recurrence, host-callable
   and device-callable, taking $r$ and $b$ (**not** $\varepsilon$) and writing
   into a flat array indexed by the order-$2p$ slot map.
3. Hand-code $b_\emptyset$, $\partial_a$, $\partial_a\partial_b$ and
   $\partial_a\partial_b\partial_c$ from §2 in the test as the oracle for
   $|k| \le 3$. These are the in-repo reference; nothing here depends on a file
   outside the repository.
4. Above $|k| = 3$ the available oracle is a central finite difference of
   $\varphi$ itself, Richardson-extrapolated. Use it to check $|k| = 4$ through
   $2p$ at the orders the later tasks use, and state the tolerance it is checked
   at on the assertion.
5. Check the recurrence against the closed forms at several $r$ spanning
   $|r| \ll \sqrt b$, $|r| \sim \sqrt b$ and $|r| \gg \sqrt b$, and at $b$ values
   spanning the downstream solver's $\varepsilon = 0.025$ ($b = 6.25\times10^{-4}$).
   The $b \to 0$ limit is *not* in range: $b > 0$ is a precondition, and
   $b = 0$ at $r = 0$ divides by zero. Assert loudly on $b \le 0$ rather than
   returning a defaulted value.
6. Nothing in this task is a contract member. No sweep is instantiated, no trait
   is declared, and `src/Canopy_CartesianTaylorBasis.hpp` does not yet compile as
   a basis.

**Exit criterion:** `make cmake_check_build_system` then
`make -j 4 Canopy_Test_CartesianTaylor_SERIAL` in `build-tuolumne/` succeeds, and
the target run under `ctest -V -R Canopy_Test_CartesianTaylor_SERIAL` passes with
every body green. Specifically: the index map is a bijection at orders 0-6; the
recurrence reproduces all four §2 closed forms exactly at every sampled $(r, b)$;
and the finite-difference check passes at $|k| = 4 \ldots 2p$. **Failure
direction:** perturbing one coefficient of the §3 recurrence — change
$-2\sum_j k_j r_j$ to $-\sum_j k_j r_j$ — must fail the $|k| \ge 2$ closed-form
assertions, and perturbing the index map's inverse must fail the bijection
assertion, each with a message naming the multi-index. Record both perturbation
outcomes in the log.

**Checkpoint commit** at the end of this task.

---

### T2 — The contract surface and the kernel-blind operators — **NOT STARTED**

**Depends on:** T1.

**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` — every member in
[The complete contract](#the-complete-contract-a-farfield-type-must-supply)
except the three M2L stages' bodies and `build_m2l_operators`'s body;
`tests/tstCartesianTaylor.hpp` — three new bodies.

**Reference:** `tests/CanopyTest_MonopoleBasis.hpp` in full, as the worked
example of the shape: the traits block (`:101-263`), the struct-template
`aux_tables_type` and its builder (`:286-322`), and the units-and-conventions
header comment (`:44-88`). [canopy-questions.md](canopy-questions.md) §4 for the M2M, L2L and
evaluation formulas. The five call sites' offset senses are tabulated in
[The complete contract](#the-complete-contract-a-farfield-type-must-supply).

**Do:**

1. Declare every trait, typedef, alias template and struct template in the
   contract list, with the values this document's contract section gives, and
   with units and sign conventions on each declaration.
2. Implement `p2m_contribution` ($M_q \mathrel{+}= d^q/q! \cdot s$, atomic — the
   sweep runs one thread per particle and many particles share a leaf),
   `m2m_translate` and `l2l_translate` (binomial shifts, non-atomic — one team
   per parent), and `l2p_evaluate` with the **analytic** gradient. Ignore
   `w_self`, `w_child`, `w_parent`: this basis carries physical, un-normalized
   coefficients.
3. `build_aux_tables` returns whatever precomputed order-dependent data the
   operators need — at minimum the multi-index tables T1 built, if they are worth
   precomputing rather than recomputing. If nothing is needed, return an empty
   struct, as `MonopoleBasis` does. Neither sweep names a member of it.
4. `m2l_pre_cell` is a no-op and must not write to scratch: the accumulator there
   is zeroed once per team and carried across every pair of that team.
   `m2l_post_cell` flushes the accumulator into `L_out` with `+=`, not `=` — on a
   shared target, L2L from a shallower depth has already written there.
5. `build_m2l_operators` and `m2l_core` **abort loudly** in this task —
   `Kokkos::abort` with a message naming T3 — rather than filling zeros. A
   defined-but-wrong operator would let a solve run and produce a plausible
   field, which is the failure mode this ordering exists to prevent. No task in
   this document runs a solve before T3.
6. Add a compile-only body asserting `Solver<TEST_MS, TEST_ES, double, p,
   NComps, CartesianTaylorBasis>` is complete via `sizeof(S) > 0` — which forces
   the class body, its data members, and therefore `UpwardSweep`, `DownwardSweep`
   and `P2P` to instantiate, running all six class-scope guards against this
   basis — plus `std::is_same_v<S::kernel_type, CartesianTaylorBasis<double, p,
   NComps>>`. Naming the type alone instantiates nothing.
7. Test the three shift operators against host references over synthetic
   coefficient arrays, with no sweep involved: M2M then an inverse shift by $-s$
   must return the original moments; L2L likewise; L2P against direct evaluation
   of $\sum_p a^p \ell_p / p!$ and its analytic derivative against a
   Richardson-extrapolated finite difference of the same polynomial.

**Exit criterion:** `make -j 4 Canopy_Test_CartesianTaylor_SERIAL` succeeds and
`ctest -V -R Canopy_Test_CartesianTaylor_SERIAL` passes every body, including
T1's. **Failure direction:** temporarily setting `sets_per_component = 0` must
fail to compile quoting `src/Canopy_DownwardSweep.hpp:154-158`, and setting
`scalars_per_coeff = 2` must fail quoting `src/Canopy_UpwardSweep.hpp:74-79` —
demonstrating the guards actually run against this basis rather than being
satisfied vacuously. Revert both perturbations by inverting the edit, **not** by
`git checkout <file>`, which would discard the task's uncommitted work. Record
both diagnostics in the log.

**Checkpoint commit** at the end of this task.

---

### T3 — M2L, with the sign and normalization convention pinned — **NOT STARTED**

**Depends on:** T2.

**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` — `build_m2l_operators`,
`m2l_core`, `m2l_translate`, and the one function that is the single source of
truth for an operator entry; `tests/tstCartesianTaylor.hpp` — new bodies.

**Reference:** [canopy-questions.md](canopy-questions.md) §4 for
$\ell_p^A = \sum_q (-1)^{|q|} b_{p+q}(R) M_q^B$ and for the statement that
`_expansion_batch` already **is** $\ell_0$ for a single source box.
`tests/CanopyTest_MonopoleBasis.hpp:378-494` for the single-source-of-truth
pattern — `m2l_operator_entry`, `m2l_set_operator` and `m2l_accumulate`, each the
sole home of one value — and `:651-708` for what `build_m2l_operators` is handed. The $R$ sign is
in [The sign of $R$](#the-sign-of-r-which-the-key-does-not-give-directly) above.

**Do:**

1. Write **one** function that maps a key plus `unit_w` plus `kernel_params` to
   the dense $(N_t, N_s)$ operator, and reach it from `build_m2l_operators`, from
   `m2l_translate` and from every host reference. Do not duplicate the
   expression anywhere. This is what makes the conformance comparisons exact
   rather than merely close.
2. $R = -(ii, jj, kk) \cdot$ `unit_w[k.max_d]`, and $b = $
   `kernel_params.softening`$^2$. `n_levels` bounds `max_d`; assert
   `0 <= k.max_d < n_levels` rather than indexing past the end.
3. Fill `ops(slot(p), slot(q), j) = (-1)^{|q|} b_{p+q}(R)` over all
   $|p|, |q| \le p_{\rm order}$, evaluating the ladder to $2p_{\rm order}$ once
   per key. `ops` is allocated `WithoutInitializing`; write every entry.
4. `m2l_core` accumulates
   `acc(slot(p), c) += ops(slot(p), slot(q), op_idx) * M(source, slot(q), c)`
   over $q$, through a named-local multiply-accumulate helper for the
   `-ffp-contract` reason in [Conventions](#conventions). Keep
   `num_coeffs_per_cell` and the scratch extents compile-time constants
   (**R3**).
5. `m2l_translate` reconstructs the same key from the physical geometry it is
   handed — `w_unit = min(w_source, w_target)`,
   $(ii,jj,kk) = \mathrm{round}((dx,dy,dz)/w_{\rm unit})$, and `dd` from
   $w_{\rm target}/w_{\rm source} = 2^{dd}$ exactly — and reaches the same
   operator function, so the fused and fallback paths agree. It is atomic: the
   sweep runs one team per pair and two pairs can share a target.
   `MonopoleBasis::m2l_translate` (`:567-649`) is the worked reconstruction.
   **It cannot recover `max_d` from the widths alone** — it needs the physical
   half-width, which it has — so build $R$ directly from `(dx, dy, dz)` with the
   sign flipped, and assert in the test that this agrees with the table path
   for the same pair.

**Checkpoint commit here** — the operator builder compiles and the convention
checks below pass — before wiring anything else.

6. Pin the convention against the reference, which is the whole point of this
   task. Two checks, both host-only over hand-built inputs, no sweep:
   - **$\ell_0$ reproduces `_expansion_batch`.** For one source box with moments
     $G, D, Q$ (orders 0/1/2) and one target box at separation $R$, the $p = 0$
     local coefficient computed here must equal the reference formula's value.
     Transcribe `_expansion_batch`'s three arrays into the test from
     [canopy-questions.md](canopy-questions.md) §2's statement of them —
     $K = r_a P_1$, $dK = \delta_{ab}P_1 - 3r_ar_bP_2$,
     $ddK = -3(\delta_{ab}r_c + \delta_{ac}r_b + \delta_{bc}r_a)P_2 + 15r_ar_br_cP_3$,
     each being **minus** the corresponding §2 tensor, shifted by one index —
     and state on the assertion that the oracle is §2 and not a file outside the
     repository.
   - **The parity identity.** $\sum_q (-1)^{|q|} b_{p+q}(R) M_q$ and
     $(-1)^{|p|}\sum_q b_{p+q}(-R) M_q$ must agree to round-off for every $p$.
     Disagreement means the sign or the parity has been applied twice.
7. Add an end-to-end algebraic check with no FMM in it: place a handful of
   sources in one box and a handful of targets in a well-separated box, run
   P2M → M2L → L2P by hand over the basis's own operators, and compare against a
   direct softened sum over the same particles. At $p = 2$ and an admissible
   separation this is the first number that says the basis computes the right
   field rather than a self-consistent one. Assert against the separation-dependent
   bound $(c\,w/R)^{p+1}$ rather than a round number, and print the achieved
   ratio.

**Exit criterion:** `make -j 4 Canopy_Test_CartesianTaylor_SERIAL` succeeds and
`ctest -V -R Canopy_Test_CartesianTaylor_SERIAL` passes every body. Specifically:
$\ell_0$ matches the reference formula to round-off; the parity identity holds;
the fused and per-pair operator paths agree exactly for a sampled set of keys;
and the hand-run P2M → M2L → L2P beats the truncation bound. **Failure
direction:** dropping the $(-1)^{|q|}$ multiplier must fail the $\ell_0$ check at
the first odd $|q|$, and omitting the negation of $R$ must fail the parity check
— each with a message naming which. A build in which both are dropped must still
fail, since the two errors do not cancel for $|p| > 0$. Record all three
outcomes in the log.

**Checkpoint commit** at the end of this task.

---

### T4 — A full-pipeline solve at `near_softening_factor = 0` — **NOT STARTED**

**Depends on:** T3.

**Fill in:** new `tests/tstCartesianTaylorSolve.hpp`; one line in
`tests/CMakeLists.txt` `UNIT_MPI_TESTS` (`:48-58`); new
`scripts/tuolumne/run_ctest_cartesian_taylor.flux`.

**Reference:** `tests/tstLaplaceSolve.hpp:1490-1590` for the direct-sum body's
structure — the brute-force $O(N^2)$ reference over gathered last-step state, the
global-scale normalization rather than a per-particle ratio, and the
measured-then-pinned tolerance comment at `:155-187`.
`tests/tstFarFieldContract.hpp:287-360` for `ContractFixture`, the pattern for
driving a pipeline at 1-6 ranks.

**Do:**

1. Configure `FmmConfig` with `near_softening_factor = 0` and an **explicit
   positive** `softening`. The default `softening = -1.0` selects
   distribution-based auto-softening, whose effective $\varepsilon$ moves with
   the particle distribution and would make the tolerance unpinnable. Choose the
   downstream solver's $\varepsilon = 0.025$ and state it on the constant.
2. Drive `Solver<…, double, 2, NComps, CartesianTaylorBasis>` — $p = 2$, the
   reference's order — through a full solve at np 1-6.
3. Compare against a direct **softened** sum: $\phi_i = \sum_{j\ne i} q_j (r^2 + b)^{-1/2}$
   and $\nabla\phi_i = -\sum_{j \ne i} q_j\, d\, (r^2+b)^{-3/2}$ with
   $d = x_i - x_j$ and $b = \varepsilon^2$. This is the sign convention
   `tests/tstLaplaceSolve.hpp:1535-1538` uses for the unsoftened case, and it
   matches `Solver`'s output.
4. Run the comparison at **two** admissibilities: `mac_theta = 0.3`, matching the
   reference treecode, and `mac_theta = 0.5`, matching Canopy's frozen gate. The
   0.3 arm is the matched-admissibility comparison that makes the accuracy claim
   against the reference apples-to-apples; the 0.5 arm is what Canopy actually
   runs.
5. Pin both achieved deviations as named constants with a provenance comment
   stating the configuration they were measured at — particle count, `ncrit`,
   `max_depth`, `replication_depth`, `softening`, `mac_theta`, rank counts — in
   the style of `LS_CROSS_RANK_TOL` / `LS_DIRECT_SUM_TOL`
   (`tests/tstLaplaceSolve.hpp:155-187`). Assert against the pinned value with a
   small margin, not against a round number, so a regression is visible rather
   than merely inside a loose bound.
6. Print `total_fallback_pair_count()`, `m2l_n_unique_ops()` per rank and the
   effective operator cap on every run, so a later failure is attributable
   without a rebuild. A non-zero fallback count is not a failure — the paths
   agree, by T3 — but it changes which arithmetic ran and must be visible
   (**R4**).
7. Expect the first real errors in `Solver`'s member bodies rather than in its
   declarations; this is the first work that runs a solve through the `FarField`
   slot (**R7**).
8. **Do not** add anything to `REGRESSION_MPI_TESTS`.

**Exit criterion:** `make cmake_check_build_system`, then `make -j 4
Canopy_Test_CartesianTaylorSolve_MPI_SERIAL Canopy_Test_LaplaceSolve_MPI_SERIAL`
in `build-tuolumne/`, then one flux job running both suites under `ctest -V`:

- `ctest -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL` passes at np 1-6, with
  the $\theta = 0.3$ relative deviation on both potential and gradient **at or
  below 1e-3** at $p = 2$ — the reference treecode's documented accuracy at the
  reference treecode's order and admissibility — and the $\theta = 0.5$ deviation
  measured and pinned alongside it.
- `ctest -R Canopy_Test_LaplaceSolve_MPI_SERIAL` passes all three checks —
  `bitForBitArtifacts` at np 1-2 against the committed bytes with **no
  regeneration of `tests/data/laplace_solve_P6.txt`**, `crossRankAgreement` at
  np 2-6, `matchesDirectSum` at np 1-6.

**Failure direction:** the same solve run against `LaplaceKernel` at
`near_softening_factor = 0` and the same positive `softening` must **fail** the
direct-softened-sum comparison by orders of magnitude — that is the unsoftened
far field being compared against a softened reference, and it is the measurement
that shows the new basis is doing the work rather than the tolerance being loose.
Record its deviation in the log beside the new basis's.

**Checkpoint commit** at the end of this task.

---

### T5 — Measure the operator-cache behaviour across a rebalance — **NOT STARTED**

**Depends on:** T4.

**Fill in:** `tests/tstCartesianTaylorSolve.hpp` — one measurement body and its
printed line. No file under `src/` changes.

**Reference:** `src/Canopy_DownwardSweep.hpp:406-416` for `set_root_half_width`'s
cache-clearing rule, `:421-447` for the counters, and
[abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md)
§T9 for the mechanism and for the baseline it measured on a
`key_needs_level = false` basis (zero keys rebuilt across an
`invalidate_interaction_list()`).

**Do:**

1. Run a multi-step solve whose particles move enough that the root bounding box
   drifts between steps, and record `m2l_op_keys_built_count()`,
   `m2l_op_cache_size()` and `interaction_list_build_count()` after each build,
   per rank, at np 1-6.
2. Print them on a single provenance line per rank so a later session can read
   the numbers out of a `ctest -V` log without a rebuild.
3. **Do not design or implement a fix**, and do not touch
   `src/Canopy_DownwardSweep.hpp`. The behaviour is understood; what is missing
   is its magnitude on a realistic distribution. See **R6** for what the numbers
   mean.
4. Record the measured numbers in the log against **R6**, and state whether the
   cache retained anything at all across a drifting box.

**Exit criterion:** `ctest -V -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`
passes at np 1-6 with the measurement line present at every rank, and the
progress log carries the per-rank `m2l_op_keys_built_count()` figures across at
least two consecutive rebuilds with a drifting root box, stated against the T9
baseline. **Failure direction:** the body must assert
`m2l_op_keys_built_count() > 0` after the first build — a zero there would mean
no operator was ever constructed and the measurement is vacuous.

**Checkpoint commit** at the end of this task.

## Known risks

**R1 — the sign and normalization convention between the moment definition and
the $(-1)^{|q|}$ multiplier is subtly wrong.** The reference author flags this as
the one place needing care, and getting it wrong produces a plausible-looking
field rather than an obvious failure: the expansion is still a polynomial, still
decays correctly, and still converges — to the wrong thing. **Presents as:** T3's
$\ell_0$ check failing, or T3 passing while T4's direct-sum comparison misses
1e-3 by one to three orders at every rank count and at both $\theta$.
**Distinguished from R2** by which order it breaks at: a convention error is
present at $|q| = 1$, the first odd multi-index, while an index-map error is
invisible below $|k| = 4$ where the map first has non-trivial structure.
**Do:** T3's $\ell_0$-reproduces-`_expansion_batch` check is the discriminator and
must be written before any solve is run. If T4 misses and T3 passes, re-check the
$1/q!$ placement: the factorial belongs in the moment and in the L2P evaluation,
never in $b_k$.

**R2 — the symmetric-tensor index map is wrong above $|k| = 3$.** Hand-derived
Cartesian FMMs habitually go wrong here, and the closed forms that serve as the
oracle stop at $|k| = 3$. **Presents as:** T1 green, T3's parity identity green
(it is symmetric in the map), and T4's accuracy missing at $p \ge 2$ — where the
M2L reaches $b_{p+q}$ with $|p+q| = 4$ — while $p = 1$ would have passed.
**Do:** T1's finite-difference check at $|k| = 4 \ldots 2p$ is the only oracle
above the closed forms and is not optional. If T4 misses, re-run T1 at the
specific $b$ and $r$ scales T4 uses before touching anything else: the FD check's
tolerance is scale-dependent and a pass at one scale is not a pass at another.

**R3 — a trait indirection deoptimizes the fused M2L kernel.**
`num_coeffs_per_cell`, `sets_per_component` and `m2l_scratch_bytes` drive
unrolling and the scratch size. If any becomes a runtime value the kernel slows
down **with no correctness signal**. **Presents as:** every test passing and the
solve being slower. **This risk has already fired once**, at roughly +18% on the
M2L kernel when the contraction first moved behind the basis interface; two
candidate micro-causes were tested and both excluded, so the cost is intrinsic to
the indirection rather than to any detail
([abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md),
§T3). **Do:** keep all three `constexpr`. No exit criterion here depends on a
timing figure; if one is wanted, measure `M2L kernel (all depths)` from the
`DownwardSweep::execute()` table in `build-tuolumne-prof/`, never in
`build-tuolumne/`, whose `Canopy_ENABLE_PROFILING` is `OFF` and which emits no
`[Canopy Diagnostics]` line at all.

**R4 — the per-pair path and the table path disagree, and which pairs overflow
decides the answer.** A pair refused an operator column takes `m2l_translate`,
which is *different arithmetic* — the same mathematics evaluated per pair rather
than out of a table. If the two paths also disagree *mathematically*, the answer
depends on the cap. **Presents as:** T4's accuracy check failing at some rank
counts and not others, with `total_fallback_pair_count()` non-zero.
**Do:** that counter is the discriminator and T4 prints it. T3's
fused-versus-per-pair agreement assertion is what makes a non-zero count
harmless. At $p = 2$ the count cap (32768) binds far below any realized key
count, so the expected value is 0 — but the guard is the assertion, not the
expectation.

**R5 — the persistent operator cache holds stale operators.** The cache survives
topology changes deliberately, on the premise that a canonicalized key plus
`M2LKernelParams` determines the operator. **This basis's operator additionally
depends on `unit_w`**, which is not part of that premise — and the only thing
that saves it is `set_root_half_width` clearing the cache for a
`key_needs_level = true` basis (`src/Canopy_DownwardSweep.hpp:406-416`).
**Presents as:** correct results on the first solve and drifting results after a
rebalance — which a single-solve test would not catch. **Do:** T4's solve is
multi-step and rebalances, so it exercises this; T5 measures the counters that
show whether the clearing actually happened. If a later change makes this basis's
operator depend on anything beyond the key, `unit_w` and `kernel_params`, it must
say so and opt out of the cache.

**R6 — `key_needs_level = true` empties the cache on every rebuild when the
bounding box drifts.** `set_root_half_width` clears the **entire** operator cache
when the root half-width changes, and does so **only** for a `key_needs_level`
basis; `Solver::_push_root_half_width()` (`src/Canopy_Solver.hpp:756-770`) runs
before every one of the three `_downward.setup()` calls. On a distribution whose
bounding box drifts — which is every moving-particle workload, and exactly the
workload the cache exists for — the cache therefore empties on each rebuild.
**Presents as:** correct answers throughout and `m2l_op_keys_built_count()`
climbing by roughly the full key count at every build, against T9's measured zero
for a level-blind basis. **Not a correctness bug**, and **no task here fixes it**:
T5 measures its magnitude and nothing more. Designing a fix means changing
`src/Canopy_DownwardSweep.hpp` — plausibly by keying the cache on the physical
$R$ rather than on the level, or by tolerating a root-width change that is an
exact power of two — and that is separate work with its own gate.

**R7 — `Solver`'s member bodies have never been instantiated on a
non-`LaplaceKernel` basis.** The existing compile-only test forces the class to
be complete, which instantiates the data members and the sweeps' guards, but
**not** `solve()` — member functions are compiled only for instantiations
something calls. **Presents as:** T2 passing cleanly and T4 failing to compile,
in `src/Canopy_Solver.hpp` rather than in the basis. **Do:** treat T2's pass as
covering the declarations only. Expect T4's first failures there, read them as
expected rather than as a defect in the basis, and record in the log every
`Solver` member body that turned out to assume something `LaplaceKernel`-specific
— that list is the real output of T4's first build and belongs to whoever writes
the next basis.

**R8 — the forward recurrence loses accuracy at high order.** Each step of the §3
recurrence divides by $w$ and accumulates terms weighted by $k_j(k_j-1)$, so
error can grow with $|k|$. At $p = 2$ the ladder runs to $|k| = 4$, four steps,
and this is not a concern. **Presents as:** T1's finite-difference check passing
at $|k| = 4$ and failing at higher $|k|$ when a later session raises $p$, with the
failure worst at small $|r|/\sqrt{b}$ where $w$ is smallest. **Do:** T1's FD
check at $|k| = 2p$ is the instrument; it must be re-run, not assumed, if $p$ is
ever raised above 2. Do not widen its tolerance to accommodate a failure —
re-measure at the specific $(r, b)$ scales in use and report. $b > 0$ bounds
$w \ge b$ away from zero, so this is a conditioning question and never a division
by zero.
