# A Cartesian-Taylor far-field basis for Canopy

**Status:** IN PROGRESS — T1, T2, T3 and T4 DONE; T5 next, then T6

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
(`src/Canopy_Solver.hpp:125-134`), adds no shared arithmetic and declares no
new class-scope guard. `Solver`'s member bodies have never been compiled against
a non-`LaplaceKernel` basis, so T4 may need minimal generic fixes inside them
(**R7**); nothing else under `src/` changes.

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
$p$ has relative error $\sim (c\,W/R)^{p+1}$, with $W$ the **full width of the
source box** — not the $w = r^2 + b$ of the ladder below, which is a different
quantity sharing a letter in the source material — and $c$ between $1$ and
$\sqrt3$; at standard admissibility $R/W \approx 3$ each additional order buys
between
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

That file is a pasted email thread, and its plain-text math has lost subscripts
and mangled exponents: its line 56 writes $P_1 = w^{-3/2}$ as `1/w15`, and
`P{m+1}`, `∂l φ` and `ε{ilm}` appear with the underscore dropped. The rendering
below has been checked term by term against it, and where the two disagree the
rendering below is correct.

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

The negation is not optional and is not absorbed anywhere else. $\varphi$ is
even, so $b_n(-R) = (-1)^{|n|} b_n(R)$, and the two spellings

$$
\ell_p = \sum_q (-1)^{|q|} b_{p+q}(R)\,M_q
\qquad\text{and}\qquad
\ell_p = (-1)^{|p|} \sum_q b_{p+q}(S)\,M_q,\quad S = -R
$$

must agree numerically. **Where each spelling gets its argument decides whether
that comparison can fail at all.** The equality is an identity in the vector fed
to it, so handing both spellings the same vector makes them agree whatever its
sign: a comparison that obtains $S$ by negating the very $R$ the first spelling
used is vacuous and passes over a wrong-signed operator. The two arguments must
be sourced independently — the first through the production key-to-$R$ path,
$R = -(ii,jj,kk)\cdot w_{\rm unit}[\texttt{max\_d}]$, and the second from the
key's **raw** offset $S = (ii,jj,kk)\cdot w_{\rm unit}[\texttt{max\_d}]$, source
minus target exactly as the sweep builds it
(`src/Canopy_DownwardSweep.hpp:1464-1481`), with the $(-1)^{|p|}$ applied by the
comparison itself. So sourced, a missing negation makes the two disagree at every
odd $|p|$, and disagreement otherwise means the parity or the sign has been
applied twice. T3 asserts it this way.

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
| Per-operator test | `tests/tstCartesianTaylor.hpp`, name `CartesianTaylor` in `UNIT_SERIAL_TESTS` (`tests/CMakeLists.txt:36-40`) | Host-only math over no MPI. Target becomes `Canopy_Test_CartesianTaylor_SERIAL` (`cmake/test_harness/test_harness.cmake:104-112`). |
| Solve test | `tests/tstCartesianTaylorSolve.hpp`, name `CartesianTaylorSolve` in `UNIT_MPI_TESTS` (`tests/CMakeLists.txt:49-59`) | Target becomes `Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`, tests `..._np_<N>` for N in 1-6, CTest label `unit`. |
| Adding a name to `tests/CMakeLists.txt` | run `make cmake_check_build_system` **first** | `make -j 4 <new target>` otherwise fails with "No rule to make target": make errors out before it regenerates, because the target is not in the current Makefile. Regenerating preserves the existing cache. |
| Gating a SERIAL unit target | anchor the regex: `ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` | `Canopy_add_tests` registers a `_valgrind` variant beside every non-MPI test when valgrind is found (`cmake/test_harness/test_harness.cmake:157-162`), and it is found in `build-tuolumne/`. An unanchored `-R` runs both, and a Kokkos binary under valgrind need not fit `--time-limit=8`. No task here gates on the variant. MPI targets are unaffected — the valgrind block is in the non-MPI branch. |
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

`_expansion_batch(R, G, D, Q, blob, order)` is the whole far-field contribution
of one source box **at the target box center**, with no target-side expansion.
Extending it to a real FMM means adding the coefficients that carry the
expansion away from that center, which are just higher $b_{p+q}(R)$ off the same
ladder.

**It is not $\ell_0$**, and matching it against $\ell_0$ is a wrong comparison
rather than a loose one. It returns a *velocity*, which needs
$\nabla\varphi$: its three arrays are $-\partial\varphi$, $-\partial^2\varphi$
and $-\partial^3\varphi$ ([canopy-questions.md](canopy-questions.md) §2, "minus
these, shifted by one index"), so contracted against the degree-0/1/2 moments
they give the **$|p| = 1$** local coefficients of the scalar pass — its gradient
at the box center. $\ell_0$ is the scalar potential there and needs $b_k$ at
degrees 0, 1 and 2 instead. T3's step 6 checks both, against §2 rather than
against this file.

Its moments are also **full symmetric tensors** where this basis's are
multi-indexed. The two differ by $|q|!/q!$:
$\sum_{a,b} T_{ab} d_a d_b = \sum_{|q|=2} \frac{|q|!}{q!} T_q d^q$. Any
comparison across the two representations applies that factor.

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

`m2l_accumulator_type` is the **one exception** to the preamble above: no sweep
names it and nothing outside the basis is kept well-formed by it. Both existing
bases declare it only to reinterpret the sweep's raw scratch bytes inside their
own M2L stages (`src/Canopy_LaplaceKernel.hpp:330`,
`tests/CanopyTest_MonopoleBasis.hpp:327`). It is **basis-internal — you own its
shape**; the sweep's obligation stops at `m2l_scratch_bytes`, which is what it
actually reads.

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
| `l2p_evaluate` | particle − cell center ($a$) | `w_self` = leaf half-width | `src/Canopy_DownwardSweep.hpp:2668-2670`, with the potential accumulate at `:2673-2674` |

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

### T1 — The index map and the derivative ladder — **DONE**

**Depends on:** none.

**Fill in:** new `src/Canopy_CartesianTaylorBasis.hpp` — the multi-index ↔ flat
slot map and the $b_k$ evaluator, nothing else; new
`tests/tstCartesianTaylor.hpp`; one line in `src/CMakeLists.txt` `HEADERS_PUBLIC`
(`:3-18`); one line in `tests/CMakeLists.txt` `UNIT_SERIAL_TESTS` (`:36-40`); new
`scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux`, the batch wrapper for
the SERIAL unit target, which T2 and T3 reuse unchanged.

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
the target run under `ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes with
every body green. Specifically: the index map is a bijection at orders 0-6; the
recurrence reproduces all four §2 closed forms exactly at every sampled $(r, b)$;
and the finite-difference check passes at $|k| = 4 \ldots 2p$.

**Checkpoint commit** at the end of this task.

**Met.** `make cmake_check_build_system` then
`make -j 4 Canopy_Test_CartesianTaylor_SERIAL` in `build-tuolumne/` succeeds,
and `ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes 3/3 bodies with
ctest rc 0 (flux jobs `f3YTi9NMzAE3`, and `f3YTwT7yiC2K` re-run on the exact
committed tree after both perturbations were reverted).

*The index map is a bijection at orders 0-6*: `slot` is injective and onto
$[0,\binom{p+3}{3})$ at every $p$ from 0 to 6 — 1, 4, 10, 20, 35, 56 and 84
slots — with `inverse_slot(slot(k)) == k` for every $k$ in range and
`slot(inverse_slot(s)) == s` for every $s$ in range. Each slot is also asserted
to land inside its own degree block, which is what makes the chosen *degree-graded*
order's prefix property a checked fact rather than an intention: the order-$p$
table is a prefix of the order-$2p$ one, so the M2L's $b_{p+q}$ out to
$|p+q| = 2p$ and the moments' $|q| \le p$ share one flat index.

*The recurrence reproduces all four §2 closed forms at every sampled $(r,b)$*:
worst deviation $2.911\times10^{-15}$ at $k = (3,0,0)$ against a $10^{-12}$
tolerance, over 40 samples — $b \in \{10^{-6}, 6.25\times10^{-4}, 10^{-2}, 1\}$
(the second being the solver's $\varepsilon = 0.025$ squared), each at $r = 0$
and at $|r|/\sqrt b \in \{0.01, 1, 100\}$ in three directions. Errors are
measured against $\varphi/L^{|k|}$, $L = \sqrt w$, because components vanish
identically at the sampled $r$ and a relative test would divide by zero.

*The finite-difference check passes at $|k| = 4 \ldots 2p$*: at $p = 2$ that is
$|k| = 4$, worst deviation $7.321\times10^{-7}$ against $10^{-5}$. The oracle
needed **two** Richardson steps rather than the one this section describes; one
step bottoms out at $7.8\times10^{-6}$ and leaves no usable margin. See the
divisor scan in [the log](cartesian-taylor-basis-progress-log.md) under `## T1`.

*Failure direction, both run as their own job and reverted by inverting the
edit*: perturbing $-2\sum_j k_j r_j$ to $-\sum_j k_j r_j$ failed
`closed_forms` at multi-index $(2,0,0)$, $|k| = 2$ — $|k| \le 1$ untouched, as
predicted, since the sum is empty at $k = 0$ — and `finite_difference` at
$(2,0,2)$, with `index_map_bijection` still green. Swapping $k_y$ and $k_z$ in
`inverse_slot` failed the bijection assertion at $(0,0,1)$ with the message
naming it. Both jobs exited ctest rc 8.

*Not a contract member*: the header declares no trait, typedef, alias template,
`static_assert` on the basis or operator, and instantiates no sweep and no
`Solver`. It does not yet compile as a `FarField`. That is T2.

---

### T2 — The contract surface and the kernel-blind operators — **DONE**

**Depends on:** T1.

**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` — every member in
[The complete contract](#the-complete-contract-a-farfield-type-must-supply)
except `m2l_core`'s body and `build_m2l_operators`'s body. `m2l_pre_cell` (a
no-op) and `m2l_post_cell` (a real accumulator flush) are written *here*, per
steps 4 and 5 below: no later task's **Fill in** names `m2l_post_cell`, so
deferring it would leave it unwritten permanently — which is the failure
[Deliberate deviations](#deliberate-deviations) records as compiling cleanly
while leaving every local coefficient zero. `tests/tstCartesianTaylor.hpp` —
three new bodies.

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
6. Add a compile-only body asserting `Solver<TEST_MEMSPACE, TEST_EXECSPACE,
   double, p, NComps, CartesianTaylorBasis>` is complete via `sizeof(S) > 0`
   — the macros a SERIAL unit test has are `TEST_MEMSPACE` and `TEST_EXECSPACE`
   (`cmake/test_harness/TestSERIAL_Category.hpp:16-17`);
   `TEST_MS`/`TEST_ES` are template parameter names local to
   `tests/tstFarFieldContract.hpp`'s fixtures and are not in scope elsewhere
   — which forces
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
`ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes every body, including
T1's.

**Checkpoint commit** at the end of this task.

**Met.** `make -j 4 Canopy_Test_CartesianTaylor_SERIAL` succeeded on the login
node in `build-tuolumne/` with no warnings from the new code, and
`ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passed **7/7 bodies** —
T1's three unchanged at their previously measured figures ($2.911\times10^{-15}$
closed-form, $7.321\times10^{-7}$ finite-difference) plus T2's four — at
flux job `f3YUdrmXQa1d`, re-run green as `f3YUfS9vvt9m` on the exact tree
committed at the checkpoint, both rc 0 with identical figures.

What the four new bodies actually verified, each at **both** $p = 2$ and
$p = 4$ and at `NComps = 3`, against oracles that share no code with the basis
(brute-force repeated multiplication and an explicit factorial, never
`taylor_monomials`): `p2m_contribution` against $\sum_j d_j^q/q!\,s_{jc}$ to
$7.5\times10^{-18}$ of $\max|M|$; `m2m_translate` against a **direct P2M about
the parent center** — the exact reference, since the shift is untruncated — to
$1.2\times10^{-16}$, and its round trip by $-s$ to $6.0\times10^{-17}$;
`l2l_translate`'s round trip by $-s$ to $8.9\times10^{-15}$ and the
shifted child expansion against the parent expansion **at the same physical
point** to $1.8\times10^{-15}$; `l2p_evaluate`'s potential against
$\sum_p a^p\ell_p/p!$ to $2.2\times10^{-16}$ and its **analytic** gradient
against a Richardson-extrapolated central difference of that same polynomial to
$2.6\times10^{-14}$. Every figure is at the roundoff floor, which is the
expected result: all four operators are exact rational arithmetic, not
approximations.

Both failure directions fired at the guards they were supposed to, each built
and reverted by inverting the edit. `sets_per_component = 0` failed at
`src/Canopy_DownwardSweep.hpp:154`; `scalars_per_coeff = 2` failed at
`src/Canopy_UpwardSweep.hpp:74`, and also at `:124` in `DownwardSweep` and at
the basis's own assert. Both diagnostics are quoted verbatim in the progress
log. So the six class-scope guards are not satisfied vacuously here: they run
against this basis through `Solver`'s data members.

**Scope, stated rather than implied.** `build_m2l_operators`, `m2l_core` and
`m2l_translate` `Kokkos::abort` naming T3 and are the only incomplete members.
`m2l_post_cell` is **real**, not a stub, and the M2L accumulator layout it reads
is fixed here — see the progress log. Per **R7** this pass is evidence about the
**declarations only**: `Solver`'s member function bodies are not instantiated by
a `sizeof`, so a clean T2 says nothing about whether `solve()` compiles on a
non-`LaplaceKernel` basis. Expect T4's first failures inside
`src/Canopy_Solver.hpp`.

---

### T3 — M2L, with the sign and normalization convention pinned — **DONE**

**Depends on:** T2.

**Fill in:** `src/Canopy_CartesianTaylorBasis.hpp` — `build_m2l_operators`,
`m2l_core`, `m2l_translate`, and the one function that is the single source of
truth for an operator entry; `tests/tstCartesianTaylor.hpp` — new bodies.

**Reference:** [canopy-questions.md](canopy-questions.md) §4 for
$\ell_p^A = \sum_q (-1)^{|q|} b_{p+q}(R) M_q^B$, and §2 for the closed-form
tensors that are the in-repo oracle for both convention checks in step 6.
`tests/CanopyTest_MonopoleBasis.hpp:378-494` for the single-source-of-truth
pattern — `m2l_operator_entry`, `m2l_set_operator` and `m2l_accumulate`, each the
sole home of one value — and `:651-708` for what `build_m2l_operators` is handed. The $R$ sign is
in [The sign of $R$](#the-sign-of-r-which-the-key-does-not-give-directly) above.

**Do:**

1. Write **one** function that maps the **physical** $(R[3], b)$ to the dense
   $(N_t, N_s)$ operator, and reach it from `build_m2l_operators`, from
   `m2l_translate` and from every host reference. Do not duplicate the
   expression anywhere. This is what makes the conformance comparisons exact
   rather than merely close.

   It is parameterized on $R$ and $b$ rather than on a key because
   `m2l_translate` cannot produce a key — see step 5 — while both callers can
   produce $R$: `build_m2l_operators` as
   $R = -(ii,jj,kk)\cdot$ `unit_w[k.max_d]` and `m2l_translate` as
   $R = -(dx,dy,dz)$.
2. $b = $ `kernel_params.softening`$^2$; the field is a **length** and the
   kernel's $b$ is its square. `n_levels` bounds `max_d`; assert
   `0 <= k.max_d < n_levels` rather than indexing past the end.

   Two further guards, because both values have a reachable default that would
   make the operator silently wrong rather than noisy. **Abort loudly** with a
   message naming the convention when `kernel_params.softening <= 0`:
   `M2LKernelParams::softening` defaults to `0.0`, documented as the unsoftened
   kernel a sweep driven directly by a test runs at
   (`src/Canopy_FarFieldContract.hpp:118-120`), and `derivative_ladder` requires
   $b > 0$ because $b = 0$ at $r = 0$ divides by zero. **Abort loudly** likewise
   when `unit_w[k.max_d] <= 0`: `unit_w` is all zeros until something calls
   `set_root_half_width` (`src/Canopy_DownwardSweep.hpp:402-404`), and a zero
   entry yields $R = 0$ and a finite, wrong, physical operator. Neither is
   reachable from the tasks here — T3 runs no sweep, T4 sets an explicit positive
   `softening`, and `Solver::_push_root_half_width` runs before every
   `_downward.setup()` — so these are guards for a later caller, not failures to
   expect.
3. Fill `ops(slot(p), slot(q), j) = (-1)^{|q|} b_{p+q}(R)` over all
   $|p|, |q| \le p_{\rm order}$, evaluating the ladder to $2p_{\rm order}$ once
   per key. `ops` is allocated `WithoutInitializing`; write every entry.
4. `m2l_core` accumulates
   `acc(slot(p), c) += ops(slot(p), slot(q), op_idx) * M(source, slot(q), c)`
   over $q$, through a named-local multiply-accumulate helper for the
   `-ffp-contract` reason in [Conventions](#conventions). Keep
   `num_coeffs_per_cell` and the scratch extents compile-time constants
   (**R3**).
5. `m2l_translate` builds $R = -(dx, dy, dz)$ directly from the physical offset
   it is handed and reaches the same operator function, so the fused and fallback
   paths agree. It is atomic: the sweep runs one team per pair and two pairs can
   share a target. Assert in the test that it agrees with the table path for the
   same pair.

   **It reconstructs no key, and needs none.** This operator is a function of the
   physical $R$ and $b$ alone — it has **no `dd` dependence**, so there is
   nothing a key would supply that $(dx,dy,dz)$ does not.
   `MonopoleBasis::m2l_translate` (`:567-649`) does reconstruct one, and is not
   the model to copy here: its local is dimensionless and normalized, so it needs
   $F(dd) = 2^{\max(0,-dd)}$ to convert between "separation in deeper-cell
   half-widths" and the source's own scale
   (`tests/CanopyTest_MonopoleBasis.hpp:404-427`). A physical operator has no
   normalization to undo. `m2l_translate` also cannot recover `max_d` from the
   two half-widths in any case, which is the second reason the key route is
   closed to it.

   One consequence worth expecting rather than debugging: two keys differing only
   in `dd` produce **identical operator columns**. That is duplication in the
   table, not an error — `MonopoleBasis` records the same effect for the same
   reason (`tests/CanopyTest_MonopoleBasis.hpp:156`).

**Checkpoint commit here** — the operator builder compiles and the convention
checks below pass — before wiring anything else.

6. Pin the convention, which is the whole point of this task. Three checks, all
   host-only over hand-built inputs, no sweep:
   - **$\ell_0$ against §2's closed forms.** For one source box with moments
     $M_q$, $|q| \le p$, and one target box at separation $R$,
     $\ell_0 = \sum_q (-1)^{|q|} b_q(R) M_q$ must match an oracle hand-coded
     from [canopy-questions.md](canopy-questions.md) §2 at degrees 0, 1 and 2 —
     $b_\emptyset = P_0$, $\partial_a\varphi = -r_aP_1$,
     $\partial_a\partial_b\varphi = -\delta_{ab}P_1 + 3r_ar_bP_2$. This is
     **R1**'s real discriminator: the $(-1)^{|q|}$ multiplier and the $1/q!$
     placement first bite at $|q| = 1$, and this is the check that sees them.
     State on the assertion that the oracle is §2 and not a file outside the
     repository.
   - **The $|p| = 1$ coefficients against the reference's contraction.** This is
     the `_expansion_batch` correspondence, and the only check that exercises
     $b_k$ at degree 3. Transcribe the reference's three arrays into the test
     from §2's statement of them — $K = r_a P_1$,
     $dK = \delta_{ab}P_1 - 3r_ar_bP_2$,
     $ddK = -3(\delta_{ab}r_c + \delta_{ac}r_b + \delta_{bc}r_a)P_2 + 15r_ar_br_cP_3$,
     each being **minus** the corresponding §2 tensor **shifted by one index** —
     contract them against the degree-0/1/2 moments, and assert against
     $\ell_{e_a}$ with the overall sign of $K = -\nabla\varphi$ stated on the
     assertion rather than absorbed silently.

     Both the degree shift that makes this the $|p| = 1$ check and not an
     $\ell_0$ one, and the $|q|!/q!$ multinomial factor between the reference's
     full symmetric tensors and this basis's multi-indexed moments, are in
     [The reference implementation](#the-reference-implementation-and-what-it-fixes).
     Getting either wrong is a route to the same **R1** failure.
   - **The parity identity.** $\sum_q (-1)^{|q|} b_{p+q}(R) M_q$ and
     $(-1)^{|p|}\sum_q b_{p+q}(S) M_q$ must agree to round-off for every $p$,
     with the two arguments sourced independently as
     [The sign of $R$](#the-sign-of-r-which-the-key-does-not-give-directly)
     requires: $R$ from the production key-to-$R$ path, $S$ from the key's raw
     source-minus-target offset. Sourced any other way the identity is vacuous
     and the check cannot fail. Disagreement means the sign or the parity has
     been applied twice.
7. Add an end-to-end algebraic check with no FMM in it: place a handful of
   sources in one box and a handful of targets in a well-separated box, run
   P2M → M2L → L2P by hand over the basis's own operators, and compare against a
   direct softened sum over the same particles. At $p = 2$ and an admissible
   separation this is the first number that says the basis computes the right
   field rather than a self-consistent one. Assert against the
   separation-dependent bound $(c\,W/R)^{p+1}$ rather than a round number, and
   print the achieved ratio. $W$ is a box size, not the ladder's $w = r^2 + b$;
   state on the assertion which length it was measured in — full width or
   half-width — since the achieved $c$ is only interpretable against that choice.

**Exit criterion:** `make -j 4 Canopy_Test_CartesianTaylor_SERIAL` succeeds and
`ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes every body. Specifically:
$\ell_0$ matches the §2 closed forms to round-off; the $|p| = 1$ coefficients
match the reference's $K/dK/ddK$ contraction to round-off; the parity identity
holds with its two arguments independently sourced; the fused and per-pair
operator paths agree exactly for a sampled set of keys; and the hand-run
P2M → M2L → L2P beats the truncation bound.

**Checkpoint commit** at the end of this task.

**Met.** `build_m2l_operators`, `m2l_core` and `m2l_translate` replace their T2
aborts, all three reaching one new single-source-of-truth function,
`m2l_operator_block( R[3], b, op[Nt*Ns] )`, which maps the **physical**
$(R, b)$ to the dense block $op(p,q) = (-1)^{|q|} b_{p+q}(R)$ with
$R = c_{\rm target} - c_{\rm source}$. `aux_tables_type` grew the single scalar
`b` (softening **squared**), because `m2l_translate`'s signature carries no
`M2LKernelParams` and `aux` is the only channel it has. All twelve bodies pass
(`make -j 4 Canopy_Test_CartesianTaylor_SERIAL`, then
`ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'`) — the seven from T1 and
T2, plus five new.
Measured, each against the sum of its own term magnitudes rather than against
the cancelling result, at tolerance $10^{-13}$ of that scale:

| Check | Achieved | Tolerance |
| --- | --- | --- |
| $\ell_0$ vs §2 closed forms, $p = 2$ | $2.000\times10^{-16}$ | $10^{-13}$ |
| $\ell_0$ vs §2 closed forms, $p = 3$ | $1.961\times10^{-16}$ | $10^{-13}$ |
| $\lvert p\rvert = 1$ vs the reference's $K/dK/ddK$ contraction, $p = 2$ | $1.190\times10^{-15}$ | $10^{-13}$ |
| parity identity, $R$ and $S$ independently sourced, 7 keys | $0$ (bitwise) | $10^{-13}$ |
| fused vs per-pair path, 5 keys + a 3-pair accumulation | $0$ | `ASSERT_EQ`, exact |

The hand-run P2M → M2L → L2P against a direct softened sum, at $p = 2$,
$\varepsilon = 0.025$, box **half-width** $W = 0.5$: relative error
$2.337\times10^{-3}$, $2.346\times10^{-4}$ and $2.530\times10^{-5}$ at
$R/W = 8$, $16$ and $32$, i.e. achieved $c = 1.062$, $0.987$, $0.939$ against
the pinned bound $(1.25\,W/R)^{3}$ — a factor 1.6 to 2.4 of margin.

Flux jobs, all on `pdebug` via the unchanged
`scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux`: `f3YeTspuFk3q`
(first full pass, measuring the achieved $c$ before the bound was pinned),
`f3YeUnMzVdk3` (green with the pinned bound, the checkpoint tree),
`f3YeVguWwKE7` / `f3YeX8BoVNNB` / `f3YeXxRkMxHM` (the three perturbations), and
`f3YeYsBmmaa3` (green on the exact tree committed, byte-identical figures).

**The three perturbation outcomes**, each reverted by inverting the edit:

- **Dropping the $(-1)^{|q|}$ multiplier** fails `m2l_ell0_closed_forms` first,
  by $0.2299$ absolute against a tolerance of $9.27\times10^{-14}$, at
  $R = (2,0,0)$ — the first odd $|q|$, as **R1** predicts. It also fails the
  $|p|=1$ contraction, the parity identity and the end-to-end check.
  `m2l_fused_vs_fallback` **passes** under it, correctly: both paths reach the
  same perturbed operator, so it is an agreement check and never a convention
  check.
- **Omitting the negation of $R$** in `build_m2l_operators` fails
  `m2l_parity_identity` — at $|p| = 0$, not at the first odd $|p|$ that was
  anticipated, because for even $|p|$ the two spellings still differ
  on the odd-$|q|$ terms — and fails `m2l_fused_vs_fallback`, since only the
  table path was perturbed. $\ell_0$ and the $|p|=1$ contraction **pass**: both
  call `m2l_operator_block` with their own $R$ and never touch the key path.
  That split is the discrimination the exit criterion asks for.
- **Both dropped together** still fails, and the parity identity now fails at
  $|p| = 1$ with $\rm lhs = -\rm rhs$ exactly: the two errors *do* cancel at
  $|p| = 0$ and not above it, which is why only $|p| > 0$ discriminates them.

---

### T4 — A full-pipeline solve at `near_softening_factor = 0` — **DONE**

**Depends on:** T3.

**Met.** `tests/tstCartesianTaylorSolve.hpp` drives
`Solver<…, double, P, 3, CartesianTaylorBasis>` through 4 solves with
`migrate / rebalance / migrate` between them, at np 1-6, against a direct
**softened** sum at an explicit `softening = 0.025` with
`near_softening_factor = 0`, at two admissibilities. Both suites build and one
flux job runs both under `ctest -V`, with `m2l_n_unique_ops() > 0` asserted per
rank in both arms (26180 at $\theta = 0.3$, np=1).

At $\theta = 0.3$ the relative deviation is **at or below 1e-3 on both
potential and gradient** — $1.896\times10^{-5}$ and $7.013\times10^{-4}$, the
latter clearing the bar by $1.43\times$. That arm runs at **$p = 3$**, not the
reference's 2, by explicit decision. At $p = 2$ its gradient is
$8.996\times10^{-3}$, and that is structural rather than a defect: the gradient
of a degree-$p$ Taylor local is degree $p-1$, so $\nabla\varphi$ truncates one
order before $\varphi$, while `treecode.py` has **no target-side expansion**
and carries no such term. Its documented 1e-3 is a source-side-only *velocity*
figure — the same truncation order as this solve's *potential*, which meets the
bar at $p = 2$. **R1**, **R2** and **R4** were each excluded by measurement
before the order was changed, R2 by re-running T1's oracle at this solve's
actual $(r,b)$ band, which the original sample set never covered. The
$\theta = 0.5$ deviation is measured and pinned beside it at $p = 2$.

`Canopy_Test_LaplaceSolve_MPI_SERIAL` passes all four bodies at np 1-6 with
**no** regeneration of `tests/data/laplace_solve_P6.txt`. The `LaplaceKernel`
failure direction fails the same comparison by **three orders of magnitude** on
the potential ($1.894\times10^{-2}$ against $1.896\times10^{-5}$). **R7 did not
fire: `src/Canopy_Solver.hpp` needed no edit**, so the list of
`LaplaceKernel`-specific member bodies this task was expected to produce is
empty — `Solver` is basis-agnostic as written.

The particle cloud's half-span is **0.1155**, anchored so the closest
MAC-admissible pair sits at exactly $R = 4\varepsilon$, the default
`near_softening_factor` — the separation at which Canopy itself declares an
unsoftened far field unsafe. That choice is load-bearing: on the original
`[0.05, 0.95]` the MAC put every admissible pair at $R = 15.5\varepsilon$ and
the failure direction did not fire at all. Every figure and job id is in
[the progress log](cartesian-taylor-basis-progress-log.md) §T4.

**Fill in:** new `tests/tstCartesianTaylorSolve.hpp`; one line in
`tests/CMakeLists.txt` `UNIT_MPI_TESTS` (`:49-59`); new
`scripts/tuolumne/run_ctest_cartesian_taylor.flux`; `src/Canopy_Solver.hpp`, if
and only if a member body fails to compile against this basis — the minimal edit
that makes that body well-formed for **any** conforming `FarField`, never a
branch on `CartesianTaylorBasis`. The Laplace-solve half of the exit criterion
below is what guards those edits. No other file under `src/` changes.

**Reference:** `tests/tstLaplaceSolve.hpp:1494-1583` for the direct-sum body's
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
   reference's order — through a **multi-step** solve at np 1-6: several solves
   with an integration step between them and **at least one `rebalance()`**
   (`src/Canopy_Solver.hpp:344-356`). The step loop at
   `tests/tstLaplaceSolve.hpp:872-890` is the pattern, with the one difference
   that matters: it uses `migrate()` and never `rebalance()`, so its np 1-2
   bitwise gate faces a single partition (`:860-871`). This test has no bitwise
   gate and needs the opposite — a rebalance is what moves the root box and so
   empties the operator cache, and it is the only way **R5**'s stale-operator
   path is exercised here at all.
3. Compare the gathered last-step state against a direct **softened** sum:
   $\phi_i = \sum_{j\ne i} q_j (r^2 + b)^{-1/2}$
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
   `max_depth`, `replication_depth`, `softening`, `mac_theta`, the step count,
   what ran between steps, rank counts — in the style of
   `LS_CROSS_RANK_TOL` / `LS_DIRECT_SUM_TOL`
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

**Fill in:** `tests/tstCartesianTaylorSolve.hpp` — two measurement bodies, a
per-step counter line inside the harness's step loop, and a defaulted `dt`
parameter on `with_cartesian_taylor_solve` and `runArm`. No file under `src/`
changes.

**Reference:** `src/Canopy_DownwardSweep.hpp:406-416` for `set_root_half_width`'s
cache-clearing rule, `:420-455` for the counters, and
[abstract-solver-backend-progress-log.md](abstract-solver-backend-progress-log.md)
§T9 for the mechanism and for the baseline it measured on a
`key_needs_level = false` basis (zero keys rebuilt across an
`invalidate_interaction_list()`). All three counters are reached through
`Solver::downward()` (`src/Canopy_Solver.hpp:514`), which is how the harness
already reaches `m2l_n_unique_ops()`.

**Do:**

1. Run a multi-step solve whose particles move enough that the root bounding box
   drifts visibly between steps. The larger `dt` is supplied **per body**,
   through a new defaulted parameter on `with_cartesian_taylor_solve` and
   `runArm` whose default leaves the two shipped gating arms on the trajectory
   they already run; `pos_half_span`
   (`tests/tstCartesianTaylorSolve.hpp:764`) is the pattern to follow. **Do not
   raise the `CTS_DT` constant** (`:154`): `cts_dt_for_half_span` (`:232-236`)
   reads it on behalf of both gating arms, and `CTS_DEV_TOL_THETA_REF` (`:307`)
   and `CTS_DEV_TOL_THETA_CANOPY` (`:308`) were both measured at
   `CTS_DT = 1.0e-5` — the first of them being the 1e-3 bar itself, met with a
   1.41x margin. Keep the $L^{3/2}$ scaling through `cts_dt_for_half_span`
   rather than fixing an absolute `dt`.
2. Sample `m2l_op_keys_built_count()`, `m2l_op_cache_size()`,
   `interaction_list_build_count()` and the root half-width **inside the step
   loop** (`:462-500`), after each `solve()`, and print one line per step per
   rank. A single end-of-run read cannot meet this: the existing `[ct-solve]`
   line is emitted after the loop (`:519-529`) and `_m2l_op_keys_built` is
   cumulative and never reset by `clear_m2l_op_cache()`
   (`src/Canopy_DownwardSweep.hpp:682-686`), so it carries totals and not
   per-build figures. The harness is shared, so both gating arms emit the
   per-step line as well; that is harmless — the line is echo-only and asserts
   nothing.
3. Measure at **two** arms, both at $p = 2$ (`CTS_P`): `mac_theta = 0.5`
   (`CTS_THETA_CANOPY`), Canopy's own default admissibility, and
   `mac_theta = 0.3` (`CTS_THETA_REF`), the reference treecode's. The pair
   brackets both admissibilities at the order the reference itself runs
   (`treecode.py:103`), and neither arm evaluates the ladder above $|k| = 4$, so
   **R8** does not bear on this measurement. No shipped gating arm exists at
   this combination — the $\theta = 0.3$ gating arm runs at
   `CTS_P_THETA_REF = 3` — and the harness's parameterization on order already
   admits it.
4. **Do not design or implement a fix**, and do not touch
   `src/Canopy_DownwardSweep.hpp`. The behaviour is understood; what is missing
   is its magnitude on a realistic distribution. See **R6** for what the numbers
   mean.
5. Record the measured numbers in the log against **R6**, and state whether the
   cache retained anything at all across a drifting box.

**Exit criterion:** `ctest -V -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL`
passes at np 1-6 with the per-step measurement line present at every rank of
both arms, and the progress log carries the per-rank
`m2l_op_keys_built_count()` figures across at least two consecutive rebuilds
with a drifting root box, stated against the T9 baseline. **Failure direction:**
the body must check `m2l_op_keys_built_count() > 0` after the first build — a
zero there would mean no operator was ever constructed and the measurement is
vacuous. Spell it `EXPECT`, not `ASSERT`: a fatal assertion returns from the
harness and the gather below it is collective, so one rank leaving early hangs
every other rank until the walltime and destroys the log the numbers have to be
read out of (`tests/tstCartesianTaylorSolve.hpp:540-548` states the same reason
for the `m2l_n_unique_ops()` guard).

**Checkpoint commit** at the end of this task.

---

### T6 — Validate the derivative ladder at $|k| = 6$ — **NOT STARTED**

**Depends on:** T5.

**Fill in:** `tests/tstCartesianTaylor.hpp` — the `finite_difference` body and
its two constants. No file under `src/` changes.

**Reference:** **R8** for the gap and the instrument.
[cartesian-taylor-basis-progress-log.md](cartesian-taylor-basis-progress-log.md)
§T1 for the divisor scan that sized the oracle at $|k| = 4$ and for why the
second Richardson step is there, §T4 for the $p = 3$ arm that opened the gap.

**Do:**

1. Extend the finite-difference check to $|k| = 6$ — the $2p$ of the $p = 3$
   the $\theta = 0.3$ solve arm runs at (`CTS_P_THETA_REF`,
   `tests/tstCartesianTaylorSolve.hpp:147`). Give the body its own maximum
   order rather than raising `p_order` (`tests/tstCartesianTaylor.hpp:36-37`):
   `max_k = 2 * p_order` sizes the ladder every other body in the file
   allocates, and the shift and M2L bodies choose their own orders explicitly.
2. **Re-measure the Richardson step divisor at the new order.** Do not carry
   $h = L/32$ over (`fd_h_divisor`, `tests/tstCartesianTaylor.hpp:433`) and do
   not widen `fd_tol` (`:434`) to accommodate a failure: this check is the only
   oracle above $|k| = 3$, and its sharpness is what bounds how small an
   index-map or recurrence error has to be to slip through (**R2**, **R8**). At
   $|k| = 4$ the scan over $h \in \{L/8, L/16, L/32, L/64\}$ gave
   $1.3\times10^{-4}$, $1.9\times10^{-6}$, $4.3\times10^{-7}$ and
   $7.2\times10^{-6}$ — a floor at $L/32$, truncation-limited above it and
   roundoff-limited below. A 6th difference amplifies cancellation more, so the
   floor is higher and need not sit at the same $h$; find it rather than
   assuming it.
3. Report the worst deviation and the multi-index it occurs at **per degree**,
   4 through 6, rather than one figure over all of them, so the margin at each
   new order is readable and the two orders stay comparable in one log.
4. Record the scan and the achieved margin at each degree in the log.

**Exit criterion:** `make -j 4 Canopy_Test_CartesianTaylor_SERIAL` succeeds and
`ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes every body, with the
finite-difference check running at $|k| = 4 \ldots 6$ and its tolerance stated
on the assertion together with the divisor it was measured at, and the progress
log carries the divisor scan and the per-degree margins. **Failure direction:**
perturbing one §3 recurrence coefficient must push the reported deviation above
tolerance at **each** of degrees 4, 5 and 6 — which is what says the two new
degrees are exercised rather than merely enumerated, and not that one low-degree
slot carries the whole failure.
`scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` runs this unchanged.

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
**Do:** T3's two convention checks are the discriminator and must be written
before any solve is run — $\ell_0$ against §2's degree-0/1/2 closed forms, which
is the one sensitive at $|q| = 1$, and the $|p| = 1$ coefficients against the
reference's $K/dK/ddK$ contraction. If T4 misses and T3 passes, re-check the
$1/q!$ placement: the factorial belongs in the moment and in the L2P evaluation,
never in $b_k$. Then re-check the $|q|!/q!$ multinomial factor between the
reference's full symmetric tensors and this basis's multi-indexed coefficients,
which is the other route to the same wrong field.

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
the next basis. T4 fixes each one in place, with the minimal edit that makes the
body well-formed for any conforming `FarField` rather than a branch on this
basis; the Laplace-solve gate is what shows the edit moved nothing for the
existing one.

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

**R8 IS NOW LIVE.** T4 raised its $\theta = 0.3$ arm to $p = 3$, whose ladder
runs to $|k| = 6$, while T1's finite-difference oracle is validated at
$|k| = 4$ and nowhere else. That arm is therefore gating on arithmetic the
oracle has not checked. **T6 closes it** with the instrument above — the FD
check re-run at $|k| = 6$, the Richardson step divisor re-measured at the new
order rather than the tolerance widened. Nothing should lean further on
$p = 3$ until it is done.
