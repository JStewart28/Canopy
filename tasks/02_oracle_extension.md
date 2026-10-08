# Validate `CartesianTaylorBasis` at order 4: an oracle for the derivative ladder to $|k|=8$

**Status:** NOT STARTED

## Problem

`CartesianTaylorBasis<Scalar, P_ORDER, NComps>`
(`src/Canopy_CartesianTaylorBasis.hpp:355`) builds its M2L operator from the
derivative ladder: `Canopy::CartesianTaylor::derivative_ladder(r, b, max_order,
out)` (`:193-265`) fills the raw derivatives $\partial^k\varphi$ of the softened
kernel $\varphi = w^{-1/2}$, $w = |r|^2 + b$, for every multi-index
$|k| \le$ `max_order`. The M2L reads $b_{p+q}$ with both multi-indices running
to degree $p$, so it calls the ladder at `m2l_ladder_order = 2 * P_ORDER`
(`:901`), into a stack array of `num_slots(2p)` doubles (`:913-914`, `:992-993`).
At $p=4$ that is degree 8 and 165 slots.

**The ladder is validated by an independent oracle only to $|k|=6$**, which is
$2p$ at $p=3$:

- $|k| \le 3$: the closed-form tensors `ref0..ref3`
  (`tests/tstCartesianTaylor.hpp:111-167`), checked at `closed_form_tol = 1e-12`
  (`:400`) in `testClosedForms` (`:402-456`).
- $|k| = 4 \ldots 6$: a tensor-product central finite difference with two
  Richardson steps (`stencil1D` `:199-259`, `fdDerivative` `:261-281`,
  `fdRichardson` `:283-293`, `testFiniteDifference` `:516-583`), at per-degree
  divisors `fd_h_divisor = {24, 24, 16}` and tolerances
  `fd_tol = {4e-6, 3e-4, 2e-2}` (`:510-514`).

Nothing checks degrees 7 or 8. `stencil1D` aborts on a per-axis order above 6
(`:246-257`), so the finite-difference check cannot simply be extended upward.
**And it should not be extended upward**: its measured worst case grows about
40x per degree — 3.08e-7, 2.53e-5, 1.18e-3 at degrees 4, 5, 6 (`:483-492`) —
because roundoff enters as $(L/h)^{|k|}$. Continuing that trend puts degree 7
near 5e-2 and degree 8 near 2. A double-precision finite difference cannot
resolve the ladder at degree 8 at all.

**Why order 4 is needed.** A downstream application evaluates the gradient of
$\varphi$ (a velocity) on a deforming surface, with an accuracy requirement of
1e-3 in max relative error against the direct sum, at `mac_theta` 0.3. At order
3 it measures **1.2537e-3** at its worst state. At order 4, on an earlier state
of the same surface, it measured **5.61e-5**, 8.9x better than order 3's
5.01e-4 there. It intends to make order 4 its production order at `mac_theta`
0.3. It needs the ladder validated at the degrees order 4 reaches, and the
order-4 M2L and solve checked end to end, before it adopts order 4. A plateau or
an off-model rate above $p=3$ would otherwise be indistinguishable from
truncation behaviour.

**End state.** The ladder is asserted against an independent oracle at every
degree through $|k|=8$. The M2L operator tests run at $P=4$. The $\theta=0.3$
solve arm runs at $p=4$ and shows the error falling below its $p=3$ figure.
Canopy's own defaults and instantiations do not change.

### Out of scope

- **Changing `derivative_ladder` or the M2L.** This is validation. If the oracle
  disagrees with the ladder, that is a defect to record and stop on, not to fix
  under this document.
- **Asserting degrees 9 and 10** ($p=5$). The O1 oracle is generic in degree and
  may *report* them; nothing asserts them.
- **Changing any default order or instantiation in `src/`.**
- **Performance of $p=4$.** The operator table grows from 3200 to 9800 B per key
  (`tasks/cartesian-taylor-basis.md:270-274`); that is the downstream
  application's budget to manage.
- **Running clang-format** on any file (`CLAUDE.md`).

## Approach

Three tasks, in order. Each can be verified before the next starts.

1. **O1 — an arbitrary-degree closed-form oracle**, cross-validated against both
   existing oracles where they overlap, then used to assert the ladder through
   $|k|=8$.
2. **O2 — the M2L operator tests at $P=4$.** The m2m, l2l and l2p tests already
   instantiate `<2>` and `<4>` (`tests/tstCartesianTaylor.hpp:2189-2205`), but
   none of them calls the ladder. Every M2L test is fixed at
   `constexpr int P = 2` (`:1652`, `:1760`, `:1907`, `:2078`).
3. **O3 — the $\theta=0.3$ solve arm at $p=4$**, in
   `tests/tstCartesianTaylorSolve.hpp`, beside the existing $p=3$ arm.

### The oracle: the separable closed form over the radial derivatives

$\varphi = g(w)$ with $g(w) = w^{-1/2}$, and $w = r_x^2 + r_y^2 + r_z^2 + b$ is a
sum of one square per axis. Differentiating $g(x^2 + c)$ $n$ times in $x$ gives
the Hermite-type expansion

$$
\frac{d^n}{dx^n}\, g(x^2+c) = \sum_{l=0}^{\lfloor n/2 \rfloor}
\frac{n!}{l!\,(n-2l)!}\,(2x)^{n-2l}\, g^{(n-l)}(x^2+c),
$$

and since each axis enters $w$ separately, the three axes compose into

$$
\partial^k \varphi = \sum_{l_x=0}^{\lfloor k_x/2 \rfloor}
\sum_{l_y=0}^{\lfloor k_y/2 \rfloor} \sum_{l_z=0}^{\lfloor k_z/2 \rfloor}
\Bigl[\prod_{j} \frac{k_j!}{l_j!\,(k_j-2l_j)!}\,(2r_j)^{k_j-2l_j}\Bigr]\,
g^{(|k|-|l|)}(w),
$$

$$
g^{(m)}(w) = (-1)^m\,\frac{(2m-1)!!}{2^m}\, w^{-(2m+1)/2}.
$$

It is a finite sum of exact terms with no step size. It shares no code and no
recurrence with `derivative_ladder`. The only thing it shares is the radial
power $w^{-(2m+1)/2}$, which `ref0..ref3` already use (`P(m, w)` at `:97-100`).
Evaluated in `long double`, its own rounding sits well below the ladder's, so
the measured deviation is the ladder's.

**Why not more stencils.** The finite difference needs 9-point stencils at
degrees 7 and 8, a fresh divisor scan per degree, and it still loses about 40x
per degree to roundoff (see [Problem](#problem)). The closed form has no step
to scan and no truncation term, and the same code covers every degree.

**Why not the ladder in extended precision.** That would check rounding, not the
recurrence: a wrong coefficient in the recurrence would reproduce exactly.

### Conventions

| Choice | Value | Why |
| --- | --- | --- |
| Oracle file | the new functions go in `tests/tstCartesianTaylor.hpp`, beside `testClosedForms` and `testFiniteDifference` | One file holds every ladder oracle, and they share `buildSamples` (`:66-88`). |
| Oracle precision | `long double`, with a `static_assert(std::numeric_limits<long double>::digits >= 64, ...)` | On a platform where `long double` is `double`, the oracle would silently be no better than the code it checks. Fail at compile time instead. The test is host-only math (`:31-32`). |
| Sample set | the existing 76 samples from `buildSamples` | Every existing tolerance is pinned on them, and they include the band the $p=3$ solve runs in ($|r|/\sqrt b \in [9.28, 74.2]$, `:54-58`). Do not add a separate sample set. |
| Error normalization | $|{\rm got} - {\rm want}| \,/\, (\varphi / L^{|k|})$ with $L = \sqrt w$, exactly as `testClosedForms` normalizes | One scale across all oracles makes the per-degree figures comparable. |
| Tolerance rule | each tolerance is the smallest one-significant-digit value at least 10x its measured worst, per degree | The rule the existing oracles were pinned by (`:498-500`). A tolerance is set from an on-machine measurement and never widened to accommodate a failure. |
| Degrees asserted | $0 \ldots 8$ against the closed form; 9 and 10 printed, not asserted | $2p$ at $p=4$. |
| Failure-direction check | perturb one recurrence coefficient by 0.1 % in a scratch build, show the new check fails at **each** of degrees 7 and 8, then revert | The precedent is the existing test's 0.1 % perturbation of $k_j(k_j-1)$, which failed 6500x, 913x and 142x over tolerance at degrees 4–6 (`tasks/cartesian-taylor-basis-progress-log.md` T6). A check that cannot fail on a seeded defect is not evidence. |
| New `TEST()` names | `cartesian_taylor.closed_form_any_degree` (O1); O2 adds `<4>` instantiations to the existing M2L `TEST()`s; O3 adds `CartesianTaylorSolve.matchesDirectSumThetaRefP4` | Matches the existing naming at `:2183-2253` and `tstCartesianTaylorSolve.hpp:1262`. |
| Build and run | manual mode per `systems/tuolumne/claude.md`: `make -j 4 <target>` only, `make cmake_check_build_system` after any `tests/CMakeLists.txt` edit, anchored `ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` | `CLAUDE.md` and `tasks/cartesian-taylor-basis.md` Conventions (`:327-347`). A bare `make -j` has been SIGKILLed on the login node. |
| Execution | every executable runs inside a flux batch job on `-q pdebug`, never on the login node. Copy the preamble from `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` | `--flags=waitable` is refused here. Wait with `flux job status <jobid>`; never use `flux job attach`. |
| Formatting | never run clang-format; comments state units and normalization | `CLAUDE.md`. |

### Deliberate deviations

- **The finite-difference oracle stays at $|k| = 4 \ldots 6$ and is not
  extended.** It becomes O1's cross-check, not its replacement. Two
  independent oracles agreeing where they overlap is what makes the closed form
  trustworthy above the overlap.
- **`tests/tstCartesianTaylor.hpp` carries no license header**, unlike most
  tests (`tests/tstCartesianTaylorSolve.hpp:1-10`). Edits to it do not add one;
  no new file is created.

## Current state

- Branch `investigate-m2l-cap`, clean tree. Record the commit O1 starts from in
  the log.
- `derivative_ladder` has no upper bound on `max_order`. Its one runtime guard
  is `Kokkos::abort` on `b <= 0` (`:199-202`). `CartesianTaylorBasis`
  static-asserts `P_ORDER >= 0` (`:363-365`) and nothing above it.
- The conditioning concern is recorded as R8 in
  `tasks/cartesian-taylor-basis.md` (`:1596-1617`): each step divides by $w$
  and weights terms by $k_j(k_j-1)$, so a loss would be worst at small
  $|r|/\sqrt b$. R8 is closed for $p=3$ and **open above it**
  (`tasks/cartesian-taylor-basis-progress-log.md` T6, "R8 remains open above
  $p = 3$").
- Tests at $P=4$ exist only for the m2m, l2l and l2p shifts (`:2189-2205`),
  and they sit at the roundoff floor (M2M against direct P2M 1.208e-16; L2P
  gradient against FD 2.576e-14). `m2l_ell0_closed_forms` runs at $P=2,3$ and
  is capped by `static_assert(P <= 3, ...)` (`:1544-1549`), because the §2
  closed forms stop at $|k|=3$.
- The solve: `TEST(CartesianTaylorSolve, matchesDirectSumThetaRef)`
  (`tests/tstCartesianTaylorSolve.hpp:1262`) runs `runArm<..., CTS_P_THETA_REF
  = 3, ...>` at $\theta=0.3$ against `CTS_DEV_TOL_THETA_REF`, the 1e-3 bar
  itself. It measures **7.0717918545e-04** on the gradient and
  **1.9263340835e-05** on the potential, identical over SERIAL np 1–6 and HIP
  np 1–4 (`:337-355`). `CartesianTaylorSolve` is in `UNIT_MPI_TESTS`
  (`tests/CMakeLists.txt:47-58`); `CartesianTaylor` is in `UNIT_SERIAL_TESTS`
  (`:35-39`).
- `scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux` points
  `CANOPY_BUILD` at `/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne`,
  a different checkout from this one. Before using it, confirm the build tree it
  names was configured from the checkout being edited. Its header comment about
  "$|k| = 4..2p$" is stale.

## Progress log

`tasks/02_oracle_extension-progress-log.md`. Read it before implementing a task,
changing a signature this document names, or reopening a question this document
treats as settled. It carries the measured numbers.

## Task sequence

### O1 — Assert the derivative ladder against a closed-form oracle through $|k|=8$ — **NOT STARTED**

**Depends on:** none.
**Fill in:** `tests/tstCartesianTaylor.hpp`: a `long double` closed-form
evaluator for $\partial^k\varphi$ at any $k$ (see [The
oracle](#the-oracle-the-separable-closed-form-over-the-radial-derivatives)), a
`testClosedFormAnyDegree()` body, its per-degree tolerance array, and its
`TEST(cartesian_taylor, closed_form_any_degree)` registration beside `:2187`.
**Reference:** `testClosedForms` (`:402-456`) for the loop shape and
normalization; `testFiniteDifference` (`:516-583`) for the per-degree
worst-case report and the post-loop `EXPECT_LE` pattern the T6 log explains
(`tasks/cartesian-taylor-basis-progress-log.md` T6, "the per-degree probe
needed an `EXPECT_LE` after the loop").
**Do:**
1. Write the evaluator with the multinomial and double-factorial weights
   computed exactly (integers or `long double`), never through `std::tgamma`. It
   returns `long double`.
2. **Validate the oracle before asserting anything with it**, at every sample:
   - at $|k| \le 3$, against `referenceClosedForm` (`:150-167`) under the same
     normalization, at `closed_form_tol` (`1e-12`);
   - at $|k| = 4 \ldots 6$, against `fdRichardson` at the existing divisors,
     within the existing `fd_tol` per degree;
   - against itself evaluated in `double`, reporting the per-degree worst gap.
     That gap measures the closed form's own cancellation (R1).
   If either of the first two fails, the oracle is wrong: stop and record it.
3. Assert the ladder (`derivative_ladder` at `max_order = 8`) against the
   oracle at every $k$ with $|k| \le 8$, at all 76 samples. Print the worst
   deviation per degree **and the sample it occurs at** ($b$, $|r|/\sqrt b$,
   direction). Run once with provisional tolerances of 1, read the on-machine
   worst per degree, then pin each tolerance by the rule in
   [Conventions](#conventions).
4. Print, without asserting, the worst deviation at degrees 9 and 10.
5. Run the failure direction: a 0.1 % perturbation of one recurrence coefficient
   (the $k_j(k_j-1)$ weight, `:251-258`) in a scratch build must fail
   `closed_form_any_degree` at each of degrees 7 and 8. Record the
   over-tolerance factors, then revert. `git diff src/` must be empty at the
   end.
**Exit criterion:** in a pdebug flux job, `ctest -V -R
'^Canopy_Test_CartesianTaylor_SERIAL$'` passes, including
`cartesian_taylor.closed_form_any_degree`. Its output carries a pinned tolerance
and a measured worst for every degree 0–8, each at least 10x under its
tolerance, plus the oracle's own cross-check figures from step 2. The existing
`closed_forms` and `finite_difference` tests pass unchanged. In the failure
direction, the step-5 perturbation fails the new test at degree 7 **and** at
degree 8, and the log records both factors.

### O2 — Run the M2L operator tests at $P=4$ — **NOT STARTED**

**Depends on:** O1 **DONE**.
**Fill in:** `tests/tstCartesianTaylor.hpp`: `testM2LParityIdentity` (`:1758`),
`testM2LFusedVsFallback` (`:1905`) and `testM2LEndToEnd` (`:2076`), each
templated on `P` in place of its `constexpr int P = 2`, and their `TEST()`s
(`:2215-2219`) calling `<2>` and `<4>`, the shape `m2m_shift` already uses
(`:2189-2193`).
**Reference:** `testM2MShift<P>` and its rationale for instantiating at 4
(`:601-613`: low-order checks degenerate).
**Additional information needed:** whether each of the three bodies is generic
in `P` or carries order-2 assumptions (hardcoded slot counts, expected values,
tolerances derived at $P=2$). Read each body first.
`testM2LP1Contraction` (`:1650`) is order-specific by construction and stays at
$P=2$. Any tolerance that was derived at $P=2$ is re-measured at $P=4$ and
pinned by the same rule, never reused.
**Do:**
1. For each of the three bodies, list its $P$-dependent literals in the log
   before changing anything.
2. Template the body on `P`, keep `<2>` bit-for-bit (its tolerances and expected
   values unchanged), and add `<4>`.
3. Record the measured figure at $P=4$ for each, beside its $P=2$ figure.
**Exit criterion:** `ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'` passes
in a pdebug flux job, and its output shows `m2l_parity_identity`,
`m2l_fused_vs_fallback` and `m2l_end_to_end` each exercised at $P=2$ and $P=4$.
The $P=2$ figures equal the pre-change figures. In the failure direction,
`m2l_fused_vs_fallback<4>` fails when `m2l_ladder_order` is set to `2 * P_ORDER
- 1` in a scratch build, and the log records the failure. Revert after; `git
diff src/` is empty.

### O3 — The $\theta=0.3$ solve at $p=4$ — **NOT STARTED**

**Depends on:** O2 **DONE**.
**Fill in:** `tests/tstCartesianTaylorSolve.hpp`: a constant
`CTS_P_THETA_REF_P4 = 4` beside `CTS_P_THETA_REF` (`:149`), a tolerance
`CTS_DEV_TOL_THETA_REF_P4`, and `TEST(CartesianTaylorSolve,
matchesDirectSumThetaRefP4)` beside `:1262`, calling the same `runArm` with the
same particle set, `ncrit` and $\theta$.
**Reference:** the $p=3$ arm and its tolerance comment (`:337-355`); the
existing flux runners for `CartesianTaylorSolve` in `scripts/tuolumne/`.
**Do:**
1. Run the new arm with a provisional tolerance of 1e-3 (the bar) at SERIAL np 1
   and np 4 and HIP np 1 and np 4. Record the gradient and potential error at 11
   digits, as the $p=3$ arm's comment does.
2. **The curve must keep falling.** The $p=4$ gradient error must be at most
   one quarter of the $p=3$ figure, 7.0717918545e-04. The downstream
   application measured 8.9x between $p=3$ and $p=4$ on its own geometry. A
   ratio under 4 is the plateau R3 describes: stop and record it, and do not pin
   a tolerance around it.
3. Pin `CTS_DEV_TOL_THETA_REF_P4` as the smallest one-significant-digit value at
   least 2x the worst measured gradient error across the four launches. That is
   the margin convention for solve-level bounds, since the 10x rule is for
   oracle deviations. Write the measured figures and the jobs into the comment,
   as `:337-355` does.
**Exit criterion:** the anchored `ctest` for the `CartesianTaylorSolve` targets
passes at SERIAL np 1, 4 and HIP np 1, 4 in pdebug flux jobs, including
`matchesDirectSumThetaRefP4`. The log carries the $p=4$ gradient and potential
errors at all four launches, and the $p=3$/$p=4$ gradient ratio is at least 4.
The existing `matchesDirectSumThetaRef` figures are unchanged. In the failure
direction, the new arm built at $p=3$ in a scratch build fails its pinned
tolerance.

## Known risks

**R1 — The closed form cancels at large $|r|/\sqrt b$.** Terms alternate in
sign through $(-1)^{|k|-|l|}$, so at $|r| \gg \sqrt b$ the sum loses digits.
**Presents as:** an oracle-versus-ladder deviation that grows with
$|r|/\sqrt b$ rather than shrinking. **Distinguishing measurement:** O1 step 2's
`long double`-versus-`double` gap of the oracle itself. If that gap is
comparable to the oracle-versus-ladder deviation at the same sample, the oracle
is the problem, not the ladder.

**R2 — The ladder is ill-conditioned at small $|r|/\sqrt b$** (R8 of
`tasks/cartesian-taylor-basis.md`). **Presents as:** a deviation that peaks at
$|r|/\sqrt b = 0.01$ and grows with degree faster than the oracle's own gap.
That is a finding about the recurrence. Record it with the sample, and do not
widen a tolerance to absorb it. If it is above the tolerance rule's reach at
degree 7 or 8, $p=4$ is not validated, and the downstream application must hear
that rather than a loosened pass.

**R3 — The order-4 solve plateaus.** **Presents as:** O3's $p=4$ gradient error
less than 4x below $p=3$'s. After O1 and O2 pass this cannot be a ladder or
operator defect; it is a property of the expansion on this particle set.
Record it. It does not block O1 or O2 from being **DONE**.

**R4 — Two oracles disagree in the overlap.** **Presents as:** O1 step 2 failing
at $|k| = 4 \ldots 6$ while the existing `finite_difference` test still passes.
The finite difference has been validated at those degrees on this machine
(T6), so the closed-form evaluator is wrong until shown otherwise. Check its
weights first.
