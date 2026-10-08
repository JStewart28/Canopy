# Canopy as the far-field engine for a distributed vortex-sheet solver — progress log

Session record for canopy. Companion to `canopy0.md`, which holds the design, the
task sequence and the risks; this file holds what actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `canopy0.md` can cite it by ID.
No dates: the order of the sections is the chronology. If a session covers more
than one task, name them all; if it belongs to no task, name the topic.

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

Every accuracy number written here must carry the qualification list the
design's conventions table requires: the source distribution, the rank counts,
`P_ORDER`, `ncrit`, `max_depth`, `mac_theta` and the softening it was measured
at. A bare tolerance is not a measurement.

## Topic: second read-only pass — treecode comparison merged into the design

No task was started and no code was built or run. A second read-only pass opened
`src/Canopy_LaplaceKernel.hpp` in full (887 lines) alongside a reference
Barnes–Hut treecode (`~/research-bridges/zmodel-steve/zmodel3d-amr/zmodel3d/treecode.py`,
138 lines) to answer the question **C1** had left open under *Additional
information needed*: whether a softened far field is reachable without replacing
the expansion basis. The findings were written up separately first
(`~/spack_envs/tuolumne_beatnik/beatnik/tasks/treecode-vs-canopy.md`) and have now been merged into `canopy0.md` as
**F6**, **F7**, **F8**, F4's exact-node-radius note, C1's revised four-option
step 3, **C11**, **R7** and **R8**. `~/spack_envs/tuolumne_beatnik/beatnik/tasks/treecode-vs-canopy.md` is retained as
the provenance record of that pass; `canopy0.md` is now the authoritative
statement.

What the pass changed in the design's substance, as opposed to adding to it:

- **A premise in the original framing was wrong.** M2M/M2L/L2L/L2P are *not*
  missing from Canopy's kernel — all five operators are implemented in the
  solid-harmonic basis with Greengard theorem citations
  (`src/Canopy_LaplaceKernel.hpp:403`, `:446`, `:551`, `:1220`, `:1337`, plus the
  precomputed-operator path at `:516`). What is missing is softening *inside*
  them. F1's conclusion is unchanged; only the diagnosis of why is.
- **C1's open question is answered: no.** The solid-harmonic addition theorems
  hold because $1/r$ is harmonic, and $(r^2+b)^{-1/2}$ is not
  ($\nabla^2\phi_b = -3b/(r^2+b)^{5/2}$). There is no coefficient substitution
  that makes the existing M2L evaluate the softened kernel; a softened far field
  is a second expansion basis, not a patch. Recorded as F6(a).
- **The cost of that basis was underestimated in C1's original framing.** Beyond
  the second basis, softening breaks the scale invariance every operator's width
  normalization rests on, and re-keys the precomputed M2L operator cache — whose
  documented property is that the operator *"depends only on (dd, ii, jj, kk); no
  physical width enters"* (`src/Canopy_DownwardSweep.hpp:1075`), true only for a
  kernel with no absolute length scale. F6(c).
- **A cheaper option existed that C1 did not list.** A softened M2P mode reusing
  the existing tree, partition, interaction list and P2P — dropping M2L/L2L/L2P
  from the path rather than softening them — removes the bias entirely at
  materially less work. It is now C1 option (d), and F8 holds the analysis. Its
  viability rests on one unmeasured number (per-target M2P vs. per-cell
  M2L+L2L+L2P), which is why it is a measurement in C1 step 3 and a risk (R7),
  not a recommendation.
- **The accuracy landscape collapsed to one number.** The status quo's honest
  claim and the M2P ceiling are both $\sim\!10^{-3}$. So the choice between
  them is about the *kind* of error (fixed bias vs. truncation controllable by
  `mac_theta` and order), not its size. $10^{-6}$ is out of reach for a
  low-order softened far field too, which makes "what does the consumer actually
  need" the cheapest open question on the list (R8).
- **Canopy's far-field gradient is a finite difference**, not analytic:
  six extra potential evaluations at $h = 10^{-5}w_{\rm self}$ with an in-code
  `TODO` (`src/Canopy_LaplaceKernel.hpp:1388-1416`). It contributes a third,
  $\sim\!10^{-10}$ plateau to the `P_ORDER` scan C1 step 1 and R1 are built
  around. F7, task C11.

Also: every `\f$`/`\f[` in `canopy0.md` was converted to KaTeX `$`/`$$` per the
repository's *Math in markdown* rule. The Doxygen delimiters had been rendering
as prose in a plain markdown reader, with CommonMark stripping the backslash from
every escaped punctuation character inside them.

Nothing here is a measurement. Every accuracy figure in F6/F8 is an estimate or
a number carried from `~/spack_envs/tuolumne_beatnik/beatnik/tasks/treecode.md`; the qualification list the conventions
table requires cannot be supplied for any of them, and no tolerance should be set
from them.

**Affects:** **C1** — its option set, its *Additional information needed* (now
narrowed to the Dehnen 2000/2002 literature check) and its exit criterion all
changed; do not start it from the pre-merge text. **C5** — gains step 5, the
exact-node-radius measurement. **C8** — a `Gradient`-only far field cannot skip
the potential internally until C11 lands. **C11** — new, and worth doing before
C1 step 1's scan is interpreted. **R1** — the scan has three plateaus, not two.

## C11

`LaplaceKernel::l2p_evaluate` now returns an analytic gradient. Commits
`93a254d` (step 1), `6c80b2f` (steps 2–3), `775a397` (step 4). Exit criterion
met on both backends; see the job list at the end.

**Decisions** (the first four were made before the session started):

- `tests/tstLaplaceKernel.hpp` was made to compile by signature changes only.
  Every operator call gains its half-widths, all `1.0`, so that
  $\bar M = M$ and $\bar L = L$ and every closed-form expectation keeps its
  meaning; the bare `A` table becomes the basis `aux` struct. No assertion
  changed in step 1. README Known Issues keeps only the `P2P` half (retitled
  "The `P2P` `unit` test target does not compile").
- `tests/data/laplace_solve_P6.txt` regenerated only after the unit test
  passed; `LS_CROSS_RANK_TOL` and `LS_DIRECT_SUM_TOL` unchanged.
- FP32 out of scope; `SolveFusedM2L.FP32_smokeTest` untouched.
- `SingleSolve` np=4 (C6) not in the gate and not touched.
- `l2p_evaluate` keeps its signature, its `=` writes and its $+\nabla\phi$
  sign. The potential's arithmetic is unchanged expression for expression
  (`L * rho_pow[n] * Y`, same loop order); the only change on that path is
  that $Y_{n,m}$ is tabulated once per particle (28 `Ynm` calls at $P=6$)
  instead of once per stencil point (196).
- **Departure from Do step 3.** The new test compares the analytic gradient
  with *two* finite differences, not one. Asserting
  $|g_{\rm an} - g_{\rm fd2}| / |\nabla\phi| < 10^{-8}$ against the old scheme
  (second order, $h = 10^{-5} w_{\rm self}$) failed at `3.4295694e-08`
  (job `f3cvdZuu4zbH`). The cause is the old scheme's own error, not the
  analytic form: at that step it is roundoff-dominated
  ($\sim\varepsilon S/h$, $S$ the scale of the summed terms) and reaches
  $3.43\times10^{-8}$ where the gradient is small against the potential.
  `testL2PGradientAnalyticVsFD` therefore asserts against a fourth-order
  stencil at $h = 10^{-3} w_{\rm self}$ (`< 1e-8` relative), and asserts that
  the analytic gradient is no farther from the old scheme than the old scheme
  is from the fourth-order one (`|an - fd2| <= |fd2 - fd4| + 1e-8`).
- `testL2PGradient` tightened `1e-7` → `1e-11`. That test's floor is now the
  $P=6$ truncation of the local expansion's derivative, not the gradient
  scheme: the exact gradient error of the truncated expansion at its
  geometry, computed independently at 40 digits with mpmath, is
  `8.751218e-12`; the analytic gradient returns `8.751250e-12`. The FD's
  smaller `4.333e-12` was its own error partly cancelling the truncation.
- `canopy_ctest` refuses an entry with no budget row, and `LaplaceKernel` and
  `SingleSolve` had none. Rows were calibrated as `fix-hang-rebalance.md`
  prescribes (maximum of three passes, job `f3cvZzMaek2b`, on the FD code, so
  an upper bound for the analytic one) and added to
  `scripts/tuolumne/serial_runtimes.tsv`: `LaplaceKernel` 5.87 s,
  `SingleSolve` np 1,2,3,5,6 = 6.18, 5.96, 7.49, 9.66, 10.83 s.

**Identity.** The ladder relations of the regular solid harmonics
$R_n^m = \rho^n Y_{n,m}$ (Dehnen 2014, *Comput. Astrophys. Cosmol.* 1:1,
eq. 51, $\Delta_1^l \Upsilon_n^m = (-1)^{1+l}\Upsilon_{n-1}^{m+l}$), mapped to
this file's `Ynm` by $R_n^m = (-1)^m\sqrt{(n-m)!(n+m)!}\,\Upsilon_n^m$. For
$m \ge 0$:

$$
\partial_z R_n^m = \sqrt{(n-m)(n+m)}\,R_{n-1}^m,\qquad
(\partial_x + i\partial_y) R_n^m = \sqrt{(n-m)(n-m-1)}\,R_{n-1}^{m+1},
$$

$$
(\partial_x - i\partial_y) R_n^m = -\sqrt{(n+m)(n+m-1)}\,R_{n-1}^{m-1}\ (m\ge1),\qquad
(\partial_x - i\partial_y) R_n^0 = \overline{(\partial_x + i\partial_y) R_n^0}.
$$

In the normalized variable $(\rho/w_{\rm self})^n Y_{n,m}$ each derivative
brings one $1/w_{\rm self}$. Checked numerically for $n \le 8$ against a
central difference (max abs difference 2e-10, FD-limited) before it was
written in C++, and by the unit test after.

**Error floor, operator level** — `Canopy_Test_LaplaceKernel_SERIAL`, one
process, $P$ = `P_ORDER` = 6, `NComps` = 1, softening 0. There is no tree, so
`ncrit`, `max_depth` and `mac_theta` do not apply; the separation ratios are
given instead.

| measurement | before (FD, `f3cvZzMaek2b`) | after (analytic, `f3cvrmBCmKd1`) |
| --- | --- | --- |
| `testL2PGradient`: $q = 1.5$ at the source center, target center 6 away on $x$, offset $(0.10, 0.06, -0.04)$, widths 1 | `4.333e-12` abs, `1.075e-10` rel | `8.751e-12` abs, `2.171e-10` rel (= truncation `8.7512e-12`) |
| `testL2PGradientAnalyticVsFD`: 50 sources uniform in a cube of half-width 0.4, charges $U[-1,1]$, seed 4567; target leaf half-width 0.25 at $(2.5, 1.2, -0.9)$ (ratio $(r_s + r_t)/d = 0.39$); 66 points: center, $z$-axis pole, 64 uniform in the leaf | old scheme's own error $\max|g_{\rm fd2} - g_{\rm fd4}|/|\nabla\phi|$ = `3.43e-8` | $\max|g_{\rm an} - g_{\rm fd4}|/|\nabla\phi|$ = `2.14e-11` (fd4's own floor) |

Both gradients sit `1.06e-5` relative from the direct sum at the second
geometry: truncation, unchanged by the scheme.

**Error floor, solve level** — `LaplaceSolve`'s frozen configuration: 600
particles uniform on $[0.05, 0.95]^3$ (global set, seed $1234 + P$), charges
$U[0.5, 1.5]$, 12 steps at $dt = 10^{-4}$ with `migrate()` between steps,
$P$ = 6, `NComps` = 1, `ncrit` = 16, `max_depth` = 6, `mac_theta` = 0.5,
softening 0. Deviations are normalized by the global max of each field.
"Before" is the FD kernel and the old data at the current tree (job
`f3cvvFhTd4Fh`: HEAD `775a397` with `src/Canopy_LaplaceKernel.hpp` and the
data file temporarily restored from `93a254d`, then restored and rebuilt).

| | before, SERIAL np 1–6 | after, SERIAL np 1–6 | after, HIP np 1–4 |
| --- | --- | --- | --- |
| cross-rank, gradient (np 2–6 / 2–4) | `7.8e-13` – `3.2e-12` | `3.1e-18` – `1.9e-16` | `1.7e-16` – `1.9e-16` |
| cross-rank, potential | `1.3e-13` – `6.2e-13` | `9.6e-16` – `1.5e-15` | `1.2e-15` – `1.5e-15` |
| direct sum, gradient | `4.332478e-08` – `4.332484e-08` | `4.3324813e-08` at every np, to 11 digits | same |
| direct sum, potential | `3.3063265e-07` | `3.3063265e-07` | same |

So at solve level the gradient's error against the direct sum is truncation,
and the FD contributed nothing visible there. What it did contribute was noise
in the *cross-rank* comparison: dividing a reassociation-level potential
difference by $h$ inflated it about four orders of magnitude, and through the
velocity update the positions, and hence the potential, diverged with it. With
the analytic gradient both cross-rank figures are at reassociation level
(~1e-16 to 1e-15). `LS_CROSS_RANK_TOL` (5.6e-10) is unchanged; it gates
potential and gradient separately, and against the worse of the two (potential,
`1.4936928404867131e-15`, np=2) its margin is now ~4e5×. The FD-era figures in
`tests/tstLaplaceSolve.hpp`'s header comment (potential 2.8e-13–1.2e-12,
gradient 1.5e-12–6.2e-12) were replaced by these measurements in a follow-up
commit; only the comment changed, not the tolerances.

**`LaplaceSolve` against the old data** (job `f3cvdZuu4zbH`, the `6c80b2f`
code uncommitted on `93a254d`): `bitForBitArtifacts` fails at np=1 (rank 0)
and np=2 (ranks 0 and 1), on the `locals()` hash only
(`tstLaplaceSolve.hpp:1414` and its dump at `:1422`); the operator table, the
$A_{n,m}$ table, the key list and the initial-set hash all still matched.
`crossRankAgreement` *passed* at np 2–6 against the old np=1 field (gradient
`1.117e-11`, potential `2.22e-12`, against `5.6e-10`), and
`matchesDirectSum` passed at np 1–6. The regeneration was therefore required
by the bitwise gate; the cross-rank gate alone could not see the change. The
regenerated file (job `f3cvgaNGZL3H`) differs from the old one in exactly the
three `locals` hashes and the 600-line np=1 `field` record, with
`n_unique_ops` 724 (np 1) and 393/462 (np 2) as before.

**Jobs:** `f3cvZzMaek2b` — budget calibration and step 1 (10/10 on the FD
code). `f3cvdZuu4zbH` — first analytic run, FD-only comparison failing at
3.43e-8, and `LaplaceSolve` against the old data. `f3cvfkDzozgf` —
`LaplaceKernel` 11/11 with the fourth-order reference and the `1e-11`
tolerance. `f3cvgaNGZL3H` — regeneration. `f3cvrmBCmKd1` — SERIAL gate at
`775a397`: `LaplaceKernel`; `LaplaceSolve`, `MultiSolve`, `DownwardSweep`
np 1–6; `SingleSolve` np 1,2,3,5,6; every entry `outcome=completed`.
`f3cvrmKe1Eco` — HIP gate at `775a397`: `LaplaceSolve`, `MultiSolve`,
`DownwardSweep` np 1–4, every entry `outcome=completed`. `f3cvvFhTd4Fh` —
the FD solve-level baseline above.

**Affects:** **C1** — step 1's `P_ORDER` scan now has two plateaus, not three:
truncation and softening bias. The gradient scheme's own floor is roundoff,
≤ 2.1e-11 relative at operator level (bounded by the fourth-order reference,
not by the analytic form, which matches a 40-digit truncation reference to
3e-17 absolute), and ~1e-16 in solve-level cross-rank agreement. Any plateau
the scan shows above ~1e-11 is truncation or softening bias, never the
gradient scheme. **C8** — a `Gradient`-only far field no longer needs the
potential internally: the gradient loop in `l2p_evaluate` reads only the local
coefficients and the $Y_{n,m}$ table, never $\phi$, so C8 can skip the
potential accumulation there (it is still computed unconditionally today).
**C5** — its gradient-accuracy figures will contain no finite-difference
component. **R1** — the "three plateaus" caveat no longer applies.
