# Why the MultiSolve "rebalance" cases show larger error than the stable case

**Date:** 2026-08-28
**Trigger:** `fmm_tol` lowered to `1.0e-8` in all `tests/tstMultiSolve.hpp` cases to
expose the true accuracy of the FMM solve.
**Backend / config:** `MPI_SERIAL`, np=4, `P_ORDER=8`, `mac_theta=0.5`,
`softening=0.0`, `ncrit=16`, `max_depth=6`, 200 particles/rank (800 total).

## Answer in one line

The maintenance mode is **not** the cause. `migrate`, `rebalance`, and `rebuild`
produce bit-identical trajectories when given the same integrator settings. The
rebalance/rebuild tests look less accurate only because they *also* run with a
10× larger `dt` and a 5×/50× larger `drift_multiplier`, and the assertion is on
an **accumulated trajectory**, not on the FMM force. Trajectory error grows
super-linearly with per-step particle motion, so those settings amplify the same
underlying force error by 5 orders of magnitude.

## Observed errors at `fmm_tol = 1e-8`

From `build-tuolumne/multisolve4.out` (np=4):

| Test | maintenance | dt | drift | steps | tree_tol | `pos_max_rel` | `vel_max_rel` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `StableTree_Migrate` | Migrate | 1e-4 | 1.0 | 5 | 0.1 | **< 1e-8** (passed) | 2.969e-06 |
| `IntermediateMotion_Rebalance` | Rebalance | 1e-3 | 5.0 | 5 | 0.1 | 1.797e-04 | 2.771e-04 |
| `AutoMaintain` | Auto → Rebuild | 1e-3 | 5.0 | 5 | 0.1 | 1.797e-04 | 2.771e-04 |
| `LargeMotion_Rebuild` | Rebuild | 1e-3 | 50.0 | 4 | 0.1 | 1.729e-03 | 1.600e-03 |
| `AutoRebalance` | Auto → Rebalance | 1e-3 | 2.0 | 8 | 0.3 | 5.105e-05 | 5.065e-04 |
| `M2L_BinEdge_Fallback` | Migrate | 1e-4 | 1.0 | 2 | 0.1 | < 1e-8 (passed) | 1.126e-06 |

Note that the tests vary **three** things at once — maintenance mode, `dt`, and
`drift_multiplier` — so the table on its own cannot attribute the error to any
one of them. That confound is the whole story.

### The control that is already in the suite

`IntermediateMotion_Rebalance` and `AutoMaintain` use *identical* integrator
settings (`dt=1e-3`, `drift=5.0`, 5 steps, `tree_tol=0.1`, `ncrit=16`,
`max_depth=6`) but drive **different maintenance code paths**:
`rebalance()` (build → `_finish_topology_change`) every step versus
`auto_maintain()`, which at `drift=5` sees bounding-box escape and takes the
`rebuild()` / `_full_setup` branch every step.

Their errors agree to 8–9 significant figures:

```
pos: 1.7969882928087631e-04   vs   1.7969882986542315e-04
vel: 2.7710042443852254e-04   vs   2.7710042438012350e-04
```

Two structurally different maintenance flows, same answer to ~1e-8 relative. If
the maintenance path were responsible for the 1e-4 error, these two numbers
could not match like this.

## Controlled experiment

To remove the confound I added a temporary diagnostic
(`tests/tstRebalanceDiag.hpp`, see [Reproducing](#reproducing)) that runs the
full 3×3 matrix of **maintenance mode × integrator settings** and, per step,
reports both:

- `force_max_rel` — max relative error of the FMM gradient against a brute-force
  $N^2$ reference evaluated at the FMM's **own current positions**. This is pure
  FMM accuracy for the current tree, independent of trajectory divergence.
- `pos_max_rel` / `vel_max_rel` — the accumulated trajectory error against an
  independent brute-force shadow run. This is what `tstMultiSolve.hpp` asserts on.

### Result 1 — maintenance mode is irrelevant

At np=4, for every column of the matrix, all three modes give the same
trajectory error:

| dt, drift | Migrate `pos_max_rel` | Rebalance `pos_max_rel` | Rebuild `pos_max_rel` |
| --- | --- | --- | --- |
| 1e-4, 1.0 (5 steps) | 1.0143e-09 | 1.0103e-09 | 1.0103e-09 |
| 1e-3, 5.0 (5 steps) | 1.7970e-04 | 1.7970e-04 | 1.7970e-04 |
| 1e-3, 50.0 (4 steps) | 1.7289e-03 | 1.7289e-03 | 1.7289e-03 |

`Rebalance` and `Rebuild` are bit-identical to each other in every column.
`Migrate` differs from them only in the `dt=1e-4` column, and only by 0.4%.

The decisive row is the middle one: **running `Migrate` — the "stable" mode — at
the rebalance test's `dt` and `drift` reproduces the rebalance test's error
exactly (1.7970e-04).** The mode contributes nothing; the integrator settings
contribute everything.

The maintenance path *does* perturb the FMM force slightly — repartitioning
changes summation order and the near/far split, so at `dt=1e-4` `Migrate` shows a
smooth `force_max_rel` of 3.196–3.206e-06 while `Rebalance`/`Rebuild` oscillate
over 3.27–3.87e-06 (~20% spread). That 20% force perturbation moves the
trajectory error by 0.4%, which is five orders of magnitude below the effect
being investigated.

### Result 2 — the stable case follows exact linear error accumulation

For symplectic Euler with a per-step force error $\delta g$, the accumulated
error after $n$ steps is

$$
\delta v_n \approx n \, \Delta t \, \delta g,
\qquad
\delta r_n \approx \frac{n(n+1)}{2} \, \Delta t^2 \, \text{drift} \; \delta g .
$$

The `dt=1e-4, drift=1` column matches the $n(n+1)/2$ law to within 0.1%:

| step | `pos_max_rel` observed | $n(n+1)/2 \times 6.754\text{e-}11$ |
| --- | --- | --- |
| 0 | 6.7539e-11 | 6.754e-11 |
| 1 | 2.0267e-10 | 2.026e-10 |
| 2 | 4.0544e-10 | 4.052e-10 |
| 3 | 6.7596e-10 | 6.754e-10 |
| 4 | 1.0143e-09 | 1.013e-09 |

This is the regime the `StableTree_Migrate` test lives in: the trajectory error
is a clean, predictable accumulation of a ~3e-6 relative force error, and nothing
else is happening.

### Result 3 — the rebalance settings leave that regime immediately

The `dt=1e-3, drift=5` column (np=4, Migrate mode — mode is irrelevant per
Result 1), instrumented with the root bounding-box extent and the largest
velocity magnitude in the system:

| step | ncells | box extent | max \|v\| | `force_max_rel` | `pos_max_rel` |
| --- | --- | --- | --- | --- | --- |
| 0 | 176 | 9.572e-01 | 2.259e+01 | 3.2063e-06 | 3.3827e-08 |
| 1 | 181 | 9.418e-01 | **3.103e+04** | 3.2080e-06 | 4.3039e-05 |
| 2 | 23 | **2.070e+02** | 3.103e+04 | 3.3792e-15 | 4.2019e-05 |
| 3 | 21 | **4.141e+02** | 3.103e+04 | 1.8542e-15 | 1.1915e-04 |
| 4 | 21 | **6.211e+02** | 3.103e+04 | 7.8560e-16 | 1.7970e-04 |

Contrast the stable column (`dt=1e-4, drift=1`), where the box never moves:

| step | ncells | box extent | max \|v\| | `force_max_rel` | `pos_max_rel` |
| --- | --- | --- | --- | --- | --- |
| 0 | 176 | 9.572e-01 | 2.270e+00 | 3.2063e-06 | 6.7539e-11 |
| 4 | 176 | 9.567e-01 | 1.865e+01 | 3.1960e-06 | 1.0143e-09 |

Three things jump out.

**(a) The trajectory error blows past the linear law at step 1.** The $n(n+1)/2$
law predicts 1.01e-07 at step 1; the observed value is 4.30e-05, a factor of 425
too large — in a single step. Linear accumulation of FMM truncation error is no
longer the dominant term.

**(b) The FMM force becomes essentially exact from step 2 onward, yet the
trajectory error keeps growing.** `force_max_rel` drops to ~1e-15 — round-off,
i.e. the solve is reproducing brute force to machine precision — while
`pos_max_rel` continues climbing 4.2e-05 → 1.19e-04 → 1.80e-04. **The error the
test reports at the end is not being produced by FMM inaccuracy at all.** It is
the step-0/step-1 perturbation being amplified by the dynamics.

**(c) The cause of both is a runaway particle that destroys the tree.** At step 1
the maximum velocity in the system jumps from 22.6 to **3.1e+04** — three orders
of magnitude — and stays pinned there. `softening` is 0.0, so a close pair
generates a near-singular force; at `dt·drift = 5e-3` (50× the stable case's
1e-4) one kick from that force is enough to eject a particle. From then on the
bounding box grows by a fixed ~207 per step, exactly the ejected particle's
displacement `dt · drift · |v| = 1e-3 × 5 × 3.1e4 ≈ 155` plus padding: 0.94 →
207 → 414 → 621.

That is what collapses the cell count and drives `force_max_rel` to 1e-15. With
a box extent of 621 and `max_depth=6`, the finest possible leaf is 621/64 ≈ 9.7
wide, while the other 799 particles still occupy a region of size ~1. They all
fall into a single leaf, the tree cannot refine (21 cells), every interaction
lands in the P2P near-field, and the "FMM" degenerates to direct summation —
which is exact, and which means those steps exercise essentially none of the
far-field pipeline.

In the stable column none of this happens: max \|v\| reaches only 18.6, per-step
displacement is `1e-4 × 1 × 18.6 ≈ 2e-3`, and the box extent is unchanged at
0.957 through all 5 steps with a constant 176 cells.

### Result 4 — mechanism: divergence feedback near singular pairs

Once the FMM and brute-force trajectories differ by $\delta r$, the force
difference between them is no longer the FMM truncation error $\delta g_{\rm FMM}$
but

$$
\delta g \approx \delta g_{\rm FMM} + \left| \frac{\partial g}{\partial r} \right| \delta r ,
\qquad
\left| \frac{\partial g}{\partial r} \right| \sim \frac{2q}{r^3} .
$$

With `softening=0.0` and uniformly random positions, the closest pair in 800
particles has $r \sim 10^{-2}$, so $|\partial g / \partial r| \sim 10^{6}$. The
feedback term dominates as soon as $\delta r$ is non-negligible, and $\delta r$
per step scales as $\Delta t^2 \cdot \text{drift}$ — which is $5\times10^{-6}$ for
the rebalance settings versus $10^{-8}$ for the stable settings, a factor of 500.
That is why the stable case stays in the clean linear regime for all 5 steps and
the rebalance case leaves it after one.

Comparing columns confirms the scaling plus the excess:

| quantity | predicted ratio (linear law) | observed ratio (B/A) |
| --- | --- | --- |
| `vel_max_rel` | 10 ($\Delta t$) | 93 |
| `pos_max_rel` | 500 ($\Delta t^2 \cdot$ drift) | 1.8e5 |

The leading-order scaling accounts for the first factor of 10 / 500; the residual
9× / 350× is the chaotic feedback above.

### Result 5 — the effect is problem-size sensitive, not rank sensitive

The diagnostic also ran at np=1. Because `npp` is *per rank*, np=1 is a
200-particle problem and np=4 is an 800-particle problem — these are different
physics problems, not a strong-scaling comparison. The 200-particle case has a
shallower tree (70 cells vs 176), a 10× smaller baseline force error (3.2e-07 vs
3.2e-06), and correspondingly smaller trajectory errors (2.9e-06 vs 1.7e-03 in
the `drift=50` column). The mode-independence conclusion holds identically at
both sizes.

This is worth knowing when interpreting `ctest -L regression -R MPI_SERIAL`
across ranks 1–6: each rank count is solving a differently-sized problem, so a
single `fmm_tol` has to cover all of them.

## Conclusions

1. **The FMM solve is not less accurate under rebalance or rebuild.** Instantaneous
   FMM force error is ~3e-6 relative at P=8 regardless of maintenance mode, and
   the three modes produce identical trajectories at identical integrator settings.
2. **The looser tolerances on the rebalance/rebuild tests are compensating for
   integrator settings, not for maintenance-path error.** `fmm_tol` of 1e-2 /
   2e-2 / 3e-2 tracks `dt` and `drift_multiplier`, not `Mode`.
3. **The trajectory metric is a poor accuracy oracle at these settings.** By step 2
   of the `drift=5` case the FMM is exact to 1e-15 and the reported error is pure
   accumulated divergence. A regression that degraded FMM accuracy by 2× would
   barely move `pos_max_rel`, while an unrelated change in particle ordering
   could move it a lot.
4. **The `drift=5` and `drift=50` cases degenerate the tree.** With `softening=0.0`
   a single close encounter ejects a particle at |v| ≈ 3e4, the bounding box
   grows without bound (0.94 → 621 over three steps), the cell count collapses
   from 176 to 21, and nearly all interactions become P2P. These tests exercise
   very little of the far-field (M2L/L2L) pipeline in their later steps — the
   opposite of what their names suggest. This is a test-configuration defect
   independent of the tolerance question.

## Suggested follow-ups

Not implemented — flagging for a decision.

- **Assert on force error, not trajectory error.** Add a per-step check of
  `force_max_rel` against the brute-force reference at the FMM's own positions
  (the `run_diag` harness already computes this). That is a direct, tight,
  step-count-independent measure of solve accuracy and would support a real
  tolerance like 1e-5 for every case. Keep the trajectory check as a loose
  sanity bound.
- **Set `cfg.softening` to a small non-zero value** in the multi-step tests
  (e.g. 1e-3 of the box width). This removes the singular close-pair forces that
  drive both the runaway velocities and the bounding-box explosion, and would let
  the rebalance/rebuild tests actually run with a stable tree — which is what they
  claim to be testing.
- **Reduce `drift_multiplier`** for the rebalance/rebuild cases and instead force
  topology change through the tree parameters (`ncrit`, `tree_tol`). Topology
  churn is the thing under test; runaway particle motion is a side effect that
  destroys the tree rather than exercising the maintenance path.
- **Decide the tolerances deliberately.** If the trajectory assertion stays, the
  current values are defensible but should carry a comment saying they scale with
  `dt²·drift` and are not statements about FMM accuracy.

## Reproducing

The diagnostic lives in `tests/tstRebalanceDiag.hpp`. **It is an investigation
aid, not a test — it has no assertions**, so it is deliberately *not* registered
in `tests/CMakeLists.txt`; the test suite is unchanged by this investigation.
Either delete the header, or rework it into the force-error assertion suggested
above.

To build and run it, temporarily add `RebalanceDiag` to `UNIT_MPI_TESTS` in
`tests/CMakeLists.txt`, then:

```bash
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos
cd build-tuolumne && cmake . && make -j Canopy_Test_RebalanceDiag_MPI_SERIAL

export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
flux run --ntasks=4 --nodes=1 --exclusive --cores-per-task=1 \
  tests/Canopy_Test_RebalanceDiag_MPI_SERIAL
```

Raw logs from this investigation:

- `build-tuolumne/multisolve4.out` — the `fmm_tol=1e-8` run of the real suite (np=4).
- `rebal_diag.f3Uwgc6SaXHH.log` — the full 3×3 matrix (mode × integrator settings)
  at np=4 and np=1.
- `rebal_diag2.f3Uwir8mAnBZ.log` — the same at np=4 with bounding-box extent and
  max velocity instrumented.
