# Task: near-field P2P cost blowup from the unsoftened far field

Status: **Phase 1 in progress** (diagnostic). Multi-phase.

This log is the durable reference for this problem — read it at the start of any
session that touches softening, the `near_softening_factor`, the FMM far field, or
near-field/P2P cost. It records *why* we are doing this and *how* we attacked it,
not just what changed.

## Problem

The FMM far field is a solid-harmonic expansion of the **unsoftened** `1/r`
Laplace kernel, which is only accurate where the Plummer softening is negligible
(separation `R ≫ eps`). The P2P near field uses the softened kernel
`1/sqrt(r²+eps²)`. To keep the two consistent, `near_softening_factor` (`k`,
default 4) forces any cell pair whose **center separation** `R ≤ k·eps` onto the
softened P2P path — see the floor in `mac_satisfied`
(`src/Canopy_CommunicationPlan.hpp`, the `R2 <= floor*floor` test).

That floor is a **fixed physical radius**. The primary downstream use is a fluid
interface / vortex-sheet solver (beatnik/rocketrig) where points roll up into a
plume and cluster ever more tightly. As local density `ρ` rises, the number of
P2P partners per point grows like

    ρ · (k·eps)³

i.e. a near-O(N²) blowup in the clustered region. An adaptive tree cannot absorb
it, because the floor is on physical distance, not tree level — deeper subdivision
just packs more, smaller leaves inside the same fixed `k·eps` ball.

### Why not just shrink things (rejected as the primary fix)

- Smaller `eps`: it is a physics knob (sets the interface desingularization /
  KH regularization). Shrinking it to dodge FMM cost re-sharpens the very roll-up
  being regularized.
- Smaller `k`: README tradeoff, far-field error `~1/(2k²)`. Cheap stopgap, but a
  harmonic `1/r` far field has a **hard floor** at `k ≳ 1` (the `(eps/R)²` series
  diverges for `R < eps`), so it can never reach `k → 0`.

## Multi-phase plan

- **Phase 1 — diagnostic (this phase).** Measure how fast near-field cost actually
  grows as clustering develops, in a cheap toy problem (gravitational cold
  collapse) rather than a full beatnik run. Deliverable: P2P particle-pair curves
  vs. clustering, swept over `k ∈ {0,2,4}`.
- **Phase 2 — B1: far-field softening correction terms.** Fold the leading
  Plummer corrections (`1/√(R²+eps²) = (1/R)(1 − ½(eps/R)² + …)`) into M2L/L2P so
  the achievable `k` floor drops (~4 → ~1.3). Keeps the solid-harmonic machinery.
  Re-run the Phase-1 sweep to confirm.
- **Phase 3 — B2: kernel-independent far field.** Replace the solid-harmonic
  operators with a black-box/Chebyshev (bbFMM) or equivalent-density (KIFMM) far
  field that represents the (smoother) softened kernel directly, then set `k → 0`
  so the near field reverts to the adaptive-tree-governed geometric MAC and
  decouples from clustering. Re-run the sweep to confirm.

If the cold-collapse surrogate proves insufficient for sheet-specific behavior,
the real solver (beatnik/rocketrig in `~/spack_envs/tuolumne_beatnik/beatnik`) is
the fallback — deferred to keep diagnosis in a toy problem.

## Approach / decisions (Phase 1)

- **Cold-collapse surrogate.** The metric (P2P particle-pairs vs. local density)
  depends only on how tightly points pile up, not on cluster geometry. A cold ball
  collapsing under the library's existing `1/r` gradient kernel grows density
  monotonically into a softened core — same `ρ·(k·eps)³` law as a rolled-up sheet,
  with no kernel hacks. Absolute numbers differ from a real sheet; the scaling law
  does not.
- **Ground-truth instrumentation.** The dual-tree traversal in
  `build_all_interaction_lists` already classifies every cell pair and `CellInfo`
  carries `global_count`; add a `NearFieldStats` struct populated there. Headline
  metric: `n_p2p_particle_pairs = Σ n_T·n_S` over P2P pairs. Also
  `n_softening_blocked_pairs` (geometric MAC passed but the floor overrode it) to
  isolate the floor's effect within one run. `mac_satisfied` is split into
  `geometric_mac_satisfied` + `softening_floor_satisfied` with no behavior change.
- **Experiment = `-k` sweep.** Run the example at `k = 0, 2, 4` with identical
  IC/eps; the gap between the `n_p2p_particle_pairs` curves is the
  softening-attributable near-field cost vs. clustering.

## Progress log

- 2026-06-24 — Phase 1 started. Prior design discussion captured above. Plan
  approved (see `plans/twinkling-exploring-wave.md`). Beginning implementation:
  task log + CLAUDE.md pointer, then `NearFieldStats` instrumentation, the
  `examples/05_rollup_nearfield` cold-collapse driver, and the `-k` sweep script.

- 2026-06-24 — Phase 1 implemented and first measurement obtained. Files:
  `NearFieldStats` + split `mac_satisfied` in `src/Canopy_CommunicationPlan.hpp`,
  `Solver::near_field_stats()` passthrough, `examples/05_rollup_nearfield`,
  `scripts/tuolumne/rollup_sweep.flux`, README diagnostic subsection.

  **Result (the blowup is confirmed).** Sweep on tuolumne, 1 rank, GPU/HIP,
  `-p 20000 -t 50 -s 0.004 -g 50 -e 0.02`, cold collapse (free-fall time
  t_ff ≈ 0.157). Near-field cost is flat and `k`-independent while the system is
  diffuse (`eps_over_min_hw < 1`, the floor never binds), then spikes hard at
  deepest collapse when cells shrink below `eps`:

  | step / t | eps/hw | k=0 P2P pairs | k=2 P2P | k=4 P2P | k=4 blocked |
  | --- | --- | --- | --- | --- | --- |
  | 24 / 0.096 | 0.71 | 23.5M | 23.5M | 23.6M | 0 |
  | 34 / 0.136 | 2.39 | 16.4M | 16.6M | 16.4M | 183 |
  | **39 / 0.156** | **6.30** | **23.1M** | **37.8M** | **112.5M** | **1.57M** |
  | 44 / 0.176 | 1.89–2.30 | 18.6M | 19.9M | 19.9M | 8188 |

  At peak clustering (step 39): k=4 does **4.9× more near-field particle-pairs**
  than k=0 (112.5M vs 23.1M) while its far field collapses (M2L 422k→96k as the
  floor demotes M2L→P2P). `n_softening_blocked_pairs` scales with k
  (0 → 0.27M → 1.57M for k=0/2/4) and is 0 whenever `eps_over_min_hw < 1`.
  This is the predicted `density·(k·eps)³` mechanism, switching on exactly when
  the cell size drops below the softening length.

  Caveat: cold collapse *bounces*, so the spike here is transient (the ball
  re-expands by step 44+). A sustained roll-up holds the dense state, so the real
  downstream blowup is sustained, not a transient spike — the per-step curve still
  captures the cost law correctly.

  Takeaway for the fix decision: the floor cost is steep and `~k`-superlinear
  once `eps_over_min_hw ≳ 1`. Phase 2 (B1 correction terms) should target dropping
  the binding `k` from 4 toward ~1.3, which on this data would cut the peak P2P
  work by the ratio of the k=4 to k≈1 curves. Re-run this exact sweep after B1.

  Regression gate (no-behavior-change check for the `mac_satisfied` split):
  `ctest -L regression -R MPI_SERIAL` (MultiSolve, np 1–6). All double-precision
  subtests pass at every rank count; the only failure is the pre-existing known
  FP32 budget miss at ≥2 ranks (`tstMultiSolve.hpp:1104`, README Known Issues),
  unrelated to this change. The predicate split is behavior-neutral as designed.
