# Scaling phase-breakdown figures (Tuolumne, 2026-10-15 runs)

Strong scaling (500k particles total) and weak scaling (500k particles per rank)
at 1/16/32/64 ranks (one MI300A APU per rank), for three far-field bases. Each
figure covers one solve state: cold, warm A or warm B. The runs, the driver and
the parsed CSVs are described in `CLAUDE.md`.

## Layout

```
figures/
  plot_scaling.py           grouped-bar plotter (one figure per scaling x state)
  make_figures.sh           regenerates everything below from the lustre CSVs
  check_figures.py          "other" share, rep spread, Warm B operator-build cost
  check_figures.txt         its output for the figures in this directory
  phases_extended.json      default phases + downward host/comm phases + upward exchange
  phases_ilist_split.json   extended, with the ilist build split into S3 / S4 / rest
  <basis>/{default,extended,ilist_split}/<basis>_{strong,weak}_{cold,warmA,warmB}.{pdf,png,csv}
```

`<basis>` is `laplace` (P = 6), `cartesian_taylor` (p = 2), or
`cartesian_taylor_quantized` (p = 2, `quantize_root_half_width` on). Input data
is `/p/lustre5/stewartj/Canopy/scaling{,_ct,_ctq}_results/scaling_{timers,ranks}.csv`.
`./make_figures.sh` (about 1 min) rebuilds every figure, and
`python3 check_figures.py` reruns the checks.

Bar = median over the 3 measured reps of the across-rank mean. Error bar =
median over those reps of the across-rank min and max. Each figure's `.csv`
holds the plotted numbers and an `other (total solve - plotted phases)` row.

## Which phase set to use

- **default** (`PHASES` in `plot_scaling.py`, the original chart's phases plus
  comm). It leaves much of the solve unplotted at scale: up to 63 % of Laplace
  weak np64 warm A, and up to 52 % of Cartesian-Taylor warm A.
- **extended** is the one to use. Unplotted time is within ±7 % everywhere
  except two small cases: CT strong np64 warm A (+9 %, 99 ms call) and CTQ
  strong np64 warm A (−11 %, 48 ms call). A negative "other" comes from taking
  medians per phase instead of the median of their sum. The set adds four bars:
  - Shared-locals snapshot = `dn_pre_m2l`.
  - L2L comm = `l2l_comm`.
  - Shared-locals Allreduce host copies = `dn_post_m2l` − (`m2l_allreduce` +
    `l2l_kernel` + `l2l_comm`).
  - **Upward multipole exchange (untimed)** = `upward_total` − (`p2m` + `m2m` +
    `m2m_allreduce`). This is `exchange_multipoles_at_depth`'s shared-key host
    loops, its Allreduce pack/unpack, and the coalesced point-to-point M2M
    exchange (`src/Canopy_UpwardSweep.hpp:548-689`). None of it has its own
    timer. It was the missing 16–23 % of CT warm A.
- **ilist_split** replaces "Interaction-list build" with classify pairs (S3),
  M2L operator build (S4) and the rest of `dn_build_ilist`. Use it for the
  operator-cache comparison between the two CT series.

## Run checks (all passed)

- All 21 runs: 52/52 calls, `done`, every plotted point `n_reps` = 3, 0
  fallback pairs, no `max_depth` warning, every measured maintain step
  `rebalance`. Multi-node runs segfault at exit after `done` (README Known
  Issues). Their core files are in each run directory.
- Laplace: warm B built 0 new operators (one key at strong np64 tsolve).
- CT: every warm B and tsolve rebuilt the whole table (keys built = cache size,
  cache 0 after each maintain). Largest table 8788 keys (weak np64), under the
  32768 cap. Setup cells within 0.1 % of Laplace (158,248 vs 158,158; 10,447,577
  vs 10,444,614 at 32M).
- CTQ: cache kept at every maintain step. Warm B built 0–17 keys, except weak
  np16, which built 1444 new keys (cache 7743 → 8922) at step 5 on the worst
  rank in all 4 reps. That is a partial top-up, not an octave crossing. The tree
  differs from CT: 169,773 cells at depth 7 vs 158,248 at depth 6 for 500k
  particles; weak 1.78M / 2.21M / 11.17M cells vs CT 1.44M / 2.99M / 10.45M,
  all at depth 9. The 32M tree is depth 9, not the 10 that was predicted.

## Findings worth a caption

Median whole-call wall (`drv_call_wall`) in ms:

| | strong 1 / 16 / 32 / 64 | weak 16 / 32 / 64 |
|---|---|---|
| Laplace cold | 4596 / 594 / 426 / 371 | 4908 / 4143 / 15891 |
| CT cold | 3521 / 360 / 275 / 205 | 3085 / 2536 / 8538 |
| CTQ cold | 3918 / 357 / 266 / 190 | 3468 / 1995 / 9252 |
| Laplace warm B | 4427 / 461 / 292 / 232 | 4004 / 4456 / 14484 |
| CT warm B | 3451 / 328 / 235 / 174 | 2799 / 2673 / 5366 |
| CTQ warm B | 3937 / 391 / 275 / 243 | 3570 / 2190 / 7621 |

- **The interaction-list build dominates every cold and warm B solve**, and
  within it the host-side pair classification (S3) dominates. S3 is
  basis-independent at a fixed tree: about 2.3–2.6 s at 1 rank (2.9 s for the deeper CTQ tree) and about 115 ms at strong
  np64.
- **The CT operator-cache fix buys little.** CT's full operator rebuild (S4)
  costs only 9–16 ms per warm B, and CTQ cuts that to about 0. But CTQ's
  different tree (one level deeper at 500k) costs more in S3 and
  communication. Its warm B is slower than CT's at every point except weak
  np32, where the quantized tree has 26 % fewer cells.
- **Weak scaling is dominated by replicated/shared-cell work at np64**: the
  M2L shared-locals Allreduce, its host copies, the shared-locals snapshot,
  L2L comm, and (for CT) the upward multipole exchange. The min–max spreads
  are wide, up to several seconds.
- **Rep-to-rep noise:** comm phases (Allreduces, ghost exchange, `l2l_comm`,
  `dn_post_m2l`) often vary by more than 15 % across the 3 reps, and whole
  calls by up to 48 % at CTQ weak np64 warm B (`check_figures.txt`). Treat
  differences of tens of percent in comm-dominated points as noise. For
  example, CT weak np64 warm B (5.4 s) being faster than its cold solve
  (8.5 s) is mostly that comm variance.
- Caption caveats (`CLAUDE.md`): auto softening
  (0.1 × mean spacing), charges scaled by 600/N, max_depth 12, and
  `auto_maintain`. CT/CTQ use `near_softening_factor = 0`.
