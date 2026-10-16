# Scaling phase-breakdown figures: instructions

Working notes for regenerating, checking or extending the Tuolumne strong/weak
scaling figures in this directory. `README.md` here has the layout, the check
results and the findings for the 2026-10-15 runs; read it first.

## What is measured

Driver: `examples/05_scaling_breakdown/scaling_breakdown.cpp`. It is built only
when `Canopy_ENABLE_PROFILING=ON` and is not a CTest test. One source gives two
binaries: `scaling_breakdown` (`LaplaceKernel`, P = 6) and
`scaling_breakdown_ct` (`-DCANOPY_SCALING_CT`: `CartesianTaylorBasis`, p = 2,
`near_softening_factor = 0`). Usage is in the top-level `README.md`.

Problem: the `tstLaplaceSolve` setup (θ 0.5, ncrit 16, replication_depth 2,
positions U[0.05, 0.95]³ and charges U[0.5, 1.5] from `mt19937(1234 + P)`,
sliced contiguously by rank). Symplectic Euler with dt = 1e-4, 5 steps of
integrate + `auto_maintain()` + solve. Deviations from the fixture, which
belong in any caption:

- charges × 600/N, so the total charge matches the fixture;
- `max_depth` 12;
- auto softening (`-e -1`, 0.1 × mean spacing). Without it, close pairs eject
  particles at large N and Warm B measures a degenerate tree;
- `auto_maintain()` instead of `migrate()`.

Strong = 500k total; weak = 500k per rank, in a fixed box (density grows). Rank
counts are 1/16/32/64, one rank per MI300A, 4 per node. The np = 1 run is
`scaling=shared`, and the parser emits it under both `strong` and `weak`.

States per rep:
- `cold`: first solve after setup.
- `warmA`: an immediate second solve.
- `warmB`: the solve after step 5.

Also recorded: `setup`, `maintain` (steps 1–5) and `tsolve` (steps 1–4).
`-w 1 -r 3` gives 4 reps; rep 0 is a warm-up and is excluded from the figures.
Before each call the driver fences, barriers and calls `reset_timer_totals()`,
then dumps every key (min/max/mean across ranks, plus per-rank values) and
`drv_call_wall`. Library tables print only in rep 0.

Three series:

| series | `basis` tag | binary / flag | run dirs | job list |
|---|---|---|---|---|
| Laplace | `laplace` | `scaling_build/build/…/scaling_breakdown` | `scaling_*` | `scaling_results_jobids.txt` |
| Cartesian-Taylor | `cartesian_taylor` | `scaling_build_ct/build/…/scaling_breakdown_ct` | `scaling_ct_*` | `scaling_ct_results_jobids.txt` |
| CT, quantized root | `cartesian_taylor_quantized` | CT binary with `-q` | `scaling_ctq_*` | `scaling_ctq_results_jobids.txt` |

All paths are under `/p/lustre5/stewartj/Canopy/`. Raw output is
`<run>/driver.<jobid>.log` plus `provenance.<jobid>.txt` (HEAD, git status,
binary mtime/md5).

## Build and jobs (Tuolumne)

- **Never rebuild a driver while a job using it is queued or running.** A job
  runs the binary at its path when it starts, so relinking under it can crash
  it. Check `flux jobs` for `canopy-scaling_*` / `canopy-scaling_ct*_*`.
- Each tree builds **only its own target**. Never run a bare `make`. The
  Laplace tree predates the CT target and does not know it.

  ```bash
  source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
  spack env activate ${HOME}/spack_envs/tuolumne_trilinos
  cd /p/lustre5/stewartj/Canopy/scaling_build/build    && make -j 16 scaling_breakdown    > ../build.log 2>&1
  cd /p/lustre5/stewartj/Canopy/scaling_build_ct/build && make -j 16 scaling_breakdown_ct > ../build.log 2>&1
  ```
  Configure with the `run_cmake_tuolumne.sh` flags plus
  `-DCanopy_ENABLE_PROFILING=ON -DCanopy_PROFILING_LEVEL=2`. Level 3 adds a
  rerun kernel that perturbs timings. The 2026-10-15 Laplace binary predates
  `-q`; a Laplace `-q` series needs a rebuild.
- Job scripts: `scripts/tuolumne/scaling/make_jobs.sh <run> <label> <np> <N>
  <queue> <limit> [driver args]`. Prefix `BASIS=ct` for CT, and append `-q`
  for the quantized series. The pdebug limit is 60m; use pbatch beyond that.
  Submit with `flux batch`, and add the new ID to the series' job list. The
  parser picks the newest log in a run dir that reached `done`, so old
  attempts need no deleting. Keep all points of a series on one binary.

## Is a run good?

```bash
D=/p/lustre5/stewartj/Canopy
for id in $(awk '{print $2}' $D/scaling_ct_results_jobids.txt); do
  flux jobs -a -n -o "{id.f58} {name} {status} {returncode} {runtime!F}" $id; done
```

A complete run has **52** `### CANOPY_SCALING end` lines and exactly one
`### CANOPY_SCALING done`. Also check:

- no `[Canopy] WARNING: TreeBuilder::build: ... max_depth` line;
- `fallback_pairs_max=0`, and steady `cells` across steps;
- one distinct GPU per task in the `mpibind` lines.

Multi-node runs **segfault at exit after `done`** (top-level README Known
Issues), so flux says FAILED with rc 139 even though the data is complete.
A real failure has fewer than 52 end lines and no `done`. Known log noise:
`CRAYBLAS_WARNING`, `OMPT initialization failed`, `libz.so.1: no version
information`.

Expected operator-cache evidence on the end lines:

| state | Laplace | CT | CT quantized |
|---|---|---|---|
| cold | keys built = cache | same | same |
| warmA | 0 | 0 | 0 |
| maintain | cache kept | cache **0** (root width changed) | cache kept |
| warmB / tsolve | ~0 new keys | **full rebuild** (keys = cache) | few new keys; a full rebuild means an octave crossing, say so |

For CT, keep `op_cache_size_max` under the 32768 count cap, or the M2L bar
includes per-pair fallback. The quantized series changes the tree as well as
the cache (deeper, with a different cell count). Report its setup cells/depth,
and compare cold-vs-cold across series for the tree effect and Warm B − cold
within a series for the cache effect.

## Parse and plot

```bash
cd <repo>; D=/p/lustre5/stewartj/Canopy
python3 scripts/tuolumne/scaling/parse_scaling.py --ranks \
    --out-dir $D/scaling_ct_results --repo-copy scripts/tuolumne/scaling/results_ct \
    $D/scaling_ct_{shared_np01,strong_np16,strong_np32,strong_np64,weak_np16,weak_np32,weak_np64}
figures/make_figures.sh          # all bases x phase sets, into figures/<basis>/<set>/
python3 figures/check_figures.py > figures/check_figures.txt
```

(Use `scaling_results` / `results` for Laplace and `scaling_ctq_results` /
`results_ctq` for the quantized series.) Figures and plotting code live in
`figures/`, **not on lustre**. The parsed CSVs stay on lustre:
`scaling_ranks.csv` is large, and the repo copies hold only
`scaling_{timers,calls}.csv`. PNGs are gitignored; the PDFs and CSVs are
committed.

## Phase sets and nesting

- Phases are `[label, [keys summed], [keys subtracted]?]`. Subtraction and
  exact merged error bars need `--ranks`.
- **Use `phases_extended.json`.** The default set leaves up to 63 % of a warm
  solve unplotted at weak np64.
- Check every new bar against the nesting, or time gets counted twice:
  - `dn_post_m2l` contains `m2l_allreduce`, `l2l_kernel` and `l2l_comm`;
  - `dn_m2l_call` contains `m2l_kernel`;
  - `dn_build_ilist` contains `ilist_s*` (do not plot it alongside its S3/S4
    split);
  - `upward_total` contains `p2m`, `m2m`, `m2m_allreduce` and the untimed
    multipole exchange.

  Confirm against `DownwardSweep::execute()` and `UpwardSweep::execute()` in
  `src/`.
- Total solve = `upward_total + downward_total + p2p_total`. Each figure CSV's
  `other` row is that minus the plotted bars. A small negative value comes from
  per-phase medians and is not an error. A large positive one means a phase is
  missing.
- "P2P" = `p2p_intra_kernel + p2p_inter_kernel`; the ghost exchange is its own
  bar. The original 600-particle chart's definition was not found.
- Comm phases vary more than 15 % between reps at many points. Do not read
  tens-of-percent differences at comm-dominated points as real.
- Timer fencing: every solve key is fenced or ends on a synchronizing call.
  The exception is `sort_by_leaf` (`TreePartitioner.hpp`, unfenced), which
  only affects setup/maintenance keys.
