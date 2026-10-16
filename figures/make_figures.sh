#!/bin/bash
# Regenerate every scaling phase-breakdown figure from the parsed CSVs on
# lustre (parse_scaling.py output, see CLAUDE.md here).
# Output: figures/<basis>/<phase set>/<basis>_<scaling>_<state>.{pdf,png,csv}
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=${D:-/p/lustre5/stewartj/Canopy}

# basis tag -> parsed results directory
declare -A RES=(
  [laplace]=$D/scaling_results
  [cartesian_taylor]=$D/scaling_ct_results
  [cartesian_taylor_quantized]=$D/scaling_ctq_results
)
# phase set -> --phases argument ("" = PHASES in plot_scaling.py)
declare -A SETS=(
  [default]=""
  [extended]="--phases $HERE/phases_extended.json"
  [ilist_split]="--phases $HERE/phases_ilist_split.json"
)

for basis in laplace cartesian_taylor cartesian_taylor_quantized; do
  for set in default extended ilist_split; do
    python3 "$HERE/plot_scaling.py" --basis "$basis" ${SETS[$set]} \
        --timers "${RES[$basis]}/scaling_timers.csv" \
        --ranks  "${RES[$basis]}/scaling_ranks.csv" \
        --out-dir "$HERE/$basis/$set"
  done
done
