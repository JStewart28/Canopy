#!/usr/bin/env python3
"""Grouped-bar phase breakdowns from parse_scaling.py output.

One figure per (scaling, state): x = phases, one bar per rank count, bar =
median over measured reps (rep >= 1) of the across-rank mean (or max, with
--stat max), error bars = median over reps of the across-rank min and max.
Times in ms. Writes <out>/<basis>_<scaling>_<state>.{pdf,png,csv}. The input CSVs may
hold several bases (parse_scaling.py's basis column); pick one with --basis.

Phases come from PHASES below or a --phases JSON file of the same shape:
  [["Display label", ["timer_key", ...]], ...]
A phase with several keys is their sum. An optional third element lists keys
to subtract, for the untimed remainder of an enclosing timer:
  ["Allreduce host copies", ["dn_post_m2l"],
   ["m2l_allreduce", "l2l_kernel", "l2l_comm"]] With --ranks (scaling_ranks.csv) the
sum is taken per rank before min/max/mean, so merged error bars are exact;
without it the min/max of a merged phase are the sums of the per-key min/max
(an outer bound) and the CSV says so.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The 7 phases of the original "where a cold solve spends its time" chart,
# then the communication phases. Keys are Canopy_Profiling.hpp TIMER_* names.
PHASES = [
    ["Interaction-list build", ["dn_build_ilist"]],
    ["L2P", ["l2p"]],
    ["M2L kernel", ["m2l_kernel"]],
    ["L2L kernel", ["l2l_kernel"]],
    ["P2M", ["p2m"]],
    ["M2M", ["m2m"]],
    ["P2P", ["p2p_intra_kernel", "p2p_inter_kernel"]],
    ["M2L comm", ["m2l_comm"]],
    ["M2M Allreduce", ["m2m_allreduce"]],
    ["M2L shared-locals Allreduce", ["m2l_allreduce"]],
    ["P2P ghost comm", ["p2p_ghost_comm"]],
]
# Solve total for the "other" check row in each figure's CSV.
SOLVE_TOTAL_KEYS = ["upward_total", "downward_total", "p2p_total"]

RANKS = [1, 16, 32, 64]
# Ordinal one-hue ramp (light -> dark = fewer -> more ranks), by position.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STATES = {"cold": "cold solve", "warmA": "warm solve A (back-to-back)",
          "warmB": "warm solve B (after time steps)"}


BASIS_LABELS = {"laplace": "Laplace P=6", "cartesian_taylor": "Cartesian-Taylor p=2",
                "laplace_quantized": "Laplace P=6, quantized root",
                "cartesian_taylor_quantized": "Cartesian-Taylor p=2, quantized root"}
# What happens to the M2L operator cache across maintenance, per basis.
WARMB_NOTE = {"laplace": "operator cache kept",
              "cartesian_taylor": "operator cache rebuilt",
              "cartesian_taylor_quantized": "operator cache kept"}


def bases_in(path):
    with open(path) as f:
        return sorted({r.get("basis") or "laplace" for r in csv.DictReader(f)})


def state_label(state, basis):
    if state == "warmB" and basis in WARMB_NOTE:
        return f"warm solve B (after time steps, {WARMB_NOTE[basis]})"
    return STATES[state]


def load_timers(path, basis):
    # (scaling, np, state, rep) -> key -> (min, max, mean)
    d = defaultdict(dict)
    n_of = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("basis") or "laplace") != basis:
                continue
            if r["state"] not in STATES or int(r["rep"]) < 1:
                continue
            k = (r["scaling"], int(r["np"]), r["state"], int(r["rep"]))
            d[k][r["timer_name"]] = (float(r["min_s"]), float(r["max_s"]),
                                     float(r["mean_s"]))
            n_of[(r["scaling"], int(r["np"]))] = int(r["N"])
    return d, n_of


def load_ranks(path, basis):
    # (scaling, np, state, rep) -> key -> [value per rank]
    d = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("basis") or "laplace") != basis:
                continue
            if r["state"] not in STATES or int(r["rep"]) < 1:
                continue
            k = (r["scaling"], int(r["np"]), r["state"], int(r["rep"]))
            d[k][r["timer_name"]][int(r["rank"])] = float(r["value_s"])
    return d


def phase_stats(keys, timers, ranks, minus=()):
    """(min, max, mean, exact) of the summed keys, less `minus`, for one call."""
    if minus and ranks is None:
        raise SystemExit("a phase that subtracts keys needs --ranks")
    if (len(keys) == 1 and not minus) or ranks is None:
        vals = [timers.get(k, (0.0, 0.0, 0.0)) for k in keys]
        return (sum(v[0] for v in vals), sum(v[1] for v in vals),
                sum(v[2] for v in vals), len(keys) == 1)
    nr = max(len(ranks.get(k, {})) for k in list(keys) + list(minus))
    per = [sum(ranks.get(k, {}).get(i, 0.0) for k in keys) -
           sum(ranks.get(k, {}).get(i, 0.0) for k in minus) for i in range(nr)]
    return min(per), max(per), sum(per) / len(per), True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = "/p/lustre5/stewartj/Canopy/scaling_results"
    ap.add_argument("--timers", default=os.path.join(res, "scaling_timers.csv"))
    ap.add_argument("--ranks", default=None,
                    help="scaling_ranks.csv, for exact merged-phase error bars")
    ap.add_argument("--phases", default=None, help="JSON phase list")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--stat", choices=["mean", "max"], default="mean",
                    help="bar height: across-rank mean (default) or max")
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--np", default=",".join(map(str, RANKS)),
                    help="rank counts, in bar order (at most 4)")
    ap.add_argument("--basis", default=None,
                    help="laplace | cartesian_taylor | cartesian_taylor_quantized; "
                         "required if the CSV holds more than one")
    ap.add_argument("--scalings", default="strong,weak")
    ap.add_argument("--states", default="cold,warmA,warmB")
    a = ap.parse_args()

    ranks_list = [int(x) for x in a.np.split(",")]
    colors = dict(zip(ranks_list, RAMP))
    phases = PHASES
    if a.phases:
        with open(a.phases) as f:
            phases = json.load(f)
    basis = a.basis
    if basis is None:
        found = bases_in(a.timers)
        if len(found) != 1:
            ap.error(f"--basis is required: the CSV holds {found}")
        basis = found[0]
    timers, n_of = load_timers(a.timers, basis)
    ranks = load_ranks(a.ranks, basis) if a.ranks else None
    os.makedirs(a.out_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})

    for scaling in a.scalings.split(","):
        for state in a.states.split(","):
            # np -> phase label -> (height, lo, hi, nreps, exact)
            table = {}
            other = {}
            for np_ in ranks_list:
                reps = sorted(k[3] for k in timers
                              if k[:3] == (scaling, np_, state))
                if not reps:
                    continue
                row = {}
                for label, keys, *minus in phases + [["Total solve", SOLVE_TOTAL_KEYS]]:
                    minus = minus[0] if minus else ()
                    st = [phase_stats(keys, timers[(scaling, np_, state, r)],
                                      ranks[(scaling, np_, state, r)]
                                      if ranks else None, minus) for r in reps]
                    mn = statistics.median(s[0] for s in st)
                    mx = statistics.median(s[1] for s in st)
                    mean = statistics.median(s[2] for s in st)
                    h = mx if a.stat == "max" else mean
                    row[label] = (h * 1e3, mn * 1e3, mx * 1e3, len(reps),
                                  all(s[3] for s in st))
                plotted = sum(row[p[0]][0] for p in phases)
                other[np_] = row.pop("Total solve")[0] - plotted
                table[np_] = row
            if not table:
                print(f"skip {scaling}/{state}: no data")
                continue

            stem = os.path.join(a.out_dir, f"{basis}_{scaling}_{state}")
            with open(stem + ".csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["basis", "scaling", "state", "np", "N", "phase",
                            f"bar_{a.stat}_ms", "min_ms", "max_ms",
                            "n_reps", "minmax_exact"])
                for np_, row in table.items():
                    for label, (h, mn, mx, nr, ex) in row.items():
                        w.writerow([basis, scaling, state, np_,
                                    n_of[(scaling, np_)], label, f"{h:.6g}", f"{mn:.6g}", f"{mx:.6g}",
                                    nr, int(ex)])
                    w.writerow([basis, scaling, state, np_,
                                n_of[(scaling, np_)], "other (total solve - plotted phases)",
                                f"{other[np_]:.6g}", "", "", "", ""])

            labels = [p[0] for p in phases]
            nb = len(ranks_list)
            width = 0.8 / nb
            fig, ax = plt.subplots(figsize=(max(8, 0.95 * len(labels)), 5))
            for j, np_ in enumerate(ranks_list):
                if np_ not in table:
                    continue
                xs = [i + (j - (nb - 1) / 2) * width for i in range(len(labels))]
                h = [table[np_][l][0] for l in labels]
                lo = [max(0.0, table[np_][l][0] - table[np_][l][1]) for l in labels]
                hi = [max(0.0, table[np_][l][2] - table[np_][l][0]) for l in labels]
                ax.bar(xs, h, width, color=colors[np_], edgecolor="white",
                       linewidth=1, label=f"{np_} rank{'s' if np_ > 1 else ''}",
                       yerr=[lo, hi], capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "#3d3d3a"})
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(f"time (ms), across-rank {a.stat}")
            if a.log_y:
                ax.set_yscale("log")
            ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8)
            ax.set_axisbelow(True)
            n0 = min(table)
            nlabel = (f"N = {n_of[(scaling, n0)]:,} total" if scaling == "strong"
                      else f"{n_of[(scaling, n0)] // n0:,} particles per rank")
            ax.set_title(f"{BASIS_LABELS.get(basis, basis)}: "
                         f"{scaling} scaling, {state_label(state, basis)}\n"
                         f"{nlabel}; error bars min-max across ranks",
                         fontsize=12)
            ax.legend(title="MPI ranks (1 per GPU)", frameon=False)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(f"{stem}.{ext}", dpi=300)
            plt.close(fig)
            print(f"wrote {stem}.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
