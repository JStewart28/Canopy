#!/usr/bin/env python3
"""Sanity numbers for the scaling figures (see CLAUDE.md here).

1. "other" share: unplotted solve time / total solve, from each figure CSV.
2. Rep spread: (max - min) / median over reps 1-3 of each phase key's mean_s,
   reported where it exceeds --spread and the median is above 1 ms.
3. Operator-cache cost: Warm B ilist_s4_op_table_build and dn_build_ilist
   (median over reps of the across-rank mean), for every series.
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/p/lustre5/stewartj/Canopy"
RES = {"laplace": f"{D}/scaling_results", "cartesian_taylor": f"{D}/scaling_ct_results",
       "cartesian_taylor_quantized": f"{D}/scaling_ctq_results"}
SPREAD_KEYS = ["dn_build_ilist", "m2l_kernel", "l2l_kernel", "l2p", "p2m", "m2m",
               "p2p_intra_kernel", "p2p_inter_kernel", "m2l_comm", "m2m_allreduce",
               "m2l_allreduce", "p2p_ghost_comm", "dn_pre_m2l", "l2l_comm",
               "dn_post_m2l", "drv_call_wall"]


def other_share(phase_set):
    print(f"\n## 'other' / total solve, phase set '{phase_set}'")
    for basis in RES:
        for path in sorted(glob.glob(f"{HERE}/{basis}/{phase_set}/*.csv")):
            bars = defaultdict(float)
            oth = {}
            for r in csv.DictReader(open(path)):
                if r["phase"].startswith("other"):
                    oth[int(r["np"])] = float(r[[k for k in r if k.startswith("bar_")][0]])
                else:
                    bars[int(r["np"])] += float(r[[k for k in r if k.startswith("bar_")][0]])
            share = {n: 100 * oth[n] / (bars[n] + oth[n]) for n in oth}
            print(f"{os.path.basename(path)[:-4]:45s} " +
                  " ".join(f"np{n}={s:5.1f}%" for n, s in sorted(share.items())))


def load(basis):
    d = defaultdict(list)  # (scaling, np, state, key) -> [mean_s per rep]
    for r in csv.DictReader(open(f"{RES[basis]}/scaling_timers.csv")):
        if int(r["rep"]) >= 1 and r["state"] in ("cold", "warmA", "warmB"):
            d[(r["scaling"], int(r["np"]), r["state"], r["timer_name"])].append(
                float(r["mean_s"]))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spread", type=float, default=15.0, help="percent threshold")
    a = ap.parse_args()
    for s in ("default", "extended", "ilist_split"):
        other_share(s)
    data = {b: load(b) for b in RES}
    print(f"\n## rep spread above {a.spread:g}% (phases > 1 ms)")
    for b, d in data.items():
        for (sc, n, st, k), v in sorted(d.items()):
            if k not in SPREAD_KEYS or sc == "weak" and n == 1:
                continue
            med = statistics.median(v)
            if med > 1e-3 and 100 * (max(v) - min(v)) / med > a.spread:
                print(f"{b:27s} {sc:6s} np{n:<3d} {st:6s} {k:18s} "
                      f"{100 * (max(v) - min(v)) / med:5.0f}%  median {med * 1e3:9.2f} ms")
    print("\n## Warm B operator build / ilist build / whole call (ms)")
    for sc in ("strong", "weak"):
        for n in (1, 16, 32, 64):
            row = []
            for b, d in data.items():
                f = lambda k: statistics.median(d.get((sc, n, "warmB", k), [0])) * 1e3
                row.append(f"{b[:5] + ('q' if 'quant' in b else ' ')}: "
                           f"{f('ilist_s4_op_table_build'):7.1f} / "
                           f"{f('dn_build_ilist'):7.1f} / {f('drv_call_wall'):7.1f}")
            print(f"{sc:6s} np{n:<3d} " + " | ".join(row))


if __name__ == "__main__":
    main()
