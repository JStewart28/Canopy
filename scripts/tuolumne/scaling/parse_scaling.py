#!/usr/bin/env python3
"""Parse scaling_breakdown driver logs into tidy CSVs.

Inputs are driver logs (driver.<jobid>.log) or run directories under
/p/lustre5/stewartj/Canopy/. For a directory, the newest log that reached
"### CANOPY_SCALING done" is used; if none did, the newest log is parsed as
partial data and a warning is printed.

Outputs (in --out-dir):
  scaling_timers.csv  scaling,np,N,state,rep,timer_name,min_s,max_s,mean_s,
                      imbalance,step,action,source
                      every timer of every measured call; imbalance is
                      (max - mean) / mean in percent, as the library prints it;
                      then basis: the series, from the log's config line --
                      laplace | cartesian_taylor, with "_quantized" appended
                      when quantize_root=1 (logs from before the basis field
                      are laplace)
  scaling_calls.csv   one row per call: action, ilist builds, op keys built,
                      op cache size, n_local min/max, cells, max cell depth,
                      wall_max_s, complete (end marker seen)
  scaling_ranks.csv   per-rank values (--ranks): scaling,np,N,state,rep,step,
                      timer_name,rank,value_s

A run labelled "shared" (the np=1 point serving both series) is emitted
twice, as scaling=strong and scaling=weak.
"""
import argparse
import csv
import glob
import os
import re
import sys

KV = re.compile(r"(\w+)=(\S+)")
CTX_KEYS = ("scaling", "np", "N", "state", "rep", "step")


def ctx_of(d):
    return tuple(d[k] for k in CTX_KEYS)


def pick_log(path):
    if os.path.isfile(path):
        return path
    logs = sorted(glob.glob(os.path.join(path, "driver.*.log")),
                  key=os.path.getmtime)
    if not logs:
        return None
    for log in reversed(logs):
        with open(log, errors="replace") as f:
            if any(l.startswith("### CANOPY_SCALING done") for l in f):
                return log
    print(f"warning: no finished log in {path}; using partial {logs[-1]}",
          file=sys.stderr)
    return logs[-1]


def parse_log(path, want_ranks):
    timers, calls, ranks = [], {}, []
    actions = {}
    basis = "laplace"
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("### CANOPY_SCALING config"):
                cfg = dict(KV.findall(line))
                basis = cfg.get("basis", "laplace")
                if cfg.get("quantize_root") == "1":
                    basis += "_quantized"
            elif line.startswith("CANOPY_TIMER "):
                d = dict(KV.findall(line))
                timers.append(d)
            elif line.startswith("### CANOPY_SCALING begin"):
                d = dict(KV.findall(line))
                calls.setdefault(ctx_of(d), dict(d, complete=0))
            elif line.startswith("### CANOPY_SCALING end"):
                d = dict(KV.findall(line))
                calls[ctx_of(d)] = dict(d, complete=1)
                actions[ctx_of(d)] = d.get("action", "none")
            elif want_ranks and line.startswith("CANOPY_RANKS "):
                d = dict(KV.findall(line))
                ranks.append(d)
    # A call whose end marker is missing has incomplete timer lines.
    complete = {c for c, d in calls.items() if d["complete"]}
    timers = [t for t in timers if ctx_of(t) in complete]
    ranks = [r for r in ranks if ctx_of(r) in complete]
    for t in timers:
        t["action"] = actions[ctx_of(t)]
    calls = list(calls.values())
    for row in timers + calls + ranks:
        row["basis"] = basis
    return timers, calls, ranks


def expand_shared(rows):
    for r in rows:
        if r["scaling"] == "shared":
            for s in ("strong", "weak"):
                yield dict(r, scaling=s)
        else:
            yield r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="driver logs or run directories")
    ap.add_argument("--out-dir",
                    default="/p/lustre5/stewartj/Canopy/scaling_results")
    ap.add_argument("--ranks", action="store_true",
                    help="also write the per-rank CSV (large)")
    ap.add_argument("--repo-copy", default=None,
                    help="also copy scaling_timers.csv and scaling_calls.csv "
                         "here if scaling_timers.csv is under 5 MB")
    a = ap.parse_args()

    all_t, all_c, all_r = [], [], []
    for inp in a.inputs:
        log = pick_log(inp)
        if log is None:
            print(f"warning: no driver log under {inp}", file=sys.stderr)
            continue
        t, c, r = parse_log(log, a.ranks)
        for row in t:
            row["source"] = log
        for row in c:
            row["source"] = log
        n_done = sum(x["complete"] for x in c)
        print(f"{log}: {n_done}/{len(c)} calls complete, {len(t)} timer rows",
              file=sys.stderr)
        all_t += t
        all_c += c
        all_r += r

    os.makedirs(a.out_dir, exist_ok=True)
    tcols = ["scaling", "np", "N", "state", "rep", "timer_name", "min_s",
             "max_s", "mean_s", "imbalance", "step", "action", "source",
             "basis"]
    tpath = os.path.join(a.out_dir, "scaling_timers.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(tcols)
        for t in expand_shared(all_t):
            mn, mx, mean = (float(t[k]) for k in ("min_s", "max_s", "mean_s"))
            imb = (mx - mean) / mean * 100.0 if mean > 0 else 0.0
            w.writerow([t["scaling"], t["np"], t["N"], t["state"], t["rep"],
                        t["key"], t["min_s"], t["max_s"], t["mean_s"],
                        f"{imb:.3f}", t["step"], t["action"], t["source"],
                        t["basis"]])

    ccols = ["scaling", "np", "N", "state", "rep", "step", "action",
             "ilist_builds", "op_keys_built_max", "op_cache_size_max",
             "n_local_min", "n_local_max", "fallback_pairs_max", "cells",
             "max_cell_depth", "wall_max_s", "complete", "source", "basis"]
    cpath = os.path.join(a.out_dir, "scaling_calls.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore")
        w.writeheader()
        for c in expand_shared(all_c):
            w.writerow(c)

    if a.ranks:
        rpath = os.path.join(a.out_dir, "scaling_ranks.csv")
        with open(rpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scaling", "np", "N", "state", "rep", "step",
                        "timer_name", "rank", "value_s", "basis"])
            for r in expand_shared(all_r):
                for i, v in enumerate(r["v"].split(",")):
                    w.writerow([r["scaling"], r["np"], r["N"], r["state"],
                                r["rep"], r["step"], r["key"], i, v,
                                r["basis"]])
        print(f"wrote {rpath}", file=sys.stderr)

    print(f"wrote {tpath}\nwrote {cpath}", file=sys.stderr)
    if a.repo_copy and os.path.getsize(tpath) < 5 * 1024 * 1024:
        import shutil
        os.makedirs(a.repo_copy, exist_ok=True)
        for p in (tpath, cpath):
            shutil.copy(p, a.repo_copy)
        print(f"copied CSVs to {a.repo_copy}", file=sys.stderr)


if __name__ == "__main__":
    main()
