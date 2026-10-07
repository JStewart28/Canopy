#!/usr/bin/env python3
"""Compare the tagged lines of two canopy_ctest job logs as multisets.

    compare_tagged_lines.py <baseline.log> <run.log>

Tags compared: the pre-existing lines a tree change must not move (tree-opt
A2). Prints, per tag, the line count in each log and how many matched; then
every unmatched line. Exit 0 iff every tagged line matched in both directions.
"""
import collections
import re
import sys

TAGS = re.compile(
    r"^\[(two-scale|two-scale-refusals|b0-dd|b0b-[a-z-]+|dd-hist|a1-balance|"
    r"b2-retain|ct-solve|ct-cache[a-z-]*|multisolve-dev|multisolve-probe|"
    r"fusedm2l-[a-z-]+)\]")


def tagged(path):
    out = collections.Counter()
    with open(path, errors="replace") as f:
        for line in f:
            # ctest -V prefixes each output line with "<test number>: ".
            line = re.sub(r"^\d+: ", "", line.rstrip("\n"))
            m = TAGS.match(line)
            if m:
                out[line] += 1
    return out


base, run = tagged(sys.argv[1]), tagged(sys.argv[2])
tag_of = lambda l: TAGS.match(l).group(1)
per_tag = collections.defaultdict(lambda: [0, 0, 0])
for l, c in base.items():
    per_tag[tag_of(l)][0] += c
for l, c in run.items():
    per_tag[tag_of(l)][1] += c
matched = base & run
for l, c in matched.items():
    per_tag[tag_of(l)][2] += c
for t in sorted(per_tag):
    b, r, m = per_tag[t]
    print(f"{t:22s} baseline {b:5d} run {r:5d} matched {m:5d}")
nb, nr, nm = sum(base.values()), sum(run.values()), sum(matched.values())
print(f"{'TOTAL':22s} baseline {nb:5d} run {nr:5d} matched {nm:5d}")
for l, c in sorted((base - run).items()):
    print(f"ONLY-BASELINE x{c}: {l}")
for l, c in sorted((run - base).items()):
    print(f"ONLY-RUN x{c}: {l}")
sys.exit(0 if nb == nr == nm and nb > 0 else 1)
