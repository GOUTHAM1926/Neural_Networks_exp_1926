#!/usr/bin/env python3
# Same-session A/B of two library snapshots from BENCHDENSE logs (2026-10-13, Phase 1.5).
# usage: ab_table.py <dir> <nameA> <nameB>   reads <dir>/2_<name>_r1.log and _r2.log; backward median ms.
import re, sys, statistics as st
d, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
def bench(path):
    out = {}
    for l in open(path):
        m = re.match(r"BENCH layout=(\S+) impl=\S+ dir=bwd med_ms=([\d.]+)", l)
        if m: out[m.group(1)] = float(m.group(2))
    return out
ra = [bench(f"{d}/2_{A}_r{r}.log") for r in (1, 2)]
rb = [bench(f"{d}/2_{B}_r{r}.log") for r in (1, 2)]
print(f"backward median ms, mean of 2 runs each, same session ({A} and {B} alternated A B A B)")
print(f"{'shape':16s} {A:>8s} {B:>8s} {B+'/'+A:>9s}  {'spread '+A:>10s} {'spread '+B:>10s}")
rat = []
for s in ra[0]:
    a = st.mean(r[s] for r in ra); b = st.mean(r[s] for r in rb)
    sa = 100 * abs(ra[0][s] - ra[1][s]) / a; sb = 100 * abs(rb[0][s] - rb[1][s]) / b
    rat.append(b / a)
    print(f"{s:16s} {a:8.3f} {b:8.3f} {b/a:9.3f}  {sa:9.1f}% {sb:9.1f}%")
print(f"{B}/{A}: min {min(rat):.3f} max {max(rat):.3f} mean {st.mean(rat):.3f} | {B} faster on {sum(x < 1 for x in rat)}/{len(rat)}")
