# 2026-10-16: executed warp-instructions by SASS opcode from an ncu source CSV (--page source --print-source sass).
# usage: python3 sass_mix.py prof_1016/ncu_ours_dense.source.csv [top]
import csv, sys, re
from collections import Counter
f = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rows = list(csv.reader(open(f)))
h = rows[1]; iS = h.index("Source"); iE = h.index("Instructions Executed"); iW = h.index("Warp Stall Sampling (All Samples)")
cnt, smp = Counter(), Counter(); tot = tots = 0
for r in rows[2:]:
    if len(r) <= iE or not r[iE].strip(): continue
    src = r[iS].strip()
    src = re.sub(r"^@!?U?P\w+\s+", "", src)
    op = src.split()[0] if src else "?"
    e = float(r[iE].replace(",", "")); s = float(r[iW].replace(",", "") or 0)
    cnt[op] += e; smp[op] += s; tot += e; tots += s
print(f"{f}: {tot/1e6:.1f} M warp-instr, {int(tots)} stall samples")
for op, e in cnt.most_common(top):
    print(f"  {op:22s} {e/1e6:8.2f} M  {100*e/tot:5.1f}%   samples {100*smp[op]/tots:5.1f}%")
