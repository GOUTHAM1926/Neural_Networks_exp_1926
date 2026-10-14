#!/usr/bin/env python3
# 2026-10-14: float-type instructions per opcode, ours (b2a) vs FA3: executed (millions, warp-level) and short_sb us-eq.
import csv, re, sys
from collections import Counter
def sections(path):
    rows = list(csv.reader(open(path))); idx = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    return [(rows[i][1], rows[i + 1], rows[i + 2:(idx[k + 1] if k + 1 < len(idx) else len(rows))]) for k, i in enumerate(idx)]
for name, path, k, t in (("ours b2a", sys.argv[1], 0, 2478.0), ("FA3", sys.argv[2], 1, 2481.4)):
    _, h, R = sections(path)[k]; ix = {c: i for i, c in enumerate(h)}
    g = lambda r, c: float(r[ix[c]]) if r[ix[c]] not in ("", "-") else 0.0
    tot = sum(g(r, "Warp Stall Sampling (All Samples)") for r in R)
    c = Counter(); sb = Counter()
    for r in R:
        s = re.sub(r"^@!?U?P\w+\s+", "", r[ix["Source"]].strip()); op = s.split(" ")[0]
        if op.split(".")[0] in ("FFMA", "FADD", "FMUL", "FSEL", "FMNMX", "FSETP", "MUFU", "F2FP", "FMUL2", "FFMA2", "FADD2"):
            c[op] += g(r, "Instructions Executed") / 1e6; sb[op] += g(r, "stall_short_sb") / tot * t
    print(f"== {name}: float-type instructions, executed (M, warp-level) | short_sb us-eq | total {sum(c.values()):.1f}M")
    for op, v in sorted(c.items(), key=lambda kv: -kv[1])[:14]:
        print(f"   {op:30s} {v:7.1f}M   short_sb {sb[op]:6.1f}")
