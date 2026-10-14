#!/usr/bin/env python3
# 2026-10-14: per instruction category, ours (b2a) vs FA3: executed instructions (warp-level, millions) and the
# stall time split by reason (us-equivalent = share of all samples x kernel time).
import csv, re, sys
sys.path.insert(0, sys.argv[5])
from category_us import sections, cat
def table(h, R, t):
    ix = {c: i for i, c in enumerate(h)}
    g = lambda r, c: float(r[ix[c]]) if c in ix and r[ix[c]] not in ("", "-") else 0.0
    st = [c for c in h if c.startswith("stall_") and "(Not Issued)" not in c]
    tot = sum(g(r, "Warp Stall Sampling (All Samples)") for r in R)
    out = {}
    for r in R:
        k = cat(r[ix["Source"]]); d = out.setdefault(k, {"inst": 0.0, "all": 0.0})
        d["inst"] += g(r, "Instructions Executed") / 1e6; d["all"] += g(r, "Warp Stall Sampling (All Samples)") / tot * t
        cov = 0.0
        for c in st:
            v = g(r, c) / tot * t; d[c[6:]] = d.get(c[6:], 0.0) + v; cov += v
        d["(arrive)"] = d.get("(arrive)", 0.0) + max(g(r, "Warp Stall Sampling (All Samples)") / tot * t - cov, 0.0)
    return out
ours = sections(sys.argv[1])[0]; fa3 = sections(sys.argv[2])[1]
A = table(ours[1], ours[2], float(sys.argv[3])); B = table(fa3[1], fa3[2], float(sys.argv[4]))
for k in ("float math", "GEMM issue (HGMMA)", "GEMM wait (DEPBAR)", "exp (MUFU)", "named barrier"):
    a, b = A.get(k, {}), B.get(k, {})
    print(f"\n{k}: executed instructions ours {a.get('inst',0):.1f}M, FA3 {b.get('inst',0):.1f}M | stall us-eq ours {a.get('all',0):.1f}, FA3 {b.get('all',0):.1f}")
    keys = sorted(set(a) | set(b) - {"inst", "all"}, key=lambda x: -abs(a.get(x, 0) - b.get(x, 0)))
    for x in [x for x in keys if x not in ("inst", "all")][:6]:
        print(f"   {x:16s} ours {a.get(x,0):7.1f}  FA3 {b.get(x,0):7.1f}  diff {a.get(x,0)-b.get(x,0):+7.1f}")
