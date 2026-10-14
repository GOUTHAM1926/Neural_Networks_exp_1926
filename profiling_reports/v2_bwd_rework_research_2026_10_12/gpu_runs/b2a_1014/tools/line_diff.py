#!/usr/bin/env python3
# 2026-10-14 B-2a test 3: per C++ line, before (g123) vs after (b2a), all samples and each exported stall reason, in
# us-equivalent (share of the kernel's samples x kernel time). b2a inserted 6 lines at 589, so g123 line L >= 589 is
# compared with b2a line L + 6. Prints the lines that moved most, and per reason which lines moved it.
import csv, sys
G = sys.argv[1]
def load(path, t):
    rows = list(csv.reader(open(path))); h = rows[2]; ix = {c: i for i, c in enumerate(h)}   # "cuda,sass" export
    st = [c for c in h if c.startswith("stall_") and "(Not Issued)" not in c]
    g = lambda r, c: float(r[ix[c]]) if r[ix[c]] not in ("", "-") else 0.0
    data = {}
    for r in rows[2:]:
        if len(r) < len(h) or not r[0].strip().isdigit() or r[2] != "-": continue    # per-line total rows only
        d = {c[6:]: g(r, c) for c in st}; d["ALL"] = g(r, "Warp Stall Sampling (All Samples)"); data[int(r[0])] = (d, r[1].strip())
    tot = sum(d["ALL"] for d, _ in data.values())
    return {k: ({c: v / tot * t for c, v in d.items()}, s) for k, (d, s) in data.items()}
B = load(f"{G}/p1_1014/ncu_g123li.source_cuda_sass.csv", 2526.3); A = load(f"{G}/b2a_1014/ncu_b2ali.source_cuda_sass.csv", 2478.0)
m = lambda L: L + 6 if L >= 589 else L
pairs = [(L, m(L)) for L in B if m(L) in A]
keys = list(next(iter(B.values()))[0].keys())
print("per C++ line, all samples us-equiv (before g123 line -> after b2a line), biggest moves:")
for L, La in sorted(pairs, key=lambda p: -abs(A[p[1]][0]["ALL"] - B[p[0]][0]["ALL"]))[:14]:
    b, a = B[L][0], A[La][0]; dd = {k: a[k] - b[k] for k in keys if k != "ALL"}; big = sorted(dd, key=lambda k: -abs(dd[k]))[:2]
    print(f"  {L:4d}->{La:4d} {b['ALL']:7.1f} -> {a['ALL']:7.1f} ({a['ALL']-b['ALL']:+6.1f})  " + ", ".join(f"{k} {dd[k]:+.1f}" for k in big) + f" | {A[La][1][:70]}")
new = [La for La in A if La not in {m(L) for L in B}]
print("lines only in b2a:", ", ".join(f"{La} ({A[La][0]['ALL']:.1f}: {A[La][1][:40]})" for La in sorted(new) if A[La][0]["ALL"] > 0.5))
for k in ("short_sb", "barrier", "long_sb", "mio"):
    tot_b = sum(B[L][0][k] for L in B); tot_a = sum(A[L][0][k] for L in A)
    print(f"\n{k}: {tot_b:.1f} -> {tot_a:.1f} ({tot_a-tot_b:+.1f}); lines that moved it most:")
    for L, La in sorted(pairs, key=lambda p: -abs(A[p[1]][0][k] - B[p[0]][0][k]))[:6]:
        print(f"  {L:4d}->{La:4d} {B[L][0][k]:7.1f} -> {A[La][0][k]:7.1f} ({A[La][0][k]-B[L][0][k]:+6.1f}) | {A[La][1][:80]}")
