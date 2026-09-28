#!/usr/bin/env python3
"""F2 evidence: exact kernel counts over whole micro-steps, from the nsys sqlite exports.

The 30 s nsys window cuts micro-steps at both ends, so "instances / loss-forward count" blurs by about one
micro-step. Here the window is cut ON micro-step boundaries instead: from the start of loss-forward kernel
number k to the start of number k+64. That is exactly 64 backward passes (156 weight accumulations each),
exactly one optimizer step, and the same composition in both arms.

Expected OLD -> NEW over 64 micro-steps: convert-to-fp32 -9,984, fp32 add -9,984, fused +9,984,
all kernels -9,984 (= 156 x 64).

Usage: python3 f2_exact_window.py <dir with OLD.sqlite and NEW.sqlite>
"""
import sqlite3, statistics as st, sys
from collections import Counter, defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "."
RULER = "sparse_ce_forward_kernel_vec_save_stats"
MICRO = 64

def load(arm):
    c = sqlite3.connect(f"{d}/{arm}.sqlite")
    s = dict(c.execute("select id, value from StringIds"))
    k = [(a, b, s[n]) for a, b, n in c.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    rt = [(a, s[n]) for a, n in c.execute("select start, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")]
    mc = [a for (a,) in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY")]
    ms = [a for (a,) in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMSET")]
    return k, rt, mc, ms

def window(arm, k, rt, mc, ms, i):
    L = [a for a, _, n in k if RULER in n]
    t0, t1 = L[i], L[i + MICRO]
    cnt, tim = Counter(), defaultdict(int)
    for a, b, n in k:
        if t0 <= a < t1:
            cnt[n] += 1; tim[n] += b - a
    api = Counter(n for a, n in rt if t0 <= a < t1)
    return dict(t0=t0, t1=t1, wall=(t1 - t0) / 1e6, cnt=cnt, tim=tim, api=api,
                memcpy=sum(t0 <= a < t1 for a in mc), memset=sum(t0 <= a < t1 for a in ms), nL=len(L))

data = {arm: load(arm) for arm in ("OLD", "NEW")}
nwin = {arm: sum(RULER in n for _, _, n in data[arm][0]) - MICRO for arm in data}
print(f"loss-forward kernels captured: OLD {nwin['OLD'] + MICRO}, NEW {nwin['NEW'] + MICRO}  "
      f"-> {min(nwin.values())} overlapping {MICRO}-micro-step windows usable per arm")

# Window 0 in each arm: the main table.
W = {arm: window(arm, *data[arm], 0) for arm in data}
O, N = W["OLD"], W["NEW"]
print(f"\n==== one exact window = {MICRO} micro-steps = 1 optimizer step (first window in each arm)")
print(f"wall time (loss-fwd #0 to #64): OLD {O['wall']:.1f} ms  NEW {N['wall']:.1f} ms  change {N['wall'] - O['wall']:+.1f} ms")
to, tn = sum(O["cnt"].values()), sum(N["cnt"].values())
print(f"kernels (= launches) in window: OLD {to}  NEW {tn}  change {tn - to:+d}   (expected -{156 * MICRO})")
print(f"memcpy OLD {O['memcpy']} NEW {N['memcpy']}   memset OLD {O['memset']} NEW {N['memset']}")
print("\nkernels whose count changed (OLD -> NEW over the window), with their GPU time:")
dt_o = dt_n = 0
for n in sorted(set(O["cnt"]) | set(N["cnt"]), key=lambda n: N["cnt"][n] - O["cnt"][n]):
    if N["cnt"][n] != O["cnt"][n]:
        dt_o += O["tim"][n]; dt_n += N["tim"][n]
        print(f"  {O['cnt'][n]:7d} -> {N['cnt'][n]:7d} ({N['cnt'][n] - O['cnt'][n]:+7d})   "
              f"{O['tim'][n] / 1e6:8.2f} ms -> {N['tim'][n] / 1e6:8.2f} ms   {n[:100]}")
print(f"  GPU time in these kernels: OLD {dt_o / 1e6:.2f} ms  NEW {dt_n / 1e6:.2f} ms  change {(dt_n - dt_o) / 1e6:+.2f} ms"
      f"  = {(dt_n - dt_o) / 1e6 / MICRO:+.3f} ms per micro-step")
ko, kn = sum(O["tim"].values()) / 1e6, sum(N["tim"].values()) / 1e6
print(f"all kernel GPU time in window: OLD {ko:.1f} ms  NEW {kn:.1f} ms  change {kn - ko:+.1f} ms")
print("\nCUDA API calls whose count changed in the window:")
for n in sorted(set(O["api"]) | set(N["api"])):
    if O["api"][n] != N["api"][n]:
        print(f"  {O['api'][n]:7d} -> {N['api'][n]:7d} ({N['api'][n] - O['api'][n]:+7d})   {n}")
for n in ("cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync"):
    print(f"  {n}: OLD {O['api'][n]}  NEW {N['api'][n]}")

# Every usable window: the counts must not move; the wall time shows the spread.
print(f"\n==== all {min(nwin.values())} overlapping windows (not independent samples)")
for arm in data:
    ws = [window(arm, *data[arm], i) for i in range(nwin[arm])]
    tot = sorted({sum(w["cnt"].values()) for w in ws})
    walls = [w["wall"] for w in ws]
    print(f"{arm}: kernels per window {tot}   wall ms min {min(walls):.1f} median {st.median(walls):.1f} max {max(walls):.1f}")
