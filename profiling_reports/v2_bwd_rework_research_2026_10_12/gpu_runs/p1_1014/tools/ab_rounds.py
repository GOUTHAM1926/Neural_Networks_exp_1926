#!/usr/bin/env python3
# Same-session A/B over N alternating rounds (2026-10-14, Phase 1). Reads <dir>/2_<A>_r<k>.log and 2_<B>_r<k>.log,
# k = 1..N (BENCHDENSE logs, median ms per shape). Per shape and direction: mean of A, mean of B, B/A, and the paired
# round-by-round ratio B_k/A_k with its spread and a t value (mean(ratio - 1) / (sd / sqrt(N))); |t| > ~2.6 (N=6) is
# unlikely to be noise. The bottom line pools all shapes.
import re, sys, math, statistics as st
d, A, B, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def bench(path, dirn):
    out = {}
    for l in open(path):
        m = re.match(rf"BENCH layout=(\S+) impl=\S+ dir={dirn} med_ms=([\d.]+)", l)
        if m: out[m.group(1)] = float(m.group(2))
    return out
for dirn in ("bwd", "fwd"):
    ra = [bench(f"{d}/2_{A}_r{k}.log", dirn) for k in range(1, N + 1)]
    rb = [bench(f"{d}/2_{B}_r{k}.log", dirn) for k in range(1, N + 1)]
    print(f"\n{dirn.upper()} median ms, {N} alternating rounds ({A}, {B}, {A}, {B}, ...)")
    print(f"{'shape':16s} {A:>8s} {B:>8s} {B+'/'+A:>9s} {'paired sd':>10s} {'t':>6s}")
    allr = []
    for s in ra[0]:
        a = st.mean(r[s] for r in ra); b = st.mean(r[s] for r in rb)
        pr = [rb[k][s] / ra[k][s] for k in range(N)]; allr += pr
        sd = st.stdev(pr); t = (st.mean(pr) - 1) / (sd / math.sqrt(N)) if sd > 0 else float("inf")
        print(f"{s:16s} {a:8.3f} {b:8.3f} {b/a:9.4f} {100*sd:9.2f}% {t:6.1f}")
    m = st.mean(allr); sd = st.stdev(allr); t = (m - 1) / (sd / math.sqrt(len(allr)))
    faster = sum(1 for s in ra[0] if st.mean(r[s] for r in rb) < st.mean(r[s] for r in ra))
    print(f"ALL {dirn}: paired {B}/{A} mean {m:.4f} (sd {100*sd:.2f}%, n={len(allr)}, t={t:.1f}) | {B} faster on {faster}/{len(ra[0])} shapes")
