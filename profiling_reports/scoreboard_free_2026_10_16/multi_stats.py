# 2026-10-16: paired per-round stats of ab_multi_1016.sh runs: every candidate vs the base library (the first one), per
# shape and direction; free clocks. usage: python3 multi_stats.py <run folder> <base> <candidate> ...
import re, sys, glob, math, statistics as st
L, base, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
def get(n, r, shape, impl, d):
    for l in open(f"{L}/race_{r:02d}_{n}_{shape}.log"):
        if l.startswith("BENCH") and f"impl={impl} " in l and f"dir={d} " in l: return float(re.search(r"med_ms=([0-9.]+)", l).group(1))
rounds = sorted({int(re.search(r"race_(\d+)_", f).group(1)) for f in glob.glob(f"{L}/race_*_{base}_dense.log")})
for c in cands:
    for shape, impl in (("dense", "ours_dense"), ("packed", "ours_A_plan")):
        for d in ("fwd", "bwd"):
            x = [(get(c, r, shape, impl, d) - get(base, r, shape, impl, d)) / get(base, r, shape, impl, d) * 100 for r in rounds]
            m, sd = st.mean(x), st.stdev(x)
            print(f"{c:8s} vs {base}: {shape:6s} {d}: base {st.median(get(base, r, shape, impl, d) for r in rounds):.4f} ms, "
                  f"{c} {st.median(get(c, r, shape, impl, d) for r in rounds):.4f} ms | mean {m:+.2f}% sd {sd:.2f} t={m / (sd / math.sqrt(len(x))):+.1f} "
                  f"faster in {sum(v < 0 for v in x)}/{len(x)}")
