# 2026-10-16: paired race statistics of ab_mask_1016.sh runs (new vs old library, per round), and the ncu comparison.
# usage: python3 race_stats.py <run folder>
import re, sys, csv, math, statistics as st
d = sys.argv[1]
def get(f, impl, dr):
    for l in open(f):
        if l.startswith("BENCH") and f"impl={impl} " in l and f"dir={dr} " in l:
            return float(re.search(r"med_ms=([0-9.]+)", l).group(1))
print(open(f"{d}/status").read().splitlines()[1])
for shape, impl in [("dense", "ours_dense"), ("packed", "ours_A_plan")]:
    for dr in ["bwd", "fwd"]:
        rows = [(get(f"{d}/race_{r:02d}_old_{shape}.log", impl, dr), get(f"{d}/race_{r:02d}_new_{shape}.log", impl, dr)) for r in range(1, 13)]
        x = [(n - o) / o * 100 for o, n in rows]
        m, sd = st.mean(x), st.stdev(x); t = m / (sd / math.sqrt(len(x)))
        print(f"{shape:6s} {dr}: old {st.median(o for o, _ in rows):.4f} new {st.median(n for _, n in rows):.4f} ms | new vs old: mean {m:+.2f}% "
              f"sd {sd:.2f}% t={t:+.2f} new faster in {sum(v < 0 for v in x)}/12 | " + " ".join(f"{v:+.1f}" for v in x))
def load(p):
    rows = list(csv.reader(open(p))); return dict(zip(rows[0], rows[2]))
num = lambda s: float(s.replace(",", ""))
for shape in ["dense", "packed"]:
    o = load(f"../bwd_scoreboard_2026_10_16/prof_1016/ncu_ours_{shape}.raw.csv"); n = load(f"{d}/ncu_new_{shape}.raw.csv")
    print(f"ncu {shape}: " + " | ".join(f"{lab} {num(o[k]) / 1e6:.2f}M -> {num(n[k]) / 1e6:.2f}M ({100 * (num(n[k]) - num(o[k])) / num(o[k]):+.1f}%)"
          for k, lab in [("smsp__inst_issued.sum", "instructions"), ("gpc__cycles_elapsed.max", "cycles")]))
