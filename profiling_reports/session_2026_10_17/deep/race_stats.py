# 2026-10-17: paired statistics of an ab_multi.sh run (free clocks): every candidate vs a base, and every candidate pair.
# Per round: % change of the candidate's median (and p10) vs the base's, same round. Prints mean %, t (mean / (sd / sqrt n)),
# first half vs second half of the rounds, and in how many rounds the candidate was faster.
# usage: python3 race_stats.py <run folder> <base lib> <candidate lib> ... [--pairs]
import re, sys, glob, math, statistics as st
args = [a for a in sys.argv[1:] if not a.startswith("--")]; pairs = "--pairs" in sys.argv
L, base, cands = args[0], args[1], args[2:]
def get(n, r, shape, d, k):
    impl = "ours_dense" if shape == "dense" else "ours_A_plan"
    for l in open(f"{L}/race_{r:02d}_{n}_{shape}.log"):
        if l.startswith("BENCH") and f"impl={impl} " in l and f"dir={d} " in l: return float(re.search(k + r"=([0-9.]+)", l).group(1))
    return None
rounds = sorted({int(re.search(r"race_(\d+)_", f).group(1)) for f in glob.glob(f"{L}/race_*_{base}_dense.log")})
dirs = [d for d in ("fwd", "bwd") if get(base, rounds[0], "dense", d, "med_ms") is not None]
print(f"run {L.rstrip('/').split('/')[-1]}: {len(rounds)} rounds, directions {dirs}")
print("medians over the rounds (ms):")
for shape in ("dense", "packed"):
    for d in dirs:
        print(f"  {shape:6s} {d}: " + "  ".join(f"{n.replace('vj2_', '')} {st.median(get(n, r, shape, d, 'med_ms') for r in rounds):.4f}" for n in [base] + cands))
def comp(c, b):
    for shape in ("dense", "packed"):
        for d in dirs:
            out = []
            for k in ("med_ms", "p10_ms"):
                x = [(get(c, r, shape, d, k) - get(b, r, shape, d, k)) / get(b, r, shape, d, k) * 100 for r in rounds]
                m, sd = st.mean(x), st.stdev(x); out.append((m, m / (sd / math.sqrt(len(x))) if sd > 0 else float("inf"), x))
            x = out[0][2]; h = len(x) // 2
            print(f"  {c.replace('vj2_', ''):5s} vs {b.replace('vj2_', ''):5s} {shape:6s} {d}: {out[0][0]:+6.2f}% t={out[0][1]:+6.1f} | p10 {out[1][0]:+6.2f}% "
                  f"t={out[1][1]:+6.1f} | halves {st.mean(x[:h]):+.2f} / {st.mean(x[h:]):+.2f} | faster {sum(v < 0 for v in x)}/{len(x)}")
print("paired per round, % change (negative = faster):")
for c in cands: comp(c, base)
if pairs:
    for i, a in enumerate(cands):
        for b2 in cands[i + 1:]: comp(b2, a)
