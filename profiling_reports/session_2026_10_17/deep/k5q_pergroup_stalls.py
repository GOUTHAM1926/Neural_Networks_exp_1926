# per-group stall breakdown in cycle-equivalents (FA3 total = 100) and per execution; 'other' = All Samples - listed reasons
# (the source page has no warpgroup_arrive column)
import collections, sys
exec(open("session_2026_10_17/deep/ncu_k5q_vs_fa3.py").read().split("\nM = [")[0].replace("NHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 25", ""))
shape = sys.argv[1] if len(sys.argv) > 1 else "dense"
cyc = {n: f(raw_row(p, t)["sm__cycles_elapsed.avg"]) for n, p, t in RAW[shape] if n in ("FA3", "K5q")}
R = ["wait", "dispatch", "selected", "not_selected", "barrier", "long_sb", "short_sb", "mio", "math", "branch_resolving", "no_inst", "sleep", "other"]
for n, p, t in SRC[shape]:
    if n not in cyc: continue
    ins = src_kernel(p, t); tot = sum(s for _, _, s, _ in ins); sc = 100 * cyc[n] / cyc["FA3"] / tot
    g = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    for src, ex, s, st in ins:
        b = g[ex]; b[0] += 1; b[1] += s
        for k, v in st.items(): b[2][k] += v
        b[2]["other"] += s - sum(st.values())
    print(f"\n== {n} {shape}: cycle-equivalents (FA3 = 100) per group, split by stall reason")
    print(f"{'exec':>8s} {'instr':>5s} {'cyc-eq':>7s} {'per 1M exec':>11s} | " + " ".join(f"{r[:8]:>8s}" for r in R))
    for ex, (ni, s, st) in sorted(g.items(), key=lambda kv: -kv[1][1]):
        if s / tot < 0.01: continue
        print(f"{ex:8d} {ni:5d} {s * sc:7.1f} {s * sc / ex * 1e6:11.2f} | " + " ".join(f"{st[r] * sc:8.2f}" for r in R))
    rest = {r: sum(b[2][r] for ex, b in g.items() if b[1] / tot < 0.01) * sc for r in R}
    print(f"{'small':>8s} {'':5s} {sum(b[1] for b in g.values() if b[1] / tot < 0.01) * sc:7.1f} {'':11s} | " + " ".join(f"{rest[r]:8.2f}" for r in R))
