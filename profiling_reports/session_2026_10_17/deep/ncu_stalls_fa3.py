import csv
def load(path):
    rows = list(csv.reader(open(path))); h = rows[0]
    return [dict(zip(h, r)) for r in rows[2:]]
srcs = [("FA3", "v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/ncu_libs_fwd.raw.csv", "FlashAttnFwdSm90"),
        ("F3c", "session_2026_10_17/box_runs/multi_1017/ncu_vj2_F3c_dense.raw.csv", "gqa_v62"),
        ("K1", "session_2026_10_17/box_runs/k1_1017/ncu_vj2_K1_dense.raw.csv", "gqa_v62")]
K = [(n, [r for r in load(p) if t in r["Kernel Name"]][0]) for n, p, t in srcs]
pre = "smsp__pcsamp_warps_issue_stalled_"
keys = sorted({k[len(pre):] for _, r in K for k in r if k.startswith(pre) and not k.endswith("_not_issued")})
def f(v):
    try: return float(v.replace(",", ""))
    except: return 0.0
tot = {n: sum(f(r.get(pre + k, "0")) for k in keys) for n, r in K}
cyc = {n: f(r["sm__cycles_elapsed.avg"]) for n, r in K}
# samples are taken per warp at a fixed interval: convert each reason's share into "warp-cycles per SM" = share * cycles * warps
print("stall reason: share of samples (%) | and in cycles-equivalents relative to FA3's total (FA3 total = 100)")
print(f"{'reason':28s}" + "".join(f"{n:>9s}" for n, _ in K) + "".join(f"{n+' eq':>10s}" for n, _ in K))
rows = []
for k in keys:
    sh = {n: 100 * f(r.get(pre + k, "0")) / tot[n] for n, r in K}
    eq = {n: sh[n] * cyc[n] / cyc["FA3"] for n, _ in K}
    rows.append((eq["K1"] - eq["FA3"], k, sh, eq))
for d, k, sh, eq in sorted(rows, reverse=True):
    if max(sh.values()) < 0.3: continue
    print(f"{k:28s}" + "".join(f"{sh[n]:9.1f}" for n, _ in K) + "".join(f"{eq[n]:10.1f}" for n, _ in K) + f"   K1-FA3 {d:+6.1f}")
print("total samples:", {n: int(t) for n, t in tot.items()}, "| cycles M:", {n: round(c / 1e6, 3) for n, c in cyc.items()})
