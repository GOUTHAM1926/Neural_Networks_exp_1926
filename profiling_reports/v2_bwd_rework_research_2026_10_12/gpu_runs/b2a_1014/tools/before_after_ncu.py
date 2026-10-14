#!/usr/bin/env python3
# 2026-10-14 B-2a test 3: Vj2 backward kernel before (g123, test 1.4) vs after (b2a) vs FA3 (prof_1013), dense B8 18/6.
# 1) whole kernel: time, tensor-pipe busy, every stall reason as share of samples and as us-equivalent (share x time).
# 2) per GEMM-issue instruction (HGMMA) and per GEMM wait (WARPGROUP.DEPBAR) / arrive: all samples and the samples no
#    exported per-line reason covers (= "warpgroup arrive", which the per-line CSV does not export), in us-equivalent.
# Usage: before_after_ncu.py <gpu_runs dir>
import csv, re, sys
G = sys.argv[1]
RUNS = [("before g123", f"{G}/p1_1014/ncu_g123li.raw.csv", 0, f"{G}/p1_1014/ncu_g123li.source_sass.csv", 0),
        ("after b2a", f"{G}/b2a_1014/ncu_b2ali.raw.csv", 0, f"{G}/b2a_1014/ncu_b2ali.source_sass.csv", 0),
        ("FA3", f"{G}/prof_1013/ncu_libs_bwd.raw.csv", 1, f"{G}/prof_1013/ncu_libs_bwd.source.csv", 1)]
def raw(path, k):
    R = list(csv.reader(open(path))); h = R[0]; r = R[2 + k]; ix = {c: i for i, c in enumerate(h)}
    f = lambda c: float(r[ix[c]])
    t_us = f("gpu__time_duration.sum") * (1000.0 if R[1][ix["gpu__time_duration.sum"]] == "ms" else 1.0)
    n = f("smsp__pcsamp_sample_count")
    reasons = {c[len("smsp__pcsamp_warps_issue_stalled_"):]: f(c) for c in h
               if c.startswith("smsp__pcsamp_warps_issue_stalled_") and not c.endswith("_not_issued")}
    return dict(name=r[ix["Kernel Name"]], t=t_us, tensor=f("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                clk=f("sm__cycles_elapsed.avg.per_second"), n=n, reasons=reasons)
def source(path, k):
    rows = list(csv.reader(open(path))); idx = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    i = idx[k]; j = idx[k + 1] if k + 1 < len(idx) else len(rows)
    h = rows[i + 1]; ix = {c: n for n, c in enumerate(h)}
    st = [c for c in h if c.startswith("stall_") and "(Not Issued)" not in c]
    g = lambda r, c: float(r[ix[c]]) if r[ix[c]] not in ("", "-") else 0.0
    out = []
    for r in rows[i + 2:j]:
        if len(r) < len(h): continue
        allv = g(r, "Warp Stall Sampling (All Samples)"); cov = sum(g(r, c) for c in st)
        top = max(st, key=lambda c: g(r, c)) if allv else ""
        out.append(dict(addr=r[ix["Address"]][-5:], src=re.sub(r"\s+", " ", r[ix["Source"]].strip()), ex=g(r, "Instructions Executed"),
                        all=allv, unexp=max(allv - cov, 0.0), top=top[6:], topv=g(r, top) if top else 0.0))
    return out
res = []
for name, rp, rk, sp, sk in RUNS:
    a = raw(rp, rk); a["src"] = source(sp, sk); a["tot"] = sum(x["all"] for x in a["src"]); a["label"] = name; res.append(a)
print("== whole kernel, dense B8 18/6, one launch under ncu ==")
print(f"{'':34s}" + "".join(f"{a['label']:>14s}" for a in res))
print(f"{'kernel time (us)':34s}" + "".join(f"{a['t']:14.1f}" for a in res))
print(f"{'tensor pipe busy (% of time)':34s}" + "".join(f"{a['tensor']:14.1f}" for a in res))
print(f"{'SM clock (GHz)':34s}" + "".join(f"{a['clk']:14.3f}" for a in res))
print(f"{'stall samples (raw counter)':34s}" + "".join(f"{a['n']:14.0f}" for a in res))
keys = sorted(set().union(*[a["reasons"] for a in res]), key=lambda k: -max(a["reasons"].get(k, 0) / a["n"] for a in res))
print("\n== stall reasons: share of samples / us-equivalent (share x kernel time) ==")
for k in keys:
    if max(a["reasons"].get(k, 0) / a["n"] for a in res) < 0.005: continue
    print(f"{k:34s}" + "".join(f"{100*a['reasons'].get(k,0)/a['n']:6.1f}% {a['reasons'].get(k,0)/a['n']*a['t']:6.1f}" for a in res))
for a in res:
    print(f"\n== {a['label']}: GEMM issue (HGMMA), GEMM wait (DEPBAR), warpgroup ARRIVE; us-equiv = samples / all x {a['t']:.0f} us ==")
    print(f"{'addr':6s} {'executed':>9s} {'all us':>8s} {'unexpl us':>9s}  top exported reason   instruction")
    for x in a["src"]:
        if not re.search(r"HGMMA|WARPGROUP", x["src"]): continue
        if x["all"] / a["tot"] < 0.0005: continue
        print(f"{x['addr']:6s} {x['ex']:9.0f} {x['all']/a['tot']*a['t']:8.1f} {x['unexp']/a['tot']*a['t']:9.1f}  {x['top']:12s} {x['topv']/a['tot']*a['t']:6.1f}  {x['src'][:70]}")
    u = sum(x["unexp"] for x in a["src"])
    print(f"unexplained total {100*u/a['tot']:.1f}% of samples = {u/a['tot']*a['t']:.1f} us-equiv (raw warpgroup_arrive {100*a['reasons'].get('warpgroup_arrive',0)/a['n']:.1f}%)")
