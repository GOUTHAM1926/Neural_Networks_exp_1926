# 2026-10-19: a candidate vs K5q (first written for SKV, work doc 45.48; now any candidate: LH, LHO 45.49), from ncu on the same box (31.56.109.89, GPU 0,
# CUDA 13.4; same settings: --set full, 4th launch, one kernel). K5q = ncu_fa3_k5q_1019 (ours_*), the candidate =
# <run folder>/ncu_vj2_<candidate>_* (printed in the column named SKV). Where do the candidate's cycles go?
#   1. totals: cycles, instructions, tensor-busy / tensor-idle cycles, DRAM, L2
#   2. stall samples by kind of instruction (cycle-equivalents: share of the kernel's samples x its cycles)
#   3. the mbarrier waits one by one (executed count = how often the code runs; samples = time spent there)
# run from profiling_reports/:  python3 session_2026_10_19/deep/ncu_cand_vs_k5q.py <candidate> <run folder>   (SKV skv_1019)
import csv, re, collections, sys

B = "session_2026_10_19/box_runs/"
RUNS = {"K5q": (B + "ncu_fa3_k5q_1019/ours_{}.raw.csv", B + "ncu_fa3_k5q_1019/ours_{}.source.csv"),
        "SKV": (B + sys.argv[2] + "/ncu_vj2_" + sys.argv[1] + "_{}.raw.csv", B + sys.argv[2] + "/ncu_vj2_" + sys.argv[1] + "_{}.source.csv")}
TAG = "gqa_v62"

def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0

def raw_row(path):
    rows = list(csv.reader(open(path))); h = rows[0]
    return [dict(zip(h, r)) for r in rows[2:] if TAG in r[h.index("Kernel Name")]][0]

def src_kernel(path):   # [(sass, executed, samples, {stall: samples})]
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if TAG in b["name"]); h = b["rows"][0]; idx = {c: i for i, c in enumerate(h)}
    out = []
    for r in b["rows"][1:]:
        if len(r) < len(h): continue
        g = lambda c: f(r[idx[c]]) if c in idx else 0.0
        st = {c[len("stall_"):]: g(c) for c in h if c.startswith("stall_") and "Not Issued" not in c}
        out.append((r[idx["Source"]].strip(), int(g("Instructions Executed")), g("Warp Stall Sampling (All Samples)"), st))
    return out

def opc(s):
    s = re.sub(r"^@!?U?P[T0-9]+\s+", "", s)
    return s.split()[0] if s else ""

def kind(s, prev):
    o = opc(s)
    if "TRYWAIT" in o: return "mbarrier wait (TRYWAIT)"
    if o.startswith("BRA") and prev is not None and "TRYWAIT" in opc(prev): return "mbarrier wait (TRYWAIT)"
    if o.startswith("SYNCS.ARRIVE") or o.startswith("SYNCS.EXCH") : return "mbarrier arrive / expect-tx"
    if o.startswith("BAR"): return "named barrier (baton, order_bar)"
    if o.startswith("WARPGROUP.DEPBAR") or o.startswith("WARPGROUP.ARRIVE") or o.startswith("WARPSYNC"): return "wgmma wait / fence"
    if o.startswith("HGMMA"): return "HGMMA issue"
    if o.startswith("MUFU"): return "exp (MUFU)"
    if o.startswith("UTMALDG") or o.startswith("UBLKCP") or o.startswith("UTMASTG"): return "TMA"
    return "other"

M = [("time us", "gpu__time_duration.sum", 1), ("SM clock MHz", "smsp__cycles_elapsed.avg.per_second", 1e-6),
     ("cycles (avg SM)", "sm__cycles_elapsed.avg", 1), ("instructions M", "smsp__inst_executed.sum", 1e-6),
     ("tensor busy %", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", 1),
     ("DRAM read MB", "dram__bytes_read.sum", 1e-6), ("L2 hit %", "lts__t_sector_hit_rate.pct", 1)]
for case in ("dense", "packed"):
    print(f"\n===== {case} =====")
    raw = {n: raw_row(p[0].format(case)) for n, p in RUNS.items()}
    for label, key, sc in M:
        v = {n: f(raw[n].get(key, "nan")) * sc for n in RUNS}
        print(f"  {label:18s} K5q {v['K5q']:>14,.1f}   SKV {v['SKV']:>14,.1f}   SKV/K5q {v['SKV'] / v['K5q'] if v['K5q'] else 0:.4f}")
    cyc = {n: f(raw[n]["sm__cycles_elapsed.avg"]) for n in RUNS}
    busy = {n: cyc[n] * f(raw[n]["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"]) / 100 for n in RUNS}
    print(f"  tensor-BUSY cycles K5q {busy['K5q']:,.0f}  SKV {busy['SKV']:,.0f}  | tensor-IDLE K5q {cyc['K5q'] - busy['K5q']:,.0f}"
          f"  SKV {cyc['SKV'] - busy['SKV']:,.0f}  (SKV - K5q: idle {cyc['SKV'] - busy['SKV'] - cyc['K5q'] + busy['K5q']:+,.0f},"
          f" total {cyc['SKV'] - cyc['K5q']:+,.0f})")
    src = {n: src_kernel(p[1].format(case)) for n, p in RUNS.items()}
    # 2. stall samples by kind, in cycle-equivalents (share x cycles)
    print("  samples by kind, cycle-equivalents per SM (share of the kernel's samples x its cycles):")
    agg = {}
    for n, rows in src.items():
        tot = sum(r[2] for r in rows); a = collections.Counter(); prev = None
        for s, ex, sm, st in rows:
            a[kind(s, prev)] += sm; prev = s
        agg[n] = {k: v / tot * cyc[n] for k, v in a.items()}
    for k in sorted(set(agg["K5q"]) | set(agg["SKV"]), key=lambda k: -agg["SKV"].get(k, 0)):
        print(f"    {k:34s} K5q {agg['K5q'].get(k, 0):>10,.0f}  SKV {agg['SKV'].get(k, 0):>10,.0f}  diff {agg['SKV'].get(k, 0) - agg['K5q'].get(k, 0):>+9,.0f}")
    # 3. each mbarrier wait site (TRYWAIT + its branch), with its executed count and time
    for n, rows in src.items():
        tot = sum(r[2] for r in rows)
        print(f"  {n}: mbarrier wait sites (executed, cycle-equivalents incl. the retry branch):")
        for i, (s, ex, sm, st) in enumerate(rows):
            if "TRYWAIT" not in opc(s): continue
            smp = sm + (rows[i + 1][2] if i + 1 < len(rows) and opc(rows[i + 1][0]).startswith("BRA") else 0)
            if ex == 0 and smp == 0: continue
            print(f"    line {i:5d} exec {ex:>9,}  cyc-eq {smp / tot * cyc[n]:>9,.0f}   {s[:90]}")
