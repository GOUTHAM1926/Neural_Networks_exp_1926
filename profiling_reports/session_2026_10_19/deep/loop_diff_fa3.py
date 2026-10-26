# 2026-10-19: FA3's steady dense loop step vs ours, instruction by instruction, grouped by JOB (work doc 45.52). Both from ncu on
# the same box (--set full, one launch; source page = SASS + executed count + warp-stall samples per instruction). Per job:
# instructions in the loop body, executions per step per warp (group-only code counts its share), and the time spent there
# per step = share of the loop's samples x the step period measured by the level-2 timeline traces (FA3 2,367, ours 2,551).
# usage (from profiling_reports/): python3 session_2026_10_19/deep/loop_diff_fa3.py [--list]   (--list: every instruction)
import csv, re, sys, collections

B = "session_2026_10_19/box_runs/"
KERNELS = [("FA3", B + "ncu_fa3_k5q_1019/fa3_dense.source.csv", "FlashAttnFwdSm90", 2367.0),
           ("ours", B + "lh_1019/ncu_vj2_LHO_dense.source.csv", "gqa_v62", 2551.0)]   # LHO dense kernel == LHOd dense (SASS check)

def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0

def load(path, tag):
    rows = list(csv.reader(open(path))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
    body = [r for r in rows[k + 2:] if len(r) >= len(h)]
    RS = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
    out = []
    for r in body:
        st = {c[6:]: f(r[idx[c]]) for c in RS}
        allx = f(r[idx["Warp Stall Sampling (All Samples)"]])
        st["gmma (unlabeled)"] = max(0.0, allx - sum(st.values()))
        out.append(dict(src=r[idx["Source"]].strip(), ex=int(f(r[idx["Instructions Executed"]])), smp=allx, st=st))
    return out

def job(s):
    pred = re.match(r"^@!?U?P[T0-9]+\s+", s)
    t = re.sub(r"^@!?U?P[T0-9]+\s+", "", s).rstrip(" ;")
    op = t.split()[0] if t else ""
    if op.startswith("HGMMA"): return "HGMMA P.V" if ".tnsp" in t or re.search(r"HGMMA\S* R\d+, R\d+", t) else "HGMMA QK"
    if op.startswith("WARPGROUP.ARRIVE"): return "wgmma fence (WARPGROUP.ARRIVE)"
    if op.startswith("WARPGROUP.DEPBAR"): return "wgmma wait " + ("QK (0x1)" if t.endswith("0x1") else "P.V (0x0)")
    if op.startswith("BAR."): return "named barrier (baton sync / arrive)"
    if "TRYWAIT" in op: return "mbarrier wait (K / V landed)"
    if op.startswith("SYNCS.ARRIVE"): return "mbarrier arrive (release / order trick)"
    if op.startswith("SYNCS"): return "mbarrier other"
    if op.startswith("FMNMX"): return "row max (FMNMX)"
    if op.startswith("SHFL"): return "shuffle (row max / sum across lanes)"
    if op.startswith("MUFU.EX2"): return "exp (MUFU.EX2)"
    if op.startswith("MUFU"): return "MUFU other"
    if op.startswith("FFMA"): return "FFMA (exp argument / scale)"
    if op.startswith("FADD"): return "FADD (row sums, max diff)"
    if op.startswith("FMUL"): return "FMUL (O rescale, scale)"
    if op.startswith("F2FP"): return "F2FP (pack P to bf16)"
    if op.startswith(("FSEL", "FSETP", "FCHK", "FMNMX", "FRND", "F2F")): return "float select / compare (check_inf)"
    if op.startswith(("U", "R2UR", "S2UR", "VOTEU")): return "uniform datapath (loop state, descriptors)"
    if op.startswith(("BRA", "BSSY", "BSYNC", "EXIT", "CALL", "RET", "WARPSYNC", "NOP")): return "branch / control"
    if op.startswith(("IMAD", "IADD", "LEA", "ISETP", "SEL", "LOP3", "SHF", "VIADD", "MOV", "PLOP3", "IABS", "I2F", "F2I", "PRMT", "LDC", "S2R", "CS2R")):
        return "integer / predicate (loop, stage, addresses)"
    if op.startswith(("LDS", "STS", "LDG", "STG", "LD.", "ST.", "ATOM", "RED", "STSM", "LDSM")): return "memory"
    return "other: " + op

def loop_of(ins):
    by_ex = collections.Counter()
    for x in ins:
        if x["ex"] >= 1000: by_ex[x["ex"]] += x["smp"]
    lx = by_ex.most_common(1)[0][0]
    lo = min(i for i, x in enumerate(ins) if x["ex"] == lx); hi = max(i for i, x in enumerate(ins) if x["ex"] == lx)
    return lx, lo, hi

res = {}
for name, path, tag, period in KERNELS:
    ins = load(path, tag); lx, lo, hi = loop_of(ins)
    L = ins[lo:hi + 1]; tot = sum(x["smp"] for x in L)
    agg = collections.defaultdict(lambda: dict(n=0, per_step=0.0, cyc=0.0, st=collections.Counter()))
    for x in L:
        a = agg[job(x["src"])]; a["n"] += 1; a["per_step"] += x["ex"] / lx; a["cyc"] += x["smp"] / tot * period
        for k, v in x["st"].items(): a["st"][k] += v / tot * period
    res[name] = (agg, lx, lo, hi, period)
    if "--list" in sys.argv:
        print(f"===== {name} steady loop {lo}..{hi}")
        for i, x in enumerate(L):
            top = sorted(x["st"].items(), key=lambda kv: -kv[1])[:3]
            print(f"{lo + i:5d} {x['ex'] / lx:4.2f} {x['smp'] / tot * period:6.1f} {job(x['src'])[:28]:28s} {x['src'][:70]:70s} "
                  + " ".join(f"{k}:{v / tot * period:.1f}" for k, v in top if v > 0))

jobs = sorted(set(res["FA3"][0]) | set(res["ours"][0]), key=lambda j: -(res["ours"][0].get(j, {}).get("cyc", 0) + res["FA3"][0].get(j, {}).get("cyc", 0)))
print(f"{'job':48s} | {'FA3 instr':>9s} {'/step':>6s} {'cycles':>7s} | {'ours instr':>10s} {'/step':>6s} {'cycles':>7s} | {'diff cyc':>8s}")
tot_f = tot_o = 0.0
for j in jobs:
    a = res["FA3"][0].get(j); b = res["ours"][0].get(j)
    ca = a["cyc"] if a else 0.0; cb = b["cyc"] if b else 0.0; tot_f += ca; tot_o += cb
    print(f"{j:48s} | {a['n'] if a else 0:9d} {a['per_step'] if a else 0:6.2f} {ca:7.0f} | {b['n'] if b else 0:10d} {b['per_step'] if b else 0:6.2f} {cb:7.0f} | {cb - ca:+8.0f}")
print(f"{'total (= period)':48s} | {'':9s} {'':6s} {tot_f:7.0f} | {'':10s} {'':6s} {tot_o:7.0f} | {tot_o - tot_f:+8.0f}")
