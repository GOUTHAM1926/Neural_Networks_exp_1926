# 2026-10-19: the unit-boundary regions (unit start, first tile, diagonal step, last P.V + epilogue) of FA3 and ours, by stall
# reason and by kind of instruction, in cycles PER UNIT (consumer samples only; = share x cycles / units). Work doc 45.53.
import csv, re, collections
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
B = "session_2026_10_19/box_runs/"
UNITS = 4608 / 132
K = {"FA3": (B + "ncu_fa3_k5q_1019/fa3_dense", "FlashAttnFwdSm90",
             [("unit start + first tile", 0, 1558), ("last P.V + epilogue", 2027, 2302)], 2302),
     "SM3m": (B + "sm3m_1019/ncu_vj2_SM3m_dense", "gqa_v62",
              [("unit start + first tile", 0, 1042), ("diagonal step (+ loop exit)", 1523, 2142), ("last P.V + epilogue", 2142, 2707)], 2707)}
def kind(o):
    for pre, k in (("HGMMA", "HGMMA"), ("WARPGROUP.DEPBAR", "wgmma wait"), ("WARPGROUP.ARRIVE", "wgmma fence"), ("BAR", "named barrier"),
                   ("SYNCS.PHASECHK", "mbarrier wait"), ("SYNCS", "mbarrier other"), ("MUFU", "MUFU"), ("FMNMX", "FMNMX"),
                   ("SHFL", "SHFL"), ("F", "other FP"), ("ST", "store"), ("LD", "load"), ("U", "uniform"), ("R2UR", "uniform"), ("BRA", "branch")):
        if o.startswith(pre): return k
    return "integer / other"
for name, (base, tag, regs, prod) in K.items():
    rows = list(csv.reader(open(base + ".source.csv"))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
    body = [r for r in rows[k + 2:] if len(r) >= len(h)]
    RS = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
    src = [r[idx["Source"]].strip() for r in body]; smp = [f(r[idx["Warp Stall Sampling (All Samples)"]]) for r in body]
    op = [re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else "" for s in src]
    raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
    cyc = f([dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
    cons = sum(smp[:prod]); sc = cyc / cons / UNITS
    for nm, a, b in regs:
        reas = collections.Counter(); kinds = collections.Counter()
        for i in range(a, b):
            lab = sum(f(body[i][idx[c]]) for c in RS)
            for c in RS: reas[c[6:]] += f(body[i][idx[c]]) * sc
            reas["gmma (unlabeled)"] += (smp[i] - lab) * sc
            kinds[kind(op[i])] += smp[i] * sc
        t = sum(smp[a:b]) * sc
        print(f"== {name}: {nm}: {t:,.0f} cycles per unit")
        print("   by reason: " + ", ".join(f"{k} {v:,.0f}" for k, v in reas.most_common(8) if v >= 20))
        print("   by kind:   " + ", ".join(f"{k} {v:,.0f}" for k, v in kinds.most_common(9) if v >= 20))
