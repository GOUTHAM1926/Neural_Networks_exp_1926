# 2026-10-19: one dense UNIT's consumer time, FA3 vs ours, from ncu PC sampling (work doc 45.53). Regions by instruction index
# (from deep/kernel_map.py); consumer time only (the producer warp's samples are left out): cycles per SM of a region =
# its samples / all consumer samples x the kernel's cycles per SM; per unit = / units per SM (4,608 / 132); per step = the
# loop's / steps per warp per SM. FA3 runs the diagonal block FIRST (its first tile, masked) and p plain steps; ours runs
# tile 0 first, p - 1 plain steps and the diagonal LAST (a separate masked step).
import csv, sys
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
B = "session_2026_10_19/box_runs/"
K = {"FA3": (B + "ncu_fa3_k5q_1019/fa3_dense", "FlashAttnFwdSm90",
             [("setup + unit start (Q wait)", 0, 330), ("first tile: QK + softmax (the diagonal, masked)", 330, 1558),
              ("steady loop (plain steps)", 1558, 2027), ("last P.V + epilogue (O, LSE stores)", 2027, 2302)], 2302),
     "LHOd": (B + "lh_1019/ncu_vj2_LHO_dense", "gqa_v62",
              [("setup + unit start (Q wait)", 0, 199), ("first tile: QK + softmax (tile 0, plain)", 199, 1072),
               ("steady loop (plain steps)", 1072, 1557), ("loop exit + diagonal step (masked)", 1557, 2210),
               ("last P.V + epilogue (O, LSE stores)", 2210, 2775)], 2775),
     "PF": (B + "pf_1019/ncu_vj2_PF_dense", "gqa_v62",
              [("setup + unit start (Q wait)", 0, 199), ("first tile: QK + softmax (tile 0, plain)", 199, 1042),
               ("steady loop (plain steps)", 1042, 1523), ("loop exit + diagonal step (masked)", 1523, 2142),
               ("last P.V + epilogue (O, LSE stores)", 2142, 2707)], 2707),
     "SM3m": (B + "sm3m_1019/ncu_vj2_SM3m_dense", "gqa_v62",
              [("setup + unit start (Q wait)", 0, 199), ("first tile: QK + softmax (tile 0, plain)", 199, 1042),
               ("steady loop (plain steps)", 1042, 1523), ("loop exit + diagonal step (masked)", 1523, 2142),
               ("last P.V + epilogue (O, LSE stores)", 2142, 2707)], 2707)}
UNITS = 4608 / 132
for name, (base, tag, regs, prod) in K.items():
    rows = list(csv.reader(open(base + ".source.csv"))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
    body = [r for r in rows[k + 2:] if len(r) >= len(h)]
    smp = [f(r[idx["Warp Stall Sampling (All Samples)"]]) for r in body]; ex = [int(f(r[idx["Instructions Executed"]])) for r in body]
    raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
    cyc = f([dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
    cons = sum(smp[:prod]); prod_s = sum(smp[prod:])
    lo, hi = next((a, b) for nm, a, b in regs if nm.startswith("steady"))
    lx = max(ex[lo:hi]); steps = lx / 8 / 132          # loop executions per consumer warp per SM
    print(f"== {name}: {cyc:,.0f} cycles per SM | producer warp {100 * prod_s / (cons + prod_s):.1f}% of samples (left out) | "
          f"{UNITS:.2f} units and {steps:.1f} plain steps per SM (per consumer warp)")
    tot = 0
    for nm, a, b in regs:
        c = sum(smp[a:b]) / cons * cyc; tot += c
        extra = f" = {c / steps:,.0f} per plain step" if nm.startswith("steady") else f" = {c / UNITS:,.0f} per unit"
        print(f"   {nm:52s} {c:10,.0f}{extra}")
    print(f"   {'total consumer':52s} {tot:10,.0f}")
