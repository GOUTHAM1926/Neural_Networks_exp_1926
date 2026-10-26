# 2026-10-19: the unit-boundary regions of our dense forward (PF) vs FA3, by job, per unit (work doc 45.55): instructions
# executed per unit (executions / (units x 8 consumer warps)) and consumer time per unit (ncu PC sampling).
# FA3: start + first tile 0-1557 (its first tile = the masked diagonal), last P.V + epilogue 2027-2301.
# ours: start + first tile 0-1041 (tile 0, unmasked), last P.V + epilogue 2142-2706.
import csv, re, collections
exec(open("session_2026_10_19/deep/loop_diff_fa3.py").read().split("res = {}")[0])
UNITS = 4608 / 132
def load_k(base, tag, prod):
    ins = load(base + ".source.csv", tag)
    raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
    cyc = f([dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
    return ins, cyc, sum(x["smp"] for x in ins[:prod])
fa3 = load_k(B + "ncu_fa3_k5q_1019/fa3_dense", "FlashAttnFwdSm90", 2302)
ours = load_k(B + "pf_1019/ncu_vj2_PF_dense", "gqa_v62", 2707)
def table(k, a, b):
    ins, cyc, cons = k; agg = collections.defaultdict(lambda: [0.0, 0.0, collections.Counter()])
    for x in ins[a:b]:
        j = job(x["src"]); agg[j][0] += x["ex"] / (4608 * 8); agg[j][1] += x["smp"] / cons * cyc / UNITS
        for r, v in x["st"].items(): agg[j][2][r] += v / cons * cyc / UNITS
    return agg
for title, (fa, fb), (oa, ob) in (("start + first tile", (0, 1558), (0, 1042)), ("last P.V + epilogue", (2027, 2302), (2142, 2707))):
    F, O = table(fa3, fa, fb), table(ours, oa, ob)
    keys = sorted(set(F) | set(O), key=lambda k: -abs(O.get(k, [0, 0])[1] - F.get(k, [0, 0])[1]))
    print(f"===== {title} (per unit) | FA3: instr  cycles | ours: instr  cycles | diff | ours' top stalls")
    tf = to = 0
    for k in keys:
        a = F.get(k, [0, 0, collections.Counter()]); b = O.get(k, [0, 0, collections.Counter()]); tf += a[1]; to += b[1]
        if abs(b[1] - a[1]) < 15 and max(a[1], b[1]) < 60: continue
        print(f"   {k:46s} | {a[0]:6.1f} {a[1]:6.0f} | {b[0]:6.1f} {b[1]:6.0f} | {b[1] - a[1]:+6.0f} | " + ", ".join(f"{r} {v:.0f}" for r, v in b[2].most_common(3) if v >= 10))
    print(f"   {'total':46s} | {'':6s} {tf:6.0f} | {'':6s} {to:6.0f} | {to - tf:+6.0f}")
