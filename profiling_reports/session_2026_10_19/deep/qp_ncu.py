# 2026-10-19 (work doc 45.59): PF vs the QP candidates from ONE ncu session on the 8x node (box_runs/qp_1019): per kernel the SM
# cycles, and (dense: per unit = cycles / 34.9; packed: per SM) the regions of a unit: consumer code before the steady loop
# (unit start + first tile), the steady loop (per step), consumer code after it (last steps + epilogue), the producer warp
# (from its USETMAXREG.DEALLOC), and the consumers' wait for Q (mbarrier qfull 0x38030 -> its retry branch).
# usage (from profiling_reports/): python3 session_2026_10_19/deep/qp_ncu.py
import re, collections
exec(open("session_2026_10_19/deep/waits_fa3_pf.py").read().split("KERN = {")[0])
B2 = "session_2026_10_19/box_runs/qp_1019/"
LIBS = ["PF", "QP", "QPU", "QPD", "QPUD"]
def regions(ins, cyc, per):
    by = collections.Counter()
    for x in ins:
        if x["ex"] >= 100000: by[x["ex"]] += x["smp"]
    lx = by.most_common(1)[0][0]
    lo = min(i for i, x in enumerate(ins) if x["ex"] == lx); hi = max(i for i, x in enumerate(ins) if x["ex"] == lx)
    prod = next(i for i, x in enumerate(ins) if "USETMAXREG.DEALLOC" in x["src"])
    first = next(i for i, x in enumerate(ins) if x["ex"] >= 30000)
    tot = sum(x["smp"] for x in ins)
    c = lambda a, b: sum(x["smp"] for x in ins[a:b]) / tot * cyc / per
    q = 0.0
    for i, x in enumerate(ins):
        if "TRYWAIT" in x["src"] and "+0x38030]" in x["src"] and x["ex"] > 0:
            p = re.search(r"TRYWAIT (P\d)", x["src"]).group(1)
            j = next((j for j in range(i + 1, min(i + 60, len(ins))) if re.search(r"@!?" + p + r"\s+BRA", ins[j]["src"])), i)
            q += c(i, j + 1)
    return dict(pre=c(first, lo), loop=c(lo, hi + 1), steps=lx, post=c(hi + 1, prod), prod=c(prod, len(ins)), q=q)
for shape, per, label in (("dense", 4608 / 132, "cycles per unit"), ("packed", 1.0, "cycles per SM")):
    print(f"===== {shape} ({label})")
    print(f"{'':6s} {'kernel cyc/SM':>14s} {'vs PF':>7s} | {'pre-loop':>9s} {'loop':>9s} {'per step':>9s} {'post-loop':>9s} {'producer':>9s} | {'Q wait':>7s}")
    base = None
    for n in LIBS:
        ins, cyc = load(B2 + f"ncu_vj2_{n}_{shape}", "gqa_v62")
        r = regions(ins, cyc, per)
        steps = r["steps"] / (4608 * 8) if shape == "dense" else r["steps"] / 8
        if base is None: base = cyc
        print(f"{n:6s} {cyc:14,.0f} {100 * (cyc / base - 1):+6.2f}% | {r['pre']:9.0f} {r['loop']:9.0f} {r['loop'] / steps if shape == 'dense' else 0:9.1f} {r['post']:9.0f} {r['prod']:9.0f} | {r['q']:7.0f}")
