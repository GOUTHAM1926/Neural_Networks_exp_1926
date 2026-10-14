#!/usr/bin/env python3
# 2026-10-14 test 1.4: time per instruction CATEGORY (stall-sample share x kernel time), ours (Vj2 g123) vs FA3 backward,
# dense B8 18/6. Every instruction of the kernel is counted (loop and non-loop), split by which warps run it:
# producer warps (loader / dQ writer: they wait on purpose) vs compute warps.
import csv, re, sys
from collections import Counter, defaultdict
def sections(path):
    rows = list(csv.reader(open(path))); idx = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    out = []
    for k, i in enumerate(idx):
        j = idx[k + 1] if k + 1 < len(idx) else len(rows); out.append((rows[i][1], rows[i + 1], rows[i + 2:j]))
    return out
CAT = [("HGMMA", "GEMM issue (HGMMA)"), ("WARPGROUP.ARRIVE", "GEMM issue (HGMMA)"), ("WARPGROUP.DEPBAR", "GEMM wait (DEPBAR)"),
       ("MUFU", "exp (MUFU)"), ("FFMA", "float math"), ("FADD", "float math"), ("FMUL", "float math"), ("FSEL", "float math"), ("FMNMX", "float math"),
       ("F2FP", "bf16 pack"), ("LDS", "smem load"), ("LDSM", "smem load"), ("STSM", "smem store"), ("STS", "smem store"),
       ("FENCE", "fence"), ("MEMBAR", "fence"), ("BAR", "named barrier"), ("SYNCS", "mbarrier wait/arrive"),
       ("UBLKRED", "TMA / bulk copy"), ("UBLKCP", "TMA / bulk copy"), ("UTMA", "TMA / bulk copy"),
       ("BRA", "branch / reconverge"), ("BSSY", "branch / reconverge"), ("BSYNC", "branch / reconverge"),
       ("R2UR", "address / descriptor / int"), ("IMAD", "address / descriptor / int"), ("LOP3", "address / descriptor / int"),
       ("LEA", "address / descriptor / int"), ("SHF", "address / descriptor / int"), ("IADD3", "address / descriptor / int"),
       ("ISETP", "address / descriptor / int"), ("VIADD", "address / descriptor / int"), ("SEL", "address / descriptor / int")]
def cat(src):
    s = re.sub(r"^@!?U?P\w+\s+", "", src.strip()); op = s.split(" ")[0] if s else ""
    if op.startswith("U") and not op.startswith(("UBLK", "UTMA")): return "uniform datapath (U*)"
    for k, v in CAT:
        if op.startswith(k): return v
    return "other"
def analyse(name, h, R, t_us):
    ix = {c: i for i, c in enumerate(h)}
    g = lambda r, c: float(r[ix[c]]) if r[ix[c]] not in ("", "-") else 0.0
    tot = sum(g(r, "Warp Stall Sampling (All Samples)") for r in R)
    loop = Counter(int(g(r, "Instructions Executed")) for r in R if "HGMMA" in r[ix["Source"]]).most_common(1)[0][0]
    us = defaultdict(float); us_prod = 0.0
    for r in R:
        e = g(r, "Instructions Executed"); s = g(r, "Warp Stall Sampling (All Samples)")
        if 0 < e < loop / 4 and e * 8 >= loop / 4 and not "HGMMA" in r[ix["Source"]]:
            pass
        us[cat(r[ix["Source"]])] += s / tot * t_us
    return us, tot, loop
src = {}
ours = sections(sys.argv[1])[0]; fa3 = sections(sys.argv[2])[1]
t_ours, t_fa3 = float(sys.argv[3]), float(sys.argv[4])
uo, to, lo = analyse("ours", ours[1], ours[2], t_ours); uf, tf, lf = analyse("fa3", fa3[1], fa3[2], t_fa3)
print(f"time per instruction category = share of stall samples x kernel time (ours {t_ours:.0f} us, FA3 {t_fa3:.0f} us)")
print(f"{'category':30s} {'ours us':>9s} {'FA3 us':>9s} {'ours - FA3':>11s}")
for k in sorted(set(uo) | set(uf), key=lambda k: -(uo.get(k, 0) - uf.get(k, 0))):
    print(f"{k:30s} {uo.get(k,0):9.1f} {uf.get(k,0):9.1f} {uo.get(k,0)-uf.get(k,0):+11.1f}")
print(f"{'TOTAL':30s} {sum(uo.values()):9.1f} {sum(uf.values()):9.1f} {sum(uo.values())-sum(uf.values()):+11.1f}")
