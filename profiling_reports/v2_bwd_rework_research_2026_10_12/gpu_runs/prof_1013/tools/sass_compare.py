#!/usr/bin/env python3
# Per-opcode comparison of ncu source pages (SASS), 2026-10-13. Each input = <file>[:<section index>] where a
# multi-kernel export holds one "Kernel Name" section per kernel. Prints, per opcode family: executed
# warp-instructions (millions) and the share of all stall samples on those lines; then the top stall reasons.
import csv, re, sys
from collections import Counter, defaultdict
def sections(path):
    rows = list(csv.reader(open(path)))
    idx = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    out = []
    for k, i in enumerate(idx):
        j = idx[k + 1] if k + 1 < len(idx) else len(rows)
        out.append((rows[i][1], rows[i + 1], rows[i + 2:j]))
    return out
def load(spec):
    p, _, s = spec.partition(":")
    name, h, R = sections(p)[int(s or 0)]
    ix = {c: i for i, c in enumerate(h)}
    return name, h, ix, R
def fam(src):
    s = re.sub(r"^@!?U?P\w+\s+", "", src.strip()); op = s.split(" ")[0] if s else ""
    base = op.split(".")[0]
    groups = {"HGMMA": "HGMMA", "MUFU": "MUFU (exp)", "FFMA": "FFMA", "FMUL": "FMUL", "FADD": "FADD", "FMNMX": "FMNMX (max)",
              "F2FP": "F2FP (cvt bf16)", "SHFL": "SHFL", "BAR": "BAR", "SYNCS": "SYNCS (mbarrier)", "LDS": "LDS", "STS": "STS",
              "STSM": "STSM", "LDSM": "LDSM", "LDG": "LDG", "STG": "STG", "LD": "LD (generic)", "ST": "ST (generic)",
              "UTMALDG": "TMA load", "UTMASTG": "TMA store", "UBLKCP": "bulk copy", "WARPGROUP": "WARPGROUP (arrive/depbar)",
              "BRA": "BRA", "FSEL": "FSEL", "FSETP": "FSETP", "ISETP": "ISETP", "IMAD": "IMAD", "IADD3": "IADD3", "LOP3": "LOP3",
              "R2UR": "R2UR", "MOV": "MOV", "IMNMX": "IMNMX", "SEL": "SEL", "LEA": "LEA", "SHF": "SHF"}
    if base.startswith("U") and base not in groups: return "uniform (U*)"
    return groups.get(base, "other")
specs = sys.argv[1:]
data = []
for sp in specs:
    name, h, ix, R = load(sp)
    g = lambda r, c: float(r[ix[c]]) if c in ix and r[ix[c]] not in ("", "-") else 0.0
    tot = sum(g(r, "Warp Stall Sampling (All Samples)") for r in R)
    ex, smp, reas = Counter(), Counter(), defaultdict(Counter)
    for r in R:
        f = fam(r[ix["Source"]]); e = g(r, "Instructions Executed"); s = g(r, "Warp Stall Sampling (All Samples)")
        ex[f] += e; smp[f] += s; ex["(all)"] += e; smp["(all)"] += s
        for c in h:
            if c.startswith("stall_") and "Not Issued" not in c: reas[f][c[6:]] += g(r, c)
    data.append((sp, name, tot, ex, smp, reas))
fams = sorted({f for d in data for f in d[3]}, key=lambda f: -max(d[3][f] for d in data))
print("executed warp-instructions (M) | stall-sample share %")
print(f"{'family':26s}" + "".join(f"{d[0].split('/')[-1][:22]:>24s}" for d in data))
for f in fams:
    if max(d[3][f] for d in data) < 0.2e6 and max(100 * d[4][f] / d[2] for d in data) < 0.5: continue
    print(f"{f:26s}" + "".join(f"{d[3][f]/1e6:12.2f} {100*d[4][f]/d[2]:9.1f}%  " for d in data))
for d in data:
    print(f"\n{d[0]}: {d[1][:90]}")
    for f in sorted(d[4], key=lambda f: -d[4][f])[:7]:
        if f == "(all)": continue
        top = ", ".join(f"{k} {100*v/d[2]:.1f}" for k, v in d[5][f].most_common(3))
        print(f"   {f:26s} {100*d[4][f]/d[2]:5.1f}%  [{top}]")
