# 2026-10-19: PF vs PFT dense by REGION (work doc 45.56): a region = a maximal run of instructions whose executed counts are
# within 2x of the run's first (as kernel_map.py; tiny runs of < 4 instructions are merged into the previous region); the two kernels' regions are paired in order. Per
# region: instructions, executed count, time per unit (share of samples x SM cycles / units) and the stall reasons that moved.
import csv, re, sys, collections
exec(open("session_2026_10_19/deep/align_pf_pft.py").read().split("A, ca = load")[0])
A, ca = load(B + "pf_1019/ncu_vj2_PF_dense"); P, cp = load(B + "pft_1019/ncu_vj2_PFT_dense")
def regions(ins):
    out = []
    for i, x in enumerate(ins):
        if out and out[-1]["ex"] / 2 <= x["ex"] <= out[-1]["ex"] * 2:   # kernel_map.py's rule: within 2x of the run's first
            out[-1]["hi"] = i
        else:
            out.append(dict(lo=i, hi=i, ex=x["ex"]))
    m = []
    for r in out:   # merge runs shorter than 4 instructions into the previous region
        if m and r["hi"] - r["lo"] + 1 < 4: m[-1]["hi"] = r["hi"]
        else: m.append(dict(r))
    for r in m:
        seg = ins[r["lo"]:r["hi"] + 1]; r["cyc"] = sum(x["cyc"] for x in seg); r["st"] = collections.Counter()
        for x in seg: r["st"].update(x["stc"])
        r["ops"] = collections.Counter(x["op"].split(".")[0] for x in seg)
    return m
RA, RP = regions(A), regions(P)
print(f"regions: PF {len(RA)}, PFT {len(RP)}; per unit PF {ca / UNITS:,.0f} PFT {cp / UNITS:,.0f} ({(cp - ca) / UNITS:+,.0f})")
import difflib
sm = difflib.SequenceMatcher(None, [r["ex"] for r in RA], [r["ex"] for r in RP], autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    a = RA[i1:i2]; p = RP[j1:j2]
    ac = sum(r["cyc"] for r in a); pc = sum(r["cyc"] for r in p)
    if abs(pc - ac) < 10 and max(ac, pc) < 300: continue
    sa = collections.Counter(); sp = collections.Counter()
    for r in a: sa.update(r["st"])
    for r in p: sp.update(r["st"])
    dr = sorted(((k, sp[k] - sa[k]) for k in set(sa) | set(sp)), key=lambda kv: -abs(kv[1]))[:4]
    ra = f"{a[0]['lo']}-{a[-1]['hi']}" if a else "-"; rp = f"{p[0]['lo']}-{p[-1]['hi']}" if p else "-"
    ex = (a or p)[0]["ex"]; ops = (p or a)[0]["ops"]
    big = [o for o in ("HGMMA", "MUFU", "BAR", "SYNCS", "UTMASTG", "UTMALDG", "STG", "LDS", "STSM", "FENCE") if any(r["ops"][o] for r in (p or a))]
    print(f"{tag:8s} PF {ra:>11s} PFT {rp:>11s} ex {ex:>8,} | PF {ac:7.0f} PFT {pc:7.0f} {pc - ac:+6.0f} | " + ", ".join(f"{k} {v:+.0f}" for k, v in dr) + " | " + " ".join(f"{o}{sum(r['ops'][o] for r in (p or a))}" for o in big))
