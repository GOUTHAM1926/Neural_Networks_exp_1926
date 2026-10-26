# 2026-10-19: PF vs PFT dense (work doc 45.56): both ncu source pages (SASS + executed count + stall samples per instruction)
# aligned by opcode sequence (difflib); per aligned block, time per unit = share of the kernel's samples x the kernel's SM
# cycles / units (4608 units over 132 SMs), so the two kernels' totals differ by exactly their cycle difference. Shows where
# PFT's extra cycles sit and by which stall reason.
# usage (from profiling_reports/): python3 session_2026_10_19/deep/align_pf_pft.py [min cycles]
import csv, re, sys, difflib, collections
B = "session_2026_10_19/box_runs/"
UNITS = 4608 / 132
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
def load(base, tag="gqa_v62"):
    rows = list(csv.reader(open(base + ".source.csv"))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
    body = [r for r in rows[k + 2:] if len(r) >= len(h)]
    RS = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
    raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
    cyc = f([dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
    ins = []
    for r in body:
        st = {c[6:]: f(r[idx[c]]) for c in RS}; allx = f(r[idx["Warp Stall Sampling (All Samples)"]])
        st["gmma"] = max(0.0, allx - sum(st.values()))
        src = r[idx["Source"]].strip(); op = re.sub(r"^@!?U?P[T0-9]+\s+", "", src).split()[0] if src else ""
        ins.append(dict(src=src, op=op, ex=int(f(r[idx["Instructions Executed"]])), smp=allx, st=st))
    tot = sum(x["smp"] for x in ins)
    for x in ins:
        x["cyc"] = x["smp"] / tot * cyc / UNITS; x["stc"] = {k: v / tot * cyc / UNITS for k, v in x["st"].items()}
    return ins, cyc
A, ca = load(B + "pf_1019/ncu_vj2_PF_dense"); P, cp = load(B + "pft_1019/ncu_vj2_PFT_dense")
print(f"per unit: PF {ca / UNITS:,.0f}  PFT {cp / UNITS:,.0f}  diff {(cp - ca) / UNITS:+,.0f} cycles")
sm = difflib.SequenceMatcher(None, [x["op"] for x in A], [x["op"] for x in P], autojunk=False)
mn = float(sys.argv[1]) if len(sys.argv) > 1 else 15
blocks = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    a = sum(x["cyc"] for x in A[i1:i2]); p = sum(x["cyc"] for x in P[j1:j2])
    blocks.append((tag, i1, i2, j1, j2, a, p))
print(f"{'tag':8s} {'PF range':>11s} {'PFT range':>11s} {'PF cyc':>8s} {'PFT cyc':>8s} {'diff':>7s}  where")
for tag, i1, i2, j1, j2, a, p in blocks:
    if abs(p - a) < mn: continue
    ex = A[i1]["ex"] if i2 > i1 else P[j1]["ex"]
    ra = collections.Counter(); rp = collections.Counter()
    for x in A[i1:i2]: ra.update(x["stc"])
    for x in P[j1:j2]: rp.update(x["stc"])
    dr = sorted(((k, rp[k] - ra[k]) for k in set(ra) | set(rp)), key=lambda kv: -abs(kv[1]))[:3]
    ops = collections.Counter(x["op"].split(".")[0] for x in (P[j1:j2] if j2 > j1 else A[i1:i2])).most_common(4)
    print(f"{tag:8s} {i1:5d}-{i2:<5d} {j1:5d}-{j2:<5d} {a:8.0f} {p:8.0f} {p - a:+7.0f}  ex {ex:>7,} | " + ", ".join(f"{k} {v:+.0f}" for k, v in dr) + " | " + " ".join(f"{o}x{n}" for o, n in ops))
print(f"sum of shown + hidden: {sum(p - a for *_, a, p in blocks):+.0f}")
