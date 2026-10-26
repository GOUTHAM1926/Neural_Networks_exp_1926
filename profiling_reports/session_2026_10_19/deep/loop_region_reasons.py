# 2026-10-19: the stall REASONS inside the issue, softmax-A and rest regions of one steady loop step (see loop_wait_split.py;
# work doc 45.51), in cycles per step (share x the traced period), plus the instruction mix. Reasons (ncu): selected = issuing;
# not_selected = eligible, but the scheduler issued another warp (the other group, the producer); math = pipe busy (MUFU / tensor
# issue throttle); short_sb = waiting for a MUFU / shuffle / shared result; wait = fixed-latency dependency; dispatch = dispatch
# stall; no_inst = instruction fetch; branch_resolving; mio = memory-I/O queue; barrier / long_sb as in loop_wait_split.py.
# usage: python3 session_2026_10_19/deep/loop_region_reasons.py <name> <source.csv> <kernel tag> <period>
import csv, re, sys, collections
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
name, path, tag, period = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rows = list(csv.reader(open(path))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
body = [r for r in rows[k + 2:] if len(r) >= len(h)]
RS = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
src = [r[idx["Source"]].strip() for r in body]; ex = [int(f(r[idx["Instructions Executed"]])) for r in body]
smp = [f(r[idx["Warp Stall Sampling (All Samples)"]]) for r in body]
opc = [re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else "" for s in src]
by_ex = collections.Counter()
for i in range(len(body)):
    if ex[i] >= 1000: by_ex[ex[i]] += smp[i]
lx = by_ex.most_common(1)[0][0]
lo = min(i for i in range(len(body)) if ex[i] == lx); hi = max(i for i in range(len(body)) if ex[i] == lx)
L = range(lo, hi + 1); tot = sum(smp[i] for i in L)
dep1 = next(i for i in L if opc[i].startswith("WARPGROUP.DEPBAR") and src[i].rstrip(" ;").endswith("0x1"))
dep0 = next(i for i in L if i > dep1 and opc[i].startswith("WARPGROUP.DEPBAR") and src[i].rstrip(" ;").endswith("0x0"))
last_hg = max(i for i in L if i < dep1 and opc[i].startswith("HGMMA"))
regs = {"issue": range(lo, last_hg + 1), "softmax A": range(dep1 + 1, dep0), "rest": range(dep0 + 1, hi + 1)}
print(f"== {name} (period {period:.0f}; cycles per step, both groups' average)")
for rn, rg in regs.items():
    c = collections.Counter()
    for i in rg:
        for col in RS: c[col[6:]] += f(body[i][idx[col]])
    mix = collections.Counter(opc[i].split(".")[0] for i in rg)
    cyc = {kk: v / tot * period for kk, v in c.items()}
    top = sorted(cyc.items(), key=lambda x: -x[1])[:8]
    print(f"   {rn:10s} {len(rg):3d} instr, {sum(cyc.values()):5.0f} cycles: " + ", ".join(f"{kk} {v:.0f}" for kk, v in top))
    print(f"   {'':10s} mix: " + ", ".join(f"{kk} {v}" for kk, v in mix.most_common(9)))
