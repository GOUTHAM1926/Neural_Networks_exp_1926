# 2026-10-17: the per-unit lines (run once per unit per consumer warp) cut into blocks at the kernel's waits, with cycles
# (sample share x kernel cycles), instructions per warp per unit, and the main stalls. usage: unit_blocks.py <src> <raw> <substr> [units] [loop]
import csv, sys
from collections import Counter
src, raw, want = sys.argv[1:4]; units = float(sys.argv[4]) if len(sys.argv) > 4 else 36864
rows = list(csv.reader(open(src))); blocks = []; cur = None
for r in rows:
    if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
    if cur is not None: cur["rows"].append(r)
blk = next(b for b in blocks if want in b["name"]); h = blk["rows"][0]
st = [k for k in h if k.startswith("stall_") and "Not Issued" not in k]
L = []
for k, r in enumerate(blk["rows"][1:]):
    if len(r) <= h.index("Instructions Executed"): continue
    f = lambda c: float((r[h.index(c)] or "0").replace(",", ""))
    L.append(dict(k=k, src=r[h.index("Source")].strip(), ex=f("Instructions Executed"), smp=f("Warp Stall Sampling (All Samples)"), st={s: f(s) for s in st}))
R0 = list(csv.reader(open(raw)))
cyc = next(float(dict(zip(R0[0], rr))["gpc__cycles_elapsed.max"].replace(",", "")) for rr in R0[2:] if want in dict(zip(R0[0], rr))["Kernel Name"])
tot = sum(x["smp"] for x in L)
from collections import Counter as C2
hg = C2(round(x["ex"], -3) for x in L if "HGMMA" in x["src"]); n = max(hg, key=lambda v: (hg[v], v))
hot = [x["k"] for x in L if abs(x["ex"] - n) / n < 0.02]; lo, hi = hot[0], hot[-1]
U = [x for x in L if not (lo <= x["k"] <= hi) and abs(x["ex"] - units) / units < 0.08]
LOOP = len(sys.argv) > 5 and sys.argv[5] == "loop"
if LOOP: U = [x for x in L if lo <= x["k"] <= hi]
# cut: a new block after every wait-like instruction (DEPBAR, SYNCS try-wait loop branch, BAR) or a jump in line numbers > 40
out = []; b = []
for x in U:
    if b and x["k"] - b[-1]["k"] > 40: out.append(b); b = []
    b.append(x)
    if any(t in x["src"] for t in ("WARPGROUP.DEPBAR", "BAR.SYNC", "BAR.ARV", "SYNCS.PHASECHK")): out.append(b); b = []
if b: out.append(b)
print(f"== {blk['name'][:60]} cycles {cyc/1e6:.3f}M, loop {lo}-{hi}; per-unit total {sum(x['smp'] for x in U)/tot*cyc/1e6:.3f}M")
for b in out:
    s = sum(x["smp"] for x in b); c = C2()
    for x in b:
        for k2, v in x["st"].items(): c[k2] += v
    ops = C2()
    for x in b: ops[x["src"].split()[0] if not x["src"].startswith("@") else x["src"].split()[1]] += x["ex"] / (n if LOOP else units)
    print(f"  lines {b[0]['k']:5d}-{b[-1]['k']:5d} {'(after loop)' if b[0]['k'] > hi else '(before loop)':13s} {s/tot*cyc/1e3:6.1f}k cyc  {sum(x['ex'] for x in b) / (n if LOOP else units):6.1f} instr/{'tile' if LOOP else 'unit'} | "
          + ", ".join(f"{k2[6:]} {v/max(s,1)*100:.0f}%" for k2, v in c.most_common(2)) + " | " + " ".join(f"{o.split('.')[0]}x{v:.0f}" for o, v in ops.most_common(6)))
