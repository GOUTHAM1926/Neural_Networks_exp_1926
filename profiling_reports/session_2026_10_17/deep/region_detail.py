# 2026-10-17: where the samples outside / inside the forward's hot loop sit, line by line, for ours vs FA3 (same ncu, same box).
# Regions as compare_versions.py: hot loop = lines run as often as the loop's HGMMA; "per unit" = lines run once per 128-row
# unit per consumer warp (dense B8 18/6: 4608 units x 8 warps = 36864, +-8%); rest = loader / setup.
# Prints per region: samples -> M cycles (share x kernel cycles), the stall reasons, the top lines with their main stall.
# usage: python3 region_detail.py <source.csv> <raw.csv> <kernel substring> [units=36864] [top=25] [region=unit|loop|rest|all]
import csv, sys, re
from collections import Counter
src, raw, want = sys.argv[1:4]
units = float(sys.argv[4]) if len(sys.argv) > 4 else 36864; top = int(sys.argv[5]) if len(sys.argv) > 5 else 25
which = sys.argv[6] if len(sys.argv) > 6 else "all"
rows = list(csv.reader(open(src))); blocks = []; cur = None
for r in rows:
    if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
    if cur is not None: cur["rows"].append(r)
blk = next(b for b in blocks if want in b["name"]); h = blk["rows"][0]
ix = {k: h.index(k) for k in ("Source", "Instructions Executed", "Warp Stall Sampling (All Samples)")}
st = [k for k in h if k.startswith("stall_") and "Not Issued" not in k]
L = []
for k, r in enumerate(blk["rows"][1:]):
    if len(r) <= ix["Instructions Executed"]: continue
    f = lambda c: float((r[h.index(c)] or "0").replace(",", ""))
    L.append(dict(k=k, src=r[ix["Source"]].strip(), ex=f("Instructions Executed"), smp=f("Warp Stall Sampling (All Samples)"),
                  st={s: f(s) for s in st}))
for rr in list(csv.reader(open(raw)))[2:]:
    d = dict(zip(list(csv.reader(open(raw)))[0], rr))
    if want in d["Kernel Name"]: cyc = float(d["gpc__cycles_elapsed.max"].replace(",", "")); break
tot = sum(x["smp"] for x in L)
hg = Counter(round(x["ex"], -3) for x in L if "HGMMA" in x["src"]); n = max(hg, key=lambda v: (hg[v], v))
hot = [x["k"] for x in L if abs(x["ex"] - n) / n < 0.02]; lo, hi = hot[0], hot[-1]
reg = {"loop": [x for x in L if lo <= x["k"] <= hi],
       "unit": [x for x in L if not (lo <= x["k"] <= hi) and abs(x["ex"] - units) / units < 0.08],
       "rest": [x for x in L if not (lo <= x["k"] <= hi) and abs(x["ex"] - units) / units >= 0.08]}
print(f"== {blk['name'][:70]}  cycles {cyc/1e6:.3f}M  loop lines {lo}-{hi} run {n:.0f}x")
for name, R in reg.items():
    if which not in ("all", name): continue
    s = sum(x["smp"] for x in R); c = Counter()
    for x in R:
        for k2, v in x["st"].items(): c[k2] += v
    ins = sum(x["ex"] for x in R)
    print(f"-- {name}: {s/tot*cyc/1e6:.3f}M cycles ({s/tot*100:.1f}% of samples), {len(R)} lines, {ins/1e6:.2f}M warp-instr | stalls: " +
          ", ".join(f"{k2[6:]} {v/s*100:.0f}%" for k2, v in c.most_common(6)))
    for x in sorted(R, key=lambda x: -x["smp"])[:top]:
        m = max(x["st"], key=x["st"].get)
        print(f"   {x['k']:5d} {x['smp']/tot*cyc/1e3:7.1f}k  x{x['ex']:>9.0f}  {m[6:]:>12s}  {x['src'][:70]}")
