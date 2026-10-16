# 2026-10-16: the forward's hot loop, instruction by instruction, from an ncu source CSV (--page source --print-source sass).
# Hot loop = the instructions executed once per (warp, key tile): the count of the loop's HGMMA lines.
# Prints: per-opcode counts per warp per key tile, the stall samples by reason, and the top instructions by samples.
# usage: python3 hotloop.py <source.csv> <kernel-name substring> [top]
import csv, sys, re
from collections import Counter, defaultdict
path, want = sys.argv[1], sys.argv[2]; top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = list(csv.reader(open(path)))
blocks = []; cur = None
for r in rows:
    if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
    if cur is not None: cur["rows"].append(r)
blk = next(b for b in blocks if want in b["name"])
h = blk["rows"][0]; data = blk["rows"][1:]
iS, iE, iW = h.index("Source"), h.index("Instructions Executed"), h.index("Warp Stall Sampling (All Samples)")
sc = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
L = []
for k, r in enumerate(data):
    if len(r) <= iE: continue
    src = r[iS].strip(); e = float(r[iE] or 0); w = float(r[iW] or 0)
    st = {c: float(r[h.index(c)] or 0) for c in sc}
    L.append((k, src, e, w, st))
tot_w = sum(x[3] for x in L)
hg = Counter(round(x[2], -3) for x in L if "HGMMA" in x[1])
loop_n = max(hg, key=lambda v: (hg[v], v))           # the most common HGMMA execution count
hot = [x for x in L if abs(x[2] - loop_n) / loop_n < 0.02]
lo, hi = hot[0][0], hot[-1][0]
region = [x for x in L if lo <= x[0] <= hi]
op = lambda s: re.sub(r"^@!?U?P\w+\s+", "", s).split()[0] if s else "?"
print(f"== {blk['name'][:90]}")
print(f"   loop count per line {loop_n:.0f}; hot-loop lines {len(hot)} (region {lo}..{hi}, {len(region)} lines)")
per = Counter(); per_s = Counter()
for x in region: per[op(x[1])] += x[2] / loop_n; per_s[op(x[1])] += x[3]
tot_reg = sum(per.values()); w_reg = sum(x[3] for x in region)
print(f"   instructions per warp per key tile (region): {tot_reg:.0f}; samples in region {100 * w_reg / tot_w:.0f}% of all")
print("   opcode                 per tile   samples%")
for o, n in per.most_common(top):
    print(f"   {o:22s} {n:8.1f}   {100 * per_s[o] / w_reg:6.1f}")
agg = Counter()
for x in region:
    for c, v in x[4].items(): agg[c] += v
print("   stall reasons in the region: " + "  ".join(f"{c[6:]} {100 * v / w_reg:.1f}%" for c, v in agg.most_common(10)))
print("   top instructions by samples:")
for x in sorted(region, key=lambda x: -x[3])[:top]:
    rs = max(x[4].items(), key=lambda kv: kv[1])
    print(f"   line {x[0]:5d}  {100 * x[3] / w_reg:5.2f}%  ({rs[0][6:]})  {x[1][:80]}")
