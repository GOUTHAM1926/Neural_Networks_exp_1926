# 2026-10-16: the softmax part of the forward loop (after wait-for-QK to loop end), per opcode: count per tile, stall samples
# per executed instruction (cycles the warp spent stalled at it, relative), and the main stall reason.
import csv, sys, re
from collections import Counter, defaultdict
path, want = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(path))); blocks = []; cur = None
for r in rows:
    if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
    if cur is not None: cur["rows"].append(r)
blk = next(b for b in blocks if want in b["name"])
h = blk["rows"][0]; data = blk["rows"][1:]
iS, iE, iW = h.index("Source"), h.index("Instructions Executed"), h.index("Warp Stall Sampling (All Samples)")
sc = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
L = [(k, r[iS].strip(), float(r[iE] or 0), float(r[iW] or 0), {c: float(r[h.index(c)] or 0) for c in sc}) for k, r in enumerate(data) if len(r) > iE]
hg = Counter(round(x[2], -3) for x in L if "HGMMA" in x[1]); n = max(hg, key=lambda v: (hg[v], v))
hot = [x for x in L if abs(x[2] - n) / n < 0.02]; lo, hi = hot[0][0], hot[-1][0]
reg = [x for x in L if lo <= x[0] <= hi]; tot = sum(x[3] for x in L)
d1 = next(x[0] for x in reg if "DEPBAR.LE gsb0, 0x1" in x[1])
sm = [x for x in reg if x[0] > d1 and "DEPBAR" not in x[1]]
by = defaultdict(lambda: [0.0, 0.0, Counter()])
for x in sm:
    o = re.sub(r"^@!?U?P\w+\s+", "", x[1]).split()[0]
    by[o][0] += x[2] / n; by[o][1] += x[3]
    for c, v in x[4].items(): by[o][2][c] += v
S = sum(v[1] for v in by.values())
print(f"== {blk['name'][:60]}: softmax part {sum(v[0] for v in by.values()):.0f} instr/tile, {100 * S / tot:.1f}% of all samples")
print("   opcode                  per tile   share of softmax samples   samples per instruction   main stalls")
for o, (c, s, st) in sorted(by.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"   {o:24s} {c:6.1f}   {100 * s / S:6.1f}%                    {s / max(c, 1e-9) / (S / sum(v[0] for v in by.values())):5.2f}x avg      " + ", ".join(f"{k[6:]} {100 * v / max(s, 1):.0f}%" for k, v in st.most_common(3)))
