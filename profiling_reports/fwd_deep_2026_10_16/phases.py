# 2026-10-16: split the forward's hot loop into phases (issue GEMMs | wait QK | between the waits | wait P.V | after the
# waits) and sum instructions per tile and stall samples per phase. Phase borders are the loop's two WARPGROUP.DEPBAR.LE
# lines (wait_group 1 = QK done, wait_group 0 = P.V done) and its last HGMMA.
# usage: python3 phases.py <source.csv> <kernel substring>
import csv, sys, re
from collections import Counter
path, want = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(path))); blocks = []; cur = None
for r in rows:
    if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
    if cur is not None: cur["rows"].append(r)
blk = next(b for b in blocks if want in b["name"])
h = blk["rows"][0]; data = blk["rows"][1:]
iS, iE, iW = h.index("Source"), h.index("Instructions Executed"), h.index("Warp Stall Sampling (All Samples)")
sc = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
L = [(k, r[iS].strip(), float(r[iE] or 0), float(r[iW] or 0), {c: float(r[h.index(c)] or 0) for c in sc})
     for k, r in enumerate(data) if len(r) > iE]
hg = Counter(round(x[2], -3) for x in L if "HGMMA" in x[1]); n = max(hg, key=lambda v: (hg[v], v))
hot = [x for x in L if abs(x[2] - n) / n < 0.02]; lo, hi = hot[0][0], hot[-1][0]
reg = [x for x in L if lo <= x[0] <= hi]
tot = sum(x[3] for x in L)
d1 = next(x[0] for x in reg if "DEPBAR.LE gsb0, 0x1" in x[1]); d0 = next(x[0] for x in reg if "DEPBAR.LE gsb0, 0x0" in x[1] and x[0] > d1)
lastg = max(x[0] for x in reg if "HGMMA" in x[1] and x[0] < d1)
ph = {"1 issue: baton, K/V wait, QK + P.V HGMMA": (lo, lastg), "2 wait for QK": (lastg + 1, d1), "3 between the waits": (d1 + 1, d0 - 1),
      "4 wait for P.V": (d0, d0), "5 after the waits (to loop end)": (d0 + 1, hi)}
print(f"== {blk['name'][:70]}: loop lines {lo}..{hi}; loop = {100 * sum(x[3] for x in reg) / tot:.0f}% of all samples")
for name, (a, b) in ph.items():
    xs = [x for x in reg if a <= x[0] <= b]
    ins = sum(x[2] for x in xs) / n; smp = sum(x[3] for x in xs)
    ops = Counter(re.sub(r"^@!?U?P\w+\s+", "", x[1]).split()[0] for x in xs if x[1])
    st = Counter()
    for x in xs:
        for c, v in x[4].items(): st[c] += v
    top = ", ".join(f"{o} {sum(x[2] for x in xs if re.sub(r'^@!?U?P\w+\s+', '', x[1]).startswith(o)) / n:.0f}" for o, _ in ops.most_common(5))
    print(f"  {name:38s} instr/tile {ins:5.0f}  samples {100 * smp / tot:5.1f}%  [{', '.join(f'{c[6:]} {100 * v / max(smp, 1):.0f}%' for c, v in st.most_common(3))}]")
    print(f"      ops: {top}")
