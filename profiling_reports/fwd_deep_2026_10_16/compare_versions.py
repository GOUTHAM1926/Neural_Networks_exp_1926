# 2026-10-16: two forward profiles side by side, converted to cycles (share of samples x the kernel's cycles):
# the loop phases (phases.py's borders), and outside the loop: consumer unit start / unit end (lines run once per
# 128-row unit per consumer warp), loader / other. Usage: compare_versions.py <label:source.csv:raw.csv:substring> ...
import csv, sys, re
from collections import Counter
def load_src(path, want):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    blk = next(b for b in blocks if want in b["name"]); h = blk["rows"][0]
    iS, iE, iW = h.index("Source"), h.index("Instructions Executed"), h.index("Warp Stall Sampling (All Samples)")
    return [(k, r[iS].strip(), float(r[iE] or 0), float(r[iW] or 0)) for k, r in enumerate(blk["rows"][1:]) if len(r) > iE]
def cycles(raw, want):
    rows = list(csv.reader(open(raw)))
    for r in rows[2:]:
        d = dict(zip(rows[0], r))
        if want in d["Kernel Name"]: return float(d["gpc__cycles_elapsed.max"].replace(",", ""))
out = {}
for spec in sys.argv[1:]:
    lab, src, raw, want = spec.split(":")
    L = load_src(src, want); cyc = cycles(raw, want) / 1e6; tot = sum(x[3] for x in L)
    hg = Counter(round(x[2], -3) for x in L if "HGMMA" in x[1]); n = max(hg, key=lambda v: (hg[v], v))
    hot = [x[0] for x in L if abs(x[2] - n) / n < 0.02]; lo, hi = hot[0], hot[-1]
    reg = [x for x in L if lo <= x[0] <= hi]
    d1 = next(x[0] for x in reg if "DEPBAR.LE gsb0, 0x1" in x[1]); d0 = next(x[0] for x in reg if "DEPBAR.LE gsb0, 0x0" in x[1] and x[0] > d1)
    lastg = max(x[0] for x in reg if "HGMMA" in x[1] and x[0] < d1)
    units = 36864
    def S(sel): return sum(x[3] for x in L if sel(x)) / tot * cyc
    out[lab] = {
        "total": cyc,
        "loop: issue": S(lambda x: lo <= x[0] <= lastg),
        "loop: wait QK": S(lambda x: lastg < x[0] <= d1),
        "loop: between waits": S(lambda x: d1 < x[0] < d0),
        "loop: wait P.V": S(lambda x: x[0] == d0),
        "loop: after waits": S(lambda x: d0 < x[0] <= hi),
        "unit start": S(lambda x: x[0] < lo and abs(x[2] - units) / units < 0.08),
        "unit end": S(lambda x: x[0] > hi and abs(x[2] - units) / units < 0.08),
        "loader / other": S(lambda x: not (lo <= x[0] <= hi) and abs(x[2] - units) / units >= 0.08),
    }
labs = list(out)
print("M cycles".ljust(22) + "".join(l.rjust(10) for l in labs) + ("   " + labs[1] + " - " + labs[0] if len(labs) > 1 else ""))
for k in out[labs[0]]:
    vals = [out[l][k] for l in labs]
    print(k.ljust(22) + "".join(f"{v:10.3f}" for v in vals) + (f"   {vals[1] - vals[0]:+.3f}" if len(vals) > 1 else ""))
