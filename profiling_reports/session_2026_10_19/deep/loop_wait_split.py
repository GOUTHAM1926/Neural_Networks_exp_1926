# 2026-10-19: one steady loop step of an FA3-style forward, split by ncu's stall REASONS (work doc 45.51). The named-barrier
# (baton) wait is deferred (BAR.SYNC.DEFER_BLOCKING) and shows as "barrier" samples on later instructions, so a split by
# position mislabels it. Categories, over the steady loop (instructions run once per step per warp, plus the group-only ones
# in between): baton = all "barrier" samples | K/V wait = samples on the TRYWAIT and the branch right after it | QK wait = the
# WARPGROUP.DEPBAR ... 0x1 | P.V wait = the DEPBAR ... 0x0 | issue = the other samples from the loop top to the last HGMMA |
# softmax A = between the two DEPBARs | rest = after the P.V wait. Cycles = share x the period of the level-2 timeline trace;
# both groups' warps run this code, so each figure is the two groups' average.
# usage (from profiling_reports/): python3 session_2026_10_19/deep/loop_wait_split.py <name> <source.csv> <kernel tag> <period>
import csv, re, sys, collections
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
name, path, tag, period = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rows = list(csv.reader(open(path))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
body = [r for r in rows[k + 2:] if len(r) >= len(h)]
src = [r[idx["Source"]].strip() for r in body]; ex = [int(f(r[idx["Instructions Executed"]])) for r in body]
smp = [f(r[idx["Warp Stall Sampling (All Samples)"]]) for r in body]; bar = [f(r[idx["stall_barrier"]]) for r in body]
opc = [re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else "" for s in src]
by_ex = collections.Counter()
for i in range(len(body)):
    if ex[i] >= 1000: by_ex[ex[i]] += smp[i]
lx = by_ex.most_common(1)[0][0]
lo = min(i for i in range(len(body)) if ex[i] == lx); hi = max(i for i in range(len(body)) if ex[i] == lx)
L = range(lo, hi + 1)
dep1 = next(i for i in L if opc[i].startswith("WARPGROUP.DEPBAR") and src[i].rstrip(" ;").endswith("0x1"))
dep0 = next(i for i in L if i > dep1 and opc[i].startswith("WARPGROUP.DEPBAR") and src[i].rstrip(" ;").endswith("0x0"))
last_hg = max(i for i in L if i < dep1 and opc[i].startswith("HGMMA"))
kv = set()
for i in L:
    if "TRYWAIT" in opc[i]:
        kv.add(i)
        if i + 1 <= hi and opc[i + 1] == "BRA": kv.add(i + 1)
cat = collections.Counter()
for i in L:
    b = bar[i]; cat["baton wait (barrier stalls)"] += b; rest = smp[i] - b
    if i in kv: cat["K/V wait (group 0)"] += rest
    elif i == dep1: cat["QK wait"] += rest
    elif i == dep0: cat["P.V wait"] += rest
    elif i <= last_hg: cat["issue (descriptors, 16 HGMMA, baton arrive)"] += rest
    elif i < dep1: cat["between issue and QK wait"] += rest
    elif i < dep0: cat["softmax A (before the P.V wait)"] += rest
    else: cat["rest (exps after the wait, rescale, packs, loop)"] += rest
tot = sum(cat.values())
n_ex = lambda a, b: sum(1 for i in range(a, b) if opc[i] == "MUFU.EX2")
print(f"== {name}: loop {lo}..{hi} ({hi - lo + 1} instr, {lx:,} executions), exps before / after the P.V wait {n_ex(dep1, dep0)} / {n_ex(dep0, hi + 1)}, period {period:.0f}")
for c in ("baton wait (barrier stalls)", "K/V wait (group 0)", "issue (descriptors, 16 HGMMA, baton arrive)", "between issue and QK wait", "QK wait",
          "softmax A (before the P.V wait)", "P.V wait", "rest (exps after the wait, rescale, packs, loop)"):
    print(f"   {c:50s} {100 * cat[c] / tot:5.1f}% = {cat[c] / tot * period:6.0f} cycles")
