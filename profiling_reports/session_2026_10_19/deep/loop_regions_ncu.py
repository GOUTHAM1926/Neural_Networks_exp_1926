# 2026-10-19: where a consumer warp spends one STEADY loop step, from ncu's warp-stall sampling (no instrumentation), split
# at the SASS markers every FA3-style loop has (work doc 45.51):
#   wait K + baton : loop top .. the scheduler BAR.SYNC (incl. group 0's K / V TRYWAIT)
#   issue          : after the BAR.SYNC .. the last HGMMA of the step + the baton BAR.ARV
#   QK wait        : the WARPGROUP.DEPBAR ... 0x1
#   softmax A      : after it .. before the WARPGROUP.DEPBAR ... 0x0
#   P.V wait       : the WARPGROUP.DEPBAR ... 0x0
#   after the wait : the rest of the step (exps, rescale, packs, releases, loop control)
# The steady loop = the instructions run most often (once per step per warp); a step's share of a region = its samples /
# all samples of the loop body; cycles = that share x the step period measured by the timeline traces (deep/timeline_read.py
# on the level-2 traces of the same box). Both consumer groups run the same code, so each figure is the two groups' average.
# usage (from profiling_reports/): python3 session_2026_10_19/deep/loop_regions_ncu.py <name> <source.csv> <kernel tag> <period>
import csv, re, sys, collections

def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0

def src_kernel(path, tag):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if tag in b["name"]); h = b["rows"][0]; idx = {c: i for i, c in enumerate(h)}
    out = []
    for r in b["rows"][1:]:
        if len(r) < len(h): continue
        out.append((r[idx["Source"]].strip(), int(f(r[idx["Instructions Executed"]])), f(r[idx["Warp Stall Sampling (All Samples)"]])))
    return out

def opc(s):
    return re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else ""

def regions(name, path, tag, period):
    ins = src_kernel(path, tag)
    # steady-loop execution count: the count carrying the most samples among instructions run >= 1000 times
    by_ex = collections.Counter()
    for s, ex, smp in ins:
        if ex >= 1000: by_ex[ex] += smp
    lx = by_ex.most_common(1)[0][0]
    body = [i for i, (s, ex, smp) in enumerate(ins) if ex == lx]
    lo, hi = body[0], body[-1]
    seg = ins[lo:hi + 1]
    ops = [opc(s) for s, _, _ in seg]
    def first(pred, start=0):
        return next(i for i in range(start, len(seg)) if pred(i))
    bar = first(lambda i: ops[i].startswith("BAR.SYNC") or ops[i].startswith("BAR.RED"))
    dep1 = first(lambda i: ops[i].startswith("WARPGROUP.DEPBAR") and seg[i][0].rstrip(" ;").endswith("0x1"))
    dep0 = first(lambda i: ops[i].startswith("WARPGROUP.DEPBAR") and seg[i][0].rstrip(" ;").endswith("0x0"), dep1 + 1)
    last_issue = max(i for i in range(bar, dep1) if ops[i].startswith("HGMMA") or ops[i].startswith("BAR.ARV"))
    cut = [("wait K + baton", 0, bar + 1), ("issue QK + P.V", bar + 1, last_issue + 1), ("(after issue, before the QK wait)", last_issue + 1, dep1),
           ("QK wait", dep1, dep1 + 1), ("softmax before the P.V wait", dep1 + 1, dep0), ("P.V wait", dep0, dep0 + 1),
           ("after the P.V wait (rest of the step)", dep0 + 1, len(seg))]
    tot = sum(smp for _, _, smp in seg)
    print(f"== {name}: steady loop = instructions {lo}..{hi} ({len(seg)} instr, executed {lx:,} times each), period {period} cycles")
    out = {}
    for nm, a, b in cut:
        part = seg[a:b]
        smp = sum(x for _, _, x in part)
        n_ex = sum(1 for p in part if opc(p[0]) == "MUFU.EX2"); n_hg = sum(1 for p in part if opc(p[0]).startswith("HGMMA"))
        cyc = smp / tot * period
        out[nm] = cyc
        print(f"   {nm:40s} {b - a:4d} instr (EX2 {n_ex:2d}, HGMMA {n_hg:2d}) | {100 * smp / tot:5.1f}% of the samples = {cyc:6.0f} cycles")
    return out

if __name__ == "__main__":
    regions(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
