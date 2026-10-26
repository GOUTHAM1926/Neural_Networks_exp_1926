# 2026-10-19: a map of a whole kernel from its ncu source page (PC sampling): contiguous blocks of instructions with the same
# executed count, merged into runs; per run: instruction range, executed count, share of all samples, cycle-equivalents per
# SM (share x the kernel's average SM cycles), and what is in it (HGMMA / exps / barriers / TMA / stores). Used to find where a
# kernel's time goes beyond its steady loop (work doc 45.53).
# usage (from profiling_reports/): python3 session_2026_10_19/deep/kernel_map.py <source.csv> <raw.csv> <kernel tag> [min %]
import csv, re, sys, collections
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
path, rawp, tag = sys.argv[1], sys.argv[2], sys.argv[3]; minpct = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
rows = list(csv.reader(open(path))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
body = [r for r in rows[k + 2:] if len(r) >= len(h)]
raw = list(csv.reader(open(rawp))); rh = raw[0]
rr = [dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]
cyc = f(rr["sm__cycles_elapsed.avg"])
src = [r[idx["Source"]].strip() for r in body]; ex = [int(f(r[idx["Instructions Executed"]])) for r in body]
smp = [f(r[idx["Warp Stall Sampling (All Samples)"]]) for r in body]; tot = sum(smp)
op = [re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else "" for s in src]
# runs: consecutive instructions whose executed counts are within 2x of the run's first (merges retry branches, predication)
runs = []; i = 0
while i < len(body):
    j = i; e0 = ex[i]
    while j + 1 < len(body) and ((e0 == 0 and ex[j + 1] == 0) or (e0 > 0 and ex[j + 1] > 0 and 0.5 <= ex[j + 1] / e0 <= 2.0)):
        j += 1
    runs.append((i, j)); i = j + 1
print(f"{tag}: {len(body)} instructions, {cyc:,.0f} cycles per SM, {tot:,.0f} samples")
print(f"{'range':>13s} {'executed':>10s} {'share':>6s} {'cyc/SM':>8s}  content")
for a, b in runs:
    s = sum(smp[a:b + 1])
    if 100 * s / tot < minpct: continue
    c = collections.Counter(op[a:b + 1])
    hg = sum(v for kk, v in c.items() if kk.startswith("HGMMA")); mu = c.get("MUFU.EX2", 0)
    br = sum(v for kk, v in c.items() if kk.startswith("BAR")); sy = sum(v for kk, v in c.items() if kk.startswith("SYNCS"))
    tma = sum(v for kk, v in c.items() if kk.startswith(("UTMALDG", "UBLKCP", "UTMASTG"))); stg = sum(v for kk, v in c.items() if kk.startswith(("STG", "STS", "STSM")))
    fm = c.get("FMNMX", 0) + sum(v for kk, v in c.items() if kk.startswith("FMNMX"))
    desc = ", ".join(x for x in (f"HGMMA {hg}" if hg else "", f"EX2 {mu}" if mu else "", f"FMNMX {fm}" if fm else "", f"BAR {br}" if br else "",
                                  f"SYNCS {sy}" if sy else "", f"TMA {tma}" if tma else "", f"stores {stg}" if stg else "") if x)
    print(f"{a:5d}-{b:<5d} {ex[a]:>10,} {100 * s / tot:5.1f}% {s / tot * cyc:8,.0f}  {b - a + 1:4d} instr  {desc}")
