#!/usr/bin/env python3
"""Summarize the library-level benchmark runs (run_lib_bench.sh).

usage: summarize_lib.py <runs dir> <rounds> <OLD dir> <NEW dir>
Refuses a run whose mapped libtensor.so is not the expected one, or whose fingerprint
(GPU bf16 70144 * 2) is not old = 0x7f80 (Inf, Bug D) / new = 0x4809.
"""
import os, re, statistics, sys

runs, rounds, old_dir, new_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
expect = {"old": (old_dir, "0x7f80"), "new": (new_dir, "0x4809")}
data = {}      # (case) -> {"old": [us per round], "new": [...], "prec": {v: (w, nb, cs, kind)}}
bad_runs = []
for r in range(1, rounds + 1):
    for v in ("old", "new"):
        f = os.path.join(runs, f"{v}.r{r}.txt")
        if not os.path.exists(f):
            bad_runs.append(f"{v} r{r}: missing"); continue
        txt = open(f).read()
        lib = re.search(r"^LIB\|(.*)$", txt, re.M); fp = re.search(r"^FP\|(\S+)", txt, re.M)
        if not lib or not lib.group(1).startswith(os.path.join(expect[v][0], "lib/")):
            bad_runs.append(f"{v} r{r}: mapped {lib.group(1) if lib else '?'} (expected {expect[v][0]}/lib/)"); continue
        if not fp or fp.group(1).lower() != expect[v][1]:
            bad_runs.append(f"{v} r{r}: fingerprint {fp.group(1) if fp else '?'} (expected {expect[v][1]})"); continue
        # the name may itself contain "|": take the last five fields from the right
        for m in re.finditer(r"^CASE\|([^\n]*)\|([^|\n]*)\|([^|\n]*)\|([^|\n]*)\|([^|\n]*)\|(\S+)$", txt, re.M):
            name, us, w, nb, cs, kind = m.groups()
            d = data.setdefault(name, {"old": {}, "new": {}, "prec": {}})
            d[v][r] = float(us); d["prec"][v] = (int(w), int(nb), cs, kind)

print(f"library benchmark: {rounds} rounds, old = {old_dir}, new = {new_dir}")
if bad_runs:
    print("REFUSED RUNS (not counted):"); [print("   " + b) for b in bad_runs]
hdr = f"{'case':44s} {'old us':>10s} {'new us':>10s} {'speedup':>8s} {'new/old min..max':>18s}  {'old wrong':>9s} {'new wrong':>9s}  {'same bits old/new':>17s}"
print(hdr); print("-" * len(hdr))
rows = []
for name, d in data.items():
    pairs = [(d["old"][r], d["new"][r]) for r in range(1, rounds + 1) if r in d["old"] and r in d["new"]]
    if not pairs:
        continue
    o = statistics.median(p[0] for p in pairs); n = statistics.median(p[1] for p in pairs)
    ratios = [p[1] / p[0] for p in pairs]
    po, pn = d["prec"].get("old", (0, 0, "", "")), d["prec"].get("new", (0, 0, "", ""))
    same = "yes" if po[2] == pn[2] else "no"
    kind = pn[3]
    ow = "-" if kind == "none" else str(po[0] + po[1]); nw = "-" if kind == "none" else str(pn[0] + pn[1])
    rows.append((name, o, n, min(ratios), max(ratios), statistics.median(ratios), po, pn, kind, same))
    print(f"{name:44s} {o:10.2f} {n:10.2f} {o / n:7.2f}x {min(ratios):8.3f}..{max(ratios):<8.3f}  {ow:>9s} {nw:>9s}  {same:>17s}" + ("   (approx device math)" if kind == "approx" else ""))

print("\n==================== SUMMARY ====================")
ctrl = [r for r in rows if r[0].startswith("CONTROL")]
ctrl_ok = len(ctrl) > 0 and all(0.97 <= r[5] <= 1.03 for r in ctrl)   # no controls ran -> not trustworthy
for r in ctrl:
    print(f"control {r[0]:40s} new/old median {r[5]:.3f}  ({'ok' if 0.97 <= r[5] <= 1.03 else 'NOISY'})")
print(f"measurement trustworthy (all {len(ctrl)} controls within 3%): {'YES' if ctrl_ok and not bad_runs else 'NO'}")
slow = [r for r in rows if not r[0].startswith("CONTROL") and r[3] > 1.03]
fast = [r for r in rows if not r[0].startswith("CONTROL") and r[4] < 0.97]
print("slower (every round new/old > 1.03):", "none" if not slow else "")
for r in slow: print(f"   {r[0]:44s} median {r[5]:.3f} (min {r[3]:.3f})")
print("faster (every round new/old < 0.97):", "none" if not fast else "")
for r in fast: print(f"   {r[0]:44s} {1 / r[5]:.2f}x")
exact = [r for r in rows if r[8] == "exact"]
print(f"precision, exact-reference cases: old wrong {sum(r[6][0] + r[6][1] for r in exact)}, new wrong {sum(r[7][0] + r[7][1] for r in exact)}")
approx = [r for r in rows if r[8] == "approx"]
for r in approx:
    print(f"   approx case {r[0]}: old wrong {r[6][0]}, new wrong {r[7][0]} (device --use_fast_math division; must be equal or lower)")
print(f"RESULT: {'PASS' if sum(r[7][0] + r[7][1] for r in exact) == 0 and all(r[7][0] <= r[6][0] for r in approx) and not bad_runs else 'CHECK'}")
