# 2026-10-19: reads a clock_power_check.sh run (work doc 45.49): for every run, the nvidia-smi samples from 4 s after its
# start to its end -> mean SM clock, mean instant power, share of samples with the sw_power_cap flag active, and the
# run's own time per forward; then per entry and shape the mean of its two runs (mirrored order).
# usage (from profiling_reports/): python3 session_2026_10_19/deep/clock_power_read.py session_2026_10_19/box_runs/lh_1019
import collections, datetime, os, re, statistics, sys

D = sys.argv[1]
fmt = lambda s: datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f")
samples = []
for line in open(os.path.join(D, "clocks.csv")):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 6: continue
    try: samples.append((fmt(p[0]), float(p[1]), float(p[2]), p[3] == "Active", float(p[4]), float(p[5])))
    except ValueError: continue
per = collections.defaultdict(list)
print(f"{'run':34s} {'samples':>7s} {'clock MHz':>9s} {'power W':>8s} {'cap %':>6s} {'temp':>5s} {'ms/fwd':>8s}")
for line in open(os.path.join(D, "runs.txt")):
    a, b, rc, log = [x.strip() for x in line.split("|")]
    shape, name, d0, t0 = a.split(None, 3)
    t0, t1 = fmt(d0 + " " + t0), fmt(b)
    w = [s for s in samples if t0 + datetime.timedelta(seconds=4) <= s[0] <= t1 and s[5] > 50]
    txt = open(os.path.join(D, log)).read()
    m = re.search(r"med_ms=([0-9.]+)", txt) or re.search(r"per_iter_ms=([0-9.]+)", txt)
    ms = float(m.group(1)) if m else float("nan")
    if not w or rc != "rc=0":
        print(f"{log:34s} no samples / {rc}"); continue
    clk = statistics.mean(s[1] for s in w); pw = statistics.mean(s[2] for s in w)
    cap = 100 * sum(s[3] for s in w) / len(w); tmp = statistics.mean(s[4] for s in w)
    print(f"{log:34s} {len(w):7d} {clk:9.1f} {pw:8.1f} {cap:6.0f} {tmp:5.1f} {ms:8.4f}")
    per[(shape, name)].append((clk, pw, cap, ms, len(w)))
print("\nper entry (mean of its runs):")
for shape in ("dense", "packed"):
    base = per.get((shape, "vj2_K5q"))
    bclk = statistics.mean(r[0] for r in base) if base else None
    bms = statistics.mean(r[3] for r in base) if base else None
    for (sh, name), rs in per.items():
        if sh != shape: continue
        clk = statistics.mean(r[0] for r in rs); pw = statistics.mean(r[1] for r in rs); ms = statistics.mean(r[3] for r in rs)
        rel = f" | clock vs K5q {100 * (clk / bclk - 1):+.2f}%  time vs K5q {100 * (ms / bms - 1):+.2f}%" if bclk else ""
        print(f"  {shape:6s} {name:8s} runs {len(rs)}  clock {clk:7.1f} MHz (runs {' / '.join(f'{r[0]:.0f}' for r in rs)})"
              f"  power {pw:6.1f} W  cap {statistics.mean(r[2] for r in rs):3.0f}%  ms/fwd {ms:.4f}{rel}")
