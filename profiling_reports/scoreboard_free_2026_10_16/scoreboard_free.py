# 2026-10-16: summary of scoreboard_free_1016.sh (runs ON THE BOX: the raw nsys traces stay there; only this text comes back).
# Per run (one contender per process, free clocks, kernel times from nsys): the forward main kernel, the backward main
# kernel and the whole backward call (helpers included, bwd_breakdown.calls), medians after skipping the first 20 calls;
# the GPU-busy share of the timed window; the median SM clock and power while busy (nvidia-smi, 100 ms).
# Then per shape: the mean of the two ABBA rounds and ours / rival ratios. usage: python3 scoreboard_free.py <run folder>
import csv, glob, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_names import short
from bwd_breakdown import calls
L = sys.argv[1]
FWD = {"ours": None, "fa3": "FA3_fwd", "fa4": "FA4_fwd", "cudnn": "CUDNN_fwd"}
def one(path, shape, who):
    r = list(csv.DictReader(open(path)))
    k = [(short(x["Name"]), int(x["Start (ns)"]), int(x["Duration (ns)"] or 0), x["Name"]) for x in r if x.get("Duration (ns)")]
    if len(k) < 100: return None   # failed run (empty trace)
    if who == "ours":
        fkey = "gqa_v62_sm90<(int)0>" if shape == "dense" else "gqa_v62_sm90<(int)1>"
        fwd = [d for s, t, d, n in k if fkey in n]
        bkey = "OURS_dense" if shape == "dense" else "OURS_packed"
    else:
        fwd = [d for s, t, d, n in k if s == FWD[who]]
        bkey = {"fa3": "FA3", "fa4": "FA4", "cudnn": "CUDNN"}[who]
    cs = calls(path)
    bl = cs.get(bkey, [])
    bmain = [p["MAIN"] for p, _ in bl][20:]; bspan = [s for _, s in bl][20:]
    t0 = k[len(k) // 10][1]; t1 = max(t + d for _, t, d, _ in k)          # last 90% of the run: the timed loop
    busy = sum(d for _, t, d, _ in k if t >= t0) / max(1, t1 - t0)
    return dict(fwd=st.median(fwd[20:]) / 1e3 if len(fwd) > 40 else float("nan"),
                bmain=st.median(bmain) / 1e3 if bmain else float("nan"),
                bcall=st.median(bspan) / 1e3 if bspan else float("nan"), n=len(fwd), busy=100 * busy)
def clocks(path):
    rows = [x for x in csv.reader(open(path)) if len(x) >= 6 and x[1].strip().endswith("MHz")]
    act = [(int(x[1].split()[0]), float(x[3].split()[0])) for x in rows if x[5].strip().split()[0].isdigit() and int(x[5].split()[0]) >= 90]
    return (st.median(a for a, _ in act), st.median(p for _, p in act), len(act)) if act else (float("nan"), float("nan"), 0)
res = {}
for f in sorted(glob.glob(f"{L}/*_cuda_gpu_trace.csv")):
    m = re.match(r"(dense|packed)_(ours|fa3|fa4|cudnn)_r(\d)_cuda_gpu_trace\.csv", os.path.basename(f))
    if not m: continue
    shape, who, rnd = m.groups()
    d = one(f, shape, who)
    if d is None: print(f"skipped {os.path.basename(f)}: empty or failed run"); continue
    d["mhz"], d["watt"], d["busy_n"] = clocks(f"{L}/{shape}_{who}_r{rnd}.clocks.csv")
    res.setdefault(shape, {}).setdefault(who, {})[rnd] = d
for shape, byw in res.items():
    print(f"\n=== {shape} (kernel times in us, free clocks; r1 / r2 = the two ABBA rounds) ===")
    print(f"{'':7s} {'fwd main r1/r2':>18s} {'bwd main r1/r2':>18s} {'bwd call r1/r2':>18s} {'busy%':>11s} {'MHz':>11s} {'W':>11s} calls")
    mean = {}
    for who in ("ours", "fa3", "fa4", "cudnn"):
        if who not in byw: continue
        r = byw[who]; g = lambda key: [r[x][key] for x in sorted(r)]
        mean[who] = {key: st.mean(g(key)) for key in ("fwd", "bmain", "bcall")}
        fmt = lambda v: "/".join(f"{x:.1f}" for x in v)
        print(f"{who:7s} {fmt(g('fwd')):>18s} {fmt(g('bmain')):>18s} {fmt(g('bcall')):>18s} {fmt(g('busy')):>11s} "
              f"{'/'.join(f'{x:.0f}' for x in g('mhz')):>11s} {'/'.join(f'{x:.0f}' for x in g('watt')):>11s} {g('n')}")
    if "ours" in mean:
        o = mean["ours"]
        print(f"ours / rival (mean of the rounds; below 1.00 = ours faster):")
        for who in ("fa3", "fa4", "cudnn"):
            if who not in mean: continue
            m = mean[who]
            print(f"   vs {who:6s} forward {o['fwd'] / m['fwd']:.3f} | backward main {o['bmain'] / m['bmain']:.3f} | backward call {o['bcall'] / m['bcall']:.3f}"
                  f" | forward + backward call {(o['fwd'] + o['bcall']) / (m['fwd'] + m['bcall']):.3f}")
