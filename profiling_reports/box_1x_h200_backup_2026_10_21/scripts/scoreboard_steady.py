# 2026-10-19: scoreboard_free.py restricted to the STEADY state: only calls that start >= 2 s after the run's first forward
# kernel (the first ~2 s run on a cooler GPU, a little faster; ours starts its GPU work within ~1 s of launch, the rivals
# after ~5 s of Python start-up, so the cool phase could favour one side). Same kernels, same backward-call grouping
# (bwd_breakdown BEFORE / AFTER helpers). usage: python3 scoreboard_steady.py <run folder> [skip seconds, default 2]
import csv, glob, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_names import short
from bwd_breakdown import BEFORE, AFTER
L = sys.argv[1]; SKIP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
FWD = {"fa3": "FA3_fwd", "fa4": "FA4_fwd", "cudnn": "CUDNN_fwd"}
def one(path, shape, who):
    r = [(short(x["Name"]), int(x["Start (ns)"]), int(x["Duration (ns)"] or 0), x["Name"]) for x in csv.DictReader(open(path)) if x.get("Duration (ns)")]
    if who == "ours":
        key = "gqa_v62_sm90<(int)0>" if shape == "dense" else "gqa_v62_sm90<(int)1>"
        fk = [(t, d) for s, t, d, n in r if key in n]
    else:
        fk = [(t, d) for s, t, d, n in r if s == FWD[who]]
    t0 = fk[0][0] + SKIP * 1e9
    fwd = [d for t, d in fk if t >= t0]
    bmain, bcall = [], []
    for i, (s, t, d, n) in enumerate(r):
        if "_bwd_main" not in s: continue
        grp = s.split("_bwd_main")[0].split("_")[0]
        a = i
        while a > 0 and r[a - 1][0] in BEFORE[grp]: a -= 1
        b = i
        while b + 1 < len(r) and r[b + 1][0] in AFTER[grp]: b += 1
        if r[a][1] < t0: continue
        bmain.append(d); bcall.append(r[b][1] + r[b][2] - r[a][1])
    return dict(fwd=st.median(fwd) / 1e3, bmain=st.median(bmain) / 1e3, bcall=st.median(bcall) / 1e3, nf=len(fwd), nb=len(bmain))
res = {}
for f in sorted(glob.glob(f"{L}/*_cuda_gpu_trace.csv")):
    m = re.match(r"(dense|packed)_(ours|fa3|fa4|cudnn)_r(\d)_cuda_gpu_trace\.csv", os.path.basename(f))
    if m: res.setdefault(m.group(1), {}).setdefault(m.group(2), {})[m.group(3)] = one(f, m.group(1), m.group(2))
for shape, byw in res.items():
    print(f"\n=== {shape}, steady state only (calls starting >= {SKIP:.0f} s after the first forward), kernel times in us ===")
    print(f"{'':7s} {'fwd main r1/r2':>16s} {'bwd main r1/r2':>16s} {'bwd call r1/r2':>16s}  calls fwd/bwd")
    mean = {}
    for who in ("ours", "fa3", "fa4", "cudnn"):
        r = byw[who]; g = lambda k: [r[x][k] for x in sorted(r)]
        mean[who] = {k: st.mean(g(k)) for k in ("fwd", "bmain", "bcall")}
        print(f"{who:7s} {'/'.join(f'{x:.1f}' for x in g('fwd')):>16s} {'/'.join(f'{x:.1f}' for x in g('bmain')):>16s} "
              f"{'/'.join(f'{x:.1f}' for x in g('bcall')):>16s}  {g('nf')}/{g('nb')}")
    o = mean["ours"]
    for who in ("fa3", "fa4", "cudnn"):
        m = mean[who]
        print(f"   ours / {who:6s} forward {o['fwd'] / m['fwd']:.3f} | backward main {o['bmain'] / m['bmain']:.3f} | backward call "
              f"{o['bcall'] / m['bcall']:.3f} | forward + backward call {(o['fwd'] + o['bcall']) / (m['fwd'] + m['bcall']):.3f}")
