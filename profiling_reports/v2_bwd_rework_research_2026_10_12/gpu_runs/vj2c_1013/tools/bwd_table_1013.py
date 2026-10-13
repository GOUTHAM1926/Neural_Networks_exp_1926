#!/usr/bin/env python3
# Backward table for test 2a (2026-10-13): our Vj1d / fixed Vj2 (mean of two same-session runs each) vs the
# libraries from the same session, plus yesterday's first Vj2 (other session, for reference only).
import re, sys, statistics as st
d, y = sys.argv[1], sys.argv[2]          # today's 2a dir, yesterday's vj2_speed_1012 dir
def bench(path, impl_override=None):
    out = {}
    for l in open(path):
        m = re.match(r"BENCH layout=(\S+) impl=(\S+) dir=bwd med_ms=([\d.]+)", l)
        if m: out[(m.group(1), impl_override or m.group(2))] = float(m.group(3))
    return out
runs = {k: [bench(f"{d}/2a_{k}_r{r}.log") for r in (1, 2)] for k in ("vj1d", "vj2")}
libs = bench(f"{d}/2a_libs_dense.log")
old = bench(f"{y}/vj2_r1.log"), bench(f"{y}/vj2_r2.log")
shapes = [s for (s, _) in runs["vj2"][0]]
print("backward, median ms per call; Vj1d and Vj2 = mean of 2 same-session runs (run spread in brackets)")
print(f"{'shape':16s} {'Vj1d':>6s} {'Vj2':>6s} {'spr%':>5s} {'FA3':>6s} {'FA3det':>6s} {'FA4':>6s} {'cuDNN':>6s} | "
      f"{'Vj2/Vj1d':>8s} {'Vj2/FA3':>7s} {'Vj2/FA3d':>8s} {'Vj2/cuDNN':>9s} | {'old Vj2':>7s} {'new/old':>7s}")
r_v1, r_fa3, r_cud, r_det, r_old = [], [], [], [], []
for s in shapes:
    a = st.mean(r[(s, "ours_dense")] for r in runs["vj1d"])
    b_runs = [r[(s, "ours_dense")] for r in runs["vj2"]]
    b = st.mean(b_runs)
    spr = 100 * (max(b_runs) - min(b_runs)) / b
    f3, f3d, f4, cu = (libs.get((s, k), float("nan")) for k in ("fa3", "fa3_det", "fa4", "sdpa_cudnn"))
    o = st.mean(r[(s, "ours_dense")] for r in old)
    r_v1.append(b / a); r_fa3.append(b / f3); r_det.append(b / f3d); r_cud.append(b / cu); r_old.append(b / o)
    print(f"{s:16s} {a:6.3f} {b:6.3f} {spr:5.1f} {f3:6.3f} {f3d:6.3f} {f4:6.3f} {cu:6.3f} | "
          f"{b/a:8.3f} {b/f3:7.3f} {b/f3d:8.3f} {b/cu:9.3f} | {o:7.3f} {b/o:7.3f}")
def rng(v): return f"min {min(v):.3f}  max {max(v):.3f}  mean {st.mean(v):.3f}"
print("Vj2/Vj1d  ", rng(r_v1), f"| shapes >=10% faster than Vj1d: {sum(x <= 0.90 for x in r_v1)}/{len(r_v1)}")
print("Vj2/FA3   ", rng(r_fa3))
print("Vj2/FA3det", rng(r_det), f"| faster than FA3det on {sum(x < 1 for x in r_det)}/{len(r_det)}")
print("Vj2/cuDNN ", rng(r_cud), f"| faster than cuDNN on {sum(x < 1 for x in r_cud)}/{len(r_cud)}")
print("new/old Vj2 (other session)", rng(r_old))
