# 2026-10-17 20:50: K5q (today's best forward) vs FA3 from ncu -- the same four views as 45.32 (K1), for dense AND packed.
#   1. totals: time, clock, cycles, instructions, issue / tensor activity
#   2. stall reasons, in cycle-equivalents (share of samples x cycles; FA3's whole kernel = 100)
#   3. regions by executed count (an instruction's executed count says how often its code runs: per loop step, per unit, ...)
#   4. K5q's hottest instructions
# Clocks: K5q (2x box, 10-17) and F3c (10-16) ran locked at ~1,336 MHz; FA3 (10-13) ~1,620 dense / ~1,730 packed; K1 ~1,680.
# F3c at both clocks (multi1_1016 1.336 GHz vs multi_1017 1.693 GHz): +1.7% cycles dense, +0.5% packed at the higher clock,
# so cycle comparisons across these clocks are good to ~1-2% (FA3's cycles would be ~1% LOWER at 1,336 MHz).
# run from profiling_reports/:  python3 session_2026_10_17/deep/ncu_k5q_vs_fa3.py [hot-spot count]
import csv, collections, re, sys

P13 = "v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/"
K5 = "session_2026_10_17/box_runs/2x_1017/ncu_k5q_1017/"
K5b = "session_2026_10_17/box_runs/2x_1017/ncu_k5q_1017_rep2/"
RAW = {"dense": [("FA3", P13 + "ncu_libs_fwd.raw.csv", "FlashAttnFwdSm90"),
                 ("F3c", "scoreboard_free_2026_10_16/multi1_1016/ncu_vj2_F3c_dense.raw.csv", "gqa_v62"),
                 ("K1", "session_2026_10_17/box_runs/k1_1017/ncu_vj2_K1_dense.raw.csv", "gqa_v62"),
                 ("K5q", K5 + "ncu_vj2_K5q_dense.raw.csv", "gqa_v62"), ("K5q#2", K5b + "ncu_vj2_K5q_dense.raw.csv", "gqa_v62")],
       "packed": [("FA3", P13 + "ncu_fa3_packed_fwd.raw.csv", "FlashAttnFwdSm90"),
                  ("F3c", "scoreboard_free_2026_10_16/multi1_1016/ncu_vj2_F3c_packed.raw.csv", "gqa_v62"),
                  ("K5q", K5 + "ncu_vj2_K5q_packed.raw.csv", "gqa_v62"), ("K5q#2", K5b + "ncu_vj2_K5q_packed.raw.csv", "gqa_v62")]}
SRC = {"dense": [("FA3", P13 + "ncu_libs_fwd.source.csv", "FlashAttnFwdSm90"),
                 ("K1", "session_2026_10_17/box_runs/k1_1017/ncu_vj2_K1_dense.source.csv", "gqa_v62"),
                 ("K5q", K5 + "ncu_vj2_K5q_dense.source.csv", "gqa_v62")],
       "packed": [("FA3", P13 + "ncu_fa3_packed_fwd.source.csv", "FlashAttnFwdSm90"),
                  ("K5q", K5 + "ncu_vj2_K5q_packed.source.csv", "gqa_v62")]}
NHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 25

def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0

def raw_row(path, tag):
    rows = list(csv.reader(open(path))); h = rows[0]
    return [dict(zip(h, r)) for r in rows[2:] if tag in r[h.index("Kernel Name")]][0]

def src_kernel(path, tag):   # [(sass, executed, samples, {stall: samples})]
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if tag in b["name"]); h = b["rows"][0]; idx = {c: i for i, c in enumerate(h)}
    out = []
    for r in b["rows"][1:]:
        if len(r) < len(h): continue
        g = lambda c: f(r[idx[c]]) if c in idx else 0.0
        st = {c[len("stall_"):]: g(c) for c in h if c.startswith("stall_") and "Not Issued" not in c}
        out.append((r[idx["Source"]].strip(), int(g("Instructions Executed")), g("Warp Stall Sampling (All Samples)"), st))
    return out

def opc(s):
    s = re.sub(r"^@!?U?P[T0-9]+\s+", "", s)
    return s.split()[0].split(".")[0] if s else ""

M = [("time us", "gpu__time_duration.sum", None), ("SM clock MHz", "smsp__cycles_elapsed.avg.per_second", None),
     ("cycles M (avg SM)", "sm__cycles_elapsed.avg", 1e-6), ("instructions M", "smsp__inst_executed.sum", 1e-6),
     ("issue active %", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
     ("tensor busy %", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", 1),
     ("XU (exp) pipe %", "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", 1),
     ("SM busy %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
     ("DRAM read MB", "dram__bytes_read.sum", 1e-6), ("DRAM write MB", "dram__bytes_write.sum", 1e-6),
     ("L2 read requests M", "lts__t_requests_srcunit_tex_op_read.sum", 1e-6),
     ("registers / thread", "launch__registers_per_thread", 1), ("shared mem / block KB", "launch__shared_mem_per_block", None)]

for shape in ("dense", "packed"):
    K = [(n, raw_row(p, t)) for n, p, t in RAW[shape]]
    print(f"\n################ {shape.upper()} ################\n\n1. TOTALS")
    print(f"{'':24s}" + "".join(f"{n:>11s}" for n, _ in K) + f"{'K5q/FA3':>10s}")
    for label, key, sc in M:
        vals = {}
        for n, r in K:
            v = f(r.get(key, ""))
            if key == "gpu__time_duration.sum":    # units differ per report: ms vs us
                v = v * 1000 if v < 50 else v
            elif key == "smsp__cycles_elapsed.avg.per_second": v = v * 1000 if v < 50 else v / 1e6
            elif key == "launch__shared_mem_per_block": v = v / 1024 if v > 1024 else v
            else: v = v * sc
            vals[n] = v
        ratio = vals["K5q"] / vals["FA3"] if vals.get("FA3") else 0
        print(f"{label:24s}" + "".join(f"{vals[n]:11.2f}" for n, _ in K) + f"{ratio:10.3f}")
    cyc = {n: f(r["sm__cycles_elapsed.avg"]) for n, r in K}

    pre = "smsp__pcsamp_warps_issue_stalled_"
    keys = sorted({k[len(pre):] for _, r in K for k in r if k.startswith(pre) and not k.endswith("_not_issued")})
    use = [n for n, _ in K if n in ("FA3", "K1", "K5q", "K5q#2")]
    R = dict(K)
    tot = {n: sum(f(R[n].get(pre + k, "0")) for k in keys) for n in use}
    print("\n2. STALL REASONS: share of samples % | cycle-equivalents (share x cycles; FA3 total = 100) | K5q - FA3")
    print(f"{'reason':26s}" + "".join(f"{n:>8s}" for n in use) + "".join(f"{n + ' eq':>10s}" for n in use) + f"{'K5q-FA3':>9s}")
    rows = []
    for k in keys:
        sh = {n: 100 * f(R[n].get(pre + k, "0")) / tot[n] for n in use}
        eq = {n: sh[n] * cyc[n] / cyc["FA3"] for n in use}
        rows.append((eq["K5q"] - eq["FA3"], k, sh, eq))
    for d, k, sh, eq in sorted(rows, reverse=True):
        if max(sh.values()) < 0.3: continue
        print(f"{k:26s}" + "".join(f"{sh[n]:8.1f}" for n in use) + "".join(f"{eq[n]:10.1f}" for n in use) + f"{d:+9.1f}")
    print(f"{'TOTAL':26s}" + " " * 8 * len(use) + "".join(f"{100 * cyc[n] / cyc['FA3']:10.1f}" for n in use) +
          f"{100 * (cyc['K5q'] - cyc['FA3']) / cyc['FA3']:+9.1f}")

    print("\n3. REGIONS by executed count (samples x cycles / samples total -> cycle-equivalents, FA3 total = 100)")
    S = {n: src_kernel(p, t) for n, p, t in SRC[shape]}
    groups = {}
    for n, ins in S.items():
        tot_s = sum(s for _, _, s, _ in ins)
        g = collections.defaultdict(lambda: [0, 0, 0.0, collections.Counter(), collections.Counter()])
        for src, ex, s, st in ins:
            b = g[ex]; b[0] += 1; b[1] += ex; b[2] += s; b[3].update(st); b[4][opc(src)] += 1
        groups[n] = (g, tot_s, sum(ex for _, ex, _, _ in ins))
        loop_ex = max((ex for ex, b in g.items() if ex > 0), key=lambda ex: g[ex][0])   # the count with the most instructions
        eqk = lambda s: 100 * s / tot_s * cyc[n] / cyc["FA3"]
        print(f"\n== {n}: {len(ins)} instructions in SASS, {sum(ex for _, ex, _, _ in ins) / 1e6:.1f}M executed, {int(tot_s)} samples;"
              f" main loop = executed {loop_ex} times ({g[loop_ex][0]} instructions)")
        print(f"   {'exec count':>10s} {'instr':>6s} {'exec M':>7s} {'samples':>8s} {'share':>6s} {'cyc-eq':>7s}  top stall reasons | top opcodes")
        for ex, (ni, exs, s, st, ops) in sorted(g.items(), key=lambda kv: -kv[1][2]):
            if s / tot_s < 0.01: continue
            top = ", ".join(f"{k} {100 * v / s:.0f}%" for k, v in st.most_common(3) if v > 0)
            top_ops = ", ".join(f"{k} {v}" for k, v in ops.most_common(4))
            print(f"   {ex:10d} {ni:6d} {exs / 1e6:7.2f} {int(s):8d} {100 * s / tot_s:5.1f}% {eqk(s):7.1f}  {top} | {top_ops}")
        loop_s = g[loop_ex][2]; rest = tot_s - loop_s
        print(f"   SPLIT: main loop {eqk(loop_s):6.1f} | everything else {eqk(rest):6.1f} | total {eqk(tot_s):6.1f}")

    if "K5q" in S:
        ins = S["K5q"]; tot_s = sum(s for _, _, s, _ in ins)
        print(f"\n4. K5q {shape}: the {NHOT} hottest instructions (index, executed, samples, share, top stalls, SASS)")
        for i, (src, ex, s, st) in sorted(enumerate(ins), key=lambda x: -x[1][2])[:NHOT]:
            top = ",".join(f"{k}:{int(v)}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:2] if v > 0)
            print(f"   {i:5d} {ex:8d} {int(s):6d} {100 * s / tot_s:5.2f}%  {top:34s} {src[:80]}")
