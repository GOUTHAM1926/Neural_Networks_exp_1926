import csv, sys
def load(path):
    rows = list(csv.reader(open(path))); h = rows[0]
    return [dict(zip(h, r)) for r in rows[2:]]
srcs = [("FA3", "v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/ncu_libs_fwd.raw.csv", "FlashAttnFwdSm90"),
        ("F3c", "session_2026_10_17/box_runs/multi_1017/ncu_vj2_F3c_dense.raw.csv", "gqa_v62"),
        ("K1", "session_2026_10_17/box_runs/k1_1017/ncu_vj2_K1_dense.raw.csv", "gqa_v62")]
M = [("time us", "gpu__time_duration.sum", 1e-3), ("SM clock MHz", "smsp__cycles_elapsed.avg.per_second", 1e-6),
     ("cycles (elapsed, avg SM) M", "sm__cycles_elapsed.avg", 1e-6),
     ("instructions M (warp-level)", "smsp__inst_executed.sum", 1e-6),
     ("issue active % (per SMSP)", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
     ("tensor (hmma pipe) busy %", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", 1),
     ("XU (exp) pipe %", "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", 1),
     ("FMA pipe %", "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", 1),
     ("ALU pipe %", "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", 1),
     ("uniform pipe %", "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active", 1),
     ("SM busy %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
     ("DRAM read MB", "dram__bytes_read.sum", 1e-6), ("L2 read requests M", "lts__t_requests_srcunit_tex_op_read.sum", 1e-6),
     ("warps active per SMSP", "smsp__warps_active.avg.per_cycle_active", 1)]
K = []
for name, path, tag in srcs:
    rows = [r for r in load(path) if tag in r["Kernel Name"]]
    K.append((name, rows[0]))
print(f"{'':32s}" + "".join(f"{n:>12s}" for n, _ in K))
for label, key, sc in M:
    vals = []
    for n, r in K:
        v = r.get(key, "")
        try: vals.append(f"{float(v.replace(',', '')) * sc:12.2f}")
        except: vals.append(f"{'n/a':>12s}")
    print(f"{label:32s}" + "".join(vals))
