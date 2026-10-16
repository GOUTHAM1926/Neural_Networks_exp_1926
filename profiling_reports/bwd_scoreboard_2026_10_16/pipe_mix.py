# 2026-10-16: instruction mix per pipe (warp-level instructions, millions) of each profiled backward main kernel, from ncu raw CSVs.
import csv, glob
from seq_names import short
def num(s):
    try: return float(s.replace(",", ""))
    except: return float("nan")
pick = {}
for f in ["ours_dense", "libs_dense", "ours_packed", "libs_packed"]:
    rows = list(csv.reader(open(f"prof_1016/ncu_{f}.raw.csv"))); h = rows[0]
    for r in rows[2:]:
        d = dict(zip(h, r))
        n = short(d["Kernel Name"].replace("<0>(", "<(bool)0>(").replace("<1>(", "<(bool)1>(")).replace("_bwd_main", "")
        if n == "FA3":   # deterministic build: more cycles for the same work
            dense = "dense" in f
            if num(d["gpc__cycles_elapsed.max"]) > (4.6e6 if dense else 2.1e6): n = "FA3det"
        pick.setdefault(f.split("_")[1] + ":" + n, d)
order = [o for o in ["dense:OURS_dense", "dense:FA3", "dense:FA4", "dense:CUDNN", "dense:FA3det", "packed:OURS_packed", "packed:FA3", "packed:FA4", "packed:CUDNN", "packed:FA3det"] if o in pick]
mets = ["smsp__inst_issued.sum", "gpc__cycles_elapsed.max"] + [f"sm__inst_executed_pipe_{p}.sum" for p in
        ["alu", "fma", "fma_type_fp16", "fmaheavy", "xu", "lsu", "uniform", "cbu", "adu", "tensor_op_gmma", "tma", "tex"]] + \
       [f"smsp__sass_inst_executed_op_{o}.sum" for o in ["shared_ld", "shared_st", "global_ld", "global_st", "tma_red", "dshared"]] + \
       ["smsp__inst_executed_op_global_red.sum", "smsp__inst_executed_op_ldsm.sum", "smsp__inst_executed_op_shared_stsm.sum", "smsp__inst_executed_op_branch.sum"]
print("metric (millions)".ljust(40) + "".join(o.replace("dense:", "d:").replace("packed:", "p:")[:12].rjust(13) for o in order))
for m in mets:
    if m not in pick[order[0]]: continue
    print(m.replace("sm__inst_executed_pipe_", "pipe ").replace("smsp__sass_inst_executed_op_", "op ").replace("smsp__inst_executed_op_", "op ")[:40].ljust(40)
          + "".join(f"{num(pick[o].get(m, 'nan')) / 1e6:13.2f}" for o in order))
# per pipe: only "% of peak" is exported, so work = pct_of_peak_sustained_elapsed x elapsed cycles (same peak per pipe -> comparable
# between kernels, not between pipes). Printed relative to our kernel = 1.00.
print("\nper-pipe work relative to OURS (pct of peak x elapsed cycles), and OURS pct of peak")
pipes = ["alu", "fma", "fmaheavy", "fma_type_fp16", "xu", "lsu", "uniform", "cbu", "adu", "tensor_op_gmma", "tma"]
for grp in ["dense", "packed"]:
    os_ = [o for o in order if o.startswith(grp)]
    base = os_[0]
    print(grp.ljust(22) + "".join(o.split(":")[1][:10].rjust(11) for o in os_))
    for p in pipes:
        k = f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_elapsed"
        if k not in pick[base]: continue
        w = {o: num(pick[o][k]) * num(pick[o]["gpc__cycles_elapsed.max"]) for o in os_}
        if w[base] == 0: continue
        print(f"  {p:14s}{num(pick[base][k]):5.1f}%" + "".join(f"{w[o] / w[base]:11.2f}" for o in os_))
