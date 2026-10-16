# 2026-10-16: one backward call, kernel by kernel, from the nsys traces of prof_bwd.sh (prof_1016/*_cuda_gpu_trace.csv).
# For every backward main kernel: the helper kernels right before it (zero-fills, prep = rowsum(dO*O)) and right after it
# (dQ fp32->bf16 convert, dK/dV head sums or converts). Median over the steady calls (first 10 calls of each contender skipped:
# FA4 JIT-compiles, cuDNN builds its graph). "span" = first helper start -> last helper end, gaps included.
# usage: python3 bwd_breakdown.py prof_1016
import csv, sys, statistics as st
from seq_names import short

BEFORE = {"OURS": {"ours_prep", "ours_prep_chats"},
          "FA3": {"torch_zero_f32", "torch_fill", "fa3_prep", "fa3_varlen_sched"},
          "FA4": {"torch_zero_f32", "fa4_prep"},
          "CUDNN": {"memset", "cudnn_prep", "cudnn_tma_setup"},
          "FA2": {"torch_copy", "fa2_prep"}}
AFTER = {"OURS": {"ours_dkdv_headsum", "ours_dq_convert"},
         "FA3": {"fa3_dq_convert", "fa3_dkdv_convert"},
         "FA4": {"fa4_post"},
         "CUDNN": {"cudnn_dq_convert", "cudnn_dkdv_headsum"},
         "FA2": {"fa2_dq_convert", "torch_reduce"}}

def calls(path):
    r = [(short(x["Name"]), int(x["Start (ns)"]), int(x["Duration (ns)"] or 0), x["Name"]) for x in csv.DictReader(open(path))]
    mains = sorted({x[3] for x in r if "_bwd_main" in x[0]}, key=lambda n: n)
    # FA3: two main kernels (normal, deterministic); the slower one by median is the deterministic one
    out = {}
    for i, (s, t, d, full) in enumerate(r):
        if "_bwd_main" not in s: continue
        who = s.split("_bwd_main")[0]
        if s.startswith("OURS"): who = s.replace("_bwd_main", "")
        a = i
        while a > 0 and r[a - 1][0] in BEFORE[who.split("_")[0]]: a -= 1
        b = i
        while b + 1 < len(r) and r[b + 1][0] in AFTER[who.split("_")[0]]: b += 1
        parts = {}
        for k in range(a, b + 1):
            name = r[k][0] if k != i else "MAIN"
            parts[name] = parts.get(name, 0) + r[k][2]
        span = r[b][1] + r[b][2] - r[a][1]
        out.setdefault((who, full), []).append((parts, span))
    # name FA3 normal / deterministic
    fa3 = [k for k in out if k[0] == "FA3"]
    if len(fa3) == 2:
        med = {k: st.median(p["MAIN"] for p, _ in out[k]) for k in fa3}
        lo, hi = sorted(fa3, key=lambda k: med[k])
        out[("FA3", "")] = out.pop(lo); out[("FA3_det", "")] = out.pop(hi)
    return {k[0]: v for k, v in out.items()}

def report(path):
    print(f"== {path}")
    for who, lst in sorted(calls(path).items()):
        lst = lst[10:]
        keys = sorted({k for p, _ in lst for k in p}, key=lambda k: (k != "MAIN", k))
        med = {k: st.median(p.get(k, 0) for p, _ in lst) / 1e3 for k in keys}
        span = st.median(s for _, s in lst) / 1e3
        helpers = sum(v for k, v in med.items() if k != "MAIN")
        print(f"  {who:13s} calls {len(lst):3d} | span {span:8.1f} us | main {med['MAIN']:8.1f} | helpers {helpers:6.1f} | gaps {span - med['MAIN'] - helpers:5.1f} | "
              + "  ".join(f"{k} {v:.1f}" for k, v in med.items() if k != "MAIN"))

if __name__ == "__main__":
    d = sys.argv[1]
    for f in ["ours_dense", "libs_dense", "ours_packed", "libs_packed"]:
        report(f"{d}/{f}_cuda_gpu_trace.csv")
