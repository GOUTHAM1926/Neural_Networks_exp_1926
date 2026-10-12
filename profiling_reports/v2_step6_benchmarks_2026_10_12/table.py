import sys, re, collections
rows = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for line in open(f):
        m = re.match(r"BENCH layout=(\S+) impl=(\S+) dir=(\S+) med_ms=(\S+) p10_ms=(\S+) p90_ms=(\S+) tflops=(\S+)", line)
        if m: rows[m.group(1)][(m.group(2), m.group(3))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)))
impls = ["ours_A_plan", "ours_B_segstart", "fa3_varlen", "fa3_varlen_det", "fa4_varlen", "cudnn_varlen_gqa", "cudnn_varlen_exp",
         "cudnn_varlen_expsum", "torch_varlen", "flex_docmask",
         "sdpa_cudnn_mask", "ours_dense_fullrow", "sdpa_cudnn_fullrow", "sdpa_flash_fullrow"]
for d in ("fwd", "bwd"):
    print(f"\n{d}: median ms (TFLOP/s of useful work)")
    print("layout".ljust(22) + "".join(i.replace("_varlen", "").replace("sdpa_", "")[:15].rjust(16) for i in impls))
    for lay, r in rows.items():
        print(lay.ljust(22) + "".join((f"{r[(i, d)][0]:.3f} ({r[(i, d)][3]:.0f})" if (i, d) in r else "-").rjust(16) for i in impls))
