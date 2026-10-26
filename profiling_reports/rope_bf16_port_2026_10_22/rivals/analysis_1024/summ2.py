# per-kernel summary of ncu raw csv exports, using the units row (row 1) instead of assuming ns / bytes
import csv, sys
M = {"t": "gpu__time_duration.sum", "dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
     "rd": "dram__bytes_read.sum", "wr": "dram__bytes_write.sum", "occ": "sm__warps_active.avg.pct_of_peak_sustained_active",
     "grid": "launch__grid_size", "blk": "launch__block_size", "regs": "launch__registers_per_thread"}
TS = {"ns": 1e-3, "usecond": 1, "us": 1, "msecond": 1e3, "ms": 1e3, "nsecond": 1e-3}
BS = {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1, "Gbyte": 1e3, "KB": 1e-3, "MB": 1, "GB": 1e3, "B": 1e-6}
num = lambda s: float(s.replace(",", ""))
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path))); h, u = rows[0], rows[1]
    ix = {k: h.index(v) for k, v in M.items()}; kn = h.index("Kernel Name")
    tot = 0; print("==", path.split("/")[-1][:-4])
    for x in rows[2:]:
        if not x or not x[0].strip().isdigit(): continue
        t = num(x[ix["t"]]) * TS[u[ix["t"]]]; tot += t
        mb = num(x[ix["rd"]]) * BS[u[ix["rd"]]] + num(x[ix["wr"]]) * BS[u[ix["wr"]]]
        print(f"  {x[kn][:52]:52s} {t:7.1f} us DRAM {num(x[ix['dram']]):5.1f}% {mb:6.1f} MB occ {num(x[ix['occ']]):5.1f}% "
              f"grid {x[ix['grid']]:>6s}x{x[ix['blk']]:>4s} regs {x[ix['regs']]}")
    print(f"  total {tot:.1f} us")
