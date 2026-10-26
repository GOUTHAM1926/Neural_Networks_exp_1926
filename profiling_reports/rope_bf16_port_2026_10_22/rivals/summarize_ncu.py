# One line per kernel from `ncu --import X --page raw --csv --metrics ...` exports (2026-10-23, step 3).
# usage: python summarize_ncu.py <csv> [<csv> ...]
import csv, sys

M = ["gpu__time_duration.sum", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum",
     "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__grid_size", "launch__block_size",
     "launch__registers_per_thread", "dram__bytes_write.sum"]
num = lambda s: float(s.replace(",", ""))
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    h = rows[0]
    ix = {k: h.index(k) for k in ["Kernel Name"] + M}
    tot = 0.0
    print(f"== {path.split('/')[-1][:-4]}")
    for x in rows[2:]:
        if not x or not x[0].strip().isdigit():
            continue
        t = num(x[ix[M[0]]]) / 1000; tot += t
        mb = (num(x[ix[M[2]]]) + num(x[ix[M[7]]])) / 1e6
        print(f"   {x[ix['Kernel Name']][:40]:40s} {t:7.1f} us  DRAM {num(x[ix[M[1]]]):5.1f}% {mb:7.1f} MB"
              f"  occ {num(x[ix[M[3]]]):5.1f}%  grid {x[ix[M[4]]]:>6s} x {x[ix[M[5]]]:>4s}  regs {x[ix[M[6]]]}")
    print(f"   total {tot:.1f} us")
