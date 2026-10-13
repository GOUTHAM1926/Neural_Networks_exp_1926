#!/usr/bin/env python3
# ncu before/after for Vj2 (2026-10-13): yesterday's first Vj2 vs today's fixed Vj2, same shape (dense B8 18/6 T4096),
# same capture (11th launch, --set full). Prints headline metrics, stall shares, and SASS opcode counts
# (executed instructions per opcode family, from the source page).
import csv, re, sys
from collections import Counter
old_dir, new_dir = sys.argv[1], sys.argv[2]
def raw(d):
    rows = list(csv.reader(open(f"{d}/ncu_vj2.raw.csv")))
    return {h: (u, v) for h, u, v in zip(rows[0], rows[1], rows[2])}
def num(r, k):
    try: return float(r[k][1].replace(",", ""))
    except Exception: return float("nan")
o, n = raw(old_dir), raw(new_dir)
M = [("kernel time (ms)", "gpu__time_duration.sum"),
     ("instructions executed", "smsp__inst_executed.sum"),
     ("registers per thread", "launch__registers_per_thread"),
     ("L2 read requests", "lts__t_requests_srcunit_tex_op_read.sum"),
     ("DRAM read (MB)", "dram__bytes_read.sum"),
     ("tensor pipe busy (% of elapsed)", "sm__pipe_tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma_cycles_active.avg.pct_of_peak_sustained_elapsed")]
print(f"{'metric':34s} {'first Vj2 (10-12)':>18s} {'fixed Vj2 (10-13)':>18s} {'new/old':>8s}")
for name, k in M:
    a, b = num(o, k), num(n, k)
    print(f"{name:34s} {a:18,.3f} {b:18,.3f} {b/a:8.3f}")
print("\nwarp stall samples, share of all samples")
st = [k for k in o if k.startswith("smsp__pcsamp_warps_issue_stalled_") and "not_issued" not in k]
to, tn = num(o, "smsp__pcsamp_sample_count"), num(n, "smsp__pcsamp_sample_count")
for k in sorted(st, key=lambda k: -num(o, k)):
    a, b = num(o, k) / to * 100, num(n, k) / tn * 100
    if max(a, b) >= 0.5:
        print(f"  {k.replace('smsp__pcsamp_warps_issue_stalled_', ''):22s} {a:6.1f}% -> {b:6.1f}%")
def sass(d):
    rows = list(csv.reader(open(f"{d}/ncu_vj2.source.csv")))
    hdr = rows[1]; i_src = hdr.index("Source"); i_ex = hdr.index("Instructions Executed")
    c, lines = Counter(), Counter()
    for r in rows[2:]:
        if len(r) <= i_ex: continue
        s = r[i_src].strip()
        s = re.sub(r"^@!?U?P\w+\s+", "", s)
        op = s.split(" ")[0] if s else ""
        fam = op.split(".")[0]
        key = {"LD": "LD (generic)", "ST": "ST (generic)", "LDS": "LDS (shared)", "STS": "STS (shared)",
               "LDG": "LDG (global)", "STG": "STG (global)", "HGMMA": "HGMMA", "BAR": "BAR", "SYNCS": "SYNCS (mbarrier)",
               "R2UR": "R2UR", "UIADD3": "UIADD3", "IMAD": "IMAD", "FFMA": "FFMA", "MUFU": "MUFU"}.get(fam)
        try: ex = int(float(r[i_ex]))
        except Exception: ex = 0
        lines["(all)"] += 1; c["(all)"] += ex
        if key: lines[key] += 1; c[key] += ex
    return c, lines
(co, lo), (cn, ln) = sass(old_dir), sass(new_dir)
print("\nSASS: static lines and executed warp-instructions per opcode family")
print(f"  {'family':18s} {'lines old':>9s} {'lines new':>9s} {'executed old':>15s} {'executed new':>15s}")
for k in ["(all)", "LD (generic)", "ST (generic)", "LDS (shared)", "STS (shared)", "LDG (global)", "STG (global)",
          "HGMMA", "R2UR", "UIADD3", "IMAD", "FFMA", "MUFU", "BAR", "SYNCS (mbarrier)"]:
    print(f"  {k:18s} {lo[k]:9d} {ln[k]:9d} {co[k]:15,d} {cn[k]:15,d}")
