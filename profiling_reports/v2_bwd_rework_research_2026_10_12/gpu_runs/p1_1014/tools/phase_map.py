#!/usr/bin/env python3
# 2026-10-14 test 1.4: per-phase stall map of the Vj2 backward (g123, lineinfo build, SASS identical to g123).
# address -> C++ line from the "cuda,sass" export; program order + per-instruction metrics from the "sass" export.
# Inlined helper lines (< 420) inherit the phase of the kernel-body line before them (call site).
import csv, re, sys
from collections import Counter, defaultdict
D = sys.argv[1] if len(sys.argv) > 1 else "."
R = list(csv.reader(open(f"{D}/ncu_g123li.source_cuda_sass.csv")))
a2l, cur = {}, None
for r in R[3:]:
    if r and r[0].strip().isdigit(): cur = int(r[0])
    elif len(r) > 2 and r[2].startswith("0x") and cur is not None: a2l[r[2]] = cur
S = list(csv.reader(open(f"{D}/ncu_g123li.source_sass.csv"))); h = S[1]; ix = {c: i for i, c in enumerate(h)}
rows = S[2:]
g = lambda r, c: float(r[ix[c]]) if r[ix[c]] not in ("", "-") else 0.0
PH = [(420, 446, "loader warp (TMA loads)"), (447, 462, "dQ writer warp"), (463, 494, "compute setup"),
      (495, 502, "P0 wait for step data (q_full)"), (503, 515, "P1 LSE loads"), (516, 518, "P2 issue S + dP GEMMs"),
      (519, 519, "P3 wait S GEMM"), (520, 535, "P4 exp + causal mask"), (536, 541, "P5 D loads"), (542, 542, "P6 wait dP GEMM"),
      (543, 557, "P7 dS math + bf16 pack"), (558, 571, "P8 dS write (stmatrix)"), (572, 583, "P9 dV GEMM issue"),
      (584, 585, "P10 dS fence + both-groups barrier"), (586, 591, "P11 dQ + dK GEMM issue"), (592, 593, "P12 wait dV + dQ GEMMs"),
      (594, 594, "P13 wait dQ buffer free (writer)"), (595, 595, "P14 dQ store"), (596, 597, "P15 dQ fence + hand-off"),
      (598, 599, "P16 wait dK GEMM"), (600, 602, "P17 release step buffers"), (603, 640, "epilogue (dK/dV out)")]
def phase_of(line):
    for a, b, n in PH:
        if a <= line <= b: return n
    return None
tot = sum(g(r, "Warp Stall Sampling (All Samples)") for r in rows)
cur_phase = "prologue"; smp = Counter(); ex = Counter(); reas = defaultdict(Counter); order = []
for r in rows:
    line = a2l.get(r[ix["Address"]])
    if line is not None and line >= 420:
        p = phase_of(line)
        if p: cur_phase = p
    if cur_phase not in order: order.append(cur_phase)
    s = g(r, "Warp Stall Sampling (All Samples)"); smp[cur_phase] += s; ex[cur_phase] += g(r, "Instructions Executed")
    for c in h:
        if c.startswith("stall_") and "Not Issued" not in c: reas[cur_phase][c[6:]] += g(r, c)
print(f"Vj2 g123 backward, dense B8 18/6, ncu PC sampling: {tot:,.0f} samples; mapped {len(a2l)} of {len(rows)} instructions to lines")
print(f"{'phase':40s} {'samples %':>9s} {'instr (M)':>10s}  top wait reasons (% of all samples)")
for p in order:
    top = ", ".join(f"{k} {100*v/tot:.1f}" for k, v in reas[p].most_common(3) if v > 0)
    print(f"{p:40s} {100*smp[p]/tot:8.1f}% {ex[p]/1e6:10.2f}  {top}")
