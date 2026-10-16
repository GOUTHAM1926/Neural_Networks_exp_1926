#!/usr/bin/env python3
"""2026-10-16 scoreboard tables (backward first): ours = mean of the 2 rounds (before / after the libraries)."""
import re, sys, glob, collections
D = sys.argv[1] if len(sys.argv) > 1 else "sb_1016"
pat = re.compile(r"^BENCH layout=(\S+) impl=(\S+) dir=(\S+) med_ms=([\d.]+)")
def read(f):
    out = {}
    for line in open(f):
        m = pat.match(line)
        if m: out[(m[1], m[2], m[3])] = float(m[4])
    return out
def mean2(a, b):
    return {k: (a[k] + b[k]) / 2 for k in a if k in b}
spread = []
def ours(prefix):
    r1, r2 = read(f"{D}/{prefix}_r1.log"), read(f"{D}/{prefix}_r2.log")
    for k in r1:
        if k in r2: spread.append(abs(r1[k] - r2[k]) / min(r1[k], r2[k]))
    return mean2(r1, r2)
def table(title, layouts, vj2, vj1d, libs, our_impl, rivals, d):
    print(f"\n### {title} -- {d}: time ratio = ours Vj2 / rival (below 1 = ours faster)\n")
    print("| layout | ours Vj2 (ms) | " + " | ".join(r for r, _ in rivals) + " |")
    print("|---|---|" + "---|" * len(rivals))
    wins = collections.Counter(); ratios = collections.defaultdict(list)
    for L in layouts:
        o = vj2.get((L, our_impl, d))
        if o is None: continue
        cells = []
        for name, src in rivals:
            v = (vj1d.get((L, our_impl, d)) if src == "vj1d" else libs.get((L, src, d)))
            if v is None: cells.append("-"); continue
            r = o / v; ratios[name].append(r); wins[name] += r < 1
            cells.append(f"{r:.3f} ({v:.3f})")
        print(f"| {L} | {o:.3f} | " + " | ".join(cells) + " |")
    n = len([L for L in layouts if (L, our_impl, d) in vj2])
    print("| **mean ratio (ours faster on)** | | " + " | ".join(
        f"**{sum(ratios[nm]) / len(ratios[nm]):.3f}** ({wins[nm]}/{len(ratios[nm])})" if ratios[nm] else "-" for nm, _ in rivals) + " |")
dv2, dv1, dl = ours("dense_vj2"), ours("dense_vj1d"), read(f"{D}/dense_libs.log")
pv2, pv1, pl = ours("packed_vj2"), ours("packed_vj1d"), read(f"{D}/packed_libs.log")
dlay = sorted({k[0] for k in dv2}, key=lambda s: (int(re.search(r"B(\d+)", s)[1]), s))
play = list(dict.fromkeys(k[0] for k in pv2))
drv = [("Vj1d", "vj1d"), ("FA3", "fa3"), ("FA3 det", "fa3_det"), ("FA4", "fa4"), ("cuDNN", "sdpa_cudnn"), ("FA2", "sdpa_flash")]
prv = [("Vj1d", "vj1d"), ("FA3 varlen", "fa3_varlen"), ("FA3 det", "fa3_varlen_det"), ("FA4 varlen", "fa4_varlen"),
       ("cuDNN varlen", "cudnn_varlen_gqa"), ("Flex", "flex_docmask"), ("FA2 (torch varlen)", "torch_varlen")]
for d in ("bwd", "fwd"):
    table("DENSE (normal causal attention), 13 shapes", dlay, dv2, dv1, dl, "ours_dense", drv, d)
    table("PACKED (SFT rows, per-chat mask), ours = seqlens plan", play, pv2, pv1, pl, "ours_A_plan", prv, d)
# packed: seqlens vs our own dense full row (how much packing saves)
print("\n### PACKED: ours seqlens vs ours dense full row (same kernels, no chat mask)\n")
print("| layout | bwd seqlens / dense row | fwd seqlens / dense row |\n|---|---|---|")
for L in play:
    b = pv2[(L, "ours_A_plan", "bwd")] / pv2[(L, "ours_dense_fullrow", "bwd")]
    f = pv2[(L, "ours_A_plan", "fwd")] / pv2[(L, "ours_dense_fullrow", "fwd")]
    print(f"| {L} | {b:.3f} | {f:.3f} |")
spread.sort()
print(f"\nours round 1 vs round 2: median difference {100 * spread[len(spread) // 2]:.2f}%, max {100 * spread[-1]:.2f}% ({len(spread)} pairs)")
