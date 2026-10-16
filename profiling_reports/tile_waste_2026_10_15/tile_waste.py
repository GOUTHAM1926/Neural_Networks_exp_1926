#!/usr/bin/env python3
"""Count, per query-key pair, the attention work our kernels COMPUTE vs the work that is REAL (key <= query, same chat),
on the stage-0 packed data, following the kernels' own tile rules (2026-10-15 code):
  forward v62:  64-query tiles x 128-key tiles; design A restarts the grid at every segment (chat or pad tail):
                q-tile qt of a segment visits key tiles 0 .. floor((64qt+63)/128); q-tiles past L are skipped
                (GQA_fwd_sm90.cu:488-491).
  backward Vj2: 128-key tiles x 64-query steps; design A: key tile kt visits steps 2kt .. ceil(L/64)-1
                (GQA_bwd_vj2_sm90.cu:468).
  design B (removed): the same tiles on the ROW grid; forward pair p starts at key tile seg_start[128p]/128,
                backward key tile kt visits steps q >= 2kt while seg_start[64q] <= 128kt+127.
  v1: one chat per 4096-token row, plain causal over the whole row (pads included).
Counts are per head (every head does the same)."""
import sys, numpy as np
d = sys.argv[1]; T = 4096
L = np.fromfile(d + "/seq_lengths.bin", dtype=np.int32); off = np.fromfile(d + "/seq_offsets.bin", dtype=np.int64)
N = len(off) - 1
def cdiv(a, b): return -(-a // b)
# ---- per segment, design A
def fwdA(l):   # computed pairs, and the split: real, diagonal-tile waste, rows past the end
    nq = cdiv(l, 64); comp = 0; diag = 0; tail = 0
    for qt in range(nq):
        nk = (64 * qt + 63) // 128 + 1
        comp += 64 * 128 * nk
        rows = np.arange(64 * qt, 64 * qt + 64)
        real_rows = rows[rows < l]
        tail += (64 - len(real_rows)) * 128 * nk
        diag += int((128 * nk - (real_rows + 1)).sum())       # keys computed but hidden, for real rows
    return comp, diag, tail
def bwdA(l):
    nq = cdiv(l, 64); comp = 0
    for kt in range(cdiv(l, 128)): comp += 128 * 64 * (nq - 2 * kt)
    return comp
tot = dict(real=0, fA=0, fA_diag=0, fA_tail=0, fA_pad=0, bA=0, bA_pad=0, fB=0, bB=0, fV=0, bV=0, fDense=0, bDense=0)
fV_row = sum(64 * 128 * ((64 * qt + 63) // 128 + 1) for qt in range(T // 64))
bV_row = sum(128 * 64 * (T // 64 - 2 * kt) for kt in range(T // 128))
nchats = 0; npads = 0
for r in range(N):
    lens = [int(x) for x in L[off[r]:off[r + 1]]]
    segs = [(x, False) for x in lens]
    if sum(lens) < T: segs.append((T - sum(lens), True))
    seg = np.empty(T, dtype=np.int64); t = 0
    for l, pad in segs:
        seg[t:t + l] = t; t += l
        c, dg, tl = fwdA(l); b = bwdA(l)
        tot["fA"] += c; tot["bA"] += b
        if pad: tot["fA_pad"] += c; tot["bA_pad"] += b; npads += 1
        else:
            tot["real"] += l * (l + 1) // 2; tot["fA_diag"] += dg; tot["fA_tail"] += tl; nchats += 1
            tot["fV"] += fV_row; tot["bV"] += bV_row              # v1: this chat alone in a padded 4096 row
    for qt in range(T // 64):                                    # design B on the row grid
        n0 = seg[128 * (qt // 2)] // 128
        tot["fB"] += 64 * 128 * ((64 * qt + 63) // 128 + 1 - n0)
    for kt in range(T // 128):
        q = 2 * kt
        while q < T // 64 and seg[64 * q] <= 128 * kt + 127: q += 1
        tot["bB"] += 128 * 64 * (q - 2 * kt)
    tot["fDense"] += fV_row; tot["bDense"] += bV_row             # the packed row as ONE causal sequence (wrong, no mask)
R = tot["real"]
print(f"stage-0 packed: {N} rows, {nchats} chats, {npads} pad tails; REAL query-key pairs (key <= query, same chat) = {R:,} per head\n")
print(f"{'':44s} {'forward':>15s} {'x real':>7s} {'backward':>15s} {'x real':>7s}")
for name, f, b in [("v1 (one chat per padded 4096 row)", "fV", "bV"),
                   ("packed, NO chat mask (wrong: chats mix)", "fDense", "bDense"),
                   ("packed, design B (row grid; removed)", "fB", "bB"),
                   ("packed, design A (grid per chat; ours)", "fA", "bA")]:
    print(f"{name:44s} {tot[f]:>15,} {tot[f] / R:>7.3f} {tot[b]:>15,} {tot[b] / R:>7.3f}")
print(f"\ndesign A forward, where the computed-but-unused pairs are (per head):")
w = tot["fA"] - R
for k, lab in [("fA_diag", "inside the diagonal tiles (key > query; also keys past a chat's end)"),
               ("fA_tail", "query rows past a chat's end in its last 64-row tile"),
               ("fA_pad", "the pad-tail segments (whole rows of padding)")]:
    print(f"  {lab:72s} {tot[k]:>14,} = {100 * tot[k] / tot['fA']:5.2f}% of computed")
print(f"  {'total thrown away':72s} {w:>14,} = {100 * w / tot['fA']:5.2f}% of computed")
print(f"  (a plain causal sequence of 4096 tokens with the same tiles throws away {100 * (fV_row - T * (T + 1) // 2) / fV_row:.2f}% too)")
