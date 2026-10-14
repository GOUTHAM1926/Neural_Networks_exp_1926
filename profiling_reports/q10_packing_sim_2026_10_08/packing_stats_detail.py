#!/usr/bin/env python3
# 2026-10-14: the extra numbers of work_docs/sft_packing_algorithms_bfd_ffd.md (examples 15, 16, 18 and section 9).
# Stage-0 (5,000 UltraChat chats, rows of 4096): first fit decreasing rows in detail; TRL-style packing per group
# (bfd = best fit decreasing per group; wrapped = one stream per group cut every 4096); step counts per step size.
import json, sys, math, numpy as np
from collections import Counter
d = sys.argv[1]; T = int(json.load(open(d + "/meta.json")).get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
L = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab], dtype=np.int64)      # slots each chat fills
trained = np.array([(r != 65535).sum() for r in lab], dtype=np.int64)               # answer tokens trained per chat
N, real = len(L), int(L.sum())
def fit(order, best):
    rows = []                                                                     # [used, n_chats, trained]
    for i in order:
        x = int(L[i]); ok = [r for r in rows if r[0] + x <= T]
        if not ok: rows.append([x, 1, int(trained[i])]); continue
        r = min(ok, key=lambda r: T - r[0]) if best else ok[0]
        r[0] += x; r[1] += 1; r[2] += int(trained[i])
    return rows
order = sorted(range(N), key=lambda i: -L[i])
ffd = fit(order, False)
print(f"== stage-0: {N} chats, {real:,} slots of real tokens, {int(trained.sum()):,} trained (answer) tokens")
print(f"   chat length: mean {L.mean():.1f}, median {np.median(L):.0f}, shortest {L.min()}, longest {L.max()}")
print(f"   minimum rows = ceil({real:,} / {T}) = {math.ceil(real / T)}; {math.ceil(real / T)} rows hold {math.ceil(real / T) * T:,} slots, "
      f"{math.ceil(real / T) - 1} rows hold {(math.ceil(real / T) - 1) * T:,}")
print(f"== first fit decreasing: {len(ffd)} rows")
c = Counter(r[1] for r in ffd)
print("   chats per row -> number of rows: " + ", ".join(f"{k}: {c[k]}" for k in sorted(c)))
emp = np.array([T - r[0] for r in ffd])
bins = [(0, 0), (1, 16), (17, 64), (65, 256), (257, 4096)]
print("   empty slots per row -> number of rows: " + ", ".join(f"{a}-{b}: {int(((emp >= a) & (emp <= b)).sum())}" for a, b in bins))
print(f"   empty slots total {int(emp.sum()):,} = {emp.sum() / T:.2f} rows' worth; mean {emp.mean():.1f}, median {np.median(emp):.0f}, largest {emp.max()}")
one = [r for r in ffd if r[1] == 1]
print(f"   rows holding a single chat: {len(one)}; that chat's length: shortest {min(r[0] for r in one)}, longest {max(r[0] for r in one)}")
# the longest chats (they open the first rows)
print(f"   the 5 longest chats: {sorted(L.tolist(), reverse=True)[:5]}; the 5 shortest: {sorted(L.tolist())[:5]}")
# TRL-style groups
p = np.random.default_rng(1234).permutation(N)
print("== packing per group of G chats (chats shuffled with seed 1234, as TRL shuffles before packing)")
for G in (10, 100, 1000, 5000):
    groups = [p[i:i + G] for i in range(0, N, G)]
    bfd_rows = 0; partial = 0
    for g in groups:
        rows = fit(sorted(g, key=lambda i: -L[i]), True); bfd_rows += len(rows); partial += sum(1 for r in rows if T - r[0] > 256)
    wr = sum(math.ceil(int(L[g].sum()) / T) for g in groups)
    print(f"   G = {G:5d}: {len(groups):4d} groups | bfd {bfd_rows} rows ({100 * real / (bfd_rows * T):.2f}% real, {partial} rows with >256 empty)"
          f" | wrapped {wr} rows ({100 * real / (wr * T):.2f}% real, {len(groups)} shorter last rows)")
# step counts
print("== optimizer steps per epoch (1 GPU, 8 rows per micro-batch)")
cpr = N / len(ffd)
tr_row = trained.sum() / len(ffd)
print(f"   v1 (one chat per row): 5000 rows; 64 rows per step = 8 micro-batches x 8 rows -> {N / 64:.3f} steps per epoch, "
      f"64 chats, ~{64 * trained.mean():,.0f} trained tokens per step")
for rps in (64, 32, 24, 16, 8):
    print(f"   v2 packed, {rps:2d} rows per step ({rps // 8} micro-batches): {len(ffd) / rps:6.2f} steps per epoch, "
          f"~{rps * cpr:6.1f} chats and ~{rps * tr_row:,.0f} trained tokens per step")
print(f"   v2 packed, ~{64 / cpr:.1f} rows per step (keeps 64 chats per step): {N / 64:.2f} steps per epoch")

# ---- the trainer's own formula (bluscript_zero_ckpt.cpp:943-958, SftDataLoader.h:56), 1 GPU, 2 epochs ----
def trainer(rows_total, B, rps, epochs=2, warm=0.03):
    if rps % B: return None
    M = rows_total // B; ga = rps // B; steps = int(epochs * M / ga)
    return dict(M=M, unused=rows_total - M * B, ga=ga, steps=steps, warm=max(1, math.floor(warm * steps + 0.5)))
print("== what the trainer would run (2 epochs, 1 GPU; max_steps = epochs x micro-batches per epoch / GRAD_ACCUM)")
for name, rows_total, B, rps in (("v1 today", N, 8, 64), ("v2", len(ffd), 8, 64), ("v2", len(ffd), 8, 24),
                                 ("v2", len(ffd), 8, 16), ("v2", len(ffd), 4, 20), ("v2", len(ffd), 8, 20)):
    t = trainer(rows_total, B, rps)
    if t is None:
        print(f"   {name:8s} B {B} rows/step {rps}: REFUSED by the trainer (rows per step must be a multiple of B x GPUs = {B})"); continue
    cps = rps * (N / rows_total)
    print(f"   {name:8s} B {B} rows/step {rps:2d}: {t['M']} micro-batches/epoch ({t['unused']} rows unused), GRAD_ACCUM {t['ga']}, "
          f"{t['steps']} steps for 2 epochs, {t['warm']} warmup steps, ~{cps:.1f} chats per step")
# ---- the tiny group example of example 16 (rows of 10; chats 6, 6, 4, 4) ----
def tiny(Lx, Tx):
    rows = []
    for x in sorted(Lx, reverse=True):
        ok = [r for r in rows if sum(r) + x <= Tx]
        (min(ok, key=lambda r: Tx - sum(r)).append(x)) if ok else rows.append([x])
    return rows
print("== tiny group example, rows of 10, chats 6, 6, 4, 4")
print(f"   all four in one group: {tiny([6, 6, 4, 4], 10)}")
print(f"   group 1 = [6, 6]: {tiny([6, 6], 10)}   group 2 = [4, 4]: {tiny([4, 4], 10)}")

# ---- example 15: gaps no chat can ever fill (smaller than the shortest chat) ----
short = int(L.min())
small = [T - r[0] for r in ffd if 0 < T - r[0] < short]
print(f"== gaps smaller than the shortest chat ({short} slots): {len(small)} rows, {sum(small):,} empty slots in them; "
      f"rows with a gap of {short} or more: {sum(1 for r in ffd if T - r[0] >= short)}")
print(f"== the row holding a single chat: {[ (r[0], T - r[0]) for r in ffd if r[1] == 1 ]} (used, empty)")
print(f"== trained tokens per packed row: {trained.sum() / len(ffd):,.1f}; per chat {trained.mean():,.1f}")
