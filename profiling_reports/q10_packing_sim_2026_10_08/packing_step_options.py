#!/usr/bin/env python3
# 2026-10-14: ways to keep "about 64 chats per step" with packing (item 15 / packing doc section 9.6), stage-0.
# (A) Kathir's token-budget micro-batches (his partition(), copied from kathir_sft/sft.py:106-124) on a sample step.
# (B) v1 today: 64 chats per step (reference).
# (C) "pack each step": draw 64 chats per step (shuffled, seed 1234), pack only those with First-Fit-Decreasing.
# (D) "pack once + token budget": pack all chats once (FFD), shuffle the rows, fill each step with rows until it holds
#     at least v1's average trained tokens per step.
import json, sys, math
import numpy as np
d = sys.argv[1]; T = int(json.load(open(d + "/meta.json")).get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
L = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab], dtype=np.int64)
TR = np.array([(r != 65535).sum() for r in lab], dtype=np.int64)
N = len(L)

def partition(lengths, budget, cost):               # kathir_sft/sft.py:106-124, unchanged
    n = len(lengths)
    best, cut = [0.0] + [float("inf")] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            if j - i > 1 and (j - i) * lengths[i] > budget:
                break
            v = best[i] + (j - i) * lengths[i] + cost
            if v < best[j]:
                best[j], cut[j] = v, i
    groups, j = [], n
    while j:
        groups.append((cut[j], j)); j = cut[j]
    return groups[::-1]

print("== (A) Kathir's micro-batches: a small example (lengths sorted, budget 16,384 padded tokens, cost 512 per micro-batch)")
ex = [3000, 2900, 2800, 1200, 1100, 1000, 900, 300, 250, 200]
for i, j in partition(ex, 16384, 512):
    g = ex[i:j]; print(f"   micro-batch: {g} -> {len(g)} rows padded to {g[0]} = {len(g) * g[0]:,} padded tokens, real {sum(g):,}")
p = np.random.default_rng(1234).permutation(N)
step = sorted(L[p[:64]].tolist(), reverse=True)
mbs = partition(step, 16384, 512)
pad = sum((j - i) * step[i] for i, j in mbs)
print(f"== (A) Kathir's micro-batches on one real step of 64 stage-0 chats: {len(mbs)} micro-batches, chats per micro-batch "
      f"{[j - i for i, j in mbs]}, padded tokens {pad:,} for {sum(step):,} real ({100 * sum(step) / pad:.1f}% real); "
      f"our trainer today: 8 micro-batches x 8 rows x 4,096 = 262,144 slots ({100 * sum(step) / 262144:.1f}% real)")
steps_all = [sorted(L[p[k:k + 64]].tolist(), reverse=True) for k in range(0, N - 63, 64)]
nm = [len(partition(s, 16384, 512)) for s in steps_all]
rp = [sum(s) / sum((j - i) * s[i] for i, j in partition(s, 16384, 512)) for s in steps_all]
print(f"   over all {len(steps_all)} steps of an epoch: micro-batches per step min {min(nm)}, mean {np.mean(nm):.1f}, max {max(nm)}; "
      f"real share mean {100 * np.mean(rp):.1f}%")

def ffd(lengths):
    rows = []
    for x in sorted(lengths, reverse=True):
        for r in rows:
            if r[0] + x <= T: r[0] += x; r[1] += 1; break
        else: rows.append([x, 1])
    return rows
print(f"== (B) v1 today: 64 chats per step, {N // 64} full steps per epoch, mean trained tokens per step {64 * TR.mean():,.0f}")
rows_per_step = []; real = 0; slots = 0
for k in range(0, N - 63, 64):
    rs = ffd(L[p[k:k + 64]].tolist()); rows_per_step.append(len(rs)); real += sum(r[0] for r in rs); slots += len(rs) * T
print(f"== (C) pack each step's 64 chats on their own: rows per step min {min(rows_per_step)}, mean {np.mean(rows_per_step):.1f}, "
      f"max {max(rows_per_step)}; real tokens {100 * real / slots:.2f}% (packing all 5,000 at once: 99.60%); "
      f"{len(rows_per_step)} steps per epoch; exactly 64 chats per step")
# (D) pack once, shuffle rows, fill steps to a trained-token budget
order = np.argsort(-L, kind="stable")
rows = []                                           # [used, chats, trained]
for i in order:
    x = int(L[i])
    for r in rows:
        if r[0] + x <= T: r[0] += x; r[1] += 1; r[2] += int(TR[i]); break
    else: rows.append([x, 1, int(TR[i])])
rr = [rows[i] for i in np.random.default_rng(1234).permutation(len(rows))]
budget = 64 * TR.mean(); steps = []; cur = []
for r in rr:
    cur.append(r)
    if sum(x[2] for x in cur) >= budget: steps.append(cur); cur = []
nrow = [len(s) for s in steps]; nch = [sum(x[1] for x in s) for s in steps]; ntr = [sum(x[2] for x in s) for s in steps]
print(f"== (D) pack once ({len(rows)} rows) + fill each step to >= {budget:,.0f} trained tokens: {len(steps)} full steps per epoch "
      f"({len(cur)} rows left over); rows per step min {min(nrow)}, mean {np.mean(nrow):.1f}, max {max(nrow)}; "
      f"chats per step min {min(nch)}, mean {np.mean(nch):.1f}, max {max(nch)}; trained tokens per step mean {np.mean(ntr):,.0f}")
