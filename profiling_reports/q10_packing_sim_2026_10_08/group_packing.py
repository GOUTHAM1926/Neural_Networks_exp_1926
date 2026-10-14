#!/usr/bin/env python3
# 2026-10-14: TRL packs inside each dataset.map batch (default 1,000 chats), not across the whole set. Stage-0 rows when
# the 5,000 chats are shuffled (seed 1234) and packed per group of G chats, best fit decreasing, vs the whole set at once.
import json, sys, math, numpy as np
d = sys.argv[1]; T = int(json.load(open(d + "/meta.json")).get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
L = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab], dtype=np.int64)
def bfd(s):
    rows = []
    for x in sorted(s, reverse=True):
        best = None
        for r in rows:
            if r[0] + x <= T and (best is None or r[0] > best[0]): best = r
        if best is None: rows.append([x])
        else: best[0] += x
    return len(rows)
p = np.random.default_rng(1234).permutation(L).tolist()
print(f"chats {len(L)}, real tokens {int(L.sum()):,}, minimum rows {math.ceil(L.sum() / T)}")
for G in (100, 1000, 5000):
    rows = sum(bfd(p[i:i + G]) for i in range(0, len(p), G))
    print(f"groups of {G:5d} chats: {rows} rows, real tokens {100 * L.sum() / (rows * T):.2f}%")
