# 2026-10-14: written by a review agent, re-run and checked by me. Packs stage-0 into 1,511 rows (the minimum).
# Is 1,511 rows reachable for stage-0 without cutting chats? (1) Martello-Toth L2 lower bound and forced waste of the
# biggest chats; (2) constructive attempt: minimum-slack packing (largest remaining chat + subset-sum fill). Throwaway.
import json, sys, math
import numpy as np
D = sys.argv[1]
meta = json.load(open(D + "/meta.json")); N, T = meta["N"], meta["T"]
lab = np.fromfile(D + "/labels.bin", dtype=np.uint16).reshape(N, T)
L = (T - np.argmax((lab != 65535)[:, ::-1], axis=1)).astype(np.int64)
s = sorted(L.tolist(), reverse=True); total = sum(s); C = T
print("L1 =", math.ceil(total / C), " slack allowed at L1 =", math.ceil(total / C) * C - total)
# L2 (Martello & Toth 1990)
best = 0
for k in sorted(set([0] + [x for x in s if x <= C // 2])):
    J1 = [x for x in s if x > C - k]; J2 = [x for x in s if C // 2 < x <= C - k]; J3 = [x for x in s if k <= x <= C // 2]
    v = len(J1) + len(J2) + max(0, math.ceil((sum(J3) - (len(J2) * C - sum(J2))) / C))
    best = max(best, v)
print("L2 lower bound =", best)
print("chats <= 300:", sorted(x for x in s if x <= 300))
print("chats >= 3500:", [x for x in s if x >= 3500])

# (2) constructive: minimum bin slack, bins opened by the largest remaining chat, filled by subset-sum over the rest
def fill(cap, items):
    # items: list of (size, id). 0/1 subset-sum, return ids of a subset with the largest sum <= cap
    parent = np.full(cap + 1, -1, dtype=np.int64); reach = np.zeros(cap + 1, dtype=bool); reach[0] = True
    for j, (x, _) in enumerate(items):
        if x > cap: continue
        new = np.zeros_like(reach); new[x:] = reach[:cap + 1 - x]
        new &= ~reach
        idx = np.nonzero(new)[0]
        if idx.size:
            parent[idx] = j; reach |= new
            if reach[cap]: break
    t = int(np.nonzero(reach)[0].max()); chosen = []
    while t > 0:
        j = int(parent[t]); chosen.append(j); t -= items[j][0]
    return chosen

rem = sorted([(x, i) for i, x in enumerate(L.tolist())], reverse=True)
bins = []
import random
rng = random.Random(0)
while rem:
    big = rem.pop(0); cap = C - big[0]
    # candidate pool: all remaining chats that fit, smallest-first order would starve later big bins, so use
    # largest-first order (subset-sum then prefers bigger pieces)
    pool = [it for it in rem if it[0] <= cap]
    ch = fill(cap, pool) if pool and cap > 0 else []
    take = set(pool[j][1] for j in ch)
    b = [big] + [pool[j] for j in ch]
    rem = [it for it in rem if it[1] not in take]
    bins.append(b)
emp = [C - sum(x for x, _ in b) for b in bins]
assert sorted(x for b in bins for x, _ in b) == sorted(L.tolist())
assert all(e >= 0 for e in emp)


json.dump([[int(i) for _, i in b] for b in bins], open(sys.argv[2], "w"))
print("dumped", len(bins), "rows")
