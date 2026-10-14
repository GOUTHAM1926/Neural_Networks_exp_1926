# 2026-10-14: independent check of a packing (reads chat lengths from labels.bin; every chat once; no row over 4096).
# independent checker: reads chat lengths straight from labels.bin (row-by-row loop) and the dumped rows of chat indices
import json, sys
import numpy as np
D, P = sys.argv[1], sys.argv[2]
T = 4096
lab = np.fromfile(D + "/labels.bin", dtype=np.uint16).reshape(-1, T)
Ls = []
for r in lab:
    nz = np.flatnonzero(r != 65535)
    Ls.append(int(nz[-1]) + 1)
rows = json.load(open(P))
flat = [i for r in rows for i in r]
print("rows", len(rows), "| chats placed", len(flat), "| distinct", len(set(flat)), "| all 0..4999 present", sorted(flat) == list(range(len(Ls))))
used = [sum(Ls[i] for i in r) for r in rows]
print("max used", max(used), "<= 4096:", max(used) <= T, "| total empty", sum(T - u for u in used), "| full rows", sum(u == T for u in used))
print("non-full rows (used, empty, chat lengths):")
for r, u in zip(rows, used):
    if u < T: print("  ", u, T - u, sorted((Ls[i] for i in r), reverse=True))
from collections import Counter
print("chats per row:", dict(sorted(Counter(len(r) for r in rows).items())))
