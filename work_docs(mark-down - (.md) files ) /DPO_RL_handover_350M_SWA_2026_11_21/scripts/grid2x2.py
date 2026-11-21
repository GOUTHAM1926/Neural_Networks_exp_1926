#!/usr/bin/env python3
"""grid2x2.py -- 2026-11-21: the 2 x 2 grid of the SFT window study once arm B is in (SWA doc 12.2).
A = kv4 (full) + full SFT, B = kv4 (full) + window SFT, C = kv4_swa3 (window) + window SFT, D = kv4_swa3 + full SFT.
A / C / D: the 2026-11-20 numbers (trainer final val; HF val groups, native attention, all 311 rows). B: from its run
(the trainer's final val from armB27.out / run.log, the groups from valgroups/B.json).
usage: python3 grid2x2.py <B trainer final val> <valgroups/B.json>
"""
import json
import sys

GROUPS = ["total", "chats<=1024", "chats>1024", "pos0-1023", "pos1024-2047", "pos2048+"]
TRAINER = {"A": 1.178506, "C": 1.176781, "D": 1.176567}
NATIVE = {   # SWA doc 12.2, "Val loss by groups", native rows (total = all 311 rows)
    "A": [1.182986, 1.191021, 1.174183, 1.135404, 1.277946, 1.796139],
    "C": [1.181186, 1.187510, 1.174256, 1.132616, 1.279082, 1.803708],
    "D": [1.180943, 1.187694, 1.173547, 1.132838, 1.276007, 1.804080],
}
tB = float(sys.argv[1])
rows = json.load(open(sys.argv[2]))
nat = [r for r in rows if r["mode"] == "native"][0]["loss"]
oth = [r for r in rows if r["mode"] == "other"][0]["loss"]
TRAINER["B"] = tB
NATIVE["B"] = [nat[g] for g in GROUPS]
print("trainer final val: " + "  ".join(f"{k} {TRAINER[k]:.6f}" for k in "ABCD"))
print(f"B HF native trainer rows {nat.get('trainer_rows', float('nan')):.6f} vs trainer {tB:.6f}")
print("\nval loss by group (HF, native attention):")
print(f"{'group':14s}" + "".join(f"{k:>11s}" for k in "ABCD"))
for i, g in enumerate(GROUPS):
    print(f"{g:14s}" + "".join(f"{NATIVE[k][i]:11.6f}" for k in "ABCD"))
print("\nthe two paths through the grid (minus = lower loss after the step):")
pairs = [("A->B", "window in SFT, full-attention base", "A", "B"), ("D->C", "window in SFT, window base", "D", "C"),
         ("A->D", "window in pretraining, full SFT", "A", "D"), ("B->C", "window in pretraining, window SFT", "B", "C"),
         ("A->C", "both", "A", "C")]
print(f"{'step':6s} {'what changes':38s} {'trainer':>9s}" + "".join(f"{g:>14s}" for g in GROUPS))
for name, what, a, b in pairs:
    print(f"{name:6s} {what:38s} {TRAINER[b] - TRAINER[a]:+9.4f}" + "".join(f"{NATIVE[b][i] - NATIVE[a][i]:+14.4f}" for i in range(len(GROUPS))))
print("\ninteraction (A->B) - (D->C) = does the window in SFT cost differently on the two bases:")
print(f"{'':45s} {(TRAINER['B'] - TRAINER['A']) - (TRAINER['C'] - TRAINER['D']):+9.4f}"
      + "".join(f"{(NATIVE['B'][i] - NATIVE['A'][i]) - (NATIVE['C'][i] - NATIVE['D'][i]):+14.4f}" for i in range(len(GROUPS))))
print("\nB served with full attention (its pretraining attention) vs with its window:")
print("".join(f"{g:>14s}" for g in GROUPS))
print("".join(f"{oth[g] - nat[g]:+14.4f}" for g in GROUPS))
