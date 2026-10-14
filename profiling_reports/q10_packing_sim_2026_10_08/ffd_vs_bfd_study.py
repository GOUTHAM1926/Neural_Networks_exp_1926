#!/usr/bin/env python3
# 2026-10-14: First-Fit-Decreasing (FFD) vs Best-Fit-Decreasing (BFD), measured. Rows of T = 4096 slots.
# Both sort longest first. FFD: first row (lowest index) with room. BFD: the row left with the fewest empty slots
# (ties: lowest index). Fast versions: FFD = segment tree over rows (max room), leftmost row with room >= x;
# BFD = for each "room" value 0..T a min-heap of row indices + a segment tree over room values -> smallest room >= x.
# Both fast versions are checked against the simple versions (same rows, same layout).
import json, sys, math, time, heapq, random
import numpy as np
d = sys.argv[1]; T = int(json.load(open(d + "/meta.json")).get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
L = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab], dtype=np.int64)

def simple(lengths, best, count=False):                 # returns rows (lists of indices) and rows checked
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    room, rows, checked = [], [], 0
    for i in order:
        x = lengths[i]; pick = -1
        if best:
            checked += len(room); bestroom = T + 1
            for k, r in enumerate(room):
                if x <= r < bestroom: bestroom, pick = r, k
        else:
            for k, r in enumerate(room):
                checked += 1
                if r >= x: pick = k; break
            else: checked += 0
        if pick < 0: room.append(T - x); rows.append([i])
        else: room[pick] -= x; rows[pick].append(i)
    return (rows, checked) if count else rows

def fast_ffd(lengths):
    n = len(lengths); size = 1
    while size < n: size *= 2
    tree = [-1] * (2 * size)                              # max room in the subtree; -1 = no row yet
    order = sorted(range(n), key=lambda i: -lengths[i]); rows = []; nrows = 0
    for i in order:
        x = lengths[i]
        if tree[1] >= x:
            k = 1
            while k < size: k = 2 * k if tree[2 * k] >= x else 2 * k + 1
            r = k - size; rows[r].append(i); v = tree[k] - x
        else:
            r = nrows; nrows += 1; rows.append([i]); k = r + size; v = T - x
        tree[k] = v; k //= 2
        while k: tree[k] = max(tree[2 * k], tree[2 * k + 1]); k //= 2
    return rows

def fast_bfd(lengths):
    size = 1
    while size < T + 1: size *= 2
    present = [0] * (2 * size)                            # 1 if some row has exactly this room (leaf), OR above
    heaps = [[] for _ in range(T + 1)]
    def setp(c, v):
        k = c + size; present[k] = v; k //= 2
        while k: present[k] = present[2 * k] | present[2 * k + 1]; k //= 2
    def smallest_at_least(x):                             # smallest room value >= x that some row has, or -1
        def rec(k, lo, hi):
            if hi < x or not present[k]: return -1
            if lo == hi: return lo
            mid = (lo + hi) // 2
            r = rec(2 * k, lo, mid)
            return r if r >= 0 else rec(2 * k + 1, mid + 1, hi)
        return rec(1, 0, size - 1)
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i]); rows = []; room = []
    for i in order:
        x = lengths[i]; c = smallest_at_least(x)
        if c < 0:
            r = len(rows); rows.append([i]); room.append(T - x)
        else:
            r = heapq.heappop(heaps[c])
            if not heaps[c]: setp(c, 0)
            rows[r].append(i); room[r] -= x
        heapq.heappush(heaps[room[r]], r); setp(room[r], 1)
    return rows

def stats(rows, lengths):
    used = np.array([sum(lengths[i] for i in r) for r in rows]); emp = T - used; c = np.array([len(r) for r in rows])
    att = np.array([sum(lengths[i] ** 2 for i in r) for r in rows], dtype=float)
    return dict(rows=len(rows), real=100 * used.sum() / (len(rows) * T), full=int((emp == 0).sum()),
                e1_16=int(((emp >= 1) & (emp <= 16)).sum()), e17_64=int(((emp >= 17) & (emp <= 64)).sum()),
                e65_256=int(((emp >= 65) & (emp <= 256)).sum()), e256=int((emp > 256).sum()), single=int((c == 1).sum()),
                maxc=int(c.max()), cpr=" ".join(f"{k}:{int((c == k).sum())}" for k in range(1, c.max() + 1)),
                att_mean=att.mean(), att_max=att.max(), att_cv=att.std() / att.mean())

Ll = L.tolist(); lb = math.ceil(sum(Ll) / T)
print(f"== E1 stage-0 real data: {len(Ll)} chats, lower bound {lb} rows")
f_s, f_chk = simple(Ll, False, True); b_s, b_chk = simple(Ll, True, True); f_f = fast_ffd(Ll); b_f = fast_bfd(Ll)
print(f"   fast versions give the same layout as the simple ones: FFD {f_s == f_f}, BFD {b_s == b_f}")
for name, rows in (("FFD", f_s), ("BFD", b_s)):
    s = stats(rows, Ll)
    print(f"   {name}: {s['rows']} rows, {s['real']:.2f}% real | empty per row: 0: {s['full']}, 1-16: {s['e1_16']}, 17-64: {s['e17_64']}, "
          f"65-256: {s['e65_256']}, >256: {s['e256']} | single-chat rows {s['single']}, most chats in a row {s['maxc']}")
    print(f"        chats per row -> rows: {s['cpr']}")
    print(f"        attention work per row (sum of L^2): mean {s['att_mean']:.3e}, largest {s['att_max']:.3e}, spread (std/mean) {s['att_cv']:.3f}")
row_of = {}
for k, r in enumerate(f_s):
    for i in r: row_of[i] = ("F", tuple(sorted(r)))
same = sum(1 for r in b_s if ("F", tuple(sorted(r))) in set(row_of.values()))
print(f"   rows that are identical in both layouts: {same} of {len(b_s)}")
print(f"   simple versions, rows looked at per chat: FFD {f_chk / len(Ll):.1f}, BFD {b_chk / len(Ll):.1f} (BFD always looks at every open row)")

def compare(gen, sizes_counts, label):
    print(f"== {label}")
    for n, k in sizes_counts:
        w = t = l = 0; diffs = []; gaps = []
        for s in range(k):
            x = gen(n, s); a = len(fast_ffd(x)); b = len(fast_bfd(x)); lbx = math.ceil(sum(x) / T)
            if a < b: w += 1
            elif a == b: t += 1
            else: l += 1
            diffs.append(a - b); gaps.append(min(a, b) - lbx)
        print(f"   {n:6d} chats x {k:3d} sets: FFD fewer rows in {w}, same in {t}, BFD fewer rows in {l} | "
              f"mean (FFD rows - BFD rows) {np.mean(diffs):+.3f} | mean rows above the lower bound {np.mean(gaps):.2f}")
compare(lambda n, s: np.random.default_rng(1000 + s).choice(L, n).tolist(), ((100, 300), (1000, 300), (5000, 60), (20000, 10)),
        "E2 random sets drawn from our real chat lengths (with replacement)")
gens = {
    "uniform 1..4096": lambda n, s: np.random.default_rng(s).integers(1, T + 1, n).tolist(),
    "uniform 1..2048": lambda n, s: np.random.default_rng(s).integers(1, T // 2 + 1, n).tolist(),
    "uniform 1024..2048": lambda n, s: np.random.default_rng(s).integers(T // 4, T // 2 + 1, n).tolist(),
    "mix: 70% short 100..600, 30% long 2800..4000": lambda n, s: [int(v) for v in np.where(np.random.default_rng(s).random(n) < 0.7,
        np.random.default_rng(s + 7).integers(100, 601, n), np.random.default_rng(s + 9).integers(2800, 4001, n))],
    "hard mix: just over T/2, T/3, T/4": lambda n, s: [int(T // q + 1 + v) for q, v in zip(np.random.default_rng(s).choice([2, 3, 4], n),
        np.random.default_rng(s + 3).integers(0, 40, n))],
}
for g, f in gens.items():
    compare(f, ((1000, 200),), f"E3 synthetic lengths: {g}")

print("== E4 TRL-style groups of 1,000 chats (stage-0)")
for label, order in (("no shuffle (TRL default, shuffle_dataset=False)", list(range(len(Ll)))),
                     ("shuffled, seed 1234", np.random.default_rng(1234).permutation(len(Ll)).tolist())):
    for name, fn in (("FFD", fast_ffd), ("BFD", fast_bfd)):
        rows = 0; big = 0
        for g in range(0, len(order), 1000):
            part = [Ll[i] for i in order[g:g + 1000]]; rs = fn(part); rows += len(rs)
            big += sum(1 for r in rs if T - sum(part[i] for i in r) > 256)
        print(f"   {label:48s} {name}: {rows} rows, {100 * sum(Ll) / (rows * T):.2f}% real, {big} rows with >256 empty")

print("== E5 speed (seconds; pure Python; fast = tree versions; simple = linear scan)")
for n in (5000, 50000, 200000):
    x = np.random.default_rng(7).choice(L, n).tolist() if n != 5000 else Ll
    line = f"   {n:7d} chats:"
    for name, fn in (("fast FFD", fast_ffd), ("fast BFD", fast_bfd)):
        t0 = time.time(); r = fn(x); line += f" {name} {time.time() - t0:7.2f} s ({len(r)} rows);"
    if n <= 50000:
        for name, best in (("simple FFD", False), ("simple BFD", True)):
            t0 = time.time(); r = simple(x, best); line += f" {name} {time.time() - t0:7.2f} s;"
    else:
        line += " simple versions skipped (too slow in Python at this size)"
    print(line)

# pads' share of today's (v1) work, with pack_sim.py's work model
d_model, layers, V = 2304, 26, 50304
lin = 6 * (1.4722e9 + V * d_model); att = 6 * 2 * 2304 * layers / 2
v1 = len(Ll) * (lin * T + att * T * T); useful = sum(lin * x + att * x * x for x in Ll)
print(f"== v1 (one chat per row): pads are {100 * (1 - sum(Ll) / (len(Ll) * T)):.1f}% of the slots and {100 * (1 - useful / v1):.1f}% of the work "
      f"(pack_sim.py's model; attention is {100 * len(Ll) * att * T * T / v1:.1f}% of v1's work)")
