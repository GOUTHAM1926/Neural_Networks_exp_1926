#!/usr/bin/env python3
# 2026-10-14: every worked example of work_docs/packing_rules_and_long_chats.md, computed (not hand-made).
# first fit decreasing (FFD), best fit decreasing (BFD), plain first fit in arrival order, and TRL's three strategies
# for chats longer than the row: bfd (cut the end), bfd_split (cut into pieces of T, last piece shorter), wrapped
# (one stream, cut every T tokens). Chat lengths here = positions in a row.
import math
def first_fit(order, T):
    rows = []
    for x in order:
        for r in rows:
            if sum(r) + x <= T: r.append(x); break
        else: rows.append([x])
    return rows
def ffd(L, T): return first_fit(sorted(L, reverse=True), T)
def bfd(L, T):
    rows = []
    for x in sorted(L, reverse=True):
        best = None
        for r in rows:
            if sum(r) + x <= T and (best is None or T - sum(r) < T - sum(best)): best = r
        if best is None: rows.append([x])
        else: best.append(x)
    return rows
def show(name, rows, T):
    real = sum(map(sum, rows))
    print(f"  {name:28s} {len(rows)} rows: " + "  ".join(f"{r}(empty {T-sum(r)})" for r in rows)
          + f"  | real {real}/{len(rows)*T} = {100*real/(len(rows)*T):.1f}%")
def case(title, L, T, extra=None):
    print(f"\n{title}\n  T = {T}, chats {L}, sum {sum(L)}, minimum rows = ceil({sum(L)}/{T}) = {math.ceil(sum(L)/T)}")
    if max(L) <= T: show("first fit decreasing", ffd(L, T), T); show("best fit decreasing", bfd(L, T), T)
    if extra: extra(L, T)
def unsorted(L, T): show("first fit, arrival order", first_fit(L, T), T)
def long_chats(L, T):
    keep = [x for x in L if x <= T]; print(f"  drop (ours, torchtitan): chats kept {keep}, lost {[x for x in L if x > T]}")
    show("  -> then FFD", ffd(keep, T), T)
    cut = [min(x, T) for x in L]; print(f"  bfd (TRL default, cut the end): chats become {cut}, tokens lost {sum(L)-sum(cut)}")
    show("  -> then BFD", bfd(cut, T), T)
    pieces = [p for x in L for p in ([T] * (x // T) + ([x % T] if x % T else []))]
    print(f"  bfd_split (cut into pieces of {T}): pieces {pieces}, tokens lost 0")
    show("  -> then BFD", bfd(pieces, T), T)
    names = "ABCDEFGH"; stream = "".join(names[i] * x for i, x in enumerate(L))
    rows = [stream[i:i + T] for i in range(0, len(stream), T)]
    print(f"  wrapped (one stream in arrival order, cut every {T}): " + " | ".join(rows) + f"  ({len(rows)} rows)")
case("E1 why pack", [6, 3, 5, 2], 8)
case("E2 user's example", [4000, 3000, 2500, 300, 200, 100], 4000)
case("E3 rules pick different rows, same count", [3796, 3196, 700, 150], 4096)
case("E4 best fit needs fewer rows", [15, 9, 7, 4, 3, 2], 20)
case("E5 first fit needs fewer rows", [15, 9, 7, 3, 2, 2, 2], 20)
case("E6 why sort longest first", [3, 3, 3, 7, 7, 7], 10, unsorted)
case("E7 a chat exactly T long", [10, 5, 5], 10)
case("E8 every chat just over half a row", [6, 6, 6, 6], 10)
case("E9 many one-token chats", [1] * 25, 10)
case("E10 equal halves", [5] * 7, 10)
case("E12 one chat longer than T", [13, 4, 3], 10, long_chats)
case("E13 one chat far longer than T", [25, 2], 10, long_chats)

# ---- E17/E18 (added 2026-10-14): named chats, rows of 4096, the running example of section 5 of the doc ----
def named_fit(chats, T, best):
    rows = []                                                    # each row: list of (name, length)
    for name, x in sorted(chats, key=lambda c: -c[1]):
        used = [sum(l for _, l in r) for r in rows]
        ok = [i for i in range(len(rows)) if used[i] + x <= T]
        if not ok: rows.append([(name, x)]); continue
        i = min(ok, key=lambda i: T - used[i]) if best else ok[0]
        rows[i].append((name, x))
    return rows
def show_named(label, rows, T):
    print(f"  {label}: {len(rows)} rows")
    for k, r in enumerate(rows):
        u = sum(l for _, l in r)
        print(f"    row {k}: " + " | ".join(f"{n} ({l})" for n, l in r) + f" | empty {T-u}  -> {100*u/T:.1f}% used")
def wrapped_named(chats, T):
    rows, cur, pos = [], [], 0
    for name, x in chats:
        start = 1
        while x > 0:
            take = min(x, T - pos); cur.append((name, start, start + take - 1)); x -= take; pos += take; start += take
            if pos == T: rows.append(cur); cur, pos = [], 0
    if cur: rows.append(cur)
    return rows
def show_wrapped(chats, T):
    rows = wrapped_named(chats, T); print(f"  wrapped (arrival order, cut every {T}): {len(rows)} rows")
    for k, r in enumerate(rows):
        u = sum(b - a + 1 for _, a, b in r)
        print(f"    row {k}: " + " | ".join(f"{n} tokens {a}-{b}" for n, a, b in r) + f" | empty {T-u}")
T = 4096
five = [("A", 1500), ("B", 800), ("C", 3000), ("D", 500), ("E", 2000)]
print(f"\nE17 five chats, rows of 4096: {five}, sum {sum(x for _, x in five)}")
show_named("first fit decreasing", named_fit(five, T, False), T); show_named("best fit decreasing", named_fit(five, T, True), T)
show_wrapped(five, T)
long5 = [("A", 1500), ("B", 800), ("C", 5500), ("D", 500), ("E", 2000)]
print(f"\nE18 same, but C = 5500 (longer than 4096): {long5}")
show_named("drop C, then best fit", named_fit([c for c in long5 if c[1] <= T], T, True), T)
show_named("bfd: C cut to 4096 (1404 lost), then best fit", named_fit([(n, min(x, T)) for n, x in long5], T, True), T)
show_named("bfd_split: C -> C1 (4096) + C2 (1404), then best fit",
           named_fit([("A", 1500), ("B", 800), ("C1", 4096), ("C2", 1404), ("D", 500), ("E", 2000)], T, True), T)
show_wrapped(long5, T)
