#!/usr/bin/env python3
# 2026-10-14: first fit decreasing (FFD) vs best fit decreasing (BFD) for packing chats into rows of T = 4096 slots.
# Laptop, CPU only. Prints every number of the FFD-vs-BFD comparison; writes nothing to disk.
# Both rules sort the chats longest first (equal lengths: lower chat index first), then place them one by one:
#   FFD: put the chat into the FIRST row (lowest index) with enough empty slots, else open a new row.
#   BFD: of the rows with enough empty slots, the row left with the FEWEST empty slots (ties: earliest row), else a new row.
# E1 stage-0 layouts in detail | E2 resampled real lengths | E3 synthetic lengths (E3b: every small instance, T = 20)
# E4 TRL-style groups of 1,000 | E5 speed of naive and fast versions, and TRL's own BFD code (from the local clone)
# Usage: python3 -I ffd_vs_bfd_full_study.py [<stage0 dir>/train] [<trl clone>]      (about 10 minutes on one core)
import ast, gc, itertools, json, math, os, platform, subprocess, sys, time
from collections import Counter, defaultdict, deque
from heapq import heappop, heappush
from typing import Literal
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/blu-bridge016/Downloads/sft_data/stage0_ultrachat_5k_T4096/train"
TRL_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/blu-bridge016/Downloads/sft_research/trl"
START = time.perf_counter()


# ============================================================================ the packing rules
# Every function takes xs = chat lengths in placement order (already sorted longest first) and T. It returns
# (row of every chat, in placement order; empty slots of every row). Rows are numbered in the order they are opened.

def ffd_naive(xs, T):
    """FFD, plain: scan the rows from the first; stop at the first row with enough empty slots."""
    space = []; rows = []
    for x in xs:
        for r, s in enumerate(space):
            if s >= x:
                space[r] = s - x
                break
        else:
            r = len(space); space.append(T - x)
        rows.append(r)
    return rows, space


def bfd_naive(xs, T):
    """BFD, plain: scan ALL open rows, keep the one with the fewest empty slots that still fits (ties: earliest)."""
    space = []; rows = []
    for x in xs:
        best = -1; best_s = T + 1
        for r, s in enumerate(space):
            if x <= s < best_s:
                best = r; best_s = s
        if best < 0:
            best = len(space); space.append(T - x)
        else:
            space[best] = best_s - x
        rows.append(best)
    return rows, space


def ffd_fast(xs, T):
    """FFD in O(log n) per chat: a max-tree over row slots (leaf r = empty slots of row r; rows not opened yet = T) finds
    the leftmost row with at least x empty slots. Unopened rows all lie right of the open ones, so when no open row fits
    the search lands on the next new row by itself."""
    n = len(xs); size = 1
    while size < n: size <<= 1
    tree = [T] * (2 * size)
    rows = []; nrows = 0
    for x in xs:
        i = 1
        while i < size:
            i <<= 1
            if tree[i] < x: i += 1
        r = i - size
        tree[i] -= x
        i >>= 1
        while i:                                   # refresh the maxima above; stop when one does not change
            a = tree[2 * i]; b = tree[2 * i + 1]
            m = a if a >= b else b
            if tree[i] == m: break
            tree[i] = m; i >>= 1
        rows.append(r)
        if r == nrows: nrows += 1
    return rows, tree[size:size + nrows]


def bfd_fast(xs, T):
    """BFD in O(log T) per chat: a max-tree over the VALUES 1..T (leaf v holds v while some row has exactly v empty slots,
    else 0; leaf T always stands for 'open a new row', as in TRL) finds the smallest empty-slot count >= x. A heap per
    value gives the EARLIEST row with exactly that many empty slots, so the layout equals bfd_naive's."""
    size = 1
    while size < T: size <<= 1
    tree = [0] * (2 * size)
    heaps = [None] * (T + 1)
    i = size + T - 1
    while i: tree[i] = T; i >>= 1
    rows = []; space = []
    for x in xs:
        i = 1
        while i < size:
            i <<= 1
            if tree[i] < x: i += 1
        s = tree[i]
        if s == T:
            r = len(space); space.append(0)
        else:
            h = heaps[s]; r = heappop(h)
            if not h:                              # no row has exactly s empty slots any more: clear leaf s
                tree[i] = 0; i >>= 1
                while i:
                    a = tree[2 * i]; b = tree[2 * i + 1]
                    m = a if a >= b else b
                    if tree[i] == m: break
                    tree[i] = m; i >>= 1
        s -= x; space[r] = s
        if s:
            h = heaps[s]
            if h is None: h = heaps[s] = []
            if not h:                              # first row with exactly s empty slots: set leaf s
                i = size + s - 1; tree[i] = s; i >>= 1
                while i:
                    a = tree[2 * i]; b = tree[2 * i + 1]
                    m = a if a >= b else b
                    if tree[i] == m: break
                    tree[i] = m; i >>= 1
            heappush(h, r)
        rows.append(r)
    return rows, space


def bfd_trl(xs, T, SegmentTree):
    """TRL's binning loop (trl/data_utils.py:781-806) on lengths only, with TRL's own _SegmentTree class. Same best-fit
    rule; ties among rows with the same empty-slot count go first-in-first-out (a deque per value), not earliest row.
    (TRL iterates numpy int32 scalars; here the input is the same Python list of ints every other version gets.)"""
    segment_tree = SegmentTree(T)
    segment_tree.add(T)
    space_to_bin = defaultdict(deque)
    bins = []
    for idx, length in enumerate(xs):
        space = segment_tree.search(length)
        if space < T:
            bin = space_to_bin[space].popleft()
        else:
            bin = {"ids": [], "length": 0}
            bins.append(bin)
        bin["ids"].append(idx)
        bin["length"] += length
        if space < T and not space_to_bin[space]:
            segment_tree.remove(space)
        space = space - length
        space_to_bin[space].append(bin)
        if space > 0:
            segment_tree.add(space)
    rows = [0] * len(xs)
    for r, b in enumerate(bins):
        for k in b["ids"]: rows[k] = r
    return rows, [T - b["length"] for b in bins]


# ============================================================================ helpers
def sort_desc(lengths):
    """Placement order: longest first; equal lengths keep their input order (stable). Returns (order, lengths list)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")
    return order, lengths[order].tolist()


def lower_bound(xs, T):
    return -(-int(sum(xs)) // T)


def sign_test_p(a, b):
    """Exact two-sided sign test: chance of a split at least this uneven if each rule were equally likely to win."""
    n = a + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def row_sets(rows):
    g = defaultdict(list)
    for k, r in enumerate(rows): g[r].append(k)
    return {frozenset(v) for v in g.values()}


def decision_stats(xs, T, best):
    """Replays one rule (best=False: FFD, best=True: BFD) and sorts every placement into: no open row fits (new row),
    exactly one open row fits, several fit and the first fitting row IS the tightest, several fit and it is NOT. Only
    the last kind is a moment where FFD and BFD would choose different rows (from that state)."""
    space = []; c = Counter()
    for x in xs:
        first = tight = -1; tight_s = T + 1; k = 0
        for r, s in enumerate(space):
            if s >= x:
                k += 1
                if first < 0: first = r
                if s < tight_s: tight = r; tight_s = s
        if k == 0:
            c["new"] += 1; space.append(T - x); continue
        c["one" if k == 1 else ("same" if first == tight else "differ")] += 1
        space[tight if best else first] -= x
    return c


def examined_rows(rows, rule):
    """Exact number of rows the NAIVE version looks at, from a layout: FFD looks at rows 0..r (r + 1 rows) when the
    chat lands in open row r, and at every open row when it opens a new one; BFD always looks at every open row."""
    a = np.asarray(rows, dtype=np.int64)
    open_before = np.maximum.accumulate(np.concatenate(([-1], a[:-1]))) + 1
    if rule == "ffd": return int(np.where(a < open_before, a + 1, open_before).sum())
    return int(open_before.sum())


def timed(fn, xs, T, repeats):
    best, out = None, None
    for _ in range(repeats):
        gc.collect(); gc.disable()
        t = time.perf_counter(); out = fn(xs, T); dt = time.perf_counter() - t
        gc.enable()
        best = dt if best is None or dt < best else best
    return best, out


def compare_counts(label, f_rows, b_rows, lbs, indent="  "):
    f = np.asarray(f_rows); b = np.asarray(b_rows); lb = np.asarray(lbs); d = f - b
    win, tie, lose = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
    print(f"{indent}{label}: {len(d):,} instances, mean BFD rows {b.mean():,.1f}, mean lower bound {lb.mean():,.1f}")
    print(f"{indent}   FFD FEWER rows {win:,} | EQUAL {tie:,} | FFD MORE rows {lose:,}   (sign test p = {sign_test_p(win, lose):.3g})")
    print(f"{indent}   mean (FFD - BFD) = {d.mean():+.4f} rows ({100 * (d / b).mean():+.5f}% of the rows); "
          f"differences seen -> instances: {dict(sorted(Counter(d.tolist()).items()))}")
    print(f"{indent}   mean rows above the lower bound ceil(sum/T): FFD {(f - lb).mean():.4f}, BFD {(b - lb).mean():.4f}; "
          f"largest: FFD {(f - lb).max()}, BFD {(b - lb).max()}; instances AT the lower bound: FFD {(f == lb).sum():,}, BFD {(b == lb).sum():,}")
    return win, tie, lose, float(d.mean())


def pin_to_fastest_cpu():
    try:
        freq = {}
        for c in sorted(os.sched_getaffinity(0)):
            with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq") as fh: freq[c] = int(fh.read())
        top = max(freq.values()); cpu = min(c for c, v in freq.items() if v == top)
        os.sched_setaffinity(0, {cpu})
        return f"pinned to CPU {cpu} (max {top / 1e6:.1f} GHz, the highest of the {len(freq)} CPUs)"
    except Exception as e:  # not Linux or no cpufreq: run unpinned
        return f"not pinned ({type(e).__name__})"


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or "unknown"


def load_trl(trl_dir):
    """TRL's _SegmentTree, _check_if_columns_can_be_packed and _pack_bfd, taken verbatim from the local clone (the module
    itself imports transformers, which is not installed here). Returns (namespace, info) or (None, reason)."""
    path = os.path.join(trl_dir, "trl", "data_utils.py")
    try:
        import pyarrow, pyarrow.compute, pyarrow.types
        src = open(path).read()
        found = {n.name: n for n in ast.parse(src).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        names = ("_SegmentTree", "_check_if_columns_can_be_packed", "_pack_bfd")
        ns = {"np": np, "pa": pyarrow, "pc": pyarrow.compute, "pyarrow": pyarrow, "defaultdict": defaultdict,
              "deque": deque, "Literal": Literal}
        for name in names:
            exec(compile(ast.get_source_segment(src, found[name]), path, "exec"), ns)
        try:
            commit = subprocess.run(["git", "-C", trl_dir, "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, timeout=10).stdout.strip() or "unknown"
        except Exception: commit = "unknown"
        lines = ", ".join(f"{n} {found[n].lineno}-{found[n].end_lineno}" for n in names)
        return ns, f"{path} at commit {commit} (lines: {lines}); pyarrow {pyarrow.__version__}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================================================ setup
print("=" * 120)
print("FFD vs BFD, full study (ffd_vs_bfd_full_study.py), run", time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"python {platform.python_version()}, numpy {np.__version__}, CPU {cpu_model()}; {pin_to_fastest_cpu()}; "
      f"load average {os.getloadavg()[0]:.2f}")
meta = json.load(open(DATA + "/meta.json"))
T, N = int(meta["T"]), int(meta["N"])
has = np.fromfile(DATA + "/labels.bin", dtype=np.uint16).reshape(N, T) != 65535
assert has.any(axis=1).all(), "a chat with no trained label"
L = (T - np.argmax(has[:, ::-1], axis=1)).astype(np.int64)             # index of the last label that is not 65535, + 1
assert np.array_equal(L, [np.nonzero(r)[0][-1] + 1 for r in has])      # the older scripts' row-by-row rule gives the same
assert L.min() >= 1 and L.max() <= T
del has
REAL, LB = int(L.sum()), lower_bound(L.tolist(), T)
print(f"data {DATA}: {N:,} chats, rows of T = {T}; slots per chat: shortest {L.min()}, longest {L.max()}, mean {L.mean():.1f}, "
      f"median {np.median(L):.1f}; chats longer than T/2: {(L > T // 2).sum()}; distinct lengths {len(np.unique(L)):,}")
print(f"real tokens (slots filled) {REAL:,}; lower bound on rows ceil({REAL:,} / {T}) = {LB:,}")
TRL_NS, TRL_INFO = load_trl(TRL_DIR)
print(f"TRL code: {TRL_INFO}" if TRL_NS else f"TRL code NOT loaded ({TRL_INFO}); TRL checks below are skipped (unverified)")
TRL_SEG = TRL_NS["_SegmentTree"] if TRL_NS else None

# ============================================================================ E5a correctness first
print("\n" + "=" * 120)
print("E5a CHECK FIRST: do the fast versions give exactly what the naive versions give? (layouts compared chat by chat)")
order0, xs0 = sort_desc(L)
checks = [("real stage-0, 5,000 chats", xs0)]
g = np.random.default_rng(98)
checks.append(("resampled real lengths, 20,000 chats (seed 98)", sort_desc(g.choice(L, 20000))[1]))
SYN_CHECK = [("uniform 1..T", lambda g, n: g.integers(1, T + 1, n)),
             ("uniform T/4..T/2", lambda g, n: g.integers(T // 4, T // 2 + 1, n)),
             ("hard mix just over T/2, T/3, T/4", lambda g, n: np.array([2048, 1365, 1024])[g.integers(0, 3, n)] + g.integers(1, 65, n)),
             ("50% 1..256 + 50% 3072..4096", lambda g, n: np.where(g.random(n) < 0.5, g.integers(1, 257, n), g.integers(3072, T + 1, n)))]
for j, (name, fn) in enumerate(SYN_CHECK):
    g = np.random.default_rng(90 + j)
    checks.append((f"synthetic {name}, 3,000 chats (seed {90 + j})", sort_desc(fn(g, 3000))[1]))
all_ok = True
for name, xs in checks:
    fa, fsp = ffd_naive(xs, T); fb, fsp2 = ffd_fast(xs, T)
    ba, bsp = bfd_naive(xs, T); bb, bsp2 = bfd_fast(xs, T)
    ok_f = fa == fb and sorted(fsp) == sorted(fsp2); ok_b = ba == bb and sorted(bsp) == sorted(bsp2)
    line = (f"  {name}: naive FFD {len(fsp):,} rows, fast FFD {len(fsp2):,} rows, same layout: {'YES' if ok_f else 'NO'} | "
            f"naive BFD {len(bsp):,}, fast BFD {len(bsp2):,}, same layout: {'YES' if ok_b else 'NO'}")
    if TRL_SEG:
        ta, tsp = bfd_trl(xs, T, TRL_SEG)
        same_n = len(tsp) == len(bsp) and sorted(tsp) == sorted(bsp)
        line += (f" | TRL's loop {len(tsp):,} rows, same rows and same empty-slot list: {'YES' if same_n else 'NO'}, "
                 f"chats in a different row than naive BFD: {sum(p != q for p, q in zip(ta, ba)):,}")
        all_ok &= same_n
    all_ok &= ok_f and ok_b
    print(line)
print(f"  => every check passed: {'YES' if all_ok else 'NO'}")
assert all_ok, "a fast version disagrees with its naive version; stop"

# ============================================================================ E1
print("\n" + "=" * 120)
print("E1 REAL STAGE-0 DATA, 5,000 chats packed all at once")
shortest = int(L.min())
layouts = {"FFD": ffd_naive(xs0, T)[0], "BFD": bfd_naive(xs0, T)[0]}
if TRL_SEG: layouts["BFD, TRL tie rule (first-in-first-out)"] = bfd_trl(xs0, T, TRL_SEG)[0]
x0 = np.asarray(xs0, dtype=np.int64)
E1 = {}
for name, rows in layouts.items():
    a = np.asarray(rows); R = int(a.max()) + 1
    used = np.zeros(R, np.int64); np.add.at(used, a, x0)
    attn = np.zeros(R, np.int64); np.add.at(attn, a, x0 * x0)
    empty = T - used; cnt = np.bincount(a, minlength=R)
    E1[name] = dict(R=R, empty=empty, cnt=cnt, attn=attn)
    small = empty[(empty > 0) & (empty < shortest)]; big = empty[empty >= shortest]
    c = Counter(cnt.tolist())
    print(f"  {name}")
    print(f"    rows {R:,} (lower bound {LB:,}, +{R - LB}); real tokens {100 * REAL / (R * T):.3f}% of {R * T:,} slots")
    print(f"    empty slots per row -> rows:  0: {(empty == 0).sum()}  |  1-16: {((empty >= 1) & (empty <= 16)).sum()}  |  "
          f"17-64: {((empty >= 17) & (empty <= 64)).sum()}  |  65-256: {((empty >= 65) & (empty <= 256)).sum()}  |  >256: {(empty > 256).sum()}")
    print(f"    empty slots: total {int(empty.sum()):,} ({empty.sum() / T:.2f} rows' worth), mean {empty.mean():.2f}, "
          f"median {np.median(empty):.0f}, largest {empty.max()}; the 6 largest gaps {sorted(empty.tolist())[-6:]}")
    print(f"    gaps smaller than the shortest chat ({shortest}): {len(small)} rows, {int(small.sum()):,} slots; "
          f"gaps of {shortest} or more: {len(big)} rows, {int(big.sum()):,} slots")
    print(f"    chats per row -> rows: " + ", ".join(f"{k}: {c[k]}" for k in sorted(c)) + f"   (maximum {cnt.max()}, mean {cnt.mean():.3f})")
    print(f"    rows holding a single chat: {(cnt == 1).sum()} -> " + ", ".join(f"chat of {int(used[r])} slots ({int(empty[r])} empty)" for r in np.nonzero(cnt == 1)[0]))
    print(f"    attention work per row (sum of L*L over its chats): mean {attn.mean():,.0f}, largest {attn.max():,}, "
          f"smallest {attn.min():,}, coefficient of variation (std/mean) {attn.std() / attn.mean():.4f}")


def compare_layouts(na, nb):
    a, b = np.asarray(layouts[na]), np.asarray(layouts[nb])
    diff = np.nonzero(a != b)[0]
    sa, sb = row_sets(layouts[na]), row_sets(layouts[nb]); common = sa & sb
    moved = N - sum(len(s) for s in common)
    print(f"  {na} layout vs {nb} layout (same chats, same sorted order):")
    print(f"    chats whose row number differs (rows numbered in the order they are opened): {len(diff):,} of {N:,} ({100 * len(diff) / N:.1f}%)")
    print(f"    chats whose set of row-mates differs (does not depend on row numbers): {moved:,} of {N:,} ({100 * moved / N:.1f}%)")
    print(f"    rows holding exactly the same chats in both layouts: {len(common):,} (of {len(sa):,} and {len(sb):,})")
    if len(diff):
        k = int(diff[0])
        print(f"    first chat placed differently: sorted position {k} (length {xs0[k]}; {na} row {a[k]}, {nb} row {b[k]}); "
              f"every chat before it in the sorted order is placed the same")
    print(f"    same empty-slot list (sorted): {'YES' if np.array_equal(np.sort(E1[na]['empty']), np.sort(E1[nb]['empty'])) else 'NO'}; "
          f"same chats-per-row histogram: {'YES' if np.array_equal(np.sort(E1[na]['cnt']), np.sort(E1[nb]['cnt'])) else 'NO'}")


print("  every placement of the 5,000 chats, in each rule's own run (a naive replay that also looks for the other rule's row):")
for name, best in (("FFD", False), ("BFD", True)):
    c = decision_stats(xs0, T, best)
    assert c["new"] == E1[name]["R"] and sum(c.values()) == N
    print(f"    {name}: no open row fits, new row {c['new']:,} | exactly one open row fits {c['one']:,} | several fit and the first "
          f"is also the tightest {c['same']:,} | several fit and the first is NOT the tightest {c['differ']:,}")
print("    (only the last kind is a placement where FFD and BFD pick different rows from the same state)")
compare_layouts("FFD", "BFD")
if TRL_SEG:
    compare_layouts("BFD", "BFD, TRL tie rule (first-in-first-out)")
    # TRL's real _pack_bfd on a pyarrow table whose tokens are the chat ids, in dataset order (one group of 5,000)
    import pyarrow as pa
    offs = np.concatenate(([0], np.cumsum(L))).astype(np.int32)
    table = pa.table({"input_ids": pa.ListArray.from_arrays(pa.array(offs), pa.array(np.repeat(np.arange(N, dtype=np.int32), L)))})
    t = time.perf_counter(); out = TRL_NS["_pack_bfd"](table, seq_length=T, on_seq_length_overflow="truncate"); dt_trl = time.perf_counter() - t
    col = out.column("input_ids").combine_chunks(); o = col.offsets.to_numpy(); v = col.flatten().to_numpy()
    sl = out.column("seq_lengths").to_pylist()
    trl_rows = []
    for r in range(len(col)):
        seg = v[o[r] - o[0]:o[r + 1] - o[0]]
        starts = np.r_[0, np.nonzero(np.diff(seg))[0] + 1]
        assert np.diff(np.r_[starts, len(seg)]).tolist() == sl[r]
        trl_rows.append(seg[starts].tolist())
    mine = defaultdict(list)
    for k, r in enumerate(layouts["BFD, TRL tie rule (first-in-first-out)"]): mine[r].append(int(order0[k]))
    same = trl_rows == [mine[r] for r in range(len(mine))]
    print(f"  TRL's own _pack_bfd (the real function, run on all 5,000 chats as one group): {len(trl_rows):,} rows, "
          f"identical layout to the TRL-tie-rule BFD above (same chats, same order in every row): {'YES' if same else 'NO'}; took {dt_trl:.3f} s")

# ============================================================================ E2
print("\n" + "=" * 120)
print("E2 RESAMPLED REAL LENGTHS: each instance draws n chat lengths WITH replacement from the 5,000 real lengths")
E2 = {}
for n, count, seed, naive_check in ((100, 5000, 100, 1000), (1000, 5000, 1000, 300), (5000, 1000, 5000, 20), (20000, 300, 20000, 0)):
    g = np.random.default_rng(seed); fr, br, lbs = [], [], []
    for k in range(count):
        xs = sort_desc(g.choice(L, n))[1]
        f_rows, f_sp = ffd_fast(xs, T); b_rows, b_sp = bfd_fast(xs, T)
        if k < naive_check:
            assert ffd_naive(xs, T)[0] == f_rows and bfd_naive(xs, T)[0] == b_rows
        fr.append(len(f_sp)); br.append(len(b_sp)); lbs.append(lower_bound(xs, T))
    note = f"naive versions re-run on the first {naive_check:,} instances: same layouts" if naive_check else "fast versions only"
    E2[n] = compare_counts(f"n = {n:,} chats (seed {seed}; {note})", fr, br, lbs)

# ============================================================================ E3
print("\n" + "=" * 120)
print("E3 SYNTHETIC LENGTHS (rows of T = 4096; all lengths within 1..T; integers drawn uniformly inside each range)")
DISTS = [("uniform 1..4096 (1..T)", lambda g, n: g.integers(1, T + 1, n)),
         ("uniform 1..2048 (1..T/2)", lambda g, n: g.integers(1, T // 2 + 1, n)),
         ("uniform 1..1024 (1..T/4)", lambda g, n: g.integers(1, T // 4 + 1, n)),
         ("uniform 1024..2048 (T/4..T/2)", lambda g, n: g.integers(T // 4, T // 2 + 1, n)),
         ("50% very short 1..256 + 50% very long 3072..4096", lambda g, n: np.where(g.random(n) < 0.5, g.integers(1, 257, n), g.integers(3072, T + 1, n))),
         ("80% short 1..512 + 20% long 2049..4096", lambda g, n: np.where(g.random(n) < 0.8, g.integers(1, 513, n), g.integers(T // 2 + 1, T + 1, n))),
         ("hard mix, thirds just over T/2, T/3, T/4: 2049..2112, 1366..1429, 1025..1088",
          lambda g, n: np.array([2048, 1365, 1024])[g.integers(0, 3, n)] + g.integers(1, 65, n)),
         ("hard mix, wider: thirds in 2049..2304, 1366..1621, 1025..1280",
          lambda g, n: np.array([2048, 1365, 1024])[g.integers(0, 3, n)] + g.integers(1, 257, n))]
E3 = {}
for j, (name, fn) in enumerate(DISTS):
    print(f"  {name}")
    for n, count, naive_check in ((1000, 2000, 100), (10000, 200, 0)):
        seed = 300 + 10 * j + (1 if n == 1000 else 2)
        g = np.random.default_rng(seed); fr, br, lbs = [], [], []
        for k in range(count):
            xs = sort_desc(fn(g, n))[1]
            assert 1 <= xs[-1] and xs[0] <= T
            f_rows, f_sp = ffd_fast(xs, T); b_rows, b_sp = bfd_fast(xs, T)
            if k < naive_check:
                assert ffd_naive(xs, T)[0] == f_rows and bfd_naive(xs, T)[0] == b_rows
            fr.append(len(f_sp)); br.append(len(b_sp)); lbs.append(lower_bound(xs, T))
        E3[(name, n)] = compare_counts(f"n = {n:,} (seed {seed})", fr, br, lbs, indent="    ")

print("\nE3b EVERY SMALL INSTANCE: rows of T = 20 (the size of examples 4 and 5 in the doc); every multiset of n chat lengths,")
print("    each length 1..20, counted once; naive versions")
Ts, tot = 20, Counter()
NMAX = 8
for n in range(2, NMAX + 1):
    c = Counter(); ex = {}
    for combo in itertools.combinations_with_replacement(range(Ts, 0, -1), n):   # already longest first
        f = len(ffd_naive(combo, Ts)[1]); b = len(bfd_naive(combo, Ts)[1])
        key = "FFD fewer" if f < b else ("BFD fewer" if b < f else "equal")
        c[key] += 1
        if key != "equal" and key not in ex: ex[key] = (combo, f, b)
    tot.update(c); m = sum(c.values())
    print(f"  n = {n}: {m:,} instances | FFD fewer rows {c['FFD fewer']:,} ({100 * c['FFD fewer'] / m:.3f}%) | equal {c['equal']:,} | "
          f"BFD fewer rows {c['BFD fewer']:,} ({100 * c['BFD fewer'] / m:.3f}%)"
          + "".join(f" | first '{k}' case: {list(e[0])} -> FFD {e[1]}, BFD {e[2]} rows" for k, e in sorted(ex.items())))
m = sum(tot.values())
print(f"  all n = 2..{NMAX}: {m:,} instances | FFD fewer {tot['FFD fewer']:,} ({100 * tot['FFD fewer'] / m:.3f}%) | equal {tot['equal']:,} "
      f"({100 * tot['equal'] / m:.3f}%) | BFD fewer {tot['BFD fewer']:,} ({100 * tot['BFD fewer'] / m:.3f}%)")

# ============================================================================ E4
print("\n" + "=" * 120)
print("E4 TRL-STYLE GROUPS: shuffle the 5,000 real chats, cut into groups of G in that order, pack every group alone")
perm = np.random.default_rng(1234).permutation(N)
assert np.array_equal(np.random.default_rng(1234).permutation(L), L[perm])   # same order as group_packing.py


def grouped(perm, G):
    fr = br = fp = bp = 0
    for i in range(0, N, G):
        xs = sort_desc(L[perm[i:i + G]])[1]
        _, f_sp = ffd_fast(xs, T); _, b_sp = bfd_fast(xs, T)
        fr += len(f_sp); br += len(b_sp); fp += sum(s > 256 for s in f_sp); bp += sum(s > 256 for s in b_sp)
    return fr, br, fp, bp


for G in (10, 100, 1000, 5000):
    fr, br, fp, bp = grouped(perm, G)
    print(f"  seed 1234, G = {G:5,}: {-(-N // G):4d} groups | FFD {fr:,} rows ({100 * REAL / (fr * T):.2f}% real, {fp} rows with >256 empty) | "
          f"BFD {br:,} rows ({100 * REAL / (br * T):.2f}% real, {bp} rows with >256 empty) | FFD - BFD = {fr - br:+d}")
for G in (1000, 100):
    fr_all, br_all = [], []
    for s in range(1000):
        fr, br, _, _ = grouped(np.random.default_rng(s).permutation(N), G)
        fr_all.append(fr); br_all.append(br)
    compare_counts(f"G = {G:,}, 1,000 different shuffles (seeds 0..999), total rows per shuffle", fr_all, br_all, [LB] * 1000)

# ============================================================================ E5b timing
print("\n" + "=" * 120)
print("E5b SPEED (one CPU core, pure Python, input already sorted; best of k runs, garbage collector off while timing)")
print("    naive FFD scans rows from the first and stops at the first fit; naive BFD scans every open row;")
print("    fast FFD = max-tree over rows (depth log2 n); fast BFD = max-tree over empty-slot values 1..4096 (depth 12) + a heap")
print("    per value; TRL's loop = TRL's own _SegmentTree class and binning loop (depth 12, deque per value). Every version gets")
print("    the same Python list of lengths and returns the row of every chat; the sort (shared by both rules) is timed apart.")
LIMIT = 180.0
rate = {}
g = np.random.default_rng(55)
plan = [("real stage-0", L, 5, 5), ("resampled (seed 55)", g.choice(L, 50000), 5, 3),
        ("resampled (seed 55)", g.choice(L, 200000), 5, 1), ("resampled (seed 55)", g.choice(L, 1000000), 3, 1)]
E5 = []
for label, lengths, reps_fast, reps_naive in plan:
    n = len(lengths)
    gc.collect(); t = time.perf_counter(); order, xs = sort_desc(lengths); t_sort = time.perf_counter() - t
    res = {}
    t_ff, (ff_rows, ff_sp) = timed(ffd_fast, xs, T, reps_fast)
    t_bf, (bf_rows, bf_sp) = timed(bfd_fast, xs, T, reps_fast)
    res["fast FFD"], res["fast BFD"] = t_ff, t_bf
    if TRL_SEG:
        t_tr, (tr_rows, tr_sp) = timed(lambda a, b: bfd_trl(a, b, TRL_SEG), xs, T, reps_fast)
        res["TRL's BFD loop"] = t_tr
        assert len(tr_sp) == len(bf_sp)
    ex = {"naive FFD": examined_rows(ff_rows, "ffd"), "naive BFD": examined_rows(bf_rows, "bfd")}
    print(f"  {label}, n = {n:,} chats: FFD rows {len(ff_sp):,}, BFD rows {len(bf_sp):,} (lower bound {lower_bound(xs, T):,}); "
          f"sort {1e3 * t_sort:.1f} ms")
    print(f"    rows the naive versions look at (exact count): naive FFD {ex['naive FFD']:,}, naive BFD {ex['naive BFD']:,} "
          f"(BFD / FFD = {ex['naive BFD'] / ex['naive FFD']:.3f}); per chat: FFD {ex['naive FFD'] / n:,.1f}, BFD {ex['naive BFD'] / n:,.1f}")
    skipped = {}
    for name, fn, ref in (("naive FFD", ffd_naive, ff_rows), ("naive BFD", bfd_naive, bf_rows)):
        if name in rate and rate[name][0] * ex[name] > LIMIT:
            skipped[name] = (f"NOT RUN: it would look at {ex[name]:,} rows; at the {rate[name][0] * 1e9:.1f} ns per row measured at "
                             f"n = {rate[name][1]:,} that is about {rate[name][0] * ex[name] / 60:.0f} minutes (projection, not a measurement)")
            res[name] = None; continue
        t_n, (rows, sp) = timed(fn, xs, T, reps_naive)
        assert rows == ref, f"{name} layout differs from the fast version"
        res[name] = t_n; rate[name] = (t_n / ex[name], n)
    for name in ("naive FFD", "naive BFD", "fast FFD", "fast BFD", "TRL's BFD loop"):
        if name not in res: continue
        tt = res[name]
        reps = reps_naive if name.startswith("naive") else reps_fast
        print(f"    {name:15s}: " + (skipped[name] if tt is None else f"{tt:9.4f} s  ({1e6 * tt / n:7.2f} us per chat; best of {reps})"))
    ratios = [f"{kind} BFD / {kind} FFD time = {res[kind + ' BFD'] / res[kind + ' FFD']:.2f}" for kind in ("naive", "fast")
              if res.get(kind + " BFD") and res.get(kind + " FFD")]
    if ratios: print("    " + "; ".join(ratios))
    E5.append((label, n, res, ex))
print("  summary (seconds; '-' = not run, see the projection above):")
print(f"    {'n':>10s} {'naive FFD':>10s} {'naive BFD':>10s} {'fast FFD':>10s} {'fast BFD':>10s} {'TRL loop':>10s}")
for label, n, res, ex in E5:
    cells = ["-" if res.get(k) is None else f"{res[k]:.4f}" for k in ("naive FFD", "naive BFD", "fast FFD", "fast BFD", "TRL's BFD loop")]
    print(f"    {n:>10,} " + " ".join(f"{c:>10s}" for c in cells))
print(f"\ntotal run time {time.perf_counter() - START:.0f} s; load average at the end {os.getloadavg()[0]:.2f}")
