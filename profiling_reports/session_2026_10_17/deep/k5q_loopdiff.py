import collections, re, sys
sys.path.insert(0, "session_2026_10_17/deep")
exec(open("session_2026_10_17/deep/ncu_k5q_vs_fa3.py").read().split("\nM = [")[0].replace("NHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 25", ""))   # reuse the loaders + paths only
def full_op(s):
    s = re.sub(r"^@!?U?P[T0-9]+\s+", "", s); return s.split()[0] if s else ""
def loop(path, tag):
    ins = src_kernel(path, tag)
    g = collections.defaultdict(float)
    for src, ex, s, st in ins: g[ex] += s
    lx = max(g, key=g.get)                       # the count holding the most samples = the steady loop
    body = [(i, src, s, st) for i, (src, ex, s, st) in enumerate(ins) if ex == lx]
    return lx, body, sum(s for _, _, s, _ in ins)
L = {}
for shape in ("dense", "packed"):
    for n, p, t in SRC[shape]:
        if n == "K1": continue
        L[(n, shape)] = loop(p, t)
def table(a, b):
    (la, ba, ta), (lb, bb, tb) = L[a], L[b]
    ca = collections.Counter(full_op(s) for _, s, _, _ in ba); cb = collections.Counter(full_op(s) for _, s, _, _ in bb)
    sa = collections.Counter(); sb = collections.Counter()
    for _, s, x, _ in ba: sa[full_op(s)] += x
    for _, s, x, _ in bb: sb[full_op(s)] += x
    print(f"\n=== loop body {a[0]} {a[1]} (executed {la}x, {len(ba)} instr, {100*sum(sa.values())/ta:.1f}% of samples) vs "
          f"{b[0]} {b[1]} (executed {lb}x, {len(bb)} instr, {100*sum(sb.values())/tb:.1f}% of samples)")
    print(f"{'opcode':34s} {a[0]:>5s} {b[0]:>5s} {'diff':>5s}   samples% {a[0]:>6s} {b[0]:>6s}")
    for k in sorted(set(ca) | set(cb), key=lambda k: (-(ca[k] - cb[k]), k)):
        if ca[k] != cb[k] or abs(sa[k] / ta - sb[k] / tb) > 0.004:
            print(f"{k:34s} {ca[k]:5d} {cb[k]:5d} {ca[k]-cb[k]:+5d}   {'':8s} {100*sa[k]/ta:6.2f} {100*sb[k]/tb:6.2f}")
table(("K5q", "dense"), ("FA3", "dense"))
table(("K5q", "packed"), ("FA3", "packed"))
table(("K5q", "dense"), ("K5q", "packed"))
