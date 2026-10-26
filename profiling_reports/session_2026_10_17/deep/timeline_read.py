# 2026-10-17: the per-step TIMELINE of a V62_TRACE dump (deep/make_trace.py, K1 family): for each CTA, both consumer groups'
# clocks at every loop step of their first pair: s0 step start | s1 baton acquired (+ K/V landed for group 0) | s2 QK(n) done |
# s3 part A done (just before the wait for P@V(n-1)) | s4 that wait passed | s5 step end. Prints the steady-state averages per
# group, how the two groups are offset, an estimate of when the tensor cores have no work, and a text chart of 4 steps.
# usage: python3 timeline_read.py <trace.bin> [cta for the chart, default 0]
import struct, sys, statistics as st
path = sys.argv[1]; chart_cta = int(sys.argv[2]) if len(sys.argv) > 2 else 0
b = open(path, "rb").read()
hdr = struct.unpack_from("<16q", b, 0); NS, MR, W = hdr[7], hdr[8], hdr[9]
off = 128 + NS * 2 * 4 * 8 + NS * 2 * MR * W * 8
TLS, TLW = 64, 8
assert len(b) >= off + NS * 2 * TLS * TLW * 4, "no timeline in this file (older trace build)"
tl = struct.unpack_from(f"<{NS*2*TLS*TLW}I", b, off)
def step(cta, c, n):
    q = tl[((cta * 2 + c) * TLS + n) * TLW:((cta * 2 + c) * TLS + n + 1) * TLW]
    return q if q[5] != 0 and q[6] == n else None
def rel(a, base): return (a - base) & 0xffffffff
rows = []
for cta in range(NS):
    s0 = [step(cta, 0, n) for n in range(TLS)]; s1 = [step(cta, 1, n) for n in range(TLS)]
    ns = [n for n in range(1, TLS) if s0[n] and s1[n]]
    if len(ns) < 12: continue
    lo, hi = ns[3], ns[-4]                       # steady state: drop the first 3 and the last 3 steps
    for n in range(lo, hi):
        if not (s0[n] and s1[n] and s0[n + 1] and s1[n + 1]): continue
        a, c1, a2 = s0[n], s1[n], s0[n + 1]
        base = a[1]
        rows.append(dict(cta=cta, n=n,
            g0=[rel(a[i + 1], a[i]) for i in range(5)], g1=[rel(c1[i + 1], c1[i]) for i in range(5)],
            per0=rel(a2[1], a[1]), per1=rel(s1[n + 1][1], c1[1]),
            off1=rel(c1[1], base), qk0=rel(a[2], base), qk1=rel(c1[2], base), nxt0=rel(a2[1], base)))
if not rows: sys.exit("no steady-state steps found")
print(f"{len(rows)} steady-state steps (both groups), {len(set(r['cta'] for r in rows))} CTAs")
names = ["wait K/V + baton (s0->s1)", "issue + wait QK (s1->s2)", "part A (s2->s3)", "wait P.V (s3->s4)", "part B + rescale + pack (s4->s5)"]
for g in ("g0", "g1"):
    print(f"  group {g[1]}: " + " | ".join(f"{nm.split(' (')[0]} {st.mean(r[g][i] for r in rows):.0f}" for i, nm in enumerate(names))
          + f" | period {st.mean(r['per' + g[1]] for r in rows):.0f}")
print(f"  offsets inside one step (cycles after group 0 got the baton): group 1 gets it at {st.mean(r['off1'] for r in rows):.0f}, "
      f"group 0's QK done at {st.mean(r['qk0'] for r in rows):.0f}, group 1's QK done at {st.mean(r['qk1'] for r in rows):.0f}, "
      f"group 0's next baton at {st.mean(r['nxt0'] for r in rows):.0f}")
# tensor work estimate: each group's GEMMs of a step = QK (512) + P.V (512) in issue order; busy from the later of (issue, end of
# the previously queued work); gaps = no GEMM work queued. Rough: assumes 512 cycles per 8-HGMMA group when the pipe is free.
idle = []
for cta in sorted(set(r['cta'] for r in rows)):
    ev = []
    for n in range(1, TLS):
        for c in (0, 1):
            q = step(cta, c, n)
            if q: ev.append(q[1])               # baton acquired = issue start of this group's QK(n) + P.V(n-1)
    ev.sort(); t = None; gaps = 0; span = 0
    for e in ev:
        if t is None: t = e + 1024; first = e; continue
        if (e - t) & 0x80000000 == 0 and e > t: gaps += e - t; t = e + 1024
        else: t = t + 1024
    span = t - first
    if span > 0: idle.append(gaps / span)
print(f"  tensor cores with no GEMM queued (estimate, 1024 cycles per group step): {st.mean(idle) * 100:.1f}% of the first pair's loop")
# chart: one CTA, 4 steady-state steps, 1 character = 50 cycles
r0 = [r for r in rows if r['cta'] == chart_cta] or rows
cta = r0[0]['cta']; n0 = r0[0]['n']
base = step(cta, 0, n0)[0]
print(f"\n  chart, CTA {cta}, steps {n0}..{n0 + 3} (1 char = 50 cycles; k = K/V + baton wait, q = issue + wait QK, A = part A, w = wait P.V,"
      f" B = part B + rescale + pack, . = between steps)")
for c in (0, 1):
    line = []
    for n in range(n0, n0 + 4):
        q = step(cta, c, n)
        if not q: break
        marks = [(q[0], 'k'), (q[1], 'q'), (q[2], 'A'), (q[3], 'w'), (q[4], 'B'), (q[5], '.')]
        for (t_a, ch), (t_b, _) in zip(marks, marks[1:] + [(step(cta, c, n + 1)[0] if step(cta, c, n + 1) else q[5], '.')]):
            a = rel(t_a, base) // 50; z = rel(t_b, base) // 50
            while len(line) < z: line.append(' ')
            for i in range(a, z): line[i] = ch
    print(f"  group {c}: " + "".join(line)[:200])
