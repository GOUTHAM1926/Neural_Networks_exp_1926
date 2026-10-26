# 2026-10-17: read a V62_TRACE dump (deep/make_trace.py builds) and print where the forward's time goes, at the clock the run
# had (free clocks). Records: per (CTA, consumer group, query pair). Phases (cycles, clock64): t0 pair start (decoded) | t1 Q
# landed | t2 tile 0 K/V landed | t3 prologue done | t4 loop done | t5 last P.V + inactive tail + next-unit fetch done | t6
# epilogue done. Per step of the loop (32-bit clock, summed per pair): kv = wait K/V (group 0) + baton | qk = issue QK(n) + P.V(n-1)
# + wait QK | sm = softmax (+ pack, K1) | pv = wait P.V(n-1) | rs = rescale (+ pack, F3c).
# usage: python3 trace_read.py <trace.bin> [more .bin ...]
import struct, sys, statistics as st
from collections import defaultdict
def load(path):
    b = open(path, "rb").read()
    hdr = struct.unpack_from("<16q", b, 0)
    magic, mode, B, Hq, Hkv, S, n_units, NS, MR, W = hdr[:10]
    assert magic == 0x76363254, "not a V62 trace"
    off = 128
    cta = struct.unpack_from(f"<{NS*2*4}Q", b, off); off += NS * 2 * 4 * 8
    rec = struct.unpack_from(f"<{NS*2*MR*W}Q", b, off)
    recs, ctas = [], []
    for i in range(NS * 2):
        n = cta[i * 4 + 2]; ctas.append((i // 2, i % 2, cta[i * 4], cta[i * 4 + 1], n))
        for k in range(min(n, MR)):
            r = rec[(i * MR + k) * W:(i * MR + k + 1) * W]
            info = r[11]
            recs.append(dict(cta=i // 2, c=i % 2, k=k, gt0=r[0], t=r[1:8], gt1=r[8], fetch=r[9], uidx=r[10], sm=info & 0xff,
                             half=(info >> 8) & 1, my=(info >> 16) & 0xffff, nkv=(info >> 32) & 0xffff, pair=(info >> 48) & 0xffff,
                             L=r[12], ac=r[13:18], an=r[18]))
    return dict(mode=mode, B=B, Hq=Hq, S=S, n_units=n_units, NS=NS, recs=recs, ctas=ctas)
def summary(path):
    T = load(path); R = T["recs"]; NS = T["NS"]
    print(f"== {path.split('/')[-1]}: {'packed (A)' if T['mode'] else 'dense'}, B {T['B']} Hq {T['Hq']} S {T['S']}, {len(R)} pair records, "
          f"{sum(1 for c in T['ctas'] if c[4] > 64)} groups over 64 records (truncated)")
    # clock: cycles / ns over each group's whole span (first pair start -> last pair end), median over groups
    span = defaultdict(list)
    for r in R: span[(r["cta"], r["c"])].append(r)
    ghz = []
    for key, rs in span.items():
        rs.sort(key=lambda r: r["k"])
        cyc = rs[-1]["t"][6] - rs[0]["t"][0]; ns = rs[-1]["gt1"] - rs[0]["gt0"]
        if ns > 0: ghz.append(cyc / ns)
    clk = st.median(ghz); print(f"   SM clock during the traced launch: {clk:.3f} GHz (median over {len(ghz)} groups)")
    k0 = min(c[2] for c in T["ctas"] if c[2]); k1 = max(c[3] for c in T["ctas"])
    print(f"   kernel span (first consumer start -> last end): {(k1 - k0) / 1e3:.1f} us = {(k1 - k0) * clk / 1e6:.3f} M cycles")
    ends = sorted(c[3] for c in T["ctas"] if c[3])
    print(f"   group end times before the last: median {(k1 - st.median(ends)) / 1e3:.1f} us, 10% {(k1 - ends[len(ends) // 10]) / 1e3:.1f} us; "
          f"idle at the tail = {sum(k1 - e for e in ends) / len(ends) / (k1 - k0) * 100:.2f}% of group-time")
    act = [r for r in R if r["my"] > 0]
    ph = ["Q wait", "tile-0 K/V wait", "prologue (QK0 + softmax0)", "loop", "last P.V + tail + fetch", "epilogue"]
    tot = sum(r["t"][6] - r["t"][0] for r in act)
    print(f"   per pair (active groups, {len(act)} pairs; cycles; share of the pairs' total):")
    for i, name in enumerate(ph):
        v = [r["t"][i + 1] - r["t"][i] for r in act]
        print(f"     {name:28s} mean {st.mean(v):8.0f}  median {st.median(v):8.0f}  share {sum(v) / tot * 100:5.1f}%")
    gaps = []
    for key, rs in span.items():
        for a, b in zip(rs, rs[1:]): gaps.append(b["t"][0] - a["t"][6])
    if gaps: print(f"     {'between pairs (decode etc.)':28s} mean {st.mean(gaps):8.0f}  median {st.median(gaps):8.0f}")
    fz = [r["fetch"] for r in act]; print(f"     {'(fetch_unit inside the above)':28s} mean {st.mean(fz):8.0f}")
    an = sum(r["an"] for r in act)
    if an:
        names = ["wait K/V + baton", "issue + wait QK", "softmax (+pack K1)", "wait P.V", "rescale (+pack F3c)"]
        per = [sum(r["ac"][i] for r in act) / an for i in range(5)]
        print(f"   per loop step (mean over {an} steps; cycles per group per tile): total {sum(per):.0f}  (ideal tensor-bound period per"
              f" group = 2048, both groups' GEMMs)")
        for n_, v in zip(names, per): print(f"     {n_:22s} {v:7.0f}  ({v / sum(per) * 100:4.1f}%)")
        for c in (0, 1):
            a2 = [r for r in act if r["c"] == c]; n2 = sum(r["an"] for r in a2)
            if n2: print(f"     group {c}: " + "  ".join(f"{names[i].split()[0]} {sum(r['ac'][i] for r in a2) / n2:.0f}" for i in range(5)))
        lp = [r for r in act if r["an"] > 0]
        perstep_loop = sum(r["t"][4] - r["t"][3] for r in lp) / sum(r["an"] for r in lp)
        print(f"   loop cycles per step from t4 - t3: {perstep_loop:.0f} (vs the per-step sum above: overhead between steps = "
              f"{perstep_loop - sum(per):.0f})")
for p in sys.argv[1:]: summary(p)
