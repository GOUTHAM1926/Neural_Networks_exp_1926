# loop phases (instruction index ranges of the steady loop, from the skeletons), per 1M loop executions, cycle-equivalents (FA3 = 100)
import collections, sys
exec(open("session_2026_10_17/deep/ncu_k5q_vs_fa3.py").read().split("\nM = [")[0].replace("NHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 25", ""))
PH = {"FA3": [("QK issue (barrier, descriptors, 8 QK HGMMA)", 1558, 1598), ("PV issue (8 PV HGMMA, release)", 1599, 1639),
              ("softmax (QK wait .. loop end)", 1640, 2026)],
      "K5q": [("QK issue (barrier, descriptors, 8 QK HGMMA)", 1072, 1143), ("PV issue (8 PV HGMMA, release)", 1144, 1171),
              ("softmax (QK wait .. loop end)", 1172, 1556)]}
R = ["wait", "dispatch", "selected", "not_selected", "barrier", "short_sb", "mio", "math", "other"]
cyc = {n: f(raw_row(p, t)["sm__cycles_elapsed.avg"]) for n, p, t in RAW["dense"] if n in ("FA3", "K5q")}
res = {}
for n, p, t in SRC["dense"]:
    if n not in PH: continue
    ins = src_kernel(p, t); tot = sum(s for _, _, s, _ in ins); sc = 100 * cyc[n] / cyc["FA3"] / tot
    lx = collections.Counter(ex for _, ex, _, _ in ins).most_common(1)[0][0]
    lx = max({ex for _, ex, _, _ in ins}, key=lambda e: sum(s for _, x, s, _ in ins if x == e))
    for name, a, b in PH[n]:
        seg = [(src, ex, s, st) for i, (src, ex, s, st) in enumerate(ins) if a <= i <= b]
        other_ex = collections.Counter(ex for _, ex, _, _ in seg if ex != lx)
        if other_ex: print(f'   note: {n} {name}: also contains instructions executed', dict(other_ex), 'times (counted with their samples)')
        c = collections.Counter()
        for src, ex, s, st in seg:
            for k, v in st.items(): c[k] += v
            c["other"] += s - sum(st.values())
        tot_seg = sum(s for _, _, s, _ in seg)
        res[(n, name)] = (len(seg), tot_seg * sc / lx * 1e6, {r: c[r] * sc / lx * 1e6 for r in R})
print(f"{'phase':46s} {'':4s} {'instr':>5s} {'per 1M':>7s} | " + " ".join(f"{r[:8]:>8s}" for r in R))
for name, _, _ in PH["FA3"]:
    for n in ("FA3", "K5q"):
        ni, t, d = res[(n, name)]
        print(f"{name:46s} {n:4s} {ni:5d} {t:7.2f} | " + " ".join(f"{d[r]:8.2f}" for r in R))
    a, b = res[("FA3", name)], res[("K5q", name)]
    print(f"{'':46s} {'diff':4s} {b[0]-a[0]:+5d} {b[1]-a[1]:+7.2f} | " + " ".join(f"{b[2][r]-a[2][r]:+8.2f}" for r in R))
