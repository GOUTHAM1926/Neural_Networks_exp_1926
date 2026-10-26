import csv, collections, re
def blocks(path):
    rows = list(csv.reader(open(path))); out = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; out.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    return out
def kern(path, want):
    b = next(b for b in blocks(path) if want in b["name"]); h = b["rows"][0]
    idx = {c: i for i, c in enumerate(h)}
    res = []
    for r in b["rows"][1:]:
        if len(r) < len(h): continue
        g = lambda c: float(r[idx[c]].replace(",", "") or 0) if c in idx else 0.0
        st = {c[len("stall_"):]: g(c) for c in h if c.startswith("stall_") and "Not Issued" not in c}
        res.append((r[idx["Source"]].strip(), int(g("Instructions Executed")), g("Warp Stall Sampling (All Samples)"), st))
    return res
K = {"FA3": kern("v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/ncu_libs_fwd.source.csv", "FlashAttnFwdSm90"),
     "K1": kern("session_2026_10_17/box_runs/k1_1017/ncu_vj2_K1_dense.source.csv", "gqa_v62")}
for name, ins in K.items():
    tot = sum(s for _, _, s, _ in ins)
    bucket = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    for src, ex, s, st in ins:
        b = bucket[ex]; b[0] += 1; b[1] += s; b[2].update(st)
    print(f"\n== {name}: {len(ins)} instructions, {int(tot)} samples; executed-count groups holding >= 1.5% of the samples:")
    print(f"   {'exec count':>10s} {'instr':>6s} {'samples':>8s} {'share':>6s}  top stall reasons")
    for ex, (n, s, st) in sorted(bucket.items(), key=lambda kv: -kv[1][1]):
        if s / tot < 0.015: continue
        top = ", ".join(f"{k} {100 * v / s:.0f}%" for k, v in st.most_common(4) if v > 0)
        print(f"   {ex:10d} {n:6d} {int(s):8d} {100 * s / tot:5.1f}%  {top}")

import sys
if len(sys.argv) > 1:
    ins = K["K1"]
    for want in (36864, 17280, 17148):
        grp = [(i, src, s, st) for i, (src, ex, s, st) in enumerate(ins) if ex == want]
        tot = sum(s for _, _, s, _ in grp)
        print(f"\n-- K1 exec-count {want}: {len(grp)} instructions, {int(tot)} samples; hottest 14 (index, samples, top stalls, instruction):")
        for i, src, s, st in sorted(grp, key=lambda x: -x[2])[:14]:
            top = ",".join(f"{k}:{int(v)}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:2] if v > 0)
            print(f"   {i:5d} {int(s):6d}  {top:32s} {src[:70]}")
        ops = collections.Counter(re.sub(r"^@!?U?P[T0-9]+\s+", "", src).split()[0].split(".")[0] for _, src, _, _ in grp)
        print("   opcode mix:", ", ".join(f"{k} {v}" for k, v in ops.most_common(14)))
