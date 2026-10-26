# 2026-10-19: the softmax instruction SCHEDULE of the steady forward loop, ours vs FA3, from ncu source CSVs of the same box.
# For each kernel: the loop body (the executed count holding the most stall samples), cut at the GEMM issue / waits; then for
# the softmax part: where the 66 MUFU.EX2 sit, how many instructions separate consecutive exps (one exp keeps the exp unit
# busy ~8 cycles per warp, so <7 other instructions between two exps = the warp stalls), how far each exp is from the
# instruction that wrote its input and from the first one that reads its result, and the stall samples per opcode.
# usage: python3 softmax_sched.py <ours source.csv> <fa3 source.csv>
import csv, re, sys, collections, statistics as st

def load(path, tag):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if tag in b["name"]); h = b["rows"][0]; ix = {c: i for i, c in enumerate(h)}
    out = []
    for r in b["rows"][1:]:
        if len(r) < len(h): continue
        g = lambda c: float(r[ix[c]].replace(",", "") or 0) if c in ix else 0.0
        st_ = {c[6:]: g(c) for c in h if c.startswith("stall_") and "Not Issued" not in c}
        out.append(dict(src=r[ix["Source"]].strip(), ex=int(g("Instructions Executed")), s=g("Warp Stall Sampling (All Samples)"), st=st_))
    return out

def op(s): return re.sub(r"^@!?U?P[T0-9]+\s+", "", s).split()[0] if s else ""
def regs(s):
    s = re.sub(r"^@!?U?P[T0-9]+\s+", "", s).replace(".reuse", "")
    parts = s.split(None, 1)
    if len(parts) < 2: return None, []
    ops = [o.strip() for o in parts[1].split(",")]
    dst = re.findall(r"\bR(\d+)\b", ops[0]) if ops else []
    src = [int(x) for o in ops[1:] for x in re.findall(r"\bR(\d+)\b", o)]
    return (int(dst[0]) if dst else None), src

def analyse(name, ins):
    tot = sum(i["s"] for i in ins)
    g = collections.defaultdict(float)
    for i in ins: g[i["ex"]] += i["s"]
    lx = max(g, key=g.get)
    body = [(k, i) for k, i in enumerate(ins) if i["ex"] == lx]
    # the softmax part = from the first WARPGROUP.DEPBAR (QK done) to the end of the body
    first_dep = next(p for p, (k, i) in enumerate(body) if op(i["src"]).startswith("WARPGROUP.DEPBAR"))
    sm = body[first_dep:]
    ops = [op(i["src"]) for _, i in sm]
    mufu = [p for p, o in enumerate(ops) if o.startswith("MUFU")]
    gaps = [b - a - 1 for a, b in zip(mufu, mufu[1:])]
    # dependency distances inside the softmax part (producer = last earlier writer of the source register)
    dist_in, dist_out = [], []
    for p in mufu:
        d, srcs = regs(sm[p][1]["src"])
        prod = [q for q in range(p - 1, -1, -1) if regs(sm[q][1]["src"])[0] in srcs]
        dist_in.append(p - prod[0] - 1 if prod else 99)
        cons = [q for q in range(p + 1, len(sm)) if d in regs(sm[q][1]["src"])[1]]
        dist_out.append(cons[0] - p - 1 if cons else 99)
    s_sm = sum(i["s"] for _, i in sm); s_mufu = sum(sm[p][1]["s"] for p in mufu)
    by = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
    for _, i in sm:
        o = op(i["src"]).split(".")[0]; by[o][0] += 1; by[o][1] += i["s"]; by[o][2].update(i["st"])
    print(f"\n=== {name}: loop body {len(body)} instructions (executed {lx}x, {100 * sum(i['s'] for _, i in body) / tot:.1f}% of samples); "
          f"softmax part from the QK wait = {len(sm)} instructions, {100 * s_sm / tot:.1f}% of samples")
    print(f"  exps: {len(mufu)} | instructions between consecutive exps: median {st.median(gaps)}, "
          f"0-1: {sum(g <= 1 for g in gaps)}, 2-6: {sum(2 <= g <= 6 for g in gaps)}, >=7: {sum(g >= 7 for g in gaps)}")
    print(f"  input written by the instruction just before (distance 0): {sum(d == 0 for d in dist_in)} of {len(mufu)} | "
          f"distance median {st.median(dist_in)} | result read by the next instruction: {sum(d == 0 for d in dist_out)} | median {st.median(dist_out)}")
    print(f"  stall samples on the exps: {100 * s_mufu / tot:.1f}% of the kernel; top reasons: " +
          ", ".join(f"{k} {100 * v / max(1, s_mufu):.0f}%" for k, v in collections.Counter({k: sum(sm[p][1]['st'].get(k, 0) for p in mufu) for k in sm[mufu[0]][1]['st']}).most_common(4)))
    print("  per opcode (count, % of kernel samples, top stalls):")
    for o, (n, s, c) in sorted(by.items(), key=lambda kv: -kv[1][1])[:9]:
        print(f"     {o:10s} {n:4d} {100 * s / tot:5.1f}%  " + ", ".join(f"{k} {100 * v / max(1, s):.0f}%" for k, v in c.most_common(3)))
    # the shape: a compact stream (M = exp, f = FFMA, a = FADD, m = FMUL, p = F2FP, x = FMNMX, s = SHFL, W = DEPBAR, . = other)
    code = {"MUFU": "M", "FFMA": "f", "FADD": "a", "FMUL": "m", "F2FP": "p", "FMNMX": "x", "SHFL": "s", "WARPGROUP": "W", "SYNCS": "Y", "BAR": "B"}
    stream = "".join(code.get(o.split(".")[0], ".") for o in ops)
    print("  stream (M exp, f FFMA, a FADD, m FMUL, p F2FP pack, x FMNMX max, s SHFL, W wait, Y mbarrier, . other):")
    for k in range(0, len(stream), 100): print("     " + stream[k:k + 100])
    return dict(gaps=gaps, din=dist_in, dout=dist_out)

o = analyse("ours K5q (dense)", load(sys.argv[1], "gqa_v62"))
f = analyse("FA3 (dense)", load(sys.argv[2], "FlashAttnFwdSm90"))
