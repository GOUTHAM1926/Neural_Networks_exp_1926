# loop skeleton: every non-math instruction of the steady loop (+ HGMMA / barriers / waits), with samples and stalls;
# runs of math opcodes are folded into one line "[n math: FMNMX a, MUFU b, ...] samples s"
import collections, re, sys
exec(open("session_2026_10_17/deep/ncu_k5q_vs_fa3.py").read().split("\nM = [")[0].replace("NHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 25", ""))
MATH = {"FMNMX", "MUFU", "FFMA", "FMUL", "FADD", "F2FP", "FSEL", "FSETP"}
def skel(name, shape):
    p, t = [(p, t) for n, p, t in SRC[shape] if n == name][0]
    ins = src_kernel(p, t); tot = sum(s for _, _, s, _ in ins)
    g = collections.defaultdict(float)
    for src, ex, s, st in ins: g[ex] += s
    lx = max(g, key=g.get)
    print(f"\n######## {name} {shape}: steady loop = executed {lx}x; samples as % of the whole kernel")
    run = []; run_s = 0.0
    def flush():
        nonlocal run, run_s
        if run:
            c = collections.Counter(run)
            print(f"          [{len(run):3d} math: " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + f"]  {100 * run_s / tot:5.2f}%")
        run = []; run_s = 0.0
    for i, (src, ex, s, st) in enumerate(ins):
        if ex != lx: continue
        o = opc(src)
        if o in MATH: run.append(o); run_s += s; continue
        flush()
        top = ",".join(f"{k}:{int(v)}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:3] if v > 0)
        print(f"   {i:5d} {100 * s / tot:5.2f}%  {top:40s} {src[:84]}")
    flush()
skel(sys.argv[1], sys.argv[2])
