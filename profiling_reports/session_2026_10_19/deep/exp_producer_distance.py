# 2026-10-19: for every MUFU.EX2 (and F2FP / FMUL) of the steady loop step: how many instructions sit between it and the
# instruction that produces its input register, and the "wait" stall time it carries (ncu samples x traced period). A
# fixed-latency producer (FFMA ~4+ cycles) right before its consumer makes the consumer stall ("wait"). Work doc 45.52.
# usage (from profiling_reports/): python3 session_2026_10_19/deep/exp_producer_distance.py
import re, collections
exec(open("session_2026_10_19/deep/loop_diff_fa3.py").read().split("res = {}")[0])
def regs(t):
    return re.findall(r"\bR\d+\b", t)
for name, path, tag, period in KERNELS:
    ins = load(path, tag); lx, lo, hi = loop_of(ins); L = ins[lo:hi + 1]; tot = sum(x["smp"] for x in L)
    T = [re.sub(r"^@!?U?P[T0-9]+\s+", "", x["src"]).rstrip(" ;") for x in L]
    for target in ("MUFU.EX2", "F2FP", "FMUL"):
        dist = []; waits = []; prods = collections.Counter()
        for i, t in enumerate(T):
            if not t.startswith(target): continue
            r = regs(t)
            if len(r) < 2: continue
            srcs = r[1:]
            best = None
            for s in srcs:
                for j in range(i - 1, -1, -1):          # last writer of s before i within the step
                    rj = regs(T[j])
                    if rj and rj[0] == s and not T[j].startswith(("ST", "HGMMA")):
                        if best is None or j > best[0]: best = (j, T[j].split()[0])
                        break
            if best is None: continue
            d = i - best[0] - 1; dist.append(d); prods[best[1].split(".")[0]] += 1
            waits.append(L[i]["st"].get("wait", 0) / tot * period)
        if not dist: continue
        near = sum(1 for d in dist if d <= 1)
        wn = sum(w for d, w in zip(dist, waits) if d <= 1); wf = sum(w for d, w in zip(dist, waits) if d > 1)
        print(f"{name:4s} {target:8s}: {len(dist)} instr | producer 0-1 instructions before: {near:2d} (their wait {wn:5.0f} cyc), "
              f"farther: {len(dist) - near:2d} (wait {wf:5.0f} cyc) | median distance {sorted(dist)[len(dist) // 2]} | producers {dict(prods)}")
