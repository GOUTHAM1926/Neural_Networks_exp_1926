# 2026-10-19: spacing between consecutive exps (MUFU.EX2) of the steady loop step, and the "wait" stall (ncu PC sampling x
# traced period) each exp carries, per distance to the previous exp. The exp unit takes a warp's MUFU about 8 cycles, so an
# exp issued 0-2 instructions after the previous one waits (ptxas encodes the wait in the control bits -> "wait" reason).
# Work doc 45.52. usage (from profiling_reports/): python3 session_2026_10_19/deep/exp_spacing.py
import re, collections
exec(open("session_2026_10_19/deep/loop_diff_fa3.py").read().split("res = {}")[0])
for name, path, tag, period in KERNELS:
    ins = load(path, tag); lx, lo, hi = loop_of(ins); L = ins[lo:hi + 1]; tot = sum(x["smp"] for x in L)
    T = [re.sub(r"^@!?U?P[T0-9]+\s+", "", x["src"]).rstrip(" ;") for x in L]
    ex = [i for i, t in enumerate(T) if t.startswith("MUFU.EX2")]
    buckets = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for k, i in enumerate(ex):
        d = (i - ex[k - 1] - 1) if k else 99
        b = "0" if d == 0 else "1" if d == 1 else "2" if d == 2 else "3-4" if d <= 4 else "5-8" if d <= 8 else ">8"
        w = L[i]["st"].get("wait", 0) / tot * period; a = L[i]["smp"] / tot * period
        buckets[b][0] += 1; buckets[b][1] += w; buckets[b][2] += a
    print(f"{name}: 66 exps by instructions since the previous exp -> count, their 'wait' cycles, all cycles (per step)")
    for b in ("0", "1", "2", "3-4", "5-8", ">8"):
        n, w, a = buckets[b]
        print(f"   {b:>4s}: {n:2d} exps | wait {w:5.0f} | all {a:5.0f}")
