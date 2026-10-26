# 2026-10-19: our DIAGONAL (masked) step vs a PLAIN step, per execution, by job (work doc 45.55): instructions executed per
# step and stall time (consumer samples / units or steps). PF dense ncu; regions from deep/kernel_map.py: plain step body
# 1042-1522 (535,680 executions), diagonal step 1537-2141 (35,712 = both groups, units with more than one tile).
import csv, re, collections
exec(open("session_2026_10_19/deep/loop_diff_fa3.py").read().split("res = {}")[0])
base = B + "pf_1019/ncu_vj2_PF_dense"
ins = load(base + ".source.csv", "gqa_v62")
raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
cyc = f([dict(zip(rh, x)) for x in raw[2:] if "gqa_v62" in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
cons = sum(x["smp"] for x in ins[:2707])
units, steps = 4608 / 132, 535680 / 8 / 132
def table(a, b, per, norm_ex):
    agg = collections.defaultdict(lambda: [0.0, 0.0, collections.Counter()])
    for x in ins[a:b]:
        j = job(x["src"]); agg[j][0] += x["ex"] / norm_ex; agg[j][1] += x["smp"] / cons * cyc / per
        for k2, v in x["st"].items(): agg[j][2][k2] += v / cons * cyc / per
    return agg
P = table(1042, 1523, steps, 535680)
D = table(1537, 2142, units, 35712)
keys = sorted(set(P) | set(D), key=lambda k: -abs(D.get(k, [0, 0])[1] - P.get(k, [0, 0])[1]))
print(f"{'job':48s} | plain: instr/step  cycles | diagonal: instr/step  cycles | diff cycles | diagonal's top stall reasons")
tp = td = 0
for k in keys:
    p = P.get(k, [0, 0, collections.Counter()]); d = D.get(k, [0, 0, collections.Counter()]); tp += p[1]; td += d[1]
    top = ", ".join(f"{r} {v:.0f}" for r, v in d[2].most_common(3) if v >= 5)
    print(f"{k:48s} | {p[0]:8.1f} {p[1]:8.0f} | {d[0]:8.1f} {d[1]:8.0f} | {d[1] - p[1]:+8.0f} | {top}")
print(f"{'total':48s} | {'':8s} {tp:8.0f} | {'':8s} {td:8.0f} | {td - tp:+8.0f}")
