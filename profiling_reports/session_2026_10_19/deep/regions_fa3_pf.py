# 2026-10-19 (work doc 45.57): FA3 vs PF dense, region by region with the SAME cut points in the unit's life, per unit (cycles =
# share of all samples x SM cycles / units; warp instructions = executions / units): instructions, cycles, stall reasons, and
# the opcodes that differ. Regions from deep/kernel_map.py (executed-count runs) and the wait list (deep/waits_fa3_pf.py):
#   start      = after the previous unit's epilogue .. the first tile's K landed (Q wait, K wait, scheduler)
#   first tile = the first QK, its softmax (FA3: + the causal mask: its first tile is the diagonal) .. before the loop
#   loop       = the steady QK(n) + P.V(n-1) + softmax(n) steps (FA3 15.5 per unit, PF 14.53)
#   diagonal   = PF only: the loop exit + the masked diagonal step (FA3 has one more loop step instead)
#   end        = the last P.V + epilogue (+ FA3's TMA-store warp)
#   producer   = the producer warp
import csv, re, collections, sys
exec(open("session_2026_10_19/deep/waits_fa3_pf.py").read().split("KERN = {")[0])
R = {"FA3": (B + "ncu_fa3_k5q_1019/fa3_dense", "FlashAttnFwdSm90",
             [("prologue", 0, 208), ("start", 208, 330), ("first tile", 330, 1558), ("loop", 1558, 2027), ("diagonal", 2027, 2027),
              ("end", 2027, 2563), ("producer", 2563, 10 ** 6)]),
     "PF": (B + "pf_1019/ncu_vj2_PF_dense", "gqa_v62",
            [("prologue", 0, 199), ("start", 199, 306), ("first tile", 306, 1042), ("loop", 1042, 1523), ("diagonal", 1523, 2142),
             ("end", 2142, 2707), ("producer", 2707, 10 ** 6)])}
def op(s):
    t = [w for w in s.split() if not w.startswith("@")]
    return t[0].split(".")[0] if t else "?"
data = {}
for n, (base, tag, regs) in R.items():
    ins, cyc = load(base, tag)
    d = {}
    for name, lo, hi in regs:
        seg = ins[lo:min(hi, len(ins))]
        st = collections.Counter(); ops = collections.Counter()
        for x in seg: st.update(x["stc"]); ops[op(x["src"])] += x["ex"] / 4608
        d[name] = dict(cyc=sum(x["cyc"] for x in seg), n=sum(x["ex"] for x in seg) / 4608, st=st, ops=ops)
    data[n] = (d, cyc)
names = [r[0] for r in R["PF"][2]]
print(f"{'region':12s} | {'FA3 instr':>9s} {'cycles':>7s} | {'PF instr':>9s} {'cycles':>7s} | {'d instr':>8s} {'d cycles':>8s} | biggest stall changes | biggest opcode changes")
tot = [0, 0, 0, 0]
for nm in names:
    a = data["FA3"][0][nm]; b = data["PF"][0][nm]
    tot[0] += a["n"]; tot[1] += a["cyc"]; tot[2] += b["n"]; tot[3] += b["cyc"]
    ds = sorted(((k, b["st"][k] - a["st"][k]) for k in set(a["st"]) | set(b["st"])), key=lambda kv: -abs(kv[1]))[:4]
    do = sorted(((k, b["ops"][k] - a["ops"][k]) for k in set(a["ops"]) | set(b["ops"])), key=lambda kv: -abs(kv[1]))[:5]
    print(f"{nm:12s} | {a['n']:9.0f} {a['cyc']:7.0f} | {b['n']:9.0f} {b['cyc']:7.0f} | {b['n'] - a['n']:+8.0f} {b['cyc'] - a['cyc']:+8.0f} | "
          + ", ".join(f"{k} {v:+.0f}" for k, v in ds) + " | " + ", ".join(f"{k} {v:+.0f}" for k, v in do))
print(f"{'total':12s} | {tot[0]:9.0f} {tot[1]:7.0f} | {tot[2]:9.0f} {tot[3]:7.0f} | {tot[2] - tot[0]:+8.0f} {tot[3] - tot[1]:+8.0f}")
