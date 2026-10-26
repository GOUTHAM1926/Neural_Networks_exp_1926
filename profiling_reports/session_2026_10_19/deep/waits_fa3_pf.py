# 2026-10-19 (work doc 45.57): every WAIT in FA3's and PF's dense forward, per unit, from the same-box ncu source pages:
# mbarrier waits (SYNCS...TRYWAIT -> its retry branch `@!Pn BRA`, where the warp sits), named barriers (BAR.SYNC: the wait
# shows on the next instruction, DEFER_BLOCKING) and wgmma waits (WARPGROUP.DEPBAR). Cycles per unit = share of all samples x
# the kernel's SM cycles / units (4608 units, 132 SMs). Labels: FA3 hopper/flash_fwd_kernel_sm90.h SharedStorage order
# (barrier_Q, barrier_Qv, barrier_O, pipeline_k, pipeline_v ...), ours fwd_PF.cu (full[3], empty[3], qfull[2], qempty[2],
# sched_full[2], order_bar[2] from the mbarrier base).
# usage (from profiling_reports/): python3 session_2026_10_19/deep/waits_fa3_pf.py
import csv, re, collections
B = "session_2026_10_19/box_runs/"; UNITS = 4608 / 132
def f(v):
    try: return float(str(v).replace(",", ""))
    except ValueError: return 0.0
def load(base, tag):
    rows = list(csv.reader(open(base + ".source.csv"))); blk = [i for i, r in enumerate(rows) if r and r[0] == "Kernel Name"]
    k = next(b for b in blk if tag in rows[b][1]); h = rows[k + 1]; idx = {c: i for i, c in enumerate(h)}
    body = [r for r in rows[k + 2:] if len(r) >= len(h)]
    RS = [c for c in h if c.startswith("stall_") and "Not Issued" not in c]
    raw = list(csv.reader(open(base + ".raw.csv"))); rh = raw[0]
    cyc = f([dict(zip(rh, x)) for x in raw[2:] if tag in x[rh.index("Kernel Name")]][0]["sm__cycles_elapsed.avg"])
    ins = []
    for r in body:
        st = {c[6:]: f(r[idx[c]]) for c in RS}; allx = f(r[idx["Warp Stall Sampling (All Samples)"]]); st["gmma"] = max(0.0, allx - sum(st.values()))
        ins.append(dict(src=r[idx["Source"]].strip(), ex=int(f(r[idx["Instructions Executed"]])), smp=allx, st=st))
    tot = sum(x["smp"] for x in ins)
    for x in ins: x["cyc"] = x["smp"] / tot * cyc / UNITS; x["stc"] = {k2: v / tot * cyc / UNITS for k2, v in x["st"].items()}
    return ins, cyc
KERN = {"FA3": (B + "ncu_fa3_k5q_1019/fa3_dense", "FlashAttnFwdSm90", 2302,
                {0x28800: "Q landed (barrier_Q)", 0x28830: "K landed (pipeline_k full)", 0x28850: "V landed (pipeline_v full)"}),
        "PF": (B + "pf_1019/ncu_vj2_PF_dense", "gqa_v62", 2707,
               {0x38000: "K+V landed (full[stage])", 0x38030: "Q landed (qfull[c])", 0x38050: "next unit published (sched_full)"})}
def label(src, labels):
    m = re.search(r"\+0x([0-9a-f]+)\]", src)
    if not m: return "?"
    off = int(m.group(1), 16)
    best = max((o for o in labels if o <= off), default=None)
    return labels[best] if best is not None and off - best < 0x18 else hex(off)
out = {}
for name, (base, tag, hi, labels) in KERN.items():
    ins, cyc = load(base, tag)
    rows = []
    for i in range(hi):
        s = ins[i]["src"]; ex = ins[i]["ex"]
        if ex == 0: continue
        if "TRYWAIT" in s:
            p = re.search(r"TRYWAIT (P\d)", s).group(1)
            j = next((j for j in range(i + 1, min(i + 60, len(ins))) if re.search(r"@!?" + p + r"\s+BRA", ins[j]["src"])), i)
            c = sum(x["cyc"] for x in ins[i:j + 1])
            rows.append((i, ex, "mbarrier: " + label(s, labels), c, f"retry branch at {j}"))
        elif s.split()[0].startswith("BAR.SYNC") or re.match(r"@!?U?P\w+\s+BAR\.SYNC", s):
            c = ins[i]["cyc"] + ins[i + 1]["cyc"]
            rows.append((i, ex, "named barrier " + " ".join(s.split()[-2:]), c, "BAR.SYNC + next instr"))
        elif "WARPGROUP.DEPBAR" in s:
            rows.append((i, ex, "wgmma wait " + s.split()[-1], ins[i]["cyc"], ""))
    out[name] = (rows, cyc)
    print(f"===== {name}: per unit {cyc / UNITS:,.0f} cycles")
    for i, ex, what, c, note in rows:
        if c >= 8: print(f"  {i:5d} ex {ex:>8,} {what:45s} {c:7.0f} cycles/unit   {note}")
