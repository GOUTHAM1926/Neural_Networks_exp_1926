# 2026-10-17: a SIMPLE in-order issue model of one warp running a SASS region alone on its SM sub-partition: one instruction
# per cycle at most, in order; an instruction issues when its source registers are ready (fixed latencies below) and its
# unit is free (MUFU: 8 cycles per warp instruction = 16 lanes/clk/SM; F2FP assumed 2). Rough on purpose: the point is to
# compare two kernels' softmax schedules under the SAME assumptions, not to predict absolute cycles.
# usage: sass_timing.py <file.sass | source.csv> <region spec>   (see main: it picks the region after a wait-QK)
import re, sys, csv
LAT = {"FMNMX": 5, "FFMA": 4, "FADD": 4, "FMUL": 4, "IMAD": 4, "IADD3": 4, "LOP3": 4, "SHF": 4, "LEA": 4, "ISETP": 4, "FSEL": 4,
       "MOV": 2, "SEL": 4, "MUFU": 18, "SHFL": 24, "F2FP": 6, "PRMT": 4, "VIADD": 4, "FSETP": 4, "IMNMX": 4, "UMOV": 2}
UNIT = {"MUFU": ("xu", 8), "F2FP": ("cvt", 2), "SHFL": ("mio", 2)}
def regs(tok):
    out = []
    for m in re.finditer(r"\bR(\d+)(\.64)?\b", tok):
        r = int(m.group(1)); out += [f"R{r}", f"R{r+1}"] if m.group(2) else [f"R{r}"]
    out += [f"P{m.group(1)}" for m in re.finditer(r"\bP(\d)\b", tok)]
    return out
def simulate(ins):
    ready = {}; unit_free = {}; t = 0; per = []
    for s in ins:
        s = s.strip().rstrip(";")
        pred = re.match(r"@!?(U?P\d)\s+", s); body = s[pred.end():] if pred else s
        op = body.split()[0].split(".")[0]; ops = body[len(body.split()[0]):]
        parts = [p.strip() for p in ops.split(",")]
        dst = regs(parts[0]) if parts and op not in ("SYNCS", "STG", "STS", "BAR", "BRA", "WARPGROUP") else []
        src = [r for p in parts[1:] for r in regs(p)] + ([pred.group(1)] if pred and not pred.group(1).startswith("U") else [])
        if op in ("ISETP", "FSETP") and parts: dst = regs(parts[0]) + (regs(parts[1]) if len(parts) > 1 else []); src = [r for p in parts[2:] for r in regs(p)]
        if op == "SHFL" and parts: dst = regs(parts[1]) if len(parts) > 1 else []; src = [r for p in parts[2:] for r in regs(p)]
        start = max([t] + [ready.get(r, 0) for r in src])
        if op in UNIT:
            u, occ = UNIT[op]; start = max(start, unit_free.get(u, 0)); unit_free[u] = start + occ
        for r in dst: ready[r] = start + LAT.get(op, 4)
        t = start + 1; per.append((op, start))
    end = max([t] + list(ready.values()))
    return end, per
def from_sass(path, fn_index, nth_wait):
    txt = open(path).read(); f = re.split(r"\n\s+Function : ", txt)[1 + fn_index]
    return [m.group(1).strip() for m in re.finditer(r"/\*[0-9a-f]{4,6}\*/\s+([^;]*);", f)]
def from_csv(path, want):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if want in b["name"]); h = b["rows"][0]; i = h.index("Source")
    return [r[i].strip() for r in b["rows"][1:] if len(r) > i]
def region(ins, nth, to):   # from the nth "wait QK" to: 'wait0' (the P.V wait) or 'hgmma' (the next HGMMA / loop end)
    ws = [k for k, s in enumerate(ins) if "DEPBAR.LE gsb0, 0x1" in s]; a = ws[nth]
    w0 = next(k for k in range(a, len(ins)) if "DEPBAR.LE gsb0, 0x0" in ins[k])
    if to == "wait0": return ins[a + 1:w0], ins[w0 + 1:next(k for k in range(w0, len(ins)) if "HGMMA" in ins[k] or "BRA" in ins[k] and k > w0 + 60)]
    return ins[a + 1:w0], []
if __name__ == "__main__":
    for label, ins, nth in [("K1 (loop copy 2)", from_sass(sys.argv[1], 0, 0), 1), ("F3c", from_sass(sys.argv[2], 0, 0), 0),
                            ("FA3 (unmasked loop)", from_csv(sys.argv[3], "FlashAttnFwdSm90"), 1)]:
        pre, post = region(ins, nth, "wait0")
        e1, p1 = simulate(pre); e2, p2 = simulate(post)
        mu1 = sum(1 for o, _ in p1 if o == "MUFU"); mu2 = sum(1 for o, _ in p2 if o == "MUFU")
        print(f"{label:22s} before the P.V wait: {len(pre):4d} instr, {mu1:2d} MUFU -> {e1:5d} cycles alone | after it: {len(post):4d} instr, "
              f"{mu2:2d} MUFU -> {e2:5d} cycles alone | sum {e1 + e2}")
