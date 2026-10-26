# 2026-10-19 (work doc 45.58): the STEADY LOOP of the forward kernel (dense = gqa_v62_sm90<0>, packed = <1>) from a cubin or a
# library, for the gate "a candidate that does not target the loop must leave its SASS identical". The loop = the smallest
# range [target, backward BRA] holding 16 HGMMA (8 QK + 8 P.V) and >= 64 MUFU.EX2. Normalised: branch targets inside the loop
# relative to its start, outside it as OUT, kernel-parameter offsets c[0x0][..] as c[P]. Also prints, per kernel, the whole
# kernel's instruction count and the counts the candidates change (UTMAPF = TMA L2 prefetch, I2F.RP = the start of an integer
# division, ISETP / FSEL = the causal mask) and their positions relative to the loop (before / in / after).
# usage: python3 loop_sass.py <base.cubin|.so> <cand.cubin|.so> [more candidates ...]   (cuobjdump from /usr/local/cuda/bin)
# Exit status: 0; the gate itself is read by the caller from the printed lines.
import re, subprocess, sys, difflib

NAMES = {k: f"_ZN9OwnTensor4cuda5gqa6212gqa_v62_sm90ILi{i}EEEv14CUtensorMap_stS3_S3_PK13__nv_bfloat16PS4_PfiiiifPK4int4i"
         for i, k in ((0, "dense"), (1, "packed"))}   # PF's signature (3 tensor maps); the QP* candidates keep it

def functions(path):
    # only the two forward kernels (-fun): a whole libtensor.so would be gigabytes of SASS text
    out = "".join(subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", "-fun", NAMES[k], path], capture_output=True, text=True).stdout
                  for k in ("dense", "packed"))
    funs, cur = {}, None
    for line in out.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            name = m.group(1)
            cur = "dense" if "gqa_v62_sm90ILi0E" in name else ("packed" if "gqa_v62_sm90ILi1E" in name else None)
            if cur: funs[cur] = []
            continue
        m = re.match(r"\s+/\*([0-9a-f]{4,6})\*/\s+(.*?)\s*;?\s*/\*", line)
        if m and cur: funs[cur].append((int(m.group(1), 16), m.group(2).strip().rstrip(";").strip()))
    return funs

def steady_loop(ins):
    idx = {a: i for i, (a, _) in enumerate(ins)}
    best = None
    for i, (a, t) in enumerate(ins):
        m = re.search(r"\bBRA (0x[0-9a-f]+)", t)
        if not m: continue
        tgt = int(m.group(1), 16)
        if tgt >= a or tgt not in idx: continue
        lo = idx[tgt]; seg = ins[lo:i + 1]
        hg = sum(1 for _, x in seg if re.search(r"(^|\s)HGMMA", x)); ex = sum(1 for _, x in seg if "MUFU.EX2" in x)
        if hg == 16 and ex >= 64 and (best is None or i - lo < best[1] - best[0]): best = (lo, i)
    return best

def norm(ins, lo, hi):
    a0, a1 = ins[lo][0], ins[hi][0]
    out = []
    for a, t in ins[lo:hi + 1]:
        def rel(m):
            tgt = int(m.group(1), 16)
            return f"BRA L+{tgt - a0:#x}" if a0 <= tgt <= a1 else "BRA OUT"
        t = re.sub(r"\bBRA (0x[0-9a-f]+)", rel, t)
        t = re.sub(r"c\[0x0\]\[0x[0-9a-f]+\]", "c[P]", t)
        out.append(re.sub(r"\s+", " ", t))
    return out

def where(ins, lo, hi, pat):
    b = sum(1 for i, (_, t) in enumerate(ins) if i < lo and re.search(pat, t))
    w = sum(1 for i, (_, t) in enumerate(ins) if lo <= i <= hi and re.search(pat, t))
    a = sum(1 for i, (_, t) in enumerate(ins) if i > hi and re.search(pat, t))
    return f"{b}/{w}/{a}"

paths = sys.argv[1:]
F = {p: functions(p) for p in paths}
for k in ("dense", "packed"):
    base = F[paths[0]][k]; bl = steady_loop(base); bn = norm(base, *bl)
    print(f"== {k}")
    for p in paths:
        ins = F[p][k]; lp = steady_loop(ins)
        if lp is None: print(f"   {p}: NO steady loop found"); continue
        ln = norm(ins, *lp)
        d = sum(1 for x in difflib.unified_diff(bn, ln, lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("+++", "---")))
        nm = p.split('/')[-2] if p.endswith('libtensor.so') else p.split('/')[-1]   # libs/<name>/libtensor.so -> <name>
        print(f"   {nm:14s} kernel {len(ins):5d} instr | loop {lp[1] - lp[0] + 1:4d} instr, {d:3d} lines differ from the base's loop | "
              f"before/in/after the loop: UTMAPF {where(ins, *lp, r'UTMAPF')}, I2F.RP {where(ins, *lp, r'I2F\.RP')}, "
              f"ISETP {where(ins, *lp, r'^(@\S+ )?ISETP')}, FSEL {where(ins, *lp, r'^(@\S+ )?FSEL')}, HGMMA {where(ins, *lp, r'HGMMA')}")
