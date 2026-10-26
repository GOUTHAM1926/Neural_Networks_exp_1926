# 2026-10-17: the SHAPE of the forward's softmax around the P.V wait, from static SASS (cuobjdump -sass) or an ncu source CSV.
# For every "wait QK" (WARPGROUP.DEPBAR.LE gsb0, 0x1) it counts the instructions up to the next "wait P.V" (DEPBAR ..., 0x0)
# and from there to the next HGMMA / end of block: FA3 (ncu 2026-10-13) has the whole row max + 13 exps BEFORE the P.V wait.
# usage: sass_shape.py <file.sass | source.csv> [kernel substring for csv]
import sys, re, csv
from collections import Counter
path = sys.argv[1]
if path.endswith(".csv"):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    want = sys.argv[2]; blk = next(b for b in blocks if want in b["name"]); h = blk["rows"][0]; i = h.index("Source")
    funcs = {blk["name"][:60]: [r[i].strip() for r in blk["rows"][1:] if len(r) > i]}
else:
    txt = open(path).read(); funcs = {}
    for f in re.split(r"\n\s+Function : ", txt)[1:]:
        name = f.split("\n", 1)[0].strip()
        funcs[name[:60]] = [m.group(1).strip() for m in re.finditer(r"/\*[0-9a-f]{4,6}\*/\s+([^;]*);", f)]
KEY = ("FMNMX", "MUFU", "FFMA", "FADD", "FMUL", "F2FP", "FSEL", "SHFL", "BRA", "HGMMA")
def op(s): t = s.split(); t = t[1:] if t and t[0].startswith("@") else t; return t[0].split(".")[0] if t else ""
for name, ins in funcs.items():
    print(f"== {name}  ({len(ins)} instructions)")
    for k, s in enumerate(ins):
        if "DEPBAR.LE gsb0, 0x1" not in s: continue
        j = next((q for q in range(k + 1, len(ins)) if "DEPBAR.LE gsb0, 0x0" in ins[q]), None)
        if j is None: continue
        e = next((q for q in range(j + 1, len(ins)) if op(ins[q]) == "HGMMA" or "DEPBAR.LE gsb0, 0x1" in ins[q]), len(ins))
        a, b = Counter(op(x) for x in ins[k + 1:j]), Counter(op(x) for x in ins[j + 1:e])
        print(f"  wait QK @{k}: before P.V wait {j - k - 1:4d} instr [" + " ".join(f"{o} {a[o]}" for o in KEY if a[o]) +
              f"] | after {e - j - 1:4d} [" + " ".join(f"{o} {b[o]}" for o in KEY if b[o]) + "]")
