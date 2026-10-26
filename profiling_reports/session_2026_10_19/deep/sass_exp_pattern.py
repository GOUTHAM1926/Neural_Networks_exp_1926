# 2026-10-19: the steady loop's exp spacing and letter pattern from a plain SASS listing (cuobjdump output split per kernel,
# *.fsass), to compare a LOCAL build with the box's ncu SASS before any GPU run (work doc 45.52). The steady loop = the part
# between the 1st and 2nd QK-wait (WARPGROUP.DEPBAR ... 0x1) of the mask-free step body: from the 16 HGMMAs before the
# QK wait to the next step's HGMMAs.
# usage: python3 sass_exp_pattern.py <file.fsass> [which QK wait; default: the shortest step with all 66 exps = mask-free]
import re, sys
L = [re.sub(r"^@!?U?P[T0-9]+\s+", "", l.strip()).rstrip(" ;") for l in open(sys.argv[1])]
dep1 = [i for i, t in enumerate(L) if t.startswith("WARPGROUP.DEPBAR") and t.endswith("0x1")]
def body(k):
    a = dep1[k]; nxt = dep1[k + 1] if k + 1 < len(dep1) else len(L)
    end = next((i for i in range(a + 1, nxt) if L[i].startswith("HGMMA")), nxt)   # until the next HGMMA (next step's issue)
    return a, end
cands = [body(k) for k in range(len(dep1))]
full = [ab for ab in cands if sum(1 for t in L[ab[0]:ab[1]] if t.startswith("MUFU.EX2")) >= 60]
a, e = cands[int(sys.argv[2])] if len(sys.argv) > 2 else min(full, key=lambda ab: ab[1] - ab[0])   # the mask-free step: shortest with all exps
seg = L[a:e]
ex = [i for i, t in enumerate(seg) if t.startswith("MUFU.EX2")]
gaps = [ex[k] - ex[k - 1] - 1 for k in range(1, len(ex))]
dep0 = next((i for i, t in enumerate(seg) if t.startswith("WARPGROUP.DEPBAR") and t.endswith("0x0")), None)
before = sum(1 for i in ex if dep0 is not None and i < dep0)
def letter(t):
    for pre, ch in (("MUFU.EX2", "M"), ("FMNMX", "x"), ("FFMA", "f"), ("FADD", "a"), ("FMUL", "m"), ("F2FP", "p"), ("SHFL", "s"),
                    ("WARPGROUP.DEPBAR", "W"), ("HGMMA", "H"), ("BAR", "b"), ("SYNCS", "y")):
        if t.startswith(pre): return ch
    return "u" if t.startswith(("U", "R2UR")) else "i"
pat = "".join(letter(t) for t in seg)
print(f"{sys.argv[1].split('/')[-1]}: softmax segment {len(seg)} instr, exps {len(ex)} ({before} before the P.V wait) | exps right after the previous one: "
      f"0 between {sum(g == 0 for g in gaps)}, 1 between {sum(g == 1 for g in gaps)}, 2 between {sum(g == 2 for g in gaps)}, 3-4 {sum(3 <= g <= 4 for g in gaps)}, 5+ {sum(g >= 5 for g in gaps)}")
print("   " + pat[:200]); print("   " + pat[200:400])
