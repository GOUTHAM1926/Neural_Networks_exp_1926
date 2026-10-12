import re, sys
def parse(path):
    funcs, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (.*)$", line)
        if m: cur = m.group(1).strip(); funcs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*/\*", line)
        if m: funcs[cur].append(m.group(1).strip())
    return funcs
o, n = parse(sys.argv[1] + "/orig.sass"), parse(sys.argv[1] + "/new.sass")
print("functions in original:", len(o), "| in new:", len(n))
# helper kernels: same name in both files -> must be identical
for k in sorted(o):
    if "gqa_bwd_vj1dILi" in k: continue
    print(f"  {k[:60]:60s} {'IDENTICAL' if o[k] == n.get(k) else 'DIFFERENT'} ({len(o[k])} instr)")
orig = [v for k, v in o.items() if "gqa_bwd_vj1dILi" in k][0]
dense = [v for k, v in n.items() if "gqa_bwd_vj1dILi64ELi64ELi128ELb0E" in k][0]
packed = [v for k, v in n.items() if "gqa_bwd_vj1dILi64ELi64ELi128ELb1E" in k][0]
print(f"original Vj1d        : {len(orig)} instructions")
print(f"new <false> (dense)  : {len(dense)} instructions -> {'IDENTICAL' if orig == dense else 'DIFFERENT'} to the original")
if orig != dense:
    for i, (a, b) in enumerate(zip(orig, dense)):
        if a != b: print('   first diff at', i, '|', a, '|', b); break
print(f"new <true> (packed)  : {len(packed)} instructions -> {'IDENTICAL' if orig == packed else 'DIFFERENT'} to the original (must differ)")
cnt = lambda ins, p: sum(1 for x in ins if re.sub(r"^@!?U?P\w+\s+", "", x).startswith(p))   # strip a guard predicate first
for p in ("LDG", "BAR", "VOTE", "LDS", "STS"):
    print(f"  {p:5s}: original {cnt(orig,p):4d}  dense {cnt(dense,p):4d}  packed {cnt(packed,p):4d}")
