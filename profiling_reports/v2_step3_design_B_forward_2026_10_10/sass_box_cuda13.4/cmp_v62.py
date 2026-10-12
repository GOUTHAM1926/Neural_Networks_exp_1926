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
orig = [v for k, v in o.items() if "gqa_v62_sm90" in k][0]
dense = [v for k, v in n.items() if "gqa_v62_sm90ILb0E" in k][0]
packed = [v for k, v in n.items() if "gqa_v62_sm90ILb1E" in k][0]
print(f"original v62        : {len(orig)} instructions")
print(f"new <false> (dense) : {len(dense)} instructions -> {'IDENTICAL' if orig == dense else 'DIFFERENT'} to the original")
if orig != dense:
    for i, (a, b) in enumerate(zip(orig, dense)):
        if a != b: print('   first diff at', i, '|', a, '|', b); break
print(f"new <true> (packed) : {len(packed)} instructions -> {'IDENTICAL' if orig == packed else 'DIFFERENT'} to the original (must differ)")
ld = lambda ins: sum(1 for x in ins if x.startswith('LDG'))
print(f"global loads (LDG*) : original {ld(orig)}, dense {ld(dense)}, packed {ld(packed)}")
