import re, sys, difflib
def parse(path):
    funcs, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (.*)$", line)
        if m: cur = m.group(1).strip(); funcs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*/\*", line)
        if m: funcs[cur].append(m.group(1).strip())
    return funcs
def pick(f, key): return [v for k, v in f.items() if key in k][0]
a = pick(parse(sys.argv[1]), sys.argv[2]); b = pick(parse(sys.argv[3]), sys.argv[4])
norm = lambda ins: [re.sub(r"0x[0-9a-f]+", "ADDR", x) if re.search(r"\b(BRA|BSSY|CALL|JMP|BRX)\b", x) else x for x in ins]
na, nb = norm(a), norm(b)
print(f"  {sys.argv[5]}: {len(a)} -> {len(b)} instr; {'IDENTICAL' if a == b else ('identical apart from branch addresses' if na == nb else 'DIFFERENT')}")
if na != nb:
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
        if op != "equal": print(f"     {op:7s} {na[a0:a1]} -> {nb[b0:b1]}")
