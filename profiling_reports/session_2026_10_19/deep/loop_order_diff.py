# 2026-10-19 (work doc 45.58): is a candidate's steady loop the SAME SCHEDULE as the base's, register names aside? Registers
# (R, UR, P, UP, B) are renamed by order of first appearance inside the loop; then the two loops are compared line by line, and
# by opcode only (difflib alignment: which instructions were inserted / deleted / moved). Also the MUFU.EX2 positions.
# usage: python3 loop_order_diff.py <base.cubin|.so> <cand.cubin|.so> [dense|packed]
import re, sys, difflib, collections
sys.argv, args = sys.argv[:1], sys.argv[1:]
exec(open(__file__.replace("loop_order_diff.py", "loop_sass.py")).read().split("paths = sys.argv[1:]")[0])
def canon(lines):
    m, out = {}, []
    for t in lines:
        def r(mm):
            k = mm.group(0)
            if k in ("RZ", "URZ", "PT", "UPT"): return k
            if k not in m: m[k] = f"{re.match(r'[A-Z]+', k).group(0)}#{len([x for x in m if x[:2] == k[:2]])}"
            return m[k]
        out.append(re.sub(r"\b(UR\d+|UP\d+|R\d+|P\d+|B\d+)\b", r, t))
    return out
def op(t): return re.sub(r"^@!?U?P\S+\s+", "", t).split()[0]
base, cand = args[0], args[1]
kinds = [args[2]] if len(args) > 2 else ["dense", "packed"]
for k in kinds:
    A = functions(base)[k]; Bf = functions(cand)[k]
    la, lb = steady_loop(A), steady_loop(Bf)
    na, nb = canon(norm(A, *la)), canon(norm(Bf, *lb))
    same = sum(1 for x, y in zip(na, nb) if x == y)
    oa, ob = [op(t) for t in na], [op(t) for t in nb]
    sm = difflib.SequenceMatcher(None, oa, ob, autojunk=False)
    ch = [(tag, i2 - i1, j2 - j1, collections.Counter(oa[i1:i2]), collections.Counter(ob[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    ea = [i for i, o in enumerate(oa) if o.startswith("MUFU.EX2")]; eb = [i for i, o in enumerate(ob) if o.startswith("MUFU.EX2")]
    print(f"== {k}: loop {len(na)} vs {len(nb)} instr | identical after register renaming: {same}/{len(na)} lines | opcode order: {sum(1 for c in ch)} changed blocks")
    print(f"   opcode ratio {sm.ratio():.4f}; inserted / deleted opcodes: +{dict(sum((c[4] for c in ch), collections.Counter()))} -{dict(sum((c[3] for c in ch), collections.Counter()))}")
    print(f"   MUFU.EX2 positions equal: {ea == eb} (first diffs: {[(x, y) for x, y in zip(ea, eb) if x != y][:6]})")
