import re, sys, glob, os
def parse(path):
    funcs, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (.*)$", line)
        if m: cur = m.group(1).strip(); funcs[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*/\*", line)
        if m: funcs[cur].append(m.group(1).strip())
    return funcs
def key(name):           # base name + template args list
    m = re.match(r"void (.*?)<(.*?)>\(", name)
    return (m.group(1), [a.strip() for a in m.group(2).split(",")]) if m else (name, [])
total = same = 0
for f in sorted(glob.glob("orig/*.sass")):
    o, n = parse(f), parse("new/" + os.path.basename(f))
    nmap = {}
    for nm in n:
        b, args = key(nm); nmap[(b, tuple(args))] = nm
    for on, oins in o.items():
        b, args = key(on)
        target = nmap.get((b, tuple(args + ["(bool)0"]))) if args else (on if on in n else None)
        total += 1
        if target is None:
            print("NO MATCH for", on[:120]); continue
        ok = (oins == n[target])
        same += ok
        print(("IDENTICAL " if ok else "DIFFERENT ") + f"{len(oins):4d} vs {len(n[target]):4d} instr  {b.split('::')[-1]}<{', '.join(args)}> -> <..., false>")
        if not ok:
            for i, (x, y) in enumerate(zip(oins, n[target])):
                if x != y: print("   first diff at", i, "|", x, "|", y); break
print(f"\n{same}/{total} original kernels have byte-for-byte identical SASS in the new HAS_POS=false instantiation")
