import re, csv, collections, sys
def sass_fn(path, tag):
    txt = open(path).read()
    f = [f for f in re.split(r"\n\s+Function : ", txt)[1:] if tag in f.split("\n", 1)[0]][0]
    return [m.group(1).strip() for m in re.finditer(r"/\*[0-9a-f]{4,6}\*/\s+([^;]*);", f)]
def csv_fn(path, want):
    rows = list(csv.reader(open(path))); blocks = []; cur = None
    for r in rows:
        if r and r[0] == "Kernel Name": cur = {"name": r[1], "rows": []}; blocks.append(cur); continue
        if cur is not None: cur["rows"].append(r)
    b = next(b for b in blocks if want in b["name"]); h = b["rows"][0]
    i = h.index("Source"); e = h.index("Instructions Executed")
    return [(r[i].strip(), int(r[e] or 0)) for r in b["rows"][1:] if len(r) > i]
def op(s):
    s = re.sub(r"^@!?U?P[T0-9]+\s+", "", s)
    return s.split()[0].split(".")[0] if s else ""
k5 = sass_fn(sys.argv[1], "ILi0E")
qk = [i for i, s in enumerate(k5) if s.startswith("HGMMA") and "RZ, !UPT" in s]
steps = [(qk[i], qk[i + 1]) for i in range(len(qk) - 1) if qk[i + 1] - qk[i] > 300]
fa = csv_fn(sys.argv[2], "FlashAttnFwdSm90")
cnt = collections.Counter(e for s, e in fa if e > 0)
loopn = max(c for c in cnt if cnt[c] > 200)
fa_loop = [s for s, e in fa if e == loopn]
print("FA3: instructions executed", loopn, "times (its main loop):", len(fa_loop), "per step")
print("K5q static step regions (QK group start -> next QK group start):", [(a, b, b - a) for a, b in steps])
a, b = steps[int(sys.argv[3]) if len(sys.argv) > 3 else 0]
ours = k5[a:b]
co = collections.Counter(op(s) for s in ours); cf = collections.Counter(op(s) for s in fa_loop)
print(f"ours {len(ours)} per step | FA3 {len(fa_loop)} per step")
print(f"{'opcode':12s} {'ours':>5s} {'FA3':>5s} {'diff':>5s}")
for k in sorted(set(co) | set(cf), key=lambda k: -(co[k] - cf[k])):
    if co[k] != cf[k]: print(f"{k:12s} {co[k]:5d} {cf[k]:5d} {co[k] - cf[k]:+5d}")
