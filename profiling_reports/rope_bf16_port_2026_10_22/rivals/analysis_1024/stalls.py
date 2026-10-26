# warp stall reasons summed over all SASS lines (ncu source page csv, columns "stall_*" style names)
import csv, sys
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    i = next(k for k, r in enumerate(rows) if r and r[0] == "Address"); h = rows[i]
    ex = h.index("Instructions Executed")
    body = [r for r in rows[i + 1:] if len(r) > ex and r[ex].strip().isdigit()]
    cols = [k for k, n in enumerate(h) if n.startswith("stall_") or "Warp Stall Sampling (All Samples) " in n or n.startswith("Stall")]
    if not cols:  # fall back: list headers that look like stall reasons
        cols = [k for k, n in enumerate(h) if "stall" in n.lower() and "Sampling" not in n]
    tot = {h[k]: sum(float(r[k] or 0) for r in body if r[k].replace('.', '', 1).isdigit()) for k in cols}
    s = sum(tot.values()) or 1
    print("==", path.split("/")[-1])
    for n, v in sorted(tot.items(), key=lambda x: -x[1])[:8]:
        print(f"   {n:45s} {v:8.0f} ({100*v/s:4.1f}%)")
