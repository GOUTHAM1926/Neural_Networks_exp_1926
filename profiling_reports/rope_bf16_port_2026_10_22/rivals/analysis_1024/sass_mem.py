# memory instructions of a kernel's SASS (ncu source page csv): opcode, times executed (per warp), stall samples
import csv, sys, re, collections
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))
    i = next(k for k, r in enumerate(rows) if r and r[0] == "Address"); h = rows[i]
    src, ex, smp = h.index("Source"), h.index("Instructions Executed"), h.index("Warp Stall Sampling (All Samples)")
    body = [r for r in rows[i + 1:] if len(r) > ex and r[ex].strip().isdigit()]
    tot_ex = sum(int(r[ex] or 0) for r in body); tot_s = sum(int(r[smp] or 0) for r in body)
    agg = collections.Counter(); samp = collections.Counter()
    for r in body:
        op = r[src].strip().split()[0] if r[src].strip() else ""
        if op.startswith("@"): op = r[src].strip().split()[1]
        if re.match(r"(LDG|STG|LDS|STS|LDC|SHFL|BAR|RED|ATOM|LDGSTS|LDSM|MUFU)", op):
            agg[op] += int(r[ex] or 0); samp[op] += int(r[smp] or 0)
    print(f"== {path.split('/')[-1]}: {len(body)} SASS lines, {tot_ex:,} warp-instructions executed, {tot_s:,} stall samples")
    for op, n in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"   {op:22s} executed {n:>10,}   stall samples {samp[op]:>6,} ({100*samp[op]/max(tot_s,1):4.1f}%)")
