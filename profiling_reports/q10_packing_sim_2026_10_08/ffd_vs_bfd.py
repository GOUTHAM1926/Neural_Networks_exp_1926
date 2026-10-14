# FFD (our pack_sim.py / test harness rule) vs BFD (TRL's rule) on the real stage-0 chats. Laptop only.
import json, sys, numpy as np
d = sys.argv[1]; T = int(json.load(open(d + "/meta.json")).get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
L = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab], dtype=np.int64)
s = sorted(L.tolist(), reverse=True)
def ffd(s):
    bins = []
    for x in s:
        for b in bins:
            if b[0] + x <= T: b[0] += x; b[1].append(x); break
        else: bins.append([x, [x]])
    return bins
def bfd(s):
    bins = []
    for x in s:
        best = None
        for b in bins:
            if b[0] + x <= T and (best is None or T - b[0] < T - best[0]): best = b
        if best is None: bins.append([x, [x]])
        else: best[0] += x; best[1].append(x)
    return bins
real = int(L.sum())
for name, bins in (("first-fit decreasing (ours)", ffd(s)), ("best-fit decreasing (TRL)", bfd(s))):
    pad = [T - b[0] for b in bins]; k = [len(b[1]) for b in bins]
    print(f"{name:28s} rows {len(bins):5d}  real tokens {100*real/(len(bins)*T):5.2f}%  mean pad/row {np.mean(pad):5.1f}  "
          f"chats/row {min(k)}-{max(k)}  rows with pad>256: {sum(p > 256 for p in pad)}")
print(f"chats {len(L)}, real tokens {real:,}, lower bound rows = ceil(real/T) = {-(-real // T)}")
