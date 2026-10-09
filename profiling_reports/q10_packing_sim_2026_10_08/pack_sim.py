"""v1 (pad every chat to 4096) vs v1.5 (length buckets) vs v2 (packing) on the real stage-0 chats.
Work model per micro-batch (our 1.5B config): linear+lm_head work ~ tokens computed (incl. padding);
causal attention work ~ sum over attended blocks of L^2/2. Ratios only; same MFU assumed.
Usage: python pack_sim.py <stage0 dir>/train"""
import json, sys, numpy as np
d = sys.argv[1]
meta = json.load(open(d + "/meta.json"))
T = int(meta.get("T", 4096))
lab = np.fromfile(d + "/labels.bin", dtype=np.uint16).reshape(-1, T)
N = lab.shape[0]
last = np.array([np.nonzero(r != 65535)[0][-1] + 1 for r in lab])      # real length (inputs) of each chat
L = last.astype(np.int64)
d_model, layers, V = 2304, 26, 50304
lin_per_tok = 6 * (1.4722e9 + V * d_model)                 # fwd+bwd FLOPs per token (layers + lm_head)
attn_per_L2 = 6 * 2 * 2304 * layers / 2                    # causal attention, fwd+bwd, per L^2

def cost(rows):            # rows: list of (computed_length, [chat lengths inside], block_diagonal?)
    lin = sum(r[0] for r in rows) * lin_per_tok
    attn = sum(sum(l * l for l in r[1]) if r[2] else r[0] ** 2 for r in rows) * attn_per_L2
    return lin, attn

real = int(L.sum())
B = 8
v1 = [(T, [int(x)], False) for x in L]                                   # v1: every chat padded to T
s = np.sort(L)[::-1]
v15 = []                                                                 # v1.5 ideal: whole set sorted
for i in range(0, N, B):
    g = s[i:i + B]
    t = int(-(-g.max() // 256) * 256)
    v15 += [(t, [int(x)], False) for x in g]
bins = []                                                                # v2: best-fit-decreasing packing
for x in s:
    for b in bins:
        if b[0] + x <= T:
            b[0] += int(x); b[1].append(int(x)); break
    else:
        bins.append([int(x), [int(x)]])
v2 = [(T, b[1], True) for b in bins]
out = {}
for name, rows in (("v1 pad to 4096", v1), ("v1.5 buckets of 8, pad to max (x256)", v15), ("v2 packing, block-diagonal mask", v2)):
    lin, attn = cost(rows)
    computed = sum(r[0] for r in rows)
    out[name] = dict(rows=len(rows), tokens_computed=computed, useful=real / computed, work=lin + attn)
base = out["v1 pad to 4096"]["work"]
print(f"chats {N}, real tokens {real:,}, mean length {L.mean():.0f}, median {np.median(L):.0f}, p90 {np.percentile(L,90):.0f}, max {L.max()}")
for k, v in out.items():
    print(f"{k:40s} rows {v['rows']:5d}  tokens computed {v['tokens_computed']:>11,}  useful {100*v['useful']:5.1f}%  "
          f"total work {v['work']/base*100:5.1f}% of v1  -> speed-up x{base/v['work']:.2f}")
rng = np.random.default_rng(1234)                                        # v1.5 realistic: sort inside each 64-chat step
p = rng.permutation(L)
v15r = []
for i in range(0, N, 64):
    st = np.sort(p[i:i + 64])[::-1]
    for j in range(0, len(st), B):
        g = st[j:j + B]
        t = int(-(-g.max() // 256) * 256)
        v15r += [(t, [int(x)], False) for x in g]
lin, attn = cost(v15r); computed = sum(r[0] for r in v15r)
print(f"{'v1.5 realistic (sort within each 64-chat step)':40s} rows {len(v15r):5d}  tokens computed {computed:>11,}  useful {100*real/computed:5.1f}%  "
      f"total work {(lin+attn)/base*100:5.1f}% of v1  -> speed-up x{base/(lin+attn):.2f}")
lin, attn = cost([(r[0], r[1], False) for r in v2])                      # v2 without the mask (wrong)
print(f"{'v2 WITHOUT mask (chats see each other)':40s} total work {(lin+attn)/base*100:5.1f}% of v1  -> speed-up x{base/(lin+attn):.2f}")
print(f"v2 rows {len(bins)}, mean pad per row {(len(bins)*T - real)/len(bins):.1f}, chats per row {min(len(b[1]) for b in bins)}-{max(len(b[1]) for b in bins)}")
