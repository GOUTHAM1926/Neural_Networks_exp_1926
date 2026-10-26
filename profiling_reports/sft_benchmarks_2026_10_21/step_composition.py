#!/usr/bin/env python3
# step_composition.py (2026-10-21): which chats each optimizer step of the two SFT runs (phase C) trained on, and which
# val chats each run validated on. Explains why the per-step train loss of padding and packing differs.
#   chat identity: every padded chat is found in the packed file by the hash of its tokens (all 5,000 must match)
#   steps: the loader never shuffles; the trainer takes a continuous stream of micro-batches over the epochs
#          (padding: B 8, 8 per step, 625 per epoch; packing: B 4, 5 per step, 379 per epoch). The reconstruction is
#          checked against the trainer's own trained_tokens column of every step.
#   val: the loader drops the last incomplete micro-batch (padding B 8: 4 of 500 rows; packing B 4: 3 of 147 rows)
# usage: PYTHONPATH=<numpy> python3 step_composition.py <sft_data folder> <runs folder>
import csv, hashlib, sys
import numpy as np

T = 4096
D, R = sys.argv[1], sys.argv[2]
P, K = f"{D}/stage0_ultrachat_5k_T4096", f"{D}/stage0_ultrachat_5k_T4096_packed"
h = lambda a: hashlib.md5(a.tobytes()).hexdigest()


def load(d, split):
    i = np.fromfile(f"{d}/{split}/inputs.bin", np.uint16).reshape(-1, T)
    l = np.fromfile(f"{d}/{split}/labels.bin", np.uint16).reshape(-1, T)
    return i, l


def packed_row_of(split):
    """packed row that holds each padded chat (padded row r = chat r)"""
    pi, pl = load(P, split); ki, _ = load(K, split)
    lens = np.fromfile(f"{K}/{split}/seq_lengths.bin", np.int32)
    offs = np.fromfile(f"{K}/{split}/seq_offsets.bin", np.int64)
    seg = {}
    for r in range(len(ki)):
        t = 0
        for c in range(offs[r], offs[r + 1]):
            L = int(lens[c]); seg[h(ki[r, t:t + L])] = (r, L); t += L
    lengths = sorted({L for _, L in seg.values()})
    out = []
    for r in range(len(pi)):
        lo = int(np.nonzero(pl[r] != 65535)[0][-1]) + 1
        out.append(next((seg[h(pi[r, :L])][0] for L in lengths if L >= lo and h(pi[r, :L]) in seg), None))
    return out, offs


def steps(nrows, B, per_step, nsteps):
    per_epoch = nrows // B
    return [[r for m in range(s * per_step, (s + 1) * per_step) for r in range((m % per_epoch) * B, (m % per_epoch) * B + B)]
            for s in range(nsteps)]


row_of, offs = packed_row_of("train")
print(f"train: {sum(x is not None for x in row_of)} of {len(row_of)} padded chats found in the packed file")
_, pl = load(P, "train"); _, kl = load(K, "train")
tp = (pl != 65535).sum(1); tk = (kl != 65535).sum(1)
SP, SK = steps(len(pl), 8, 8, 156), steps(len(kl), 4, 5, 151)
cp = [int(x["trained_tokens"]) for x in csv.DictReader(open(f"{R}/C_P_padded/sft_adamw.csv"))]
ck = [int(x["trained_tokens"]) for x in csv.DictReader(open(f"{R}/C_K_packed/sft_adamw.csv"))]
print("step reconstruction == trainer's trained_tokens column:", [int(tp[s].sum()) for s in SP] == cp,
      [int(tk[s].sum()) for s in SK] == ck)
nk = [int(sum(offs[r + 1] - offs[r] for r in s)) for s in SK]
print(f"chats per step: padding 64 always; packing {min(nk)}-{max(nk)} (median {int(np.median(nk))})")
print(f"trained tokens per step: padding {min(cp):,}-{max(cp):,}; packing {min(ck):,}-{max(ck):,}")
ov = [sum(1 for r in SP[s] if row_of[r] in set(SK[s])) for s in range(151)]
print(f"chats shared by padded step s and packed step s: mean {np.mean(ov):.2f} of 64, max {max(ov)}")
lp = [float(x["train_loss"]) for x in csv.DictReader(open(f"{R}/C_P_padded/sft_adamw.csv"))]
lk = [float(x["train_loss"]) for x in csv.DictReader(open(f"{R}/C_K_packed/sft_adamw.csv"))]
print(f"step 0 (same weights): padding train {lp[0]:.6f} on {cp[0]:,} trained tokens; packing {lk[0]:.6f} on {ck[0]:,}")
print(f"step-to-step change of the train loss (std, steps 10+): padding {np.std(np.diff(lp[10:])):.4f}, "
      f"packing {np.std(np.diff(lk[10:])):.4f}")
print(f"last step: padding {lp[-1]:.6f}, packing {lk[-1]:.6f}; epoch means padding {np.mean(lp[:78]):.4f} / "
      f"{np.mean(lp[78:]):.4f}, packing {np.mean(lk[:75]):.4f} / {np.mean(lk[75:]):.4f}")
vrow, _ = packed_row_of("val")
up = set(range(496)); uk = {r for r in range(len(vrow)) if vrow[r] < 144}
print(f"val: padded run uses {len(up)} chats, packed run {len(uk)}, both {len(up & uk)}, only padded {len(up - uk)}, "
      f"only packed {len(uk - up)}")
