#!/usr/bin/env python3
# build_samesteps.py (2026-10-21): packed SFT data whose optimizer step s holds EXACTLY the 64 chats that step s of the
# padded run (run_all_sftb.sh phase C, B 8 x 8 micro-batches, 156 steps) trained on, in the same order of steps, so a packed
# run can be compared with that padded run step by step (same chats, same trained tokens, same LR at every step).
#   train: 156 groups x R rows (R = the smallest multiple of 4 that holds every group); a group's chats go into its R rows
#          by worst-fit decreasing (the longest chats first, each into the row with the most free space), so every row
#          gets at least one chat (the loader refuses a row without a trained token: SftDataLoader.h "no trained token").
#          Each chat's inputs and labels are copied from the existing packed file (chats labelled alone), found by the
#          hash of its tokens. One epoch of this file = the 156 steps (run it with BLU_SFT_EPOCHS=1).
#   val:   padded val rows 0-495 = exactly the rows the padded run validated on (B 8 dropped rows 496-499); dense rows.
#   checks (before writing): the loader's file rules, every group == the padded step's chats (token hashes), and the trained
#          tokens of every group == the padded run's trained_tokens column.
# usage: PYTHONPATH=<numpy> python3 build_samesteps.py <sft_data folder> <padded run folder with sft_adamw.csv> <out folder>
import csv, hashlib, json, os, sys
import numpy as np

T, PAD, IGN = 4096, 50256, 65535
D, RUN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P, K = f"{D}/stage0_ultrachat_5k_T4096", f"{D}/stage0_ultrachat_5k_T4096_packed"
h = lambda a: hashlib.md5(a.tobytes()).hexdigest()
rd = lambda p, dt: np.fromfile(p, dt)

pi = rd(f"{P}/train/inputs.bin", np.uint16).reshape(-1, T); pl = rd(f"{P}/train/labels.bin", np.uint16).reshape(-1, T)
ki = rd(f"{K}/train/inputs.bin", np.uint16).reshape(-1, T); kl = rd(f"{K}/train/labels.bin", np.uint16).reshape(-1, T)
lens = rd(f"{K}/train/seq_lengths.bin", np.int32); offs = rd(f"{K}/train/seq_offsets.bin", np.int64)
seg = {}
for r in range(len(ki)):
    t = 0
    for c in range(offs[r], offs[r + 1]):
        L = int(lens[c]); seg[h(ki[r, t:t + L])] = (r, t, L); t += L
lengths = sorted({v[2] for v in seg.values()})
chat = []   # padded row (= chat) -> (packed row, start, length) of the same chat in the packed file
for r in range(len(pi)):
    lo = int(np.nonzero(pl[r] != IGN)[0][-1]) + 1
    chat.append(next(seg[h(pi[r, :L])] for L in lengths if L >= lo and h(pi[r, :L]) in seg))
assert len(chat) == 5000

per_epoch, per_step, B, NSTEP = len(pi) // 8, 8, 8, 156          # the padded run's stream (no shuffle)
steps = [[(m % per_epoch) * B + j for m in range(s * per_step, (s + 1) * per_step) for j in range(B)] for s in range(NSTEP)]
want = [int(x["trained_tokens"]) for x in csv.DictReader(open(f"{RUN}/sft_adamw.csv"))]
assert len(want) == NSTEP


def place(group, R):   # worst-fit decreasing; None if a chat does not fit
    rows, free = [[] for _ in range(R)], [T] * R
    for c in sorted(group, key=lambda c: -chat[c][2]):
        i = int(np.argmax(free))
        if free[i] < chat[c][2]: return None
        rows[i].append(c); free[i] -= chat[c][2]
    return rows if all(rows) else None


R = 4
while any(place(g, R) is None for g in steps): R += 4
print(f"rows per step R = {R} (multiple of 4 for B 4); total rows {NSTEP * R}")
N = NSTEP * R
inp = np.full((N, T), PAD, np.uint16); lab = np.full((N, T), IGN, np.uint16)
slen, soff = [], [0]
for s, g in enumerate(steps):
    for k, row in enumerate(place(g, R)):
        rr, t = s * R + k, 0
        for c in row:
            r0, t0, L = chat[c]
            inp[rr, t:t + L] = ki[r0, t0:t0 + L]; lab[rr, t:t + L] = kl[r0, t0:t0 + L]
            slen.append(L); t += L
        soff.append(len(slen))
cnt = (lab != IGN).sum(1).astype(np.uint32)
# ---- checks -----------------------------------------------------------------------------------------------------
slen_a, soff_a = np.array(slen, np.int32), np.array(soff, np.int64)
assert (cnt > 0).all(), "a row without a trained token"
assert (np.diff(soff_a) >= 1).all() and soff_a[-1] == len(slen_a)
for r in range(N):
    assert slen_a[soff_a[r]:soff_a[r + 1]].sum() <= T
got = [int(cnt[s * R:(s + 1) * R].sum()) for s in range(NSTEP)]
assert got == want, "trained tokens per step differ from the padded run"
for s, g in enumerate(steps):   # the chats of every group, by token hash, == the padded step's chats
    mine, t = [], None
    for r in range(s * R, (s + 1) * R):
        t = 0
        for c in range(soff_a[r], soff_a[r + 1]):
            L = int(slen_a[c]); mine.append(h(inp[r, t:t + L])); t += L
    theirs = [h(ki[chat[c][0], chat[c][1]:chat[c][1] + chat[c][2]]) for c in g]
    assert sorted(mine) == sorted(theirs), f"step {s}: different chats"
print(f"checks OK: every row has a trained token, every group == the padded step's 64 chats, trained tokens per step == the "
      f"padded run's ({min(got):,}-{max(got):,}); real-token share {slen_a.sum() / (N * T):.3f}")
# ---- write --------------------------------------------------------------------------------------------------------
vi = rd(f"{P}/val/inputs.bin", np.uint16).reshape(-1, T)[:496]; vl = rd(f"{P}/val/labels.bin", np.uint16).reshape(-1, T)[:496]
for split in ("train", "val"):
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
inp.tofile(f"{OUT}/train/inputs.bin"); lab.tofile(f"{OUT}/train/labels.bin"); cnt.tofile(f"{OUT}/train/valid_counts.bin")
slen_a.tofile(f"{OUT}/train/seq_lengths.bin"); soff_a.tofile(f"{OUT}/train/seq_offsets.bin")
vi.tofile(f"{OUT}/val/inputs.bin"); vl.tofile(f"{OUT}/val/labels.bin"); (vl != IGN).sum(1).astype(np.uint32).tofile(f"{OUT}/val/valid_counts.bin")
note = {"made_by": "build_samesteps.py 2026-10-21", "train_rule": f"step s = the 64 chats of padded step s, {R} rows, worst-fit decreasing; 1 epoch = 156 steps",
        "val_rule": "padded val rows 0-495 (what the padded B 8 run validated on)", "pad_id": PAD, "ignore_id": IGN}
json.dump({"N": N, "T": T, "packed": True, **note}, open(f"{OUT}/train/meta.json", "w"), indent=1)
json.dump({"N": 496, "T": T, "packed": False, **note}, open(f"{OUT}/val/meta.json", "w"), indent=1)
print(f"wrote {OUT}: train {N} rows ({len(slen_a)} chats), val 496 rows")
