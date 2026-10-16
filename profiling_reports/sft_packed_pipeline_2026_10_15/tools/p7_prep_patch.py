#!/usr/bin/env python3
"""P7 (2026-10-15): prepare_sft_data.py --pack ffd writes the chats of every row as a LIST of lengths (TRL's
seq_lengths; the cu_seqlens way) instead of one seg_start value per token. --pack none (v1) is unchanged.
Every replacement must match exactly once. usage: p7_prep_patch.py <BluTrain_sft dir>"""
import sys, os
p = os.path.join(sys.argv[1], "sft/prepare_sft_data.py")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

rep("""  seg_start.bin     uint16 [N, T]  --pack ffd only (v2 packed rows): where the chat of each token starts in its row;
                                   the row's pad tail is a segment of its own. RoPE positions = t - seg_start[t].""",
"""  seq_lengths.bin   int32  [sum]   --pack ffd only (v2 packed rows): the lengths of the chats of every row, row after
                                   row, in the order they sit in the row (TRL's seq_lengths column, trl/data_utils.py).
  seq_offsets.bin   int64  [N + 1] --pack ffd only: row r's chats are seq_lengths[seq_offsets[r] : seq_offsets[r+1]].
                                   The rest of a row (T - sum of its chats) is its pad tail. The trainer turns the list
                                   into the attention kernels' per-chat plan (each chat attends only to itself) and
                                   into RoPE positions that restart at 0 for every chat (as TRL's position_ids).""")
rep("""def build_packed_row(chats, T):
    \"\"\"chats: list of (ids, mask) packed into one row. Each chat is labelled on its own (inputs = ids[:-1],
    labels = ids[1:] where trained), so the last target of a chat is its own end token and nothing points into the
    next chat. Returns inputs, labels, seg_start, trained-token count.\"\"\"
    inp = np.full(T, PAD_ID, dtype=np.uint16)
    lab = np.full(T, IGNORE_ID, dtype=np.uint16)
    seg = np.empty(T, dtype=np.uint16)
    t, nvalid = 0, 0
    for ids, mask in chats:
        ci, cl, cv = build_row(ids, mask, len(ids) - 1)     # the chat alone, exactly its length (no padding)
        n = len(ids) - 1
        inp[t:t + n], lab[t:t + n], seg[t:t + n] = ci, cl, t
        t += n; nvalid += cv
    assert t <= T, "row overflow"
    seg[t:] = t                                             # the pad tail is a segment of its own (if any)
    return inp, lab, seg, nvalid""",
"""def build_packed_row(chats, T):
    \"\"\"chats: list of (ids, mask) packed into one row. Each chat is labelled on its own (inputs = ids[:-1],
    labels = ids[1:] where trained), so the last target of a chat is its own end token and nothing points into the
    next chat. Returns inputs, labels, the chats' lengths in row order, trained-token count.\"\"\"
    inp = np.full(T, PAD_ID, dtype=np.uint16)
    lab = np.full(T, IGNORE_ID, dtype=np.uint16)
    lens = []
    t, nvalid = 0, 0
    for ids, mask in chats:
        ci, cl, cv = build_row(ids, mask, len(ids) - 1)     # the chat alone, exactly its length (no padding)
        n = len(ids) - 1
        inp[t:t + n], lab[t:t + n] = ci, cl
        lens.append(n)
        t += n; nvalid += cv
    assert t <= T, "row overflow"
    return inp, lab, lens, nvalid                           # positions t.. T-1 (if any) are the row's pad tail""")
rep("""    pin, plab, pseg, pv = build_packed_row([(ids, mask), (ids2, mask2)], 128)
    n1, n2 = len(ids) - 1, len(ids2) - 1
    assert list(pseg[:n1]) == [0] * n1 and list(pseg[n1:n1 + n2]) == [n1] * n2 and (pseg[n1 + n2:] == n1 + n2).all()""",
"""    pin, plab, plens, pv = build_packed_row([(ids, mask), (ids2, mask2)], 128)
    n1, n2 = len(ids) - 1, len(ids2) - 1
    assert plens == [n1, n2] and (pin[n1 + n2:] == PAD_ID).all() and (plab[n1 + n2:] == IGNORE_ID).all()""")
rep("""    seg = None
    if a.pack == "none":                        # v1: one chat per row, in the sampled (random) order""",
"""    seq_lengths = None
    if a.pack == "none":                        # v1: one chat per row, in the sampled (random) order""")
rep("""        seg = np.empty((N, T), dtype=np.uint16)
        valid = np.empty(N, dtype=np.uint32)
        for r, idxs in enumerate(row_chats):
            inputs[r], labels[r], seg[r], valid[r] = build_packed_row([chats[i] for i in idxs], T)
        # packing checks: every chat exactly once, rows never overflow, and seg_start follows the kernel contract
        # (AttentionKernels.h): seg[0] = 0, seg[t] <= t, and seg[t] is t (a segment starts) or seg[t-1]
        flat = sorted(i for idxs in row_chats for i in idxs)
        assert flat == list(range(C)), "a chat is missing or packed twice"
        assert all(sum(lengths[i] for i in idxs) <= T for idxs in row_chats), "a row overflows"
        tt = np.arange(T)
        assert (seg[:, 0] == 0).all() and (seg <= tt).all(), "seg_start > t"
        assert ((seg[:, 1:] == tt[1:]) | (seg[:, 1:] == seg[:, :-1])).all(), "seg_start breaks the contract"
        n_starts = int((seg == tt).sum())       # chat starts + pad-tail starts
        pad_tails = int(sum(1 for idxs in row_chats if sum(lengths[i] for i in idxs) < T))
        assert n_starts == C + pad_tails, "segment count != chats + pad tails\"""",
"""        valid = np.empty(N, dtype=np.uint32)
        row_lens = []
        for r, idxs in enumerate(row_chats):
            inputs[r], labels[r], lens_r, valid[r] = build_packed_row([chats[i] for i in idxs], T)
            row_lens.append(lens_r)
        seq_lengths = np.asarray([n for lr in row_lens for n in lr], dtype=np.int32)
        seq_offsets = np.zeros(N + 1, dtype=np.int64)
        seq_offsets[1:] = np.cumsum([len(lr) for lr in row_lens])
        # packing checks: every chat exactly once, rows never overflow, the lengths are the chats' own lengths in
        # row order, every chat's last position predicts its own end token (so the stored boundaries are the real
        # ones), and the rest of every row is padding that trains nothing
        flat = sorted(i for idxs in row_chats for i in idxs)
        assert flat == list(range(C)), "a chat is missing or packed twice"
        assert all(sum(lengths[i] for i in idxs) <= T for idxs in row_chats), "a row overflows"
        assert len(seq_lengths) == C and seq_offsets[-1] == C and (seq_lengths >= 1).all(), "seq_lengths count"
        for r, idxs in enumerate(row_chats):
            lr = seq_lengths[seq_offsets[r]:seq_offsets[r + 1]]
            assert list(lr) == [lengths[i] for i in idxs], f"row {r}: seq_lengths != the chats' lengths"
            ends = np.cumsum(lr)
            assert (labels[r, ends - 1] == PAD_ID).all(), f"row {r}: a chat's last target is not its end token"
            assert (inputs[r, ends[-1]:] == PAD_ID).all() and (labels[r, ends[-1]:] == IGNORE_ID).all(), \\
                f"row {r}: the pad tail is not padding\"""")
rep("""    if seg is not None:
        seg.tofile(os.path.join(a.out, "seg_start.bin"))""",
"""    if seq_lengths is not None:
        seq_lengths.tofile(os.path.join(a.out, "seq_lengths.bin"))
        seq_offsets.tofile(os.path.join(a.out, "seq_offsets.bin"))""")
rep("""            "seg_start": "uint16 [N,T]: start of the chat of token t in its row; pad tail = its own segment",
            "positions": "RoPE position of token t = t - seg_start[t]",""",
"""            "seq_lengths": "int32: every row's chat lengths in row order (TRL's seq_lengths); row r = "
                           "seq_lengths[seq_offsets[r]:seq_offsets[r+1]]; the rest of a row is its pad tail",
            "seq_offsets": "int64 [N+1]",
            "positions": "RoPE position = index of the token inside its chat (restarts at 0 for every chat)",""")
rep("""    if seg is not None:
        print(f"packed (First-Fit-Decreasing): {C:,} chats -> {N:,} rows (minimum possible "
              f"{meta['packing']['min_rows_possible']:,}), real tokens {100 * meta['packing']['real_token_share']:.1f}% "
              f"of the slots, chats per row mean {meta['packing']['chats_per_row']['mean']} max "
              f"{meta['packing']['chats_per_row']['max']}; seg_start checks OK")""",
"""    if seq_lengths is not None:
        print(f"packed (First-Fit-Decreasing): {C:,} chats -> {N:,} rows (minimum possible "
              f"{meta['packing']['min_rows_possible']:,}), real tokens {100 * meta['packing']['real_token_share']:.1f}% "
              f"of the slots, chats per row mean {meta['packing']['chats_per_row']['mean']} max "
              f"{meta['packing']['chats_per_row']['max']}; chat-length checks OK")""")
rep("""        tag = kept_ids[k] if seg is None else f"{len(row_chats[k])} chats\"""",
    """        tag = kept_ids[k] if seq_lengths is None else f"{len(row_chats[k])} chats\"""")
rep("""          + (", seg_start.bin" if seg is not None else ""))""",
    """          + (", seq_lengths.bin + seq_offsets.bin" if seq_lengths is not None else ""))""")
rep("""                    help="none = v1 (one chat per row); ffd = v2 packed rows (First-Fit-Decreasing) + seg_start.bin")""",
    """                    help="none = v1 (one chat per row); ffd = v2 packed rows (First-Fit-Decreasing) + seq_lengths.bin")""")
assert "seg_start" not in s and "seg[" not in s and "seg " not in s, [l for l in s.splitlines() if "seg" in l]
open(p, "w", encoding="utf-8").write(s)
print("patched prepare_sft_data.py")
