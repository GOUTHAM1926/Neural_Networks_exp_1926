#!/usr/bin/env python3
"""Turn a Hugging Face chat dataset into BluTrain SFT files (SFT report sections 3.2-3.10).

Output directory:
  inputs.bin        uint16 [N, T]  token ids the model reads, padded with 50256 (<|endoftext|>)
  labels.bin        uint16 [N, T]  token to predict at each position; 65535 = ignore (prompt, padding)
  valid_counts.bin  uint32 [N]     number of non-ignored labels in each row (for the global denominator)
  seq_lengths.bin   int32  [sum]   --pack ffd only (v2 packed rows): the lengths of the chats of every row, row after
                                   row, in the order they sit in the row (TRL's seq_lengths column, trl/data_utils.py).
  seq_offsets.bin   int64  [N + 1] --pack ffd only: row r's chats are seq_lengths[seq_offsets[r] : seq_offsets[r+1]].
                                   The rest of a row (T - sum of its chats) is its pad tail. The trainer turns the list
                                   into the attention kernels' per-chat plan (each chat attends only to itself) and
                                   into RoPE positions that restart at 0 for every chat (as TRL's position_ids).
  meta.json         settings, dataset revision, tool versions, counts, drop reasons, length histogram

Template "tulu-plain-v1" (option A: role markers are plain text, each answer ends with the real end token):
  <|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...
Each piece is tokenized on its own, so a marker never merges with the text next to it (BPE).
Only answer tokens and the end token after each answer are targets; every assistant turn is trained.
Text that looks like a special token (e.g. "<|endoftext|>" inside a user message) is encoded as plain text.
Default data: HuggingFaceH4/ultrachat_200k (English-only, MIT), so no language filter is needed.

Rows are labelled the usual way: inputs = ids[:-1], labels = ids[1:] with prompt positions set to 65535.
--pack none (default, v1): one chat per row, padded. --pack ffd (v2): every chat is labelled on its own first (so no
target ever crosses into the next chat), then whole chats are packed into rows with First-Fit-Decreasing (the rule
chosen in work_docs/sft_packing_algorithms_bfd_ffd.md section 13), and the rows are shuffled. Chats longer than T are
dropped in both modes (splitting them like TRL's bfd_split is a future task, SFT work doc section 45.4).
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

PAD_ID = 50256          # input padding = <|endoftext|>, a real GPT-2 id the embedding handles (report 3.8)
IGNORE_ID = 65535       # label marker: no loss, no gradient (report 3.5)
VOCAB_ROWS = 50304      # model vocab rows (50,257 GPT-2 ids padded to 50,304)
TEMPLATE = "tulu-plain-v1"
ROLES = ("system", "user", "assistant")


def tokenize_conversation(messages, enc):
    """Return (ids, train_mask) for one conversation, or (None, reason) if the row is unusable."""
    if not messages:
        return None, "empty_row"
    if any(m.get("role") not in ROLES for m in messages):
        return None, "unknown_role"
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant_turn"
    if messages[-1]["role"] != "assistant":
        return None, "last_turn_not_assistant"
    ids, mask = [], []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "assistant" and not content.strip():
            return None, "empty_answer"
        marker = enc.encode_ordinary(f"<|{role}|>\n")
        ids += marker
        mask += [False] * len(marker)
        body = enc.encode_ordinary(content)
        trained = role == "assistant"
        ids += body
        mask += [trained] * len(body)
        if trained:
            ids.append(PAD_ID)          # the end token is a target: the model must learn to stop
            mask.append(True)
        else:
            nl = enc.encode_ordinary("\n")
            ids += nl
            mask += [False] * len(nl)
    return (ids, mask), None


def build_row(ids, mask, T):
    """inputs = ids[:-1]; labels = ids[1:] where the next token is trained, else IGNORE. Padded to T."""
    n = len(ids) - 1
    inp = np.full(T, PAD_ID, dtype=np.uint16)
    lab = np.full(T, IGNORE_ID, dtype=np.uint16)
    inp[:n] = ids[:-1]
    nxt = np.asarray(ids[1:], dtype=np.int64)
    keep = np.asarray(mask[1:], dtype=bool)
    lab[:n] = np.where(keep, nxt, IGNORE_ID)
    return inp, lab, int(keep.sum())


def ffd_pack(lengths, T):
    """First-Fit-Decreasing: chats longest first; each goes into the FIRST row that still has room, else a new row.
    A max segment tree over the rows' free space finds that row in O(log rows). Returns a list of rows, each a list
    of chat indices. Same result as the plain loop (checked in self_test)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    size = 1
    while size < max(1, len(lengths)):
        size *= 2
    tree = [0] * (2 * size)          # leaf size + r = free slots of row r (0 = no row yet, or full)
    rows, used = [], 0
    for i in order:
        L = lengths[i]
        if tree[1] >= L:             # some open row has room: walk down to the leftmost one
            node = 1
            while node < size:
                node = 2 * node if tree[2 * node] >= L else 2 * node + 1
            r = node - size
        else:                        # no row has room: open a new one
            r = used; used += 1
            node = size + r
            tree[node] = T
            assert r < size, "more rows than chats (cannot happen)"
            rows.append([])
        rows[r].append(i)
        tree[node] -= L
        node //= 2
        while node:
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
            node //= 2
    return rows


def ffd_pack_plain(lengths, T):
    """The same rule written as the plain loop (used only by self_test to check ffd_pack)."""
    rows, free = [], []
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        for r in range(len(rows)):
            if free[r] >= lengths[i]:
                rows[r].append(i); free[r] -= lengths[i]
                break
        else:
            rows.append([i]); free.append(T - lengths[i])
    return rows


def build_packed_row(chats, T):
    """chats: list of (ids, mask) packed into one row. Each chat is labelled on its own (inputs = ids[:-1],
    labels = ids[1:] where trained), so the last target of a chat is its own end token and nothing points into the
    next chat. Returns inputs, labels, the chats' lengths in row order, trained-token count."""
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
    return inp, lab, lens, nvalid                           # positions t.. T-1 (if any) are the row's pad tail


def show_row(enc, inp, lab):
    """Decode one row; the trained part (what the labels ask for) is wrapped in [[ ]].
    End tokens of earlier answers are also INPUTS, so padding is not "the first 50256": the row's real
    length is fixed by its last target (every row ends with the trained end token)."""
    n = int(np.nonzero(lab != IGNORE_ID)[0][-1]) + 1
    ids = [int(inp[0])] + [int(lab[t]) if lab[t] != IGNORE_ID else int(inp[t + 1]) for t in range(n)]
    trained = [False] + [bool(lab[t] != IGNORE_ID) for t in range(n)]
    out = []
    for is_trained, grp in itertools.groupby(zip(ids, trained), key=lambda x: x[1]):
        toks = [t for t, _ in grp]
        text, run = [], []
        for t in toks + [None]:                      # decode runs between end tokens in one go (keeps UTF-8 whole)
            if t is None or t == PAD_ID:
                if run:
                    text.append(enc.decode(run)); run = []
                if t == PAD_ID:
                    text.append("<|endoftext|>")
            else:
                run.append(t)
        out.append("[[" + "".join(text) + "]]" if is_trained else "".join(text))
    return "".join(out)


def self_test(enc):
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye <|endoftext|>"}, {"role": "assistant", "content": "Bye!"}]
    (ids, mask), err = tokenize_conversation(msgs, enc)
    assert err is None
    inp, lab, nvalid = build_row(ids, mask, 64)
    trained = [int(t) for t in lab if t != IGNORE_ID]
    expect = enc.encode_ordinary("Hello.") + [PAD_ID] + enc.encode_ordinary("Bye!") + [PAD_ID]
    assert trained == expect, (trained, expect)
    assert nvalid == len(expect)
    assert enc.decode(enc.encode_ordinary("<|endoftext|>")) == "<|endoftext|>"          # stays plain text
    assert PAD_ID not in enc.encode_ordinary("Bye <|endoftext|>")                       # never the real id
    assert show_row(enc, inp, lab).count("[[") == 2
    print("self-test OK:", show_row(enc, inp, lab).replace("\n", "\\n"))
    # packing: the tree version gives the same rows as the plain First-Fit-Decreasing loop
    rng = np.random.default_rng(7)
    for trial in range(300):
        T_ = int(rng.choice([8, 20, 64, 4096]))
        lens = [int(x) for x in rng.integers(1, T_ + 1, size=int(rng.integers(1, 60)))]
        assert ffd_pack(lens, T_) == ffd_pack_plain(lens, T_), (T_, lens)
    # a packed row: chats back to back, each labelled alone, pad tail = its own segment
    (ids2, mask2), _ = tokenize_conversation(msgs[:2], enc)
    pin, plab, plens, pv = build_packed_row([(ids, mask), (ids2, mask2)], 128)
    n1, n2 = len(ids) - 1, len(ids2) - 1
    assert plens == [n1, n2] and (pin[n1 + n2:] == PAD_ID).all() and (plab[n1 + n2:] == IGNORE_ID).all()
    assert plab[n1 - 1] == PAD_ID and plab[n1 + n2 - 1] == PAD_ID, "each chat's last target is its own end token"
    assert pv == nvalid + int((np.asarray(mask2[1:])).sum())
    print(f"packing self-test OK: tree FFD == plain FFD on 300 random sets; packed row of chats {n1}+{n2} + pad")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--revision", default="8049631c405ae6576f93f445c6b8166f76f5505a",
                    help="dataset commit, pinned so the data is reproducible")
    ap.add_argument("--split", default="train_sft", help="train_sft for training, test_sft for validation")
    ap.add_argument("--num", type=int, default=5000, help="rows to keep (random sample)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out", required=False)
    ap.add_argument("--show", type=int, default=2, help="decoded rows to print as a check")
    ap.add_argument("--pack", choices=["none", "ffd"], default="none",
                    help="none = v1 (one chat per row); ffd = v2 packed rows (First-Fit-Decreasing) + seq_lengths.bin")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    assert enc.eot_token == PAD_ID and enc.n_vocab == 50257
    if a.self_test:
        self_test(enc)
        return
    T = a.seq_len
    if T % 256 or T > 4096:
        sys.exit("--seq-len must be a multiple of 256 and <= 4096 (attention kernel + RoPE cache limits)")
    if not a.out:
        sys.exit("--out is required")
    os.makedirs(a.out, exist_ok=True)

    import datasets
    t0 = time.time()
    ds = datasets.load_dataset(a.dataset, split=a.split, revision=a.revision)
    order = np.random.default_rng(a.seed).permutation(len(ds))
    print(f"loaded {a.dataset}@{a.revision[:8]} {a.split}: {len(ds):,} rows in {time.time() - t0:.0f}s; sampling {a.num:,}")

    chats = []                                  # (ids uint16 array, mask bool array) of every kept chat
    lengths, kept_src = [], collections.Counter()
    drops, drop_src = collections.Counter(), collections.Counter()
    kept_ids, scanned = [], 0
    for idx in order:
        if len(kept_ids) == a.num:
            break
        row = ds[int(idx)]
        scanned += 1
        src = row.get("source") or a.dataset
        msgs = row["messages"]
        res, reason = tokenize_conversation(msgs, enc)
        if reason is None and len(res[0]) - 1 > T:
            reason = "too_long"
        if reason:
            drops[reason] += 1; drop_src[(src, reason)] += 1
            continue
        chats.append((np.asarray(res[0], dtype=np.uint16), np.asarray(res[1], dtype=bool)))
        lengths.append(len(res[0]) - 1)
        kept_src[src] += 1
        kept_ids.append(str(row.get("id") or row.get("prompt_id") or int(idx)))
    C = len(kept_ids)                           # chats kept
    if C < a.num:
        print(f"WARNING: only {C:,} usable chats after scanning everything")
    seq_lengths = None
    if a.pack == "none":                        # v1: one chat per row, in the sampled (random) order
        N = C
        inputs = np.empty((N, T), dtype=np.uint16)
        labels = np.empty((N, T), dtype=np.uint16)
        valid = np.empty(N, dtype=np.uint32)
        for k, (ids, mask) in enumerate(chats):
            inputs[k], labels[k], valid[k] = build_row(ids, mask, T)
        row_chats = [[k] for k in range(N)]
    else:                                       # v2: First-Fit-Decreasing, then the rows in random order
        packed = ffd_pack(lengths, T)
        perm = np.random.default_rng(a.seed + 2).permutation(len(packed))
        row_chats = [packed[r] for r in perm]
        N = len(row_chats)
        inputs = np.empty((N, T), dtype=np.uint16)
        labels = np.empty((N, T), dtype=np.uint16)
        valid = np.empty(N, dtype=np.uint32)
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
            assert (inputs[r, ends[-1]:] == PAD_ID).all() and (labels[r, ends[-1]:] == IGNORE_ID).all(), \
                f"row {r}: the pad tail is not padding"

    # ---- checks (fail loudly: a bad file trains silently) ----
    assert inputs.max() < VOCAB_ROWS, "input id >= vocab rows"
    lab_ok = (labels < VOCAB_ROWS) | (labels == IGNORE_ID)
    assert lab_ok.all(), "label outside [0, vocab) and not the ignore marker"
    assert (valid > 0).all(), "a row with zero trained tokens (0/0 = NaN in the loss)"
    for k in range(N):  # the last trained target of every row is the end token
        last = labels[k][labels[k] != IGNORE_ID][-1]
        assert last == PAD_ID, f"row {k}: answer does not end with the end token"
    eos_targets = int((labels == PAD_ID).sum())

    inputs.tofile(os.path.join(a.out, "inputs.bin"))
    labels.tofile(os.path.join(a.out, "labels.bin"))
    valid.tofile(os.path.join(a.out, "valid_counts.bin"))
    if seq_lengths is not None:
        seq_lengths.tofile(os.path.join(a.out, "seq_lengths.bin"))
        seq_offsets.tofile(os.path.join(a.out, "seq_offsets.bin"))
    chats_per_row = np.asarray([len(idxs) for idxs in row_chats])
    lens = np.asarray(lengths)
    hist_edges = [0, 256, 512, 1024, 2048, 3072, 4096]
    meta = {
        "format": {"inputs": "uint16 [N,T] little-endian, row-major", "labels": "uint16 [N,T], 65535 = ignore",
                   "valid_counts": "uint32 [N]", "label_rule": "labels = ids[1:] masked; inputs = ids[:-1]"},
        "N": N, "T": T, "pad_id": PAD_ID, "ignore_id": IGNORE_ID, "vocab_rows": VOCAB_ROWS,
        "packed": a.pack != "none",
        "packing": None if a.pack == "none" else {
            "rule": "first-fit-decreasing (max segment tree), rows shuffled with seed + 2",
            "seq_lengths": "int32: every row's chat lengths in row order (TRL's seq_lengths); row r = "
                           "seq_lengths[seq_offsets[r]:seq_offsets[r+1]]; the rest of a row is its pad tail",
            "seq_offsets": "int64 [N+1]",
            "positions": "RoPE position = index of the token inside its chat (restarts at 0 for every chat)",
            "chats": C, "rows": N, "min_rows_possible": int(-(-sum(lengths) // T)),
            "real_token_share": round(float(sum(lengths)) / (N * T), 4),
            "chats_per_row": {"mean": round(float(chats_per_row.mean()), 2), "min": int(chats_per_row.min()),
                              "max": int(chats_per_row.max())},
            "row_chats": [[kept_ids[i] for i in idxs] for idxs in row_chats]},
        "template": TEMPLATE, "tokenizer": f"tiktoken gpt2 ({tiktoken.__version__})",
        "dataset": a.dataset, "revision": a.revision, "split": a.split, "seed": a.seed,
        "filters": {"max_tokens": T, "language": "none (the dataset is English-only)"},
        "rows_scanned": scanned, "rows_kept": N, "chats_kept": C, "drops": dict(drops),
        "drops_by_source": {f"{s} | {r}": c for (s, r), c in sorted(drop_src.items())},
        "kept_by_source": dict(kept_src.most_common()),
        "trained_tokens": int(valid.sum()), "positions": int(N * T),
        "trained_share_of_positions": round(float(valid.sum()) / (N * T), 4),
        "end_token_targets": eos_targets,
        "length_tokens": {"mean": round(float(lens.mean()), 1), "median": int(np.median(lens)),
                          "p95": int(np.percentile(lens, 95)), "max": int(lens.max()),
                          "histogram": {f"{lo}-{hi}": int(((lens > lo) & (lens <= hi)).sum())
                                        for lo, hi in zip(hist_edges[:-1], hist_edges[1:])}},
        "kept_row_ids": kept_ids,
        "tools": {"python": sys.version.split()[0], "numpy": np.__version__, "datasets": datasets.__version__},
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"kept {C:,} chats of {scanned:,} scanned | drops {dict(drops)}")
    if seq_lengths is not None:
        print(f"packed (First-Fit-Decreasing): {C:,} chats -> {N:,} rows (minimum possible "
              f"{meta['packing']['min_rows_possible']:,}), real tokens {100 * meta['packing']['real_token_share']:.1f}% "
              f"of the slots, chats per row mean {meta['packing']['chats_per_row']['mean']} max "
              f"{meta['packing']['chats_per_row']['max']}; chat-length checks OK")
    print(f"trained tokens {meta['trained_tokens']:,} = {100 * meta['trained_share_of_positions']:.1f}% of "
          f"{N * T:,} positions | length mean {meta['length_tokens']['mean']} median "
          f"{meta['length_tokens']['median']} max {meta['length_tokens']['max']}")
    print(f"checks OK: ids < {VOCAB_ROWS}, labels valid or {IGNORE_ID}, every row trains >= 1 token "
          f"and ends with the end token ({eos_targets:,} end-token targets)")
    rng = np.random.default_rng(a.seed + 1)
    for k in rng.choice(N, size=min(a.show, N), replace=False):
        tag = kept_ids[k] if seq_lengths is None else f"{len(row_chats[k])} chats"
        print(f"\n----- row {k} ({tag}): trained part in [[ ]] -----")
        print(show_row(enc, inputs[k], labels[k])[:1500])
    print(f"\nwrote {a.out}: inputs.bin + labels.bin ({2 * N * T * 2 / 1e6:.1f} MB), valid_counts.bin, meta.json"
          + (", seq_lengths.bin + seq_offsets.bin" if seq_lengths is not None else ""))


if __name__ == "__main__":
    main()
