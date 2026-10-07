#!/usr/bin/env python3
"""Turn a Hugging Face chat dataset into BluTrain SFT files (SFT report sections 3.2-3.10).

Output directory:
  inputs.bin        uint16 [N, T]  token ids the model reads, padded with 50256 (<|endoftext|>)
  labels.bin        uint16 [N, T]  token to predict at each position; 65535 = ignore (prompt, padding)
  valid_counts.bin  uint32 [N]     number of non-ignored labels in each row (for the global denominator)
  meta.json         settings, dataset revision, tool versions, counts, drop reasons, length histogram

Template "tulu-plain-v1" (option A: role markers are plain text, each answer ends with the real end token):
  <|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...
Each piece is tokenized on its own, so a marker never merges with the text next to it (BPE).
Only answer tokens and the end token after each answer are targets; every assistant turn is trained.
Text that looks like a special token (e.g. "<|endoftext|>" inside a user message) is encoded as plain text.
Default data: HuggingFaceH4/ultrachat_200k (English-only, MIT), so no language filter is needed.

Rows are labelled the usual way: inputs = ids[:-1], labels = ids[1:] with prompt positions set to 65535.
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

    inputs = np.empty((a.num, T), dtype=np.uint16)
    labels = np.empty((a.num, T), dtype=np.uint16)
    valid = np.empty(a.num, dtype=np.uint32)
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
        k = len(kept_ids)
        inputs[k], labels[k], valid[k] = build_row(res[0], res[1], T)
        lengths.append(len(res[0]) - 1)
        kept_src[src] += 1
        kept_ids.append(str(row.get("id") or row.get("prompt_id") or int(idx)))
    N = len(kept_ids)
    if N < a.num:
        print(f"WARNING: only {N:,} usable rows after scanning everything")
    inputs, labels, valid = inputs[:N], labels[:N], valid[:N]

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
    lens = np.asarray(lengths)
    hist_edges = [0, 256, 512, 1024, 2048, 3072, 4096]
    meta = {
        "format": {"inputs": "uint16 [N,T] little-endian, row-major", "labels": "uint16 [N,T], 65535 = ignore",
                   "valid_counts": "uint32 [N]", "label_rule": "labels = ids[1:] masked; inputs = ids[:-1]"},
        "N": N, "T": T, "pad_id": PAD_ID, "ignore_id": IGNORE_ID, "vocab_rows": VOCAB_ROWS,
        "template": TEMPLATE, "tokenizer": f"tiktoken gpt2 ({tiktoken.__version__})",
        "dataset": a.dataset, "revision": a.revision, "split": a.split, "seed": a.seed,
        "filters": {"max_tokens": T, "language": "none (the dataset is English-only)"},
        "rows_scanned": scanned, "rows_kept": N, "drops": dict(drops),
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

    print(f"kept {N:,} of {scanned:,} scanned | drops {dict(drops)}")
    print(f"trained tokens {meta['trained_tokens']:,} = {100 * meta['trained_share_of_positions']:.1f}% of "
          f"{N * T:,} positions | length mean {meta['length_tokens']['mean']} median "
          f"{meta['length_tokens']['median']} max {meta['length_tokens']['max']}")
    print(f"checks OK: ids < {VOCAB_ROWS}, labels valid or {IGNORE_ID}, every row trains >= 1 token "
          f"and ends with the end token ({eos_targets:,} end-token targets)")
    rng = np.random.default_rng(a.seed + 1)
    for k in rng.choice(N, size=min(a.show, N), replace=False):
        print(f"\n----- row {k} ({kept_ids[k]}): trained part in [[ ]] -----")
        print(show_row(enc, inputs[k], labels[k])[:1500])
    print(f"\nwrote {a.out}: inputs.bin + labels.bin ({2 * N * T * 2 / 1e6:.1f} MB), valid_counts.bin, meta.json")


if __name__ == "__main__":
    main()
