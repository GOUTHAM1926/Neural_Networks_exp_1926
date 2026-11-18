#!/usr/bin/env python3
"""End-to-end gate for an SFT setup (2026-10-28): the trainer's step-0 val loss must equal a numpy reference.

--data <split dir> (inputs.bin, labels.bin, valid_counts.bin, seq_lengths.bin, seq_offsets.bin, meta.json from
prepare_sft_data.py --pack ffd) and --weights <dir from hf_to_blutrain.py>. Takes the first --rows packed rows, runs
every chat ALONE through the reference forward of hf_to_blutrain.py (fp32 math, positions 0..len-1: what the packed
attention + RoPE positions must give), and prints the loss the trainer reports: the sum of the cross-entropy over every
trained label divided by the number of trained labels (one global denominator).
--make-subset <out>: also writes those rows as a data folder with train/ and val/ (the same rows), for the trainer:
its step-0 "[val] ... val_loss" must match within the bf16 tolerance (the trainer runs bf16, the reference fp32).
Rows must be a multiple of the trainer's micro-batch B (it drops leftover rows).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_to_blutrain import ref_logits  # noqa: E402

IGNORE = 65535


def load_weights(d):
    W = {}
    for line in open(os.path.join(d, "index.txt")):
        parts = line.split()
        W[parts[0]] = np.fromfile(os.path.join(d, parts[0] + ".bin"), dtype=np.float32).reshape(
            [int(x) for x in parts[1:]])
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--make-subset", default=None)
    ap.add_argument("--skip-ref", action="store_true", help="only write the subset (reference_loss null); run again "
                    "without it on the subset's val/ to get the number")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    N, T = meta["N"], meta["T"]
    assert meta["packed"], "packed data only"
    R = min(a.rows, N)
    inp = np.fromfile(os.path.join(a.data, "inputs.bin"), dtype=np.uint16).reshape(N, T)
    lab = np.fromfile(os.path.join(a.data, "labels.bin"), dtype=np.uint16).reshape(N, T)
    cnt = np.fromfile(os.path.join(a.data, "valid_counts.bin"), dtype=np.uint32)
    lens = np.fromfile(os.path.join(a.data, "seq_lengths.bin"), dtype=np.int32)
    offs = np.fromfile(os.path.join(a.data, "seq_offsets.bin"), dtype=np.int64)

    wmeta = json.load(open(os.path.join(a.weights, "meta.json")))["training_meta"]
    m = {"d": int(wmeta["d_model"]), "L": int(wmeta["n_layers"]), "Hq": int(wmeta["q_heads"]),
         "Hk": int(wmeta["kv_heads"]), "hd": int(wmeta["head_dim"]), "F": int(wmeta["ffn_hidden"]),
         "V": int(wmeta["vocab_size"]), "theta": float(wmeta["rope_theta"]), "rms_eps": float(wmeta["rms_eps"]),
         "qk_eps": float(wmeta["qk_eps"])}
    W = load_weights(a.weights) if not a.skip_ref else None

    total, count, chats = 0.0, 0, 0
    for r in range(R if not a.skip_ref else 0):
        t = 0
        for c in range(offs[r], offs[r + 1]):
            n = int(lens[c])
            ids, y = inp[r, t:t + n].astype(np.int64), lab[r, t:t + n]
            keep = y != IGNORE
            if keep.any():
                lg = ref_logits(W, m, ids).astype(np.float64)[keep]
                yy = y[keep].astype(np.int64)
                lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
                total += float((lse - lg[np.arange(len(yy)), yy]).sum())
                count += int(keep.sum())
            t += n
            chats += 1
        assert count == int(cnt[:r + 1].sum()), "trained-label count differs from valid_counts.bin"
    ref = total / count if count else None
    if ref is not None:
        print(f"reference: {R} rows, {chats} chats, {count:,} trained labels -> loss {ref:.6f} nats (sum {total:.3f})")

    if a.make_subset:
        for split in ("train", "val"):
            o = os.path.join(a.make_subset, split)
            os.makedirs(o, exist_ok=True)
            inp[:R].tofile(os.path.join(o, "inputs.bin"))
            lab[:R].tofile(os.path.join(o, "labels.bin"))
            cnt[:R].tofile(os.path.join(o, "valid_counts.bin"))
            lens[offs[0]:offs[R]].tofile(os.path.join(o, "seq_lengths.bin"))
            (offs[:R + 1] - offs[0]).astype(np.int64).tofile(os.path.join(o, "seq_offsets.bin"))
            # 2026-10-31: no seq_complete.bin any more: long chats are dropped, never cut (prepare_sft_data.py)
            sm = {k: meta[k] for k in ("T", "pad_id", "end_id", "ignore_id", "vocab_rows", "tokenizer_ids", "bos_id",
                                       "packed", "template", "tokenizer") if k in meta}
            sm.update({"N": R, "subset_of": os.path.abspath(a.data), "reference_loss": ref,
                       "trained_tokens": int(cnt[:R].sum())})
            json.dump(sm, open(os.path.join(o, "meta.json"), "w"), indent=1)
        shutil.copy(__file__, os.path.join(a.make_subset, "ref_val_loss.py"))
        print(f"wrote {a.make_subset}/train and /val ({R} rows each); reference loss {ref}")


if __name__ == "__main__":
    main()
