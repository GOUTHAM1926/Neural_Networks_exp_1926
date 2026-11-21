#!/usr/bin/env python3
"""hf_val_groups.py -- 2026-11-20, step 13 of the SWA plan (the evaluation of the SFT arms A / C / D): the val loss of a
Hugging Face model folder on our packed SFT val split, chat by chat, in two attention modes, split into groups.

Modes:
  native  config.json as written (HF transformers reads layer_types per layer: arm C and the kv4_swa3 base run the
          window, arms A / D and the kv4 base run full attention);
  other   the other attention type with the same weights: a window model -> full attention in every layer (what vLLM and
          SGLang serve for our layout); a full-attention model -> the 350M SWA layout (window --window in layer i unless
          (i + 1) % --every == 0).
Groups (every labeled token is in exactly one group of each grouping):
  by chat length: chats of at most 1,024 tokens / longer chats (a window of 1,024 changes nothing in the short ones);
  by the token's position inside its chat: 0-1,023 / 1,024-2,047 / 2,048 and up (the window hides keys only from 1,024 on).
Loss = the trainer's definition: the sum of the next-token cross entropy over every labeled position (label != 65535),
divided by their count; each chat runs alone (positions 0..L-1), as the packed trainer computes it. So the native total
must equal the trainer's val loss of the same weights (step 9: within 0.00007).
usage: python hf_val_groups.py <hf folder> <data dir with val/> --name <tag> [--modes native,other] [--window 1024]
       [--every 4] [--json <file>]
Derived from hf_val_loss.py (step 9), which stays unchanged.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

LEN_EDGE = 1024                                    # chat-length groups: L <= 1024 / L > 1024
POS_GROUPS = (("pos0-1023", 0, 1024), ("pos1024-2047", 1024, 2048), ("pos2048+", 2048, 1 << 30))
# 2026-11-20 21:20: + "trainer_rows" = only the rows the trainer's val loader uses (it drops the last incomplete
# micro-batch: the Tulu val split has 311 rows, B 4 -> rows 0-307), so this total is the like-for-like check against the
# trainer's val loss (the "total" covers every row). Old line (kept):
# KEYS = ["total", "chats<=1024", "chats>1024"] + [g[0] for g in POS_GROUPS]
KEYS = ["total", "trainer_rows", "chats<=1024", "chats>1024"] + [g[0] for g in POS_GROUPS]


def val_rows(data):
    return json.load(open(os.path.join(data, "val", "meta.json")))["N"]


def val_chats(data):
    m = json.load(open(os.path.join(data, "val", "meta.json")))
    N, T = m["N"], m["T"]
    inp = np.memmap(os.path.join(data, "val", "inputs.bin"), dtype=np.uint16, mode="r", shape=(N, T))
    lab = np.memmap(os.path.join(data, "val", "labels.bin"), dtype=np.uint16, mode="r", shape=(N, T))
    so = np.fromfile(os.path.join(data, "val", "seq_offsets.bin"), dtype=np.int64)
    sl = np.fromfile(os.path.join(data, "val", "seq_lengths.bin"), dtype=np.int32)
    for r in range(N):
        t = 0
        for L in sl[so[r]:so[r + 1]]:
            # Old line (kept; the row index is now yielded too, for the trainer_rows total):
            # yield np.array(inp[r, t:t + L], dtype=np.int64), np.array(lab[r, t:t + L], dtype=np.int64)
            yield r, np.array(inp[r, t:t + L], dtype=np.int64), np.array(lab[r, t:t + L], dtype=np.int64)
            t += int(L)


def pattern(cfg):
    lt = getattr(cfg, "layer_types", None)
    return "".join("G" if t == "full_attention" else "L" for t in lt) if lt else "-"


def is_windowed(cfg):
    lt = getattr(cfg, "layer_types", None) or []
    return bool(getattr(cfg, "sliding_window", None)) and any(t == "sliding_attention" for t in lt)


def make_config(folder, mode, window, every):
    cfg = AutoConfig.from_pretrained(folder)
    n = cfg.num_hidden_layers
    if mode == "other":
        if is_windowed(cfg):                       # window model -> full attention everywhere
            cfg.use_sliding_window, cfg.sliding_window = False, None
            cfg.layer_types = ["full_attention"] * n
        else:                                      # full model -> the SWA layout
            cfg.use_sliding_window, cfg.sliding_window, cfg.max_window_layers = True, window, n
            cfg.layer_types = ["full_attention" if (i + 1) % every == 0 else "sliding_attention" for i in range(n)]
    return cfg


@torch.no_grad()
def run(folder, data, mode, window, every, device, max_chats=0, trainer_batch=4):
    cfg = make_config(folder, mode, window, every)
    model = AutoModelForCausalLM.from_pretrained(folder, config=cfg, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    desc = f"sliding_window {model.config.sliding_window}, layer_types {pattern(model.config)}"
    s = dict.fromkeys(KEYS, 0.0)
    c = dict.fromkeys(KEYS, 0)
    n, n_long, t0 = 0, 0, time.time()
    trainer_rows = (val_rows(data) // trainer_batch) * trainer_batch
    for r, ids, lab in val_chats(data):
        keep = lab != 65535
        if not keep.any():
            continue
        logits = model(input_ids=torch.from_numpy(ids)[None].to(device), use_cache=False).logits[0].float()
        k = torch.from_numpy(keep).to(device)
        ce = torch.nn.functional.cross_entropy(logits[k], torch.from_numpy(lab).to(device)[k],
                                               reduction="none").double().cpu().numpy()
        pos = np.nonzero(keep)[0]
        L = len(ids)
        for key in ("total", "chats<=1024" if L <= LEN_EDGE else "chats>1024") + (("trainer_rows",) if r < trainer_rows else ()):
            s[key] += float(ce.sum())
            c[key] += int(ce.size)
        for key, lo, hi in POS_GROUPS:
            m = (pos >= lo) & (pos < hi)
            s[key] += float(ce[m].sum())
            c[key] += int(m.sum())
        n += 1
        n_long += int(L > LEN_EDGE)
        if max_chats and n >= max_chats:
            break
    del model
    torch.cuda.empty_cache()
    res = {key: (s[key] / c[key] if c[key] else float("nan")) for key in KEYS}
    return res, c, n, n_long, desc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--modes", default="native,other")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-chats", type=int, default=0, help="only the first N chats (smoke test); 0 = all")
    a = ap.parse_args()
    out = []
    for mode in a.modes.split(","):
        res, cnt, n, n_long, desc, dt = run(a.folder, a.data, mode, a.window, a.every, "cuda", a.max_chats)
        parts = " | ".join(f"{key} {res[key]:.6f} ({cnt[key]:,})" for key in KEYS)
        print(f"HF val groups | {a.name} | {mode} | {parts} | {desc} | {n} chats ({n_long} longer than 1,024) | {dt:.0f} s",
              flush=True)
        out.append({"name": a.name, "mode": mode, "folder": a.folder, "loss": res, "tokens": cnt, "chats": n,
                    "chats_longer_1024": n_long, "attention": desc, "seconds": round(dt, 1)})
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
