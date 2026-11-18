#!/usr/bin/env python3
"""
Supervised fine-tuning with the pretraining run's optimizer (ZeRO-1 Muon + AdamW, zero_muon.py)
and numerics (blu_model.py).

    torchrun --nproc_per_node 8 sft.py configs/sft_lr2e-5_10k.yaml     # H200 box
    python sft.py configs/sft_lr2e-5_10k.yaml                          # single process

Config: YAML, $VARS in strings are expanded from the environment (source env.sh first).

Batching
  * Global batch = `global_batch` conversations, drawn from the subset (order.npy[:subset], the
    held-out tail of order.npy excluded, then `exclude_sources` removed from the training set) in
    a fresh shuffle per epoch; a final partial batch is dropped. Every rank builds the same batch.
  * The batch is sorted by length and dealt round-robin to ranks, then each rank splits its share
    into micro-batches of at most `micro_tokens` padded tokens (right padding with <|pad|>), cut
    where padded tokens + `micro_cost` per micro-batch is smallest (partition()).
  * Loss = sum of per-token cross-entropy over trained tokens / trained tokens in the WHOLE global
    batch. Gradients summed over micro-batches and ranks are then exactly the gradient of the
    global token mean, independent of the rank count and of how micro-batches are cut.
  * Logits are computed in fp32 only at trained positions.
Schedule: linear warmup over `warmup_ratio` of the steps, then linear decay to min_lr_ratio * lr.
max_steps > 0 stops (and saves) after that many steps without changing the schedule; rerunning the
same config without it resumes from last/ (e.g. max_steps = one epoch's steps, then continue).
Saving: every `save_every` steps the resume state last/ is rewritten (to last.tmp, then swapped in,
so a crash while saving keeps the previous one; the swap needs room for two copies). HF exports
step_<N>/ are kept at every `export_every` steps, at the end of every epoch and at the stop step.
A resumed run first drops log.jsonl lines past the resumed step (those steps are redone).

Outputs in out_dir
  step_<N>/    HF checkpoint (model, tokenizer with chat template, train_info.json), for eval
  last/        resume state: native params + one optimizer shard per rank (same rank count needed)
  log.jsonl    one line per step (and per validation)
  config.yaml  the resolved config
"""

import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from blu_model import load_model
from zero_muon import ZeroMuonAdamW

DEFAULTS = {
    "model_type": "blu", "model_path": None, "tokenizer": None, "data": None,
    "subset": 0, "exclude_sources": [], "val_examples": 512, "val_every": 0, "epochs": 2, "global_batch": 128,
    "micro_tokens": 16384, "micro_cost": 512, "lr": 2.0e-5, "warmup_ratio": 0.03, "min_lr_ratio": 0.0,
    "weight_decay": 0.1, "momentum": 0.95, "ns_steps": 5, "betas": [0.9, 0.95], "eps": 1.0e-8,
    "clip": 1.0, "seed": 8, "grad_checkpointing": True, "compile": False, "save_every": 0, "export_every": 0, "out_dir": None,
    "resume": True, "log_every": 1, "device": "cuda", "max_steps": 0,
}


def load_config(path, defaults=DEFAULTS, required=("model_path", "tokenizer", "data", "out_dir")):
    with open(path) as f:
        cfg = {**defaults, **(yaml.safe_load(f) or {})}
    cfg = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()}
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    # PyYAML reads "2e-5" (no decimal point) as a string; coerce every float-valued key.
    for k, v in defaults.items():
        if isinstance(v, float):
            cfg[k] = float(cfg[k])
    for k in required:
        if not cfg[k]:
            raise ValueError(f"config needs {k}")
    return cfg


class SFTData:
    def __init__(self, path, subset, val_examples, exclude_sources=()):
        self.tokens = np.memmap(os.path.join(path, "tokens.bin"), dtype=np.uint32, mode="r")
        self.mask = np.memmap(os.path.join(path, "mask.bin"), dtype=np.uint8, mode="r")
        self.index = np.load(os.path.join(path, "index.npy"))
        order = np.load(os.path.join(path, "order.npy"))
        self.val = order[len(order) - val_examples:] if val_examples else order[:0]
        train = order[:len(order) - val_examples]
        if exclude_sources:          # training set only, so the validation set stays the same
            with open(os.path.join(path, "sources.json")) as f:
                names = json.load(f)
            unknown = set(exclude_sources) - set(names)
            if unknown:
                raise ValueError(f"exclude_sources not in {path}/sources.json: {sorted(unknown)}")
            ids = [names.index(n) for n in exclude_sources]
            train = train[~np.isin(self.index["source"][train], ids)]
        self.train = train[:subset] if subset else train

    def example(self, j):
        r = self.index[j]
        a, n = int(r["offset"]), int(r["length"])
        return self.tokens[a:a + n], self.mask[a:a + n]

    def n_loss(self, items):
        return int(self.index["n_loss"][items].sum())


def partition(lengths, budget, cost):
    """Split `lengths` (sorted descending) into contiguous groups [(i, j)], each padded to its first
    (longest) item and holding at most `budget` padded tokens (a single longer item gets its own
    group), minimizing padded tokens + `cost` per group (dynamic programming; `cost` stands for the
    fixed overhead of one more forward/backward)."""
    n = len(lengths)
    best, cut = [0.0] + [float("inf")] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            if j - i > 1 and (j - i) * lengths[i] > budget:
                break
            v = best[i] + (j - i) * lengths[i] + cost
            if v < best[j]:
                best[j], cut[j] = v, i
    groups, j = [], n
    while j:
        groups.append((cut[j], j))
        j = cut[j]
    return groups[::-1]


def micro_batches(data, items, rank, world, micro_tokens, pad_id, micro_cost=512):
    """This rank's micro-batches for one global batch: list of (input, target, loss_mask) int64/bool."""
    items = sorted(items, key=lambda j: (-int(data.index["length"][j]), int(j)))[rank::world]
    lengths = [int(data.index["length"][j]) for j in items]
    groups = [items[i:j] for i, j in partition(lengths, micro_tokens, micro_cost)]
    out = []
    for g in groups:
        T = int(data.index["length"][g[0]]) - 1
        x = np.full((len(g), T), pad_id, dtype=np.int64)
        y = np.full((len(g), T), pad_id, dtype=np.int64)
        m = np.zeros((len(g), T), dtype=bool)
        for b, j in enumerate(g):
            ids, mask = data.example(j)
            n = len(ids) - 1
            x[b, :n], y[b, :n], m[b, :n] = ids[:-1], ids[1:], mask[1:] != 0
        out.append((torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)))
    return out


def micro_loss(model, x, y, m):
    """Summed fp32 cross-entropy over trained positions."""
    h = model.hidden(x)
    logits = F.linear(h[m].float(), model.output_weight().float())
    return F.cross_entropy(logits, y[m], reduction="sum")


def lr_at(step, total, cfg):
    warm = max(1, round(cfg["warmup_ratio"] * total))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    frac = (step + 1 - warm) / max(1, total - warm)
    return cfg["lr"] * (1.0 - (1.0 - cfg["min_lr_ratio"]) * min(frac, 1.0))


def all_sum(t):
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


@torch.no_grad()
def validate(model, data, rank, world, cfg, pad_id, device):
    if len(data.val) == 0:
        return None
    model.eval()
    tot = torch.zeros(2, dtype=torch.float64, device=device)
    for x, y, m in micro_batches(data, list(data.val), rank, world, cfg["micro_tokens"], pad_id, cfg["micro_cost"]):
        x, y, m = x.to(device), y.to(device), m.to(device)
        tot[0] += micro_loss(model, x, y, m).double()
        tot[1] += m.sum()
    all_sum(tot)
    model.train()
    return (tot[0] / tot[1]).item()


def main():
    cfg = load_config(sys.argv[1])
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        backend = "nccl" if cfg["device"] == "cuda" else "gloo"
        dist.init_process_group(backend)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if cfg["device"] == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    else:
        device = torch.device("cpu")
    master = rank == 0
    out = cfg["out_dir"]
    log_path = os.path.join(out, "log.jsonl")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    pad_id = tok.pad_token_id
    data = SFTData(cfg["data"], cfg["subset"], cfg["val_examples"], cfg["exclude_sources"])
    G = cfg["global_batch"]
    steps_per_epoch = len(data.train) // G
    total = steps_per_epoch * cfg["epochs"]
    stop = min(total, cfg["max_steps"]) if cfg["max_steps"] else total   # stop early, same schedule
    if steps_per_epoch == 0:
        raise ValueError(f"subset of {len(data.train)} examples is smaller than one global batch ({G})")

    model = load_model(cfg["model_type"], cfg["model_path"], cfg["grad_checkpointing"]).to(device)
    model.train()
    if cfg["compile"]:
        model.compile_blocks()
    opt = ZeroMuonAdamW(model, cfg["lr"], cfg["weight_decay"], cfg["momentum"], cfg["ns_steps"],
                        tuple(cfg["betas"]), cfg["eps"], cfg["clip"])

    start = 0
    last = os.path.join(out, "last")
    if cfg["resume"] and os.path.isfile(os.path.join(last, "state.json")):
        with open(os.path.join(last, "state.json")) as f:
            st = json.load(f)
        from safetensors.torch import load_file
        model.load_state_dict({k: v.to(device) for k, v in load_file(os.path.join(last, "native.safetensors")).items()},
                              strict=False)
        opt.load_state_dict(torch.load(os.path.join(last, f"optim_rank{rank}.pt"), map_location=device))
        start = st["step"]

    if master:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        n_muon = sum(r["param"].numel() for r in opt.records if r["muon"])
        print(f"[sft] {cfg['model_type']} {cfg['model_path']}: {n_params / 1e6:.1f}M params "
              f"({n_muon / 1e6:.1f}M Muon), world {world}, owned elems per rank {opt.load}", flush=True)
        print(f"[sft] data {cfg['data']}: {len(data.train)} train / {len(data.val)} val conversations, "
              f"{data.n_loss(data.train) / 1e6:.1f}M trained tokens per epoch | {steps_per_epoch} steps/epoch x "
              f"{cfg['epochs']} = {total} steps (global batch {G}) | start at step {start}", flush=True)
        if start:
            print(f"[sft] resumed from {last}", flush=True)

    def save(step, export):
        tag = os.path.join(out, f"step_{step}")
        tmp = last + ".tmp"
        if master:
            if export:
                info = {"step": step, "total_steps": total, "config": cfg,
                        "trained_conversations": step * G, "epochs_done": step / steps_per_epoch, "phase": "sft"}
                model.save_hf(tag, tok, info)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            model.save_native(os.path.join(tmp, "native.safetensors"))
        if distributed:
            dist.barrier()
        torch.save(opt.state_dict(), os.path.join(tmp, f"optim_rank{rank}.pt"))
        if distributed:
            dist.barrier()
        if master:
            with open(os.path.join(tmp, "state.json"), "w") as f:
                json.dump({"step": step, "world": world}, f)
            shutil.rmtree(last, ignore_errors=True)
            os.rename(tmp, last)
            print(f"[sft] saved {last} (step {step})" + (f" and {tag}" if export else ""), flush=True)
        if distributed:
            dist.barrier()

    def log(rec):
        if master:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    if master and start and os.path.isfile(log_path):     # steps after the resume point are redone
        with open(log_path) as f:
            kept = [l for l in f if json.loads(l)["step"] <= start]
        with open(log_path, "w") as f:
            f.writelines(kept)

    if start == 0 and cfg["val_every"]:
        v = validate(model, data, rank, world, cfg, pad_id, device)
        log({"step": 0, "val_loss": v})
        if master:
            print(f"[sft] step 0 val loss {v:.4f}", flush=True)

    t_last, tok_count, n_since = time.time(), 0, 0
    t_launch = t_last
    for step in range(start, stop):
        epoch, b = divmod(step, steps_per_epoch)
        perm = np.random.default_rng(cfg["seed"] + epoch).permutation(data.train)
        items = perm[b * G:(b + 1) * G]
        n_global = data.n_loss(items)
        lr = lr_at(step, total, cfg)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        real = padded = 0
        for x, y, m in micro_batches(data, items, rank, world, cfg["micro_tokens"], pad_id, cfg["micro_cost"]):
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            ls = micro_loss(model, x, y, m)
            (ls / n_global).backward()
            loss_sum += ls.detach().double()
            real += int((x != pad_id).sum())
            padded += x.numel()
        stats = opt.step(lr)
        counts = all_sum(torch.tensor([loss_sum.item(), real, padded], dtype=torch.float64, device=device))
        loss = counts[0].item() / n_global
        tok_count += int(counts[1].item())
        n_since += 1
        rec = {"step": step + 1, "epoch": epoch, "lr": lr, "loss": loss, "loss_tokens": n_global,
               "tokens": int(counts[1].item()), "pad_frac": 1 - counts[1].item() / max(counts[2].item(), 1),
               **stats}
        if (step + 1) % cfg["log_every"] == 0 or step + 1 == stop:
            now = time.time()
            dt = now - t_last
            rec["tok_per_s"] = tok_count / max(dt, 1e-9)
            rec["step_time"] = dt / n_since       # wall seconds per step, incl. the previous step's val/save
            if device.type == "cuda":
                rec["mem_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            eta = (now - t_launch) / (step + 1 - start) * (stop - step - 1)
            t_last, tok_count, n_since = now, 0, 0
            if master:
                print(f"[sft] step {step + 1}/{total} ep {epoch} lr {lr:.3e} loss {loss:.4f} "
                      f"|g| {stats['grad_norm']:.3f} (muon {stats['grad_norm_muon']:.3f} adamw {stats['grad_norm_adamw']:.3f} "
                      f"clip {stats['clip_coef']:.3f}) pad {rec['pad_frac']:.2f} {rec['tok_per_s']:.0f} tok/s {rec['step_time']:.2f}s/step "
                      f"eta {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m"
                      + (f" mem {rec['mem_gib']:.1f}G" if "mem_gib" in rec else "")
                      + (" SKIPPED" if stats["skipped"] else ""), flush=True)
        log(rec)
        if cfg["val_every"] and ((step + 1) % cfg["val_every"] == 0 or step + 1 == stop):
            v = validate(model, data, rank, world, cfg, pad_id, device)
            log({"step": step + 1, "val_loss": v})
            if master:
                print(f"[sft] step {step + 1} val loss {v:.4f}", flush=True)
        export = ((cfg["export_every"] and (step + 1) % cfg["export_every"] == 0)
                  or (step + 1) % steps_per_epoch == 0 or step + 1 == stop)
        if export or (cfg["save_every"] and (step + 1) % cfg["save_every"] == 0):
            save(step + 1, export)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
