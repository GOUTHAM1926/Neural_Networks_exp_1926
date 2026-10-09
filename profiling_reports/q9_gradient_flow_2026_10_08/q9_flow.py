#!/usr/bin/env python3
"""Q9: where the loss and the gradients go in SFT, measured on a tiny model.

One chat, tokenized by our real prepare_sft_data.py (tulu-plain-v1), padded to T=28.
Tiny pre-norm transformer like ours: tied embedding (50304 rows), RMSNorm, causal attention, SwiGLU.
fp64 so exact zeros are exact.

E1  one forward + backward: per-position loss, logit gradients, residual-stream gradients per layer,
    per-position contributions to every weight, and the two halves of the tied embedding gradient.
E2  checks: change the padding -> gradients bit-identical; change one prompt word -> they change.
E3  train 60 steps with the SFT mask and 60 steps with every position trained (pretraining style),
    and track -log p(next token) at EVERY position, trained or not.
Writes q9_results.json next to this file.
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/blu-bridge016/Downloads/BluTrain_sft/sft")
import tiktoken                     # noqa: E402
import prepare_sft_data as P        # noqa: E402

torch.set_default_dtype(torch.float64)
V, D, H, L, T = 50304, 64, 4, 4, 28
IGN = P.IGNORE_ID
enc = tiktoken.get_encoding("gpt2")


def make_row(user, answer):
    (ids, mask), err = P.tokenize_conversation(
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], enc)
    assert err is None, err
    inp, lab, n = P.build_row(ids, mask, T)
    nxt = list(ids[1:]) + [P.PAD_ID] * (T - (len(ids) - 1))      # the true next token, even where ignored
    return (torch.tensor(inp.astype("int64")), torch.tensor(lab.astype("int64")),
            torch.tensor(nxt), len(ids) - 1, n)


class RMSNorm(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D), RMSNorm(D)
        self.q = torch.nn.Linear(D, D, bias=False)
        self.k = torch.nn.Linear(D, D, bias=False)
        self.v = torch.nn.Linear(D, D, bias=False)
        self.o = torch.nn.Linear(D, D, bias=False)
        self.w1 = torch.nn.Linear(D, 4 * D, bias=False)
        self.w3 = torch.nn.Linear(D, 4 * D, bias=False)
        self.w2 = torch.nn.Linear(4 * D, D, bias=False)
        self.attn_w = None

    def forward(self, x):
        t = x.shape[0]
        a = self.n1(x)
        q = self.q(a).view(t, H, D // H).transpose(0, 1)
        k = self.k(a).view(t, H, D // H).transpose(0, 1)
        v = self.v(a).view(t, H, D // H).transpose(0, 1)
        s = q @ k.transpose(-1, -2) / math.sqrt(D // H)
        s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
        w = s.softmax(-1)
        self.attn_w = w.detach()
        x = x + self.o((w @ v).transpose(0, 1).reshape(t, D))
        b = self.n2(x)
        return x + self.w2(F.silu(self.w1(b)) * self.w3(b))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(V, D)
        torch.nn.init.normal_(self.wte.weight, std=0.02)
        self.blocks = torch.nn.ModuleList(Block() for _ in range(L))
        self.nf = RMSNorm(D)

    def forward(self, idx, keep=None):
        h = [self.wte(idx)]
        if keep is not None:
            h[0].retain_grad()
        x = h[0]
        for blk in self.blocks:
            x = blk(x)
            if keep is not None:
                x.retain_grad()
            h.append(x)
        hf = self.nf(x)
        logits = hf @ self.wte.weight.T                                   # tied: the same table
        if keep is not None:
            logits.retain_grad()
            keep.update(h=h, hf=hf, logits=logits)
        return logits


def sft_loss(logits, lab):
    """Our rule: sum of -log p over trained positions / N (ignored labels give 0 and are not counted)."""
    n = int((lab != IGN).sum())
    return F.cross_entropy(logits, lab, ignore_index=IGN, reduction="sum") / n, n


def contributions(model, idx, lab):
    """Per-position share of every weight gradient: dW = sum_t g_t (x) x_t, so |g_t|*|x_t| per position."""
    store = {}

    def hook(name):
        def f(mod, args, out):
            x = args[0].detach()
            out.register_hook(lambda g: store.__setitem__(name, (g.norm(dim=-1) * x.norm(dim=-1)).tolist()))
        return f

    hs = [getattr(blk, n).register_forward_hook(hook(f"L{i + 1}.{n}"))
          for i, blk in enumerate(model.blocks) for n in ("q", "k", "v", "o", "w1", "w3", "w2")]
    model.zero_grad()
    loss, _ = sft_loss(model(idx), lab)
    loss.backward()
    for h in hs:
        h.remove()
    return store


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def main():
    torch.manual_seed(1926)
    model = Tiny()
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    idx, lab, nxt, n_real, n_train = make_row("What is the capital of France?", "Paris.")
    pieces = [enc.decode([int(t)]) for t in idx]
    out = {"config": {"V": V, "D": D, "H": H, "L": L, "T": T, "dtype": "float64", "seed": 1926,
                      "template": P.TEMPLATE, "n_real_inputs": n_real, "n_trained": n_train},
           "tokens": [{"pos": t, "input": int(idx[t]), "input_text": pieces[t],
                       "label": int(lab[t]), "label_text": enc.decode([int(lab[t])]) if lab[t] != IGN else None,
                       "next_text": enc.decode([int(nxt[t])]),
                       "kind": "pad" if t >= n_real else ("trained" if lab[t] != IGN else "prompt")}
                      for t in range(T)]}

    # ---------------- E1: one forward + backward
    keep = {}
    model.zero_grad()
    logits = model(idx, keep)
    loss, n = sft_loss(logits, lab)
    loss.backward()
    p = logits.detach().softmax(-1)
    per_pos = []
    for t in range(T):
        tgt = int(nxt[t])
        row = {"pos": t, "p_next": float(p[t, tgt]), "nll_next": float(-math.log(p[t, tgt]))}
        row["loss_counted"] = float(-math.log(p[t, int(lab[t])])) if lab[t] != IGN else 0.0
        g = keep["logits"].grad[t]
        row["logit_grad_norm"] = float(g.norm())
        row["logit_grad_right"] = float(g[int(lab[t])]) if lab[t] != IGN else 0.0
        row["logit_grad_right_formula"] = float((p[t, int(lab[t])] - 1) / n) if lab[t] != IGN else 0.0
        row["logit_grad_nonzero_cols"] = int((g != 0).sum())
        per_pos.append(row)
    out["E1"] = {
        "N": n, "loss": float(loss),
        "per_pos": per_pos,
        "resid_grad": [[float(x) for x in h.grad.norm(dim=-1)] for h in keep["h"]],   # [layer 0..L][pos]
        "attn_last_layer_head0": model.blocks[-1].attn_w[0].tolist(),
        "contrib": contributions(model, idx, lab),
    }
    model.zero_grad()
    logits = model(idx, keep)
    sft_loss(logits, lab)[0].backward()
    g_logits = keep["logits"].grad                       # [T, V]
    out_side = g_logits.T @ keep["hf"].detach()          # output-layer half of the tied gradient
    in_side = torch.zeros(V, D).index_add_(0, idx, keep["h"][0].grad)   # embedding-lookup half
    tot = model.wte.weight.grad
    out["E1"]["wte"] = {
        "rows_nonzero_output_side": int((out_side.abs().sum(-1) > 0).sum()),
        "rows_nonzero_input_side": int((in_side.abs().sum(-1) > 0).sum()),
        "input_side_rows": sorted({enc.decode([int(i)]) for i in idx[(keep["h"][0].grad.abs().sum(-1) > 0)]}),
        "pad_row_input_side_norm": float(in_side[P.PAD_ID].norm()),
        "sum_matches_autograd_maxdiff": float((out_side + in_side - tot).abs().max()),
    }
    base = grads(model)

    # ---------------- E2: checks
    idx2 = idx.clone()
    g = torch.Generator().manual_seed(7)
    idx2[n_real:] = torch.randint(0, 50257, (T - n_real,), generator=g)
    model.zero_grad()
    sft_loss(model(idx2), lab)[0].backward()
    pad_diff = max(float((p_.grad - base[k]).abs().max()) for k, p_ in model.named_parameters())
    idx3, lab3, *_ = make_row("What is the capital of Japan?", "Paris.")
    assert (lab3 != IGN).sum() == n and len(idx3) == len(idx)
    model.zero_grad()
    l3, _ = sft_loss(model(idx3), lab3)
    l3.backward()
    prompt_diff = max(float((p_.grad - base[k]).abs().max()) for k, p_ in model.named_parameters())
    out["E2"] = {"pad_changed_to": [enc.decode([int(t)]) for t in idx2[n_real:]],
                 "pad_change_max_grad_diff": pad_diff,
                 "prompt_change": "France -> Japan", "prompt_change_loss": [float(loss), float(l3)],
                 "prompt_change_max_grad_diff": prompt_diff}

    # ---------------- E3: train with the SFT mask vs every position trained
    lab_all = torch.where(torch.arange(T) < n_real, nxt, torch.full((T,), IGN))
    curves = {}
    for name, labels in (("sft_mask", lab), ("all_positions", lab_all)):
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        rows = []
        for step in range(61):
            with torch.no_grad():
                lp = model(idx).log_softmax(-1)
                rows.append({"step": step, "nll": [float(-lp[t, int(nxt[t])]) for t in range(n_real)]})
            if step == 60:
                break
            opt.zero_grad()
            sft_loss(model(idx), labels)[0].backward()
            opt.step()
        curves[name] = rows
    out["E3"] = {"lr": 3e-3, "steps": 60, "optimizer": "AdamW", "curves": curves}

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q9_results.json")
    with open(path, "w") as f:
        json.dump(out, f)

    # ---------------- text summary
    print(f"tokens: {n_real} real inputs, {n} trained, {T - n_real} pad;  loss {float(loss):.4f}")
    for r, tk in zip(per_pos, out["tokens"]):
        print(f"{r['pos']:2d} {tk['kind']:7s} in={tk['input_text']!r:14s} label={tk['label_text']!r:16s} "
              f"loss={r['loss_counted']:.3f} |dlogits|={r['logit_grad_norm']:.2e} "
              f"right={r['logit_grad_right']:+.4f} (formula {r['logit_grad_right_formula']:+.4f})")
    print("residual-stream gradient norm, layers 0..L (rows) x positions:")
    for l, row in enumerate(out["E1"]["resid_grad"]):
        print(f"  h{l}: " + " ".join("  0   " if x == 0 else f"{x:.0e}" for x in row))
    print("wte:", out["E1"]["wte"])
    print("E2:", {k: v for k, v in out["E2"].items()})
    for name, rows in curves.items():
        first, last = rows[0]["nll"], rows[-1]["nll"]
        kinds = [tk["kind"] for tk in out["tokens"][:n_real]]
        pr = [i for i, k in enumerate(kinds) if k == "prompt"]
        tr = [i for i, k in enumerate(kinds) if k == "trained"]
        mean = lambda xs, ids: sum(xs[i] for i in ids) / len(ids)
        print(f"E3 {name}: prompt mean nll {mean(first, pr):.2f} -> {mean(last, pr):.2f} | "
              f"trained mean nll {mean(first, tr):.2f} -> {mean(last, tr):.4f}")


if __name__ == "__main__":
    main()
