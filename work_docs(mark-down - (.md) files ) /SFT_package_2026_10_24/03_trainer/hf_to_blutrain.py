#!/usr/bin/env python3
"""Convert a Hugging Face Qwen3-layout export of a BluTrain model (e.g. Navin's 135M base, step 428945) back into
BluTrain's own weights, for the SFT trainer's BLU_INIT_FROM_RAW (2026-10-28).

Input  --hf-dir: model.safetensors (bf16), config.json, training_meta.json (BluTrain's export metadata).
Output --out:    one <name>.bin per BluTrain weight (fp32, little-endian, BluTrain layout), index.txt
                 ("<name> <dim0> [<dim1>]" per line, in the trainer's load order) and meta.json (sources, checks).

The mapping (BluTrain names = bluscript_zero_ckpt.cpp export_safetensors, preproduction-135M f37c616):
  wte.weight                  <- lm_head.weight                [vocab, d]  (exact: see the embed check below)
  layers.i.attn_norm.weight   <- model.layers.i.input_layernorm.weight
  layers.i.wq/wk/wv/wo.weight <- self_attn.{q,k,v,o}_proj.weight, TRANSPOSED (HF Linear is [out, in], BluTrain [in, out])
  layers.i.q_gamma / k_gamma  <- self_attn.q_norm.weight / k_norm.weight
  layers.i.mlp_norm.weight    <- post_attention_layernorm.weight
  layers.i.gate_up.weight     <- [gate_proj^T | up_proj^T] along the out axis (SwiGLUKernels_sm90.cu: gate half first)
  layers.i.down.weight        <- mlp.down_proj.weight^T
  norm_f.weight               <- model.norm.weight
model.embed_tokens is NOT used: BluTrain ties the embedding to lm_head and multiplies the embedding OUTPUT by
sqrt(d_model) (embed_scale); the HF file folds that scale into embed_tokens and unties lm_head. The script checks
embed_tokens == bf16(lm_head x sqrt(d_model)) exactly, so lm_head is the original table.

--new-tokens "<|system|>,<|user|>,<|assistant|>" (arm B of the special-token ablation, 2026-10-28): the strings become
single special tokens at the first reserved ids (49152, 49153, ...; the model already has those rows, no resize). Each
new row starts as the MEAN of the rows of the pieces the ORIGINAL tokenizer splits the string into, counted as they occur
(<|assistant|> = mean of <, |, ass, istant, |, >), rounded to bf16: Laguna M.1/XS.2 report (arXiv 2605.27605) 4.1.1,
"initialized as the mean of its constituent subtokens' embeddings". wte is tied to lm_head, so the one row serves both.
NOT the current reserved row: it was never an input and pretraining only pushed it down as an output. The extended
tokenizer.json (the new strings as added special tokens) is written next to the weights.
--check: a numpy reference forward of the whole model (fp32 math) on a few texts; prints the next-token loss of the
mapping as built and of wrong mappings (gate/up swapped, GPT-J RoPE, no embed scale). The right mapping must give the
lowest loss by a clear margin.

2026-11-20 step 9 (SWA in SFT):
  * the sliding window travels with the model: read from config.json (use_sliding_window, sliding_window, layer_types,
    else max_window_layers) and written into meta.json as "swa_window" / "swa_global_every" (layer i global iff
    (i + 1) % swa_global_every == 0, the trainers' rule). The SFT trainer compares them with its own BLU_SWA_WINDOW /
    BLU_SWA_GLOBAL_EVERY at start-up and stops on a mismatch. A layer pattern that is not "every N-th layer global" is
    refused (the trainers cannot run it). No window in config.json: swa_window 0. --check's reference forward uses the
    same window (--window / --global-every only confirm it: a different value is refused);
  * merged from profiling_reports/swa_sft_2026_11_17/hf_to_blutrain_350m.py (the 350M copy of 2026-11-17, strict
    generalizations): the export metadata may be training_meta.json or blu_meta.json; model.safetensors may be F32 as
    well as BF16 (the trainer's exact load check still needs bf16-exact values, checked at write time as before); the
    reference's embedding scale = bf16(sqrt(d_model)) (22.625 for the 135M's 512, as before).
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def round_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        head.pop("__metadata__", None)
        base, out = 8 + n, {}
        for name, e in head.items():
            # 2026-11-20 step 9 (SWA in SFT) (from the 350M copy): F32 exports too. Old lines (kept):
            #     assert e["dtype"] == "BF16", (name, e["dtype"])
            #     out[name] = bf16_to_f32(np.frombuffer(f.read(b - a), dtype=np.uint16)).reshape(e["shape"])
            assert e["dtype"] in ("BF16", "F32"), (name, e["dtype"])
            a, b = e["data_offsets"]
            f.seek(base + a)
            if e["dtype"] == "F32":
                out[name] = np.frombuffer(f.read(b - a), dtype="<f4").reshape(e["shape"]).astype(np.float32)
            else:
                out[name] = bf16_to_f32(np.frombuffer(f.read(b - a), dtype=np.uint16)).reshape(e["shape"])
    return out


def convert(hf, L):
    W = {"wte.weight": hf["lm_head.weight"]}
    for i in range(L):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        W[q + "attn_norm.weight"] = hf[p + "input_layernorm.weight"]
        for a, b in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"), ("wo", "o_proj")):
            W[q + a + ".weight"] = np.ascontiguousarray(hf[p + "self_attn." + b + ".weight"].T)
        W[q + "q_gamma"] = hf[p + "self_attn.q_norm.weight"]
        W[q + "k_gamma"] = hf[p + "self_attn.k_norm.weight"]
        W[q + "mlp_norm.weight"] = hf[p + "post_attention_layernorm.weight"]
        W[q + "gate_up.weight"] = np.ascontiguousarray(np.concatenate(
            [hf[p + "mlp.gate_proj.weight"].T, hf[p + "mlp.up_proj.weight"].T], axis=1))
        W[q + "down.weight"] = np.ascontiguousarray(hf[p + "mlp.down_proj.weight"].T)
    W["norm_f.weight"] = hf["model.norm.weight"]
    return W


# ---- numpy reference forward (BluTrain semantics, fp32 math) ----
def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, theta, interleaved):
    T, _, D = x.shape
    inv = theta ** (-np.arange(0, D // 2, dtype=np.float64) * 2 / D)
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :].astype(np.float32), np.sin(ang)[:, None, :].astype(np.float32)
    if not interleaved:                                    # NeoX half-split (BluTrain rope_interleaved=false, HF)
        a, b = x[..., :D // 2], x[..., D // 2:]
        return np.concatenate([a * c - b * s, b * c + a * s], -1)
    a, b = x[..., 0::2], x[..., 1::2]                      # GPT-J interleaved (a wrong variant, for the check)
    o = np.empty_like(x)
    o[..., 0::2], o[..., 1::2] = a * c - b * s, b * c + a * s
    return o


# Old signature (kept, 2026-11-20 step 9 (SWA in SFT)): def ref_logits(W, m, ids, swap_gate_up=False, interleaved=False, scale=True):
def ref_logits(W, m, ids, swap_gate_up=False, interleaved=False, scale=True, window=0, global_every=0):
    """Logits [T, vocab] of one sequence (positions 0..T-1), fp32 math. window > 0: layer i is global iff
    (i + 1) % global_every == 0, the others see key j from query i only when i - window < j <= i (HF sliding_window)."""
    d, L, Hq, Hk, hd, F = m["d"], m["L"], m["Hq"], m["Hk"], m["hd"], m["F"]
    G, T = Hq // Hk, len(ids)
    # 2026-11-20 step 9 (SWA in SFT) (from the 350M copy): bf16(sqrt(d_model)) (22.625 for 512, 32 for 1024). Old line (kept):
    #     x = W["wte.weight"][ids] * (np.float32(22.625) if scale else np.float32(1))   # bf16(sqrt(512)), as BluTrain
    es = float(round_bf16(np.array([np.sqrt(float(d))], dtype=np.float32))[0])
    x = W["wte.weight"][ids] * (np.float32(es) if scale else np.float32(1))
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    wmask = mask.copy()                                    # 2026-11-20 step 9 (SWA in SFT): the windowed layers' mask
    if window > 0:
        wmask[np.tril(np.ones((T, T), dtype=bool), -window)] = -np.inf
    for i in range(L):
        q_ = f"layers.{i}."
        h = rms(x, W[q_ + "attn_norm.weight"], m["rms_eps"])
        q = rms((h @ W[q_ + "wq.weight"]).reshape(T, Hq, hd), W[q_ + "q_gamma"], m["qk_eps"])
        k = rms((h @ W[q_ + "wk.weight"]).reshape(T, Hk, hd), W[q_ + "k_gamma"], m["qk_eps"])
        v = (h @ W[q_ + "wv.weight"]).reshape(T, Hk, hd)
        q, k = rope(q, m["theta"], interleaved), rope(k, m["theta"], interleaved)
        o = np.empty((T, Hq, hd), dtype=np.float32)
        for hh in range(Hq):
            kv = hh // G                                   # query head hh reads kv head hh // G (HF repeat_kv)
            # 2026-11-20 step 9 (SWA in SFT): the windowed layers use wmask. Old line (kept): sc = q[:, hh] @ k[:, kv].T / np.sqrt(hd) + mask
            local = window > 0 and global_every > 0 and (i + 1) % global_every != 0
            sc = q[:, hh] @ k[:, kv].T / np.sqrt(hd) + (wmask if local else mask)
            p = np.exp(sc - sc.max(-1, keepdims=True))
            o[:, hh] = (p / p.sum(-1, keepdims=True)) @ v[:, kv]
        x = x + o.reshape(T, Hq * hd) @ W[q_ + "wo.weight"]
        gu = rms(x, W[q_ + "mlp_norm.weight"], m["rms_eps"]) @ W[q_ + "gate_up.weight"]
        g, u = (gu[:, F:], gu[:, :F]) if swap_gate_up else (gu[:, :F], gu[:, F:])
        x = x + (g / (1 + np.exp(-g)) * u) @ W[q_ + "down.weight"]
    logits = rms(x, W["norm_f.weight"], m["rms_eps"]) @ W["wte.weight"].T
    return logits


def ref_loss(W, m, ids, **kw):
    """Mean next-token loss over the whole sequence (nats)."""
    lg = ref_logits(W, m, ids, **kw)[:-1].astype(np.float64)
    lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
    return float((lse - lg[np.arange(len(ids) - 1), ids[1:]]).mean())


CHECK_TEXTS = [
    "The Industrial Revolution began in Britain in the late eighteenth century. New machines changed how goods were "
    "made, and factories grew in towns that had been small villages. Workers moved from farms to cities, where they "
    "found jobs but also crowded housing and long hours.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n"
    "        a, b = b, a + b\n    return a\n\n\nprint(fibonacci(10))\n",
    "Solve for x: 3x + 7 = 22. Subtract 7 from both sides to get 3x = 15. Divide both sides by 3, so x = 5. Check: "
    "3 times 5 is 15, and 15 plus 7 is 22.",
]


def window_from_config(cfg, L):
    """2026-11-20 step 9 (SWA in SFT): (swa_window, swa_global_every) of a Hugging Face config, in the trainers' rule (layer i global iff
    (i + 1) % every == 0). (0, 4) when the config has no window. Raises when the layer pattern cannot be written so."""
    sw = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else None   # HF Qwen3: off unless use_sliding_window
    if not sw:
        return 0, 4, "no sliding window in config.json (use_sliding_window false or sliding_window null)"
    lt = cfg.get("layer_types")
    if lt is None:                                         # HF Qwen2/3 without layer_types: i >= max_window_layers slide
        mwl = int(cfg.get("max_window_layers", L))
        lt = ["sliding_attention" if i >= mwl else "full_attention" for i in range(L)]
    assert len(lt) == L, f"layer_types has {len(lt)} entries, the model {L} layers"
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    if not full:
        every = L + 1                                      # every layer windowed
    else:
        every = full[0] + 1
        want = [i for i in range(L) if (i + 1) % every == 0]
        if full == list(range(L)):
            return 0, 4, f"sliding_window {sw} but every layer is full_attention: no window"
        if full != want:
            raise SystemExit(f"layer_types: full-attention layers {full} are not 'every {every}-th layer' ({want}); "
                             "the trainers cannot run this pattern")
    pat = "".join("G" if t == "full_attention" else "L" for t in lt)
    return int(sw), every, f"sliding window {sw}, every {every}-th layer global: {pat}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", default="",
                    help="comma-separated strings made single special tokens at the reserved ids 49152.. (arm B)")
    ap.add_argument("--check", action="store_true", help="numpy reference forward on a few texts (needs tokenizers)")
    ap.add_argument("--window", type=int, default=None,
                    help="2026-11-20 step 9 (SWA in SFT): only confirms config.json's sliding window (a different value is refused)")
    ap.add_argument("--global-every", type=int, default=None, help="the same for the global-layer period")
    a = ap.parse_args()

    # 2026-11-20 step 9 (SWA in SFT) (from the 350M copy): training_meta.json or blu_meta.json (same field names). Old line (kept):
    #     meta = json.load(open(os.path.join(a.hf_dir, "training_meta.json")))
    mp = os.path.join(a.hf_dir, "training_meta.json")
    meta = json.load(open(mp if os.path.exists(mp) else os.path.join(a.hf_dir, "blu_meta.json")))
    m = {"d": int(meta["d_model"]), "L": int(meta["n_layers"]), "Hq": int(meta["q_heads"]),
         "Hk": int(meta["kv_heads"]), "hd": int(meta["head_dim"]), "F": int(meta["ffn_hidden"]),
         "V": int(meta["vocab_size"]), "theta": float(meta["rope_theta"]), "rms_eps": float(meta["rms_eps"]),
         "qk_eps": float(meta["qk_eps"])}
    assert meta.get("embed_scale") == "1" and meta.get("weight_tying") == "1", "expected embed_scale=1, tied"
    # 2026-11-20 step 9 (SWA in SFT): the sliding window, from config.json; the export metadata may carry it too (Navin's blu_meta.json:
    # swa_window / swa_global_every), and then it must agree.
    cfgj = json.load(open(os.path.join(a.hf_dir, "config.json")))
    swa_w, swa_e, swa_desc = window_from_config(cfgj, m["L"])
    if "swa_window" in meta:
        mw = int(meta["swa_window"]); me = int(meta.get("swa_global_every", swa_e))
        assert mw == swa_w and (mw == 0 or me == swa_e), f"export metadata window {mw}/{me} != config.json {swa_w}/{swa_e}"
    if a.window is not None and a.window != swa_w:
        raise SystemExit(f"--window {a.window} but config.json says {swa_w} ({swa_desc})")
    if a.global_every is not None and swa_w > 0 and a.global_every != swa_e:
        raise SystemExit(f"--global-every {a.global_every} but config.json says {swa_e} ({swa_desc})")
    print(f"window: {swa_desc} -> meta.json swa_window {swa_w}, swa_global_every {swa_e}")
    st = os.path.join(a.hf_dir, "model.safetensors")
    hf = read_safetensors(st)
    assert len(hf) == 11 * m["L"] + 3, f"{len(hf)} tensors, expected {11 * m['L'] + 3}"

    # the folded embedding scale: embed_tokens == bf16(lm_head x sqrt(d)) -> lm_head is the original tied table
    E, Lh = hf["model.embed_tokens.weight"], hf["lm_head.weight"]
    mism = int((round_bf16(Lh * np.float32(np.sqrt(float(m["d"])))) != E).sum())
    print(f"embed check: embed_tokens vs bf16(lm_head x sqrt({m['d']})): {mism} of {E.size} differ (must be 0)")
    assert mism == 0

    W = convert(hf, m["L"])
    new_tokens = [x for x in a.new_tokens.split(",") if x.strip()]
    tok_json = json.load(open(os.path.join(a.hf_dir, "tokenizer.json")))
    added = []
    if new_tokens:
        import tokenizers
        base_tk = tokenizers.Tokenizer.from_file(os.path.join(a.hf_dir, "tokenizer.json"))
        n_ids = base_tk.get_vocab_size(with_added_tokens=True)
        W["wte.weight"] = np.array(W["wte.weight"], dtype=np.float32)          # writable copy
        for k, t in enumerate(new_tokens):
            assert base_tk.token_to_id(t) is None, f"{t} is already a token"
            rid = n_ids + k
            assert rid < m["V"], f"no reserved row left for {t}"
            enc = base_tk.encode(t, add_special_tokens=False)
            row = round_bf16(W["wte.weight"][enc.ids].mean(0, dtype=np.float64).astype(np.float32))
            W["wte.weight"][rid] = row
            added.append({"id": rid, "content": t, "pieces": enc.tokens, "piece_ids": enc.ids,
                          "row_norm": round(float(np.linalg.norm(row)), 4)})
            tok_json["added_tokens"].append({"id": rid, "content": t, "single_word": False, "lstrip": False,
                                             "rstrip": False, "normalized": False, "special": True})
            print(f"new token {t!r} -> id {rid}, row = mean of {enc.tokens} (norm {np.linalg.norm(row):.4f})")
        os.makedirs(a.out, exist_ok=True)
        json.dump(tok_json, open(os.path.join(a.out, "tokenizer.json"), "w"), ensure_ascii=False)
        chk = tokenizers.Tokenizer.from_file(os.path.join(a.out, "tokenizer.json"))
        for x in added:
            assert chk.token_to_id(x["content"]) == x["id"]
            assert chk.encode(x["content"], add_special_tokens=False).ids == [x["id"]]
        print(f"extended tokenizer written: {chk.get_vocab_size(with_added_tokens=True)} ids")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "index.txt"), "w") as idx:
        for name, t in W.items():
            t = np.ascontiguousarray(t, dtype=np.float32)
            # 2026-11-20 step 9: the QK-norm scales are fp32 parameters in the trainer (ModelConfig::qk_dtype(), pinned by the
            # fused QK-norm + RoPE kernel), so an F32 export (the 350M's) holds non-bf16 values there and the trainer stores
            # them exactly; every other weight is a bf16 parameter (BLU_ALL_BF16=1) and must be bf16-exact. The trainer's
            # exact load check (every weight read back bit for bit) covers both. Old line (kept; it stopped the 350M
            # conversion at layers.0.q_gamma):
            #     assert np.array_equal(round_bf16(t), t), f"{name}: not bf16-exact"   # the trainer's exact check relies on it
            if not name.endswith(("q_gamma", "k_gamma")):
                assert np.array_equal(round_bf16(t), t), f"{name}: not bf16-exact"   # the trainer's exact check relies on it
            t.tofile(os.path.join(a.out, name + ".bin"))
            idx.write(name + " " + " ".join(str(s) for s in t.shape) + "\n")
    info = {"source": os.path.abspath(st), "source_sha256": hashlib.sha256(open(st, "rb").read()).hexdigest(),
            "training_meta": meta, "tensors": len(W), "new_tokens": added, "layout": "BluTrain in_out, fp32",
            "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "swa_window": swa_w, "swa_global_every": swa_e}   # 2026-11-20 step 9 (SWA in SFT): read by the SFT trainer's start-up check
    json.dump(info, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"wrote {len(W)} weights to {a.out} ({sum(t.size for t in W.values()):,} values)")

    if a.check:
        import tokenizers
        tk = tokenizers.Tokenizer.from_file(os.path.join(a.hf_dir, "tokenizer.json"))
        variants = [("as built", {}), ("gate/up swapped", {"swap_gate_up": True}),
                    ("GPT-J interleaved RoPE", {"interleaved": True}), ("no embed scale", {"scale": False})]
        print("reference forward, next-token loss (nats) per text; the 'as built' row must be the lowest")
        for name, kw in variants:
            losses = []
            for txt in CHECK_TEXTS:
                ids = np.asarray([0] + tk.encode(txt, add_special_tokens=False).ids)   # every doc starts with id 0
                # 2026-11-20 step 9 (SWA in SFT): with the model's window. Old line (kept): losses.append(ref_loss(W, m, ids, **kw))
                losses.append(ref_loss(W, m, ids, window=swa_w, global_every=swa_e, **kw))
            print(f"  {name:24s} " + "  ".join(f"{x:6.3f}" for x in losses) + f"   mean {np.mean(losses):6.3f}")


if __name__ == "__main__":
    main()
