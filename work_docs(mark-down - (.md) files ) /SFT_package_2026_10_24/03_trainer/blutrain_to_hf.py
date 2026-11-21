#!/usr/bin/env python3
"""Turn a BluTrain weight export (BLU_EXPORT=<file>.safetensors: fp32, BluTrain names, [in, out] Linears) of the 135M
model into a Hugging Face Qwen3 model folder, for the benchmark tools (2026-10-28). The inverse of hf_to_blutrain.py,
with the conventions of Navin's export of the base (navin_135m_checkpoints_tokenizer/):
  - embed_tokens = bf16(wte x sqrt(d_model)) (HF Qwen3 has no embedding scale, so it is folded in), lm_head = bf16(wte),
    tie_word_embeddings false; every Linear transposed back to [out, in]; gate_up split into gate_proj / up_proj;
  - config.json / generation_config.json / special_tokens_map.json from --base-hf (the base model's folder);
  - tokenizer.json from --tokenizer (the base one for arm A, the extended one for arm B), and our chat template
    (tulu-plain-v1 / tulu-special-v1 render the same TEXT; arm B's tokenizer turns the markers into single ids):
      <|endoftext|><|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>\\n{answer}<|endoftext|><|user|>\\n...
    generation prompt "<|assistant|>\\n", generation stops at <|endoftext|> (id 0). Message text is trimmed at both
    ends, as prepare_sft_data.py does for this tokenizer (so the template and the training ids match token for token).
--format-version vN (the component-9 study, 2026-10-30): --tokenizer must be chat_formats.py's full tokenizer (the
version's init folder holds it; checked); generation_config.json stops on the version's stop ids
(chat_formats.end_ids); the chat template only raises an error, because a text template cannot give the training ids
of every version (plan 4.48 rule 9): the study evaluates through sft/fmt_lm_eval.py, which builds the ids itself.
--check-against <model.safetensors>: compare every output tensor with that file bit for bit (the round-trip test:
Navin's file -> hf_to_blutrain.py -> an export -> this script must give Navin's file back).
2026-11-20 step 9 (SWA in SFT): the sliding window of the RUN (the export's __metadata__ swa_window / swa_global_every, written by
bluscript_sft.cpp since 2026-11-20) is written into config.json (use_sliding_window, sliding_window, layer_types,
max_window_layers), whatever --base-hf's config.json says; an export without these fields keeps the base's config (a
line says so).
"""
import argparse
import json
import os
import shutil
import struct

import numpy as np

CHAT_TEMPLATE = (
    "{{ '<|endoftext|>' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|system|>\n' + (message['content'] | trim) + '\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|user|>\n' + (message['content'] | trim) + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + (message['content'] | trim) + '<|endoftext|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}")


def round_bf16_u16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        meta = head.pop("__metadata__", {})
        base, out = 8 + n, {}
        for name, e in head.items():
            a, b = e["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if e["dtype"] == "F32":
                out[name] = np.frombuffer(raw, dtype=np.float32).reshape(e["shape"])
            elif e["dtype"] == "BF16":
                out[name] = np.frombuffer(raw, dtype=np.uint16).reshape(e["shape"])
            else:
                raise SystemExit(f"{name}: dtype {e['dtype']}")
    return out, meta


def write_st_bf16(path, tensors):
    """tensors: name -> uint16 array (bf16 bits). Same layout as torch's save_file: sorted header, 8-byte aligned."""
    hdr, off, order = {}, 0, sorted(tensors)
    for k in order:
        nb = tensors[k].size * 2
        hdr[k] = {"dtype": "BF16", "shape": list(tensors[k].shape), "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "pt"}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in order:
            f.write(np.ascontiguousarray(tensors[k]).tobytes())
    os.replace(path + ".tmp", path)


def hf_window_fields(W, every, L):
    """2026-11-20 step 9 (SWA in SFT): the Hugging Face Qwen3 window fields of a model with window W (0 = none), layer i global iff
    (i + 1) % every == 0 (the trainers' rule). HF reads layer_types per layer and drops the window entirely unless
    use_sliding_window is true (configuration_qwen3.py); vLLM / SGLang do NOT honour layer_types for Qwen3 (they serve
    every layer with full attention: SWA analysis doc section 11.5, serving checklist)."""
    if W <= 0:
        return {"use_sliding_window": False, "sliding_window": None, "max_window_layers": L,
                "layer_types": ["full_attention"] * L}
    return {"use_sliding_window": True, "sliding_window": int(W), "max_window_layers": L,
            "layer_types": ["full_attention" if (i + 1) % every == 0 else "sliding_attention" for i in range(L)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="the BLU_EXPORT safetensors of a run")
    ap.add_argument("--base-hf", required=True, help="the base model's HF folder (config.json etc.)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the run (base for arm A, extended for B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-against", default=None)
    ap.add_argument("--format-version", default="", help="component-9 study version v1..v22 (see above)")
    a = ap.parse_args()

    W, meta = read_st(a.export)
    assert meta.get("format") == "blutrain" and meta.get("linear_layout") == "in_out", meta
    assert meta.get("embed_scale") == "1" and meta.get("weight_tying") == "1", meta
    d, L, F = int(meta["d_model"]), int(meta["n_layers"]), int(meta["ffn_hidden"])
    W = {k: np.asarray(v, dtype=np.float32) for k, v in W.items()}
    H = {"lm_head.weight": round_bf16_u16(W["wte.weight"]),
         "model.embed_tokens.weight": round_bf16_u16(W["wte.weight"] * np.float32(np.sqrt(float(d)))),
         "model.norm.weight": round_bf16_u16(W["norm_f.weight"])}
    for i in range(L):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        H[p + "input_layernorm.weight"] = round_bf16_u16(W[q + "attn_norm.weight"])
        for a_, b_ in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"), ("wo", "o_proj")):
            H[p + "self_attn." + b_ + ".weight"] = round_bf16_u16(W[q + a_ + ".weight"].T)
        H[p + "self_attn.q_norm.weight"] = round_bf16_u16(W[q + "q_gamma"])
        H[p + "self_attn.k_norm.weight"] = round_bf16_u16(W[q + "k_gamma"])
        H[p + "post_attention_layernorm.weight"] = round_bf16_u16(W[q + "mlp_norm.weight"])
        gu = W[q + "gate_up.weight"]
        assert gu.shape[1] == 2 * F
        H[p + "mlp.gate_proj.weight"] = round_bf16_u16(gu[:, :F].T)
        H[p + "mlp.up_proj.weight"] = round_bf16_u16(gu[:, F:].T)
        H[p + "mlp.down_proj.weight"] = round_bf16_u16(W[q + "down.weight"].T)
    assert len(H) == 11 * L + 3

    os.makedirs(a.out, exist_ok=True)
    write_st_bf16(os.path.join(a.out, "model.safetensors"), H)
    for fn in ("config.json", "generation_config.json", "special_tokens_map.json"):
        shutil.copy(os.path.join(a.base_hf, fn), os.path.join(a.out, fn))
    # 2026-11-20 step 9 (SWA in SFT): the run's window into config.json (the base's config may have none, or another one on purpose)
    cfg_path = os.path.join(a.out, "config.json")
    cfg = json.load(open(cfg_path))
    before = {k: cfg.get(k) for k in ("use_sliding_window", "sliding_window")}
    if "swa_window" in meta:
        cfg.update(hf_window_fields(int(meta["swa_window"]), int(meta.get("swa_global_every", 4)), L))
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        print(f"config.json window from the run's export: use_sliding_window {cfg['use_sliding_window']}, "
              f"sliding_window {cfg['sliding_window']}, layer_types "
              + "".join("G" if t == "full_attention" else "L" for t in cfg["layer_types"]) + f" (base config: {before})")
    else:
        print(f"note: the export has no window fields (made before 2026-11-20): config.json kept from --base-hf {before}")
    shutil.copy(a.tokenizer, os.path.join(a.out, "tokenizer.json"))
    tc = json.load(open(os.path.join(a.base_hf, "tokenizer_config.json")))
    tc["chat_template"] = CHAT_TEMPLATE
    if a.format_version:
        import chat_formats as cf
        f = cf.FORMATS[a.format_version]
        tok = json.load(open(a.tokenizer))
        have = {t["content"]: (t["id"], t["special"]) for t in tok["added_tokens"]}
        assert all(have.get(n) == (i, True) for n, i in cf.NEW_ID.items()), "--tokenizer is not the full tokenizer"
        stop = cf.end_ids(f)[2]
        gc = json.load(open(os.path.join(a.out, "generation_config.json")))
        gc["eos_token_id"] = stop
        json.dump(gc, open(os.path.join(a.out, "generation_config.json"), "w"), indent=1)
        tc["chat_template"] = ("{{ raise_exception('component-9 study version " + f.name + ": evaluate through "
                               "sft/fmt_lm_eval.py (exact training ids), not through a text chat template') }}")
        print(f"{f.name}: generation stops on {stop}; text chat template disabled")
    json.dump(tc, open(os.path.join(a.out, "tokenizer_config.json"), "w"), indent=1)
    json.dump({"export_metadata": meta, "source": os.path.abspath(a.export), "format_version": a.format_version or None},
              open(os.path.join(a.out, "blutrain_export_meta.json"), "w"), indent=1)
    print(f"wrote {a.out}: {len(H)} tensors (bf16), config + tokenizer + chat template")

    if a.check_against:
        ref, _ = read_st(a.check_against)
        bad = [k for k in H if k not in ref or ref[k].shape != H[k].shape or not np.array_equal(ref[k], H[k])]
        print(f"round trip vs {a.check_against}: {len(H) - len(bad)}/{len(H)} tensors bit-identical"
              + (f"; first differing: {bad[:3]}" if bad else ""))
        assert not bad and len(ref) == len(H)


if __name__ == "__main__":
    main()
