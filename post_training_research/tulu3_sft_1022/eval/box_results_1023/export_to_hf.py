# Export a BluTrain SFT checkpoint to a Hugging Face model folder (Qwen3ForCausalLM layout), for evaluation (2026-10-23).
#
# Input : a DCP checkpoint folder (metadata.bin + data_rank_0.bin), e.g. runs/tulu3_sft_1022/checkpoints/blumodel_step_11590.
# Output: config.json, generation_config.json, model.safetensors, the GPT-2 tokenizer with our chat template, export_info.json.
#
# Why Qwen3: our block is Llama-style + per-head QK RMSNorm applied BEFORE RoPE, exactly Qwen3's layout
# (bluscript_sft.cpp:5-12; QK-norm gamma before the rotation: FusedRMSRoPE_bf16.cu `ys = v * rstd * g`, then rotate):
#   RMSNorm y = x * rsqrt(mean(x^2) + eps) * gamma (LayerNormKernels.cu:346-358) = HF Qwen3RMSNorm;
#   GQA 18 / 6 heads x 128, NeoX (half-split) RoPE theta 50000 = HF rotate_half; SwiGLU gate_up laid out [gate | up]
#   (bluscript_sft.cpp:701-705); no biases; final RMSNorm; lm_head tied to the embedding; vocab 50304 (GPT-2 50257 padded).
# One difference: HF Qwen3 uses ONE rms_norm_eps for all norms; ours is 1e-5 for the block norms and 1e-6 for the QK-norm.
#   config.json gets 1e-5; verify_export.py measures the effect of the QK-norm eps (it can set 1e-6 exactly).
#
# Our checkpoint layout (read from the code, checked here):
#   metadata.bin = CheckpointMetadata (dist/Distributed_Checkpointing/include/Metadata.h): magic 0x4D504344, version 3..5,
#     epoch, step, loss, per-rank RNG states, optimizer config, then every tensor: name, global shape, dtype, replicated,
#     shards (rank, file, file offset, offsets, shape, [v4+] checksum, [v5] data/padded length). Parsed to the last byte.
#   The model weights are "param_000000" .. "param_000281" in model.parameters() order (bluscript_sft.cpp:2143-2147).
#     parameters() returns a module's OWN parameters first, then its children's, each in registration order
#     (nn/NN.cpp:21-29). CausalGQA registers the children norm, wQ, wK, wV, wO and the own parameters q_gamma, k_gamma
#     (bluscript_sft.cpp:597-601); SwiGLUMLP the children norm, gate_up, down (723-725); per layer attention then MLP
#     (773-774); then wte and the final norm (785-786). So per layer i: 10*i + 0 q_gamma, 1 k_gamma, 2 attention RMSNorm,
#     3 wQ, 4 wK, 5 wV, 6 wO, 7 MLP RMSNorm, 8 gate_up, 9 down; then 280 wte, 281 final norm. Checked against the
#     shapes in metadata.bin of blumodel_step_11590 (same pattern in all 28 layers; q/k gammas fp32, the rest bf16).
#     Shapes cannot tell q_gamma from k_gamma or wK from wV; verify_export.py's val loss would expose a swap.
#   Our Linear stores W as [in, out] and computes y = x @ W (nn/NN.cpp:239, :270); HF stores [out, in] -> transposed.
#
# usage: python export_to_hf.py <checkpoint dir> <output dir> [--layers 28] [--rope-theta 50000]
import argparse, hashlib, json, os, struct
import numpy as np
import torch
from safetensors.torch import save_file

DTYPES = {8: (np.uint16, torch.bfloat16, "bfloat16"), 10: (np.float32, torch.float32, "float32")}   # Dtype.h enum
MAGIC = 0x4D504344


def read_metadata(path):
    b = open(path, "rb").read()
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o)
        o += struct.calcsize("<" + fmt)
        return v

    def rd_str():
        nonlocal o
        (n,) = rd("Q")
        s = b[o:o + n].decode()
        o += n
        return s

    magic, version = rd("II")
    assert magic == MAGIC, f"not a BluTrain checkpoint: magic {magic:#x}"
    assert 3 <= version <= 5, f"unknown metadata version {version}"
    epoch, step, loss = rd("iif")
    (n_rng,) = rd("Q")
    for _ in range(n_rng):                                  # rank, cpu_state blob, has_gpu, gpu seed, gpu offset
        rd("i"); rd_str(); rd("I"); rd("QQ")
    (n_opt,) = rd("Q")
    opt = {}
    for _ in range(n_opt):
        k = rd_str(); (v,) = rd("f"); opt[k] = v
    (n_t,) = rd("Q")
    tensors = {}
    for _ in range(n_t):
        name = rd_str()
        (nd,) = rd("Q"); shape = rd(f"{nd}q")
        (dt,) = rd("i"); (rep,) = rd("I"); (ns,) = rd("Q")
        shards = []
        for _ in range(ns):
            (rank,) = rd("i"); file = rd_str(); (foff,) = rd("Q")
            (no,) = rd("Q"); offs = rd(f"{no}q")
            (nsh,) = rd("Q"); sshape = rd(f"{nsh}q")
            if version >= 4: rd("I"); rd("Q")
            dlen = rd("QQ")[0] if version >= 5 else None
            shards.append(dict(rank=rank, file=file, offset=foff, offsets=offs, shape=sshape, data_length=dlen))
        tensors[name] = dict(shape=tuple(shape), dtype=dt, replicated=bool(rep), shards=shards)
    assert o == len(b), f"metadata parsed {o} of {len(b)} bytes: the format is not what this reader expects"
    return dict(version=version, epoch=epoch, step=step, loss=loss, opt=opt, tensors=tensors)


def read_tensor(ckpt_dir, t):
    assert len(t["shards"]) == 1, "only single-rank checkpoints (our 1-GPU run) are handled"
    s = t["shards"][0]
    assert tuple(s["shape"]) == t["shape"] and all(x == 0 for x in s["offsets"]), "the one shard must be the whole tensor"
    np_dt, torch_dt, _ = DTYPES[t["dtype"]]
    n = int(np.prod(t["shape"]))
    nbytes = n * np.dtype(np_dt).itemsize
    if s["data_length"] is not None:
        assert s["data_length"] >= nbytes, (s["data_length"], nbytes)
    f = s["file"] if os.path.isabs(s["file"]) and os.path.exists(s["file"]) else os.path.join(ckpt_dir, os.path.basename(s["file"]))
    a = np.fromfile(f, dtype=np_dt, count=n, offset=s["offset"]).reshape(t["shape"])
    assert a.size == n, f"short read from {f}"
    if np_dt == np.uint16:
        return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(a.copy())


CHAT_TEMPLATE = (   # tulu-plain-v1 (sft/prepare_sft_data.py:16-17, 48-75): markers are plain text; answers end with <|endoftext|>
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>\n' + m['content'] }}"
    "{% if m['role'] == 'assistant' %}{{ '<|endoftext|>' }}{% else %}{{ '\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=28); ap.add_argument("--rope-theta", type=float, default=50000.0)
    ap.add_argument("--tokenizer", default="openai-community/gpt2")
    a = ap.parse_args()
    d, Hq, Hkv, hd, F, V, L = 2304, 18, 6, 128, 6144, 50304, a.layers

    meta = read_metadata(os.path.join(a.ckpt, "metadata.bin"))
    params = sorted(k for k in meta["tensors"] if k.startswith("param_") and len(k) == 12 and k[6:].isdigit())
    print(f"metadata v{meta['version']}: step {meta['step']}, epoch {meta['epoch']}, loss {meta['loss']:.6f}, "
          f"{len(meta['tensors'])} tensors, {len(params)} model weights")
    assert len(params) == 10 * L + 2, f"expected {10 * L + 2} model weights for {L} layers, found {len(params)}"

    want = []   # (param index, HF name, expected [in,out]-style shape of OUR tensor, how to convert)
    for i in range(L):
        p, h = 10 * i, f"model.layers.{i}."
        want += [(p + 0, h + "self_attn.q_norm.weight", (hd,), "copy"),
                 (p + 1, h + "self_attn.k_norm.weight", (hd,), "copy"),
                 (p + 2, h + "input_layernorm.weight", (d,), "copy"),
                 (p + 3, h + "self_attn.q_proj.weight", (d, Hq * hd), "T"),
                 (p + 4, h + "self_attn.k_proj.weight", (d, Hkv * hd), "T"),
                 (p + 5, h + "self_attn.v_proj.weight", (d, Hkv * hd), "T"),
                 (p + 6, h + "self_attn.o_proj.weight", (Hq * hd, d), "T"),
                 (p + 7, h + "post_attention_layernorm.weight", (d,), "copy"),
                 (p + 8, h + "mlp.", (d, 2 * F), "gate_up"),
                 (p + 9, h + "mlp.down_proj.weight", (F, d), "T")]
    want += [(10 * L, "model.embed_tokens.weight", (V, d), "copy"), (10 * L + 1, "model.norm.weight", (d,), "copy")]

    sd, dtypes_seen = {}, {}
    for idx, name, shape, how in want:
        t = meta["tensors"][f"param_{idx:06d}"]
        assert t["shape"] == shape, f"param_{idx:06d} ({name}): shape {t['shape']}, expected {shape}"
        w = read_tensor(a.ckpt, t)
        assert torch.isfinite(w.float()).all(), f"param_{idx:06d} has NaN/Inf"
        dtypes_seen[DTYPES[t["dtype"]][2]] = dtypes_seen.get(DTYPES[t["dtype"]][2], 0) + 1
        if how == "copy":
            sd[name] = w.contiguous()
        elif how == "T":
            sd[name] = w.t().contiguous()
        else:   # gate_up [d, 2F] = [gate | up] along the output columns
            sd[name + "gate_proj.weight"] = w[:, :F].t().contiguous()
            sd[name + "up_proj.weight"] = w[:, F:].t().contiguous()
    print(f"weights read: {len(sd)} HF tensors, dtypes {dtypes_seen}; "
          f"layer-0 input norm mean {sd['model.layers.0.input_layernorm.weight'].float().mean():.4f}, "
          f"q_norm mean {sd['model.layers.0.self_attn.q_norm.weight'].float().mean():.4f}")

    os.makedirs(a.out, exist_ok=True)
    save_file(sd, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    cfg = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
           "vocab_size": V, "hidden_size": d, "intermediate_size": F, "num_hidden_layers": L,
           "num_attention_heads": Hq, "num_key_value_heads": Hkv, "head_dim": hd, "hidden_act": "silu",
           "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": a.rope_theta, "rope_scaling": None,
           "attention_bias": False, "attention_dropout": 0.0, "tie_word_embeddings": True,
           "use_sliding_window": False, "sliding_window": None, "max_window_layers": L, "initializer_range": 0.02,
           "use_cache": True, "torch_dtype": "bfloat16",
           "bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    # no max_new_tokens here: it would override a benchmark's own limit (lm-eval passes max_length = prompt + max_gen_toks;
    # a max_new_tokens of 2048 here won over IFEval's 1280 in the 10-23 smoke test)
    json.dump({"bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256, "do_sample": False},
              open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    assert tok.eos_token_id == 50256 and len(tok) == 50257, (tok.eos_token_id, len(tok))
    tok.chat_template = CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    tok.save_pretrained(a.out)

    md5 = {}
    for fn in ("metadata.bin",):
        md5[fn] = hashlib.md5(open(os.path.join(a.ckpt, fn), "rb").read()).hexdigest()
    json.dump({"source": os.path.abspath(a.ckpt), "source_md5": md5, "meta_step": meta["step"], "meta_epoch": meta["epoch"],
               "meta_loss": meta["loss"], "layers": L, "rope_theta": a.rope_theta, "template": "tulu-plain-v1",
               "qk_norm_eps_in_training": 1e-6, "block_norm_eps_in_training": 1e-5},
              open(os.path.join(a.out, "export_info.json"), "w"), indent=2)
    print(f"wrote {a.out}: model.safetensors ({os.path.getsize(os.path.join(a.out, 'model.safetensors')) / 1e9:.2f} GB), "
          f"config.json, tokenizer + chat template")


if __name__ == "__main__":
    main()
