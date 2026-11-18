# Proof that an exported model (export_to_hf.py) IS our checkpoint (2026-10-23): it recomputes the val loss of the
# 2,000 held-out chats with the Hugging Face model and compares it with what our trainer printed for the same weights.
#
#   Val data = sft_data/tulu3_noaya/val: inputs.bin / labels.bin uint16 [320, 4096], seq_lengths.bin int32 (the chats of
#   every row, in row order), seq_offsets.bin int64 [321] (sft/prepare_sft_data.py:4-13, 136-152). Each chat sits in its
#   row as inputs[t:t+n], labels[t:t+n], already labelled on its own; 65535 = no loss. Our trainer attends each chat only
#   to itself with positions from 0, which is what running each chat ALONE through the HF model does.
#   The trainer's val is the sum of -ln p(label) over all 898,276 trained tokens divided once by that count
#   (bluscript_sft.cpp:2304-2329); computed here the same way, in float64.
#   Expected (the trainer's [val] lines; val at step N = the weights after N updates = checkpoint blumodel_step_N):
#     blumodel_step_6000  -> 0.859895 (run.log, [val] step 6000: the SAME weights, so the match must be close)
#     blumodel_step_11590 -> 0.850100 is [val] step 11589, one update before the checkpoint (lr 5e-6 there)
#   Differences come only from different kernels (bf16 matmuls, attention) and from the QK-norm eps (HF uses rms_norm_eps
#   1e-5 for q_norm/k_norm, our trainer 1e-6): the script runs the val set with 1e-6 (exactly ours) and then with 1e-5
#   (what lm-eval / any HF loader will use), so both differences are measured.
#
# usage: python verify_export.py <export dir> <val dir> [--expect 0.859895] [--limit-rows N]
import argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("val")
ap.add_argument("--expect", type=float, default=None); ap.add_argument("--limit-rows", type=int, default=0)
a = ap.parse_args()
T, IGNORE = 4096, 65535

inp = np.fromfile(f"{a.val}/inputs.bin", np.uint16).reshape(-1, T)
lab = np.fromfile(f"{a.val}/labels.bin", np.uint16).reshape(-1, T)
lens = np.fromfile(f"{a.val}/seq_lengths.bin", np.int32)
offs = np.fromfile(f"{a.val}/seq_offsets.bin", np.int64)
N = inp.shape[0] if not a.limit_rows else a.limit_rows
assert len(offs) == inp.shape[0] + 1 and offs[-1] == len(lens), "seq_offsets / seq_lengths do not match the rows"
print(f"val: {inp.shape[0]} rows, {len(lens)} chats, {int((lab != IGNORE).sum()):,} trained tokens; using {N} rows")

model = AutoModelForCausalLM.from_pretrained(a.export, dtype=torch.bfloat16).cuda().eval()   # transformers 5.x: dtype=
print(f"model: {type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B parameters (tied embedding counted once)")


def set_qk_eps(eps):
    for layer in model.model.layers:
        layer.self_attn.q_norm.variance_epsilon = eps
        layer.self_attn.k_norm.variance_epsilon = eps


def run(eps):
    set_qk_eps(eps)
    tot, cnt, pad_mass, t0 = 0.0, 0, 0.0, time.time()
    with torch.no_grad():
        for r in range(N):
            t = 0
            for n in lens[offs[r]:offs[r + 1]]:
                n = int(n)
                x = torch.from_numpy(inp[r, t:t + n].astype(np.int64))[None].cuda()
                y = torch.from_numpy(lab[r, t:t + n].astype(np.int64)).cuda()
                m = y != IGNORE
                if m.any():
                    lp = torch.log_softmax(model(x).logits[0].float(), dim=-1)[m]
                    tot += float(-lp.gather(1, y[m][:, None]).double().sum())
                    cnt += int(m.sum())
                    pad_mass = max(pad_mass, float(lp[:, 50257:].exp().sum(-1).max()))   # ids past GPT-2's 50257
                t += n
            assert (lab[r, t:] == IGNORE).all(), f"row {r}: labels after the last chat"
    return tot / cnt, cnt, pad_mass, time.time() - t0


for eps, what in [(1e-6, "QK-norm eps 1e-6 = exactly our trainer"), (1e-5, "QK-norm eps 1e-5 = config.json, what HF tools use")]:
    loss, cnt, pm, sec = run(eps)
    line = f"val loss {loss:.6f} over {cnt:,} trained tokens ({what}; {sec:.0f} s); largest probability on ids >= 50257: {pm:.2e}"
    if a.expect is not None:
        line += f" | trainer printed {a.expect:.6f}, difference {loss - a.expect:+.6f}"
    print(line)
