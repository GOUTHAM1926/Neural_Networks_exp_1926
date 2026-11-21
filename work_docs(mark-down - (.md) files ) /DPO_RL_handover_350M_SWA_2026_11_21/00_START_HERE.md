# Handover to the DPO and RL teams: the 350M SFT checkpoints (sliding-window study) and how our SFT works

Written 2026-11-21 by the SFT side (BluTrain). For the DPO team and the RL team, who continue from these checkpoints.
Everything needed to start is in this folder: which checkpoint to take, where to download it and how to check it, how
to load it, how to continue training it in BluTrain, the two window rules that both teams must follow, and the full
documentation of our SFT (data, trainer, kernels, flags, context parallelism, sliding window, procedures, results).

---

## 1. Which checkpoint to take

Four 350M checkpoints, one per arm of a controlled 2 x 2 study (same data, same settings, same code; only the
attention changes):

| Arm | Pretraining | SFT | Folder in the bucket | Use it for |
|---|---|---|---|---|
| A | full attention (`kv4`) | full attention | `swa350m_sft_A_full_full` | the full-attention reference |
| B | full attention (`kv4`) | sliding window | `swa350m_sft_B_full_window` | ablation only (a full-attention model switched to the window in SFT) |
| **C** | **sliding window (`kv4_swa3`)** | **sliding window** | **`swa350m_sft_C_window_window`** | **the main checkpoint: the 350M window model** |
| D | sliding window (`kv4_swa3`) | full attention | `swa350m_sft_D_window_full` | ablation only (the window skipped in SFT) |

**Take C** unless you want the full-attention reference (A). C is pretrained with the window and fine-tuned with it
(the published practice); its val loss is the same as D's and its inference KV cache is 44% of full attention's at
4,096 tokens (27% at 32,768). Details and results: `01_checkpoints.md`, `07_results.md`.

Bucket (public, no login needed): `hf://buckets/gautam1926/checkpoints_sft`, folder
`checkpoints_sft_swa_experimentations/`. Each checkpoint folder holds:
- `hf/`: the Hugging Face transformers model (`Qwen3ForCausalLM`, bf16, the window in `config.json`, tokenizer, chat
  template): for RL frameworks, evaluation, serving;
- `blumodel_step_5652/`: the BluTrain checkpoint at the last SFT step (weights + optimizer state + the window record
  `model_config.json`): for continuing in BluTrain (DPO);
- `README.md` (the model card) and `MD5SUMS.txt`.

## 2. Quick start

**Download and check** (example: C):

```
hf sync hf://buckets/gautam1926/checkpoints_sft/checkpoints_sft_swa_experimentations/swa350m_sft_C_window_window ./C
cd C && md5sum -c MD5SUMS.txt          # every line must say OK
```

(`hf` comes with `pip install huggingface_hub`; single files also work with `curl -L -O
https://huggingface.co/buckets/gautam1926/checkpoints_sft/resolve/checkpoints_sft_swa_experimentations/swa350m_sft_C_window_window/<path>`.)

**Load in Hugging Face transformers** (5.21.0 was used; it honours the per-layer window):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("./C/hf")
model = AutoModelForCausalLM.from_pretrained("./C/hf", torch_dtype=torch.bfloat16).cuda()
ids = tok.apply_chat_template([{"role": "user", "content": "Hello!"}], add_generation_prompt=True,
                              return_tensors="pt").cuda()
print(tok.decode(model.generate(ids, max_new_tokens=64, eos_token_id=0)[0, ids.shape[1]:], skip_special_tokens=True))
```

**Continue in BluTrain (DPO)** from C (all of these are required; `04_flags_and_settings.md` explains each):

```
BLU_INIT_FROM=<path>/C/blumodel_step_5652  BLU_N_LAYERS=28  BLU_ROPE_THETA=50000
BLU_D_MODEL=1024  BLU_Q_HEADS=8  BLU_KV_HEADS=4  BLU_FFN_HIDDEN=2560  BLU_VOCAB=49280  BLU_EMBED_SCALE=1
BLU_SWA_WINDOW=1024  BLU_SWA_GLOBAL_EVERY=4
```

The DPO reference model is the same checkpoint. The chat format the DPO / RL data must use is in
`01_checkpoints.md` section 4.

## 3. The two window rules (C and B; both teams)

1. **Training keeps the window.** Every forward and backward of the policy (and of the reference model) runs layer i
   with a window of 1,024 keys unless (i + 1) % 4 == 0 (then full causal attention): 21 windowed layers, 7 global,
   `LLLG` x 7. In BluTrain this is `BLU_SWA_WINDOW=1024 BLU_SWA_GLOBAL_EVERY=4`; code that calls our attention kernels
   directly passes `window = 1024` or `0` per layer (`05_cp_and_swa.md`). The trainer refuses a window that differs
   from the checkpoint's record (`model_config.json`) unless `BLU_SWA_MISMATCH_OK=1`.
2. **Generation keeps the window** (RL rollouts, sampling DPO pairs from the policy, evaluation, serving). Hugging Face
   transformers honours the per-layer window of the config. **vLLM and SGLang do not** (for Qwen3 configs they run
   full attention in every layer, without any error): for C this raises the loss past position 2,048 from 1.804 to
   2.031. Use transformers, or patch the engine for per-layer windows and check its logits against transformers on a
   prompt longer than 1,024 tokens first.

For A and D (full attention in every layer) neither rule applies; any engine serves them as trained.

## 4. What is in this package

| File | What it answers |
|---|---|
| `00_START_HERE.md` | this page |
| `01_checkpoints.md` | the four checkpoints in full: files, sizes, md5, the model, tokenizer and chat format, base models, SFT data and settings, how each run went, how to load and continue |
| `02_how_sft_is_wired.md` | how our SFT works end to end: data preparation, the packed files, the loader, the trainer, the kernels (packed attention, window, RoPE per chat, loss), how the binary links the libraries, checkpoints and exports |
| `03_sft_vs_pretraining_trainer.md` | every difference between the SFT trainer and the pretraining trainer, kernel side and non-kernel side, and why |
| `04_flags_and_settings.md` | every setting and flag: `run_sft.sh` variables, trainer `BLU_*` flags, kernel flags, how to set a kernel flag through `run_sft.sh`, the exact settings of our runs |
| `05_cp_and_swa.md` | context parallelism (Ulysses) and sliding-window attention: how they work, alone and together, limits, what changes for DPO and RL |
| `06_procedures.md` | step-by-step procedures: access, build, data, base conversion, SFT runs (1 GPU, CP, window), monitoring, resume, export, checks, evaluation, starting DPO, RL rollouts, pitfalls |
| `07_results.md` | the results: the 2 x 2 (val loss, groups, serving, benchmarks), the kernel race against FA3 / FA4 / FA2 / FlexAttention, speed, the CP memory ladder |
| `docs/` | the detailed documents these pages point to (the BluTrain SFT guide, the kernel-difference study, the sliding-window analysis, the CP documents, the benchmarks, the optimization history, the 1.7B DPO handover note) |

## 5. Where the code is

| Repository | Branch | Commit | What |
|---|---|---|---|
| `BlubridgeAI/BluTrain` | `dist_side_gautam_lottn` | `651f9c4` (code; docs `bfc712b`) | trainers, data preparation, loader, scripts, export / convert tools |
| `BlubridgeAI/Tensor-Implementations` (submodule) | `dist_side_gautam_lottn` | `dd5b61a` | the kernels (attention with packed rows and window, RoPE, loss, ...) |
| BluBridge-BLAS (submodule of Tensor-Implementations) | - | `cfa0e92` | the GEMM library |

All four checkpoints were trained with this code (trainer binary md5 `3af3e48d`). Access and cloning:
`docs/how_to_handle_sft.md` sections 1-2.

**Read next:** `01_checkpoints.md`, then `06_procedures.md`. For the why behind each part: `02`, `03`, `05` and `docs/`.
