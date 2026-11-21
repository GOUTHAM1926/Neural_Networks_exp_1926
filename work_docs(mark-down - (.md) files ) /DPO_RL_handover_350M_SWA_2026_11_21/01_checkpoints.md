# The four 350M SFT checkpoints: files, model, data, settings, runs

Everything about the checkpoints of the 2 x 2 sliding-window study, in the order needed to use them. The bucket is
public (no login): `hf://buckets/gautam1926/checkpoints_sft`, folder `checkpoints_sft_swa_experimentations/`.

---

## 1. The checkpoints and their files

| Arm | Pretraining | SFT | Folder | Trained (IST) |
|---|---|---|---|---|
| A | full attention (`kv4`) | full attention | `swa350m_sft_A_full_full` | 2026-11-20 20:51-21:41, P035 GPU 5 |
| B | full attention (`kv4`) | window (on purpose, `BLU_SWA_MISMATCH_OK=1`) | `swa350m_sft_B_full_window` | 2026-11-21 12:23-13:11, P015 GPU 3 |
| **C** | **window (`kv4_swa3`)** | **window** | **`swa350m_sft_C_window_window`** | 2026-11-20 20:09-20:58, P035 GPU 7 |
| D | window (`kv4_swa3`) | full attention (on purpose, `BLU_SWA_MISMATCH_OK=1`) | `swa350m_sft_D_window_full` | 2026-11-20 20:51-21:41, P035 GPU 4 |

All four: the same binary (md5 `3af3e48d`: BluTrain `651f9c4` + Tensor-Implementations `dd5b61a`), data, settings and
step count. B ran on P015 (P035 was busy) with its inputs rebuilt there and checked against P035 by md5; the one
difference, P015's CUDA 13.0 runtime instead of 13.4, does not touch the model's math (`07_results.md` section 1).

Each folder (4.61 GB):

| Path | What | Size |
|---|---|---|
| `hf/model.safetensors` | the weights in Hugging Face names, bf16 (`embed_tokens` and `lm_head` separate: the input scale x32 is folded into `embed_tokens`) | 818.6 MB |
| `hf/config.json` | `Qwen3ForCausalLM` config: the shape and the attention layout (`use_sliding_window`, `sliding_window`, `layer_types`) | 1.4 kB |
| `hf/tokenizer.json`, `hf/tokenizer_config.json` (chat template), `hf/special_tokens_map.json`, `hf/generation_config.json` | tokenizer and chat format | 1.8 MB |
| `hf/blutrain_export_meta.json` | the trainer's export metadata (shape, rope_theta, embed scale, window, steps) | 0.5 kB |
| `blumodel_step_5652/data_rank_0.bin` + `metadata.bin` | BluTrain DCP checkpoint at the last step (5,652), saved by 1 GPU: the model weights **plus** the optimizer state (Muon momentum, AdamW moments, fp32 master weights) | 3.79 GB |
| `blumodel_step_5652/model_config.json` | the model record: shape, rope_theta, embed scale, tying, and the window (`swa_window` 1024 / 0, `swa_global_every` 4); the trainer checks it at start-up | 0.4 kB |
| `README.md` | the model card | - |
| `MD5SUMS.txt` | md5 of every file above | - |

Download a whole folder and check it:

```
hf sync hf://buckets/gautam1926/checkpoints_sft/checkpoints_sft_swa_experimentations/<folder> ./<folder>
cd <folder> && md5sum -c MD5SUMS.txt
```

or single files: `curl -L -O https://huggingface.co/buckets/gautam1926/checkpoints_sft/resolve/checkpoints_sft_swa_experimentations/<folder>/<path>`.
The upload was checked by downloading every file through these public links and comparing md5 with the lists below.

**`swa350m_sft_A_full_full`**

| File | Bytes | md5 |
| --- | --- | --- |
| `README.md` | - | `170ad7a34730c22d1f3cfb96dc7bd218` |
| `blumodel_step_5652/data_rank_0.bin` | 3,790,796,800 | `c07bbed438f8eda79b130427afa146f7` |
| `blumodel_step_5652/metadata.bin` | 109,061 | `7caf5c2d388d30c942fa0fdc7e284859` |
| `blumodel_step_5652/model_config.json` | 355 | `60fb23524157bbf066806a3a58fc3b99` |
| `hf/blutrain_export_meta.json` | 542 | `25b5345116aaf281144b22840b13dee8` |
| `hf/config.json` | 1,390 | `cac3b9b78e431f1434f355e766b2e914` |
| `hf/generation_config.json` | 82 | `0ab1e3fbf3a8228e2677c18bb783e49a` |
| `hf/model.safetensors` | 818,579,792 | `db226eca13a8a7ba22817dbfd4d3ea7c` |
| `hf/special_tokens_map.json` | 126 | `15a616d8d0294182442a510bd90fb3e4` |
| `hf/tokenizer.json` | 1,769,424 | `c56e5ce791e7b14131a94ba4aa32b686` |
| `hf/tokenizer_config.json` | 679 | `d9ef3898ec82a20c62833b4f274548ab` |

**`swa350m_sft_B_full_window`**

| File | Bytes | md5 |
| --- | --- | --- |
| `README.md` | - | `1723a81b530b8a72949a3088f9bbb01d` |
| `blumodel_step_5652/data_rank_0.bin` | 3,790,796,800 | `acde05685e53f2cfa912ae16fab447b0` |
| `blumodel_step_5652/metadata.bin` | 109,061 | `30baa7ce4b87c5c4cf4e44705e198518` |
| `blumodel_step_5652/model_config.json` | 358 | `7e55efd32c3fac1ae571d3e106588d8f` |
| `hf/blutrain_export_meta.json` | 544 | `ba628523c93ce3cb3d4fc1a2dabbcd02` |
| `hf/config.json` | 1,452 | `9f190caee8a17695fe7d20a016ef10bd` |
| `hf/generation_config.json` | 82 | `0ab1e3fbf3a8228e2677c18bb783e49a` |
| `hf/model.safetensors` | 818,579,792 | `8b2734b841835681e7709d8c4bbdecf7` |
| `hf/special_tokens_map.json` | 126 | `15a616d8d0294182442a510bd90fb3e4` |
| `hf/tokenizer.json` | 1,769,424 | `c56e5ce791e7b14131a94ba4aa32b686` |
| `hf/tokenizer_config.json` | 679 | `d9ef3898ec82a20c62833b4f274548ab` |

**`swa350m_sft_C_window_window`**

| File | Bytes | md5 |
| --- | --- | --- |
| `README.md` | - | `02ccfa5cf46dcf4c16020a38f78e07db` |
| `blumodel_step_5652/data_rank_0.bin` | 3,790,796,800 | `2868cf615d38c334116e010501aa3990` |
| `blumodel_step_5652/metadata.bin` | 109,061 | `ed5fab9b056c725c7f62acaf0cbd9524` |
| `blumodel_step_5652/model_config.json` | 358 | `7e55efd32c3fac1ae571d3e106588d8f` |
| `hf/blutrain_export_meta.json` | 545 | `54dece6145f2cc389f7c4fa9f28067eb` |
| `hf/config.json` | 1,452 | `9f190caee8a17695fe7d20a016ef10bd` |
| `hf/generation_config.json` | 82 | `0ab1e3fbf3a8228e2677c18bb783e49a` |
| `hf/model.safetensors` | 818,579,792 | `156ffa253edf12529e7a3862fdad89ee` |
| `hf/special_tokens_map.json` | 126 | `15a616d8d0294182442a510bd90fb3e4` |
| `hf/tokenizer.json` | 1,769,424 | `c56e5ce791e7b14131a94ba4aa32b686` |
| `hf/tokenizer_config.json` | 679 | `d9ef3898ec82a20c62833b4f274548ab` |

**`swa350m_sft_D_window_full`**

| File | Bytes | md5 |
| --- | --- | --- |
| `README.md` | - | `2a826027bc72d417bbaae7d2e441ae4e` |
| `blumodel_step_5652/data_rank_0.bin` | 3,790,796,800 | `ee3510c085d91af5183bddad9451d0f9` |
| `blumodel_step_5652/metadata.bin` | 109,061 | `268728010fe61a097c5302d19433a055` |
| `blumodel_step_5652/model_config.json` | 355 | `60fb23524157bbf066806a3a58fc3b99` |
| `hf/blutrain_export_meta.json` | 543 | `036ae2740748dd480dcbb5da53c44470` |
| `hf/config.json` | 1,390 | `cac3b9b78e431f1434f355e766b2e914` |
| `hf/generation_config.json` | 82 | `0ab1e3fbf3a8228e2677c18bb783e49a` |
| `hf/model.safetensors` | 818,579,792 | `b492b94cc2bb5470b7e82c9eca7cbdc1` |
| `hf/special_tokens_map.json` | 126 | `15a616d8d0294182442a510bd90fb3e4` |
| `hf/tokenizer.json` | 1,769,424 | `c56e5ce791e7b14131a94ba4aa32b686` |
| `hf/tokenizer_config.json` | 679 | `d9ef3898ec82a20c62833b4f274548ab` |

## 2. The model

| | |
|---|---|
| Parameters | 358.8M (embedding tied with the output layer in BluTrain, counted once); the HF export stores 409.3M (separate `embed_tokens` and `lm_head`) |
| Layers | 28 decoder blocks: RMSNorm -> attention -> residual, RMSNorm -> SwiGLU MLP -> residual; final RMSNorm |
| Width | d_model 1,024; SwiGLU hidden 2,560 (`gate_up` stored as [gate \| up] in BluTrain) |
| Attention | grouped-query: 8 query heads, 4 key/value heads (G = 2), head size 128, causal, no biases |
| QK-norm | per-head RMSNorm on q and k (gamma of size 128, eps 1e-6), before RoPE; the gammas are fp32 in BluTrain |
| RoPE | NeoX style (first half / second half), theta **50,000**, context 4,096 |
| Norms | RMSNorm `x * rsqrt(mean(x^2) + eps) * gamma`, eps 1e-5 |
| Embedding | input embedding x sqrt(1,024) = 32 (`BLU_EMBED_SCALE=1`); tied with the output layer |
| Vocab | 49,280 rows = 49,152 tokenizer ids padded; `<\|endoftext\|>` = id **0** = begin, end and padding |
| Attention layout, B and C | window 1,024 (the token itself and the 1,023 before it, inside its own chat) in layers where (i + 1) % 4 != 0; layers 3, 7, 11, 15, 19, 23, 27 global: `LLLGLLLGLLLGLLLGLLLGLLLGLLLG` |
| Attention layout, A and D | full causal attention in all 28 layers |
| Precision in training | all bf16 (`BLU_ALL_BF16=1`), fp32 optimizer state |

## 3. How to use them

**Hugging Face transformers** (`hf/`): `AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)`
(the config says float32; the weights are bf16). transformers 5.21.0 reads `layer_types` and runs the window per
layer. Generation: `eos_token_id=0`. The exports give the trainer's val loss: HF 1.178484 vs trainer 1.178506 (A),
1.176537 vs 1.176567 (D), 1.180420 vs 1.180413 (B), on the same rows.

**vLLM / SGLang**: they ignore the per-layer window of a Qwen3 config and run full attention everywhere (no error).
Fine for A and D; for C it costs +0.23 val loss past position 2,048. Patch them for per-layer windows, or use
transformers (`05_cp_and_swa.md` section 4).

**BluTrain** (`blumodel_step_5652/`): see `00_START_HERE.md` section 2 for the full line. `BLU_INIT_FROM` loads the
weights only (reads `metadata.bin` + `data_rank_0.bin`; the optimizer state is not used: DPO / RL start with a fresh
optimizer). Shape and theta must be given (the trainer's built-in model is a different one: 26 layers, d 2304, theta
10,000): a wrong layer count or width stops with a shape error, **a wrong theta or embed scale gives no error, only a
worse model**. For B and C also `BLU_SWA_WINDOW=1024 BLU_SWA_GLOBAL_EVERY=4`; the trainer compares them with
`model_config.json` and stops on a mismatch.

**From the HF folder into BluTrain** (if only `hf/` is at hand): `python sft/hf_to_blutrain.py --hf-dir hf/ --out raw/
--check`, then `BLU_INIT_FROM_RAW=raw/` instead of `BLU_INIT_FROM` (the converter reads the window from `config.json`
into `meta.json`; checked exact on the 350M bases: 284 files equal to the earlier conversion).

## 4. Tokenizer and chat format (DPO and RL data must use exactly this)

- Tokenizer: the 135M / 350M tokenizer (`tokenizer.json`, 49,152 ids, sha256 `b2b3ad98...`); `<|endoftext|>` = id 0
  is the begin, end and padding token. Vocab rows 49,280 (padded).
- Chat template `tulu-plain-v1` (in `tokenizer_config.json`):

```
<|endoftext|><|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n{next user turn}\n<|assistant|>\n...
```

- The role markers (`<|system|>`, `<|user|>`, `<|assistant|>`) are **plain text**, not special tokens. In our data
  each piece (marker, message, `\n`) is tokenized on its own (`sft/prepare_sft_data.py`), so a marker never merges with
  its neighbour; tokenizing the whole string at once (as `apply_chat_template(tokenize=True)` does) can differ at the
  joins. For training data, use the piecewise tokenization of `prepare_sft_data.py`.
- Every assistant answer ends with the end token 0; the model learned to stop there. Trained positions: the answer
  tokens + their end token; everything else (system, user, markers, padding) has label 65535 = ignored.
- For DPO: the prompt ends with `<|assistant|>\n`; the chosen and the rejected answer each end with `<|endoftext|>`.

## 5. The base models

| | `kv4` (A, B) | `kv4_swa3` (C, D) |
|---|---|---|
| What | in-house 350M pretraining screen run (Navin's `hp350m` series) | the same with the window: the chosen 350M configuration |
| Attention | full | window 1,024, every 4th layer global |
| Pretraining | 5.0B tokens, rows of 4,096, the same init, data and steps for both | same |
| Into BluTrain | Navin's HF export -> `sft/hf_to_blutrain.py --check` (numpy reference forward) -> `raw/` (`BLU_INIT_FROM_RAW`) | same; `meta.json` records window 1,024 / every 4 |
| Val loss at SFT step 0 (trainer) | 1.672250 (A, full); 1.678060 (B, with the window) | 1.668373 (C, window); 1.681077 (D, without the window) |

## 6. SFT data

| | |
|---|---|
| Dataset | `allenai/tulu-3-sft-olmo-2-mixture-0225`, revision `d91a0785ade02942520280fb484866fce41e448f` |
| Removed | the Aya part (`ai2-adapt-dev/tulu_v3.9_aya_100k`, multilingual) and the hard-coded identity chats (`olmo_hardcoded`) |
| Validation | 2,000 chats held out from the same mix (311 rows, 850,765 trained tokens), never trained on |
| Dropped | chats longer than 4,096 tokens (the only method: a chat is never split) |
| Training chats | 744,023, packed into **113,052 rows of 4,096 tokens** (First-Fit-Decreasing); each chat attends only to itself and its RoPE positions restart at 0 |
| Tokens per epoch | 463.1M positions, **311.6M trained (answer) tokens** |
| Chat length | 17.8% of the chats are longer than 1,024 tokens, 2.9% longer than 2,048 (the window changes only those) |
| Prep | `sft/prepare_sft_data.py --mix sft/mixes/m2_tulu.json --pack ffd --seq-len 4096 --template tulu-plain-v1 --tokenizer <tokenizer.json> --seed 1234 --out <dir>` (the mix file lists the dataset, revision, 2,000 val chats and the two dropped sources); output folder `m2_tulu_A` |

## 7. SFT settings (identical for the four arms)

```
sft/run_sft.sh with:
  N_LAYERS=28 ROPE_THETA=50000 D_MODEL=1024 Q_HEADS=8 KV_HEADS=4 FFN_HIDDEN=2560 VOCAB=49280 EMBED_SCALE=1
  SFT_DATA=<m2_tulu_A> ROWS=20 B=4 EPOCHS=1 LR=1e-4 WARMUP=0.03 VAL_FREQ=500 OPT=muon EXPORT=1 NP=1
  INIT_FROM_RAW=<raw base>   (kv4 for A / B, kv4_swa3 for C / D)
  C: SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4      B: SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4 SWA_MISMATCH_OK=1
  D: SWA_WINDOW=0 SWA_MISMATCH_OK=1          A: (nothing: no window)
run_sft.sh adds: BLU_ALL_BF16=1 BLU_EMB_BWD=segmented, cosine schedule, checkpoints every 1,000 steps (newest 2 kept)
```

| | |
|---|---|
| Steps | 1 epoch = **5,652 optimizer steps**; one step = 20 packed rows (4 rows x 5 accumulation micro-steps, 1 GPU) = 81,920 tokens (~55k trained) |
| Optimizer | Muon for the 2-D weight matrices (308M params): momentum 0.95, 5 Newton-Schulz steps, weight decay 0.1; AdamW for the embedding and norms: betas 0.9 / 0.95, weight decay 0.1 |
| Learning rate | peak **1e-4** for both groups; linear warm-up 3% (170 steps), then cosine down to 1e-5 |
| Grad clip | 1.0 |
| Loss | cross-entropy over the trained tokens, divided by the step's total trained-token count |
| Validation | every 500 steps and at the end, over the whole val set (the trainer reads 308 of the 311 rows: 77 micro-batches of 4) |
| Hardware | 1 x H200 per arm; 466-483 ms per step (~170k tokens/s), ~48 min per arm |

## 8. How the runs went

Val loss during training (the trainer's val):

| Step | A (full / full) | B (full / window) | B - A | C (window / window) | D (window / full) | C - D |
|---|---|---|---|---|---|---|
| 0 | 1.672250 | 1.678060 | +0.0058 | 1.668373 | 1.681077 | -0.0127 |
| 500 | 1.395124 | 1.397860 | +0.0027 | 1.394243 | 1.395336 | -0.0011 |
| 1,000 | 1.338270 | 1.340520 | +0.0023 | 1.337176 | 1.337712 | -0.0005 |
| 1,500 | 1.304564 | 1.307104 | +0.0025 | 1.303532 | 1.303804 | -0.0003 |
| 2,000 | 1.279233 | 1.281483 | +0.0023 | 1.277365 | 1.277615 | -0.0003 |
| 2,500 | 1.257578 | 1.259550 | +0.0020 | 1.255676 | 1.255697 | 0.0000 |
| 3,000 | 1.237261 | 1.239287 | +0.0020 | 1.235225 | 1.235015 | +0.0002 |
| 3,500 | 1.219137 | 1.221239 | +0.0021 | 1.217331 | 1.217229 | +0.0001 |
| 4,000 | 1.204701 | 1.206776 | +0.0021 | 1.202782 | 1.202523 | +0.0003 |
| 4,500 | 1.193088 | 1.195056 | +0.0020 | 1.191410 | 1.191190 | +0.0002 |
| 5,000 | 1.185149 | 1.187097 | +0.0019 | 1.183511 | 1.183230 | +0.0003 |
| 5,500 | 1.179824 | 1.181739 | +0.0019 | 1.178129 | 1.177882 | +0.0002 |
| 5,651 (final) | 1.178506 | 1.180413 | +0.0019 | 1.176781 | 1.176567 | +0.0002 |

Final val loss by group (HF transformers on the exports, native attention, all 311 rows), and B run with full
attention:

| Model | trainer final val | total (all 311 rows) | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|
| A | 1.178506 | 1.182986 | 1.191021 | 1.174183 | 1.135404 | 1.277946 | 1.796139 |
| B | 1.180413 | 1.185001 | 1.191413 | 1.177976 | 1.135756 | 1.285109 | 1.813233 |
| C | 1.176781 | 1.181186 | 1.187510 | 1.174256 | 1.132616 | 1.279082 | 1.803708 |
| D | 1.176567 | 1.180943 | 1.187694 | 1.173547 | 1.132838 | 1.276007 | 1.804080 |
| B served with full attention | (HF trainer rows 1.179826) | 1.184336 | 1.191413 | 1.176582 | 1.135761 | 1.281470 | 1.809601 |

No arm overfits (the val loss falls to the end). Benchmarks and the full analysis: `07_results.md`.
