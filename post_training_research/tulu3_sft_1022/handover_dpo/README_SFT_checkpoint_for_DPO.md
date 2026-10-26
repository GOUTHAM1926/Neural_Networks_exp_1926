# Tülu 3 SFT of the 1.7B model: checkpoint, settings and data for the DPO run

Everything needed to start DPO from the SFT model: where to download it, the exact model, tokenizer and chat format, the
SFT data and settings, how the run went, and the files. Benchmark results follow separately.

---

## 1. The checkpoint

**Final SFT model = `blumodel_step_11590`** (end of epoch 2, the last step of the run). Public, no login needed.

```
mkdir -p blumodel_step_11590 && cd blumodel_step_11590
curl -L -O https://huggingface.co/buckets/gautam1926/checkpoints_sft/resolve/blumodel_step_11590/metadata.bin
curl -L -O https://huggingface.co/buckets/gautam1926/checkpoints_sft/resolve/blumodel_step_11590/data_rank_0.bin
```

Check that the download is complete (both checksums must match exactly):

| File | Bytes | md5 |
| --- | --- | --- |
| `data_rank_0.bin` | 17,479,868,928 | `171a164edb6b307205ac0db5b257e374` |
| `metadata.bin` | 109,061 | `0f9cdf52b14ed40a31e0101cc7911bb0` |

- It is a BluTrain DCP checkpoint saved by 1 GPU: the model weights (282 tensors: bf16, the 56 QK-norm gammas fp32) **plus**
  the optimizer state (Muon momentum, AdamW moments, fp32 master weights). The weights alone are 3.4 GB of the 17.5 GB.
- Also in the same place: `blumodel_step_6000` (205 steps into epoch 2, so practically the end-of-epoch-1 model;
  `data_rank_0.bin` md5 `548c8a2be0cd8c134523c3ed7d66c670`, `metadata.bin` md5 `80aba476ea84a77e0f3ee0968b3f601e`), same
  URL pattern with `blumodel_step_6000`.

## 2. How to start DPO from it in BluTrain (important)

```
BLU_INIT_FROM=<path>/blumodel_step_11590  BLU_N_LAYERS=28  BLU_ROPE_THETA=50000
```

- **`BLU_N_LAYERS=28` and `BLU_ROPE_THETA=50000` are both required.** The trainer defaults are 26 layers and theta 500000.
  A wrong layer count stops with a shape error, but **a wrong theta gives no error at all**: the model just works worse
  (measured on this model's base at step 0: val loss 1.1757 with 50000, 1.2301 with 10000, 1.2596 with 500000).
- `BLU_INIT_FROM` loads the **weights only** (it reads `metadata.bin` + `data_rank_0.bin`); the optimizer state in the file
  is not used, so DPO starts with a fresh optimizer.
- The DPO **reference model** is this same checkpoint.

## 3. The model

| | |
| --- | --- |
| Base (pretrained) model | `navingv/fullchess` on Hugging Face (Navin's pretraining run: step 22,600, about 22.6B tokens, Muon + AdamW) |
| Parameters | 1.701B (the embedding is tied to the output layer, counted once) |
| Layers | 28 decoder blocks: RMSNorm → attention → residual, RMSNorm → SwiGLU MLP → residual; final RMSNorm |
| Width | d_model 2304; SwiGLU hidden 6144 (`gate_up` stored as [gate \| up]) |
| Attention | grouped-query: 18 query heads, 6 key/value heads, head size 128, causal |
| QK-norm | per-head RMSNorm on q and k (gamma of size 128, eps 1e-6), applied **before** RoPE |
| RoPE | NeoX style (first half / second half), theta **50000** |
| Norms | RMSNorm `x * rsqrt(mean(x²) + eps) * gamma`, eps 1e-5 |
| Context | 4096 tokens |
| Vocab | 50304 rows = GPT-2's 50257 padded; ids 50257–50303 are never used |
| Biases | none |
| Precision in training | all bf16 (`BLU_ALL_BF16=1`), fp32 optimizer masters |

The same layout as Hugging Face's `Qwen3ForCausalLM`. An exporter to that format exists and is proven: the exported model
gives the same val loss as our trainer (0.850077 vs 0.850100); ask if you need the HF format.

## 4. Tokenizer and chat format (DPO data must use exactly this)

- Tokenizer: **GPT-2** (tiktoken `gpt2`). End token / padding = `<|endoftext|>` = id **50256**.
- Chat template `tulu-plain-v1`:

```
<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n{next user turn}\n<|assistant|>\n...
```

- The role markers (`<|system|>`, `<|user|>`, `<|assistant|>`) are **plain text**, not special tokens. Each piece (marker,
  message, `\n`) is tokenized on its own, so a marker never merges with its neighbour.
- Every assistant answer ends with the real end token 50256; the model learned to stop there.
- Trained positions: only assistant answer tokens + their end token. Everything else (system, user, markers, padding) has
  label 65535 = ignored.
- For DPO: the prompt ends with `<|assistant|>\n`; chosen and rejected answers each end with `<|endoftext|>` (50256).

## 5. SFT data

| | |
| --- | --- |
| Dataset | `allenai/tulu-3-sft-olmo-2-mixture-0225`, revision `d91a0785ade02942520280fb484866fce41e448f` (866,138 chats) |
| Removed | the Aya part `ai2-adapt-dev/tulu_v3.9_aya_100k` (99,989 chats; multilingual, while the base model was pretrained on English) |
| Validation | 2,000 chats held out from the same mix (never trained on) |
| Dropped | 22,020 chats longer than 4096 tokens, 716 with an empty answer |
| Training chats | 741,413 |
| Packing | packed into **115,904 rows of 4096 tokens** (First-Fit-Decreasing, rows shuffled); each chat attends only to itself and its RoPE positions restart at 0 (as TRL's packing with `position_ids`) |
| Tokens per epoch | 474.7M positions, **325.7M trained (answer) tokens** (68.6%) |
| Chat length | mean 640, median 470, 95th percentile 1608 tokens |
| Prep script | `prepare_sft_data.py` (in this folder), seed 1234; train: `--num 0 --skip-kept 2000`, val: `--num 2000`, both `--dataset allenai/tulu-3-sft-olmo-2-mixture-0225 --revision d91a0785ade02942520280fb484866fce41e448f --split train --drop-source ai2-adapt-dev/tulu_v3.9_aya_100k --seq-len 4096 --pack ffd` |

## 6. SFT training settings

Exactly as run (`run_tulu3_sft.sh`, in this folder):

```
BLU_SFT_DATA=<data dir>  BLU_INIT_FROM=<fullchess dir>  BLU_ALL_BF16=1  BLU_OPTIMIZER=muon
BLU_SFT_LR=5e-5  BLU_SFT_MUON_LR=5e-5  BLU_SFT_B=4  BLU_SFT_ROWS_PER_STEP=20  BLU_SFT_EPOCHS=2  BLU_SFT_VAL_FREQ=200
BLU_N_LAYERS=28  BLU_ROPE_THETA=50000  BLU_CKPT=1  BLU_CKPT_FREQ=1000
```

| | |
| --- | --- |
| Steps | 2 epochs = **11,590 optimizer steps**; one step = 20 packed rows (4 rows × 5 accumulation micro-steps, 1 GPU) ≈ 57k trained tokens |
| Optimizer | Muon for the 2-D weight matrices (1,585M params): momentum 0.95, 5 Newton–Schulz steps, weight decay 0.1. AdamW for the embedding and norms (116M): betas 0.9 / 0.95, weight decay 0.1 |
| Learning rate | peak **5e-5** for both groups; linear warm-up over the first 3% (348 steps), then cosine down to 5e-6 |
| Grad clip | 1.0 |
| Loss | cross-entropy over trained tokens, divided by the step's total trained-token count |
| Hardware | 1 × H200; 1.86 s/step, 44k trained tokens/s, 52% MFU; about 6 h of training in total |

The run was stopped twice and continued from checkpoints 8000 and 9000; the continued runs match the stopped ones exactly
at the resume steps (same train and val loss).

## 7. How the run went

| | Val loss (2,000 held-out chats) | Train loss (mean of 500 steps) |
| --- | --- | --- |
| Base model (step 0) | 1.1757 | — |
| End of epoch 1 (step 5795) | 0.8608 | 0.8431 |
| End of epoch 2 (step 11589) | **0.8501** (the lowest of the run) | 0.7737 |

No overfitting (the val loss never went up). Epoch 2 added a small real gain and some memorization of the training chats.
Benchmark results (6000 vs 11590) come separately.

## 8. Files in this folder

| File | What |
| --- | --- |
| `README_SFT_checkpoint_for_DPO.md` | this note |
| `tulu3_sft_1022_full_run.log` | the complete training log (all 11,590 steps; three parts joined as they are, header says which part is which) |
| `trajectory.csv` | every step: train loss, val loss, lr, grad norm, trained tokens, speed |
| `loss_train_val.png`, `loss_val.png` | loss charts |
| `run_tulu3_sft.sh` | the launch script with all settings |
| `prepare_sft_data.py` | the data preparation script |
| `bluscript_sft.cpp` | the SFT trainer source (md5 `f2024ab2`) |
| `SftDataLoader.h` | the SFT data loader the trainer uses (`sft/SftDataLoader.h`, md5 `83c88d41`): reads the prepared rows in file order, masked labels, packed-chat positions, trained-token counts for the loss denominator |
