# The SFT trainer against the pretraining trainer: what differs and why

Files: `train/production_zero/bluscript_zero_ckpt.cpp` (pretraining, 3,444 lines) and
`train/production_zero/bluscript_sft.cpp` (SFT, 4,308 lines) at BluTrain `651f9c4`. The full line-by-line difference is
`docs/pretraining_to_sft_trainer.diff` (`diff -u`; today +936 / -89 lines in 51 hunks). The study of every op, with
what the other libraries change: `docs/kernel_diff_sft_pretraining.md`.

---

## 1. The short answer

- **Kernel side: exactly three kernels change for packed SFT**: RoPE (per-chat positions), attention (each chat attends
  only to itself) and the loss (ignore the untrained tokens; one denominator for the whole step). Every other kernel
  runs the pretraining code unchanged. Every library we checked (HF transformers, TRL, torchtitan, Megatron-LM,
  open-instruct, LLaMA-Factory, axolotl, Liger, FlashAttention, TransformerEngine) changes the same three things.
- **The window is not an SFT difference**: both trainers have it (`BLU_SWA_WINDOW`, `BLU_SWA_GLOBAL_EVERY`, the model
  record `model_config.json`); in SFT it is counted inside each chat.
- **Non-kernel side**: the data loader, the loss denominator and the backward seed, the SFT settings, the start weights,
  the packed-row inputs built every micro-batch, validation over the whole val set, and SFT-only extras (shape
  settings, raw init, export, trained-rows switch, context parallelism).
- **Without `BLU_SFT_DATA` the SFT trainer trains exactly like the pretraining trainer.**

## 2. Kernel-side differences

| Op | Pretraining | SFT (packed rows) | Why |
|---|---|---|---|
| RoPE | positions 0..T-1 along the row | positions restart at 0 for every chat (`g_packed.positions`) | a chat must look the same wherever it sits in a row (as TRL's packing with `position_ids`) |
| Attention | plain causal over the whole row | causal inside each chat only (`g_packed.chats`, the chat plan) | chats packed into one row must not see each other |
| Loss | mean over all tokens of the micro-batch | sum over trained tokens (label != 65535), divided by the step's global trained-token count | only answers are trained; every trained token of the step weighs the same, whatever its micro-batch |
| (window, both) | window per layer over the row | window per layer inside each chat | the window is per sequence |

## 3. Non-kernel differences

| What | Pretraining | SFT | Where (search in `bluscript_sft.cpp`) |
|---|---|---|---|
| Mode switch | - | `BLU_SFT_DATA=<dir>` | `"SFT mode: BLU_SFT_DATA"` |
| Data | token shards (the pretraining `DataLoader`, data mixture, shard fingerprints) | `SftDataLoader`: inputs, labels, valid counts, chat lengths; the pretraining loader is never created | `train_loader`, `SftDataLoader` |
| Packed-row inputs | none | positions + chat plan every micro-batch, freed at exit | `packed_rows_of()`, `g_packed` |
| Loss denominator | per micro-batch | the step's trained-token count, all-reduced over the GPUs | `valid_tokens_ahead`, `sft_denom` |
| Backward seed | 1 / GRAD_ACCUM | W (the GPU count) | `grad_scale` |
| Logged loss | averaged | the micro losses just add up (they are already normalised) | `loss_accum` |
| Batch / length | tokens per step, max steps from the YAML | `BLU_SFT_B`, `BLU_SFT_ROWS_PER_STEP`, `BLU_SFT_EPOCHS` (steps computed from the data) | `"SFT mode"` block |
| LR schedule | pretraining warm-up / stable / decay | `BLU_SFT_LR`, `BLU_SFT_MUON_LR`, `_MIN_LR`, `BLU_SFT_WARMUP`, `BLU_SFT_SCHEDULE` (cosine, linear, constant, wsd) | same block |
| Validation | `val_loss_steps` batches of the pretraining val shards | the whole SFT val set, one division by its trained-token count | `"SFT: the whole val set"` |
| Start weights | random init or resume | `BLU_INIT_FROM` (DCP, weights only, fresh optimizer) or `BLU_INIT_FROM_RAW` (converted HF model); a warning if neither | `"initialise the weights from another run"` |
| Model shape | the YAML / built-in model | `BLU_N_LAYERS`, `BLU_ROPE_THETA`, `BLU_D_MODEL`, `BLU_Q_HEADS`, `BLU_KV_HEADS`, `BLU_FFN_HIDDEN`, `BLU_VOCAB`, `BLU_EMBED_SCALE` | `"BLU_N_LAYERS / BLU_ROPE_THETA"`, `"135M base"` |
| Export | - | `BLU_EXPORT` (final safetensors with metadata) | `"BLU_EXPORT"` |
| lm_head on trained tokens only | - | `BLU_SFT_TRAINED_ROWS=1` (off by default) | `trained_rows`, `gather_rows` |
| Context parallelism | - (pretraining uses data parallelism / ZeRO) | `BLU_SFT_CP=U` (Ulysses) | `"struct SftCp"` |
| CSV | the branch's columns + per-domain columns | the branch's columns + `trained_tokens` | `trained_tokens` |
| Resume | from the step | also skips exactly the micro-batches already trained | `skip_micro_batches` |
| Old sequential RoPE path | used | throws for packed rows (it has no positions) | - |

**The same in both:** the model code, the optimizers (Muon + AdamW, ZeRO-1, fp32 masters), the DCP checkpoints and the
model record, NCCL / BluCCL, the GEMM library, the speed changes of 2026-11-03 to 11-05 (batched Newton-Schulz,
fewer copies / launches / CPU waits, GEMM tile tables on the GPU), the window, the precision settings (`BLU_ALL_BF16`,
`BLU_EMB_BWD=segmented` set by `run_sft.sh` so SFT matches pretraining).

## 4. How the SFT trainer came to be

- 2026-10-15/22: the SFT parts written on the pretraining trainer of that time (61 lines of difference on 2026-10-22).
- 2026-10-27: brought onto `dist_side_gautam_lottn` by a 3-way merge (10 conflicts, each keeping both sides; MPI calls
  replaced by the branch's `dist::comm` helpers; the SFT loaders take the local rank as their GPU).
- Since then: the shape settings and raw init (10-27), the export, the trained-rows switch (11-03), context
  parallelism (11-06/07), the window and the model record (11-20). Each is described in `docs/README_sft_BluTrain.md`
  (sections 5.1-5.3, 6.4, 6.5) and listed by date in `docs/SFT_package_START_HERE.md` section 2.

## 5. What this means for a DPO or RL trainer built on BluTrain

- Take the SFT trainer's packed-row path as the template: positions per sequence, the chat plan per micro-batch, the
  window per layer, the loss over the trained tokens with one denominator per step.
- The start weights come from `BLU_INIT_FROM` (our `blumodel_step_5652`) with the shape, theta and window flags; the
  start-up window check protects against a wrong window.
- Everything in section 3's "same in both" can stay as it is.
