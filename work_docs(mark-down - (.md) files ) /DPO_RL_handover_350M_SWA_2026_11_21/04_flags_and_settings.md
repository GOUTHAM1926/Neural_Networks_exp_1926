# Flags and settings: every knob of our SFT, what it does, its default, what we tested

Code: BluTrain `dist_side_gautam_lottn` `651f9c4` + Tensor-Implementations `dd5b61a`. The defaults below are read from
that code (`train/production_zero/bluscript_sft.cpp`, `sft/run_sft.sh`, the kernel files).

**How settings reach the trainer.** The trainer reads environment variables `BLU_*`. `sft/run_sft.sh` is the normal
launcher: it takes short names (`LR=1e-4`, `SWA_WINDOW=1024`, ...), turns them into `BLU_*` variables, **removes every
other `BLU_*` variable first**, and starts the trainer under `mpirun`. So a flag that `run_sft.sh` does not know never
reaches the trainer through it (section 5 shows how to pass one anyway).

---

## 1. `sft/run_sft.sh` variables

| Variable | Default | Becomes | What | Our 350M arms |
|---|---|---|---|---|
| `SFT_DATA` | required | `BLU_SFT_DATA` | folder with `train/` and `val/` from `prepare_sft_data.py`; turns SFT mode on | `m2_tulu_A` |
| `RUN_DIR` | required | logs, `checkpoints/`, CSVs | a new folder for this run | one per arm |
| `INIT_FROM` | - | `BLU_INIT_FROM` | start from a BluTrain DCP checkpoint (weights only, fresh optimizer) | - (DPO: `blumodel_step_5652`) |
| `INIT_FROM_RAW` | - | `BLU_INIT_FROM_RAW` | start from `sft/hf_to_blutrain.py` output (a Hugging Face model) | the raw base (`kv4` / `kv4_swa3`) |
| `N_LAYERS`, `ROPE_THETA` | 26, 10000 (built-in) | `BLU_N_LAYERS`, `BLU_ROPE_THETA` | the base's layer count and RoPE theta; **always set both** | 28, 50000 |
| `D_MODEL`, `Q_HEADS`, `KV_HEADS`, `FFN_HIDDEN`, `VOCAB`, `EMBED_SCALE` | 2304, 18, 6, 6144, 50304, 0 (built-in) | `BLU_<name>` | the rest of the base's shape; a wrong width / heads / ffn / vocab stops the load, a wrong `EMBED_SCALE` does not | 1024, 8, 4, 2560, 49280, 1 |
| `ROWS` | 20 | `BLU_SFT_ROWS_PER_STEP` | packed rows per optimizer step (all micro-batches, all GPUs); a multiple of B x NP (/ U with CP) | 20 |
| `B` | 4 | `BLU_SFT_B` | rows per micro-batch per GPU | 4 |
| `EPOCHS` | 2 | `BLU_SFT_EPOCHS` | passes over the train set; the step count is computed from it | 1 (5,652 steps) |
| `LR` | 5e-5 | `BLU_SFT_LR` and `BLU_SFT_MUON_LR` | peak learning rate of both optimizer groups (the min LR is LR / 10) | 1e-4 |
| `WARMUP` | 0.03 | `BLU_SFT_WARMUP` | linear warm-up as a fraction of all steps | 0.03 (170 steps) |
| `SCHEDULE` | cosine | `BLU_SFT_SCHEDULE` | after warm-up: `cosine`, `linear`, `constant`, `wsd` (decay fraction 0.2) | cosine |
| `VAL_FREQ` | 200 | `BLU_SFT_VAL_FREQ` | validation every N steps (and at step 0 and the end), over the whole val set | 500 |
| `OPT` | muon | `BLU_OPTIMIZER` | `muon` (Muon for matrices + AdamW for the rest) or `adamw` | muon |
| `EXPORT` | 0 | `BLU_EXPORT=<RUN_DIR>/final.safetensors` | 1 = write the final weights (fp32, BluTrain names) for the HF converter | 1 |
| `NP`, `GPUS` | 1, 0 | `mpirun -np`, `CUDA_VISIBLE_DEVICES` | processes and the GPUs they use | 1, one GPU |
| `BIN` | `./bluscript_sft_bin` | - | the trainer binary (or a wrapper script, section 5) | `bin/sft_swa` (md5 3af3e48d) |
| `RESUME` | 0 | `BLU_CKPT_LOAD=1` | 1 = continue this RUN_DIR's newest checkpoint (log to `run_resume_*.log`) | 0 |
| `SFT_T` | from the data's `meta.json` | `BLU_SFT_T` | row length; must equal the data's T | 4096 (automatic) |
| `SFT_TRAINED_ROWS` | off | `BLU_SFT_TRAINED_ROWS` | 1 = lm_head + loss only on labelled tokens (same math, experimental) | off |
| `SFT_CP` | off | `BLU_SFT_CP` | U > 1: context parallelism, U GPUs share every row (`05_cp_and_swa.md`) | off |
| `SWA_WINDOW` | off | `BLU_SWA_WINDOW` | sliding window of W keys (multiple of 128) in the windowed layers | C, B: 1024; D: 0 |
| `SWA_GLOBAL_EVERY` | 4 | `BLU_SWA_GLOBAL_EVERY` | layer i is global iff (i + 1) % N == 0 | 4 |
| `SWA_MISMATCH_OK` | off | `BLU_SWA_MISMATCH_OK` | 1 = start although the base's window record differs (deliberate arm; warning printed) | B, D: 1 |

Fixed by `run_sft.sh` (not variables): `BLU_ALL_BF16=1`, `BLU_EMB_BWD=segmented` (the embedding backward of
pretraining: fp32 sums, deterministic), `BLU_CKPT=1`, checkpoints every 1,000 steps keeping the newest 2
(`BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=2`), `BLU_SFT_MUON_LR = BLU_SFT_LR`. Safety checks before the start: no other
BluTrain training on the GPUs, >= 60 GB free GPU memory, **>= 50 GB free disk** in `RUN_DIR`.

## 2. Trainer flags (`BLU_*`) that matter for SFT, DPO and RL

Set by `run_sft.sh` from section 1 unless marked "direct" (then they need section 5, or your own launcher).

| Flag | Default | What |
|---|---|---|
| `BLU_SFT_DATA` | empty = pretraining | SFT mode and its data folder |
| `BLU_SFT_B`, `BLU_SFT_ROWS_PER_STEP`, `BLU_SFT_EPOCHS` | from the config, 64, 2 | batch shape and length (section 1) |
| `BLU_SFT_LR`, `BLU_SFT_MIN_LR` | 2e-5, LR / 10 | AdamW group: peak and final LR |
| `BLU_SFT_MUON_LR`, `BLU_SFT_MUON_MIN_LR` | required with Muon, LR / 10 | Muon group (the pretraining Muon LR would wreck the weights) |
| `BLU_SFT_WARMUP`, `BLU_SFT_SCHEDULE`, `BLU_SFT_DECAY_FRAC` | 0.03, cosine, 0.2 | the LR shape (`wsd` uses the decay fraction; direct) |
| `BLU_SFT_VAL_FREQ` | 10 | validation period in steps |
| `BLU_SFT_T` | the config's context length | row length = RoPE cache length; must equal the data's T |
| `BLU_SFT_TRAINED_ROWS` | 0 | lm_head + loss on labelled rows only; not with CP |
| `BLU_SFT_CP` | 1 | context parallelism degree U |
| `BLU_SWA_WINDOW`, `BLU_SWA_GLOBAL_EVERY`, `BLU_SWA_MISMATCH_OK` | 0, 4, 0 | the window (section 1; `05_cp_and_swa.md`) |
| `BLU_INIT_FROM` | - | DCP checkpoint to start from (weights only) |
| `BLU_INIT_FROM_RAW` | - | raw folder (HF model converted by `hf_to_blutrain.py`) to start from |
| `BLU_INIT_VERIFY` | 0 | direct; 1 = check every loaded weight's CRC32 against the checkpoint |
| `BLU_N_LAYERS`, `BLU_ROPE_THETA`, `BLU_D_MODEL`, `BLU_Q_HEADS`, `BLU_KV_HEADS`, `BLU_FFN_HIDDEN`, `BLU_VOCAB`, `BLU_EMBED_SCALE` | the built-in 26-layer model | the model shape; must be the checkpoint's |
| `BLU_OPTIMIZER` | from the config | `muon` or `adamw` |
| `BLU_ALL_BF16` | 0 | 1 = residual stream, logits and embedding in bf16 (`BLU_BF16_RESID` / `_LOGITS` / `_EMBED` one by one) |
| `BLU_EMB_BWD` | the library's default (bf16 atomicAdd) | `segmented` (what pretraining and `run_sft.sh` use), `atomic`, `cooperative` |
| `BLU_QKROPE_FP32` | 0 | direct; 1 = the fp32 QK-norm + RoPE path instead of the bf16 kernel |
| `BLU_EXPORT` | - | path of the final safetensors export |
| `BLU_CKPT`, `BLU_CKPT_DIR`, `BLU_CKPT_FREQ`, `BLU_CKPT_KEEP`, `BLU_CKPT_LOAD` | on, -, 1600, 4, 0 | checkpointing (run_sft.sh: on, `RUN_DIR/checkpoints`, 1000, 2, `RESUME`) |
| `BLU_MAX_STEPS` | 0 | pretraining only: **ignored in SFT mode** (the log says `[sft] BLU_MAX_STEPS is ignored in SFT mode`); a short SFT run uses a small data split instead (e.g. the first 640 rows = 32 steps of 20 rows, as in our smoke runs; `sft/ref_val_loss.py --make-subset` writes one) |
| `BLU_GLOBAL_CLIP`, `BLU_GLOBAL_CLIP_STRICT` | 0 | direct; clip by the global norm over all ranks |
| `BLU_STEP_WATCHDOG`, `BLU_STEP_TIMEOUT_S` | 0, 600 | direct; abort a step that hangs longer than the timeout |
| `BLU_DIAG` | 0 | direct; extra diagnostics in the log |
| `BLU_RUN_DIR`, `BLU_LOG_CSV` | - | where the monitor files and the CSV go (run_sft.sh sets them) |

Not for SFT / DPO / RL on one node (pretraining or multi-node only): the data-mixture flags (`BLU_MIX_*`, `BLU_W_*`),
the inter-node transport flags (`BLU_IB_*`, `BLU_AR_*`, `BLU_RS_*`, `BLU_SYMM_IB`, ...), the ZeRO / Muon communication
tuning (`BLU_ZERO_V2`, `BLU_MUON_BUCKETS`, `BLU_MUON_PIPELINE`, `BLU_NO_OVERLAP*`, `BLU_OVERLAP_REFRESH`,
`BLU_COMM_*`), the monitor and profiling flags. They keep their pretraining defaults in SFT.

## 3. Kernel flags (Tensor-Implementations; read by the library, so always "direct")

| Flag | Default | What | Tested |
|---|---|---|---|
| `BLU_GQA_MX` | on | forward "mixed units": an idle compute group of a chat with an odd number of 64-row tiles runs the same tile of another query head of the same KV head; `=0` turns it off | -21..-26% forward on chats of <= 64 tokens; real SFT rows -0.5% |
| `BLU_GQA_MX_MIN_PCT` | 2 | mixed units only when they are at least this share (%) of the plan's units (the 2% rule); 0 = always (the test suite) | sweep of 2026-11-13: below 2% mixing was slower |
| `BLU_VJ2_HSUM` | 1 | backward idea 4: one block per (key tile, KV head) adds the query heads' dK / dV in registers (no partial buffers, no head-sum launches); `=0` = the per-query-head path | part of the backward pushed 2026-11-18 (ideas 1, 2, 4; numbers in `docs/sft_kernel_benchmarks.md` section 8) |
| `BLU_VJ2_HSUM_MINKV` | 2 | idea 4 only when the call has at least this many KV heads (rule 2b: decided by the shape, never by the batch, so a chat takes the same path packed or alone) | the 350M (4 KV heads) uses idea 4; CP with 1 KV head per GPU does not |
| `BLU_VJ2_HSUM_SECT` | 0 | experimental sectioning of the head sums; leave at 0 | - |
| `BLU_VJ2_DKDV_F32` | 0 | 1 = fp32 dK / dV partials (more precise, each step slower, no visible gain in loss; turns idea 4 off); the log then says `[vj2] fp32 dK/dV partials` | 2026-10-31 long-context checks |
| `BLU_BWD_TWO_STREAM` | off | a second CUDA stream for the matmul backward (value-checked: `0`, `false`, `no`, empty = off) | not used in SFT |

## 4. Precision, determinism, speed: what we run and why

- `BLU_ALL_BF16=1`: the production setting of pretraining; the 350M arms ran with it (fp32 optimizer masters).
- `BLU_EMB_BWD=segmented`: deterministic fp32 sums in the embedding backward (the bf16 atomicAdd default is not
  deterministic); set by `run_sft.sh` since 2026-10-30 (rule: SFT uses the same settings as pretraining).
- The attention kernels are deterministic for a given layout; a chat gives bit-identical results packed or alone, and a
  batch whose chats all fit in the window runs the window-free kernels bit for bit.

## 5. How to pass a flag that `run_sft.sh` does not pass (kernel flags, `BLU_MAX_STEPS`, ...)

`run_sft.sh` removes every `BLU_*` variable before it starts the trainer, so give it a wrapper as `BIN` that sets the
flag and then runs the real trainer:

```
cat > my_trainer.sh <<'EOF2'
#!/bin/bash
export BLU_VJ2_DKDV_F32=1          # the flag(s) to add
exec /path/to/bluscript_sft_bin "$@"
EOF2
chmod +x my_trainer.sh
BIN=$PWD/my_trainer.sh SFT_DATA=... RUN_DIR=... bash sft/run_sft.sh
```

Check the run's log for the line the flag prints (for example `[vj2] fp32 dK/dV partials (BLU_VJ2_DKDV_F32=1)`; no line
= the flag did not arrive). Or start the trainer with your own `mpirun` line and export the variables yourself
(section 1 lists everything `run_sft.sh` sets).

## 6. The settings of our SFT runs, for reference

| Run | Model | Data | Main settings |
|---|---|---|---|
| 350M window study (2026-11-20/21) | 350M, 28 layers, d 1024, heads 8/4, theta 50,000, embed scale 1 | `m2_tulu_A` (Tülu 3 minus Aya and hard-coded) | 1 epoch, LR 1e-4, 20 rows of 4,096 per step, Muon + AdamW, bf16, val every 500; window arms `SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4` |
| 1.7B Tülu 3 (2026-10-22) | 1.7B, 28 layers, d 2304, heads 18/6, theta 50,000, GPT-2 vocab | Tülu 3 minus Aya | 2 epochs, LR 5e-5, 20 rows per step, Muon + AdamW, bf16, val every 200 (`docs/README_SFT_checkpoint_for_DPO.md`) |
| 135M grid (2026-10-28) | 135M, 37 layers, d 512, heads 4/2, theta 50,000 | 12 mixes (`sft/mixes/`) | `docs/plan_grid_search_sft.md` |
