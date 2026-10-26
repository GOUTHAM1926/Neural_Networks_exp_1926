---
name: project-sft-workflow-1006
description: "The SFT working method agreed 2026-10-06 — edit only in the laptop's ~/Downloads/BluTrain_sft (dist_side_merge); the box gets a fresh clone plus copies of the changed files daily; push from the laptop at the end"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-06T16:14:45.741Z
---

Agreed with the user on 2026-10-06, when the task-2 (`BLU_INIT_FROM`) test started.

**The master copy:**
- **`~/Downloads/BluTrain_sft`** (laptop) is the ONLY place SFT edits are made.
- It is the folder name of a 09-29 clone of BlubridgeAI/BluTrain on the team branch **`dist_side_merge`** (`909ab75`), with TI `dist_side_merge` at `d96b991`. It is NOT a new branch, and no extra branch is made.

**The box is for tests only** (H200 node `ssh -p 2224`; our GPUs are 5, and 5+6). Each day the user:
1. clones `dist_side_merge` fresh from GitHub, using a token, a `set-url` cleanup, and an `insteadOf` for the TI submodule;
2. copies over EVERY file that `git status` on the laptop shows as changed. I give the exact list each time, because unpushed edits from earlier days are not on GitHub.

Only the changed source files are copied: never the laptop's sm_86 build outputs, the `.orig_init` backups, or the hidden local tests.

**Push:** from the laptop to `dist_side_merge` once SFT is complete, or daily if the user decides that later. The user runs every push.

**Before copying a whole file:** check that the box clone's HEAD == the laptop's base commit. If colleagues changed that file upstream, apply our diff on top of their version instead, and show the user their changes first.

**Logging:** note every change (doc + memory): what, why, the test result, and the commit.

**State on 2026-10-06:**
- The only local edit is `bluscript_zero_ckpt.cpp` (+80 −1, md5 `ad7c5f68…`). It is identical to the documented `profiling_reports/sft_task1_2026_09_29/bluscript_init_from.diff`.
- The checkpoint (metadata.bin + data_rank_0.bin) is in the box's `~/gautam_k/ckpt/blumodel_step_8000/`. I downloaded it WITHOUT permission ([[feedback-a<REDACTED:api-key>]]); the user kept it.

Related: [[project-sft-gautam-branch-facts]], [[feedback-edit-local-push-pull]].

**TASK 2 PASSED 2026-10-06 (GPU 5):**
- 262/262 weights match the checkpoint CRC;
- step-0 val is 2.4538 on flux val (train 2.3274), compared with ~10.8 for random weights;
- no out-of-range ids;
- peak 125,086 MiB at B=8 with all bf16.
Steps 1-3 blew up (8.6 → 13.3) because a fresh AdamW at the pretraining LR of 1.2e-3 was used; SFT needs its own LR. Written up as SFT doc §35.1 (master + mirror). The code is still NOT pushed (laptop only).

**SFT data, checked on HF 2026-10-06:**
- public, not gated:
  - `allenai/tulu-3-sft-mixture`: 939,343 rows, ODC-BY;
  - `allenai/Dolci-Instruct-SFT` (OLMo 3 Instruct): 2,152,112 rows, ODC-BY;
  - `allenai/Dolci-Think-SFT-7B`: 2,268,178 rows, 36 GB, for the think model later;
  - `HuggingFaceTB/smoltalk`: ~1.1M rows in "all" (the dataset-server total of 2.2M counts every config).
- `GAIR/lima` is GATED (HF login plus accepting the terms), with license "other".
- The Llama 2 (27,540) and InstructGPT (~13k) SFT sets were never released, so they cannot be used.

**Stage-0 data, final (2026-10-06):**
- The user's choice: **an English-only dataset instead of filtering** ("why make it complex"). The data is **HuggingFaceH4/ultrachat_200k** @ `8049631c` (MIT, English, `messages` format, Zephyr's SFT data):
  - train = 5,000 random rows of `train_sft` (seed 1234); 4.77M trained tokens (23.3% of positions); mean length 1,238 tokens;
  - val = 500 rows of `test_sft`.
  - Files: `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/{train,val}/` (inputs.bin, labels.bin, valid_counts.bin, meta.json).
- **Script:** `~/Downloads/BluTrain_sft/sft/prepare_sft_data.py`, 236 lines. It has no language filter and is not committed. Its self-test is `--self-test`.
- **Deleted:** the Tülu stage-0 data, the fastText lid.176 model and the `fasttext-wheel` package. The Tülu HF cache (1.4 GB) is still in `~/.cache/huggingface`.
- **Dataloader: option A, DONE (2026-10-06, with the user's OK).**
  - `sft/SftDataLoader.h`: 174 lines, new, uncommitted. Syntax-checked with g++ only.
  - The only change from the pretraining loader: the target comes from `labels.bin` (row-aligned), plus the valid counts.
- **Loader test PASSED (2026-10-06, laptop RTX 3060, 0.8 s).** `sft/test_sft_dataloader.cpp` is new and uncommitted. It builds with g++ against the laptop libtensor and needs `-lcuda`; the binary goes outside the repo. Results:
  - 5/5 checks pass, with 0 byte mismatches over 625 batches;
  - every trained label equals the next input token, and every row ends with 50256;
  - the W=2 split has no overlap (8 tail rows unused);
  - wrap, reset and skip all work;
  - 9 corrupt-file cases each throw with a named row.
  - The step-0 global count at 64 rows/step is **60,263** (the same on 1 and 2 GPUs).
  - Log: scratchpad `loader_test_2026_10_06.log`; not copied to profiling_reports yet.
- **TASK 6 CODED 2026-10-06 (laptop only; not built or run yet).** `bluscript_zero_ckpt.cpp` changed by +154/−23 on top of task 2 (the task-2-only file rebuilt from the saved diff has md5 `ad7c5f68`; the task-6-only diff is in scratchpad `task6.diff`).
  - **Switch and knobs:** `BLU_SFT_DATA=<dir>` turns SFT mode on, using `<dir>/train` and `<dir>/val`. Knobs: `BLU_SFT_ROWS_PER_STEP` 64, `_EPOCHS` 2, `_LR` 2e-5, `_MIN_LR` LR/10, `_WARMUP` 0.03 (then cosine, stable 0), `_VAL_FREQ` 10, `_MUON_LR` (required with muon).
  - **Defaults:** checkpoints go to `checkpoints_sft[_muon]`; the CSV is `sft_adamw.csv`/`sft_muon.csv` with an appended `trained_tokens` column; the step line ends with ` | trained N`.
  - **Loss path:** the mixture loader is not created in SFT mode (unique_ptr), so `BLU_MIX_ROOT` is not needed. Loss uses `(65535, &sft_denom)`, and `sft_denom` = `valid_tokens_ahead(GA)` all-reduced with `MPI_INT64_T`. Seed W. The logged loss is a plain sum. Val goes over the whole val set with a global N.
  - **Checks done:** the trainer compiled cleanly with `mpic++ -fsyntax-only` and the Makefile INCLUDES, and a control copy with one broken call failed as it should. The ZeRO path has no hidden 1/GA (`grad_accum_steps` is 1, the trainer doesn't set it, and `reduce_grads_now` only averages plus applies the LossScaler unscale).
- **STAGE-0 SMOKE RUN PASSED (2026-10-06, 21:38–21:44, GPU 5, AdamW).**
  - It was stopped on purpose after 20 steps, once all 6 goals were met. The user gave a 10-minute budget, OKed the stop, and I killed the run by its exact mpirun PID.
  - **Results:**
    - val went 1.446 → 1.353 (step 10) → 1.332 (step 20);
    - train went 1.450 → about 1.19–1.33;
    - norm went 0.74 → 0.20;
    - trained N was exact (step 0 = 60,263; val = 441,382);
    - 5.3 s/step and 55% MFU, about 78 GB.
  - Steps 0, 1 and 6 are slow (about 35–43 s) because of the trainer's `empty_cache()` at steps 0 and 5. Per the user (10-06), that call is a colleague's old experiment that got copied into every trainer, and **the colleague will remove it himself**. Leave it alone and don't patch it in our files.
  - Logs are in `profiling_reports/sft_stage0_2026_10_06/` (`run.log`, `sft_adamw.csv`). On the box they are under `~/gautam_k/sft_runs/stage0_adamw/`, and the build is `~/gautam_k/BluTrain_sft/_runner_` at md5 `f8686d20`.
  - **Trap:** the user reused this morning's shell. mpirun on one node passes the WHOLE environment to the job, whatever `-x` lists, so `BLU_CKPT=0`, `BLU_MAX_STEPS=4`, `BLU_INIT_VERIFY=1`, `BLU_MIX_ROOT` and `BLU_TOKEN_MAX` leaked in. `BLU_CKPT=0` meant no checkpoint. **For the next run, use a fresh shell or unset these.**
- **NEXT (not started):**
  1. the full 156-step AdamW run with a checkpoint;
  2. the Muon run, which needs a `BLU_SFT_MUON_LR` decision;
  3. eval: convert the checkpoint to HF and generate (ask Kathir about his `Model-Eval-Infer-Suite-PostTrain`), checking that answers end with the end token, plus HellaSwag base vs SFT;
  4. stage 1 (more data, plus the LR sweep 5e-6/1e-5/2e-5).
  - Code is still laptop-only. The files to push are `bluscript_zero_ckpt.cpp` and `sft/{SftDataLoader.h, prepare_sft_data.py, test_sft_dataloader.cpp}`; never `sft/__pycache__` or `.orig_init`.
- **Settings approved on 10-06 ("lets proceed"):**
  - LR 2e-5;
  - 3% warmup, then cosine;
  - 2 epochs;
  - 64 rows/step;
  - AdamW first, then Muon;
  - `BLU_BF16_LOGITS=1`;
  - val every 10 steps.
