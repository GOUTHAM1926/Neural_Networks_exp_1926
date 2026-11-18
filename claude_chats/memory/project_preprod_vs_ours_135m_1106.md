---
name: project_preprod_vs_ours_135m_1106
description: "11-06 task: 135M before/after = preproduction-135M as-is vs preproduction-135M + our branch merged (1 pretraining + 1 SFT run, P032), for the team leads before pushing our work to preproduction-135M"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-06T13:37:14.510Z
---

**User, 2026-11-06 afternoon:** on the 135M config of the 10-28 production SFT (Navin's 900B base), run one pretraining run
and one SFT run on preproduction-135M and on our branch. The user shows the difference to the team leads, then pushes our
branch into preproduction-135M "so everyone uses the optimized ones". 135M only; 1.6B and other models later. Run on P032.

**Config:**
- Model (training_meta.json): 512 / 37 layers / 4q 2kv / hd 128 / ffn 1408 / vocab 49280 / theta 50000 / embed_scale 1 /
  tied / T 4096. The base is 428,945 steps x 2,097,152 tokens = 900B.
- SFT (stage 1, c128 lr1e-4): 1 GPU, B=13 x 2 micro-batches (ROWS=26), Muon, WARMUP 0.03, cosine, 2 epochs = 6,996
  steps, m1_smol_A, ckpt_A. 0.47-0.51 s/step on 10-28.
- Pretraining: `bluscript_zero_ckpt_135m_eqab.cpp` (repo root, preproduction-135M ONLY) + MixedDataloader135.h. Env-driven:
  BLU_VOCAB, BLU_EMBED_SCALE, BLU_MICRO_B, BLU_GLOBAL_BATCH, BLU_OPTIMIZER=muon. B=16, global 2,097,152. Data
  BLU_MIX_ROOT default /localstorage/dinesh/tok50b_v2/ours_49152/s1. Ran on 16 GPUs. The CSV is mix_ratio_1.csv (muon) in
  cwd; it refuses to overwrite without BLU_LOG_OVERWRITE=1.

**Findings 11-06:**
1. The preproduction SFT trainer has no 135M knobs and no INIT_FROM_RAW. Those came in our 334ecbb (10-30), which was
   never merged. Kiruthik's 10-29 merge took us only to 5c0999c.
2. The 135M trainers never got e9e310c (empty_cache step 0 only). Porting it = 9 lines in 2 files.
3. The 135M trainer hard-codes rope_theta 10000 with no env knob, while Navin's meta says 50000, so his run used a local
   edit.
4. merge-tree: TI is clean (tree 8a3d423); BluTrain is clean except the TI gitlink (tree f30d423).

**Package:** `profiling_reports/preprod_vs_ours_2026_11_06/pkg/`. `make_src.sh` builds `src_pre.tar.gz` (60754ce /
a9e37e6 / ea666ab / CP cf1372d) and `src_ours.tar.gz` (the merge trees + BLAS a6d4292). Both are verified.

**Asked the user (pending):**
- the P032 GPUs, and 8 vs 1 GPU for pretraining;
- SFT "before" = preproduction + only the 334ecbb SFT files;
- port the empty_cache change to the 135M trainers;
- read-only box checks of the data.

**11-06 later:**
- The user said proceed on P032 GPUs 2 + 3 (one pretraining and one SFT run, old vs new). The plan was 4 runs of 30
  steps plus a 3-step smoke of each build, ~45 min.
- Read-only checks: all 8 GPUs were free; sft135 data/m1_smol_A (1.5G) + ckpt_A are still there; nvcc is at
  /usr/local/cuda/bin (not on PATH); 224 cores.
- Dinesh's tok50b_v2 data is NOT on P032 (README only). tok2t/ours_49152/root is dead symlinks. The only real
  ours_49152 shards are tok2t_wtest/s1/code (3 x ~95M tokens + val), so the plan was to point all 4 domains at those
  shards. MixedDataloader135 matches "train"/"val" in the file names.
- **STOPPED by the user before anything was uploaded or run: "all are using" (the GPUs were taken).** Nothing was written
  on P032. Resume on the user's word: scripts -> upload -> build -> smoke -> 4 runs.

**RESULTS 11-06 (P032 GPUs 2+3, 30 steps, median 11-29):**
- **SFT (1 GPU, B13 x 2):** 348.7 -> 309.1 ms (-11.4%).
- **Pretraining, 135M eqab trainer (bf16 ATOMIC embedding backward):** 3,930 -> 5,693 ms (+44.9%). nsys: the whole
  regression is embedding_backward_kernel_optimized, 21 -> 122 ms per call.
  - Cause: TI d067e89 (10-31 grid-limit fix) swapped the block order; atomic contention on hot tokens (`_` = 2,911 of
    65,536).
  - The 10-31 test used uniform random ids, so it missed this.
- **Pretraining, production trainer + YAML (SEGMENTED):** 3,667.8 vs 3,678.1 ms (+0.28%, noise), and the first steps
  are -65 s per run.
- **Fix S1 proposed (NOT approved yet):** old order when N <= 524,280 per call.
- **S2 (Navin's call):** a segmented switch for the 135M trainer.
- **CP doc for Adhitya:** `work_docs(...) /cp_ulysses_issues_for_adhitya.{md,docx}`.
- The 2x box closed 11-06 evening.

Method as in [[project_tomorrow_1106_full_grid]]: a 30-step CSV stop and the median of steps 11-29; same GPU, run one after
the other.
