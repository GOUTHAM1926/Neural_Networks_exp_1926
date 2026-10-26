---
name: project_tulu3_sft_run_1022
description: "10-22 Tülu 3 SFT run (user types every command): data tulu-3-sft-olmo-2-mixture-0225@d91a0785, Aya DROPPED, 2,000 held-out val chats, packed; init = navingv/fullchess (23B tokens, 8-rank ckpt, weights in data_rank_0); box 86.38.182.101 GPU 0"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-23T14:47:31.306Z
---

Decided 2026-10-22 with the user (official-paper research: Tülu 3 SFT mix used for OLMo 2 1B + Llama 3.1 8B/70B/405B;
OLMo 3 Dolci = 32K-context reasoning data, 7B/32B only, does not fit our T 4096):
- Data: `allenai/tulu-3-sft-olmo-2-mixture-0225`, revision `d91a0785ade02942520280fb484866fce41e448f`, 866,138 chats,
  19 sources. **Aya dropped** (`ai2-adapt-dev/tulu_v3.9_aya_100k`, 99,989 chats: GPT-2 tokenizer 2-4x tokens for non-English,
  English-only pretraining) -> 766,149 chats. **2,000 held-out val chats** from the mix (prep script `--num 2000` for val,
  `--num 0 --skip-kept 2000` for train, same seed / filters). Packing only ([[feedback_packing_only]]).
- Tokenizer must stay GPT-2 tiktoken (model's embedding = GPT-2 ids); the dataset is text, so no problem.
- Prep script options added on the laptop master `sft/prepare_sft_data.py` (md5 d4d8f8b1): `--drop-source` (repeatable),
  `--skip-kept N`, `--num 0` = all; tested on fake data (val/train disjoint, union = all English chats, no Aya).
- Init checkpoint: HF `navingv/fullchess` = 8-GPU save (data_rank_0 5.19 GB holds the bf16 weights + rank-0 optimizer
  shard; ranks 1-7 ~1.7 GB = optimizer only); weights-only init needs metadata.bin + data_rank_0.bin (proven on 10-21
  with the 4-GPU 8B ckpt). Box layout: ~/gautam_k/{BluTrain_sft, local (yaml-cpp), prepenv (datasets, tiktoken, numpy),
  ckpt, sft_data}; library + bluscript_sft_bin built 10-22 ~12:20 IST.
- Data prep DONE 10-22 (box `~/gautam_k/sft_data/tulu3_noaya/{train,val}`): train 741,413 chats -> 115,904 packed rows
  (min 115,894), 474.7M positions / 325.7M trained tokens per epoch, 6.4 chats/row, mean 640 tok; drops too_long 22,020,
  Aya 99,989, empty_answer 716. Val 2,000 chats -> 320 rows. ~2.6 h/epoch at 50.3k tok/s -> ~5.2 h for 2 epochs.
- Run settings agreed: ~128 chats/step = 20 rows/step (~5,800 steps/epoch); **BLU_SFT_VAL_FREQ=200** (default 10 would
  cost hours). Run PAUSED until the bf16 RoPE port (stages A-C, [[project_remind_1022_rope_restart]] context) is tested.
- bf16 RoPE port: stage A (Kruthik kernel as-is) PASS bit-identical vs fp32 path; stage B (HAS_POS added) PASS all
  3 checks bit-identical, 9/9 SASS identical to his original. Positions cost on HIS kernel +28-36% fwd+bwd (our fp32
  kernel was +2%) = ~0.7% of a step / ~2 min per run; bf16 path still saves ~3.4%. DECIDED 10-22: keep the restart,
  run now; optimise the HAS_POS=true read later (option C), never rebuild libtensor.so on the box while a run uses it.
  Full write-up: work_docs/kernel_diff_sft_pretraining.md. Next = stage C (trainer switch + step-0 loss check).
- fullchess = Navin's pretraining (log in repo root bluscript_zero_ckpt_022.log): **28 layers, rope_theta 50000**, T 4096,
  all-bf16, 1.70B params, Muon+AdamW peak 1.2e-3 -> 1.2e-4 WSD, 22,600 x 1M tokens. Our trainer defaults 26 / 500000 ->
  added BLU_N_LAYERS / BLU_ROPE_THETA env overrides (10-22). Wrong theta does NOT error (only val 1.2301@10k vs 1.2596@500k).
  Stage C PASS (step-0 identical bf16 vs fp32, 3.0% faster); trainer is NOT bit-reproducible run-to-run in backward
  (norm 0.4019 vs 0.4017 same config) -- open item. Run script profiling_reports/rope_bf16_port_2026_10_22/run_tulu3_sft.sh.
- Optimizer DECIDED by user 10-22: **Muon** (base pretrained with Muon; Moonlight Table 6 + our 10-07 sweep, work doc
  37.5-37.6). Proposed peak LR 5e-5 for Muon and AdamW part alike (Moonlight Tulu-3 SFT lr; sweep: 5e-5 no overfit).
- RUN STARTED 2026-10-22 15:51 IST (box ~/gautam_k/runs/tulu3_sft_1022, Muon lr 5e-5, theta 50000, 28 layers):
  step-0 val 1.1757 (< 1.2301 @10k, theta confirmed), 1.882 s/step, 43.5k tok/s, 52% MFU, 70% of positions trained;
  11,590 steps, ETA ~22:10-22:30 IST. Watchdog: monitor_tulu3_sft.sh -> monitor.txt (ALERT lines + live ETA).
  Next while it runs: bench_rope_bf16.cu (standalone, kernel-only fwd/bwd/reduce) in ~/gautam_k/rope_opt on a NON-0 GPU.
- Checkpoints 17 GB each (blumodel_step_N, keep 2); step 1000 at 16:26 IST, 2000 at 17:00 -> 34 min/1000 steps.
  **COPY blumodel_step_6000 to runs/tulu3_sft_1022/keep_blumodel_step_6000 between ~19:16 and ~20:24 IST** (deleted at
  step 8000). Run end ~22:26 IST. Overfitting/dev/unseen doubts documented: work_docs/overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md.
- **EOD 10-22: STOPPED at step 9194 (21:30 IST); RESUME FROM blumodel_step_9000** (2,590 steps ~1.5 h) with
  run_tulu3_sft_resume1.sh settings + a NEW log name. Resume proven exact (step 8000 identical after resume). Checkpoints
  6000(keep_)/8000/9000 stay on the box disk /dev/vda1 -> user attaches it to a new instance 10-23 (if the disk is not kept they
  must be uploaded first). Laptop backup ~/Downloads/box_backup_2026_10_22 (logs, binary, env versions; no checkpoints).
  sft_muon.csv was rewritten by the resume (8000+ only); steps 0-8723 are in run.log. Documented in work doc §45.65-45.69.
- **10-23 11:30 IST CHECK: OLD DISK NOT ATTACHED.** New box root@95.133.252.52 (short-data-lies-fin-03, 4x H200) has ONE
  fresh 300G disk (vda, created 10-23 05:47 UTC), no vdb, no ~/gautam_k anywhere (find / -xdev); teammates' folders
  (Ananya, aparna, harhsini, madhu, ram, myenv) + load avg 351. Old IP 86.38.182.101 now answers with a DIFFERENT host key
  (old instance most likely deleted, IP reused) -- I did NOT bypass the warning. Checkpoints 6000/8000/9000 exist only if
  the provider kept the old volume (user to check the dashboard / ask the admin). If lost: restart from step 0
  (fullchess from HF, re-prep data with prepare_sft_data.py d4d8f8b1, rebuild from laptop BluTrain_sft).
- **10-23 12:30 IST: OLD DISK FOUND** -- it is the root disk of a 1x RTX PRO 6000 Blackwell box root@95.133.252.123
  (clean-name-flourishes-fin-03). /root/gautam_k intact: step_9000 + keep_6000 (data_rank_0.bin 17,479,868,928 B +
  metadata.bin each), 8000, logs end at 9194. Non-ckpt total ~13.7 GB (rivalenv 5.8, ckpt 4.9, sft_data 1.9).
  Plan (user): rsync all but checkpoints -> 4x box 95.133.252.52 (laptop `ssh -A` to 1x, then rsync); upload 9000 + 6000
  to the user's HF bucket hf://buckets/gautam1926/checkpoints_sft via `hf sync` (prepenv hf 2.2.0). Box `hf` login is
  Madhuuu373 (a teammate's) -> use HF_TOKEN via `read -rs`, never `hf auth login` there.
- **10-23 12:50 IST DONE + PROVEN:** rsync to 4x = 37,297 files + 9 links, content md5-of-md5s fe9545e8 identical on
  both boxes; bucket has 9000 + 6000 at exact sizes. Checkpoint md5s (laptop box_backup_2026_10_22/checkpoint_md5_1023.txt):
  9000 data_rank_0 404c2809.., metadata c14e3418..; 6000 data_rank_0 548c8a2b.., metadata 80aba476...
  4x box: copied binary finds all libs (NO rebuild). /usr/local/cuda -> 13.4 (installed by a teammate 10-23 12:49 IST;
  13.1 also new) but the run was built+ran on 13.0 -> put /usr/local/cuda-13.0/lib64 FIRST (proven: all 5 CUDA libs then
  resolve to 13.0). Disk 66G free and falling ~22G/h (teammates) -> hold the 6000 download, BLU_CKPT_KEEP=1.
  Trainer saves at the final step too (bluscript_sft.cpp:2857) and prunes only blumodel_step_* in the ckpt dir.
- **RUN COMPLETE 10-23 14:42:58 IST, exit 0, 0 ALERTs** (4x box GPU 3, run_tulu3_sft_resume_4x.sh, log run_resume_4x.log,
  started 13:15 by me with the user's explicit OK). Step 9000 matched 10-22 exactly (train 0.567569, norm 0.1390); 9001+ drift
  4e-5..9e-5 = smaller than 10-22's own same-box resume drift (1.3e-4) -> backward non-determinism, not the box. 1.86 s/step,
  44k tok/s, 52.5% MFU. Val: 0.854744 (9000) -> 0.852799 (10000) -> 0.850836 (11000) -> **0.850100 final (11589)**, falling all
  through epoch 2. Final = blumodel_step_11590 (md5 data 171a164e.., metadata 0f9cdf52..), upload to bucket started by user.
  Logs: run.log (0-8723, old box) + run_resume1.log (8000-9194) + run_resume_4x.log (9000-11589). Eval suite NOT built
  (needs HF export + generation + harness); loss curves + epoch-2 memorization measure can be done from the logs.
- **10-23 afternoon: analysis DONE** (post_training_research/tulu3_sft_1022/: analyze_run.py, joined log, 3 charts, csvs):
  joins exact (val+train identical at 8000/9000); val 1.1757 -> 0.8608 (e1) -> 0.8501 (e2, lowest last); NO overfitting;
  memorization measured 0.06-0.07 nats steady (epoch-2 step 5795+n shares 16/20 rows with step n), at the end 0.057 vs
  real learning 0.012 (4.7x). Documented in work_docs/overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md
  (user RENAMED it 10-23; sections 11-21 + Context paragraph on every doubt).
- **EVAL BUILD (10-23):** our model = HF Qwen3ForCausalLM exactly (QK-norm gamma before RoPE; only QK eps 1e-6 vs config 1e-5).
  eval/export_to_hf.py (param order: per layer q_gamma, k_gamma, norm, wQ, wK, wV, wO, mlp norm, gate_up, down; Linear
  stored [in,out] -> transpose) + eval/verify_export.py. **11590 export PROVEN: HF val 0.850077 vs trainer 0.850100.**
  4x disk 98% full (teammates) -> everything eval goes to RAM /dev/shm/gautam_eval (335 GB tmpfs); box rivalenv got
  transformers 5.20.2 (dtype= not torch_dtype). **CLEANUP after eval (user asked):** /dev/shm/gautam_eval, /root/gautam_k/eval,
  and any lm-eval caches put under /dev/shm.
- **10-23 eval results so far:** both exports proven (6000: HF 0.859864 vs trainer 0.859895). Dev suite (chat template, lm-eval
  0.4.13, GPU 3): no task beyond 2-SE noise; PAIRED per-question test: HellaSwag +1.07 (z +6.0, 11590 better), TruthfulQA MC2
  -0.92 (z -4.1, 6000 better), rest null; end-token rate 0.88 (6000) / 0.90 (11590), median len 257 = real 256, ~10% greedy
  repetition loops. DROP in chat format is broken (stops at "."): 0.046 meaningless. User cut the plain pass and the base-model
  eval ("not needed"). Chosen: 11590. Unseen: agieval_en 0.183 (chance level), mmlu_pro RANDOM subset 150/subject (seed 1023,
  full run projected ~5 h) running. Bucket is PUBLIC; plain curl works: .../buckets/gautam1926/checkpoints_sft/resolve/<path>.
- **10-23 20:05 EVAL DONE:** mmlu_pro random subset 0.0971+-0.0065 (chance 0.10), agieval_en 0.183 (chance) -> unseen suite at
  chance for 1.7B (mmlu_pro may also lose answers to "The answer is (X)" extraction, not checked). All 79 result files pulled to
  laptop post_training_research/tulu3_sft_1022/eval/box_results_1023; box /dev/shm/gautam_eval + /root/gautam_k/eval DELETED
  (user's OK). Handover zip for the DPO colleague: post_training_research/tulu3_sft_1022/handover_dpo_tulu3_sft_1022.zip.
  Hyperparameter doubts doc: work_docs/HYPERPARAMETERS_CONFIGS_SELECTION_FOR_MODEL_TRAINING_AND_FACTORS_AFFECTING_THESE.md.
- **P032 = old 8x node (innmi1srh2-p032) being replaced: CLEANED 10-23 19:10** (user ran rm -rf /localstorage/gautam_k). Kept
  50.6 MB in laptop ~/Downloads/p032_backup_2026_10_23 (compute_dpo_reference_logprobs.cpp: untracked, author unknown;
  v2_work/snapshots kernel versions; logs/results). kk/...goutham... files there are a colleague's: not touched.
- **10-23 EOD backup (4x box may close):** laptop `~/Downloads/box_backup_2026_10_23/` = rope_opt, gautam_k_logs (4x run +
  watchdog scripts, 4 HF logs), sft_data_small (val + small train index files), sft_data_tulu3_noaya_md5.txt (all 12 files).
  The 3 big train files (inputs/labels/meta.json, 1.98 GB) NOT copied (laptop link ~100 KB/s): rebuild with prepare_sft_data.py
  (d4d8f8b1, revision d91a0785) and check against the md5. Run logs already in box_backup_2026_10_22/tulu3_run_final.
  Checkpoints 6000/9000/11590 safe in the public bucket. Work doc §45.70-45.79 = all of 10-23.
Related: [[project_sft_benchmarks_kathirvel_1021]].
