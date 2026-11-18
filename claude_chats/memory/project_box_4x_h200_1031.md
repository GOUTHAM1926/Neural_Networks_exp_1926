---
name: project_box_4x_h200_1031
description: 4x H200 box root@95.133.253.118 given 10-31 for the long-context kernel work (P022/P032 unavailable); our layout /root/gautam_k/lc; setup via laptop bundle ~/Downloads/lc_box
metadata:
  type: project
---

2026-10-31 ~12:45 IST the user connected a 4x H200 box (VS Code Remote-SSH, `SSH: 95.133.253.118`; driver 580.178.04, CUDA 13.0 driver). P022 and the 8x nodes are NOT available. At connect time GPU 0 was busy (someone's `.../aude_1031/ab_1031_0639/runner`, 97.8 GB, 100%); GPUs 1-3 free. GPU 3 is ours (user 10-31: "let us run in gpu3"); check it is free in the same command as every launch.

My ssh from the laptop to this box was BLOCKED by the permission classifier -> the user runs every box command ([[feedback_user_runs_all_box_commands]]).

Setup bundle (laptop `~/Downloads/lc_box/`): `sync_to_box.sh` (laptop -> box: BluTrain_push without .git, flash-attention 94e22c9, the bundle) and `scripts/lc_setup.sh` (box, CPU only, ~25-30 min: make blublas/profiler/tensor/dist, trainer via sft/build_sft_trainer.sh, bin/tva, bin/bench_ce, rope/rivals/libropeours.so; envs benv (FA3+FA4, torch 2.14.1 cu130), lossenv (Liger, CCE), rope/rivalenv (torch cu132, liger 0.8.5, quack 0.6.5); status in /root/gautam_k/lc/setup_status; aborts if < 40 GB free). PROVENANCE.txt lists the code versions (BluTrain 334ecbb + local drop-only changes, TI 6aac3bf + test_varlen d75095b8).

**How to apply:** check the board/GPU-free before every GPU launch; don't touch GPU 0's job; pull results off the box after each run. Related: [[project_case5_long_questions_1031]], [[feedback_length_limits_follow_every_test]].

**Results 10-31 13:25 IST (records profiling_reports/long_context_kernels_2026_10_31/, work doc 45.113):** disk had 5.2 GB free -> setup OURS_ONLY (no rival envs, ~20 GB needed: user/box-owner decision). RoPE table builder confirmed NOT built at >= 65,536 rows (fix on laptop TI tree only, bit-identical below; table error ours ~2x HF's fp32 formula, 1.9e-2 at 128k). Attention G14 on real rows 8k-128k, 4/2 and 18/6: all pass except one known dV rounding outlier (1.169, 32k 4/2); longest chat 130,533 passes. Pending user: RoPE fix into lib, double-precision table, disk for rivals, then 135M trainer at long T.

**14:00-15:00 IST (CPU only, user: "prepare all CPU work, then tell me before GPU"):** RoPE fix + no --use_fast_math in laptop TI tree (NOT pushed; push next time). Audit: grid-y bugs = RoPE table/per-token kernels (T >= 65,536) + loss backward FALLBACK sparse_ce_backward_cuda_vec (rows >= 65,536; training uses the safe stats path); latent: embedding (>524,280 tokens), SwiGLU sm120. Candidates: RoPE v0-v3 (+CPU table), loss l0-l3, launch-check modes, fast math F0 (old lib) / F1 (BluTrain_new, built). GPU plan = lc_box/tests/ab_run.sh with SKIP_BUILD=1 after an ONLY_BUILD=1 run; NOT started. Work doc 45.114-45.116, plan Q152-155.
**15:15 IST: CPU work DONE; GPU part waiting for the user's word:** `setsid nohup env SKIP_BUILD=1 bash /root/gautam_k/lc/tests/ab_run.sh` (~1 h). Disk 1.9 GB free (others filling it): ALL big writes go to /dev/shm (trainer saves a checkpoint at its last step!). Never use $L/tmp for nvcc temp (it filled the disk).
