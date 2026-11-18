---
name: project_box_4x_h200_1027
description: "From 2026-10-27: 4x H200 box root@95.133.252.206 (clear-cable-completes-fin-03), SHARED via /root/gpu_board/gpuboard claims; our folder /root/gautam_k; disk only ~11-13 GB free"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-27T05:45:42.779Z
---

**The box** (connected by the user 2026-10-27 ~11:10 IST, VS Code Remote-SSH screenshot):
- `ssh 95.133.252.206` (User root in ~/.ssh/config), hostname `clear-cable-completes-fin-03`, 4x H200 (143,771 MiB), driver 580.178.04 (CUDA 13.0), 176 CPUs, 668 GB RAM.
- nvcc: `/usr/local/cuda` -> 13.2 (13.1 also there). The build is SASS-only `-gencode arch=compute_90a,code=sm_90a`, so 13.2 on the 13.0 driver is fine. nsys 2025.5.2; NVTX headers come from `/opt/nvidia/nsight-compute/*/.../nvtx/include` (Makefile finds them).
- yaml-cpp 0.8.0 is installed system-wide (libyaml-cpp-dev), no own copy needed.
- **Disk: `/` 193 GB, only ~11-13 GB free (94%).** Check `df -h /` before big writes; never keep sqlite exports or many binaries there.
- Box clock is UTC (IST = +5:30).

**GPU sharing protocol (MANDATORY, `/root/gpu_board/PROTOCOL.md`):**
- Owner = top folder under /root -> we are `gautam_k`.
- Before ANY GPU use: `/root/gpu_board/gpuboard --as gautam_k status`, then `claim <gpu> <minutes> "<what>"`. Exit code 2 = REFUSED -> do not start; tell the user who holds it.
- Run only with `CUDA_VISIBLE_DEVICES=<claimed>`; check `inbox` while running; `release <gpu>` as soon as done or crashed.
- Never kill/renice anyone else's process. Teammates on it 10-27: aparna (GPU 0, nsys), someone's `jit_runner` (GPU 3, 97 GB), folders Ananya, Surendra, adwaid, aparna, dhana, harshini, madhu, mahesh.
- LAPTOP_SETUP.md offers a laptop hook (`laptop_hook.sh` into ~/.claude/settings.json) that enforces this; NOT installed (user decision).

**Our layout** (`/root/gautam_k`):
- `audit/tree_new` = copy of laptop `~/Downloads/BluTrain_fuse` (C3/C1 switches, no .git); `audit/build_all.sh` builds it (`make -j64 profiler tensor dist SM_ARCH=90` + `_ablation_compile ABLATION_OUT=audit/pz_new`), status in `audit/build_status`.
- `audit/box` = the 10-26 scripts; `audit/runs` = results; `pt_data/edu` = the 4 edufineweb train shards + val (954 MB, md5 = laptop `master_gau_latest/data`); EVERY domain in `pt_data/mix/{code,commoncrawl,edufineweb,math}` links ALL 5 files (as on 10-26). The 10-21 backup layout (1 shard per domain) gives a DIFFERENT step-0 loss (11.288469) — that was the first smoke's mismatch.
- Only ONE tree (no separate OLD tree): on 10-26 the switch-off arm was proven equal to the old tree (bit-identical step 0, +0.23% on the same steps).

**How to apply:** claim on the board before every GPU command (in addition to [[feedback_ask_before_every_gpu_run]] and [[feedback_gpu_free_check_every_run]]); pull results off the box after every run ([[feedback_pull_run_artifacts_immediately]]). Work continues per [[project_matmul_epilogue_audit_1024]].

**Smoke gate PASSED 2026-10-27 11:41 IST** (GPU 1, `runs/ab_fuse_np1_g1_1027_1134`, pz_new 7045ca15, libtensor fed5e960, CUDA 13.2): step-0 loss off 11.286944 / c3 11.287024 = 10-26 exactly; step 2 21,328 / 21,303 ms. Steps 0-1 are slow here (27-38 s vs ~22 s on 10-26), step 2 on is normal -> compare steps >= 2 only.
