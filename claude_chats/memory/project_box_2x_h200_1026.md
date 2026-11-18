---
name: project_box_2x_h200_1026
description: "GONE (instance cut 10-26 22:20 IST) — 2026-10-26 box for the matmul audit — 2x H200 ssh 95.133.254.29 (loud-sun-glows-fin-03), OUR GPU = 0; GPU 1 busy with a teammate's runner_ddp"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-26T13:32:16.098Z
---

**The box:**
- Connected by the user on 2026-10-26 ~19:00 IST through a VS Code Remote-SSH window (screenshot): `SSH: 95.133.254.29`, hostname `loud-sun-glows-fin-03`, 2x NVIDIA H200 (143,771 MiB each), driver 580.178.04, CUDA 13.0.
- The box clock is UTC: 13:30 there = 19:00 IST.

**GPU state at connect:**
- GPU 0: 0 MiB, 0% busy, free.
- GPU 1: 10,225 MiB used by pid 700067 `...hesh/scratch/train/runner_ddp`, a teammate's job.

**Decision:** OUR GPU = 0. Never touch GPU 1. Pin every run with `CUDA_VISIBLE_DEVICES=0`.

**Why:** this is the box for step 1 of the matmul epilogue audit (C1-C3 nsys timing, [[project_matmul_epilogue_audit_1024]]).

**How to apply:**
- The user runs every box command ([[feedback_user_runs_all_box_commands]]). Give short PULL / BUILD / RUN blocks.
- Ask before each GPU run, and say when GPU 0 is free again.
- Check what is on the disk before assuming any tree or venv exists.

**GONE:** the user cut this instance on 2026-10-26 ~22:20 IST after everything was copied to the laptop (see [[project_matmul_epilogue_audit_1024]]). Later that day GPU 1 was also ours. A new box will need the trees rebuilt from `~/Downloads/BluTrain_fuse`.
