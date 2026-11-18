---
name: feedback_gpu_free_check_every_run
description: Never run on a GPU that has another process; check right before EVERY launch (in the same command) and abort if busy; keep monitoring
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-26T15:51:11.564Z
---

User, 2026-10-26 (2x H200 95.133.254.29, both GPUs given to us):

> "if someone is running on any GPU, don't run on that GPU; run on whatever is free and whenever it is free; don't run on top of something, our tests will be spoiled ... keep on monitoring"

**Why:**
- A teammate's job shares the box. Running on top of it spoils our timings (the measurements become invalid) and slows their job.
- Earlier that day, GPU 1 went from a teammate's 97.6 GB job to empty and back. Who is on a GPU changes during a session.

**How to apply:**
- Every GPU launch command starts with a check of `nvidia-smi -i <g> --query-compute-apps=pid --format=csv,noheader`. If it shows any process, abort or pick the other free GPU.
- The check goes in the SAME ssh command as the launch, not a check from minutes earlier.
- Long scripts check before each run inside them (as `nsys_audit.sh` does).
- Monitor while running (background watcher) and report the GPU state.
- Say when a GPU is released.
- Related: [[feedback_ask_before_every_gpu_run]], [[project_box_2x_h200_1026]].

**2026-11-03 P032 incidents (shared box, others' short jobs):** (1) GPU 3: Dhana's dense_bwd_bench starts and stops every
few seconds, so it looked free between jobs; our tests started there twice and overlapped ~15-20 s (14:25). Exclude a
GPU that anyone uses intermittently, even if it looks free at the check. (2) GPU 4: the free check passed at 14:30:45
but nsys needed >2 min to start the program; our kernels ran 14:33:09-14:33:12, after Surendra's job started there
(14:32:26), ~2.5 s overlap. The check must happen right before the GPU work begins: avoid nsys on a shared box, or make
the test program itself check its GPU (no other compute process) at the moment it starts. Tell the user each overlap.
