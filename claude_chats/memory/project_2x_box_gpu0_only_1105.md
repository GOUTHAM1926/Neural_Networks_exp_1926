---
name: project_2x_box_gpu0_only_1105
description: "From 2026-11-05 14:30 IST: on the 2x H200 box 86.38.238.51 OUR GPU is ONLY GPU 0; GPU 1 is Ananya's (she uses it now and then) -- never launch on GPU 1, even when it looks free"
metadata:
  type: project
---

User, 2026-11-05 ~14:30 IST: "from now, ours is only gpu 0, ok? not gpu 1, its used by ananya from now and then".
Context: that day Ananya's jobs (/root/Ananya/claude_1105/{plain,fused}/BluTrain, ./_jit_runner_) came and went on
GPU 1 during our runs; her load on GPU 1 moved our GPU-0 step time by ~66 ms (see [[project_pretrain_before_after_1105]]).
**Why:** GPU 1 belongs to a colleague; her jobs start without warning.
**How to apply:** every launch on 86.38.238.51 pins CUDA_VISIBLE_DEVICES=0; scripts must not auto-pick GPU 1 when it
is free (run_pz.sh / nsys_pz.sh picked any free GPU -- change that before reuse); still check GPU 0 is free before
each run, and note GPU 1's state (her load affects our timings).
