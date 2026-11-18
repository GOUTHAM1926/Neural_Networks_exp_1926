---
name: project_empty_cache_test_1104
description: "11-04 item 10 tested (P032 GPUs 4/6/7): empty_cache step 5 = pure cost (same memory, +6 s B=4 / +26 s B=20) -> remove; step 0 keeps headroom (+3.4 GB and slow steps without it at B=20) -> keep; nsys after Muon fix 8.41 ms idle/step"
metadata:
  type: project
---

Work doc 45.145. Test binaries on P032 bt_new: bluscript_sft_bin_ec0 (step 0 only), bluscript_sft_bin_noec (none).
Decision pending with the user: remove step 5 in bluscript_sft.cpp (and pretraining bluscript_zero_ckpt.cpp:3268?), keep step 0.
History: beda19d (bruce-yeager, 09-08) moved step 1 -> step 5 with AllocationTracker shutdown.
nsys after the Muon fix: idle 8.41 ms/step; biggest = Newton-Schulz launch-bound (GEMM/transpose/convert ~4.4 ms),
blocking copies 2.4 ms (items 5/6). Script nsys_gaps.py (P032 muon_guard_p032 + laptop lc_box/muon_guard_p032).
**Why:** decide by tests. **How to apply:** ask before any GPU run; next per-step item by measured idle.
Related: [[project_muon_gpu_guard_1104]].

**PUSHED 11-04 ~19:20:** BluTrain e9e310c (7 trainers: sft, production_zero zero_ckpt, root zero_ckpt + _bluccl/_fused/_multi/_elastic), verified on GitHub. Not changed: step-0-and-1 trainers (bluscript_zero.cpp, bluscript_single.cpp, train/production/bluscript.cpp: DDP bucket-rebuild reason). _elastic has an existing compile error (drain_prefetch) unrelated.
