---
name: project-4b-trainer-99p8-production
description: 4B trainer is at 99.80% of cuBLAS on the PRODUCTION schedule (2026-09-12, 2xH200) - up from 96.07%. BLU_MAX_STEPS=30 costs cuBLAS 1.86% and us only 0.63%, so short-schedule numbers understate us by ~1.2 pts.
metadata:
  type: project
---

Box `sharp-fish-circles-fin-02`, 2x H200. 4B, B=2 T=4096, global_batch 524288, ZeRO-1 v2, world=2.

**PRODUCTION schedule (warmup=200 / steps=2000):**

| | step 16 | mean 10-16 |
|---|---|---|
| cuBLAS | 35,812.5 | 35,839.2 |
| ours | 35,773.4 | 35,767.9 |
| **gap** | **-0.11%** | **-0.20%** |

**99.80% of cuBLAS.** Previous best recorded was 96.07% - see
[[project-4b-standalone-wins-trainer-loses]]. That is +3.7 points.

**NEVER quote a `BLU_MAX_STEPS` number as a cuBLAS ratio.** It rewrites the LR schedule
(warmup=ms/10, stable=ms/2) and the cost is BLAS-SPECIFIC:

| arm | ms30 -> full |
|---|---|
| cuBLAS | **-1.86%** |
| ours | **-0.63%** |

So the short schedule flatters cuBLAS. Our gap reads **-1.34% at ms30** but **-0.11% at full**.
Same pattern as [[project-regime-effect-is-blas-specific]] (recorded 2.28% vs 0.53%).

**Short-schedule arm means (2 runs each, reproduce to <0.1%):** cuBLAS 36,401.2, ours 35,937.3.

**Absolute tok/s jumped ~28.7K -> ~35.9K but NONE of that is ours:** converts fell from 21.2-23.6%
of GPU time to **10.6%** because `vec8_convert_to_bf16` now runs on SM90 (Tensor-Implementations
master 1c84859). That is BluTrain-side and lifts cuBLAS equally - see
[[project-convert-kernels-future]].

**How to apply:** run both arms at the FULL schedule for any ratio claim. Both arms nsys-profiled
is fine. Compare at matched steps - see [[feedback-same-step-window-before-comparing]].
