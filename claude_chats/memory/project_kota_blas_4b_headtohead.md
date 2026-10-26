---
name: project-kota-blas-4b-headtohead
description: Kota's BLAS relinked into OUR BluTrain on 4B gives 30,078 tok/s (98.4% of cuBLAS on GEMM) vs our 28,703 (89.1%); his absolute 32.4K claim does NOT reproduce here but his near-parity ratio does.
metadata:
  type: project
---

Measured 2026-09-01, H200 x2, ZeRO-1 NP=2, 5 steps, same box/day/script for every arm.
**Pure relink**: identical `_zero_runner` and `libtensor.so`, only `libblublas.so` swapped.
Verified at runtime by reading /proc/<pid>/maps on both workers and md5-matching the mapped
file. Kota's tree = `BlubridgeAI/BluBridge-BLAS` @ `ea666ab` (2026-08-31), built with his own
Makefile (14 files get sm_90a via his per-directory gencode); all 12 symbols libtensor needs
are exported.

    arm            tok/s     MFU      total GPU   GEMM      GEMM %cuB   e2e %cuB
    cuBLAS         30,299   44.45%      349.6 s   153.5 s      --         --
    KOTA           30,078   44.13%      352.1 s   156.0 s     98.4%      99.3%
    ours pinned    28,703   42.11%      367.9 s   172.2 s     89.1%      94.7%
    ours pre-pin   27,247   ~40%        387.6 s   190.4 s     80.6%      89.9%

Non-GEMM time is 196.1 / 196.1 / 195.7 s across cuBLAS / Kota / ours — identical, which is the
control proving the relink is clean and ONLY the GEMM differs.

**His 32.4K claim does not reproduce in absolute terms** (we get 30.1K) but neither does his
32.2K cuBLAS baseline (we get 30.3K). Both his arms sit ~7% above both of ours, so his
environment is faster across the board — a different BluTrain or box, NOT a BLAS effect. The
like-for-like result is what matters: **his BLAS is 4.8% ahead of ours end-to-end and 10.4%
ahead on GEMM.**

His tile mix is completely different from ours and from cuBLAS's: **81.6% of his GEMM time is
128x256** (three configs), where cuBLAS and we use 256x128 and 192x192. He also merges the two
cuBLAS nn kernels (counts 122,400 + 24,480) into a single 146,880-launch config. His template
parameter ORDER differs from ours, so cluster/raster cannot be read off by position.

Next: per-shape and ncu comparison to find what the 128x256 family is doing right.
Related: [[project_kota_sm90_architecture]], [[project_hybrid_blublas_plan]], [[project_4b_dispatch_gap]].
