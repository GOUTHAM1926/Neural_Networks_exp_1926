---
name: project-trainer-parity-1p7b
description: CORRECTED 2026-08-28 - the 2026-08-27 "parity" was cuBLAS measured against itself; the real 1.7B trainer result is 98.36% free / 99.83% locked, and the power-to-clock mechanism DOES reach the application
metadata:
  type: project
---

**THE 2026-08-27 PARITY CLAIM (§H27bj, 100.01% free / 99.94% locked) IS VOID.** Both arms ran
cuBLAS. `WITH_BLUBLAS` is baked into a single `libtensor.so`; building `=1` then `=0` overwrote
the first with the second. The tell was in the numbers — four runs agreeing to **5 tok/s** is not
repeatability, it is one binary four times. See [[feedback-backend-ab-snapshot]].

**CORRECTED, MEASURED 2026-08-28 (§H28b) on `hazy-earth-laughs-fin-03`, CUDA 13.1, GPU 0.**
15 steps per arm, unprofiled, 60 s pre-warm, all three backend checks green.

    steady state (step>=3)   cuBLAS      BluBLAS     delta     %cuB
    free clocks             28,215.91   27,751.98   -463.93   98.36%
    locked 1400 MHz         24,929.96   24,887.46    -42.50   99.83%

Slower on **every single step** at free clocks (range -360 to -518), so not noise. Loss is
bit-identical at step 0 and diverges only ~0.01 by step 14 — numerics are sound, the gap is speed.

**THE MECHANISM DOES REACH THE APPLICATION — §H27bj concluded the opposite and was wrong.**

    LOCKED 1410: same clock, we draw +14.1 W median (+2.69%)
    FREE:        both pin the 700 W cap, sw_power_cap Active on 100% of samples,
                 our clock median 1785 vs 1815 MHz; time above 1830 MHz 23.7% vs 47.5%
    gap free 1.64%  -  gap locked 0.18%  -  clock deficit 1.93%   <- the arithmetic closes

§H27bj recorded 665 W under a 700 W cap and reasoned the trainer is never cap-limited. Measured
correctly it sits at **691-693 W with the cap Active 100% of the time**.

**KERNEL LEVEL (§H28d, nsys 5 steps):** GEMM +6.12%, everything else +0.82% on byte-identical
code (an independent read of the same clock effect). GEMM is 37-38% of kernel time.
**The 192x192 family alone is 73.3% of the GEMM deficit** at +11.07% / +13.67% per call.
See [[project-cluster-orientation-192]].

**STILL TRUE FROM BEFORE:** this is the BASELINE dispatch (tuned rungs compiled, not selected);
historical trainer figures 93.8/96.4/98.4% are NOT comparable (script changed); shards must live
in `/root/gautam_k/blublas/data` and no stray `*.bin` may sit there; optimizations must be synced
into the VENDORED `BluTrain/Tensor-Implementations/BluBridge-BLAS/`.
**Build correction:** use the toolkit that compiled the objects — on this box that is CUDA
**13.1** (`/usr/local/cuda` -> 13.1, `.bashrc` puts it on PATH). Forcing 13.0 gives
`nvlink fatal: input file newer than toolkit (131 vs 130)`.
See [[project-power-clock-solved]], [[project-open-tasks-2026-08-28]].
