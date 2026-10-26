---
name: feedback-samples-not-static-counts
description: Judge a kernel change by SAMPLE-WEIGHTED cost, never by static instruction counts - the two disagree by 100x
metadata:
  type: feedback
---

Static opcode counts do NOT predict time on the SM90 GEMM, and chasing them has wasted effort
four separate times. Measured on the V6 depth-4 STSM arm (2026-08-19):

    opcode   static   samples   % of kernel
    LDG        168        0       0.00%   <- predicated off at beta=0; costs NOTHING
    FFMA       336        0       0.00%
    F2FP       336      409       0.66%
    FMUL       672     1396       2.26%
    DEPBAR       6    11871      19.23%   <- 6 instructions
    BRA        212    19288      31.24%   <- 212 instructions

**218 instructions out of 4216 -- 5% of the code -- are 50.5% of the kernel.**

**Why:** always run the PC-sample census before proposing a fix. A large static count with zero
samples is a predicated-off branch or a cold path; a tiny static count with huge samples is a
stall. Instruction count went UP 14% on V3's SKEW adoption and it got FASTER.

**How to apply:** get `smsp__pcsamp_sample_count` per address, dedupe BY ADDRESS (ncu emits one
row per launch; summing multiplies by the launch count), and rank by samples. Only then decide
what to fix. See [[feedback-verify-claims-and-benchmark]], [[feedback-isolated-measurements-mislead]].
