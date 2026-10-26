---
name: project-sm90-cta-fills-sm
description: One SM90 GEMM CTA consumes 100% of an SM's shared memory and 98.4% of its registers, so NO concurrent kernel can ever co-reside — this kills every stream-overlap idea before it is built
metadata:
  type: project
---

**One of our SM90 GEMM CTAs occupies an entire SM.** Measured from the nsys capture
(`CUPTI_ACTIVITY_KIND_KERNEL`: `dynamicSharedMemory`, `registersPerThread`):

    dynamicSharedMemory   229,376 B  (sharedMemoryExecuted 233,472 = 228 KB) of 228 KB  = 100.0%
    registers             384 threads x 168 regs = 64,512 of 65,536                     =  98.4%
    grid                  132 CTAs on a 132-SM H200                                     =  1 CTA/SM

Only **1,024 registers** remain on the whole SM — 32 threads at 32 regs.

**Why this matters far beyond one experiment: concurrent-kernel overlap with our GEMM is
CAPACITY-IMPOSSIBLE, not merely small.** A second kernel's CTA cannot begin until one of ours
retires and frees its 224 KB. That applies to another GEMM, to a `convert_type_kernel`, to
anything. Overlap is limited to TAIL-FILLING as CTAs drain. cuBLAS makes the identical choice
(`grid(2,66,1)` = 132 CTAs, block 384), so it would hit the same wall.

**Measured consequence (§H27zX, H200, 36-layer 1.7B, 10 steps x grad_accum 4):** the backward
secondary stream was worth +0.42% tok/s at locked 1400 and **-0.01% at free clocks** (production),
against a ~0.3% noise floor, while making our own dgrad 9-11% and wgrad 5% slower and putting
3.4 points of distortion on every per-kernel comparison. Default flipped to single stream;
`BLUTRAIN_BWD_SINGLE_STREAM=0` restores two streams.

**How to apply:** before proposing ANY multi-stream / concurrent-kernel overlap on SM90, read the
occupancy first. If a CTA fills the SM, the answer is no — the win has to come from ELIMINATING
work, not overlapping it. The live example: BluTrain spends **23.6% of GPU time** on dtype
conversion (`convert_type_kernel` bf16<->fp32, 61,056 launches, 8,666 ms), and the fix is to fuse
the fp32 output into our epilogue (our accumulator is already fp32), not to overlap the converts.

Related: [[project-blutrain-integration]] [[project-pdl-everywhere-future]]
[[feedback-isolated-measurements-mislead]]
