---
name: project_4b_variants_shapes_observed
description: 4B's 8 variants and 15 shapes are now confirmed by DIRECT OBSERVATION, and the trainer's call contract is batch=1/alpha=1/beta=0/packed
metadata:
  type: project
---

2026-09-10, §H211, box 86.38.182.106 (since released). A 20-step nsys run plus a
`BLUBLAS_SM90_TRACE=1` capture replaced three separate INFERENCES with observation.

- **8 variants**, all cluster 1x1, GEMM = 55.70% of GPU: NN/NT/TN 128x256x64,
  NT/TN 192x128x64, TN 192x192x64, NN/TN 256x128x64. The 128x256 family alone is 45.4% of GPU.
- **15 shapes**, 15/15 identical to `tools/h147_cases.txt` -- the stored list is vindicated.
- **The call contract is `batchCount=1, alpha=1, beta=0, lda=ldb=ldc=0, strideA=strideB=0`**
  on every one of the 15. I had flagged that a bench at batch=1/beta=0/packed might differ
  from the trainer's real call and invalidate every standalone number ever taken. **REFUTED** --
  the old h92-style benchmarks were signature-correct.
- 5 of 8 variants serve ONE shape and are pinned by time alone; 3 of those reproduce §H203's
  in-trainer TFLOPS to within 1.1 (720.2/679.7/741.7). **The three 128x256 variants each serve
  several shapes, so no per-shape in-trainer timing is recoverable for them by any timing
  argument** -- that is what the NVTX join (`BLUBLAS_SM90_NVTX=1`, committed) is for.

**Why:** stops the recurring re-derivation of shapes/variants, and closes the call-signature
question permanently.

**How to apply:** treat the 15 shapes and 8 variants as settled. Do NOT re-derive them. If a
per-shape in-trainer number is needed for a 128x256 site, run the NVTX profile -- launch-count
pairing cannot separate those sites. See [[project_4b_standalone_wins_trainer_loses]] and
[[project_power_clock_solved]].
