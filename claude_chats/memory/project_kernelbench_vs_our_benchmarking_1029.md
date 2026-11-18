---
name: project_kernelbench_vs_our_benchmarking_1029
description: KernelBench / KernelBench-Verified pitfalls compared with our GEMM benchmarking (10-29); we never ran KernelBench; our harness L2 flush is default OFF
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T10:30:27.450Z
---

On 2026-10-29 the user asked whether we benchmark correctly, prompted by a towardsdatascience blog, "AI agents beat
PyTorch writing faster CUDA kernels". The blog itself gives HTTP 403 to automated fetches, so I never read it.

**Facts checked:**
- **KernelBench** (arXiv 2502.10517, Feb 2025): 250 PyTorch tasks; AI-written kernels are graded on correctness and
  speedup over PyTorch (fast_p).
- **KernelBench v0.1 blog:**
  - timing on the wrong stream gives fake-short times;
  - microsecond tasks are dominated by launch overhead, so tasks were resized to 1-15 ms on H100;
  - broken tasks were fixed;
  - inputs are now mixed distributions;
  - a speed-of-light check was added.
- **KernelBench-Verified** (arXiv 2607.16241, June 2026):
  - fixes: a TF32 eager baseline and hidden tests (inputs x3, x0.01, x-1);
  - GPT-5.5 geometric mean drops from 1.43x to 0.88x;
  - 28% of kernels use more memory;
  - timing is 3 warm-up runs + 10 trials with CUDA events and NO L2 flush.

**Us:**
- We never ran KernelBench; it is not in our docs or memory.
- We compare against cuBLAS, FA3/FA4 and cuDNN (stronger than eager PyTorch), with bit-exact correctness, free
  clocks, one config per process, paired crossovers, and end-to-end trainer tok/s.
- Known gap: `test_bgemm_sm90_matrix.cu` §H27zQ `BLUBLAS_L2_FLUSH` is default OFF.
  - Warm L2 inflates the standalone %cuBLAS: tn shapes read ~100% on the bench but 83-94% in the trainer.
  - The polluter test cut our lead from 102.15% to 100.04% (§H161, parked).

**How to apply:** if asked again, take from KernelBench:
- speed-of-light checks;
- scaled/negated-input correctness tests;
- L2 flush ON for standalone claims;
- peak-memory reporting.

Trust the trainer end-to-end numbers over the standalone ones. Related: [[project_l2_pollution_parked]],
[[project_test2_measurement_traps]].
