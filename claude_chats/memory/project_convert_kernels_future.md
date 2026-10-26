---
name: project-convert-kernels-future
description: FUTURE TASK (parked 2026-08-26) — the bf16<->fp32 convert kernels run at 35% of HBM peak on H200 and a vec8 replacement already exists gated to sm_103; parked because it is BluTrain work and helps cuBLAS equally
metadata:
  type: project
---

**SUPERSEDED 2026-10-21:** H100/H200 already have vector casts (`ConversionKernelsSM90.cu`, kathir d8506d5, 08-06, default on
Hopper when 16-B aligned); see [[project_future_pretraining_tasks_1021]]. The text below is the 08-26 view.

FUTURE TASK, deliberately parked 2026-08-26. All findings are in §H27zZ and §H27aa.

**WHY PARKED — the reasoning is strategic, not technical.** The work is real and large, but:
1. It is **BluTrain-side, outside BluBLAS** — the casts are written explicitly in
   `train/production/bluscript.cpp` as `autograd::to_dtype`, and the training script is not
   fixed yet and may change.
2. A **bf16 matmul should output bf16** — that is the correct library contract. An fp32-output
   epilogue is a scope change, not an optimisation.
3. **Decisive one:** vectorising the converts speeds up the cuBLAS build EXACTLY as much as ours,
   because both backends pay the identical converts. It improves the product but moves us ZERO
   against cuBLAS, and beating cuBLAS on the GEMM is the mission.

**WHAT IS ALREADY ESTABLISHED, so it need not be re-derived:**
- Converts are 23.60% of trainer GPU time: 61,056 launches, 8,666 ms (locked) / 7,422 ms (free).
- They achieve **1,656 GB/s locked / 1,934 GB/s free = 35-40% of H200's ~4,800 GB/s peak**, moving
  14,354 GB. `convert_type_kernel` is scalar, one element per thread.
- A **vectorized vec8 replacement already exists** in `ConversionKernelsSM103.cu` (bf16->fp32
  5,380 GB/s on B300 = 2.52x PyTorch) and its own header says it uses "only vectorized loads +
  bf16 intrinsics, no tcgen05 / TMA / inline PTX" — nothing requires Blackwell. It is gated on
  `is_sm103_device()` in `ConversionKernels.cu:157-198` because that is where it was TUNED.
- Site map: **6 of 7 forward sites genuinely consume fp32**, so an fp32 epilogue would remove
  their up-cast rather than move it. `wV` is the exception and is PURE WASTE — cast to fp32 at
  bluscript.cpp:258 and straight back to bf16 at :319 with no fp32 consumer between.
- Backward: `AutogradMeta.cpp:116,158,173` up-casts bf16 grads to fp32 for accumulation.

**How to apply:** if this is ever revived, do the vec8 enablement first (cheapest, no kernel
redesign), and remember the wV round trip is a trainer one-liner independent of everything else.
Related: [[project-sm90-cta-fills-sm]] [[project-blutrain-integration]]
