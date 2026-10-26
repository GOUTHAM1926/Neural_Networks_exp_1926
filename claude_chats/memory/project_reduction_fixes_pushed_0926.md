---
name: project-reduction-fixes-pushed-0926
description: "7 GPU/CPU reduction bugs fixed + pushed 2026-09-26 (lib 97735ad, BluTrain f1618d8); the June fixes had NEVER reached the team repo; 43/43 GPU+CPU, sanitizers clean"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-26T16:55:24.543Z
---

**Pushed on 2026-09-26.**
- Library `Tensor-Implementations` `dist_side_merge`: `7810aea..97735ad`, 4 files: `ReductionImplGPU.cu`, `ReductionKernels.cuh`, `ReductionOps.h`, `ReductionImpl.h`.
- BluTrain: `23ae1e5..f1618d8` (the submodule bump). Committed with the company identity.

**The 7 fixes:**

| # | What was wrong | Fix |
|---|---|---|
| 1 | Mean config/shared memory sized with the input type, not `arg_t` | Crash for bf16/fp16/int means; now `arg_t` |
| 2 | `block_x_reduce` treated the 2-D block as 1-D | Wrong last-axis results; now PyTorch's algorithm |
| 3 | Integer mean `factor = int64(1/n) = 0` | Now `MeanOps::FactorT` = double; the output is Float64, bit-identical to the CPU (user chose fp64 after the PyTorch/NumPy/TF/JAX source review) |
| 4 | GPU middle axis: config reset `num_outputs` to `inner` | Now `outer*inner` plus a `ReduceOp::outer_stride` slab remap |
| 5 | GPU Generic path had empty OffsetCalculators | New `setup_generic_path`; refuses more than 10 dims per side |
| 6 | CPU middle axis wrong | `cpu_demote_middle_to_generic` at all 6 CPU sites |
| 7 | Multi-CTA combine: `block_y_reduce` → `block_x_reduce` with no barrier | One `__syncthreads()`; racecheck went from 11,521 errors to 0 |

**Key history:** fixes 2, 4, 5 and 6 are the user's June 2026 fixes (`master_gau_latest_sm_86_RTX_3060` / BBP; see [[project_reduction_bugs]]). They were NEVER committed to the team repo. The 06-11 "merge the v2 compiler" commit (3160082) brought in the 1-D `block_x_reduce`, and every tree since carries it.

**Verification:**
- 43-case test, one process per case, GPU AND CPU against an exact double reference.
- Before: GPU 9 ok / 13 wrong / 11 crash (of 33); CPU wrong on all 6 middle-axis cases.
- After: 43/43. memcheck 0; racecheck 0.

**Not measured:**
- Speed (no A/B was run).
- argmax/argmin and variance over a middle axis.

Everything is in `profiling_reports/reduce_gpu_2026_09_26/`: the 3 patches, the test, the run script, before/after logs, the reference sources, and orig/patched copies.

**Tell colleagues:**
- The GPU integer mean now returns Float64 (was Float32, and its values were 0).
- They need `git submodule update --init` plus a libtensor rebuild. Use `make -j$(nproc)`; `ReductionImplGPU.cu` alone takes more than 4 minutes.
