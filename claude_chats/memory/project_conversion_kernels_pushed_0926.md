---
name: project-conversion-kernels-pushed-0926
description: GPU as_type (ConversionKernels.cu) fixed + pushed 2026-09-26 (lib 3a08ade, BluTrain 4e92e6c); float-box removed for wide types, complex converted, 169/169 pairs vs CPU, speed unchanged
metadata:
  type: project
---

**Pushed on 2026-09-26.** Library `dist_side_merge` `97735ad..3a08ade` (only
`src/Kernels/cuda/misc/ConversionKernels.cu`); BluTrain `f1618d8..4e92e6c`. Company identity.

**The 3 fixes:**
1. `convert_type_kernel` put EVERY value through a float temporary. That broke several things:
   - int32 above 2^24, int64, uint32/64 and double lost digits;
   - negative int → unsigned went through an out-of-range float (int8 -1 → uint8 gave **0**, not 255);
   - int/double → bf16/fp16 was rounded twice.

   Now bf16/fp16 sources keep the float path (it is exact for them). Wide sources use either
   `static_cast<Dst>(x)` or the Types.h one-rounding bf16/fp16 constructors, exactly like the CPU
   `as_type` (AsTypeTensor.cpp).
2. complex → complex was silently SKIPPED (garbage output). It is now converted component by component,
   and complex ↔ real throws in `convert_type_cuda_generic` too.
3. The 32-bit index is now 64-bit.

**Test:** `profiling_reports/conversion_kernels_2026_09_26/test_as_type_gpu.cu`. It compares GPU against CPU
for all pairs, 131,072 values each, including every bf16/fp16 bit pattern.
- Before: 110 of 169 ok. After: 169 of 169.
- Timing: 9 hot pairs, all within 1%.

**Known and NOT changed:** fp32 subnormals (< 1.18e-38) flush to 0 on the GPU, because the whole library is
built with `--use_fast_math`. One way to change it: compile this one file without FTZ.

**Test traps hit:**
- My fp32 → complex case reused the complex buffer, so memcpy overflowed the heap ("double free"). Keep a
  buffer-size guard.
- The FTZ category hid the real fp64 → bool bug in the before-run (28,629). Treat FTZ counts that change
  between before and after as findings.

**BluTrain impact today:** none. Integer `as_type` there is only token ids and labels (< 16.7M, exact through
float). `DistributedNN.h:788/801/1613` and the CP script were checked.

## ALSO OPEN (found 2026-09-07, NOT fixed by the 09-26 push): as_type is STRIDE-BLIND (was memory project_as_type_stride_blind)

`Tensor::as_type` (`src/core/AsTypeTensor.cpp:11-41`) allocates a **contiguous** output of
`this->shape()` and fills it with a **flat** convert (`convert_type_cuda_generic(this->data(),
..., numel(), stream)`). No strides. So `as_type` on a non-contiguous view relabels the bytes
instead of transposing them. Same hole in `ToDtypeBackward::apply`
(`include/autograd/backward/ReshapeBackward.h`) and `CastBackward::apply`, which both call
`as_type` on whatever grad arrives.

`GenMatmul.cu:57-58` DOES materialize (`B.is_contiguous() ? B : B.contiguous()`). That is why
the fp32 weight-tying script the [[project_weight_tying_bug_doc]] validated was safe:
transpose -> matmul, matmul fixed the view. The bf16 `bluscript_zero*.cpp` inserts
`to_dtype` BETWEEN them, so as_type sees the strided view and matmul can no longer tell.

**Why:** forward then uses relabeled bytes while the backward (TransposeBackward +
the AutogradMeta layout contract) transposes correctly by strides -> forward and backward
disagree, and loss still descends (same fixed-permutation reasoning as the original bug),
so it will NOT show up as a blow-up.

**How to apply:** fix at the library boundary — `if (!is_contiguous()) return
contiguous().as_type(new_dtype);` at the top of `Tensor::as_type`. Do NOT "fix" it in the
script by casting before transposing: that only moves the strided input into
ToDtypeBackward/CastBackward, which call as_type too. `AttentionOps.cpp:503` already does
`query.contiguous().as_type(...)` per-site, evidence the hole was hit before.

Live only when `weight_tying == true`: ON in `blutrain_prod_local/bluscript_zero_600M.cpp:99`
and `profiling_reports/bluscript_zero.cpp:99`; OFF (dead branch) in `bluscript_zero(2).cpp:99`
and `bluscript_zero.cpp`. Related: [[feedback_no_redundant_boundary_fixes]],
[[feedback_verify_reachability_not_presence]].
