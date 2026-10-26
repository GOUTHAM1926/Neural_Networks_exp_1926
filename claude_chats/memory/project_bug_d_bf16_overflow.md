---
name: project-bug-d-bf16-overflow
description: "HIGH PRIORITY — Types.h float_to_bfloat16 clamped at fp16's limit (>=65,536 -> Inf, CPU AND GPU); clamp removed on the box 2026-09-26; per-arch intrinsic research DONE; the fp16 converter is ALSO wrong (100.8M of 2^32)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-26T09:34:56.421Z
---

**Bug D.** `Tensor-Implementations/include/dtype/Types.h:85`, `detail::float_to_bfloat16`: `if (exponent > 0x8E)` returned ±Inf.
- `0x8E` = 127+15 is fp16's limit. bf16 is finite up to ~3.39e38. Old code: 1,878,982,656 of 2^32 wrong; clamp removed = 0 wrong (CPU exhaustive, 2026-09-26).
- It is `__device__ __host__`. Kernels templated on `OwnTensor::bfloat16_t` run this manual code ON THE GPU (SASS: `ISETP.GT.U32 ... 0x8e`, 17 instr + branch per convert vs 1 for the intrinsic).
- **Status 2026-09-26:** clamp removal applied on box `fair-soul-wanders-fin-02` (user's patch). Build state unknown; needs a CLEAN rebuild. Intrinsic patch + GPU exhaustive test NOT yet on the box.

**Per-arch research, done 2026-09-26** (headers + official PTX ISA + real SASS for sm_75/80/86/89/90a/100a/103a/120; evidence in `profiling_reports/bugd_2026_09_26/per_arch/`):
- fp32→bf16: `__float2bfloat16_rn` = 1 instr (F2FP.BF16) on sm_80+. On sm_75 CUDA's own software = 14 ops, no branch; ours = 18 with a branch. So use it on ALL device archs; the `>=800` guard is not needed.
- bf16→fp32: keep `b<<16` everywhere. It is 1 instr on every arch (PRMT/SHF). PTX says `cvt.f32.bf16` gives an "unspecified NaN"; the shift is exact.
- fp32↔fp16: `__float2half_rn` / `__half2float` = 1 instr on every arch sm_75+.
- f16↔bf16 direct: `cvt.bf16.f16` / `cvt.f16.bf16` (F2F, 1 instr) is sm_90+ only (ptxas errors on sm_89). No C++ intrinsic exists. Via fp32 = 2 instr, same bits (one of the two steps is exact).
- double→bf16: `__double2bfloat16` = 1 instr on sm_90+, 15-26 ops below. The struct does double→float→bf16 = double rounding (1+2^-8+2^-30 gives 0x3f80, correct is 0x3f81).
- CUDA software NaN output = 0x7FFF for both bf16 and fp16. Ours is sign|0x7FC1 (bf16).

**fp16 converter is ALSO wrong** (reverses the old "leave :164 as is"): `float_to_float16` gets 100,815,872 of 2^32 wrong. Two references (GCC _Float16, CUDA host) agree on all 2^32.
- 100,662,272 are subnormals truncated.
- 30,720 are ties rounded up instead of to even.
- 122,880 are a mantissa carry ORed instead of ADDed, so the value HALVES.
- A rewritten RNE version tested 0 wrong. `float16_to_float` is correct (0 of 65,536).

**Side findings (code-read, not run, NOT Bug D):** GPU `as_type` (`convert_type_kernel`) routes every type through a `float`, so int64/double lose precision past 2^24. Complex→complex on the GPU is silently skipped, and the output stays uninitialised.
- **User decision 2026-09-26:**
  - `ConversionKernels.cu` = DONE 2026-09-26 (lib 3a08ade), see [[project-conversion-kernels-pushed-0926]]. (Was: FUTURE task.) Fix sketch:
    - direct `(Dst)src` when neither side is 16-bit;
    - the one-rounding ctors when one side is bf16/fp16;
    - throw for complex;
    - first check every BluTrain `as_type` call site.
  - fp4.h double rounding: LEAVE, do nothing.
  - long double: leave.

**FINAL patch folder (local, 2026-09-26):** `profiling_reports/bugd_2026_09_26/final/`.
- Box-bound files: `patch_conversions.py`, `test_types_conversions_gpu.cu`, `test_conversions_e2e.cu`.
- `review/` holds the 4 patched files, `conversions.diff` (537 lines) and the CPU PASS log. Read only, never copied to the box.
- The patch edits 4 files. `ReductionImpl.h` was added after the review: 7 CPU mean sites, one rounding.
- NOT applied on the box yet.

**Patch state 2026-09-26 (scratchpad `bugd/patch/`):** `patch_conversions.py` edited 3 files (later 4, see above).
- Files: Types.h, Vectorized.h store_float_as_bf16 (CPU truncation → RNE), Activations.cpp (3 CPU GELU truncations).
- Local checks:
  - CPU test vs 7 references PASS (13.4B+ inputs);
  - whole-library compile 0 fails (planted-error control OK);
  - ScalarOps sm_90a: 4,416 clamp compares → 0, instructions −48%.
- bf16→fp32 "3 branches → shift" is a cleanup, NOT a fix (old code was correct).
- Pending: the independent reviewer, then the box steps (user runs each, with permission).

**Why:** the user asked for a strong note (2026-09-24). It is a silent wrong answer in a header every kernel includes.

**How to apply:** the final Types.h patch has 3 parts:
- on the device, use the intrinsics (not for bf16→fp32);
- on the host, use our fixed manual (clamp removed, fp16 RNE rewrite, NaN = 0x7FFF);
- then a clean rebuild, the GPU exhaustive test, the SASS census, and a trainer check.

Only with the user's permission for every box step ([[feedback_ask_permission_for_every_box_action]]). Doc: loss doc §17.
