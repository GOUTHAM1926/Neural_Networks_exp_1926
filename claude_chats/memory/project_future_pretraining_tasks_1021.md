---
name: project_future_pretraining_tasks_1021
description: "Pretraining follow-ups after the 10-21 hand-over: casts ALREADY vectorized on H200 (wiring verified), attention ahead of all rivals except a dense-fwd tie with FA3, PDL, GEMM hybrid paused"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-21T11:07:27.518Z
---

User, 2026-10-21 after the `dist_side_gautam_lottn` push: "note these as future tasks, we will do one by one". Work doc §45.63.

1. **Casts on H200: ALREADY FAST** (corrected 10-21) - `ConversionKernelsSM90.cu` (kathir d8506d5, 08-06): vec8 fp32->bf16,
   vec4 bf16->fp32, default on Hopper when 16-B aligned (`ConversionKernels.cu:198/:229`; off via BLUTRAIN_NO_SM90_CONVERT).
   Scalar `convert_type_kernel` = fallback only. WIRING VERIFIED 10-21 (to_dtype->as_type->generic->sm90; backward and ZeRO
   too; never disabled in production_zero; kernels in the box libs). Left: optional nsys alignment check. Old (outdated)
   note: [[project_convert_kernels_future]].
2. **PDL on the remaining kernels** - 0.06% under nsys = lower bound; decisive = unprofiled tok/s on vs off.
   Detail: [[project_pdl_everywhere_future]].
3. **Attention: already ahead of FA3/FA4/cuDNN everywhere except ONE tie** (corrected 10-21: ratios are ours/rival TIME,
   <1 = ours faster; sb_qpu_1019 summary_steady.txt): dense fwd vs FA3 1.004 (tie), bwd 0.968, f+b 0.976; FA4 0.919/0.989/0.972;
   cuDNN 0.863/0.891/0.884; packed faster than all. Open idea only for the tie: diagonal-first with QPU's loop order kept.
4. **GEMM hybrid** (our pins + Kota's dispatcher) - PAUSED since 09-17 by the user; never propose restarting unprompted;
   production uses Kota's BluBLAS ea666ab. See [[project_hybrid_blublas_plan]], [[project_state_2026_09_12_resume_thursday]].
- Correctness to-do: [[project_future_matmul_bf16_racecheck]].

**DPO kernel finding (colleague's `DPOLossKernels(1).cu`, one file is enough):** our types in the API, raw uint4 memory, bf16->fp32
bit shift, fp32->bf16 CUDA hardware `__floats2bfloat162_rn` (pairs) / `__float2bfloat16_rn`. Her "bfloat16_t software
conversion is issue-bound" note predates TI b975be5 (09-29), which made our type use the hardware conversion on device
(Types.h:111-114). Our loss kernels = our types via uint4. CORRECTED from SASS: NO conversion difference - ptxas pairs
our per-element __float2bfloat16_rn into F2FP.BF16.F32.PACK_AB with 2 sources (bf16 bwd: 21 F2FP, 20 paired, 1 RZ = tail),
same as her __floats2bfloat162_rn; 2 per instruction = sm90 max. CudaNativeType in the loss = spelling only. Task 1 is a
DIFFERENT kernel: convert_type_kernel, 1 element/thread (the B300 vec8 one: 2x float4 in, paired cvt, 16 B out).

**Why:** the user wants these kept and done one at a time after the hand-over.
**How to apply:** when the user asks "what next for pretraining": PDL (unprofiled A/B) or attention research; casts need
only an optional nsys confirmation; GEMM hybrid only on the user's word. Ask before any GPU run.
Related: [[project_state_2026_10_21_lottn]].
