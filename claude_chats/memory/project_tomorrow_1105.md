---
name: project_tomorrow_1105
description: "REMIND USER 11-05: plan in order = fix 4 old BluBLAS GEMM bugs -> push batched NS + addmm fix -> pretraining before/after -> remaining CUDA-call items -> CP wrappers"
metadata:
  type: project
---

User, 2026-11-04 ~22:00 IST: "we will solve tmrw, ok dont forget, note and document as immediate future tasks, remind me
tmrw". Work doc 45.153 has the same list.

Order for 2026-11-05:
1. Fix the 4 OLD BluBLAS GEMM bugs (work doc 45.151; each reproduced with the 11-03 library too, so not from 11-04 work):
   - bf16/fp16 64x64x64 with batch >= 2: wrong numbers (batch 1 and 512^3 fine)
   - odd shapes 37x45x29: fp16/fp32 crash "misaligned address" at any batch; bf16 wrong at batch >= 2 (also with padded strides)
   - fp32 with mat1 transposed (A.T @ W): wrong numbers (blublasSgemmStridedBatched opA=T)
   - bf16/fp16 with both operands transposed: crash "misaligned address"
   Tools: profiling_reports/muon_batched_ns_2026_11_04/test_gemm_odd_stride.cu (args: type M N K batch sharedB) and
   test_addmm_fix.cu (TF_TYPES, TF_SKIP); rerun with NOTHING set aside after the fix.
2. Push (user does it) BluBLAS + TI + BluTrain: batched Newton-Schulz + per-batch bias + addmm fix (+ GEMM fixes);
   patches blublas_all_v2 dd94f023, ti_batched_ns_v2 2760368b, ti_addmm_fix c3922b3f, blutrain_batched_ns_v2 c52c624e;
   verify every file on the remote after.
3. Pretraining before/after of the 3 changes (B0-B3 as work doc 45.152) -- P032 has NO pretraining mix; need a box/path
   with commoncrawl/code/math(/edufineweb) or the user's choice of a fake mix from SFT tokens.
4. Remaining CUDA-call items of 45.140 (3 copy loops, 4, 5-7, 8, 9, 11, 12).
5. Goal 2: context-parallel wrappers (Adhitya's) around our kernels.
Open decisions: NS alpha/beta rounded to bf16 at the BluBLAS API (a 3.4375, b -4.78125, c 2.03125); TI cuBLAS-build addmm
branch has the same dropped bias shapes; group message to colleagues after all pushes; SFT runs used no YAML (rule 10-30).
**Why:** user planned a full day; wants it done correctly and completely.
**How to apply:** at the start of 11-05, remind the user of this list before anything else; ask before every GPU run.
Related: [[project_batched_ns_1104]], [[project_muon_gpu_guard_1104]], [[project_empty_cache_test_1104]].

(Moved from the MEMORY.md index line on 2026-11-05:) 11-04 result: SFT step 137.4 -> 110.3 ms (-19.6%).
