---
name: project_future_logits_trained_only
description: "FUTURE (after v1 SFT, user 10-17) = Q11 logits only at trained positions — gather -> lm_head [M x 2304] -> loss -> scatter dX; CCE proves the gain (-13% even with slow kernels); est. 36 -> 9.5 ms per micro-batch"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-28T11:49:50.151Z
---

User decision 2026-10-17: NOT in the v1 SFT pipeline; do it next, after v1. It is Q11 of the SFT plan (work doc §33, §38;
re-confirmed with data in §45.33 / §45.34).
- Today the lm_head computes logits for every position; only the loss skips masked rows. ~77% of lm_head work wasted at the
  median stage-0 batch (prompt + pad); packing removes pads but not prompts.
- Plan: gather trained rows of hidden -> lm_head GEMM [M x 2304] -> our loss -> scatter dX (zeros elsewhere); round M up to a
  multiple of 256 (filler label 65535) because of [[project_blublas_sub64_transposed_gemm]].
- Evidence: Apple CCE skips masked tokens inside its lm_head and beats our cuBLAS GEMM + loss by 13% at the median batch with
  kernels at ~half cuBLAS speed. Estimate with our BLAS: ~36 -> ~9.5 ms per micro-batch (~0.2 s per 8-micro-batch step).
- Not bit-identical to today (summation order); measure in all 3 stages with and without (6 cases).

**CORRECTED 2026-10-28 (135M, packed data):** the 36 -> 9.5 ms estimate was for PADDED 1.6B rows (77% waste). With packing
the waste is only the prompt share: m1 21%, m2 33%, d8 27%. lm_head+loss = 16.8 ms of a 202 ms step (8.3%) -> trained-only
saves ~1.5-2.4% speed and 1.1-1.7 GB (logits 5.25 GB at 13x4096x49280 bf16). Bigger memory lever = fused/chunked
lm_head+loss (TRL+Liger used ~4 GB less than us on 10-28). Same math; last-bit differences only.

**DECIDED 2026-10-28 (user): do BOTH (trained-only logits + fused/chunked lm_head+loss) LATER, after the SFT work.**

**Why:** biggest SFT compute lever found in the loss study (padded data); small once packed.
**How to apply:** start only after v1 is wrapped up; fix/guard the BluBLAS M/N%64 transposed case first.
Related: [[project_loss_bench_vs_others_plan]], [[project_sft_plan_is_qa_questions]].

**RE-RESEARCHED 2026-10-31 (work doc 45.129, all from current code):** TRL's DEFAULT now drops untrained rows before the
lm_head (`add_fused_lm_head`, trainer/utils.py:1569-1720, cuBLAS tiles + Triton fold + recompute, 8 x N_t work); Unsloth
the same (unsloth_zoo chunked_logprob.py:337-376); Apple CCE drops them inside its kernel; Liger FLCE, HF, torchtune,
Megatron, torchtitan compute every row. Our 10-31 lm_head+loss benchmark LACKED TRL / Unsloth and added cuBLAS + our loss
instead of measuring our trainer: fix both before building. N_t per micro-batch (real data): 4k median 47,730 / 65,536;
128k median 64,114 / 131,072 (min 77). Plan: trained-only rows (6 x N_t) + chunked gradient-in-forward (Liger style,
sft_denom known up front) = less work than every library read.

**10-31 21:45 RESULTS (work doc 45.130):** corrected lm_head+loss bench done (lc_box/tests/lmh, records in
profiling_reports/long_context_kernels_2026_10_31/long_t/lmh). We beat all-row libs everywhere; TRL/Unsloth/CCE win at
<=57% trained (16k-128k), we win at >=69%. NEXT: build trained-only as an optional switch and measure in the same harness.
Loader vocab fix (vocab_rows required + meta.json check) done, compiles; test run waits for a GPU; in PUSH_LIST.

**10-31 22:20 BUILT + MEASURED (45.131):** switch BLU_SFT_TRAINED_ROWS (off default), FINAL5 tree on box
/dev/shm/gk_final5, laptop files in BluTrain_push (5 new TI files + SftDataLoader.h + bluscript_sft.cpp + run_sft.sh),
NOT in PUSH_LIST yet. lm_head+loss: beats every library in every case, precision identical. Trainer: loss = noise level,
but step time only -3..-7 ms and peak -1..-3 GB (less than predicted). NEXT: nsys + memory trace of the trainer off vs on.

**11-01 00:30 DROPPED FROM THE PUSH (user) -> NEXT FUTURE TASK.** Code saved in profiling_reports/long_context_kernels_2026_10_31/
trained_rows_switch_SAVED/ (5 TI files + switch versions of bluscript_sft.cpp / SftDataLoader.h / run_sft.sh + diff + local tests).
Blockers: (1) BluBLAS sm_90 tile-map stall ([[project-future-blublas-tile-map-sync]]); (2) BluBLAS sm_86 GEMM wrong rows in
the last partial 2,048-row block when M % 2048 != 0 (found locally on RTX 3060; gather + loss exact). Add back only after
both fixed or a <sm_90 guard; then re-run lmh + trainer off/on/off + local sm_86 test. Work doc 45.131 end, plan Q&A 188.
