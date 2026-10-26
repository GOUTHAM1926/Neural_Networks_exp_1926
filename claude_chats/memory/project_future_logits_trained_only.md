---
name: project_future_logits_trained_only
description: FUTURE (after v1 SFT, user 10-17) = Q11 logits only at trained positions — gather -> lm_head [M x 2304] -> loss -> scatter dX; CCE proves the gain (-13% even with slow kernels); est. 36 -> 9.5 ms per micro-batch
metadata:
  type: project
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

**Why:** biggest SFT compute lever found in the loss study.
**How to apply:** start only after v1 is wrapped up; fix/guard the BluBLAS M/N%64 transposed case first.
Related: [[project_loss_bench_vs_others_plan]], [[project_sft_plan_is_qa_questions]].
