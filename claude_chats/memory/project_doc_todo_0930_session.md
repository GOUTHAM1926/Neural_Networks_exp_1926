---
name: project-doc-todo-0930-session
description: AFTER the F4 fix, document the whole 2026-09-30 session (every doubt, answer, prompt) in simple English; anchor = the "full chain per case" D1/D2/D3 explanation the user said made it clear
metadata:
  type: project
---
User 2026-09-30: "remind this for documentation ... we will document everything from morning, every doubt, every response and prompt in detail, in simple understandable english ... after this F4 bug fix".

**DONE 2026-09-30 (evening):** written as `embedding_optimizer_zero_1_parallelism.md` (819 lines, master `~/Downloads/sft_research/` + `work_docs*/`), loss doc §26/§31, SFT doc §36. Keep this file only as the record of the user's wording.

**Order (was):** F4 fix first, then this documentation (master `~/Downloads/sft_research/`, mirror `work_docs*/`, per [[feedback-single-doc-file]]).

**Topics to cover (09-30 session):**
- embedding forward/backward op sequence (Op1 lookup ... Op7 residual add; backward in reverse; ids are an INPUT of the lookup backward, not an output)
- residual stream + resid dtype vs embed dtype (2 separate switches; Op 2 is a no-op when they match)
- RMSNorm: fp32 maths inside the kernel, no separate conversion; gamma grad always fp32
- lookup backward kernels: dist_side_merge = cooperative fp32 / atomicAdd bf16; BLUCCL = EmbeddingBwdFused (CUB sort + bounded_direct/chunks/fold, fp32 adds, one final write), active only with bluscript_zero_ckpt_bluccl_fused.cpp + BLUCCL_FUSED_EMB=1; hook ignores N (SFT risk)
- why every wte row changes (tied lm_head part, weight decay 0.1, Adam momentum); Muon does NOT update wte (AdamW does)
- bf16 rounding (gap 1/128 near 1.0, half-gap rule; fp32-then-bf16 == bf16 add, 0/1M pairs differ)
- NVLS switch: .acc::f32 but returns bf16
- F1 explained (fixed in all 3); #2 fixed (v1 only had it); F4 verified in all 3 + fix design; Muon BLU_MUON_RB leftover

**Anchor text (user: "actually I understood from this only clearly"):**
fp32 table (BLU_BF16_EMBED=0): D1 Op-2 backward widens grad to fp32 -> table grad fp32 -> fused kernel adds fp32, writes fp32: no rounding. D2 lm_head part widened to fp32, fp32+fp32: no rounding. Micro-step buffer fp32+=fp32: no rounding. Send fp32, switch adds fp32, returns fp32: no rounding. TOTAL 0 roundings to bf16 (lm_head part is born bf16 from its matmul in both cases).
bf16 table (BLU_ALL_BF16=1): D1 adds fp32, writes bf16: 1 rounding/micro-step. D2 bf16+bf16 stored bf16: 1 rounding/micro-step. Micro-step buffer fp32: none (unless BLU_GRAD_VIEW=1). Send fp32->bf16: 1/step. Switch fp32 add returned bf16: 1/step. 1 GPU: same minus the switch; the send conversion still happens = F4.

**Also document (user asked 2026-09-30): the send-dtype flags and their consequences.**
- Default SAME_AS_PARAM: send dtype = weight dtype. `BLU_BF16_REDUCE=1`: every group sends bf16 (fp32 groups get converted); `BLU_FP32_REDUCE=1`: every group sends fp32. Mutually exclusive. v1 refuses both when they differ from the weight dtype (throws); v2 + Muon accept. Trainer comment: BF16_REDUCE is "for bisection, not for production", +147.6 MiB/GPU, ~0.3 ms/step, "costs precision exactly where the model chose not to spend it" (wte). Colleague experiments with it; their committed run scripts set neither.
- Chain, fp32 table + BLU_BF16_REDUCE=1 (8 GPU): D1/D2/buffer fp32 no rounding; send fp32->bf16 1/step; switch returns bf16 1/step => 2 roundings/step. bf16 table default: 2/micro-step (D1, D2) + 2/step.
- User's point: accumulating in fp32 then sending bf16 throws part of the savings away ("maybe not all but will lose"), so with that flag a bf16 embedding table is closer to the fp32 one. Fact: the fp32 table still avoids the per-micro-step D1+D2 roundings; only the per-step part is lost. The test decides whether the rest matters.
