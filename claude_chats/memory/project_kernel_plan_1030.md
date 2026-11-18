---
name: project_kernel_plan_1030
description: "Kernel plan 10-30 (work doc §45.108): user's 6 tasks T1 long ctx 16k-64k, T2 trained-only logits+fused lm_head, T3 bfd_split, T4 user self-study, T5 attention masked work, T6 runtime metadata; 22-item kernel future list; loss >65,536-token bug blocks T1"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-30T09:17:43.657Z
---

User 2026-10-30: chat-format study POSTPONED (CPU work done). Focus = kernel side; modelling is others' job, we step in
only if they fail. Full text: `sft_work_documentation.md` §45.108 (plan, answers, 22-item list, order).

Key facts found (code-checked, BluTrain_push 5c0999c / TI f7aa40f / CP pin 661b6ef 2026-08-30):
- SFT trainer has NO context parallelism; CP repo has its own attention (ring GQA_causal_cp_*_sm90 from old v62/v44,
  Ulysses gqa_fused_flash_attn_forward) with NO chat boundaries / per-chat positions -> packed SFT cannot use CP today.
  Ulysses could reuse our design-A kernels (KV heads must divide by CP degree); ring needs new chat-aware masking.
- Single GPU: no length limit (fwd T%256, Vj2 S%128); tested 8192 rows + one 16,384 chat only.
- BLOCKER: LossKernels.cu stage-2 sum = one 256-thread block -> >65,536 tokens per micro-batch drops loss (logged/val
  only). Long-ctx SFT (8x16k, 4x32k, 2x64k = 131,072) would log half the loss.
- "Only place we lose": speed - only dense attn fwd alone 0.5-1.4% vs FA3; MEMORY - 50.0 vs TRL+Liger 46.0 GB (logits).
- Attention: design A skips other-chat and above-diagonal tiles; waste 11.54% (diag 7.48, chat-end 3.83, pad 0.23).
- Metadata: chat lengths + valid counts are in preprocessing files; positions + attention plan built on CPU per
  micro-batch, 2 async uploads; cost never measured.
- Drops at T 4096: m1 4,091 (0.9%), m2 19,162 (2.5%).

Proposed order: loss>65,536 fix + BluBLAS M/N%64 guard -> T1 -> T2 -> T6/T5 measure -> T3 -> matmul fusion (separate
tree). T4 = user reads alone (reading map in §45.108). Waiting for the user's pick.
**How to apply:** no GPU runs without asking; branch rule [[feedback_branch_rule_sft_vs_experimental]].
Related: [[project_future_logits_trained_only]], [[project_matmul_epilogue_audit_1024]], [[project_sft_loss_preexisting_defects]].

**Round 2 (10-30, §45.108 part 5):**
- EMBEDDING: SFT trainer embedding code == pretraining (identical). Kernel chosen by YAML `embedding.backward`
  (prod YAML = segmented = Kruthik's). run_sft.sh passes NO YAML + BLU_ALL_BF16=1 -> every SFT run so far (incl. 135M
  grid, 10-27 re-runs) used bf16 atomicAdd (non-deterministic, ~15-93% err frequent tokens). Fix = BLU_EMB_BWD=segmented
  in run_sft.sh + check; also audit other prod-YAML settings SFT misses. NOT done yet.
- CP: memory = tokens per GPU (B x T), not T. 1.6B 16k/32k at B2/B1 = 32,768 tokens = fits (88 GB measured at 32,768);
  64k est. no. CP = splits one row over GPUs, not a kernel swap. His CP: no doc masking, own trainer bluscriptCP.cpp,
  not referenced by our trainers/Makefiles; separate repo as submodule dist/Context_Parallelism (dist/ = plain folder);
  branch records cf1372d, laptop checkout 661b6ef stale (fetch first). Plan: measure max row per GPU -> Ulysses with OUR
  kernels + chat plan (KV heads must divide CP) -> ring only if needed (new chat-aware kernels).


**Round 4 (10-30, §45.108 part 7):**
- Adhitya reply (cp_verification_reply_2026-10-30.md): bug present in his lib; all his train/val <= 32,768 rows/call
  (64k/128k use CP 4-8); only hit = LongRoPE search CP1 T131072 pre-08-10 (PPL 7.55 vs 59.87). His LC runs used
  CP_BF16_EMBED=1 -> bf16 atomicAdd embedding bwd. Doc mask ring-only, no per-doc position restart. MISMATCH: reply
  says TI 1c84859 but quotes ad10e7d code (not in 1c84859) -> his real version unsettled; his libtensor.a is 08-18.
- Fix reaches a team only after it is in the TI branch they build (Adhitya: TI master) AND they rebuild.
- Generic attention package for junior colleague: work_docs/generic_attention_varlen_package_2026_10_30(+.zip), files
  == TI f7aa40f == 10-24 tested.
- Context length: not in weights/optimizer; RoPE settings (theta, YaRN/LongRoPE+mscale) must match the checkpoint; our
  SFT trainer builds plain RoPE only (bluscript_sft.cpp:831) -> needed before SFT on Adhitya's long checkpoints.

**Round 5 (10-30, §45.108 parts 8-9; plan file X5 + Q&A 127-132):** user rule: support every SFT length choice
correctly first (loss fix, bfd_split, RoPE match, CP), measure chat lengths before SFT, pack every chat whole. Loss
research: PyTorch nll_loss = one block LOOPING over rows; torch.sum/CUB loop + multi-block; Liger/FA/Quack/TE/Megatron
= per-row losses + torch.sum. Our Muon/grad-norm finalize kernels already loop. Fix = strided loop in
sum_reduction_kernel (bit-identical <= 65,536). Test plan 8 steps; the USER writes the fix and runs tests; I review.
NEXT: user ports test_sparse_ce_ignore_index.cu to lottn, runs before/after.
