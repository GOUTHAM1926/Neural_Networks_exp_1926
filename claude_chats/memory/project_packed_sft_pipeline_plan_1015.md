---
name: project_packed_sft_pipeline_plan_1015
description: "10-15: packed SFT pipeline built; EVENING: switched to design A (per-chat list) EVERYWHERE, design B (seg_start) REMOVED from v62/Vj1d/Vj2/API/loader/trainer; 2 GPU tests pending (G11 under Vj2 + same-chats e2e)"
metadata:
  type: project
---

User decision 2026-10-15: pause attention-kernel optimisation; first a complete SFT pipeline WITH packing (v2), then
optimise on top. Plan + verified status table: `sft_work_documentation.md` §45.3.

Verified status (laptop `~/Downloads/BluTrain_sft`): loss ready; template decided (53.0, [[project_chat_template_decision_1015]]);
RoPE positions ready in kernel+autograd (`fused_rms_rope(..., positions)`); attention fwd v62 A+B and bwd Vj1d A+B ready
in KERNELS only; packed + USE_VJ2=1 throws (GQA_bwd_generic.cu:728). MISSING: autograd attention op param (AttentionOps.h),
trainer wiring (bluscript_zero_ckpt.cpp:617 pos_offset=0, :642), dataloader per-row segments (SftDataLoader.h), FFD
packing in prepare_sft_data.py, end-to-end packed==unpacked tests. Design A = main path (§41.3); plan built per
micro-batch on host (gqa_chat_plan_build), owns device memory.

DPO mismatch: `/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/prepare_dpo_data_tokenize.py` makes <|user|>/<|assistant|>/<|system|>
special ids 50257/50258/50259 and encodes content with allowed_special, yet targets OUR plain-text SFT checkpoint
stage0_adamw_full. Mentions an sft_tokenizer.py not on the laptop; their compute_dpo_reference_logprobs.cpp is untracked
in the box's ~/gautam_k/BluTrain_sft. Decisions D1 (one tokenization for SFT+DPO) and D2 (end token 50256 vs dedicated EOT)
are open, with the DPO team. Weight tying ON and loss over 50,304 rows (spare rows were only pushed down as outputs).

**Future (10-15):** over-long chats: keep DROPPING for now; later do TRL `bfd_split`-style splitting (pieces <= T, each its
own segment) — open points in work doc §45.4 + packing doc §7.7. User also chose: Vj2 = DEFAULT backward (USE_VJ2=0 -> Vj1d),
SFT packed backward on Vj2 (design B first), pretraining-run comparison of Vj2 vs Vj1d LATER. User runs box commands himself.

**A1 CODE DONE on laptop (10-15 ~15:00), not yet built/tested on the box:** Vj2 packed rows design B (template PACKED,
seg_start last param, trip count via __syncthreads_count before the role split, +256-B seg_start bulk load per step,
mask key < seg_start[q], packed smem 199,232 B) + Vj2 = DEFAULT (enum Vj2_Default/Vj1d/Vj1_Flash/V44_DetDQ; USE_VJ2=0 -> Vj1d;
chats (design A) -> Vj1d in Vj2 mode; S%128 fallback now passes seg_start/chats). md5: vj2 ced7d3d8, generic c81c9909,
AttentionKernels.h 21d8fb3d, test_varlen_attention.cu a2ea4bd8 (G9 runs under Vj2; G11 SKIPS under Vj2 -> run with USE_VJ2=0),
sanitize_vj2.cu 2d9f8528. Local nvcc 13.0 sm_90a: 0 errors; both instantiations 168 regs 0 spills; DENSE SASS IDENTICAL to b1
(1760/1760). Backups: scratchpad a1_before/. Next: user copies 5 files to ~/gautam_k/v2_work, builds, snapshot a1, runs G9/CLASSIC/USE_VJ2=0 tests.

**PIPELINE WORKS (10-15 ~16:00), work doc §45.5.** E2E test (same 500 chats, v1 500 rows vs packed 147 rows, B=1, 1 step over all):
step-0 val 1.445038 vs 1.445084, grad norm 0.7935 vs 0.7935, packed 2.5-3.3x faster per step. Box tree ~/gautam_k/sft_v2/BluTrain_sft
(binary bluscript_sft_bin via build_sft_cmd.sh), runs ~/gautam_k/sft_v2/runs/, data ~/gautam_k/sft_data/stage0_ultrachat_5k_T4096_packed
(+ e2e_v1/e2e_packed symlink dirs). Exit hang fixed (g_packed reset after the loop; bluscript_sft.cpp 05dede59) — verify on the first real run.
NEXT: real packed sample runs (step count: 64 rows/step = ~211 chats/step -> 47 steps for 2 epochs vs v1 156) + D1/D2 token decision with DPO team.
packed64 DONE 10-15 16:19: 47 steps, val 1.4464 -> 1.3002 (v1 156 steps: 1.2839), ~6.3 min vs ~14 min, EXIT FIX CONFIRMED (exited by itself).
packed16 (189 steps, separates the step-count effect) stopped by the user at step 5 -> re-run later. Box idle, nothing of ours running (16:26).

**EVENING 10-15: DESIGN A EVERYWHERE, B REMOVED (work doc §45.7).** User was angry B was used though A was the 10-12
default. Done on laptop; COPIED + BUILT on the box 10-15 18:14 (md5 7667deec/83c88d41/204587bd, data 2e23ee29/fc96ebae,
box seg_start.bin removed; lib/trainer/test exit 0, binary tva_a in TI/); NOT yet GPU-tested: Vj2 got design A (template CHATS; plan block list
{b,t0,L,kt,hq,dqt0}; FA3-style PADDED 64-row slots for LSE/D/dQacc via bwd_prep_vj2_chats; convert_dq_frag_vj2<true>;
plain-store epilogue for a chat's last tile); B mode deleted from v62 + Vj1d + Vj2; API/autograd take
`cuda::GqaChatPlanPtr` (shared_ptr, gqa_chat_plan_make); plan = ONE alloc + ONE cudaMemcpyAsync on current stream.
Data: prepare_sft_data.py writes seq_lengths.bin (int32) + seq_offsets.bin (int64 [N+1]) = TRL seq_lengths; loader
refuses old seg_start.bin-only folders; trainer packed_rows_of() builds the plan per micro-batch (Hq = cfg.q_heads).
Proofs (no GPU): dense SASS byte-identical (v62 2280, Vj1d 1624, Vj2 1760 vs b1); A modes identical except param
offsets; Vj2-A 1688 instr 168 regs 0 spills; plan builder REAL code CPU test 254 layouts 0 fails (mutations caught);
regenerated data inputs/labels/valid byte-identical + boundaries == old seg_start; loader REAL code CPU test PASS;
test_varlen_attention G8/G9 deleted (-808 lines), G11 runs under Vj2. md5: fwd_sm90 eff8cdb6, vj1d 302ece10, vj2 7667deec,
fwd_generic c993599a, bwd_generic 2dbe59eb, AttentionKernels.h a95a7c63, test 612b7d53, prep f9dc7a42, loader 83c88d41,
bluscript_sft.cpp 204587bd. Laptop packed data updated (seg_start.bin moved out). Box data still OLD format -> loader
will refuse it until the new files are copied. PENDING GPU (ask first): (1) VARLEN_QUICK=1 VARLEN_ONLY=G11 under Vj2;
(2) same-chats e2e step 0 (v1 1.445038 / gn 0.7935).

**10-16: DESIGN A PROVEN ON GPU (2x H200 box root@86.38.238.80, work doc §45.11).** Test 1 G11 quick under Vj2:
94/94, 12 s; Vj2-A bwd 2.901 / 1.474 / 1.023 ms on real rows longest/middle/shortest (= 94.9 / 50.9 / 34.8% of dense
Vj2), ~26/22/12% faster than Vj1d-A of 10-12. Test 2 same-chats e2e packed A: step-0 val 1.445038 = v1 EXACTLY
(B gave 1.445084), grad norm 0.7935, exit 0. Next: push decision (user: push after optimizations), pull logs, §45.8 list.
