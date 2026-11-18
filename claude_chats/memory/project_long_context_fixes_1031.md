---
name: project-long-context-fixes-1031
description: "2026-10-31 long-context bug fixes (RoPE table/per-token, loss fallback, copy >2^32, fast math only for the loss file) in the LAPTOP TI tree, NOT pushed; proven 4k-128k + trainer; fp32 dK/dV measured and NOT adopted; emb/SwiGLU-sm120 candidates (old SwiGLU hidden%4 bug found) being decided; open: push + remote verify"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T13:41:01.682Z
---

State at 2026-10-31 ~17:40 IST (work doc 45.113-45.123+, plan Q&A 142-176, records
profiling_reports/long_context_kernels_2026_10_31/).

**In ~/Downloads/BluTrain_push/Tensor-Implementations (uncommitted, user pushes):**
- RoPEKernels.cu + arch/RoPE_sm90.cu: tables flat grid, per-token token in grid x, launch checks (throw).
- Makefile: global --use_fast_math removed; ONE per-file line gives LossKernels.o --use_fast_math (pf, user chose
  16:50; loss fwd without it +39%; pf = old speed; trainer FINAL2 vs OLD +0.4% = noise; step-0 val closer to fp64). 45.121.
- loss/LossKernels.cu: fallback backward = candidate l3 ((row,part) in grid x) + launch check. md5 5ade878a.
- Views/ContiguousKernel.cu + arch/ContiguousKernel_sm90.cu: copy fix c3 (split like PyTorch, 64-bit fallback kernels).
- attention/arch/GQA_bwd_vj2_sm90.cu: BLU_VJ2_DKDV_F32 switch (default off, bf16 path SASS-identical). MEASURED, NOT
  ADOPTED (user rule "only if speed same": attention bwd +3-14% at 4k-8k, trainer +0.8-1.1%; precision 32k dV
  1.169->0.571). f32wr (wait_group.read) = f32, dropped. Open: push the switch as opt-in or drop it (user). 45.122.
- Tests/kernel_tests/test_varlen_attention.cu (blocked fp64 ref, real rows).
BluTrain sft/ drop-only + T<=131072 limit (see [[project-case5-long-questions-1031]]).

**Embedding + SwiGLU-sm120 grid-y fix (45.123):** candidates e1/s1 swap-to-grid-x, e2/s2 grid-y loop, e3/s3 flat 1-D.
Embedding: all pass; e1 fastest bwd at today's sizes. SwiGLU sm120 fp32 has an OLD bug: hidden%4!=0 -> float4 row
strides wrong (all rows after 0 wrong); fix = float4 only if hidden%4==0 and 16B-aligned, else scalar typed<float>
(PyTorch can_vectorize_up_to / TE aligned-unaligned). Re-test es_var2 queued (also fixed 2 test bugs). Then apply the
chosen ones to the library + trainer check.

**run_sft.sh REMOVES every BLU_* env var of the caller before mpirun (by design).** To pass a library switch
(e.g. BLU_VJ2_DKDV_F32) into a trainer run, give BIN = a wrapper that exports it and execs the binary
(lc_box/tests/queue/sft_f32_wrap.sh). A switch "set" in the caller env silently does nothing; check its notice line.

**Proven (box 95.133.253.118):** every RoPE kernel + LeRoPE bwd + loss fallback + copy pass at 4k-131k; old fails exactly
where expected. Training RoPE = fused bf16 kernel: no bug, only the table (fixed). Logits path at 6.46e9 values passes.
G14 dense + packed 8k-128k pass (bf16 32k dV 1.169 = known; add to G14 known list = open). 135M trainer step-0 val = numpy
at 4k-131k. Memory = 8.4 GB + 0.757 MB x (B*T) tokens, confirmed again 17:25 (65,536 tok -> 58,008 MiB; 131,072 ->
107,604). Max B: 4k 42 (43 OOMs), 8k 21, 16k 10 (probes 17:24-17:27, rest running). OLD trainer at 65k aborts at start.

**GPU use 10-31 17:07-18:05: GPU 0 given for ONE hour** (normally never GPU 0). Two-worker queue
(lc_box/tests/queue/worker2.sh + jobs.sh + joblist2 file; mkdir claims; GPU 0 deadline). GPU 0 runs ~3% slower than
GPU 3 for the same job: compare pairs only on one GPU.

**18:10 update:** emb e1 + SwiGLU s2 APPLIED to the laptop TI (minimal edits; SASS = tested; FINAL3 = all 10 push
files passes trainer 4k + 131k gate 4.334521). Rivals: ours beats FA3/FA4/cuDNN fwd+bwd at 4k/16k/32k (65k/131k
running). PUSH LIST + draft commit messages + remote-verify commands:
profiling_reports/long_context_kernels_2026_10_31/PUSH_LIST_2026_10_31.md (exclude dist/Context_Parallelism pointer
and the untracked sft/fmt_* study files; GQA switch keep/drop = user).

**19:10 update:** fp32 dK/dV switch KEPT, OFF by default (user 18:20), experiment documented at the switch naming
gautam_reddy_k; checked: SASS default == original, API (scaled_dot_product_attention) off -> greduce<bf16>, trainer nsys
off -> gqa_bwd_vj2_kernel<(bool)1,(bool)0> (45.126). Rivals done at 4k-128k (45.125 attention, 45.128 loss + RoPE): loss
fastest everywhere; lm_head+loss: Liger FLCE ties us at 131k tokens with 0.37 GB vs our 12.3 GB logits; RoPE fastest +
most accurate to 32k, LOSES at 64k/128k: fused RoPE backward grid = fixed 32 chunks per row (FusedRMSRoPE_bf16.cu:804),
128k x 1 = 192 warps -> fix after push. CP analysis (45.127): Adhitya's CP builds its RoPE table on CPU (YARNOps.cpp
overrides build_rope_cache) -> the TI table bug did NOT hit his CP runs; his doc mask = ring+split only, ids from EOT,
positions not restarted; plan for SFT: no CP while a row fits one GPU, then Ulysses with OUR kernels.

**Open:** push all
(user), then VERIFY ON THE REMOTE change by change, every change in code comments and commit messages
([[feedback-verify-push-on-remote]]). Later: logits matmul redundant-compute removal; tell Adhitya's CP team about the
RoPE table bug.

**Box rule learned today:** ONE runner owns a GPU at a time; start the next only after the previous process is gone
(exact-argv count). See [[feedback-box-ops-traps]] trap 8.

**PUSHED 2026-11-01 ~00:50 IST (user ran the commands), VERIFIED ON REMOTE file by file:** TI dist_side_gautam_lottn
6aac3bf -> d067e89 (10 files: RoPE x2, LossKernels, Contiguous x2, Embedding, SwiGLU sm120, GQA_bwd_vj2 fp32 option OFF,
Makefile fast math loss-only, G14 test). BluTrain 334ecbb -> 91b6b54 (9 sft M + test_long_chat_split.py D + TI pointer;
vocab_rows required + meta check; T from meta.json; long chats dropped; README/docs). NOT pushed: trained-rows switch
(saved, [[project-future-logits-trained-only]]), dist/Context_Parallelism pointer, sft/fmt_* study.
