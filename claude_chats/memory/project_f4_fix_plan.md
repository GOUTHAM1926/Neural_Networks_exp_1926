---
name: project-f4-fix-plan
description: F4 (1-GPU bf16 grad round trip in v1/v2/Muon) fix — done ON THE H200 BOX in a fresh clone of dist_side_merge_BLUCCL, pushed from there; must be robust, future-proof, not fight colleagues' flags; deep analysis on the box's exact code first
metadata:
  type: project
---
User 2026-09-30: don't touch dist_side_merge; clone remote `dist_side_merge_BLUCCL` on the 1x H200 box (`round-earth-echoes-fin-03`, 86.38.182.174, `~/gautam_k`), analyze, apply, test all cases, push from there; locally only fetch later. "The fix should be more robust, in sync with all future changes, correct, and should not oppose or act as something negative. Deeply analyze first."

**Design (proposed, not yet approved):** a W==1 shortcut per optimizer. v1/v2 copy each param's fp32 grad straight into `shard_grad_fp32_` (zero no-grad slices + padding), skip the W=1 RS copy + cast. Muon: `owned_grad_fp32` reads the param's fp32 grad at W==1, skip staging + copy. W>1 code untouched. Recommendation so it doesn't fight colleague flags: shortcut only when reduce dtype is the default SAME_AS_PARAM (explicit BLU_BF16_REDUCE/FP32_REDUCE keep today's behaviour). Optional later: don't allocate `shard_grad_` at W=1 (~2.9 GB Muon group).

**Must stay consistent with (check each in the box's code):** F1 check + norm, loss scaler, global clip (reduce_and_measure), BLU_FUSED_ADAMW/fused clip kernels, grad bucket views (decline at W=1), overlap (off at W=1), param refresh/publish, DCP state, Muon pipeline publish + RB mode, grad_prescale_, padding.

**Tests:** W=1 bit-exact grad test per optimizer (must FAIL on old code first), F1 36 cases, 1-GPU trainer A/B old vs new; W>1 only by code review unless 2 GPUs. Then documentation of the 09-30 session ([[project-doc-todo-0930-session]]).

**STATUS 2026-09-30 ~16:27 box time: F4 patch APPLIED + VERIFIED on the box clone (not committed/pushed yet).**
- Patch: `profiling_reports/f4_2026_09_30/patch_f4.py` (15 anchors, md5-guarded vs a3aba3e), diff `f4.diff`; box files md5 == reviewed.
- Old (a3aba3e + TI b975be5): F4 test 17 FAIL / 13 PASS (bf16 norm = norm of bf16(g), rel 9.76e-4). New: 30/30 PASS (rel 1.3e-8). F1 36/36 both. Navin's test_zero PASS both.
- **Navin's a3aba3e introduced an INTERMITTENT ShardedMuon failure at W=1** (test_muon_rsag, fp32 params): always 7077/82120 elems, which scenario fails varies; e6eaeba dist/zero 6/6 PASS, a3aba3e 6/6 has a FAIL; identical before/after F4 (so not in the gradient-read path). Unverified lead: his norm rewrite removed the per-tensor host sync that protected multi_tensor_*'s static pinned descriptor buffer. His build scripts (build_and_test.sh, run_muon_rsag_test.sh) no longer compile (missing -Idist/Distributed_Checkpointing/include).
- a3aba3e also moved the TI pointer back to 38d0332 (drops our b975be5 port), added 2 symlinks, rope_theta 10000 in bluscript_zero_ckpt.cpp (SFT risk: ask Navin which rope_theta his step-8000 used).
- Next: optional 1-GPU trainer A/B (fused trainer, flux, `run_trainer_ab.sh`, binaries `_runner_fused_old/_new` built); then user commits on the box: commit 1 = 6 dist/zero files (`commit1_msg.txt`), commit 2 = TI pointer -> b975be5 (`commit2_msg.txt`), push dist_side_merge_BLUCCL.

**PUSHED 2026-09-30 (user ran it on the box): `a3aba3e..3298e74` on BluTrain `dist_side_merge_BLUCCL`.**
- `149249d` F4 (6 dist/zero files, +160/-5), `3298e74` TI pointer -> b975be5. Author Goutham Kumar (company), 0 co-author lines; pushed file md5s == tested.
- Still open: tell Navin (TI pointer, flaky test_muon_rsag since a3aba3e, stale build scripts, 2 symlinks, rope_theta question); optional 1-GPU trainer A/B (ready: `run_trainer_ab.sh`, `_runner_fused_old/_new` on box); local fetch of BLUCCL; dist_side_merge NOT patched (user: maybe later, separately); then the 09-30 documentation ([[project-doc-todo-0930-session]]).
