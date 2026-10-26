---
name: project_lower_gpu_attention_1024
description: 10-24 — generic (non-Hopper/non-B300) causal + packed-rows GQA attention BUILT in BluTrain_sft; tests on the local RTX 3060; the user arranges an Ada box next
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-24T15:58:19.398Z
---

**Audit (10-24, BluTrain_sft TI, identical to BluTrain_lottn before the edit):**
- The "generic" kernels = `GQA_fwd_generic.cu` (`gqa_v2`) + `GQA_bwd_generic.cu` (`gqa_backward_dQ` / `gqa_backward_dKdV`): sanjay 06-24, madhu 07-22. WMMA, 1 warp, no TMA, NO causal mask.
- Off Hopper, before the change: chats → throw, causal hd128 → throw, causal hd64 → full attention (a silent wrong answer).

**Built 10-24 (not optimized yet):**
- Kernels: `CAUSAL` template flag on all 3 kernels. Segments come from `GqaChatPlan::gen_segs` {b, t0, L, 0}, plus `n_gen_segs` and `gen_max_len` (appended last in the same upload). Tiles are counted from t0.
- Launches: `getCurrentStream()`.
- Dispatch: `generic_ok = is_causal && !v62_ok/!hopper_ok && arch != B300` in both qkv entries; the packed entries are routed the same way.
- B300 is untouched. Hopper source files are byte-identical to lottn and do not include `AttentionKernels.h`.
- Work dir: `profiling_reports/generic_attn_sft_2026_10_24/` (`orig/`, `new/`, `diffs/`, `logs/`, `h200_sass_proof/`).

**Test harness `test_varlen_attention.cu`:**
- `g_generic` and `VARLEN_HD` (64/128).
- Refusals for hd64 and T%256 were removed (they are accepted now).
- The reference is computed one kv-head group at a time (fits 12 GB).

**Why:** the user wants SFT packed rows on Ada (sm_89) and the RTX 3060 (sm_86), and later optimization.

**Results 10-24 (RTX 3060 quick):** hd128 292/292 + 1 KNOWN; hd64 291 pass / 1 FAIL (dQ `row 0 @2688 L=64` 1.175x = dS from the bf16 P; the fix is fp32 P for dS, NOT yet done). Packed == alone bit-exact, repeats bit-identical. The generic kernels have NO cp.async (0 LDGSTS); the FA2-style rewrite is next. SFT package: `Neural_Networks_exp_1926/SFT_package_2026_10_24.zip` (README §0-10); work doc §45.92, §45.94. Box 86.38.238.224 copied, verified (3,909 files, same bytes) → `~/Downloads/box_backup_2026_10_24_1xh200`.

**How to apply:**
- After the RTX 3060 tests: an end-to-end trainer smoke, then a report + zip of every change.
- Then optimize: FA2 style, 4 warps, cp.async, fix the dKdV hd128 spills (320 B), compute D once.
- Run the H200 regression (the varlen test G10/G11) when a box is available.
