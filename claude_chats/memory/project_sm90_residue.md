---
name: project_sm90_residue
description: "SM90 residue/partial-tile support — real-dim descriptors, validated for all real shapes on H200"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-25T11:04:41.126Z
---

SM90 GEMM residue/partial support DONE for real workloads, H200-validated (2026-07-25). See
analyzations_h200.md §H16-H16.4.

Approach (permanent, hardware-based, NOT per-shape logic): declare REAL tensor dims in the TMA
descriptor so out-of-bounds tile elements clamp + OOB-ZERO by hardware (CuTe OOBFill::ZERO ==
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, which all our descriptors already use). k_iters=ceil(K/BK) so the
last partial K-step's OOB-K auto-zeros (correct since OOB-K contributes 0 to the wgmma). Epilogue
store clamps via the C descriptor's real (M,N); beta-read guards BOTH axes (row<M && col+1<N).

Two descriptor families:
- K-major (NT's A/B, NN's A): REWRITTEN to CUTLASS's real-dim rank-2 form — globalDim {K, M/N, batch},
  box {BK, BM/BN, 1, 1} = whole tile ONE UTMALDG + residue-robust (simpler than the old 4D swizzle_w
  split). Producer coords -> (kp*BK, X, batch, 0). Handles arbitrary M, N%8, K%8 (NT gate 14/14 §H16.3).
- MN-major (TN's A/B, NN's B): the fast 5D one-shot descriptor already uses REAL atom counts {MN/64,
  K/8}, so MN%64 & K%8 TILE-residue clamps too (TN/NN gate 12/12 §H16.4, incl real partial-M 151936).

Coverage (validated): NT arbitrary M, N%8, K%8; TN/NN M%64,N%64,K%8; partial clusters (phantom CTAs)
"just work". => ALL 46 Qwen shapes + real workloads covered, no perf regression.

HARDWARE limits (not ours; cuBLAS/CUTLASS share them): TMA leading stride multiple of 16B => row
strides K (A/B) and N (C store) must be %8; M fully arbitrary. Truly-arbitrary N/K needs caller lda/ldc
padding (globalDim stays real for clamp; only allocated pitch pads).

OPEN (optional, rare): MN%8-not-%64 sub-atom residue on TN/NN needs b1 (real-dim rank-2, box {64,8}=one
atom, (BMN/64)*(BK/8) UTMALDG/tile — CUTLASS's choice, but ~16x issues). Add ONLY as a launch-time
FALLBACK preserving the fast 5D path (dispatch on MN%64), never a uniform regression. Not needed for
real shapes. Swizzle is now a single-source knob (sw_tma_enum, template<int SwizzleBytes=128>) — see
[[feedback_sm90_composable_knobs]]; finish wiring it through wgmma layout_type + consumer next.
Ties [[project_sm90_native_mn_major]].
