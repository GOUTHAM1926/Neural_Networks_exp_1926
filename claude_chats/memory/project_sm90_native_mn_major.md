---
name: project_sm90_native_mn_major
description: SM90 native MN-major TMA (replaces raw-transpose) VALIDATED on H200 for NN+TN single-CTA
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-25T10:10:53.532Z
---

SM90 (H200) native MN-major TMA path DONE & validated on H200 (2026-07-25): replaces the
raw-load+software-transpose workaround for MN-major operands (NN's B, TN's A/B). All 30 NN/TN
shapes in test_bgemm_sm90_nn_tn.cu PASS bit-exact vs cuBLAS; single-CTA ~739 TFLOPS square-4096,
qwen-shape132 710. NN≈TN everywhere = genuinely one shared implementation. MEASURED before/after
(§H15.8): native vs old raw-transpose = 13-19x faster on TN, 9-13x on NN (e.g. TN square-4096
739 vs 38 TFLOPS), ZERO regressions — matches the documented "13-19x slower than NT" bottleneck.
See analyzations_h200.md §H15–§H15.8.

Design (permanent, not a band-aid — removes the workaround, matches CUTLASS/cuBLAS semantics):
- `init_5d_tma_MN_sm90` — the MN-major operand loads DIRECTLY into MN-major-swizzled smem. Must be a
  **5D** TMA box `{mn_in 64, k_in 8, mn_atom BMN/64, k_atom BK/8, batch}` because the MN swizzle atom
  is 64(MN)x8(K) → BOTH axes split → 4 structural modes. The box compact-order then == CUTLASS's
  `tile_to_shape(Layout_MN_SW128_Atom)`. A 4D box (the first cut) lands "box-natural" order instead →
  make_gmma_desc<Major::MN> misreads → all-FAIL. K-major only needs 4D (atom 8x64, K unsplit, 3 modes)
  — the 4D box does NOT generalize to MN by axis-swap; the atom shape differs.
- Consumer: `make_gmma_desc_mn_major<MN_WIDTH>` (leading field=64, stride field=MN_WIDTH, layout B128 —
  §H11.3 probed) + wgmma `Tnsp*=1`. Per-atom A base = atom_idx*512*2 B; k16-slice stride = ks*16*BMN*2 B.
  These were correct from the start; only the producer box was wrong (5D fix, §H15.6).
- Gated behind `USE_NATIVE_MN=true` (SmemLayout); false = old raw path, kept ONLY as a bisection A/B
  aid until step 4 deletes it. NEEDS_RAW_A/B, transpose_to_kmajor_swizzled, raw barriers all go then.

Proven locally (nvcc 13.0 host, no GPU) before H200: descriptor bitfields + consumer offsets via cute
probes (scratchpad/probe_mn_offsets.cu, probe_tma_mn.cu dumping CUTLASS's real MN-major descriptor).

MN-major MULTICAST (§H15.9) DONE + H200-validated (2026-07-25): TN 1x2 A-mc, TN 2x1 B-mc, NN 2x1
B-mc all PASS (test_bgemm_sm90_cluster_mn_gate.cu, 9/9, redundant twins too). KEY: the 5D box is
sliced along **K (dim3=k_atom), NOT MN** — tile_to_shape interleaves mn_atoms inside k_atoms so only
the outermost (k_atom) mode carves contiguous slice_idx*slice_bytes chunks (K-major got away with
MN-slicing only because BK=64 -> 1 k_atom). issue_operand_5d_mn<MULTICAST>, launcher slices
A_BK_BOX/B_BK_BOX=BK/CLUSTER. CLUSTER COVERAGE NOW COMPLETE: all {NT,NN,TN}x{1CTA,A-mc,B-mc} green.

Step 4 DONE + H200-confirmed (§H15.10): raw-transpose path fully DELETED (transpose_to_kmajor_
swizzled, init_4d_tma_raw_KX_sm90, raw staging/barriers/phases, USE_NATIVE_MN flag, ~100 lines);
producer is one all-direct block; both gates re-run ALL PASS identical TFLOPS (behavior-preserving).
SM90 native-MN + universal-cluster milestone CLOSED. Next open work (not this milestone): partial-
cluster tiles (M/N not mult of CLUSTER*tile) to unlock the real 46 Qwen shapes on clusters; then
1x4/4x2/2x2 shapes, h/v raster, clock-locked sweep -> dispatcher.
Also: bf16 atom constants (64,8) → dtype-parameterize for future fp8/fp16 (clean extension).
Ties to [[project_sm90_cluster_1x2_done]] (partitioned multicast composes orthogonally per §H15.1).