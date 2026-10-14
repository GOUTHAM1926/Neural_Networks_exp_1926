# Phase 1, test 1.4 - where the Vj2 main kernel loses to FA3 (2026-10-14 11:21-11:23, GPU 7)

Build: only GQA_bwd_vj2_sm90.cu rebuilt with -lineinfo (snapshot g123li); its kernel SASS is identical to g123
(1760/1760 lines); the normal library was restored afterwards (kernel SASS identical, no line tables).
ncu --set full, dense B8 18/6, 11th launch: 2526.3 us, tensor busy 67.0%. FA3 = yesterday's capture (prof_1013, 2481.4 us).
Tools: `tools/phase_map.py` (per-phase by C++ line), `tools/category_us.py` (per instruction category, samples x time).

1. Per-line attribution is muddled by the compiler mixing lines (descriptor work of the loop maps to setup line 483, the
   dV/dK GEMM block to helper line 240), so the comparison with FA3 is done per instruction category
   (`4_category_us_ours_vs_fa3.txt`): our warps spend +238 us-equivalent more on GEMM-issue instructions (HGMMA).
2. Most of that is a stall reason the per-line CSV does not export: "warpgroup arrive" (the 4 warps of a warpgroup must
   all arrive before a wgmma group starts). Unexplained samples = exactly the raw warpgroup_arrive counter: ours 16.7% of
   samples, FA3 10.4% (`4_unexplained_gmma_stalls.txt`):

| where the 4 warps wait for each other | ours (us-equiv) | FA3 | ours - FA3 |
|---|---|---|---|
| first HGMMA of the dK GEMM (register A) | 215.5 | 141.1 | +74.4 |
| first HGMMA of the dV GEMM (register A) | 96.8 | 39.5 | +57.3 |
| first HGMMA of the dP GEMM | 96.8 | 76.8 | +20.0 |
| first HGMMA of the S GEMM | 13.4 | 1.1 | +12.3 |

3. Also: shared-memory-queue stalls when issuing the shared-memory GEMMs (S, dP, dQ): ours 121 vs FA3 49 us-equiv;
   exp (MUFU) +47; GEMM result waits (DEPBAR) +46. FA3's extra time is in its idle loader warps (branch, mbarrier).
4. FA3's step order around these GEMMs (mainloop_bwd_sm90_tma_gmma_ws.hpp:937-962): dS to smem -> dV GEMM (no wait) ->
   fence + both-group barrier -> dQ GEMM then wait until <= 1 group pending (dV done) -> release dO buffer -> dK GEMM then
   wait <= 1 (dQ done) -> dQ to smem + hand-off -> wait 0 -> release Q. Ours: dS to smem -> dV GEMM -> fence + barrier ->
   dQ GEMM -> dK GEMM (no wait in between) -> wait <= 1 -> dQ hand-off -> wait 0 -> release Q and dO together.
