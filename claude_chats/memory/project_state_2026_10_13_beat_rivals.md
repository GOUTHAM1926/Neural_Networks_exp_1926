---
name: project_state_2026_10_13_beat_rivals
description: "10-13 evening: goal = beat FA3/FA4/cuDNN/FA2 fwd, bwd, fwd+bwd, dense AND packed; profiling round tests 1-2 DONE on GPU 2, tests 3-5 (ncu) NOT run; stopped by user ~18:53"
metadata:
  type: project
---

Goal set by the user 2026-10-13: our attention (forward, backward, fwd+bwd, with the SFT packed modes) must beat
FA3, FA4, cuDNN, FA2 on H200. Method: profile everything, compare line by line with the FA source clone
(`~/Downloads/sft_research/flash-attention`, commit 94e22c9 = the benchmarked one), fix one at a time.

**Profiling round (GPU 2), records `profiling_reports/v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/`:**
- Test 1 DONE 18:48, dense same-session (`1_dense_scoreboard.txt`), ours = v62 fwd + final Vj2 f5:
  fwd ours/FA3 mean 1.08 (B8 18/6 1.21), /FA4 0.96, /cuDNN 0.93; bwd /FA3 1.01 (B8 18/6 1.04), /cuDNN 0.85 (13/13);
  fwd+bwd /FA3 1.03 (B8 18/6 1.08), /cuDNN 0.87 (13/13).
- Test 2 DONE 18:51, packed segments_1012 (`2_packed_scoreboard.txt`), ours bwd = Vj1d:
  fwd best-ours/FA3 mean 0.99, /cuDNN 0.88 (7/7); bwd /FA3 1.18, /cuDNN 1.08; fwd+bwd /FA3 1.13.
- NOT RUN (stopped by the user ~18:53, "continue later, I will say"): test 3 ncu forward per-instruction (ours v62,
  FA3, FA4, cuDNN; libs with BENCH_WARM=0 BENCH_ITERS=1, --kernel-name-base demangled, regex
  FlashAttnFwdSm90|FlashAttentionForwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_fprop), test 4 ncu backward
  (FA3/FA4/cuDNN), test 5 ncu packed real_rows_752 forward ours A vs FA3.
- Then: gap analysis (code + profiles) -> forward fixes first, backward LSE wait, Phase 2 (packed Vj2), final scoreboard.

FA4 facts (proven from source 10-13): on H200 FA4 runs its own sm90 CuTeDSL kernels (interface.py:1629, :2883), no
tcgen05/TMEM/2-CTA/exp2-emulation there; FA4 fwd is 12-17% slower than FA3 on H200, bwd about equal. Exp unit (XU)
busy <= 43% on H200, so FA4's software-exp2 trick is not expected to help on Hopper.
Related: [[project_state_2026_10_13_vj2_tests]], [[project_eod_1013_stats_reminder]], [[feedback_gpu_pin_device_1]].

**Profiling round COMPLETE 19:38 (tests 3-5 on GPU 2).** Gap list G1-G7 in chat + `prof_1013/` (3_fwd_*, 4_bwd_*, 5_packed_*).
Key proofs: fwd v62 1079 us vs FA3 877 (tensor busy 64% vs 83%, SM busy 95.7 vs 99.6%); half-width RS GEMMs in fwd P.V and
bwd dV/dK (m64n64 x2 vs FA3/FA4/cuDNN m64n128); FA3 bwd issues LSE LDS after its 1st S HGMMA. ncu DRAM unit trap again
(libs Gbyte vs ours Mbyte); packed-fwd time unit was "us" not "ms".
**G2 (uniform warpgroup index) = NEUTRAL, KEPT at the user's request (20:15)** (g2_1013/RESULT.md): bwd R2UR 59M -> 0.7M but time unchanged, fwd -1%
noise. Proves both kernels are NOT instruction-issue bound -> deprioritise pure instruction cuts (G4 dQ-scale FMUL).
Next order (proposed 20:00): G1 full-width RS GEMMs (fwd P.V + bwd dV/dK), G3 LSE before S GEMM, G5 fwd tail (v62 units
= mirrored pairs, 2304 units / 132 CTAs = 17.45 rounds -> wave quantisation), G6 fwd O via TMA store, then G7 packed Vj2.

**G-series results 10-13 (GPU 2), backward kernel ncu B8 18/6 (FA3 2481 us):** f5 2545 -> G2 2547 -> G1 2532 -> G3 2512.5 us.
- G2 neutral, KEPT (user). G1 full-width dV/dK m64n128k16 neutral (0.997), KEPT. G3 LSE before S GEMM KEPT: 0.994 (9/13),
  short_scoreboard 11.3 -> 3.1%. All: 144/144 identical errors. Records `gpu_runs/{g2,g12,g123}_1013/RESULT.md`.
- CURRENT CODE (laptop = box source): v62 forward 060e6946 (G2), Vj2 backward a956e789 (G1+G2+G3); box snapshots
  f5, g2, g12, g123 in ~/gautam_k/v2_work/snapshots. Full 605 NOT yet run on g123.
- STOPPED by the user 20:41 ("after this test stop"). NEXT: full 605 on g123; then forward fixes: G1 for the forward
  (P.V m64n128k16, V descriptor LBO = SWZ_BLK*2 = 16384 B), G5 tail, G6 O via TMA store; then G7 packed Vj2.
**Docs 10-13 written (21:xx):** work doc §43 (4191-4417) covers the fixed-Vj2 tests, Phase 1.5, FA3/FA4/cuDNN proofs,
profiling round + gap list, G2/G1/G3, start-vs-now tables, SFT walkthrough (proof outputs re-run with
~/Downloads/tf_issue_venv/bin/python3 and saved). Push block given to the user: records repo only (232 text/CSV files,
21.8 MB; binary .ncu-repz/.nsys-rep/.sqlite 107 MB left out). Code (Tensor-Implementations submodule, uncommitted since
10-10) NOT pushed: user's call.

**10-14 Phase 1 (Vj2 backward to saturation), GPU 7; user runs the commands himself when he asks for them.**
- Test 1.3 (nsys one backward call, `gpu_runs/p1_1014/RESULT_1.3.md`): helpers+fills at PARITY (ours 381, FA3 373, FA4 378,
  cuDNN 365 us); "FA4 helpers 61 us" (10-12) was wrong. Gap to FA3 = main kernel 2528.7 vs 2485.5 us (+1.7%).
  Small lever B-1: fold the 67.9 us dQ-buffer memset into bwd_prep_vj2 (FA3 Clear_dQaccum) -> up to ~20 us.
- Test 1.4 (lineinfo ncu, `p1_1014/RESULT_1.4.md`): per-line CSV lacks the "warpgroup arrive" reason; it = unexplained
  samples. Ours 16.7% vs FA3 10.4%: warps wait for siblings at the first HGMMA of dK (+74 us-eq) and dV (+57) GEMMs.
  Also mio on SS HGMMA 121 vs 49. FA3 order differs: dQ GEMM then wait<=1, release dO early, dK GEMM then wait<=1.
  Proposed B-series: B-1 memset into prep, B-2a FA3 wait placement, B-2b early dO release, B-3 MIO-class instr per step.
- **B-2a DONE + KEPT (10-14, `gpu_runs/b2a_1014/RESULT.md`):** FA3 wait before dK. Quick 144/144 identical; speed bwd 0.9806
  (t -8.7, 13/13), fwd+bwd 0.9861 (13/13); ncu kernel 2526.3 -> 2478.0 us (FA3 2481.4) = EQUAL to FA3 on the main kernel.
  Warpgroup arrive 16.7% -> 12.2% (FA3 10.4%); dK start 215.5 -> 148.2 (FA3 141.1). Costs: new wait 50, short_sb +64.
  Next gaps: mio +93 (B-3), dispatch +51, arrive +44 (B-2b), helpers +8 (B-1). Our GPU = 0 from 14:13.
- **10-14 eve, GPU 3: B-2b SKIPPED** (compute warps wait for data only 0.6 us-eq; mbarrier waits total 4.8 vs FA3 43.3 -> <=5 us upside).
  **G4 DONE + KEPT** (`gpu_runs/g4_1014/RESULT.md`, Vj2 41318b58, snapshot g4): dQ scale moved to convert kernel (FA3
  postprocess :213), -32 FMUL/step. 144/144; bwd g4/b2a 0.9729 (t -8.0, 13/13), B8 18/6 3.221->3.119 ms; fwd+bwd 0.9831.
  Remaining vs FA3 (b2a ncu): float-math short_sb +78 (LSE LDS wait at the exp FFMA, 45 us-eq on one instr), HGMMA mio
  +60, arrive +44, DEPBAR +75. Test 3 (ncu on g4) not yet run.
- **G4 test 3 (ncu, GPU 1): main kernel 2406.7 us vs FA3 2481.4 = 3.0% FASTER than FA3** (b2a 2478.0). Float instr =
  FA3 (158.1M). Biggest remaining wait: own LDS arriving late (D load -> first dS FADD 85 us-eq; LSE -> exp FFMA 42.7).
  Next proposed B-4a: D loads before the S/dP GEMM issue (G3-style). OURS = GPU 1 from 19:05.
- **B-4a (D loads before the GEMM issue) REVERTED 10-14:** 144/144 identical, but bwd b4a/g4 1.0234 (t +9.4, slower
  13/13). Laptop source back to g4 41318b58. Lesson: loads placed before the HGMMAs delay the GEMM start (same queue);
  the D wait must be fixed another way (e.g. after the S issue / before the dP issue, or prefetch one step ahead).
- **B-4a ncu (10-14):** b4a 2459.6 us vs g4 2406.7; short_sb -155 (D wait gone) but both-warpgroups barrier before dQ
  +166 (51.8->218.0 at the dQ HGMMA), mio +90, dP wait +51. The g4 "D wait" is mostly unavoidable waiting (the two
  warpgroups must meet before dQ) -> STOP trying load placements. Next: B-1 (dQ memset into prep) + same-session
  scoreboard vs FA3/FA4/cuDNN, then Phase 2 forward (fwd 1.21x FA3 on B8 18/6).
