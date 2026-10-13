# G3 (LSE loads before the S GEMM issue) - 2026-10-13 20:29-20:41, GPU 2: KEPT

Snapshot g123 = lib 1b9f444d (Vj2 a956e789 = G1 + G2 + G3; v62 060e6946 = G2), compared with g12 (lib 41dda051).
- SASS: all 8 LSE LDS.64 before the first S HGMMA (laptop CUDA 13.0 and box CUDA 13.4). 0 spills.
- Quick CLASSIC 144/144, all 88 error values identical to base; G4 dK/dV bitwise, dQ max diff 1.953e-03 (atol 1.562e-02).
- Same session g12/g123 x2 (`2_ab_g12_vs_g123.txt`): bwd g123/g12 mean 0.994, faster on 9/13, B8 18/6 0.995;
  fwd (identical code) mean 1.004 = noise check.
- ncu B8 18/6 (`3_ncu_g123.txt`): 2532.3 -> 2512.5 us (FA3 2481.4, gap 1.25%); short_scoreboard 11.3% -> 3.1%
  (FA3 5.5%); tensor busy 66.0 -> 67.0% (FA3 68.3%). The wait moved to warpgroup_arrive 13.7 -> 16.8% (waiting on wgmma).
