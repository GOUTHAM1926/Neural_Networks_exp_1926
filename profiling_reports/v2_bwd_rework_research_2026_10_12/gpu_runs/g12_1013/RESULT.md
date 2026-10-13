# G1 (full-width register-A backward GEMMs: dV, dK as m64n128k16) - 2026-10-13 20:06-20:19, GPU 2: NEUTRAL, KEPT

Snapshot g12 = lib 41dda051 (Vj2 22f130e2 = G1 + G2; v62 060e6946 = G2), compared with g2 (lib 1a00c378).
- SASS: 8 x HGMMA.64x128x16 (A=regs) + 24 x HGMMA.64x64x16 per step = FA3's mix; was 16 + 24. 0 spills.
- Quick CLASSIC 144/144, all 88 error values identical to base (dK/dV bitwise in G4).
- Same session g2/g12 x2 (`2_ab_g2_vs_g12.txt`): bwd g12/g2 mean 0.997 (8/13 faster, B8 18/6 0.997); fwd identical code
  in both, mean 1.001 (noise check, single shapes up to +-5%).
- ncu B8 18/6 (`3_ncu_g12.txt`): 2547 -> 2532 us; instructions 743 -> 731M; tensor busy 65.9 -> 66.0%; FA3 2481 us
  (gap 2.0%). short_scoreboard (LSE wait) still 11.3%.
- Kept (correct, removes 8 HGMMA per step, not slower) per [[keep neutral cleanups]].
