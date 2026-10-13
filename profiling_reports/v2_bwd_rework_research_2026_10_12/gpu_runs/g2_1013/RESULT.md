# G2 (uniform warpgroup index, __shfl_sync broadcast) - 2026-10-13 19:46-20:00, GPU 2: NEUTRAL, KEPT (user decision)

Snapshot g2 = lib 1a00c378 (v62 forward 060e6946, Vj2 backward bed7731f); base = f5 (lib a07ca28b).
- Quick CLASSIC 144/144, all 88 error values identical to base.
- Same-session speed (f5, g2 x2, 13 dense shapes): fwd g2/f5 mean 0.990 (11/13 faster, B8 18/6 0.993);
  bwd mean 0.999 (4/13, B8 18/6 1.008) (`2_ab_f5_vs_g2.txt`).
- ncu B8 18/6 (`3_ncu_g2_vs_f5.txt`): bwd R2UR 59.1M -> 0.66M, IMAD 74 -> 41M, total 796 -> 743M, but uniform-datapath
  instructions 137 -> 212M and kernel time 2545 -> 2547 us, tensor busy 66.7 -> 65.9%. fwd R2UR 40.2 -> 29.9M,
  total 360 -> 349M, time 1079 -> 1084 us.
- Conclusion: neither kernel is instruction-issue bound; the gap to FA3 is tensor-core idle time, not instruction count.
  Both files restored (v62 698d2705, Vj2 826ae057). Source of g2 kept in code_snapshot/*.g2.cu.
UPDATE 20:15: first reverted under "keep only what wins"; the user asked to KEEP it (it removes real redundant work, is
correct, and is not slower; it may pay off together with later fixes). Re-applied: v62 forward 060e6946; the backward
carries it on top of G1. G1 is measured against the g2 snapshot (G1 alone on top of G2).
