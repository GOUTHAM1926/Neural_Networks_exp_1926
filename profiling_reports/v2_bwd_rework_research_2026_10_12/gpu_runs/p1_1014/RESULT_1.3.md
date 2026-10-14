# Phase 1, test 1.3 - where the time goes in one backward call (2026-10-14 11:05-11:15, GPU 7, run by the user)

nsys --trace=cuda, dense B8 18/6 T4096, 10 warm-up + 5 timed calls. ours = Vj2 g123 (lib 1b9f444d, USE_VJ2=1);
libraries = bench_step6_libs.py --dense dense_B8_H18x6 in one capture. Medians over the 15 calls of each operation (us).
API timing in the same capture: ours forward 1.111 ms, backward 2.926 ms (p10-p90 2.924-2.929).

| piece of one backward call | ours (Vj2) | FA3 | FA4 | cuDNN |
|---|---|---|---|---|
| zero fills | memset of the fp32 dQ buffer 67.9 | 2 torch fills (dK, dV fp32 accumulators, flash_api.cpp:1493-1494) 45.0 | 2 torch fills 44.9 | memset 0.7 |
| prep (D = rowsum(dO*O), LSE) | 101.8 | preprocess 149.2 (also zeroes dQaccum inside: Clear_dQaccum=true, flash_bwd_launch_template.h:50) | preprocess 151.4 | dot(dO,O) 148.6 |
| main kernel | **2528.7** | **2485.5** | **2477.0** | 2774.1 |
| after | dK/dV head sums 2 x 52.5 + dQ convert 106.2 = 211.2 | dQ convert 104.5 + dK/dV converts 2 x 36.9 = 178.3 | postprocess 105.7 + 39.2 + 37.0 = 181.9 | dQ convert 116.2 + head sums 99.0 = 215.2 |
| **helpers + fills total** | **380.9** | **372.5** | **378.2** | **364.5** |
| **one backward call (GPU time)** | **2909.6** | **2858.0** | **2855.2** | **3138.6** |
| idle gaps inside the call | 1.0-2.1 per launch | ~1.1 per launch | ~1.1 per launch | ~1.1 per launch |

Findings:
- Helpers + fills are at parity with every library (364-381 us). The 10-12 note "FA4 helpers 61 us" counted only part
  of FA4's helpers and is wrong; FA4 also has a 151 us preprocess, three postprocess kernels and two fills.
- The end-to-end gap to FA3 (+51.6 us, 1.8%) is the main kernel (+43.2 us, 1.7%) plus 8.4 us of helpers.
- Small helper lever (B-1): our separate memset (67.9) + prep (101.8) = 169.7 us vs FA3's single preprocess that also
  zeroes the dQ buffer (149.2 us): zeroing inside our prep kernel could save up to ~20 us per call (~0.7%).
- Our dK/dV head sums + dQ convert (211.2 us) vs FA3's converts + accumulator fills (178.3 + 45.0 = 223.3 us): ours is 12 us less.
- The 15-160 us gaps BETWEEN calls are CPU launch / Python time, not GPU work.
