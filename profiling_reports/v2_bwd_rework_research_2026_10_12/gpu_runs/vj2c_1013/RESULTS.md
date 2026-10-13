# Fixed Vj2 on GPU 7, 2026-10-13 (GQA_bwd_vj2_sm90.cu md5 bbd977be, lib d2659657, tva_vj2 a68c366b)

One test at a time, each with the user's OK. Logs in this folder.

| Test | When | Result |
|---|---|---|
| 1a quick CLASSIC (USE_VJ2=1) | 14:10-14:15 | 144/144 pass; all 88 error lines identical to the first Vj2 (10-12 smoke) |
| 1b compute-sanitizer, sanitize_vj2 B1 6/2 T256 | 14:16-14:17 | memcheck 0, synccheck 0, racecheck 0 hazards (racecheck does not see TMA/wgmma) |
| 2a dense speed, 13 shapes, ABAB + libraries | 14:24-14:29 | Vj2/Vj1d 0.78-0.82; Vj2/FA3 1.05-1.15; faster than cuDNN 12/13 (`2a/bwd_table.txt`) |
| 2a reachability | 14:26 | nsys of sanitize_vj2 shows gqa_bwd_vj2_kernel (1 call) |
| 2b nsys Vj1d / Vj2 (first half, stopped 14:39 by the user) | 14:36, 14:38 | reports in `2b/first_half_1434/` |
| 2b nsys FA3 + ncu Vj2 (second half) | 15:09-15:14 | `2b/` |
| 3 full CLASSIC (USE_VJ2=1) | 15:17-15:39 | 605/605 pass; all 435 error lines identical to the first Vj2 full run (10-12); G4 dQ run-to-run: max diff unchanged (4.883e-04, atol 1.562e-02) but more elements vary (B8: 666 -> 4025; B60: 0 -> 56, max 9.5e-07) (`3/compare_vs_first_vj2_full.txt`) |

## Kernel times, nsys, dense B8 18/6 T4096, median of 15 calls (us)

| | main kernel | helper kernels | total |
|---|---|---|---|
| Vj1d | 3623 | 88 + 107 + 2x52 = 300 | 3923 |
| Vj2 first (10-12) | 3197 | 102 + 107 + 2x52 = 314 | 3511 |
| Vj2 fixed | 2761 | 101 + 106 + 2x52 = 312 | 3073 |
| FA3 (Deterministic=0) | 2486 | 150 + 105 + 2x37 = 328 | 2814 |
| FA3 (Deterministic=1) | 3081 | | |

Fixed Vj2 main kernel = 0.762x Vj1d, 0.864x first Vj2, 1.111x FA3.
Note: BENCH_ONLY=fa3 did not limit the --dense contenders, so the FA3 nsys run also ran FA2/FA4/cuDNN (kernel times unaffected).

## ncu, fixed vs first Vj2 (same shape, 11th launch): `2b/ncu_before_after.txt`

- instructions 1,095.6M -> 840.3M (0.767x); FA3 was 625M on 10-12
- generic LD/ST 161 lines / 44.9M executed -> 0; now LDS 16 lines + STS 80 lines
- tensor pipe busy 51.4% -> 59.6% of elapsed; lg_throttle stall 6.3% -> 0%
- registers 168, L2 read requests and DRAM unchanged

## Next leads found (not started)

1. dQ staging STS has bank conflicts: 96.7M wavefronts vs 41.3M ideal (2.34x); STS = 8.7% of stall samples.
2. short_scoreboard 10.1%, almost all on two FFMA lines (5.1% + 4.4%) that use the LSE value just loaded by LDS: the LSE load is not issued early enough.
3. dQ hand-off: FENCE.VIEW.ASYNC.S 4.0% + BAR.ARV 3.1% (long_scoreboard) and a MEMBAR.ALL.CTA in the dQ path.
