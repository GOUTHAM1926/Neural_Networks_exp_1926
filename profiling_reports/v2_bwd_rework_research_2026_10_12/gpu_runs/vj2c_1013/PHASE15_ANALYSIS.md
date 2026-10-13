# Phase 1.5 analysis: why the fixed Vj2 is still 1.11x FA3 (2026-10-13, no GPU runs)

Sources:
- our ncu: `2b/ncu_vj2.*` (fixed Vj2, dense B8 18/6 T4096, 11th launch)
- FA3 ncu: `../../v2_step6_benchmarks_2026_10_12/ncu_6c/libs_dense.raw.csv` (same shape, 10-12)
- FA3 source: Dao-AILab/flash-attention main @ 1fc0221183e799363001394a12dd9e78e15f5dda, `hopper/`

## 0. FA3 runs the same layout as Vj2

FA3's kernel name (our nsys, `2b/nsys_fa3_cuda_gpu_kern_sum.csv`) decoded with the template order of
`mainloop_bwd_sm90_tma_gmma_ws.hpp:30-34`: 64 queries x 128 keys per block, SdP_swapAB = 1, 2 compute
warpgroups, AtomLayoutNdKV = 2 (64 keys per group), AtomLayoutMdQ = 1 (dQ split by head columns), P and dS in
registers for dV/dK (Mma_dKV_is_RS = 1). Same work split as Vj2, so a line-by-line comparison is fair.

| | FA3 | Vj2 fixed |
|---|---|---|
| kernel time | 2.494 ms | 2.778 ms |
| instructions | 625M | 840M |
| shared-memory LOAD instructions | 19,464,192 | 19,464,192 (identical) |
| shared-memory STORE instructions | 10,911,744 | 21,823,488 (2.0x) |
| store bank conflicts | 1.8M | 56.3M (31x) |
| tensor pipe busy | 68.8% | 59.6% |
| SM busy / elapsed cycles | 98.6% | 96.9% |
| stall samples | 272,616 | 308,962 (+36,346) |
| of which mio_throttle | 10,851 | 35,222 (+24,371) |
| of which short_scoreboard | 15,121 | 31,205 (+16,084) |

The two shared-memory-queue categories are +40k samples, more than the whole +36k excess.

## 1. dQ staging stores: 2x the instructions, 2-way bank conflicts (ROOT)

`store_acc_sw128_f32` (GQA_bwd_vj2_sm90.cu:289-304) writes the fp32 dQ tile in the TMA SW128 layout: 16 STS.64
per thread per step. In each half-warp, rows r and r^1 land on the same 8 banks: 4 wavefronts per instruction
instead of 2. The epilogue's bf16 staging (`stage_acc_bf16`, :307-318, unswizzled rows of 128 B) is 8-way.
Exact check against ncu: 19,464,192 x 4 + 2,359,296 x 8 = 96,731,136 wavefronts (ncu: 96,731,136);
ideal 19,464,192 x 2 + 2,359,296 = 41,287,680 (ncu: 41,287,680).

FA3 (`mainloop_bwd...:195-198, 951-957, 645-654`): flat unswizzled fp32 buffer per warpgroup, 128-bit stores in
thread order (8 per thread per step, no conflicts), then ONE 1-D bulk reduce-add (`SM90_BULK_REDUCE_ADD`) into
a global accumulator kept in that same thread order; `FlashAttnBwdPostprocessConvertdQ` (its kernel name carries
the TiledMMA) maps it back to row-major bf16. Its convert kernel costs 104.5 us, ours 106 us.

Fix 1: the FA3 scheme. Changes: staging store, writer (1-D bulk reduce instead of the 2-D tensor reduce),
global accumulator order, convert kernel. Same fp32 values are added, so the math is unchanged.

## 2. LSE values arrive late (short_scoreboard 9.5% on two FFMA lines)

SASS bd030-bd0f0: the 8 LSE LDS.64 are issued BEFORE the wait, between the last HGMMAs of the dP GEMM; the
FFMAs at bd120 / bd150 still wait for them (4.38% + 5.09% of samples). Issued 20-60 instructions earlier, a
plain load would have returned, so they sit in the shared-memory queue (mio_throttle 11.4%), which the dQ stores
of the other group and the HGMMA issues also use.
FA3 (`:834-841`): LSE copy right after the S GEMM issue, BEFORE the dP GEMM issue; same LDS count.

Fix 2: FA3 order (move the LSE load loop between the two `gemm_kq_issue` calls). Small. Alone its gain is
uncertain while the queue is still full; it should pay after fix 1.

## 3. dQ hand-off (FENCE.VIEW.ASYNC.S 4.19% + BAR.ARV 3.10%, + MEMBAR 0.43%)

Both stall on long_scoreboard right after the 16 STS.64: the fence waits until this thread's stores have
landed. dK's GEMM is already done by then (the following wait costs 0.12%), so this wait is exposed.
FA3 uses the identical sequence (copy, fence_view_async_shared, NamedBarrier::arrive; `:955-957`).
No separate code change: its cost is the drain time of fix 1's stores. Re-measure after fix 1.
Also here: 2.16% waiting for the writer to release the buffer (DQ_EMPTY). A second dQ buffer (+32 KB) does not
fit (197.7 KB used, 227 KB max); FA3 has one buffer too.

## 4. Block order: tail at the end (NEW)

SM busy 96.9% of elapsed (FA3 98.6%). Our grid runs all 32 key blocks of a head together, so the heaviest blocks
(kt = 0, 64 steps) of the last heads start in the last wave. FA3 causal uses `SingleTileBwdLPTScheduler`
(`tile_scheduler.hpp:368-469`, `flash_bwd_launch_template.h:96-99`): 1-D grid, key block 0 (heaviest) first,
heads grouped in L2 sections (40 MB / 4 MB per head -> 8 heads for our shape).
Fix 4: same index mapping. Upper bound about 1.7% (the busy-cycle gap).

## 5. Epilogue staging 8-way conflicts (NEW, small)

Once per block: 64 STS lines, 2.19% of samples. Fix 5: swizzled (or stmatrix) bf16 staging.

## How they connect

Fix 1 removes half the store instructions and all their conflicts -> shortens the fence wait (3) and empties the
queue the LSE loads wait in (2). 4 and 5 are independent of 1-3.

Plan: one at a time, cumulative: 1 -> 2 -> 4 -> 5 (3 is re-measured after each). Per step: quick CLASSIC
(identical errors expected), same-session speed vs the previous step (library snapshots), ncu on the main
kernel. Full CLASSIC (605) once at the end.

## RESULT fix 1 (GPU 7, 16:14-16:23; snapshots base = lib d2659657 / Vj2 bbd977be, f1 = lib 84c91617 / Vj2 a4f7d5f6)

Records: `f1/`.
- Quick CLASSIC on f1: 144/144; all 88 error values identical to the base run (14:10), floors identical;
  G4: O/LSE/dK/dV bitwise, dQ max diff unchanged (4.883e-04, atol 1.562e-02).
- Same-session speed (base, f1, base, f1), backward median: f1/base 0.932-0.968, mean 0.943, faster on 13/13
  (`f1/ab_base_vs_f1.txt`). B8 18/6: 3.469 -> 3.269 ms.
- nsys B8 18/6 main kernel median: 2761.1 -> 2582.0 us (-6.5%); FA3 2485.8 us, so 1.111x -> 1.039x FA3.
  New convert kernel 105.9 us (old 106.2 us).
- ncu (`f1/ncu_base_vs_f1.txt`): kernel 2.778 -> 2.599 ms; instructions 840M -> 795M; tensor pipe busy
  59.6% -> 65.2% (FA3 68.8%); store instructions 21.8M -> 12.1M; store bank conflicts 56.3M -> 18.3M
  (the rest is mostly the epilogue staging, fix 5); mio_throttle 11.4% -> 8.0%.
  Loop stall shares: STS 6.54% -> 3.48%, BAR.ARV 3.10% -> 1.23%, MEMBAR 0.44% -> 0.04%, FENCE 4.36% -> 5.04%.
- NOT confirmed: the expectation that fix 1 also removes the LSE wait. The FFMAs waiting on the LSE loads went
  9.70% -> 10.44% of samples. Fix 2 has to deal with it directly.

## RESULT fix 2 (GPU 7, 16:27-16:35; snapshot f2 = lib 6ffda947 / Vj2 ab6e5cbb, on top of f1): NULL, REVERTED

Records: `f2/`.
- Quick CLASSIC 144/144, all 88 error values identical to base.
- Same-session speed f1/f2/f1/f2: f2/f1 0.983-1.021, mean 1.003, faster on 4/13 (noise) (`f2/ab_f1_vs_f2.txt`).
- nsys main kernel 2605.4 us (f1 2582.0); ncu 2.614 vs 2.599 ms; short_scoreboard 10.9% -> 11.8% (`f2/ncu_f1_vs_f2.txt`).
- SASS (`f2/ncu_vj2.source.csv`): the 8 LSE LDS now sit before the last S HGMMA and all 8 dP HGMMAs, yet the
  FIRST use of each loaded register still stalls: R14 (loaded ~70 instructions earlier) 3.68%, R10 1.09%,
  R6 5.05%. Later uses of the same registers do not stall. So issue time is not the cause.
- Best explanation, NOT proven: the LDS results wait for shared-memory bandwidth while the 16 SS-wgmma of the S
  and dP GEMMs read their operands. FA3 issues the same LDS at the same point and also has short_scoreboard
  stalls (5.5% of its samples).
- Reverted to the f1 code (Vj2 file back to a4f7d5f6). Possible later experiments, not planned now: FA3's
  ShuffleLSE (fewer LDS + SHFL, FA3 uses it only for head dim <= 64), or reading the next step's LSE/D at the
  end of the current step.

## RESULT fix 4 (GPU 7, 16:39-16:49; snapshot f4 = lib 521a9718 / Vj2 22c0a1e5 = f1 + block order): KEPT

Records: `f4/`.
- Quick CLASSIC 144/144, all 88 error values identical to base. G4: O/LSE/dK/dV bitwise; dQ run-to-run differences
  grew (B8: 393 -> 7144 elements, max 4.883e-04 -> 9.766e-04; B60: 0 -> 1305 elements), still 16x inside atol:
  more blocks now add into the same dQ tile at the same time.
- Same-session speed f1/f4/f1/f4: f4/f1 0.904-1.010, mean 0.967, faster on 10/13 (`f4/ab_f1_vs_f4.txt`).
  B2 shapes -5..-10%, B4 0..-7%, B8 -1.2%..+1.0% (B8 24/8 +0.9%, B8 32/8 +1.0%), B8 18/6 -0.6%.
- nsys main kernel B8 18/6: 2571.5 us (f1 2582.0).
- ncu: SMs busy 96.9% -> 98.8% of elapsed cycles (FA3 98.6%), so the tail is gone; but DRAM reads 713 -> 745 MB,
  L2 hit rate 92.15% -> 91.92%, active cycles +2.1%: on B8 the extra DRAM traffic of 8-pair sections cancels most
  of the tail gain. Follow-up to test: smaller sections (4 pairs = 16 MB working set) for large grids.

## RESULT fix 5 + section-size experiment (GPU 7, 17:03-17:14; records `f5/`)

Snapshots: f5 = f4 + epilogue swizzle (lib a07ca28b, Vj2 826ae057); f4s4 = f4 with a 20 MB section budget,
4 pairs (lib 2f23ad6c, Vj2 c5ae7761, experiment only).
- Test 1, quick CLASSIC on f5: 144/144, all 88 error values identical to base; G4 dK/dV bitwise, dQ max diff
  9.766e-04 (atol 1.562e-02).
- Test 2, same session f4, f5, f4s4 x2 (`f5/ab_f4_vs_f5.txt`, `f5/ab_f4_vs_f4s4.txt`):
  f5/f4 0.962-1.023, mean 0.989, faster on 12/13 (the one loss, B4 16/4 +2.3%, has a 6.9% f4 run-to-run spread).
  f4s4/f4 0.962-1.014, mean 0.994, faster on 7/13, B8 shapes unchanged -> NOT kept.
- Tests 3-4, ncu B8 18/6: f4 / f5 / f4s4 kernel 2.590 / 2.545 / 2.574 ms; store bank conflicts 18.3M / 1.79M /
  18.3M (FA3 1.81M); tensor busy 65.1 / 66.7 / 65.3% (FA3 68.8%); shared-memory queue stalls 8.2 / 6.3 / 8.1%;
  DRAM 745 / 744 / 724 MB. f4s4 cuts DRAM 2.8% but gains only 0.6% in time.
- KEPT: f5 = fix 1 + fix 4 + fix 5. Cross-session chain for B8 18/6 (ncu): first fixed Vj2 2.778 ms -> f5 2.545 ms
  (0.916x); FA3 2.494 ms (10-12 ncu), so f5 is about 1.02x FA3 on this shape (cross-session, to confirm in one
  session).

## RESULT test 6: full CLASSIC on f5 (GPU 7, 17:14-17:37; `f5/6_full_f5.log`)

- 605/605 pass. All 435 error values identical to the 15:17 full run of the fixed Vj2 (O 89, LSE 89, dQ 84, dK 84,
  dV 89), floors identical, including the 5000 real chats (D10).
- G4 x1000: O/LSE/dK/dV bitwise every time. dQ run-to-run: B8 max diff 1.953e-03 (15:17 run: 4.883e-04), 1,451,773
  of 75.5M elements differ (was 4,025); B60 x250 max 9.766e-04, 34,412 differ (was 0). Atol 1.562e-02 / 3.125e-02,
  so 8x / 32x inside. Cause: the new block order (fix 4) runs more key blocks of the same head at once, so the
  fp32 adds into each dQ tile happen in more different orders.
