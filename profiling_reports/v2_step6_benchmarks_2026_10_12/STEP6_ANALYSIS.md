# Step 6 (2026-10-12): benchmarks, profiling, and the test-coverage audit

All runs on the H200 box, GPU 7, steady 1980 MHz. Env: `~/gautam_k/benv` (FA3 built from FlashAttention `94e22c9`
`hopper/`, bf16 + head_dim 128 + sm90, all features for that config kept; FA4 = `flash-attn-4` from the same commit;
torch 2.14.1 + cuDNN 9.24 reused read-only from `~/envv`).

## 1. Packed attention (8 rows x 4096, 18/6 heads): ours vs the libraries

Same segments for everyone (our harness exports them). Every packed contender's output and dQ were checked against
FA3 on the same inputs: all agree to bf16 noise (O <= 1.6e-2, dQ <= 3.1e-2). Times are per API call, median of 50
interleaved rounds (`full_ours_1247.log`, `full_libs_1247.log`, `tables_1247.txt`).

Forward + backward, ms:

| Layout | ours A | ours B | FA3 | FA3 det | FA4 | Flex | torch varlen (FA2) | cuDNN + mask (HF SDPA way) |
|---|---|---|---|---|---|---|---|---|
| real rows, shortest chats | **1.49** | 1.53 | 1.60 | 1.69 | 1.62 | 2.03 | 2.43 | 14.5 |
| all chats 512 | 1.44 | **1.42** | 1.54 | 1.62 | 1.55 | 1.68 | 2.32 | 14.5 |
| real rows, middle | 2.44 | 2.57 | **2.10** | 2.30 | 2.16 | 3.14 | 3.73 | 14.5 |
| all chats 1024 | 1.96 | 2.01 | **1.83** | 1.99 | 1.85 | 2.33 | 3.08 | 14.5 |
| all chats 2048 | 3.09 | 3.22 | **2.54** | 2.81 | 2.61 | 3.63 | 4.64 | 14.5 |
| real rows, longest chats | 5.07 | 5.33 | **3.89** | 4.31 | 3.92 | 5.97 | 7.53 | 14.5 |
| one 4096 chat per row | 5.43 | 5.67 | **4.06** | 4.53 | 4.16 | 6.29 | 7.97 | 14.6 |

- Short chats: ours fastest. Middle/long chats: FA3/FA4 faster (16% on typical real rows, ~30% on long chats).
- A vs B: A equal or faster almost everywhere (forward always); B wins only the shortest-chat backward by 3%. A = main path.
- The masked-SDPA way computes the whole row: 6-10x slower than everything else.
- Noise floor: FA3 and FA3-det run the identical forward kernel yet differ ~4% (position in the round). Differences
  under ~5% in this run are not meaningful.

## 2. Dense attention at the colleagues' shapes (`dense_tables_1253.txt`)

T = 4096, hd 128, B 2/4/8 x heads 12/4, 16/4, 24/8, 32/8, plus 18/6 B8. Same run, same GPU.

- Forward: ours faster than cuDNN on 11 of 13 shapes (0.67-0.99x); slower at B8 24/8 and B8 32/8 (1.02-1.03x).
  This matches the colleagues' forward table.
- Backward: ours slower than cuDNN on all 13 shapes (1.05-1.24x), worst at B=8; slower than FA3 by 1.32-1.42x and than
  FA3-deterministic by 1.05-1.17x. This does NOT match the colleagues' backward table (they reported wins vs cuDNN on
  all 12). Their measurement method is unknown to me; this run is same-session, same GPU, same shapes.

## 3. Profiling (nsys `nsys_6b/`, ncu `--set full` `ncu_6c/`), dense B8 18/6, kernel against kernel

Per call (nsys):

| | Ours | FA3 | FA3 det | FA4 | cuDNN |
|---|---|---|---|---|---|
| forward kernel | 1075 us | 890 | 890 | 964 | 1017 |
| backward main kernel | 3633 us | 2495 | 3283 | 2482 | 2836 |
| backward helpers | 298 us + zero-fill | 255 | 255 | 61 | ~350 (PyTorch GQA expand/sum) |

Packed real middle rows: ours A forward 507 us vs FA3 427; backward main ~1498 vs FA3 1082 (det 1367).
The API-level benchmark understated the kernel gap (FA3's Python call adds ~0.1 ms per call).

ncu (main kernels):

| Kernel | Time | Instructions | SM busy | Tensor-core cycles | Shared-mem wavefronts | CTAs x threads |
|---|---|---|---|---|---|---|
| ours bwd | 3626 us | **888 M** | 44.9% | 43.0% | **662 M** | 9216 x 128 |
| FA3 bwd | 2494 us | 625 M | 68.8% | 57.3% | 419 M | 4608 x 384 |
| FA4 bwd | 2502 us | 555 M | 68.9% | 45.7% | 419 M | 4608 x 384 |
| cuDNN bwd | 2788 us | 809 M | 60.1% | 40.8% | 405 M | 132 x 384 |
| ours fwd | 1086 us | 360 M | 64.4% | 31.7% | 104 M | 132 x 384 |
| FA3 fwd | 886 us | 307 M | 82.9% | 36.6% | 102 M | 132 x 384 |

Packed real middle rows, ncu (`ours_packed.raw.csv`, `fa3_packed.raw.csv`):

| Kernel | Time | Instructions | Shared-mem wavefronts |
|---|---|---|---|
| ours A fwd | 501 us | 151 M | 40.9 M |
| FA3 varlen fwd | 432 us | 142 M | 41.2 M |
| ours A bwd | 1466 us | 335 M | 250 M |
| FA3 varlen bwd | 1086 us (det 1362) | 248 M | 159 M |

Same pattern as dense: backward +35% time, +35% instructions, +57% shared-memory traffic.

Where the time goes (ncu per-instruction stall sampling, `ours_dense.source.csv`):
- **Backward:** ~16% of samples in the fences before each dQ TMA reduce (`MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S`,
  waiting for 32 KB of staged dQ to become visible), 3.5% in the dQ staging stores, ~10% waiting for GEMMs
  (`WARPGROUP.DEPBAR`, one warpgroup, no overlap). With 64-key tiles this per-tile dQ work happens twice as often per
  key as in FA3 (128-key tiles), and P makes a round trip through shared memory (written as bf16, read back for dS).
  Result: +42% instructions and +58% shared-memory traffic vs FA3, which matches the +45% time.
- **Forward:** a third of the samples are mbarrier wait loops, mostly the producer warpgroup spinning while idle (as
  the v62 code comment already says) -- harmless. The real gap to FA3 is +17% instructions and lower tensor-core use.

## 4. What would close the gap (options, not done)

Backward, in order of expected effect:
1. 128-key tiles per CTA (halves Q/dO loads, dQ staging, fences and reduces per key). Needs a producer warpgroup plus
   2 consumer warpgroups (FA3's structure): the dK/dV accumulators of 128 keys do not fit one warpgroup's registers.
2. Keep P in registers for the dV GEMM and compute dS from fp32 P (FA3 does both). Removes the P round trip through
   shared memory AND fixes precision mechanism 1 (bf16 P in dS) at the same time.
3. Warp-specialised loads (TMA producer) to overlap loads with the GEMMs.
These amount to a new backward kernel version. Expected: close to FA3 (~2.5 ms dense, -30%).

Forward: cut instructions in the softmax/epilogue path (17% excess); smaller expected gain (~15%).

## 5. Test-coverage audit (what was missing, and what now covers it)

| Dimension | Covered before today | Gap | Now |
|---|---|---|---|
| G8 / G10 / G11 at full size (5 head setups, all 1517 rows, 1000x repeat) | quick only | yes | running (suite_1012_1319) |
| G1-G7 at full size after the sync fix | quick + G4 1000x | yes | running |
| Packed rows with T != 4096 | no | yes | G12: T 1024, 2048, 8192 |
| Chat > 8192 tokens (B backward prefix loop 2nd pass) | no | yes | G12: T 16384 (2/1 heads) |
| 4096 one-token chats in one row (A and B) | 512 only | yes | G12 |
| Large input values on packed rows | dense only (G5) | yes | to add |
| Packed batch B > 8 per call | no | yes | to add (B = 60, like G4's dense case) |
| Two streams at once / CUDA graph capture | no | yes | known limitation: Vj1d uses static scratch buffers, so it is not safe on two streams at once (pre-existing); to document or fix before the trainer uses streams |
| cuDNN varlen | no | yes | DONE 2026-10-12 evening: `cudnn_varlen_1012/CUDNN_RESULTS.md`. No Transformer Engine needed (torch 2.14 op, called as `varlen.py` does). Forward ours A within 5% of cuDNN; backward cuDNN 9-16% faster on middle/long chats, equal on short. |
| Packed FA3 ncu | partial | yes | done (`fa3_packed.raw.csv`); FA4 packed not captured |
| End-to-end: packed batch gradients == the same chats as separate v1 rows (whole model) | no | yes | Step 7's first test |

## 6. Open decisions

1. Fix the backward gap before Step 7 (new backward version, item 4) or after the trainer works.
2. The two precision items (fp32 P in dS; fp32 head sum for dK/dV): item 4.2 fixes the first one as a side effect.
3. Keep design B in the code (costs nothing) or drop it.
