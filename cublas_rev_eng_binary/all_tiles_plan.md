# ALL-TILES plan — exact-tile coverage for the remaining ~13 Qwen shapes (2026-07-07)

Decoded from nsys grid math (grid/clusterCTAs = #clusters; solve ceil(M/tm)*ceil(N/tn)=#clusters
against the KNOWN qwen shape each kernel ran on; calibrated on 128x256_2x1_v which fits 256x256
UNIQUELY = our validated geometry).

## Decoded cluster C-tiles (all kernels are _2cta_ = cta_group::2!)

| nvjet kernel | shapes | cluster C-tile (M x N) | per-CTA structure | status/need |
|---|---|---|---|---|
| 128x256_2x1_v | 12 | 256x256 UNIQUE | ours, validated | DONE |
| 128x256_2x2   | 8  | 256x512 | ours (128x256_2x2), validated | DONE |
| 128x256_2x2f  | ~5 | 256x512 dyn | ours, validated | DONE |
| 256x128_2x2 #4| 1  | 256x512 (fits) | = our 128x256_2x2 geometry | COVERED (verify via bench) |
| 256x256_2x2 #41| 1 | 512x512 UNIQUE | per-CTA M=256 = COMPOSITE 2x M_MMA-256 stacked; N 128/CTA (chunks ok) | need composite-M |
| 128x192 (2x1_v + 2x2f) | 5 | 384x128 or 192x256 (ambig; both need composite) | M-composite (256+128 -> 192/CTA, or 128+64 -> 96/CTA); N 128- or 256-per-pair | need composite-M (+interleave if 96) |
| 128x160_2x1_v #16 | 1 | 320x128 or 160x256 | M-composite 256+64 (160/CTA, interleaved M64) or 128+32 | need composite + INTERLEAVED M64 epi (§28.7) |
| 160x192_1x2 #11/21/27 | 3 | 384x160 or 320x192 | our swap-AB B2 covers these shapes TODAY (different geometry, validated); nvjet's = M-composite pair | COVERED by B2; nvjet-geometry optional |
| 160x192_2x2 #1 | 1 | 384x320 | M=192/CTA composite + N 160/CTA (sw64 - HAVE) | need composite-M |
| 320x192_2x2f #19 | 1 | 384x640 preferred | M=192/CTA composite + N 320/pair=160/CTA sw64 + dyn cluster (HAVE) | need composite-M |
| 64x64_2x2 #2 | 1 | 128x128 | per-CTA M=64: M_MMA=128 INTERLEAVED tmem (§28.7!) + N 32/CTA sw64 | need interleaved epi |
| 128x64_2x2 #7/#12 | 2 | 128x256 | per-CTA M=64 interleaved + N 64/CTA | need interleaved epi |

Ambiguities (384x128 vs 192x256 etc.): both readings legal via different composites; benchmark
vs the real nvjet kernel per shape is the arbiter. Where ambiguous we build OUR pick and measure.

## Capability gaps -> build stages

STAGE T1 — composite-M for cta_group::2 (biggest unlock: 9 shapes):
  Multi-MMA along M per K-step: M_MMA pieces from {256, 128, 64}; second/third atom at tmem col
  offsets (Layout D for 256/128 pieces at per-CTA 128/64 rows... NOTE M_MMA=128 on cta_group::2 =
  per-CTA 64 rows uses the INTERLEAVED layout §28.7 — so composite 256+128 needs the interleaved
  read for the +128 piece; 256+64 needs cta_group::2 M64-atom reads). Epilogue: extend the
  swap-AB strip-read machinery (16-lane loads exist) to the cta_group::2 interleaved variants.
  Reference first: CUTLASS mma_traits_sm100.hpp 2SM layout atoms (M_MMA=128 2x2 interleaved map,
  M_MMA=64) — pin exact (row->lane,col) maps BEFORE coding, like the Layout-F work.
STAGE T2 — interleaved-M64 epilogue alone unlocks 64x64/128x64 (3 shapes) — subset of T1 work.
STAGE T3 — config aliases + suites per decoded tile; sw32 only if the 96/80-wide readings win.
STAGE T4 — benchmark each vs its nvjet kernel; keep winners, retire losers.

NOTE: shapes #4 (256x128) + the 3 160x192_1x2 shapes are ALREADY covered by validated kernels
(our 128x256_2x2 and swap-AB B2) — 30/42 exact-or-equivalent today; T1+T2 close the rest.

## RESOLVED: composite-M + interleaved 2SM TMEM maps + swizzle parameterization

Sources: ptx_isa_9.3.txt (PTX), cutlass/include (CUTE), DeepGEMM, and our template
`BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh` (KERNEL).
All file:line cites verified 2026-07-07.

### R1 — cta_group::2 M_MMA=128 accumulator map (per-CTA 64 rows) = PTX Layout B "2x2"

PTX Data Path Layout table (ptx 33617-33631): M=128 + ::2 + Dense = "2x2" organization,
**Layout B**, TMEM lane alignment **0** (no 16-lane option — unlike Layout F). Layout B section
header: ptx 33670 (Figure 207/208 are images; exact map taken from the CUTE atom, which is the
authoritative machine-readable form and matches our validated §28.7 note).

CUTE atom (mma_traits_sm100.hpp:543-551, `tmem_frg`, N_SM=2, per-SM M_MMA_SM=64):
`Layout<Shape<_64,Shape<N/2,_2>>, Stride<_1,Stride<_128,_64>>>` over virtual addressing
DP_STRIDE=1/COL_STRIDE=128 (mma_traits_sm100.hpp:486-491). Note CUTE's `M_MMA` here is the
**per-CTA** M (mma_traits_sm100.hpp:472-474); instruction M=128 -> M_MMA_SM=64. Decoded exact map,
per-CTA row r in [0,64), mma column n in [0,N):

    n <  N/2 :  lane = r,      tmem col = base_col + n
    n >= N/2 :  lane = r + 64, tmem col = base_col + (n - N/2)

i.e. §28.7 "2x2 INTERLEAVED: half-N in DP[0,64), half in DP[64,128)" CONFIRMED, and the split is
low-half/high-half of N (h is the outer coordinate mode: n_logical = nh + (N/2)*h), not an
interleave of individual columns. Per-CTA footprint = **N/2 tmem cols at all 128 lanes**.
Cross-check: 2SM CLayout `Shape<_2,Shape<M/2,N>>` (mma_traits_sm100.hpp:2049-2050) — each CTA owns
its own M/2 rows x all N cols; B is CTA-split N-wise (BLayout `Shape<_2,Shape<N/2,K>>`,
mma_traits_sm100.hpp:2047-2048 — each CTA's smem supplies N/2 of B, exactly our "B smem holds BN/2
per CTA", KERNEL:113).

**ld shapes / warp assignment (4 epi warps):** every lane strip is a full 32-lane subpartition, so
plain `32x32b` reads it cleanly — the EXISTING `tcgen05_ld` 32x32b.x8 helper (KERNEL:417-423), no
16-lane loads needed (16x256b machinery, KERNEL:425-435, stays swap-AB-only):

    warp 0 (lanes   0-31): C rows  0-31, cols [0,   N/2)   tmem cols [base, base+N/2)
    warp 1 (lanes  32-63): C rows 32-63, cols [0,   N/2)   same cols
    warp 2 (lanes  64-95): C rows  0-31, cols [N/2, N  )   same cols
    warp 3 (lanes 96-127): C rows 32-63, cols [N/2, N  )   same cols

Each warp: 32 rows x N/2 cols. Epilogue store indexing: c_row_local = (warp_id&1)*32 + lane_id,
c_col_base = (warp_id>>1)*(N/2) — only the row/col decode changes vs the validated M256 path
(KERNEL:1493-1495); the tcgen05.ld loop shape is identical.

### R2 — cta_group::2 M_MMA=64: **ILLEGAL for kind::f16 — question is moot**

PTX Table 44 (ptx 31320-31344): kind::f16, CTA Group 2, Dense supports **only 128xNxK and
256xNxK** (N={16..256} step 16, K=16; row at ptx 31335-31337). Same restriction for ::2 Sparse and
tf32 (ptx 31374). The Data Path Layout table (ptx 33617-33631) accordingly has NO M=64/::2 row.
CUTLASS enforces it: `static_assert(M == 128 || M == 256, "...2x1SM_SS M-mode size should be 128
or 256...")` (cute/arch/mma_sm100_umma.hpp:554). The CUTE "1x4" M_MMA_SM=32 atom
(mma_traits_sm100.hpp:532-541, map: lane = r + 32*q, col = n - q*N/4 for n in quarter q) exists
only for other kinds/sparse paths — do NOT build on it for bf16.
CONSEQUENCES for the decoded table above:
- 128x160_2x1_v #16 ("256+64" or "128+32" readings): **both illegal**. Build per-CTA M=160 as the
  192-composite (256+128) with the last 32 rows dead: TMA OOB zero-fill on A + epilogue store
  predication (rows >= 160 skipped); optionally suppress the dead lanes with the MMA's
  `disable-output-lane` vector (size 8 for ::2, ptx ~34817-34830) — not required for correctness.
- Line 16's "(128+64 -> 96/CTA)" alternative reading: also uses only legal atoms (256- and
  128-instruction M) — still available.

### R3 — composite-M legality for cta_group::2

Everything the swap-AB ::1 composite relies on carries over; verified constraint list:
1. **Same .cta_group for ALL tcgen05 in the kernel** (ptx 32724) — both atoms use ::2 ✓.
2. **Per-atom idesc**: M-field differs (256>>4 vs 128>>4, bits 24-30), N-field identical — exactly
   the `sm103_make_i_desc_swap<Config, M_ATOM>` pattern (KERNEL:670-675) with the normal (non-swap)
   M/N field assignment of `sm103_make_i_desc` (KERNEL:653-664).
3. **Ordering**: mma->mma pipelining is only guaranteed for "same N and accumulator, shape and
   kind" (ptx 32396-32398). The two atoms write DISJOINT d-tmem so they need no mutual order; each
   atom's own K-step chain (same shape+accumulator+kind) IS pipelined — the same property the
   validated ::1 composite uses (KERNEL:1152-1185, 336/336 bit-exact on B300).
4. **One commit tracks all**: tcgen05.commit implicitly tracks ALL prior mma from the issuing
   thread (ptx 32410-32415 / 32421-32427) — unchanged (KERNEL:1208-1209).
5. **Issue granularity**: only the pair leader issues both atoms (single-thread semantics,
   ptx ~34803-34805) — same as today.
6. **d-tmem lane alignment**: Layout A and Layout B both require lane alignment 0 (ptx
   33617-33631), so the M128 atom CANNOT be lane-offset-16-packed the way the ::1 Layout F M64
   atom is (KERNEL:1183-1185) — its accumulator must take its own column range at lane 0.
7. **Operand offsets for the 2nd atom** (per-CTA rows 128..): each CTA supplies its OWN rows
   (ALayout `Shape<_2,Shape<M/2,K>>`, mma_traits_sm100.hpp:2045-2046), so the offset is local:
   - MN-major 128B-swizzled [K,M] slice (TN A): rows 128.. = 64-elem chunk #2 -> **+2*LBO =
     +16384 bytes** at BK=64 — the exact swap-AB constant (KERNEL:1163-1165).
   - K-major [M,K] slice (NN/NT A): rows 128.. = 8-row atom #16 -> +16*SBO = 16*1024 = **+16384
     bytes** too (SBO=1024, KERNEL:350-357). Same constant, different derivation — don't conflate.
   - b-desc: IDENTICAL for both atoms (same B, same N, same K-step).
8. **N is capped at 256 per instruction** (ptx 31335-31337). Per-pair N=320 (tiles #1/#19) is NOT
   one MMA: use an N-composite (2nd d-tmem col region; e.g. 256+64 keeps per-CTA B slices at
   whole 128/32-elem chunks, or 2x160 which needs sw32 B slices of 80 elems). The plan lines
   19-20's "N 160/CTA sw64" is amended accordingly: 80-elem B half-slices do NOT tile with 32-elem
   sw64 chunks; 160-elem half-slices (=N_inst 320) exceed the cap.

### R4 — TMEM budget (128 lanes x 512 cols, EPI_PIPE_DEPTH=2)

Per-stage accumulator columns (per CTA): M256 atom = N cols (all lanes); M128 atom = N/2 cols (all
lanes, R1); no column sharing possible between stages/atoms (R3.6).

| target tile (per-CTA rows / instr N) | atoms         | cols/stage      | x2 stages | alloc (KERNEL:878-885 granularity) | verdict |
|---|---|---|---|---|---|
| 192 / N128            | M256+M128     | 128+64  = 192   | 384       | 512 | 2 stages ✓ |
| 192 / N160 (sw64 B*)  | M256+M128     | 160+80  = 240   | 480       | 512 | 2 stages ✓ |
| 160 / N128 (masked 192-composite, R2) | M256+M128 | 192 | 384       | 512 | 2 stages ✓ |
| 256 / N128            | M256+M256     | 128+128 = 256   | 512       | 512 | 2 stages ✓ (exactly full, zero headroom) |
| 64  / N32             | M128 only     | 16              | 32        | 32  | 2 stages ✓ |
| 64  / N64             | M128 only     | 32              | 64        | 64  | 2 stages ✓ |

Nothing needs to drop to 1 stage. (*N160: per-CTA B half-slice = 80 elems -> needs sw32 chunks
(5x16) or an N-composite per R3.8; the acc-col math is unaffected.)
The `TMEM_COLS = BN*EPI_PIPE_DEPTH` formula (KERNEL:878-885) and the matching dealloc
(KERNEL:1606-1609) must become a Config-level `ACC_COLS_PER_STAGE` = sum over atoms
(N for M256 pieces, N/2 for M128 pieces).

### R5 — swizzle parameterization (confirmed vs DeepGEMM, bf16, MN-major operand, BK=64)

Selection rule (DeepGEMM csrc/jit_kernels/heuristics/utils.hpp:13-21): mode = largest of
{128,64,32,16} bytes dividing block_size*elem_size (16 = no swizzle, interleave only); per operand
independently (sm100.hpp:162-166). Propagation:
- tmap enum: mode_into_tensor_map_swizzle (csrc/jit_kernels/impls/runtime_utils.hpp:93-110);
  inner box dim forced to swizzle_mode/elem_size (runtime_utils.hpp:119-120, and
  deep_gemm/include/deep_gemm/common/tma_utils.cuh:9-12).
- descriptor bits 61-63 (PTX Table 45, ptx 31931-31953; enum values
  cute/arch/mma_sm100_desc.hpp:79-85): NONE=0, 128B_BASE32B=1, 128B=2, 64B=4, 32B=6.
- MN-major LBO/SBO (sm100_utils.cuh:105-131): LBO = BLOCK_K * chunk_elems * elem_size =
  BK * mode_bytes; SBO = 8 * chunk_elems * elem_size = 8 * mode_bytes.
- boundary/base-offset rule: pattern restarts at 1024B/512B/256B for 128B/64B/32B modes
  (PTX Table 46, ptx 31955-31967) — keep per-stage smem slices aligned accordingly.

| mode | chunk elems (bf16) | tmap enum | desc bits 61-63 | LBO (BK=64) | SBO | per-K-row byte pitch | coord divisor | status |
|---|---|---|---|---|---|---|---|---|
| 128B | 64 | CU_TENSOR_MAP_SWIZZLE_128B | 2 | 8192 | 1024 | 128 | 64 | validated (KERNEL:350-357, 364-368; tmap KERNEL:515-548) |
| 64B  | 32 | CU_TENSOR_MAP_SWIZZLE_64B  | 4 | 4096 | 512  | 64  | 32 | validated (KERNEL:370-377, 552-570, 1169-1175) |
| 32B  | 16 | CU_TENSOR_MAP_SWIZZLE_32B  | 6 | 2048 | 256  | 32  | 16 | NEW — `make_smem_desc_sw32` = the sw64 form (KERNEL:373-377) with 6ULL<<61, lbo=BK*32, sbo=256 |

(coord divisor = chunk elems: TMA box coordinates are in whole chunks, cf. A_col =
block_row*BM/SWIZZLE_W, KERNEL:532-536, and the sw64 z = block_row*(BM/32), KERNEL:552-556.)

### R6 — minimal template-change set, ordered by unlock

1. **Composite-M knob** (unlocks 9 shapes — every "need composite-M" row):
   Config gains `M_PIECES` (e.g. {256}, {256,128}, {256,256}, {128}); the ::2 MMA branch
   (KERNEL:1186-1204) issues one mma per piece per K-step, mirroring the SWAP_AB dual-issue
   (KERNEL:1152-1185): per-piece idesc (generalize KERNEL:653-664 with an M_ATOM parameter),
   a-desc + (16384 * piece_index) (R3.7), shared b-desc, d-tmem col base advancing by the
   piece's cols/stage (R4), one commit unchanged.
2. **Layout-B interleaved epilogue read** (with #1 completes the composite tiles; ALONE unlocks
   64x64_2x2 + 128x64_2x2 = 3 shapes = STAGE T2): new epi variant for M128 pieces using the R1
   warp map — reuses `tcgen05_ld` 32x32b.x8 as-is; only row/col decode + store predication differ
   from KERNEL:1457-1554.
3. **ACC_COLS accounting** (required by #1): replace BN*EPI_PIPE_DEPTH at KERNEL:878-885 and
   KERNEL:1606-1609 with the R4 per-stage sum; static_assert <= 512.
4. **Relax BN static_assert** (KERNEL:149-151) for BN=32/64 tiles + B-slice swizzle mode by width
   (BN/2 elems: 128->sw128 as today, 32->sw64, 16->sw32) via the R5 table.
5. **160-row alias** = 192-composite + store predication (R2) — config-only once #1-#3 land.
6. **sw32 row** (make_smem_desc_sw32 + CU_TENSOR_MAP_SWIZZLE_32B tmap) — only needed if the
   96/80-wide N-composite readings (R3.8) or 64x64-with-N32 slices win benchmarks (STAGE T3).

No open items.
