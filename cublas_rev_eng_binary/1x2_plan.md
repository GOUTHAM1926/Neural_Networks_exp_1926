# Plan: cta_group::1 BF16 GEMM, 160x192 CTA tile, 1x2 cluster (sm_103a / B300)

Target: match `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` (BM=160, BN=192, BK=64, 5 stages,
1x2 cluster, horizontal = 2 CTAs along N sharing A).
Covers 3 Qwen shapes (analyzations.md:897,907,913): (M,N,K) = (4096,2048,2048),
(4096,2048,1024), (4096,2048,2048) — all TN Q-projections.

Sources searched (read-only):
- PTX ISA 9.3: `ptx_docs/ptx_isa_9.3.txt` (all line refs below are into this file unless noted)
- CUTLASS: `cutlass/include` (sm100 collective builders + `cute/atom/mma_traits_sm100.hpp`)
- DeepGEMM: `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` + epilogues
- Our template: `BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh`

---

## Q1. Legal M/N for tcgen05.mma.cta_group::1.kind::f16 — M=160 is NOT legal in one MMA

**PTX ISA 9.3, Table 44 "Various combinations of .kind and shapes"
(ptx_isa_9.3.txt:31324-31338):** for `.kind::f16`, cta_group **1**, **Dense**, dtype f32,
atype/btype f16/bf16:

- **M ∈ {64, 128} only** ("64xNxK / 128xNxK", lines 31328-31331)
- **N = {8, 16, 24, … 256} in steps of 8** (line 31328-31329) → **N=192 is legal**, N=160 is legal
- **K = 16 elements** (32 B of bf16) per MMA issue (line 31328-31329) → BK=64 = 4 MMA issues/stage

The instruction descriptor *encodes* M in bits 24-28 as `M >> 4` (5 bits, line 32017-32018)
and N in bits 17-22 as `N >> 3` (line 32014-32015), so M=160 (=10<<4) and N=192 (=24<<3)
are *encodable* — but Table 44 is the documented legality contract, and it says M=64/128 only.
**A 160-row accumulator cannot exist as one MMA**: TMEM is physically 128 lanes x 512
columns x 32b per CTA (ptx_isa_9.3.txt:31266-31267).

Weight-stationary escape hatch rejected: `.ws` allows M=32/64/128 but restricts
**N ∈ {64,128,256}** (lines 31356-31358) — N=192 not allowed, and `.ws` K-handling differs.
`disable-output-lane` (ptx_isa_9.3.txt:34813-34823, vector of 4x32b = 128 mask bits for
cta_group::1) masks *TMEM writeback lanes* of a legal-shape MMA; it does not create new M
values (and does not save flops).

**Conclusion: 160 rows = TWO MMA atoms: M=128 + M=64 (only 32 of the 64 rows useful),
or avoid the problem entirely by swapping operands (see Q6).**

*Honesty note:* nvjet is SASS-emitted; the idesc M field is a runtime register value that
ptxas cannot check, so cuBLAS *could* be issuing an undocumented M=160 (its sibling
`176x128_64x8_1x2_h`, analyzations.md:958, is equally non-decomposable-friendly). Nothing
in the three provided source trees confirms or denies HW support for M=160. A 10-minute
probe (issue idesc with M-field=10, read TMEM back) would settle it, but it is UB per PTX
and unshippable; the plan below uses only documented shapes.

### Where an M=64 accumulator lives in TMEM (critical for composing 160)

Data Path Layout Organization table (ptx_isa_9.3.txt:33617-33632):

| M | cta_group | mode | DP organization | Layout | lane alignment |
|---|---|---|---|---|---|
| 128 | ::1 | dense | 4x1 (all 128 DPs) | **D** | 0 |
| 64 | ::1 | dense (non-.ws) | 4x1, **1/2 datapath utilized** | **F** | **0 or 16** |

CUTLASS gives the exact Layout F atom (`cutlass/include/cute/atom/mma_traits_sm100.hpp:500-511`):

```
// M_MMA == 64: Half subpartitions layout atom: (M,N) -> tmem_addr
Layout tmem_atom = Layout<Shape <Shape <_16,  _4>, Int<N_MMA>>,
                          Stride<Stride< _1, _32>,      _128>>{};
```

i.e. an M=64 accumulator occupies **16 lanes in each of the 4 subpartitions**
(lanes 0-15, 32-47, 64-79, 96-111; or +16 with alignment 16) x **N full columns**.
Rows 0-15 → lanes 0-15, rows 16-31 → lanes 32-47, rows 32-47 → lanes 64-79,
rows 48-63 → lanes 96-111. Two M=64 accumulators **can share the same columns** by using
the top/bottom half-subpartitions ("Interleaved" alloc mode,
mma_traits_sm100.hpp:424-427 and the `tile_stride` logic at :505-509; PTX's "lane
alignment 0 or 16" at :33623-33624).

So for a non-swapped 160-row tile: MMA#1 M=128 → lanes 0-127; MMA#2 M=64 → tile rows
128-143 land in lanes 0-15 and rows 144-159 in lanes 32-47 (the other two 16-lane strips
hold garbage rows), **at a different TMEM column offset** — exactly the "accumulator #2 at
a column offset in DPs 0-47" shape the design question anticipated.

---

## Q2. CUTLASS/DeepGEMM never build M=160 CTA tiles on sm100

- CUTLASS 1SM tiled-MMA builder hard-asserts
  `M == 64 || M == 128` — `cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:308-309`
  (`sm100_make_1sm_trivial_tiled_mma`), and 2SM asserts `M == 128 || M == 256`
  (:374-375). N is asserted `N % 8 == 0 && N <= 256` (:313, :379) → **192 fine**.
- Grep for `Shape<_160` / `CtaShape.*160` / `_160` across `cutlass/include`: the only 160
  hits are **SM90 wgmma N=160** variants (`cute/arch/mma_sm90.hpp:438` etc.) — N, never M,
  and never sm100.
- DeepGEMM sm100 BF16: `sm100_bf16_gemm.cuh:57-66` — `LAYOUT_AD_M = 128`,
  `DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or (not kSwapAB and (BLOCK_M ==
  32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)))`. BLOCK_M ≤ 128 always. Their BLOCK_M
  = 32/64 cases still issue **M=128 MMAs** over padded A TMEM
  (`UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * ...`, :89).

**How else a 160-row (or 192-col) tile can be built** — two documented mechanisms:
1. **Multiple MMA atoms along the M dim of the same CTA tile** (2 x tcgen05.mma per
   K-step, second `[d-tmem]` at a column offset; one `tcgen05.commit` tracks both —
   commit tracks *all* prior async tcgen05 ops of the thread, ptx_isa_9.3.txt:35690-35699).
2. **Swap-AB**: compute Cᵀ so the awkward dimension becomes MMA-N (any multiple of 8).
   DeepGEMM ships this as `kSwapAB` (`sm100_bf16_gemm.cuh:59`:
   `UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N`) with a transposing epilogue
   (`epilogue/sm100_store_cd_swap_ab.cuh`).

For 160x192 specifically (see Q5/Q6): swap-AB turns it into MMA-M' = 192 = **128 + 64
exactly** (zero waste) with N' = 160 (legal), while non-swap 160 = 128 + 64 wastes 32
rows of compute.

---

## Q3. 1x2 cluster, A multicast with cta_group::1 — barrier accounting

PTX ISA 9.3, cp.async.bulk.tensor multicast semantics (ptx_isa_9.3.txt:17169-17180):

> "`.multicast::cluster` … The source data is multicast to the **same offset as dstMem**
> in the shared memory of each destination CTA. When `.cta_group` is specified as:
> **`.cta_group::1` : The mbarrier signal is multicast to the same shared memory offset
> as mbar in the destination CTA.**"

and (:17123-17129): with `.cta_group::1` the mbar named in the instruction must be in the
**same CTA as dstMem**; the even/odd **parity routing exists only for `.cta_group::2`**
(:17176-17180 — that's the path our current template documents at
Bgemm_sm103_cluster_template.cuh:271-300).

So for the 1x2 pair (ranks 0,1 along N), **confirmed accounting — no parity, fully
symmetric**:

- **Bytes**: an A-load issued by either CTA with `ctaMask = 0b11` lands at the *same smem
  offset* in both CTAs (each CTA has its own full copy of the A stage buffer).
- **complete_tx**: each destination CTA's **own** full-barrier (same smem offset as the
  `mbar` operand) is incremented by the byte count that landed *in that CTA*.
- **expect_tx**: each CTA's producer (elect_sync leader of its TMA warp) arms **its own**
  full barrier per stage with `expect_tx(A_STAGE_BYTES + B_STAGE_BYTES)` — the total that
  will land locally, regardless of who issued it. (Contrast with our cta_group::2 leader
  barrier collecting 2xA+2xB, template :300.)
- **Issue split**: CUTLASS-style — CTA rank r issues rows [r·80, (r+1)·80) of the 160-row
  A tile with mask 0b11 (halves the issue traffic per CTA), plus its own 192-col B tile
  non-multicast (mask irrelevant). A single-issuer scheme (rank 0 issues all of A) is
  equally legal; expect_tx math is unchanged.
- **Empty (consumer→producer)**: because CTA r's A-issue writes into *both* CTAs' stage
  buffers, a stage may be refilled only after **both** CTAs' MMA warps consumed it. Each
  MMA warp signals both CTAs with one instruction:
  `tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.b64 [empty_mbar], 0b11`
  — `.multicast::cluster` is legal on commit for cta_group::1
  (ptx_isa_9.3.txt:35679-35684, semantics :35711-35714, "same offset as mbar" again).
  Each per-stage empty barrier therefore needs **2 arrivals per phase**
  (vs our current parity-based scheme). Our helper `tcgen05_commit_multicast`
  (template :367-375) already takes CTA_GROUP as a template parameter.

One hard constraint to remember: **"All tcgen05 instructions within a kernel must specify
the same value for the .cta_group qualifier"** (ptx_isa_9.3.txt:32724, 34801) — the
cta_group::1 path must be a separate kernel/template instantiation end-to-end
(alloc/dealloc, mma, commit), never mixed with the cta_group::2 code path at runtime.

---

## Q4. BN=192, BK=64, 128B swizzle — descriptors, TMA boxes, tcgen05.ld

- **Smem descriptor**: format at ptx_isa_9.3.txt:31930-31971 (14-bit encoded start
  address, LBO bits 16-29, SBO bits 32-45, swizzle bits 61-63 = 2 for 128B; all three
  byte quantities must be 16B-aligned, :31968-31971). A K-major 192x64 bf16 tile under
  128B swizzle is **24 swizzle atoms of 8 rows x 128 B**; 192 is a multiple of 8, so LBO/
  SBO are identical to the 128-row case — **the tile height never appears in the
  descriptor**, it comes from idesc N (192>>3 = 24, fits the 6-bit field, :32014-32015).
  The second M-atom's descriptor is just base + row-128 byte offset (16 atoms x 1024 B =
  16 KiB; 16B-aligned ✓). No new constraint vs our working 128-col descriptors.
- **TMA box**: box dims are capped at 256 per dimension (CUTLASS enforces this shape cap
  in its TMA layout coalescing, `cute/atom/copy_traits_sm90_tma.hpp:673-704`
  "coalesce…up to size-256"; the 128B-swizzle cap applies to the *contiguous* dim = BK=64
  bf16 = 128 B exactly). Boxes of 192x64 (B), 160x64 or 80x64 (A halves) are all legal.
  (The ≤256/≤swizzle-width rules originate in the CUDA `cuTensorMapEncodeTiled` docs —
  not restated in the PTX text dump; flagged as external-doc knowledge.)
- **tcgen05.ld with a 192-col accumulator**: any warp can address **all 512 columns**;
  lanes are the restricted resource — warp w of a warpgroup touches only lanes
  [32w, 32w+32) (ptx_isa_9.3.txt:32790-32804). Column index is bits 0-15 of the TMEM
  address (:31274-31281). Shapes available: `.16x64b/.16x128b/.16x256b/.32x32b` x
  `.num up to x128` (:32850-32852, Table 54 :32903-32913). Reading 192 columns per lane =
  e.g. 6 x `32x32b.x32` or 2 x `32x32b.x64` + 1 x `.x32` (our epilogue already loops
  `32x32b.x8`, template :377-384). The 16-lane shapes (`16x256b` etc.) are what read the
  M=64 atom's half-subpartition strips — DeepGEMM does exactly this in its swap-AB
  epilogue (`sm100_store_cd_swap_ab.cuh:87-90`, `SM100_TMEM_LOAD_16dp256b1x`).
- **TMEM allocation**: units of 32 columns, power of 2, range [32,512], non-increasing
  across allocations (ptx_isa_9.3.txt:32701-32703) → just allocate **512** once
  (`alloc_tmem` already parameterized on CTA_GROUP, template :353).

---

## Q5. TMEM capacity math for 160x192 (single CTA, 128 DP x 512 cols)

TMEM per CTA: 128 lanes x 512 cols x 32b (ptx_isa_9.3.txt:31266-31267). fp32 accumulator
= 1 col per N element.

**Option A — no swap (M-composite 128+64, N=192):**
| piece | lanes | cols |
|---|---|---|
| M=128 atom (rows 0-127) | 128 (Layout D) | 192 |
| M=64 atom (rows 128-159 valid) | 4x16 strips (Layout F) | 192 |
- 1 accumulator buffer = 192+192 = **384 cols ≤ 512 ✓** (the "extra 32 rows" live in the
  half-subpartition strips of the M=64 atom at column offset 192 — lanes 0-15 = rows
  128-143, lanes 32-47 = rows 144-159).
- 2 buffers (EPI_STAGES=2): 2x192 (full-DP) + 192 (the two M=64 atoms interleaved at lane
  alignment 0/16 in shared columns) = **576 > 512 ✗**. → Option A is limited to a
  **single accumulator stage** (epilogue can't overlap the next tile's MMA).
- Compute waste: MMA does 192 rows for 160 useful → **+20% flops**.

**Option B — swap-AB (compute Cᵀ: MMA-M' = 192 = 128+64, MMA-N' = 160):**
| piece | lanes | cols |
|---|---|---|
| M'=128 atom (C-cols 0-127) | 128 | 160 |
| M'=64 atom (C-cols 128-191, all 64 rows real) | 4x16 strips | 160 |
- 1 buffer = 160+160 = 320 cols; **2 buffers = 2x160 + 160 (M'=64 pair interleaved at
  alignment 0/16) = 480 ≤ 512 ✓** — 2-stage accumulator/epilogue pipeline fits, 32 cols
  spare.
- Compute waste: **zero** (192 = 128+64 exactly; 160 is a legal N').
- This is the documented CUTLASS "Interleaved" TmemAllocMode
  (mma_traits_sm100.hpp:424-427, :500-511) applied across epi stages.

Neither CUTLASS nor DeepGEMM ever lays out an M>128 single-CTA accumulator (Q2), so the
composite layout above is our construction from the documented Layout D + Layout F atoms,
not a copied pattern. The Layout F strip mapping itself is verbatim CUTLASS
(mma_traits_sm100.hpp:503-504).

**SMEM budget** (bf16, per CTA, same for A/B either option): stage = 160x64x2 (A) +
192x64x2 (B) = 44 KiB. cuBLAS's 5 stages = 220 KiB — leaves <7 KiB for store buffers in a
227 KiB carveout, so nvjet must stream its epilogue; for us **4 stages = 176 KiB + our
existing store pipeline (STORE_PIPE_DEPTH x BM x STORE_COLS x 2, template :104) fits
comfortably**; revisit 5 stages later only if K=1024 (16 k-iters) proves latency-bound.

---

## Q6. Recommended implementation for our template

**Recommendation: build the cta_group::1 Config as swap-AB with a composite-M' MMA
(128+64), i.e. Option B**, with Option A as an acceptable fallback if the transposed
epilogue proves too invasive. Rationale: these 3 shapes are compute-bound square-ish
GEMMs (4096x2048x1024/2048); Option A's +20% tensor-core waste and single-stage
accumulator directly cost throughput, while Option B is waste-free and keeps our
2-stage epilogue overlap.

Per-K-step issue (Option B), from the single MMA warp:
```
// 4 K-steps per 64-wide stage (K=16 per issue, Table 44)
tcgen05.mma.cta_group::1.kind::f16 [tmem_c0],       bT_desc,       aT_desc, idesc_M128_N160, accum;
tcgen05.mma.cta_group::1.kind::f16 [tmem_c1], bT_desc_row128,       aT_desc, idesc_M64_N160,  accum;
...x4...
tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.b64 [empty_mbar], 0b11;
```
idesc changes vs `sm103_make_i_desc` (template :576-579): N-field = 160>>3, M-field =
128>>4 / 64>>4 (two constants instead of `CLUSTER_SIZE*BM`); transpose bits recomputed
for the swapped operand roles (TN stays two K-major descriptors).

**Carries over unchanged (≈70% of the template):**
- TMA producer warp specialization + 4D TMA loads (`tma_4d_gmem2smem`, template :256-271)
  — only the multicast wrapper flips to `CTA_GROUP=1` (already a template parameter, :273-313).
- CLC persistent scheduling (clc_try_cancel :419 + response plumbing) — works at cluster
  granularity; a 1x2 cluster is the same rank-0-queries/broadcast pattern as 2x1.
- Store pipeline (tma_store_fence/commit/wait :394-403, `tma_2d_smem2gmem` :405),
  smem-descriptor builders (`make_smem_desc` :315-337 — the 3-arg LBO/SBO variant from
  the NN fix is reused for the second atom's row-128 base).
- TMEM alloc/dealloc (:353-365, flip template arg to 1), mbarrier helpers (:175-213).

**New / rewritten:**
1. **Config**: `Sm103Cluster1x2Config<160,192,64>` — CLUSTER_SIZE=1 (no MMA pair),
   CLUSTER_N=2, BN is now *per CTA* (vs per-cluster today, template :81), drop the
   `BN >= 128 % (CLUSTER_SIZE*SWIZZLE_W)` assert (:135) for `BN % 64`; BM=160 allowed via
   new asserts (BM % 32, BM ≤ 128+64).
2. **Barrier counts**: full barrier expect_tx = A+B bytes on *every* CTA (no leader
   parity, Q3); empty barriers need 2 arrivals/phase via commit-multicast mask 0b11;
   `mbarrier_init` counts change accordingly (CLUSTER_SIZE=1 ⇒ the :121 CLUSTER_CTAS
   scaling becomes x2 for the A-shared barriers only).
3. **MMA issue loop** (:1055 region): two idesc constants, two d-tmem base addresses
   (col 0 and col 160 + lane-alignment bit for the interleaved pair), second smem
   descriptor offset (+16 KiB) — all compile-time.
4. **Epilogue** (biggest change, Option B): TMEM holds Cᵀ. One warpgroup:
   `16x256b`-family loads (16-lane strips; the M'=64 atom's valid strips are lanes
   0-15/32-47/64-79/96-111, so every warp reads its own 16-lane strip) → `stmatrix`-style
   transposed smem writes → TMA store boxes of 16(M)x192(N), exactly DeepGEMM's
   `sm100_store_cd_swap_ab.cuh` scheme (STORE_BLOCK_M=16, STORE_BLOCK_N=128 warpgroup
   read, :30-90). Our current 32x32b row-wise epilogue (:377-384, BM/32 warps, :89)
   survives only in the Option A fallback (where it needs a 5th/6th 16-lane read for rows
   128-159 at column offset 192).
5. **Dispatcher**: route the 3 (M,N,K) shapes (4096,2048,{1024,2048}) to the new kernel;
   grid = 26 x 11 = 286 CTAs = 143 x (1x2) clusters (~0.97 waves of 148 SMs at 2
   CTAs/SM... actually 1 CTA/SM at 176-220 KiB smem ⇒ 1.93 waves — matches why cuBLAS
   liked the smaller 160x192 tile here vs 128x256's 2.0 waves).

**Suggested build order:** (1) Option A skeleton first (epilogue nearly unchanged) to
validate cta_group::1 plumbing, barriers, and the composite-M TMEM addressing; (2) then
swap-AB epilogue for the perf target; (3) micro-benchmark both against
`nvjet_sm103_tst_160x192_64x5_1x2_h_bz` on the B300 before choosing (per our
verify-claims rule).

## Unconfirmed / open items (explicit)
- Whether nvjet really decomposes 160 (vs an undocumented SASS M=160 encoding) — not
  determinable from PTX/CUTLASS/DeepGEMM sources; SASS dump of the nvjet kernel would
  answer it (count tcgen05/UTCMMA issues per K-step).
- Whether nvjet uses swap-AB for these tiles ("tst"/"h"/"bz" name fields not decoded).
- TMA box-size caps cited from CUDA driver docs knowledge, not from the provided trees
  (CUTLASS's 256-coalesce is corroborating, not normative).
- Perf assumption that 2-stage accumulator overlap outweighs Option A's simplicity —
  must be measured.

## Implementation checklist (build order, Option A first)

File: NEW `src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_1x2_template.cuh`
(#include the cluster template header for primitives; do NOT modify the validated file).

A-skeleton steps:
1. Config `Sm1031x2Config<160,192,64,_STAGES=4>`: CLUSTER_SIZE=1, CLUSTER_N=2 (cluster=2 CTAs
   along N), BN per-CTA, NUM_EPI_WARPS=BM_CEIL/32 where BM_CEIL=160→5 warps, A_STAGE=160*64*2,
   B_STAGE=192*64*2, asserts per plan Q6.1.
2. i_desc x2: idesc_M128 (M-field 128>>4, N-field 192>>3), idesc_M64 (64>>4, 192>>3);
   TN transpose bits as today (both operands K-major).
3. Kernel (single file, cta_group::1 everywhere): producer = same TMA pipeline; A split-issue
   rows [r*80,(r+1)*80) mask 0b11 + own B non-multicast; every CTA arms OWN full mbar
   expect_tx(A+B) (no leader mapa!). Consumer: EVERY CTA has its own MMA warp (no pair gate):
   per K-step 2 MMAs (M128 atom → tmem col base; M64 atom → col base+192, smem A desc +16KiB
   row-128 offset via 3-arg make_smem_desc), commit-multicast mask 0b11 → both CTAs' empty
   (init count 2). tmem: single accumulator stage (Option A limit), tmem_full count 1,
   tmem_empty count NUM_EPI_WARPS (local only — accumulators are per-CTA private).
4. Epilogue: rows 0-127 = today's 32x32b path (4 warps); rows 128-159 = warp 4 reads M64-atom
   strips: lanes 0-15 (rows 128-143) + lanes 32-47 (rows 144-159) at col base+192 via
   16-lane loads (tcgen05.ld.16x256b family) — warp4's lanes 0-15/16-31 map... CAREFUL:
   32x32b from warp4 (lanes 128-159 don't exist; warps address lanes [32w,32w+32) of the
   WARPGROUP — warp 4 is in the 2nd warpgroup → lanes 0-31 again ✓ reads strips 0-15/32-47?
   NO — warp reads its own 32-lane window; need 16x-lane shapes. Resolve at impl vs
   DeepGEMM sm100_store_cd.cuh non-swap path for BLOCK_M=32/64.
5. C store: same store pipeline; grid = ceil(M/160) * ceil(N/(192*2)) * 2 CTAs; CLC identical
   (rank0 queries, 2-CTA cluster = same as validated 2x1 shape).
6. Test: new suite in the test file (RUN_1X2=1): shapes incl. 4096x2048x{1024,2048} TN (the 3
   Qwen shapes) + small sanity shapes; M%160!=0 handled by TMA OOB (4096 = 25.6 tiles → ceil).
7. THEN Option B (swap-AB) per plan Q6 with DeepGEMM sm100_store_cd_swap_ab.cuh epilogue;
   benchmark A vs B vs nvjet on B300.

---

## RESOLVED: M=64-atom epilogue read (Option A)

**Verdict up front: the checklist's 5-warp Option A epilogue (step 4) is physically
impossible, and the corrected version is complex enough that Option A should be SKIPPED.
Recommended path at the end of this section.** All line refs into
`ptx_docs/ptx_isa_9.3.txt` unless noted.

### 1. Layout F row→(lane,col) map — ALL 4 strips, rows in order (plan Q1 guess of "2 strips valid" for the *atom* was wrong; for the BM=160 *tile* it is right)

The PTX Layout F figures (Figure 215/216, ptx:33758-33768) are images absent from the
text dump, so the normative map here is the CUTLASS atom, which is exact
(`cutlass/include/cute/atom/mma_traits_sm100.hpp:500-504`):

```
// M_MMA == 64: Half subpartitions layout atom: (M,N) -> tmem_addr
Layout<Shape<Shape<_16,_4>, N>, Stride<Stride<_1,_32>, _128>>
```

Row m → DP lane `(m%16) + (m/16)*32`, i.e. the M=64 accumulator occupies **all four**
16-lane half-subpartition strips, rows strictly in order:

| atom rows | TMEM lanes | our tile (BM=160, atom at col 192) |
|---|---|---|
| 0-15  | 0-15   | tile rows 128-143 (valid) |
| 16-31 | 32-47  | tile rows 144-159 (valid) |
| 32-47 | 64-79  | garbage (A rows 160-191 don't exist) |
| 48-63 | 96-111 | garbage |

Columns: all N columns at the atom's column offset (192..383 for us). Alignment "0 or 16"
(ptx:33623-33624 Layout F row of the DP-organization table, :33617-33632) selects
top/bottom half-subpartition; Interleaved mode packs a second M=64 atom at +16
(mma_traits_sm100.hpp:424-427, tile_stride=1 at :506).

### 2. tcgen05.ld lane addressing — warp window is by warp-id MOD 4, taddr bit 20 picks the half

- ptx:32794-32804: TMEM is split into 4 chunks; the table is indexed by "**ID of the warp
  within the warpgroup**": id 0→lanes 0-31, 1→32-63, 2→64-95, 3→96-127. So **warp w
  accesses lanes [32*(w%4), 32*(w%4)+32) — warp 4 (warpgroup 1, id 0) sees lanes 0-31
  again, and can NEVER touch lanes 32-47.**
- taddr = lane index in bits 31-16, column in bits 15-0 (ptx:31270-31281).
- 16-lane shapes (`.16x64b/.16x128b/.16x256b`, ptx:31551,31593,31607: "will access 16
  lanes") read lanes [base, base+16) where base is the taddr lane field; the only legal
  bases inside a warp's window are +0 and +16 — DeepGEMM selects the top half with
  `tmem_addr | 0x00100000` (= lane+16, `sm100_store_cd_swap_ab.cuh:90-92`). The
  warp-index bits of the lane field are **ignored by hardware**
  (`sm100_bf16_gemm.cuh:373-374`: "no need for tmem_ptr |= (epilogue_warp_idx*32)<<16");
  CUTLASS debug-asserts they match the executing warp anyway
  (`cute/atom/copy_traits_sm100.hpp:445`).
- `.16x256b.x1` fragment map (Copy_Traits DstLayout, copy_traits_sm100.hpp:467-481:
  `Shape<Shape<_4,_8>,Shape<_64,_2>>, Stride<Stride<_64,_256>,Stride<_1,_2048>>`):
  thread t holds lanes base+(t/4) and base+(t/4)+8; regs {r0,r1} = f32 cols
  2*(t%4), 2*(t%4)+1 of the first lane, {r2,r3} = same cols of lane+8. **Two rows per
  thread, 2-col chunks — NOT the thread==row map of `.32x32b`**, so the tail rows need a
  brand-new register→smem store indexing.

### 3. DeepGEMM never reads a Layout F accumulator in its non-swap epilogue — nothing to copy

`sm100_bf16_gemm.cuh:57-58`: `LAYOUT_AD_M = 128; UMMA_M = LAYOUT_AD_M * kNumMulticast` —
the MMA is **always M=128 (Layout D)**, even for BLOCK_M=32/64; A is padded to 128 rows
in smem (`UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M)*...`, :89,
allowed to bleed into the B-stage region, :90). The epilogue then stores only the first
BLOCK_M lanes: `STORE_BLOCK_M = min(BLOCK_M,128)`, `kNumUMMAStoreThreads = STORE_BLOCK_M`
(:74-76) → BLOCK_M/32 warps, each reading its own 32-lane window with
`SM100_TMEM_LOAD_32dp32b{4,8}x` only (`sm100_store_cd.cuh:89,96`). The **only** 16-lane
reader in all of DeepGEMM's epilogues is the swap-AB path
(`sm100_store_cd_swap_ab.cuh:88-92`), a full warpgroup where every warp does the
identical pair of `16dp256b1x` loads (halves 0 and 16) + `STSM_T` transposed store (:102),
with `STORE_BLOCK_N == 128` asserted (:32). Conclusion: for awkward M they either pad to
M=128 (cheap when BLOCK_M<128 wastes only *unused* datapaths, not flops — N is the
per-col cost) or swap. **No producer of tcgen05 code ships a non-swap Layout F read.**

### 4. Corrected Option A epilogue (if it were built) — and why not to build it

Checklist step 1/step 4 are wrong as written: NUM_EPI_WARPS=5 with warp 4 reading strips
0-15 *and* 32-47 is impossible (warp 4's window is lanes 0-31 only, and one 16-lane load
cannot span lanes 0-15 + 32-47 since they sit in different windows). The minimal correct
scheme is **4 epi warps, two passes**:

| warp | pass 1: M128 atom, cols 0-191 | pass 2: M64 atom, cols 192-383 |
|---|---|---|
| 0 | `.32x32b` lanes 0-31 = rows 0-31 | `.16x256b` half 0 → lanes 0-15 = rows 128-143 |
| 1 | `.32x32b` lanes 32-63 = rows 32-63 | `.16x256b` half 0 → lanes 32-47 = rows 144-159 |
| 2 | `.32x32b` lanes 64-95 = rows 64-95 | idle |
| 3 | `.32x32b` lanes 96-127 = rows 96-127 | idle |

Accounting: EPI bar.sync = 128 threads (4 warps, not 160); tmem_empty arrive count = 4
warp-arrivals (or 128 thread-arrivals per our template's scheme); C TMA box 160 rows is
legal (≤256 per dim). Pass 2 needs a second store-smem indexer (thread t → rows
128/144+(t/4) and +8, col pairs 2*(t%4)+8k per `.x(k)` repeat — from the fragment map in
#2), i.e. a fragment layout that exists nowhere in our template or DeepGEMM.

Cost summary of Option A: (i) bespoke second fragment→smem path, (ii) asymmetric warp
duty + warp 0/1 doing 1.25x the TMEM-load work of warps 2/3, (iii) single accumulator
stage — no MMA/epilogue overlap (plan Q5: 2 stages need 576>512 cols), (iv) +20% flops.
Its sole advertised virtue — "epilogue nearly unchanged" — is false per (i)+(ii).

### DECISION — skip Option A; validate plumbing on a BM=128 bring-up config, then build Option B directly

1. **Bring-up**: `Sm1031x2Config<128,192,64>` — single M=128 MMA (Layout D), today's
   4-warp `.32x32b` epilogue byte-for-byte, exercising 100% of the new cta_group::1
   machinery (1x2 multicast + expect_tx both-CTAs, commit-multicast mask 0b11 empty
   barriers, CLC, idesc N=192, single-CTA TMEM alloc). This replaces the "Option A
   skeleton" — same validation value, zero epilogue work.
2. **Deliverable**: Option B (swap-AB 160x192: MMA M'=128+64, N'=160) with the
   DeepGEMM swap-AB epilogue scheme (`sm100_store_cd_swap_ab.cuh:51-141`): full
   warpgroup, `16dp256b1x` pair per warp for the M'=128 atom; for the M'=64 atom each
   warp issues the **single** half-0 load (its strip: warp w%4 → atom rows 16*(w%4)..+16)
   — or half-1 (`|0x00100000`) for the interleaved second epi-stage buffer — then the
   same `STSM_T` store. 2-stage accumulator fits (480≤512, plan Q5). Zero flop waste.
3. Benchmark Option B vs `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` on the B300 (bring-up
   config is a plumbing artifact, not a perf claimant). No open items: every addressing
   fact needed for both steps is pinned above with sources.

## CRITICAL GAP found during Option B implementation start (2026-07-04, pre-code)

The 160-dim slicing conflicts with 128B swizzle chunking IN BOTH OPTIONS — the plan's Q4
"boxes of 160x64 legal" missed that block OFFSETS must be chunk-aligned:
- TN operand [K,X] uses tmap {64, K, X/64} (128B swizzle = 64-elem chunks); slicing X in
  160-wide blocks → z-coord = block*160/64 = NON-INTEGER for odd blocks. Same for 80-wide
  issue-split halves. So neither Option A (A-operand M=160 slices) nor Option B (B'-operand
  N'=160 slices) can use our current 128B-swizzle tmaps.
- HYPOTHESIS to verify FIRST next session: the 160-sliced operand must use 64B swizzle
  (SWIZZLE_64B = 32-elem chunks; 160 = 5x32 ✓): tmap {32, K, X/32}, smem descriptor swizzle
  field = 64B mode, SBO/LBO recomputed for 32-elem chunks, MMA desc supports 64B-swizzled
  operands (desc bits 61-63). Alternatively swizzle-NONE for that operand. Verify against
  CUTLASS sm100 smem layout selectors (they pick swizzle per operand extent) + PTX desc
  encoding table, THEN implement. Do NOT code Option B before resolving this.
- Also re-check: does 64B swizzle change the TN a_off/b_off chunk-major math (SWIZZLE_W=32
  for that operand) — yes, all SWIZZLE_W-derived offsets for the swapped operand must use 32.

---

## RESOLVED: 160-dim slicing & swizzle (Option B final spec)

**Verdict up front: the 64B-swizzle hypothesis is CONFIRMED and is the chosen strategy.
The 160-sliced operand (swapped B' = original A[K,M]) gets its own 64B-swizzled tmap +
64B-mode smem descriptors (LBO=4096, SBO=512); the 192-sliced operand (swapped A' =
original B[K,N]) keeps 128B swizzle unchanged. Mixed swizzle modes across the two
operands of one tcgen05.mma and across two tmaps in one kernel are both fully legal.
The 128B+padding alternative is REJECTED — it does not even load (proof in §R6). One
consequential correction to the earlier plan: the multicast issue-split of the shared
operand can no longer be an 80-row M-split (80 is not chunk-aligned); use a single-issuer
full box (§R7a). Zero open items.** Line refs: `ptx:` = ptx_docs/ptx_isa_9.3.txt,
`template :` = Bgemm_sm103_cluster_template.cuh, cutlass/DeepGEMM paths as written.

### R0. Terminology fix first (the CRITICAL GAP called these "K-major"; they are MN-major)

PTX (ptx:33810-33812, tcgen05 major-ness table): transposed A = **M-major**, transposed
B = N-major; "we will use MN-Major and K-Major throughout". Our TN stores A as
[batch,K,M] row-major (template :491-503) — the contiguous gmem dim is **M**, so the A
operand is **MN-major**, and likewise B[K,N] is MN-major. The 160-sliced dim is the
operand's *MN/contiguous* dim. This matters because swizzle selection keys on the
contiguous-dim extent — for MN-major that is exactly the 160/192 tile width.

### R1. How CUTLASS picks the swizzle (Q1) — exact rule

`sm100_smem_selector` (cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:82):
requires extent % 8 == 0 (:86-87), then picks the **largest swizzle atom that divides the
contiguous-dim extent**:
- MN-major: `BLK_MN0 % 64 → SW128; % 32 → SW64; % 16 → SW32; % 8 → INTER` (:89-108) —
  atom MN-spans for bf16 come from Layout_MN_SW{128,64,32}_Atom_Bits = 1024/512/256 bits
  (cute/atom/mma_traits_sm90_gmma.hpp:76-78, upcast :90-94) = 64/32/16 elements.
- K-major: same ladder on `BLK_K0` (:115-127) — irrelevant here (our operands are MN-major).

**For our 160-wide bf16 MN-major slice: 160 % 64 ≠ 0, 160 % 32 = 0 → Layout_MN_SW64_Atom
= 64B swizzle.** For the 192-wide slice: 192 % 64 = 0 → SW128 stays. CUTLASS builders
call the selector **separately per operand** (e.g.
sm100_mixed_tma_cpasync_umma_builder.inl:113 for A, :124 for B), so per-operand swizzle
is the normal CUTLASS situation, not an exotic one.

DeepGEMM has the identical rule at runtime: `get_swizzle_mode` returns the largest of
{128,64,32,16} dividing block-width-bytes (csrc/jit_kernels/heuristics/utils.hpp:13-20),
applied per operand to the *inner* (contiguous) dim ("Decide swizzling by the inner dim",
csrc/jit_kernels/heuristics/sm100.hpp:160-166): 160×2B = 320B → **64**. So both reference
stacks pick 64B swizzle for a 160-wide bf16 MN-major block. (DeepGEMM's sm100 heuristic
additionally *discards* candidate layouts whose swizzle ≠ 128 when either operand is
K-major, sm100.hpp:132-137 — an impl simplification tied to their
`kSwizzleMode == BLOCK_K*sizeof` K-major assert, common/sm100_utils.cuh:104, not a HW
rule, and it does not apply to us: both our operands are MN-major.)

### R2. PTX smem-descriptor encodings (Q2)

Table 45 (ptx:31936-31952): bits 61-63 swizzle mode = **0 none, 1 = 128B with 32B atomic
swizzling, 2 = 128B, 4 = 64B, 6 = 32B** (3/5/7 invalid) — same enum in CUTLASS
(cute/arch/mma_sm100_desc.hpp:79-85, `SWIZZLE_64B = 4`). Start addr, LBO, SBO all 16B
aligned and encoded `(x & 0x3FFFF) >> 4` (ptx:31939-31945, :31955, :31964-31970). Base
offset bits 49-51 = 0 iff the swizzle pattern starts on its natural boundary — 1024B for
128B swizzle, **512B for 64B swizzle**, 256B for 32B (Table 46, ptx:31959-31963);
otherwise `(pattern start addr >> 7) & 7` (ptx:31964-31965; cf. ptx:30083-30086).

LBO/SBO meaning for **MN-major swizzled** operands (tcgen05 9.7.17.3.1.1, ptx:31666-31690;
9.7.17.3.2, ptx:31712-31727; canonical layout `((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))` with
Swizzle<2,4,3> for 64B, ptx:31749-31760, 64B row :31759):
- **LBO** = byte stride from one (swizzle-span) group of MN rows to the next = our
  *chunk pitch*;
- **SBO** = byte stride from one group of 8 K-columns to the next.

For our chunk-major TMA fill (chunk pitch = BK rows × span bytes; 8-K-row group = 8 ×
span bytes), with BK=64 bf16:

| mode | desc bits 61-63 | chunk width | LBO | SBO | per-K-row pitch |
|---|---|---|---|---|---|
| 128B | 2 | 64 elem | BK·128 = **8192** | 8·128 = **1024** | 128 B |
| 64B  | 4 | 32 elem | BK·64  = **4096** | 8·64  = **512**  | 64 B  |
| 32B  | 6 | 16 elem | BK·32  = **2048** | 8·32  = **256**  | 32 B  |

The 128B row is byte-for-byte our working TN descriptor (template :345-349, :1071-1072,
:1080-1081) — validation of the derivation. The general formulas are also verbatim
DeepGEMM: MN-major `stride_byte_offset = 8*ATOM*sizeof`, `leading_byte_offset =
BLOCK_K*ATOM*sizeof`, ATOM = swizzle-bytes/elt-size (common/sm100_utils.cuh:112-131,
common/tma_utils.cuh:10-12), and desc-address K-walk stride = ATOM elements per K element
(sm100_utils.cuh:81-90) → 16-K-step pitch = 16·64B = 1024B under 64B swizzle.

CU_TENSOR_MAP_SWIZZLE_* ↔ descriptor field (cuda.h:3642-3648 vs Table 45): NONE=0↔0,
32B=1↔6, 64B=2↔**4**, 128B=3↔2 (mapping code: cute/arch/copy_sm90_desc.hpp:240,
mma_traits_sm100.hpp:145-148). The two enums are unrelated numerically — never mix them.

### R3. tcgen05.mma operand legality under 64B swizzle (Q3) and mixed modes (Q5)

- Table 60 "Layout for swizzle atoms" (ptx:33827-33843) lists tcgen05 swizzle atoms for
  **every** mode × major-ness; 64B MN-major atom = 4×8 in 128b units = **32×8 bf16**. No
  MMA-kind column exists — modes are kind-agnostic.
- The only mode restriction anywhere in 9.7.17 is Table 56's transpose rule
  (ptx:33438-33453): for kind::f16, Transpose A/B is supported, and the *only* exclusion
  is 128B-swizzle-with-32B-atomicity (mode 1). Plain 64B (mode 4), 32B and none are fully
  legal with the transpose bits we use.
- Mixed modes across the two operands: each operand has its own 64-bit descriptor with
  its own bits 61-63 (Table 45 is per-descriptor); no clause ties A's mode to B's.
  DeepGEMM ships kernels with independent `swizzle_a_mode`/`swizzle_b_mode` template
  parameters and tmaps (csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:58, :101-106).
  **A'=128B + B'=64B in one MMA is legal.**

### R4. Two tmaps with different swizzle in one kernel (Q6)

Swizzle is a per-CUtensorMap property passed to each `cuTensorMapEncodeTiled` call
(our builders already take it as a parameter, template :475, :518). Constraints from the
driver docs (cuda.h:23302, :23322-23325): boxDim[0]·eltsize must be a multiple of 16B,
and with swizzle enabled **boxDim[0]·eltsize ≤ swizzle size** — 64B swizzle → inner box
≤ 64B = **32 bf16 elements**. Nothing constrains two tmaps of one kernel to share a mode
(DeepGEMM's A/B tmaps above are the existence proof). Our 4D-chunked shape generalizes
as: 128B → {64, ·, X/64}; 64B → {32, ·, X/32}, dim-2 global stride 64B (multiple of 16 ✓).

### R5. Sanity of the 192 operand (Q5)

A' (original B[K,N], per-CTA 192-wide): 192 % 64 = 0 → SW128 by the R1 rule; tmap
{64, K, N/64} box {64, BK, 192/64=3}, z = block_col·6 + rank·3 — integer ✓. Descriptor =
today's 3-arg form (LBO 8192, SBO 1024, mode 2). Second M-atom (M'=192 = 128+64): rows
128-191 = chunk 2 → base + 2·LBO = **+16384** (16B-aligned ✓, same value as plan Q4's
"16 KiB"). Nothing else changes for this operand.

### R6. The 128B-swizzle padding alternative (Q7) — REJECTED, it cannot even load

Pad-to-192-in-smem does not fix the *gmem* problem: block b's slice starts at m = b·160,
and b·160 mod 64 = 32 for every odd b. A 64-elem-chunk tmap can only address multiples
of 64 in the chunked dim, and re-encoding cannot help: with 128B swizzle the box inner
dim is capped at 64 elements (cuda.h:23322-23325), so the chunk factorization — and the
64-element offset granularity — is forced. Absorbing the 32-element (64B) misalignment
on the smem side is also impossible: the descriptor's base-offset field re-phases the
swizzle only in whole 128B rows (`(pattern start addr >> 7) & 7`, ptx:31964-31965;
TMA-side table ptx:3835-3838), and MN-major mode-2 descriptors walk whole 64-element
lines (canonical layout ptx:31749-31760) — there is no "start 32 elements into a chunk"
encoding. The remaining 128B-preserving escapes are padding the *gmem* layout (not
allowed — we are a BLAS API on user buffers) or loading a 64-aligned 192-wide window per
block with parity-dependent descriptor origins (+20% A-side TMA traffic, garbage-column
reads, two code paths). All strictly worse than 64B swizzle, whose only cost is a
half-width TMA/MMA line for one operand. **PICKED: 64B swizzle.** (Swizzle-NONE would
also make offsets legal but forfeits bank-conflict avoidance and is never chosen by
either reference stack when a larger atom fits, sm100_common.inl:97-108,
heuristics/utils.hpp:16.)

### R7. FINAL Option B spec — `Sm1031x2Config<160,192,64>` (supersedes the CRITICAL GAP; everything else from the earlier RESOLVED section stands)

Operand roles after swap-AB: **MMA-A' = original B[K,N]** (192/CTA slices, 128B swizzle),
**MMA-B' = original A[K,M]** (160 slices, 64B swizzle, cluster-shared → multicast).

**(a) tmap B' (original A, NEW 64B-swizzle builder `init_4d_tma_A_KM_sw64`):**
`globalDim {32, K, M/32, batch}`, `globalStrides {M·2, 64, batchStride·2}`,
`boxDim {32, BK=64, 160/32=5, 1}`, swizzle = `CU_TENSOR_MAP_SWIZZLE_64B`.
Precondition M % 32 == 0 (relaxed from %64; 4096 ✓). Per stage k of block_row b:
coords `(0, k·64, b·5, batch)` — z = b·5 is always integer, gap closed. **Single-issuer:**
the pair's rank-0 producer issues the one 20480B box with multicast mask 0b11 (same box
byte-count as today's single A issue; blessed as equally legal at plan Q3 :139-140). The
old "rank r issues rows [r·80,(r+1)·80)" split is DEAD — 80/32 = 2.5 chunks. (If issue
balance ever matters, the legal even split is per-chunk: boxDim {32,64,1}, rank0 chunks
0-2 / rank1 chunks 3-4, dst += chunk·4096 — DeepGEMM's per-atom issue pattern,
common/tma_utils.cuh:25-33. Do NOT K-split: two {32,32,5} boxes give chunk pitch 2048,
breaking LBO=4096.) M=4096 tail: block 25 spans chunks 125-129 vs globalDim[2]=128 →
chunks 128-129 OOB-zero-filled by TMA ✓. expect_tx unchanged: every CTA arms its own full
barrier with the full 45056B/stage (per plan Q3).

**(b) tmap A' (original B): unchanged machinery** — `init_4d_tma_B_KN` (template
:515-534) with per-CTA slice 192: box {64, 64, 3}, 128B swizzle, z = block_col·6 +
rank·3, non-multicast, issued by each CTA's own producer.

**(c) smem per stage:** `[A' 24576 B | B' 20480 B]` = 45056 B; ×4 stages = 176 KiB.
**B' stage bytes unchanged** (160·64·2 = 5 chunks × 4096). Alignment: smem base is
`__align__(1024)` (template :727); 45056 = 44·1024 → every stage 1024-aligned → A' on a
1024B pattern boundary and B' at +24576 = 48·512 on a 512B boundary → **base-offset
field = 0 for both descriptors** (Table 46, ptx:31959-31963).

**(d) consumer walk (replaces the SWIZZLE_W-uniform math of template :1022-1055):**
loop bounds are UNCHANGED — `k1 < BK/64 (=1)`, `k2 < 64/MMA_K_STEP (=4)` — they walk K,
not chunks; the question's "BK/32?" is answered NO. Only the per-operand byte pitch
differs. With k_row = k1·64 + k2·16:
```
a_off = k_row * 128;                       // A' 128B-swizzled: k2*2048
b_off = k_row * 64;                        // B' 64B-swizzled:  k2*1024
a_desc  = make_smem_desc     (A_smem + a_off,          8192, 1024);  // mode 2
a_desc2 = make_smem_desc     (A_smem + a_off + 16384,  8192, 1024);  // M64 atom, chunk 2
b_desc  = make_smem_desc_sw64(B_smem + b_off,          4096,  512);  // mode 4, both MMAs
```
`make_smem_desc_sw64` = template :345-349 with `(4ULL << 61)` in place of `(2ULL << 61)`.
Keep per-operand constants (`SW_SPAN_Ap = 128`, `SW_SPAN_Bp = 64` bytes) instead of the
global SWIZZLE_W (template :63); drop the `BN % (CLUSTER_SIZE*SWIZZLE_W)` assert
(template :142) in favor of the asserts in (f).

**(e) idesc:** two compile-time constants, both with trans_a = trans_b = 1 (bits 15,16 —
both operands MN-major, exactly today's TN setting, template :598-599; legal for f16
under 64B swizzle per Table 56, ptx:33438-33453): N-field = 160>>3 = 20, M-field =
128>>4 = 8 for MMA#1 and 64>>4 = 4 for MMA#2 (fields per ptx:32014-32018). N=160 legal
(Table 44, ptx:31324-31338).

**(f) config asserts:** M % 32 == 0; B'-slice (160) % 32 == 0; A'-slice (192) % 64 == 0;
BK % 64 == 0. Everything not named here — cta_group::1 barriers, CLC, TMEM layout
(2×160-col buffers + interleaved M64 pair, 480 ≤ 512), DeepGEMM-style swap-AB epilogue,
BM'=128 bring-up config first (both operands 128B, no 64B code exercised), then this
spec, then benchmark vs `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` — is exactly as pinned
in the "DECISION" block of the previous RESOLVED section and in plan Q3-Q6.

**No open items.** Every formula above is triple-sourced (PTX normative text, DeepGEMM
shipping code, and — for the 128B column — our own validated kernel), and the one
plan-level casualty (the 80-row issue split) has a specified replacement.

## P5 BUG ANALYSIS: exact TMEM fragment maps

**Verdict up front: every TMEM/MMA map in the SWAP_AB path is CORRECT. The bug is in the
store-buffer write: both SWAP_AB epilogue writes (template :1308 and :1334) are LINEAR,
but `C_tmap` is built with `CU_TENSOR_MAP_SWIZZLE_128B` (template :1625-1626), so the TMA
store expects the 128B-swizzled smem arrangement. The validated non-swap epilogue applies
exactly that XOR (`swizzled_n = n ^ (row & 7)`, template :1487); the swap path dropped it.**

### 1. tcgen05.ld.32x32b.x8 fragment — our map CORRECT
CUTLASS `Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>` (copy_traits_sm100.hpp:1557-1570):
`ValID = ((_256,_32)):((_1,DP_b))`, `DstLayout = (_32,_256):(_256,_1)` → logical bit
`= 256*t + 32*i + b` decodes to lane `t`, col `i` (i = register 0..7). I.e. **thread t,
register i = (lane t, col addr+i); repetitions are 8 contiguous 32-bit columns, no
interleave.** PTX only states "accesses 32 lanes, 32*.num bits per lane"
(ptx_isa_9.3.txt:31537-31538, Figure 183 is an image). Our epilogue assumes exactly this
(template :1297-1308) and P4 (non-swap, 42/42) validates the same primitive at :1442.
DeepGEMM's swap-AB float path also uses it as (lane, col+row) (sm100_store_cd_swap_ab.cuh:74-84).

### 2. tcgen05.ld.16x256b.x1 fragment — our map CORRECT
CUTLASS `Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>` (copy_traits_sm100.hpp:464-482):
`DstLayout = ((_4,_8),(_64,_2)):((_64,_256),(_1,_2048))` over
`ValID = (_256,_16):(_1,DP_b)`. Decode: bit = 64*(t%4) + 256*(t/4) + v0 + 2048*v1 →
r0 = (lane t/4, col 2*(t%4)), r1 = (lane t/4, col 2*(t%4)+1),
r2 = (lane t/4+8, col 2*(t%4)), r3 = (lane t/4+8, col 2*(t%4)+1).
**Lane base+t/4 (one lane per 4 threads), NOT 2 apart** — exactly our comment/usage
(template :407-417, :1314-1335). PTX: "accesses 16 lanes … 256*.num bits"
(ptx_isa_9.3.txt:31607-31608). The 16-lane offset within the warp's subpartition is
selected by address lane-field bits — DeepGEMM selects the upper half with
`tmem_addr | 0x00100000` (sm100_store_cd_swap_ab.cuh:88-92), confirming the encoding our
`(warp*32 + 16*wave_stage) << 16` uses (template :1278-1279; CUTLASS asserts the
subpartition bits must match the warp, copy_traits_sm100.hpp:443-447).

### 3. Layout F (M=64, cta_group::1, non-.ws) — our strips CORRECT
CUTLASS 1SM M_MMA=64 atom (mma_traits_sm100.hpp:500-511):
`tmem_atom = ((_16,_4),N):((_1,_32),_128)` → M-rows 0-15→lanes 0-15, 16-31→32-47,
32-47→64-79, 48-63→96-111; N-col n at col base+n. PTX Data Path Layout Organization
table (ptx_isa_9.3.txt:33617-33635): M=64/::1/non-.ws = Layout F, "4x1 (1/2 datapath
utilized)", **TMEM lane alignment 0 or 16** — so the D-tmem lane offset `(16*wave_stage)<<16`
(template :1131) is legal and encoded via the address lane field. The alignment-sharing
restriction (ptx:34606-34610) binds A/D/metadata only when A is in TMEM — ours is smem.
Epilogue strip math (template :1316-1318): warp w at lane field 32w+16*wave reads its
16-lane region = atom rows 16w..16w+15 → C-cols 128+16w+t/4(+8) ✓.

### 4. Layout D (M=128, cta_group::1) — our map CORRECT
CUTLASS 1SM M_MMA=128 atom (mma_traits_sm100.hpp:513-523): `(_128,N):(_1,_128)` →
row r → lane r, one 32-bit col per N element for f32 D. So C'(row', n') sits at
(lane row', col base+n'); with D base = tmem+wave*BM (template :1127), C-col n_local =
32*warp+lane and C-row m = m0+i (template :1280, :1304-1308) ✓. DeepGEMM swap-AB uses the
identical map: lanes = n (STORE_BLOCK_N=128 "must match TMEM rows",
sm100_store_cd_swap_ab.cuh:31-32), cols = m (:60-62).

### 5. MMA operand walk under swap — CORRECT
PTX canonical layouts (ptx_isa_9.3.txt:31749-31765): MN-major 128B swizzle =
`Swizzle<3,4,3> o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))`, T=8 for bf16 → one swizzle
chunk spans 64 MN positions × 8 K rows; the m-th 64-wide MN block advances by LBO, 8 K
rows advance by SBO (LBO/SBO defs ptx:31696-31701, :31740-31744). With LBO=8192,
SBO=1024 (template :1116-1122): M128 atom walks chunks 0-1 → B-slot columns 0-127 =
M'-rows 0-127 ✓; `+16384` = chunk 2 → columns 128-191 = M64-atom rows 0-63 ✓. K offset
k2*16 rows = k2*2048 B stays swizzle-pattern aligned ✓. Instruction descriptor fields
(Table 47, ptx:31989-32022): bit15 transA→M-major A', bit16 transB→N-major B'
(ptx:33801-33804), N-field bits17-22 = BM>>3 = 16, M-field bits24-28 = 8/4 — our
`sm103_make_i_desc_swap` (template :631-635) matches exactly. The same descriptor form
drove the 192-wide slot as MMA-B in P4 (42/42), and A/B canonical walks are symmetric.

### 6. THE BUG: missing 128B-swizzle XOR in the store-buffer writes
`C_tmap`: box [BM × STORE_COLS=64] bf16 = 128 B/row with `CU_TENSOR_MAP_SWIZZLE_128B`
(template :1625-1626). TMA reads logical 16-byte group g of smem row r from position
`g ^ (r & 7)`. Producers must therefore write group g at `g ^ (r & 7)`:
- non-swap epilogue DOES (template :1486-1489, validated 42/42);
- DeepGEMM swap-AB DOES (`(col ^ row) * kNumBankGroupBytes`,
  sm100_store_cd_swap_ab.cuh:80-82 and :97-99, with the 128B-swizzle assert at :38);
- **our swap-AB epilogue does NOT** — template :1308 (`buf[(m0+i)*STORE_COLS +
  (n128-n_base)]`) and :1334 (`buf[m*STORE_COLS + (n_loc-n_base)]`) are linear.

Every output row m with (m & 7) != 0 gets its eight 16-byte groups permuted within the
64-col store chunk → deterministic same-row wrong values of GEMM magnitude. Evidence fits
bit-exactly:
- Row 0 (m&7=0): XOR is identity → row correct ("first-10 of row 0 CORRECT").
- (113,100): m&7=1; chunk1 offset 36 → g=4,e=4; reads g 4^1=5 → out = ref(113,108).
- (142,544): m=14 (block_row 1), n_local=160 → chunk2 (M64 region) offset 32 → g=4,e=0;
  reads g 4^6=2 → out = ref(142,528). Both regions fail identically because the bug is
  downstream of the (correct) TMEM reads.

### Fix (only change needed)
M128 block, replace template :1308:
```cuda
const int off = n128 - n_base;                         // [0,64)
const int r   = m0 + i;
buf[r * STORE_COLS + ((((off >> 3) ^ (r & 7)) << 3) | (off & 7))] =
    __float2bfloat16(v);
```
M64 block, replace template :1334:
```cuda
const int off = n_loc - n_base;
buf[m * STORE_COLS + ((((off >> 3) ^ (m & 7)) << 3) | (off & 7))] =
    __float2bfloat16(v);
```
(identical formula to the validated non-swap `n ^ (row & 7)` at template :1487, with the
group index recomputed because the swap path addresses single elements, not int4 groups).
All TMEM fragment maps, Layout D/F placements, lane-alignment encoding, smem descriptors
and idesc fields verified correct from primary sources — leave them untouched.
