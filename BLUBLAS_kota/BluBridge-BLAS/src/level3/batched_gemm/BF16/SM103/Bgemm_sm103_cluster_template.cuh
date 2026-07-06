#pragma once
// Bgemm_sm103_cluster_template.cuh
// Unified SM103a (B300 / Blackwell Ultra) BF16 batched GEMM kernel template.
//
// Usage — include this header and call the launcher:
//
//   using Cfg = Sm103ClusterConfig<128, 128, 64>;   // BM, BN, BK
//   launch_bgemm_sm103_cluster<Cfg, Sm103Layout::NN>(A, B, C, M, N, K, batch,
//   alpha, beta, stream);
//
// Architecture overview (identical for all three layouts):
//   - tcgen05.mma.cta_group::2.kind::f16   (2-CTA cooperative MMA — SM103a /
//   B300)
//   - CLC (Cluster Launch Control) persistent tile scheduling
//   - On-device tile coords: block_col = mn_tile % grid_n  (no GPU-side
//   tile_map)
//   - Warp layout:  [0 .. NUM_EPI_WARPS)  epilogue
//                   [NUM_EPI_WARPS]       producer
//                   [NUM_EPI_WARPS+1]     consumer  (drives MMA from CTA0)
//                   [NUM_EPI_WARPS+2]     scheduler (CLC loop)
//
// Per-layout differences (controlled by Sm103Layout template tag):
//   NN  A[batch,M,K] × B[batch,K,N]      trans_b=1 in i_desc
//   NT  A[batch,M,K] × B[batch,N,K]ᵀ    no trans bits (B stored [N,K])
//   TN  A[batch,K,M]ᵀ × B[batch,K,N]   trans_a=1 + trans_b=1 in i_desc

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ─── layout tag ──────────────────────────────────────────────────────────────

enum class Sm103Layout { NN, NT, TN };

// ─── config struct
// ────────────────────────────────────────────────────────────
//
// Maps directly to ThunderKittens' `config` struct pattern.
//
// Template parameters:
//   _BM             — M-tile rows per CTA (must be multiple of 32; cluster
//   covers 2×BM) _BN             — N-tile cols per cluster (must be multiple of
//   128 = 2×SWIZZLE_W) _BK             — K-depth per pipeline stage (must be
//   multiple of 64 = SWIZZLE_W) _LOAD_PIPE_DEPTH — smem A/B pipeline depth (=
//   QUEUE_SIZE in original code) _EPI_PIPE_DEPTH  — TMEM double-buffering slots
//   (= NUM_EPI_STAGES, default 2) _CLC_PIPE_DEPTH  — CLC pipeline depth (=
//   NUM_CLC_STAGES, default 2)

template <int _BM, int _BN, int _BK,
          int _CLUSTER_N = 1, // CTAs along N (1 = 2x1 cluster, 2 = 2x2, ...)
          int _LOAD_PIPE_DEPTH = 0, // 0 = auto-derive from smem budget
          int _EPI_PIPE_DEPTH = 2, int _CLC_PIPE_DEPTH = 2,
          int _CLUSTER_SIZE = 2, // CTAs per MMA group: 2 = cta_group::2 pair,
                                 // 1 = cta_group::1 (single-CTA MMA, DeepGEMM
                                 // kNumMulticast-style knob; 1x2_plan.md)
          bool _SWAP_AB = false> // swap-AB: compute C^T so BN becomes MMA-M'
                                 // (192 = 128+64 composite atoms) and BM
                                 // becomes MMA-N'. TN only. 1x2_plan.md R7.
struct Sm103ClusterConfig {

  // ── hardware-fixed SM103a constants ──────────────────────────────────────
  static constexpr int WARP_THREADS = 32;
  static constexpr int SWIZZLE_W = 64; // 64 bf16 = 128 bytes, one swizzle group
  static constexpr int MMA_K_STEP = 16; // inner-K step of one tcgen05 MMA call
  static constexpr int CLUSTER_SIZE =
      _CLUSTER_SIZE; // CTAs per MMA group (= the
                     // tcgen05 cta_group value everywhere)
  // ── cluster shape (parameterized) ────────────────────────────────────────
  // CLUSTER_SIZE = CTAs along M = the cta_group::2 pair (always 2; the PTX pair
  //   is the last %cluster_ctarank bit, §9.7.17.5.1).
  // CLUSTER_N    = CTAs along N (pairs). 1 → 2x1 cluster; 2 → 2x2; 4 → 2x4.
  // CLUSTER_CTAS = total CTAs launched per cluster = CLUSTER_SIZE * CLUSTER_N.
  // We launch flat (CLUSTER_CTAS,1,1) and decode rank → (cta_m, cta_n); the
  // flat launch is provably identical to a (CLUSTER_SIZE,CLUSTER_N,1) launch
  // because pairing is by ctarank parity (PTX §9.7.17.5.1). See analyzations.md
  // §30.
  static constexpr int CLUSTER_N = _CLUSTER_N;
  static constexpr bool SWAP_AB = _SWAP_AB;
  static constexpr int CLUSTER_CTAS = CLUSTER_SIZE * CLUSTER_N;
  static constexpr int STORE_COLS = 64;      // N-columns per TMA store call
  static constexpr int STORE_PIPE_DEPTH = 2; // store smem double buffer (fixed)
  static constexpr uint16_t CTA_MASK =
      (uint16_t)((1u << CLUSTER_SIZE) - 1u); // per-MMA-group commit mask (base;
                                             // kernel shifts by CS*cta_n_grant)

  // ── tile dimensions ──────────────────────────────────────────────────────
  static constexpr int BM = _BM; // M-rows per CTA
  static constexpr int BN = _BN; // N-cols per cluster (each CTA holds BN/2)
  static constexpr int BK = _BK; // K-depth per pipeline stage

  // ── pipeline depths ──────────────────────────────────────────────────────
  static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH; // TMEM slots
  static constexpr int CLC_PIPE_DEPTH = _CLC_PIPE_DEPTH; // CLC stages

  // ── warp layout ──────────────────────────────────────────────────────────
  static constexpr int NUM_EPI_WARPS =
      SWAP_AB ? 4 : (BM / WARP_THREADS); // swap: 4 warps read the C^T
                                         // accumulator (128 lanes + strips),
                                         // independent of BM. else: 1/32 rows
  static constexpr int NUM_PROD_WARPS = 1;
  static constexpr int NUM_CONS_WARPS = 1;
  static constexpr int NUM_SCHED_WARPS = 1;
  static constexpr int NUM_WARPS =
      NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS + NUM_SCHED_WARPS;
  static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

  // ── smem sizes (bytes) ───────────────────────────────────────────────────
  // A smem: BM × BK bf16 per stage (TN has [BK×BM] but same byte count)
  static constexpr int A_STAGE_BYTES = BM * BK * 2;
  // B smem: (BN/2) × BK bf16 per stage (each CTA holds half the N columns)
  static constexpr int B_STAGE_BYTES = (BN / CLUSTER_SIZE) * BK * 2;
  static constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
  // Store buffer: double-buffered STORE_COLS-wide tiles
  static constexpr int STORE_BUF_BYTES = STORE_PIPE_DEPTH * BM * STORE_COLS * 2;
  // Maximum usable smem on B300 per CTA
  static constexpr int SMEM_BUDGET = 227 * 1024;

  // ── load pipeline depth ──────────────────────────────────────────────────
  // If the caller passes 0 (default), auto-derive the maximum depth that fits.
  // If the caller passes an explicit value it is used verbatim (asserted
  // below).
  static constexpr int LOAD_PIPE_DEPTH =
      (_LOAD_PIPE_DEPTH == 0) ? (SMEM_BUDGET - STORE_BUF_BYTES) / STAGE_BYTES
                              : _LOAD_PIPE_DEPTH;

  static constexpr int SMEM_BYTES =
      LOAD_PIPE_DEPTH * STAGE_BYTES + STORE_BUF_BYTES;

  // ── CLC barrier arrive count ─────────────────────────────────────────────
  // Every warp of EVERY CTA in the cluster arrives on CTA0's CLC-empty barrier
  // each tile → scale by CLUSTER_CTAS (all 4 CTAs for 2x2), not CLUSTER_SIZE.
  static constexpr int CLC_EMPTY_ARRIVE =
      (NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS + NUM_SCHED_WARPS) *
      CLUSTER_CTAS;

  // ── warp index offsets ───────────────────────────────────────────────────
  static constexpr int PRODUCER_WARP = NUM_EPI_WARPS;
  static constexpr int CONSUMER_WARP = NUM_EPI_WARPS + 1;
  static constexpr int SCHEDULER_WARP = NUM_EPI_WARPS + 2;

  // ── validity checks ──────────────────────────────────────────────────────
  static_assert(
      BM >= 32 && BM % WARP_THREADS == 0,
      "BM must be a positive multiple of 32 (one warp per row group)");
  static_assert(
      BN >= 64 && BN % (CLUSTER_SIZE * SWIZZLE_W) == 0,
      "BN must be a multiple of CLUSTER_SIZE*SWIZZLE_W (per-CTA slice "
      "of B must be whole 64-col swizzle chunks)");
  static_assert(BK >= 64 && BK % SWIZZLE_W == 0,
                "BK must be a multiple of 64 (SWIZZLE_W)");
  static_assert(
      LOAD_PIPE_DEPTH >= 2,
      "LOAD_PIPE_DEPTH must be >= 2 (need at least one stage in-flight)");
  static_assert(EPI_PIPE_DEPTH >= 1, "EPI_PIPE_DEPTH must be >= 1");
  static_assert(CLC_PIPE_DEPTH >= 1, "CLC_PIPE_DEPTH must be >= 1");
  static_assert(SMEM_BYTES <= SMEM_BUDGET,
                "Smem footprint exceeds 227 KiB budget — reduce BM/BN/BK or "
                "LOAD_PIPE_DEPTH");
};

// Convenience aliases matching the validated configs used today. BM is fixed at
// 128 for this cta_group::2 kernel (see the BM==128 static_assert in the
// launcher and §28.7 — BM=64 needs a cta_group::1 variant, not a config alias).
using Sm103Config_128x128x64 = Sm103ClusterConfig<128, 128, 64>;
using Sm103Config_256x256x64 = Sm103ClusterConfig<128, 256, 64>;
// 2×2 cluster variants (CLUSTER_N=2 → 4 CTAs = 1 M-pair × 2 N-blocks). Each
// cluster computes 2 adjacent BN-blocks; A is shared (multicast) across them.
// P2 (§30). Same per-CTA tile (128×BN) — the cluster just covers 2× the N.
using Sm103Config_128x256_2x2 = Sm103ClusterConfig<128, 256, 64, 2>;
using Sm103Config_128x128_2x2 = Sm103ClusterConfig<128, 128, 64, 2>;
// 1x2 cluster, cta_group::1 (single-CTA MMA): bring-up tile 128x192 per CTA,
// 2 CTAs along N share A (multicast, NO parity routing — cta_group::1 sends
// bytes+tx to each destination's OWN mbar; 1x2_plan.md Q3). BN is per-CTA.
using Sm103Config_1x2_128x192 =
    Sm103ClusterConfig<128, 192, 64, 2, 0, 2, 2, /*CLUSTER_SIZE=*/1>;
// Swap-AB B1 (1x2_plan.md R7): same C-tile 128x192 but computed as C^T —
// MMA-M' = BN = 192 (composite 128+64 atoms), MMA-N' = BM = 128. Exercises the
// composite MMA + transposed epilogue with both operands still 128B-swizzled.
// The 160x192 tile (64B-swizzle operand) is B2 on top of this.
using Sm103Config_1x2swap_128x192 =
    Sm103ClusterConfig<128, 192, 64, 2, 0, 2, 2, 1, /*SWAP_AB=*/true>;
// B2 — the real nvjet tile (nvjet_sm103_tst_160x192_64x5_1x2_h_bz): C-tile
// 160x192 as C^T (M'=192 composite, N'=160). The 160-wide shared operand is
// 64B-SWIZZLED (160 = 5x32-elem chunks; 128B chunking cannot address 160-
// aligned slices — 1x2_plan.md RESOLVED/R7). TMEM: 2x160 + 160 = 480 <= 512.
using Sm103Config_1x2swap_160x192 =
    Sm103ClusterConfig<160, 192, 64, 2, 0, 2, 2, 1, /*SWAP_AB=*/true>;

// ─── device primitives (hardware-level, identical for all layouts/configs) ───

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFFu));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect(int addr, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, "
               "[%0], %1;" ::"r"(addr),
               "r"(tx)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_cluster(int addr,
                                                               int tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 "
               "_, [%0], %1;" ::"r"(addr),
               "r"(tx)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_cluster(int addr) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" ::"r"(addr)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(int addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred P;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], "
               "%1, %2;\n\t"
               "@P bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t"
               "}" ::"r"(addr),
               "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void cluster_fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n"
               "barrier.cluster.wait.acquire.aligned;\n" ::
                   : "memory");
}

__device__ __forceinline__ uint32_t get_cluster_cta_rank() {
  // Flat cluster rank across all dims (PTX §5.x %cluster_ctarank). Shape-
  // independent: 0..CLUSTER_CTAS-1 for any launch dims. The cta_group::2 pair
  // is the CTAs whose rank differs only in the last bit (PTX §9.7.17.5.1), so
  // cta_m = rank&1, cta_n = rank>>1 for our flat (CLUSTER_CTAS,1,1) launch.
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t map_smem_addr_to_cta_rank(uint32_t addr,
                                                              uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r)
               : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ uint32_t get_cluster_nctas() {
  // Number of CTAs in THIS cluster as GRANTED at runtime (PTX %cluster_nctaid).
  // Under a preferred-cluster launch (CUDA 12.8 fallback clusters) this is
  // either the preferred or the fallback size — the one piece of state a
  // DynamicCluster kernel must query (CUTLASS cluster_shape() pattern,
  // cluster_sm90.hpp:140). §31.1.
  uint32_t n;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(n));
  return n;
}

// 4D TMA: bulk async load [x × y × z × w] tile into shared memory
template <int CTA_GROUP = 1>
__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void *tmap,
                                                 int x, int y, int z, int w,
                                                 int mbar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global"
               ".mbarrier::complete_tx::bytes.cta_group::%7"
               " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
               "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar),
               "n"(CTA_GROUP)
               : "memory");
}

// Multicast variant: copies to `dst` in EVERY cluster CTA whose bit is set in
// ctaMask (each bit = a %cluster_ctarank). Used to share A across the N-partner
// CTAs of a 2×2 cluster (mask 0b0101/0b1010). PTX ISA 9.3 §9.7 (line 17169):
// with cta_group::2 the mbarrier completion routes to the even/leader CTA of
// each destination pair by ctarank parity.
template <int CTA_GROUP = 2>
__device__ __forceinline__ void
tma_4d_gmem2smem_mc(int dst, const void *tmap, int x, int y, int z, int w,
                    int mbar, uint16_t ctaMask) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global"
               ".mbarrier::complete_tx::bytes.cta_group::%8.multicast::cluster"
               " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
               "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar),
               "h"(ctaMask), "n"(CTA_GROUP)
               : "memory");
}

// A-load policy: MulticastA=false → every CTA loads its own A (redundant-A,
// validated M1). MulticastA=true → only the cta_n==0 CTA of each A-group issues
// A and multicasts it to its N-partner (mask 0b0101<<cta_m), halving A DRAM
// traffic.
//
// BOTH paths use the LEADER mbar (mapa'd to the even pair-CTA for the odd
// partner). REQUIRED for the multicast: PTX ISA 9.3 (cp.async.bulk.tensor,
// ~line 17169) — with .cta_group::2 the mbarrier signal goes to the even or
// odd CTA of EACH destination pair "based on the %cluster_ctarank parity of
// the shared memory where the mbarrier object mbar RESIDES". It is NOT
// automatic route-to-leader. Passing the raw LOCAL mbar from the odd issuer
// (odd parity) sent the +A bytes to the ODD CTA of every destination pair —
// barriers nobody waits on — starving both leaders' full barrier of A_bytes
// every stage → the P2-M2 hang. CUTLASS forces even parity the same way
// (clears the peer bit, Sm100MmaPeerBitMask, copy_sm100_tma.hpp:45/318).
// With the leader mbar each leader collects 2×A (both A-multicasts) + 2×B
// (its pair's B loads) = its expected 2×(A+B). Balanced on both pairs.
template <int CTA_GROUP, bool MulticastA>
__device__ __forceinline__ void
issue_A_load(int A_smem, const void *A_tmap, int x, int y, int z, int w,
             int leader_mbar, uint16_t a_mask, uint32_t cta_n) {
  if constexpr (MulticastA) {
    if (cta_n == 0)
      tma_4d_gmem2smem_mc<CTA_GROUP>(A_smem, A_tmap, x, y, z, w, leader_mbar,
                                     a_mask);
  } else {
    tma_4d_gmem2smem<CTA_GROUP>(A_smem, A_tmap, x, y, z, w, leader_mbar);
  }
}

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  // 1-arg form (LBO=0, SBO=1024) — correct for NN and NT operands.
  // SWIZZLE_W = 64 elements = 128 bytes — fixed for all SM103 configs
  constexpr int stride =
      64 * 2 * 8; // SWIZZLE_W * sizeof(bf16) * 8 (swizzle factor)
  return desc_encode(addr) | (desc_encode(stride) << 32ULL) | (1ULL << 46ULL) |
         (2ULL << 61ULL);
}

// 3-arg form — explicit LBO (Leading Dim Byte Offset, bits 16-29) + SBO
// (Stride Dim Byte Offset, bits 32-45). TN needs LBO=BK*SWIZZLE_W*2 (=8192 for
// BK=64) because A is stored [K,M] and its chunk-major smem spans >1 m-block;
// the 1-arg form (LBO=0) gives wrong results for TN. Ported from fixed
// Bgemm_sm103_bf16_TN_cluster.cu (2026-07-02).
__device__ __forceinline__ uint64_t make_smem_desc(int addr, int lbo_bytes,
                                                   int sbo_bytes) {
  return desc_encode(addr) | (desc_encode(lbo_bytes) << 16ULL) |
         (desc_encode(sbo_bytes) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

// 64B-swizzle descriptor (PTX Table 45 mode 4; 1x2_plan.md R2): for MN-major
// operands whose contiguous extent is a multiple of 32 but NOT 64 elements
// (e.g. the 160-wide swap-AB operand). LBO = BK*64 = 4096, SBO = 8*64 = 512.
__device__ __forceinline__ uint64_t make_smem_desc_sw64(int addr, int lbo_bytes,
                                                        int sbo_bytes) {
  return desc_encode(addr) | (desc_encode(lbo_bytes) << 16ULL) |
         (desc_encode(sbo_bytes) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void
tcgen05_mma_bf16(int tmem_addr, uint64_t a_desc, uint64_t b_desc,
                 uint32_t i_desc, int use_accum) {
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
               "}" ::"r"(tmem_addr),
               "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(use_accum),
               "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void alloc_tmem(int tmem_smem_addr, int width) {
  asm volatile(
      "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;" ::
          "r"(tmem_smem_addr),
      "r"(width), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void dealloc_tmem(int tmem_addr, int width) {
  asm volatile(
      "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr),
      "r"(width), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 2>
__device__ __forceinline__ void tcgen05_commit_multicast(int mbar_addr,
                                                         uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::%2"
               ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
               " [%0], %1;" ::"r"(mbar_addr),
               "h"(mask), "n"(CTA_GROUP)
               : "memory");
}

__device__ __forceinline__ void tcgen05_ld(float *tmp, int addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
               " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                 "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
               : "r"(addr));
}

// 16-lane TMEM load for M=64 "Layout F" atom strips (swap-AB epilogue).
// .16x256b.x1: 32 threads cover 16 lanes x 8 cols. Fragment (CUTLASS
// copy_traits_sm100.hpp:467-481): thread t -> r0=(lane t/4,  col 2*(t%4)),
// r1=(t/4, col+1), r2=(lane t/4+8, col), r3=(t/4+8, col+1). Lane base from
// the address lane field (must be 16-aligned).
__device__ __forceinline__ void tcgen05_ld_16x256b(float *tmp, int addr) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32"
               " {%0,%1,%2,%3}, [%4];"
               : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
               : "r"(addr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void tcgen05_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N> __device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void tma_2d_smem2gmem(int src, const void *tmap,
                                                 int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
               " [%0, {%2, %3}], [%1];" ::"l"(tmap),
               "r"(src), "r"(x), "r"(y)
               : "memory");
}

__device__ __forceinline__ void barrier_sync(int bar, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(n));
}

// ─── CLC primitives ──────────────────────────────────────────────────────────

__device__ __forceinline__ void clc_try_cancel(int response_addr,
                                               int mbar_addr) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta"
               ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 "
               "[%0], [%1];" ::"r"(response_addr),
               "r"(mbar_addr)
               : "memory");
}

__device__ __forceinline__ void
clc_query_response(int response_addr, uint32_t &is_valid, uint32_t &new_ctaid) {
  asm volatile(
      "{\n\t"
      ".reg .b128 handle;\n\t"
      ".reg .pred p;\n\t"
      "mov.s32 %0, 0;\n\t"
      "ld.shared.b128 handle, [%2];\n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
      "@p mov.s32 %0, 1;\n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, "
      "handle;\n\t"
      "}"
      : "=r"(is_valid), "=r"(new_ctaid)
      : "r"(response_addr)
      : "memory");
}

// ─── host-side TMA descriptor builders ───────────────────────────────────────

static inline void check_cu(CUresult e) {
  if (e != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(e, &msg);
    fprintf(stderr, "[BluBridge SM103] cuTensorMapEncodeTiled: %s (%d)\n",
            msg ? msg : "unknown", (int)e);
    exit(EXIT_FAILURE);
  }
}

// A[batch, M, K] row-major — NN and NT layouts (A is not transposed)
// TMA coords: (0, A_row, k_chunk, batch_idx)
static inline void init_4d_tma_A_MK(CUtensorMap *tmap, const __nv_bfloat16 *ptr,
                                    int M, int K, int BM, int BK,
                                    uint64_t batchCount, uint64_t batchStride,
                                    CUtensorMapSwizzle swizzle) {
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64ULL, (uint64_t)M, (uint64_t)(K / 64),
                              batchCount};
  uint64_t globalStrides[rank - 1] = {(uint64_t)K * sizeof(__nv_bfloat16),
                                      128ULL,
                                      batchStride * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {64U, (uint32_t)BM, (uint32_t)(BK / 64), 1U};
  uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};
  check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
                                  (void *)ptr, globalDim, globalStrides, boxDim,
                                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// A[batch, K, M] row-major — TN layout (opA=T: A stored transposed)
// TMA coords: (0, k*BK, A_col, batch_idx)  where A_col = block_row*BM/SWIZZLE_W
static inline void init_4d_tma_A_KM(CUtensorMap *tmap, const __nv_bfloat16 *ptr,
                                    int M, int K, int BM, int BK,
                                    uint64_t batchCount, uint64_t batchStride,
                                    CUtensorMapSwizzle swizzle) {
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64ULL, (uint64_t)K, (uint64_t)(M / 64),
                              batchCount};
  uint64_t globalStrides[rank - 1] = {(uint64_t)M * sizeof(__nv_bfloat16),
                                      128ULL,
                                      batchStride * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {64U, (uint32_t)BK, (uint32_t)(BM / 64), 1U};
  uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};
  check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
                                  (void *)ptr, globalDim, globalStrides, boxDim,
                                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// A[batch, K, M] with 64B SWIZZLE — for tiles whose M-slice is a multiple of
// 32 but not 64 (the swap-AB 160-wide operand; 1x2_plan.md R7a). tmap
// {32, K, M/32}; inner box = 32 elems (64B swizzle cap); z = block_row*(BM/32).
static inline void init_4d_tma_A_KM_sw64(CUtensorMap *tmap,
                                         const __nv_bfloat16 *ptr, int M, int K,
                                         int BM, int BK, uint64_t batchCount,
                                         uint64_t batchStride) {
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {32ULL, (uint64_t)K, (uint64_t)(M / 32),
                              batchCount};
  uint64_t globalStrides[rank - 1] = {(uint64_t)M * sizeof(__nv_bfloat16),
                                      64ULL,
                                      batchStride * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {32U, (uint32_t)BK, (uint32_t)(BM / 32), 1U};
  uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};
  check_cu(cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void *)ptr, globalDim,
      globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// B[batch, K, N] row-major — NN and TN layouts (opB=N: B not transposed)
// TMA coords: (0, k*BK, B_col, batch_idx)  where B_col = (block_col*BN +
// rank*BN/2)/SWIZZLE_W
static inline void init_4d_tma_B_KN(CUtensorMap *tmap, const __nv_bfloat16 *ptr,
                                    int K, int N, int BK, int BN,
                                    uint64_t batchCount, uint64_t batchStride,
                                    CUtensorMapSwizzle swizzle,
                                    int clusterSize = 2) {
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64ULL, (uint64_t)K, (uint64_t)(N / 64),
                              batchCount};
  uint64_t globalStrides[rank - 1] = {(uint64_t)N * sizeof(__nv_bfloat16),
                                      128ULL,
                                      batchStride * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {64U, (uint32_t)BK, (uint32_t)(BN / clusterSize / 64),
                           1U};
  uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};
  check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
                                  (void *)ptr, globalDim, globalStrides, boxDim,
                                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// B[batch, N, K] row-major — NT layout (opB=T: B stored transposed [N,K])
// TMA coords: (0, B_row, k_chunk, batch_idx)  where B_row = block_col*BN +
// rank*BN/2
static inline void init_4d_tma_B_NK(CUtensorMap *tmap, const __nv_bfloat16 *ptr,
                                    int N, int K, int BN, int BK,
                                    uint64_t batchCount, uint64_t batchStride,
                                    CUtensorMapSwizzle swizzle,
                                    int clusterSize = 2) {
  constexpr uint32_t rank = 4;
  uint64_t globalDim[rank] = {64ULL, (uint64_t)N, (uint64_t)(K / 64),
                              batchCount};
  uint64_t globalStrides[rank - 1] = {(uint64_t)K * sizeof(__nv_bfloat16),
                                      128ULL,
                                      batchStride * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {64U, (uint32_t)(BN / clusterSize),
                           (uint32_t)(BK / 64), 1U};
  uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};
  check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
                                  (void *)ptr, globalDim, globalStrides, boxDim,
                                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// C[batch*M, N] flattened — identical for all layouts
// TMA coords: (c_col, c_row)
static inline void init_2d_tma_C(CUtensorMap *tmap, const __nv_bfloat16 *ptr,
                                 uint64_t totalRows, uint64_t totalCols,
                                 uint32_t boxRows, uint32_t boxCols,
                                 CUtensorMapSwizzle swizzle) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {totalCols, totalRows};
  uint64_t globalStrides[rank - 1] = {totalCols * sizeof(__nv_bfloat16)};
  uint32_t boxDim[rank] = {boxCols, boxRows};
  uint32_t elementStrides[rank] = {1U, 1U};
  check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
                                  (void *)ptr, globalDim, globalStrides, boxDim,
                                  elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ─── i_desc builder
// ───────────────────────────────────────────────────────────
//
// tcgen05 MMA instruction descriptor (compile-time constant per Config+Layout):
//   bits [4,7,10] : K-dimension sub-descriptors (always set)
//   bit  15       : trans_a = 1  → TN only   (A stored [K,M], read as M-major
//   columns) bit  16       : trans_b = 1  → NN and TN (B stored [K,N], read as
//   row-major)
//                                → clear for NT (B stored [N,K])
//   bits [17..23] : BN >> 3      (N-tile width in 8-element units)
//   bits [24..30] : (CLUSTER_SIZE * BM) >> 4  (cluster M height in 16-element
//   units)

template <typename Config, Sm103Layout Layout>
__host__ __device__ constexpr uint32_t sm103_make_i_desc() {
  uint32_t d = (1U << 4U) | (1U << 7U) | (1U << 10U) |
               ((uint32_t)(Config::BN) >> 3U << 17U) |
               ((uint32_t)(Config::CLUSTER_SIZE * Config::BM) >> 4U << 24U);
  if constexpr (Layout == Sm103Layout::NN)
    d |= (1U << 16U);
  if constexpr (Layout == Sm103Layout::TN)
    d |= (1U << 15U) | (1U << 16U);
  // NT: no trans bits — B stored [N,K], MMA sees it as already column-major
  return d;
}

// Swap-AB instruction descriptors (1x2_plan.md R7e): the MMA computes
// C' = C^T, so N-field = BM>>3 (C'-cols = C-rows) and M-field = the atom
// height (128 or 64; M' = BN is composed of the two atoms). Both operands are
// the SAME K-major storages as TN, so both transpose bits stay set. TN only.
template <typename Config, int M_ATOM>
__host__ __device__ constexpr uint32_t sm103_make_i_desc_swap() {
  return (1U << 4U) | (1U << 7U) | (1U << 10U) |
         ((uint32_t)(Config::BM) >> 3U << 17U) |
         ((uint32_t)(M_ATOM) >> 4U << 24U) | (1U << 15U) | (1U << 16U);
}

// ─── tile-index decode (wave scheduling policy) ──────────────────────────────
//
// Maps a linear MN-tile index → (block_col, row_tile). Two policies:
//   Horizontal (default): N varies fastest (block_col = idx % grid_n). This is
//     the original behaviour.
//   Vertical (cuBLAS `128x256_2x1_v`): M varies fastest (row_tile = idx %
//   grid_m)
//     so the streamed weight column (B) stays resident in L2 across many
//     M-tiles — the dominant Qwen kernel policy (§16.7). block_row =
//     row_tile*CLUSTER
//     + cta_rank is applied by the caller.
template <bool VerticalWave>
__device__ __forceinline__ void decode_mn_tile(int mn_tile, int grid_n,
                                               int grid_m, int &block_col,
                                               int &row_tile) {
  if constexpr (VerticalWave) {
    row_tile = mn_tile % grid_m;
    block_col = mn_tile / grid_m;
  } else {
    block_col = mn_tile % grid_n;
    row_tile = mn_tile / grid_n;
  }
}

// ─── unified kernel
// ───────────────────────────────────────────────────────────
//
// Grid:  dim3(total_mn_tiles * batchCount * Config::CLUSTER_CTAS)
// Block: dim3(Config::NUM_THREADS)
// Smem:  Config::SMEM_BYTES dynamic shared memory
//
// Cluster dims are set via LAUNCH ATTRIBUTES (cudaLaunchKernelEx), not
// __cluster_dims__, so one kernel supports BOTH rigid clusters and the
// DynamicCluster (preferred+fallback, "2x2f") launch. CUTLASS does the same
// (no __cluster_dims__; cluster_launch.hpp). §31.1.
//
// DynamicCluster=true — the cuBLAS "f" variant: launched with fallback
// cluster (CLUSTER_SIZE,1,1) + preferred (CLUSTER_CTAS,1,1). Each cluster
// reads its GRANTED size at runtime and adapts barrier counts + masks +
// A-multicast. Tile decode becomes CTA-level: the PREFERRED-relative offset
// (blockIdx.x % CLUSTER_CTAS) picks this pair's N-block, so a preferred slot
// split into 2 fallback clusters still covers both N-blocks exactly once.
// M2: CLC tile-stealing ENABLED for DynamicCluster per the MEASURED B300
// contract (probe §31.3): cancellation units are REQUESTER-SIZED (a
// fallback-2 cluster cancels 2-CTA pieces, either half of a preferred unit;
// a preferred-4 cluster cancels whole units — unit accounting closes
// exactly). The stolen piece's N-block is decoded from the response ctaid:
// n_sub = ((new_ctaid %% CLUSTER_CTAS)>>1) + cta_n_grant.

template <typename Config, Sm103Layout Layout, bool VerticalWave = false,
          bool MulticastA = false, bool DynamicCluster = false>
__global__
__launch_bounds__(Config::NUM_THREADS, 1) void bgemm_sm103_cluster_kernel(
    const __nv_bfloat16
        *__restrict__ A, // global A (layout-dependent storage order)
    const __nv_bfloat16 *__restrict__ C_in, // [batch, M, N] for beta read-back
    __nv_bfloat16 *__restrict__ C,          // [batch, M, N] output
    int M, int N, int K,
    int total_mn_tiles, // (M / (CLUSTER_SIZE * BM)) * (N / BN)
    float alpha, float beta, const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int grid_n) // N / BN  (for on-device tile decode)
{
  // Unpack Config members into local aliases for readability
  constexpr int BM = Config::BM;
  constexpr int BN = Config::BN;
  constexpr int BK = Config::BK;
  constexpr int LOAD_PIPE_DEPTH = Config::LOAD_PIPE_DEPTH;
  constexpr int EPI_PIPE_DEPTH = Config::EPI_PIPE_DEPTH;
  constexpr int CLC_PIPE_DEPTH = Config::CLC_PIPE_DEPTH;
  constexpr int WARP_THREADS = Config::WARP_THREADS;
  constexpr int SWIZZLE_W = Config::SWIZZLE_W;
  constexpr int MMA_K_STEP = Config::MMA_K_STEP;
  constexpr int CLUSTER_SIZE = Config::CLUSTER_SIZE;
  constexpr int STORE_COLS = Config::STORE_COLS;
  constexpr int STORE_PIPE_DEPTH = Config::STORE_PIPE_DEPTH;
  constexpr uint16_t CTA_MASK = Config::CTA_MASK;

  constexpr int NUM_EPI_WARPS = Config::NUM_EPI_WARPS;
  constexpr int PRODUCER_WARP = Config::PRODUCER_WARP;
  constexpr int CONSUMER_WARP = Config::CONSUMER_WARP;
  constexpr int SCHEDULER_WARP = Config::SCHEDULER_WARP;
  constexpr int CLC_EMPTY_ARRIVE = Config::CLC_EMPTY_ARRIVE;

  constexpr int CLUSTER_N = Config::CLUSTER_N;
  constexpr int CLUSTER_CTAS = Config::CLUSTER_CTAS;
  constexpr bool SWAP_AB = Config::SWAP_AB;
  static_assert(!SWAP_AB || (Layout == Sm103Layout::TN && CLUSTER_SIZE == 1 &&
                             Config::BN == 192 &&
                             (Config::BM == 128 || Config::BM == 160)),
                "SWAP_AB path: TN, cta_group::1, BN=192, BM in {128 (B1), "
                "160 (B2, 64B-swizzle operand)}");
  // 160-wide operand cannot be 128B-chunked (offsets land mid-chunk) — it
  // gets its own 64B-swizzle tmap + descriptors (1x2_plan.md RESOLVED).
  constexpr bool A_SW64 = SWAP_AB && (Config::BM % 64 != 0);

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_THREADS;
  const int lane_id = tid % WARP_THREADS;
  const uint32_t cta_rank = get_cluster_cta_rank();

  // GRANTED cluster size — runtime under DynamicCluster (preferred OR
  // fallback, per cluster), compile-time otherwise. Everything barrier/mask
  // related derives from the GRANTED size: barriers never cross cluster
  // boundaries, so mixed-size clusters coexist safely (CUTLASS, §31.1).
  const int cluster_ctas =
      DynamicCluster ? (int)get_cluster_nctas() : CLUSTER_CTAS;
  const int cluster_n = cluster_ctas / CLUSTER_SIZE;

  // Cluster-rank decomposition (§30): pair is the last rank bit (PTX
  // §9.7.17.5.1).
  //   cta_m       — M-partner within the MMA pair (0/1); drives which M-half.
  //   cta_n       — ADDRESSING N-block index: which BN-block of the PREFERRED
  //                 tile-unit this pair owns. Static: rank>>1. Dynamic: from
  //                 the PREFERRED-relative CTA offset
  //                 (blockIdx.x % CLUSTER_CTAS)>>1, so a preferred slot split
  //                 into two fallback clusters still covers both N-blocks
  //                 exactly once (CUTLASS CTA-level decode, §31.1).
  //   cta_n_grant — GRANTED-relative pair index (rank>>1): gates who issues
  //                 the A multicast + shifts the pair-local commit mask.
  //   pair_leader — even rank of my pair; owns the pair's full_mbar + issues
  //                 the cta_group::2 MMA.
  const uint32_t cta_m = cta_rank % (uint32_t)CLUSTER_SIZE;
  const uint32_t cta_n =
      DynamicCluster
          ? ((blockIdx.x % (uint32_t)CLUSTER_CTAS) / (uint32_t)CLUSTER_SIZE)
          : (cta_rank / (uint32_t)CLUSTER_SIZE);
  const uint32_t cta_n_grant = cta_rank / (uint32_t)CLUSTER_SIZE;
  const uint32_t pair_leader = cta_rank - cta_m;

  // ── shared memory layout ──────────────────────────────────────────────────
  extern __shared__ __align__(1024) char smem[];
  const int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int tmem[1];
  const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

  __shared__ __align__(8) uint64_t full_mbar[LOAD_PIPE_DEPTH];
  __shared__ __align__(8) uint64_t empty_mbar[LOAD_PIPE_DEPTH];
  __shared__ __align__(8) uint64_t tmem_full[EPI_PIPE_DEPTH];
  __shared__ __align__(8) uint64_t tmem_empty[EPI_PIPE_DEPTH];
  __shared__ __align__(16) uint8_t clc_response[CLC_PIPE_DEPTH][16];
  __shared__ __align__(8) uint64_t clc_full_mbar[CLC_PIPE_DEPTH];
  __shared__ __align__(8) uint64_t clc_empty_mbar[CLC_PIPE_DEPTH];

  const int full_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(full_mbar));
  const int empty_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(empty_mbar));
  const int tmem_full_addr =
      static_cast<int>(__cvta_generic_to_shared(tmem_full));
  const int tmem_empty_addr =
      static_cast<int>(__cvta_generic_to_shared(tmem_empty));
  const int clc_response_addr =
      static_cast<int>(__cvta_generic_to_shared(clc_response));
  const int clc_full_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(clc_full_mbar));
  const int clc_empty_mbar_addr =
      static_cast<int>(__cvta_generic_to_shared(clc_empty_mbar));

  // Pipeline stage smem: A tile + B tile per stage
  // A smem: BM × BK bf16  (TN: [BK×BM] but same byte count — BK*BM == BM*BK)
  // B smem: (BN/2) × BK bf16  (each CTA holds half the N-column tile)
  constexpr int a_tile_bytes = Config::A_STAGE_BYTES;
  constexpr int b_tile_bytes = Config::B_STAGE_BYTES;
  constexpr int copy_size = Config::STAGE_BYTES;
  constexpr int store_buf_sz =
      BM * STORE_COLS * 2; // one store-smem slot (bytes)
  const int store_smem = smem_ptr + LOAD_PIPE_DEPTH * copy_size;
  __nv_bfloat16 *store_base =
      reinterpret_cast<__nv_bfloat16 *>(smem + LOAD_PIPE_DEPTH * copy_size);

  // ── initialisation ────────────────────────────────────────────────────────
  if (warp_id == PRODUCER_WARP && elect_sync()) {
    for (int i = 0; i < LOAD_PIPE_DEPTH; ++i) {
      mbarrier_init(full_mbar_addr + i * 8,
                    1 * CLUSTER_SIZE); // 1 producer × 2 CTAs
      // Operand-smem EMPTY release must span the WHOLE multicast group, not
      // just the MMA pair: A is shared across the CLUSTER_N N-blocks, so its
      // smem may only be refilled after EVERY pair-leader has finished reading
      // it. count = one arrive per pair-leader = CLUSTER_N (CUTLASS
      // multicast_consumer_arrival_count = CM/atomM + CN/atomN - 1 = CLUSTER_N
      // for atom(2,1); == 1 for the 2×1 case, so no regression). §30.x.
      mbarrier_init(empty_mbar_addr + i * 8, cluster_n);
    }
    for (int i = 0; i < EPI_PIPE_DEPTH; ++i) {
      mbarrier_init(tmem_full_addr + i * 8, 1);
      mbarrier_init(tmem_empty_addr + i * 8, NUM_EPI_WARPS * CLUSTER_SIZE);
    }
    for (int i = 0; i < CLC_PIPE_DEPTH; ++i) {
      mbarrier_init(clc_full_mbar_addr + i * 8, 1);
      // NUM_WARPS × GRANTED cluster CTAs (runtime under DynamicCluster)
      mbarrier_init(clc_empty_mbar_addr + i * 8,
                    (CLC_EMPTY_ARRIVE / CLUSTER_CTAS) * cluster_ctas);
    }
    cluster_fence_mbarrier_init();
  } else if (warp_id == CONSUMER_WARP) {
    // tcgen05.alloc column count must be a power of 2 in [32,512] (PTX
    // ~32701): round BN*EPI up (384 -> 512 for BN=192). Extra cols unused.
    constexpr int TMEM_COLS_RAW = BN * EPI_PIPE_DEPTH;
    constexpr int TMEM_COLS = (TMEM_COLS_RAW <= 32)    ? 32
                              : (TMEM_COLS_RAW <= 64)  ? 64
                              : (TMEM_COLS_RAW <= 128) ? 128
                              : (TMEM_COLS_RAW <= 256) ? 256
                                                       : 512;
    static_assert(TMEM_COLS_RAW <= 512, "accumulator exceeds TMEM columns");
    alloc_tmem<CLUSTER_SIZE>(tmem_smem_addr, TMEM_COLS);
  }

  cluster_sync();

  const int tmem_addr = tmem[0];
  const int k_iters = K / BK;

  // i_desc: compile-time constant encoding MMA shape + transposition for this
  // layout
  constexpr uint32_t i_desc = sm103_make_i_desc<Config, Layout>();

  // Each PAIR's even (leader) CTA owns that pair's full_mbar; the odd partner
  // reaches across to its pair_leader (rank 0 for pair0, rank 2 for pair1).
  const int tma_mbar_base =
      (cta_m == 0) ? full_mbar_addr
                   : map_smem_addr_to_cta_rank(full_mbar_addr, pair_leader);

  const int cta0_clc_empty_addr =
      (cta_rank == 0) ? clc_empty_mbar_addr
                      : map_smem_addr_to_cta_rank(clc_empty_mbar_addr, 0);

  // One work item = one CLUSTER (CLUSTER_CTAS CTAs). total_mn_tiles counts
  // cluster-tiles, where each cluster covers CLUSTER_SIZE*BM rows × CLUSTER_N
  // N-blocks. grid_n is the number of N-CLUSTERS (= ceil(N/BN / CLUSTER_N)).
  const int init_tile_idx = (int)(blockIdx.x / CLUSTER_CTAS);
  int batch = init_tile_idx / total_mn_tiles;
  int mn_tile = init_tile_idx % total_mn_tiles;

  // grid_m (M-tiles per column) for the vertical-wave tile decode. grid_n is
  // passed in; total_mn_tiles = grid_m * grid_n, so grid_m = total/grid_n.
  const int grid_m = total_mn_tiles / grid_n;

  // ── producer warp ─────────────────────────────────────────────────────────
  // Loads A and B tiles into the smem pipeline.
  // Layout-specific: how the global addresses map to TMA (x, y, z, w) coords.
  if (warp_id == PRODUCER_WARP && elect_sync()) {
    // decode_mn_tile gives the N-CLUSTER index; this CTA's actual N-block is
    // n_cluster*CLUSTER_N + cta_n (the CLUSTER_N adjacent N-blocks this cluster
    // owns). M-block is by the pair partner cta_m. For CLUSTER_N=1 this reduces
    // to the old 2×1 behaviour (cta_n=0, cta_m=cta_rank).
    int n_cluster, row_tile;
    decode_mn_tile<VerticalWave>(mn_tile, grid_n, grid_m, n_cluster, row_tile);
    int block_row = row_tile * CLUSTER_SIZE + (int)cta_m;
    int block_col = n_cluster * CLUSTER_N + (int)cta_n;
    int cur_batch = batch;
    // A-multicast group mask (MulticastA only): the CTAs sharing this M-half
    // across N = same cta_m, all cta_n. For 2×2: cta_m=0→{0,2}=0b0101,
    // cta_m=1→{1,3}=0b1010. cta_n==0 issues; parity routes completion to
    // leaders.
    uint16_t a_mask = 0; // CTAs sharing this A slice: same cta_m, every cta_n'
    for (int j = 0; j < cluster_n; ++j)
      a_mask |= (uint16_t)(1u << (j * CLUSTER_SIZE + (int)cta_m));

    int stage = 0, phase = 0, issued = 0;
    int clc_stage = 0, clc_full_phase = 0;

    while (true) {
      if constexpr (Layout == Sm103Layout::TN) {
        // A[K,M]: select BM M-columns by column-group index
        // B[K,N]: select BN/2 N-columns by column-group index
        constexpr int A_CHUNK_W = A_SW64 ? 32 : SWIZZLE_W;
        const int A_col = block_row * BM / A_CHUNK_W;
        const int B_col =
            (block_col * BN + (int)cta_m * (BN / CLUSTER_SIZE)) / SWIZZLE_W;

        for (int k = 0; k < k_iters; ++k) {
          if (issued >= LOAD_PIPE_DEPTH)
            mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
          ++issued;

          const int A_smem = smem_ptr + stage * copy_size;
          const int B_smem = A_smem + a_tile_bytes;
          const int mbar = tma_mbar_base + stage * 8;
          mbarrier_arrive_expect(mbar, copy_size);

          // A TN: y = k*BK (K-row), z = A_col (M col-group)
          issue_A_load<CLUSTER_SIZE, MulticastA>(A_smem, &A_tmap, 0, k * BK,
                                                 A_col, cur_batch, mbar, a_mask,
                                                 cta_n_grant);
          // B TN: y = k*BK (K-row), z = B_col (N col-group)
          tma_4d_gmem2smem<CLUSTER_SIZE>(B_smem, &B_tmap, 0, k * BK, B_col,
                                         cur_batch, mbar);

          stage = (stage + 1) % LOAD_PIPE_DEPTH;
          phase ^= (stage == 0);
        }
      } else if constexpr (Layout == Sm103Layout::NT) {
        // A[M,K]: select BM M-rows
        // B[N,K]: select BN/2 N-rows (B transposed — stored [N,K])
        const int A_row = block_row * BM;
        const int B_row = block_col * BN + (int)cta_m * (BN / CLUSTER_SIZE);

        for (int k = 0; k < k_iters; ++k) {
          if (issued >= LOAD_PIPE_DEPTH)
            mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
          ++issued;

          const int A_smem = smem_ptr + stage * copy_size;
          const int B_smem = A_smem + a_tile_bytes;
          const int mbar = tma_mbar_base + stage * 8;
          mbarrier_arrive_expect(mbar, copy_size);

          // A NT: y = A_row (M-row), z = k (K-chunk index)
          issue_A_load<CLUSTER_SIZE, MulticastA>(A_smem, &A_tmap, 0, A_row, k,
                                                 cur_batch, mbar, a_mask,
                                                 cta_n_grant);
          // B NT: y = B_row (N-row), z = k (K-chunk index)
          tma_4d_gmem2smem<CLUSTER_SIZE>(B_smem, &B_tmap, 0, B_row, k,
                                         cur_batch, mbar);

          stage = (stage + 1) % LOAD_PIPE_DEPTH;
          phase ^= (stage == 0);
        }
      } else // NN
      {
        // A[M,K]: select BM M-rows
        // B[K,N]: select BN/2 N-columns by column-group index
        const int A_row = block_row * BM;
        const int B_col =
            (block_col * BN + (int)cta_m * (BN / CLUSTER_SIZE)) / SWIZZLE_W;

        for (int k = 0; k < k_iters; ++k) {
          if (issued >= LOAD_PIPE_DEPTH)
            mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
          ++issued;

          const int A_smem = smem_ptr + stage * copy_size;
          const int B_smem = A_smem + a_tile_bytes;
          const int mbar = tma_mbar_base + stage * 8;
          mbarrier_arrive_expect(mbar, copy_size);

          // A NN: y = A_row (M-row), z = k (K-chunk index)
          issue_A_load<CLUSTER_SIZE, MulticastA>(A_smem, &A_tmap, 0, A_row, k,
                                                 cur_batch, mbar, a_mask,
                                                 cta_n_grant);
          // B NN: y = k*BK (K-row), z = B_col (N col-group)
          tma_4d_gmem2smem<CLUSTER_SIZE>(B_smem, &B_tmap, 0, k * BK, B_col,
                                         cur_batch, mbar);

          stage = (stage + 1) % LOAD_PIPE_DEPTH;
          phase ^= (stage == 0);
        }
      }

      mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
      uint32_t is_valid, new_ctaid;
      clc_query_response(clc_response_addr + clc_stage * 16, is_valid,
                         new_ctaid);
      mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
      clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
      if (clc_stage == 0)
        clc_full_phase ^= 1;

      if (!is_valid)
        break;

      const int new_tile_idx = (int)(new_ctaid / CLUSTER_CTAS);
      cur_batch = new_tile_idx / total_mn_tiles;
      const int new_mn = new_tile_idx % total_mn_tiles;
      int row_tile, n_cluster;
      decode_mn_tile<VerticalWave>(new_mn, grid_n, grid_m, n_cluster, row_tile);
      block_row = row_tile * CLUSTER_SIZE + (int)cta_m;
      // n_sub per STOLEN tile: cancellation units are REQUESTER-SIZED (B300
      // probe, §31.3) — a fallback-2 piece's position within its preferred
      // unit is encoded in the response ctaid. Degenerates to cta_n for all
      // rigid launches (response always cluster-aligned there).
      block_col =
          n_cluster * CLUSTER_N + (int)(((new_ctaid % (uint32_t)CLUSTER_CTAS) /
                                         (uint32_t)CLUSTER_SIZE) +
                                        cta_n_grant);
      // DO NOT reset `issued` here — it must grow monotonically across
      // CLC tiles so the `if (issued >= LOAD_PIPE_DEPTH) wait()` check
      // fires on the next tile's first issue. Resetting caused a
      // producer/consumer race (stage reuse while in-flight) that shows
      // up as "unspecified launch failure" at scale (many tiles, e.g.
      // 4096³). Ported from the fixed _cluster.cu files (2026-07-02).
    }
  }
  // ── consumer warp ─────────────────────────────────────────────────────────
  // Drives tcgen05.mma from CTA0.  Layout-specific: a_off / b_off reflect
  // how the K dimension strides through smem for each storage order.
  else if (warp_id == CONSUMER_WARP && elect_sync()) {
    int stage = 0, phase = 0;
    int wave_iter = 0;
    int clc_stage = 0, clc_full_phase = 0;
    // Two DISTINCT commit masks for two DISTINCT smem lifetimes (CUTLASS
    // sm100_pipeline.hpp): the accumulator/tmem barrier is PAIR-LOCAL; the
    // operand-smem barrier is CLUSTER-WIDE because A is shared across pairs.
    //
    // pair_mask (accumulator/tmem, calculate_umma_peer_mask): each pair leader
    // signals only ITS 2 CTAs. pair0 (cta_n=0)→0b0011; pair1 (cta_n=1)→0b1100.
    const uint16_t pair_mask =
        (uint16_t)(CTA_MASK << (CLUSTER_SIZE * cta_n_grant));
    // operand_empty_mask (operand smem, calculate_multicast_mask<kRowCol>): for
    // atom(2,1) every CTA participates → full cluster (0b11 for 2×1, 0xF for
    // 2×2). Each pair leader releases with this; each empty_mbar collects
    // CLUSTER_N arrives (matches the CLUSTER_N init count). Using pair_mask
    // here was the deadlock: pair1's producer never saw pair0's release. §30.x.
    const uint16_t operand_empty_mask = (uint16_t)((1u << cluster_ctas) - 1u);

    while (true) {
      // Each pair's even (leader) CTA drives that pair's MMA (ranks 0 and 2);
      // was cta_rank==0 (single leader) in the 2×1 kernel.
      if (cta_m == 0) {
        const int wave_stage = wave_iter % EPI_PIPE_DEPTH;
        const int wave_phase = (wave_iter / EPI_PIPE_DEPTH) & 1;

        if (wave_iter >= EPI_PIPE_DEPTH)
          mbarrier_wait(tmem_empty_addr + wave_stage * 8, wave_phase ^ 1);

        tcgen05_fence_after();

        for (int k = 0; k < k_iters; ++k) {
          mbarrier_wait(full_mbar_addr + stage * 8, phase);
          tcgen05_fence_after();

          const int A_smem = smem_ptr + stage * copy_size;
          const int B_smem = A_smem + a_tile_bytes;

          for (int k1 = 0; k1 < BK / SWIZZLE_W; ++k1) {
            for (int k2 = 0; k2 < SWIZZLE_W / MMA_K_STEP; ++k2) {
              const int use_accum = (k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;

              // a_off: byte offset into A smem for K-step (k1, k2)
              // NN/NT: A smem is [BM × BK], K advances along columns
              //        → k2 * MMA_K_STEP columns = k2 * MMA_K_STEP * 2 bytes
              // TN:    A smem is [BK × BM], K advances along rows
              //        → k2 * MMA_K_STEP rows = k2 * MMA_K_STEP * BM * 2 bytes
              int a_off, b_off;
              if constexpr (Layout == Sm103Layout::TN) {
                // TN A smem is CHUNK-MAJOR after TMA+128B swizzle:
                // per +K step = +SWIZZLE_W (NOT BM). Ported from
                // fixed TN _cluster.cu (2026-07-02).
                a_off = k1 * SWIZZLE_W * SWIZZLE_W * 2 +
                        k2 * MMA_K_STEP * SWIZZLE_W * 2;
              } else {
                a_off = k1 * SWIZZLE_W * BM * 2 + k2 * MMA_K_STEP * 2;
              }

              // b_off depends on B's STORAGE layout, not the GEMM name:
              //   NN & TN both store B as [K,N] → identical chunk-major smem →
              //     identical b_off (SWIZZLE_W × SWIZZLE_W form).
              //   NT stores B as [N,K] → different (k2 term is bare).
              // Validated 2026-07-02: TN 128×256 passes with the SW×SW form;
              // NN must match (same B layout). The legacy SW*(BN/CLUSTER) form
              // only equalled SW×SW at BN=128 by coincidence, giving garbage at
              // BN=256 — that was a bad over-split; corrected here.
              if constexpr (Layout == Sm103Layout::NT) {
                b_off = k1 * SWIZZLE_W * (BN / CLUSTER_SIZE) * 2 +
                        k2 * MMA_K_STEP * 2;
              } else { // NN and TN — B stored [K,N]
                b_off = k1 * SWIZZLE_W * SWIZZLE_W * 2 +
                        k2 * MMA_K_STEP * SWIZZLE_W * 2;
              }

              // Descriptor LBO/SBO is chosen PER-OPERAND by that operand's
              // storage layout, not by the whole GEMM layout:
              //   A is stored [K,M] (chunk-major, needs LBO) only when trans_a
              //     -> TN. NN/NT keep A as [M,K] (1-arg, LBO=0).
              //   B is stored [K,N] (chunk-major, needs LBO) when trans_b
              //     -> NN and TN. NT keeps B as [N,K] (1-arg, LBO=0).
              // The LBO (=BK*SWIZZLE_W*2) only matters once the operand spans
              // more than one 64-col swizzle chunk (BM=128 or BN/2=128 at
              // BN=256). At BN=128, B is a single chunk so LBO is unused and
              // the 1-arg form coincidentally worked -- which is exactly why
              // NN 128x128 passed but NN 128x256 produced garbage.
              if constexpr (SWAP_AB) {
                // C' = C^T (1x2_plan.md R7d): MMA-A' = the B smem slot
                // (192-wide, M' side, composite 128+64 atoms; 2nd atom = rows
                // 128-191 = swizzle chunk 2 -> +2*LBO = +16384), MMA-B' = the
                // A smem slot (BM-wide, N' side). Both K-major TN descriptors.
                // d-tmem: M128 atom at col wave_stage*BM (2 stages, cols
                // [0,2*BM)); M64 atom in the SHARED col region [2*BM,3*BM)
                // with lane alignment 16*wave_stage (Layout F interleave,
                // plan Q5). One commit tracks both MMAs.
                const uint64_t ap_desc = make_smem_desc(
                    B_smem + b_off, BK * SWIZZLE_W * 2, 8 * SWIZZLE_W * 2);
                const uint64_t ap_desc2 =
                    make_smem_desc(B_smem + b_off + 16384, BK * SWIZZLE_W * 2,
                                   8 * SWIZZLE_W * 2);
                // B' (the BM-wide operand): 64B-swizzled when BM%64!=0 —
                // per-K-row pitch 64B, LBO=4096, SBO=512 (R7d); else the
                // validated 128B form (pitch 128B == a_off).
                const int bp_off =
                    A_SW64 ? ((k1 * SWIZZLE_W + k2 * MMA_K_STEP) * 64) : a_off;
                const uint64_t bp_desc =
                    A_SW64 ? make_smem_desc_sw64(A_smem + bp_off, BK * 32 * 2,
                                                 8 * 32 * 2)
                           : make_smem_desc(A_smem + bp_off, BK * SWIZZLE_W * 2,
                                            8 * SWIZZLE_W * 2);
                constexpr uint32_t idesc_m128 =
                    sm103_make_i_desc_swap<Config, 128>();
                constexpr uint32_t idesc_m64 =
                    sm103_make_i_desc_swap<Config, 64>();
                tcgen05_mma_bf16<CLUSTER_SIZE>(tmem_addr + wave_stage * BM,
                                               ap_desc, bp_desc, idesc_m128,
                                               use_accum);
                tcgen05_mma_bf16<CLUSTER_SIZE>(
                    tmem_addr + 2 * BM + ((16 * wave_stage) << 16), ap_desc2,
                    bp_desc, idesc_m64, use_accum);
              } else {
                uint64_t a_desc, b_desc;
                if constexpr (Layout == Sm103Layout::TN) {
                  a_desc = make_smem_desc(A_smem + a_off, BK * SWIZZLE_W * 2,
                                          8 * SWIZZLE_W * 2);
                } else {
                  a_desc = make_smem_desc(A_smem + a_off);
                }
                if constexpr (Layout == Sm103Layout::NT) {
                  b_desc = make_smem_desc(B_smem + b_off);
                } else {
                  // NN and TN: B stored [K,N], needs LBO=BK*SWIZZLE_W*2.
                  b_desc = make_smem_desc(B_smem + b_off, BK * SWIZZLE_W * 2,
                                          8 * SWIZZLE_W * 2);
                }
                tcgen05_mma_bf16<CLUSTER_SIZE>(tmem_addr + wave_stage * BN,
                                               a_desc, b_desc, i_desc,
                                               use_accum);
              }
            }
          }

          tcgen05_commit_multicast<CLUSTER_SIZE>(empty_mbar_addr + stage * 8,
                                                 operand_empty_mask);

          stage = (stage + 1) % LOAD_PIPE_DEPTH;
          phase ^= (stage == 0);
        }

        tcgen05_commit_multicast<CLUSTER_SIZE>(tmem_full_addr + wave_stage * 8,
                                               pair_mask);
        wave_iter++;
      }

      mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
      uint32_t is_valid, new_ctaid;
      clc_query_response(clc_response_addr + clc_stage * 16, is_valid,
                         new_ctaid);
      mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
      clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
      if (clc_stage == 0)
        clc_full_phase ^= 1;

      if (!is_valid)
        break;
    }
  }
  // ── scheduler warp (identical for all layouts and configs) ───────────────
  // PROTOCOL: exactly ONE clusterlaunchcontrol.try_cancel outstanding at any
  // time (CUTLASS sm100 scheduler, sm100_gemm_tma_warpspecialized.hpp:689-721
  // — issue→consume strictly 1:1). We previously armed CLC_PIPE_DEPTH (2)
  // up-front and kept 2 in flight; the two in-flight cancels can execute out
  // of order at END-OF-GRID: the LATER one steals the last cluster (valid)
  // while the EARLIER returns invalid. Consuming stages in order hits the
  // invalid first → every warp breaks → the stolen cluster's response is
  // orphaned and its tile NEVER computed → zero-filled C tiles. Seen as flaky
  // LM-head failures (19k-tile grids: thousands of steals still landing right
  // at exhaustion; small grids orphan only invalid responses = harmless).
  // With one outstanding, an invalid response ⇒ nothing else in flight ⇒ the
  // plain `break` in every warp role stays safe. §30.12.
  else if (warp_id == SCHEDULER_WARP && elect_sync()) {
    if (cta_rank == 0) {
      // Arm ONLY stage 0. try_cancel uses multicast::cluster::all → the
      // response broadcasts to EVERY CTA of the cluster, each completing its
      // own clc_full, so expect_tx=16 is armed on all CLUSTER_CTAS CTAs.
      mbarrier_arrive_expect_cluster(clc_full_mbar_addr, 16);
      for (int r = 1; r < cluster_ctas; ++r) { // GRANTED size (mapa outside
        int remote_full = map_smem_addr_to_cta_rank(clc_full_mbar_addr, r);
        mbarrier_arrive_expect_cluster(remote_full, 16); // granted cluster=UB)
      }
      clc_try_cancel(clc_response_addr, clc_full_mbar_addr);
    }

    int clc_stage = 0, clc_full_phase = 0, clc_empty_phase = 0;
    // Next arm slot index; slot = arm_idx % CLC_PIPE_DEPTH (equals clc_stage
    // after the consume below — kept separate for clarity). Slots
    // 1..CLC_PIPE_DEPTH-1 get armed for the FIRST time inside the loop, so
    // their clc_empty has never completed — skip the empty wait for those.
    int arm_idx = 1;

    while (true) {
      mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
      uint32_t is_valid, new_ctaid;
      clc_query_response(clc_response_addr + clc_stage * 16, is_valid,
                         new_ctaid);
      mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);

      clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
      if (clc_stage == 0)
        clc_full_phase ^= 1;

      if (!is_valid)
        break; // one-outstanding ⇒ no orphaned steal possible

      if (cta_rank == 0) {
        const int s2 = arm_idx % CLC_PIPE_DEPTH;
        if (arm_idx >= CLC_PIPE_DEPTH) {
          // Slot reuse: wait until ALL warps of ALL CTAs have consumed the
          // response this slot held CLC_PIPE_DEPTH responses ago.
          mbarrier_wait(clc_empty_mbar_addr + s2 * 8, clc_empty_phase);
          if (s2 == CLC_PIPE_DEPTH - 1)
            clc_empty_phase ^= 1;
        }
        mbarrier_arrive_expect_cluster(clc_full_mbar_addr + s2 * 8, 16);
        for (int r = 1; r < cluster_ctas; ++r) { // GRANTED size
          int remote_full =
              map_smem_addr_to_cta_rank(clc_full_mbar_addr + s2 * 8, r);
          mbarrier_arrive_expect_cluster(remote_full, 16);
        }
        clc_try_cancel(clc_response_addr + s2 * 16,
                       clc_full_mbar_addr + s2 * 8);
        ++arm_idx;
      }
    }
  }
  // ── epilogue warps: SWAP_AB transposed path ──────────────────────────────
  // TMEM holds C' = C^T: M128 atom lanes = C-cols n_local 0..127 (cols = C-row
  // m, [wave*BM,+BM)); M64 atom = C-cols 128..191 in Layout-F strips (shared
  // col region [2*BM,3*BM), lane alignment 16*wave_stage). Warp w owns
  // n_local 32w..32w+31 (M128) + strip 128+16w..128+16w+15 (M64). Writes are
  // TRANSPOSED into the store buffer [BM m-rows x STORE_COLS n-cols]; the C
  // TMA store is unchanged. Scalar smem writes (bring-up; optimize later).
  else if (SWAP_AB && warp_id < NUM_EPI_WARPS) {
    constexpr int EPI_BAR = 7;
    constexpr int EPI_THREADS = NUM_EPI_WARPS * WARP_THREADS;
    constexpr int num_chunks = BN / STORE_COLS; // 3 chunks of 64 C-cols

    int n_cluster, row_tile;
    decode_mn_tile<VerticalWave>(mn_tile, grid_n, grid_m, n_cluster, row_tile);
    int block_row = row_tile * CLUSTER_SIZE + (int)cta_m;
    int block_col = n_cluster * CLUSTER_N + (int)cta_n;
    int cur_batch = batch;

    int wave_iter = 0;
    int clc_stage = 0, clc_full_phase = 0;

    while (true) {
      const int wave_stage = wave_iter % EPI_PIPE_DEPTH;
      const int wave_phase = (wave_iter / EPI_PIPE_DEPTH) & 1;

      mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
      tcgen05_fence_after();

      const int t_m128 =
          tmem_addr + wave_stage * BM + ((warp_id * WARP_THREADS) << 16);
      const int t_m64 = tmem_addr + 2 * BM +
                        ((warp_id * WARP_THREADS + 16 * wave_stage) << 16);
      const int n128 = warp_id * WARP_THREADS + lane_id; // my M128 C-col
      int store_stage = 0;

      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (warp_id == 0)
          tma_store_wait<STORE_PIPE_DEPTH - 1>();
        barrier_sync(EPI_BAR, EPI_THREADS); // buffer free for all writers

        const int n_base = chunk * STORE_COLS;
        __nv_bfloat16 *buf = store_base + store_stage * BM * STORE_COLS;

        if (n_base < 128) {
          // M128-atom chunks (C-cols 0..127): warps whose n128 falls in this
          // chunk read all BM m-values (8 per ld) and write transposed.
          if (n128 >= n_base && n128 < n_base + STORE_COLS) {
            for (int m0 = 0; m0 < BM; m0 += 8) {
              float tmp[8];
              tcgen05_ld(tmp, t_m128 + m0);
              tcgen05_wait_ld();
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                float v = tmp[i] * alpha;
                if (beta != 0.0f) {
                  const long long cr =
                      (long long)(cur_batch * M + block_row * BM + m0 + i) * N +
                      (block_col * BN + n128);
                  v += beta * __bfloat162float(C_in[cr]);
                }
                // C_tmap is 128B-swizzled: 16B group g of buffer row r is
                // fetched from g ^ (r&7) — pre-swizzle the write (P5 bug fix,
                // 1x2_plan.md "P5 BUG ANALYSIS").
                {
                  const int r = m0 + i, off = n128 - n_base;
                  const int soff = ((((off >> 3) ^ (r & 7)) << 3) | (off & 7));
                  buf[r * STORE_COLS + soff] = __float2bfloat16(v);
                }
              }
            }
          }
        } else {
          // M64-atom chunk (C-cols 128..191): every warp reads its strip via
          // 16x256b.x1 (4 regs: (t/4, 2c),(t/4,2c+1),(t/4+8,2c),(t/4+8,2c+1)).
          const int n0 = 128 + 16 * warp_id + (lane_id / 4); // strip row A
          const int n1 = n0 + 8;                             // strip row B
          const int mc = 2 * (lane_id % 4);                  // col pair
          for (int m0 = 0; m0 < BM; m0 += 8) {
            float tmp[4];
            tcgen05_ld_16x256b(tmp, t_m64 + m0);
            tcgen05_wait_ld();
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int n_loc = (j < 2) ? n0 : n1;
              const int m = m0 + mc + (j & 1);
              float v = tmp[j] * alpha;
              if (beta != 0.0f) {
                const long long cr =
                    (long long)(cur_batch * M + block_row * BM + m) * N +
                    (block_col * BN + n_loc);
                v += beta * __bfloat162float(C_in[cr]);
              }
              {
                const int off = n_loc - n_base; // swizzled write (see above)
                const int soff = ((((off >> 3) ^ (m & 7)) << 3) | (off & 7));
                buf[m * STORE_COLS + soff] = __float2bfloat16(v);
              }
            }
          }
        }

        // Signal tmem free after the LAST chunk's reads are done.
        if (chunk == num_chunks - 1) {
          tcgen05_fence_before();
          if (elect_sync())
            mbarrier_arrive_cluster(tmem_empty_addr + wave_stage * 8);
        }

        __syncwarp();
        tma_store_fence();
        barrier_sync(EPI_BAR, EPI_THREADS);

        if (warp_id == 0 && elect_sync()) {
          const int src = store_smem + store_stage * store_buf_sz;
          const int c_col = block_col * BN + chunk * STORE_COLS;
          const int c_row = cur_batch * M + block_row * BM;
          tma_2d_smem2gmem(src, &C_tmap, c_col, c_row);
          tma_store_commit();
        }
        store_stage ^= 1;
      }

      wave_iter++;

      mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
      uint32_t is_valid, new_ctaid;
      clc_query_response(clc_response_addr + clc_stage * 16, is_valid,
                         new_ctaid);
      if (elect_sync())
        mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
      clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
      if (clc_stage == 0)
        clc_full_phase ^= 1;

      if (!is_valid)
        break;

      const int new_tile_idx = (int)(new_ctaid / CLUSTER_CTAS);
      cur_batch = new_tile_idx / total_mn_tiles;
      const int new_mn = new_tile_idx % total_mn_tiles;
      int row_tile2, n_cluster2;
      decode_mn_tile<VerticalWave>(new_mn, grid_n, grid_m, n_cluster2,
                                   row_tile2);
      block_row = row_tile2 * CLUSTER_SIZE + (int)cta_m;
      block_col =
          n_cluster2 * CLUSTER_N + (int)(((new_ctaid % (uint32_t)CLUSTER_CTAS) /
                                          (uint32_t)CLUSTER_SIZE) +
                                         cta_n_grant);
    }

    if (warp_id == 0)
      tma_store_wait<0>();
    barrier_sync(EPI_BAR, EPI_THREADS);
  }
  // ── epilogue warps (identical for all layouts and configs) ───────────────
  // Reads TMEM → scales by alpha/beta → writes to smem → TMA stores to C.
  else if (warp_id < NUM_EPI_WARPS) {
    constexpr int EPI_BAR = 7;
    constexpr int EPI_THREADS = NUM_EPI_WARPS * WARP_THREADS;
    constexpr int num_chunks = BN / STORE_COLS;
    constexpr int LOADS_PER =
        STORE_COLS / 8; // 8 = tcgen05_ld loads 8 floats per call

    int n_cluster, row_tile;
    decode_mn_tile<VerticalWave>(mn_tile, grid_n, grid_m, n_cluster, row_tile);
    int block_row = row_tile * CLUSTER_SIZE + (int)cta_m;
    int block_col = n_cluster * CLUSTER_N + (int)cta_n;
    int cur_batch = batch;

    int wave_iter = 0;
    int clc_stage = 0, clc_full_phase = 0;

    while (true) {
      const int wave_stage = wave_iter % EPI_PIPE_DEPTH;
      const int wave_phase = (wave_iter / EPI_PIPE_DEPTH) & 1;

      mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
      tcgen05_fence_after();

      // TMEM is 128 DP (datapaths/lanes) × 512 col, PER CTA and LOCAL. A
      // cta_group::2 MMA splits total M=2*BM cleanly: CTA0 → rows [0,BM) in
      // CTA0's TMEM DP [0,BM); CTA1 → rows [BM,2*BM) in CTA1's OWN TMEM DP
      // [0,BM). So each CTA reads its own accumulator at DP base 0 — there is
      // NO cross-CTA lane offset (a CTA cannot address the peer's TMEM via a DP
      // index; only 128 DP exist). Verified vs CUTLASS tmem_allocator_sm100.hpp
      // /mma_traits_sm100.hpp + DeepGEMM sm100_store_cd.cuh (2026-07-03,
      // §28.7). NOTE: a prior `cta_rank*BM<<16` term was spurious — at BM=128
      // it lands on DP 128 (out of the 7-bit DP range) and is silently ignored,
      // so it only *accidentally* worked; at BM<128 it corrupts the read.
      // Removed. (The tcgen05.ld auto-distributes 32 DP per warp; the warp term
      // is a no-op the hardware ignores but is kept to match the validated
      // form.)
      const int tmem_row =
          (tmem_addr + wave_stage * BN) + ((warp_id * WARP_THREADS) << 16);
      const int row = warp_id * WARP_THREADS + lane_id;
      int store_stage = 0;

      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        if (warp_id == 0)
          tma_store_wait<STORE_PIPE_DEPTH - 1>();

        float tmp[LOADS_PER][8];
#pragma unroll
        for (int n = 0; n < LOADS_PER; ++n)
          tcgen05_ld(tmp[n], tmem_row + chunk * STORE_COLS + n * 8);
        tcgen05_wait_ld();

        if (chunk == num_chunks - 1) {
          tcgen05_fence_before();
          // Signal THIS pair's leader tmem_empty (rank0 for pair0, rank2 for
          // pair1). Odd partner reaches the leader via mapa; even leader uses
          // its local addr. (Was a bit-24 clear that always hit rank0 — wrong
          // for pair1 in a 2×2 cluster.)
          const int tmem_empty_leader =
              (cta_m == 0) ? (tmem_empty_addr + wave_stage * 8)
                           : map_smem_addr_to_cta_rank(
                                 tmem_empty_addr + wave_stage * 8, pair_leader);
          if (elect_sync())
            mbarrier_arrive_cluster(tmem_empty_leader);
        }

        barrier_sync(EPI_BAR, EPI_THREADS);

#pragma unroll
        for (int n = 0; n < LOADS_PER; ++n) {
          __nv_bfloat162 packed[4];

          if (beta != 0.0f) {
            const int c_col = block_col * BN + chunk * STORE_COLS + n * 8;
            const int c_row = cur_batch * M + block_row * BM + row;
            const __nv_bfloat16 *cp = C_in + (long long)c_row * N + c_col;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              __nv_bfloat162 old =
                  *reinterpret_cast<const __nv_bfloat162 *>(cp + i * 2);
              float2 old_f = __bfloat1622float2(old);
              float v0 = tmp[n][i * 2] * alpha + old_f.x * beta;
              float v1 = tmp[n][i * 2 + 1] * alpha + old_f.y * beta;
              packed[i] = __float22bfloat162_rn({v0, v1});
            }
          } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              packed[i] = __float22bfloat162_rn(
                  {tmp[n][i * 2] * alpha, tmp[n][i * 2 + 1] * alpha});
            }
          }

          // Swizzle n within the store-smem row to avoid bank conflicts
          const int swizzled_n = n ^ (row & 7);
          __nv_bfloat16 *wp = store_base + store_stage * BM * STORE_COLS +
                              row * STORE_COLS + swizzled_n * 8;
          *reinterpret_cast<int4 *>(wp) = *reinterpret_cast<int4 *>(packed);
        }

        __syncwarp();
        tma_store_fence();
        barrier_sync(EPI_BAR, EPI_THREADS);

        if (warp_id == 0 && elect_sync()) {
          const int src = store_smem + store_stage * store_buf_sz;
          const int c_col = block_col * BN + chunk * STORE_COLS;
          const int c_row = cur_batch * M + block_row * BM;
          tma_2d_smem2gmem(src, &C_tmap, c_col, c_row);
          tma_store_commit();
        }

        store_stage ^= 1;
      }

      wave_iter++;

      mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
      uint32_t is_valid, new_ctaid;
      clc_query_response(clc_response_addr + clc_stage * 16, is_valid,
                         new_ctaid);
      if (elect_sync())
        mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
      clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
      if (clc_stage == 0)
        clc_full_phase ^= 1;

      if (!is_valid)
        break;

      const int new_tile_idx = (int)(new_ctaid / CLUSTER_CTAS);
      cur_batch = new_tile_idx / total_mn_tiles;
      const int new_mn = new_tile_idx % total_mn_tiles;
      int row_tile, n_cluster;
      decode_mn_tile<VerticalWave>(new_mn, grid_n, grid_m, n_cluster, row_tile);
      block_row = row_tile * CLUSTER_SIZE + (int)cta_m;
      // n_sub per STOLEN tile (requester-sized pieces, §31.3) — see producer.
      block_col =
          n_cluster * CLUSTER_N + (int)(((new_ctaid % (uint32_t)CLUSTER_CTAS) /
                                         (uint32_t)CLUSTER_SIZE) +
                                        cta_n_grant);
    }

    if (warp_id == 0)
      tma_store_wait<0>();
    barrier_sync(EPI_BAR, EPI_THREADS);
  }

  cluster_sync();
  if (warp_id == 0)
    dealloc_tmem<CLUSTER_SIZE>(tmem_addr, (BN * EPI_PIPE_DEPTH <= 32)    ? 32
                                          : (BN * EPI_PIPE_DEPTH <= 64)  ? 64
                                          : (BN * EPI_PIPE_DEPTH <= 128) ? 128
                                          : (BN * EPI_PIPE_DEPTH <= 256) ? 256
                                                                         : 512);
}

// ─── unified launcher
// ─────────────────────────────────────────────────────────
//
// Template args:
//   Config  — an Sm103ClusterConfig<BM, BN, BK, ...> instantiation
//   Layout  — Sm103Layout::NN / NT / TN
//
// Replaces the three separate launch_bgemm_{nn,nt,tn}_cluster functions.

template <typename Config, Sm103Layout Layout, bool VerticalWave = false,
          bool MulticastA = false, bool DynamicCluster = false>
static void launch_bgemm_sm103_cluster(const __nv_bfloat16 *A,
                                       const __nv_bfloat16 *B, __nv_bfloat16 *C,
                                       int M, int N, int K, int batchCount,
                                       float alpha, float beta,
                                       cudaStream_t stream) {
  // This cta_group::2 kernel only supports BM=128 (total MMA M=256), the clean
  // non-interleaved 128-DP accumulator layout the epilogue reads. BM=64
  // (total M=128) is a LEGAL cta_group::2 shape but uses a 2x2 INTERLEAVED TMEM
  // layout (half-N in DP[0,64), half in DP[64,128)) that this epilogue does not
  // de-interleave — it silently miscomputes. Smaller row-tiles must use a
  // cta_group::1 (M=64) variant. See analyzations.md §28.7.
  static_assert(Config::BM == 128 || (Config::SWAP_AB && Config::BM == 160),
                "BM=128 only (cta_group::2 epilogue, §28.7) — except the "
                "swap-AB 160x192 cta_group::1 tile (1x2_plan.md R7).");
  static int s_arch_ok = -1;
  if (s_arch_ok < 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    s_arch_ok = (prop.major == 10 && prop.minor >= 3) ? 1 : 0;
    if (!s_arch_ok)
      fprintf(stderr,
              "[BluBridge SM103] WARNING: device '%s' SM %d.%d, requires SM "
              "10.3+\n",
              prop.name, prop.major, prop.minor);
  }
  if (!s_arch_ok)
    return;

  constexpr int BM = Config::BM;
  constexpr int BN = Config::BN;
  constexpr int BK = Config::BK;
  constexpr int CLUSTER_SIZE = Config::CLUSTER_SIZE;
  constexpr int CLUSTER_N = Config::CLUSTER_N;
  constexpr int CLUSTER_CTAS = Config::CLUSTER_CTAS;
  constexpr int STORE_COLS = Config::STORE_COLS;

  // ── alignment preconditions (LOUD, never silent) ─────────────────────────
  // Our chunk-factorized 4D tmaps encode the sliced dim as {chunk, ., X/chunk}
  // — integer division TRUNCATES a non-multiple tail, and TMA then zero-fills
  // reads past it: the kernel would return silently-wrong zeros in the tail
  // rows/cols (found via the M=200 B2 test, §33.2). Reject loudly instead.
  // All 42 Qwen shapes satisfy these. Proper arbitrary-M support = true-extent
  // tmaps (DeepGEMM runtime_utils.hpp:199 style) or a fallback kernel route
  // (cuBLAS ships cutlass_75_wmma align1 for exactly this) — documented TODO.
  {
    constexpr int A_CHUNK =
        (Config::SWAP_AB && (Config::BM % 64 != 0)) ? 32 : 64;
    bool aligned = true;
    if constexpr (Layout == Sm103Layout::TN)
      aligned = (M % A_CHUNK == 0) && (N % 64 == 0);
    else if constexpr (Layout == Sm103Layout::NT)
      aligned = (K % 64 == 0);
    else // NN
      aligned = (K % 64 == 0) && (N % 64 == 0);
    if (!aligned) {
      fprintf(stderr,
              "[BluBridge SM103 Cluster] UNSUPPORTED alignment for this "
              "kernel (chunked-tmap sliced dims must be chunk multiples; "
              "TN: M%%%d && N%%64, NN: K%%64 && N%%64, NT: K%%64). "
              "M=%d N=%d K=%d — refusing to compute silently-wrong output.\n",
              A_CHUNK, M, N, K);
      return;
    }
  }

  // Build TMA descriptors — layout-specific A and B initialisation
  CUtensorMap A_tmap, B_tmap, C_tmap;

  if constexpr (Layout == Sm103Layout::TN) {
    // A[batch, K, M]  batch stride = K * M. The swap-AB 160-wide slice needs
    // the 64B-swizzle tmap (R7a); everything else keeps 128B.
    if constexpr (Config::SWAP_AB && (Config::BM % 64 != 0))
      init_4d_tma_A_KM_sw64(&A_tmap, A, M, K, BM, BK, (uint64_t)batchCount,
                            (uint64_t)K * M);
    else
      init_4d_tma_A_KM(&A_tmap, A, M, K, BM, BK, (uint64_t)batchCount,
                       (uint64_t)K * M, CU_TENSOR_MAP_SWIZZLE_128B);
    // B[batch, K, N]  batch stride = K * N
    init_4d_tma_B_KN(&B_tmap, B, K, N, BK, BN, (uint64_t)batchCount,
                     (uint64_t)K * N, CU_TENSOR_MAP_SWIZZLE_128B, CLUSTER_SIZE);
  } else if constexpr (Layout == Sm103Layout::NT) {
    // A[batch, M, K]  batch stride = M * K
    init_4d_tma_A_MK(&A_tmap, A, M, K, BM, BK, (uint64_t)batchCount,
                     (uint64_t)M * K, CU_TENSOR_MAP_SWIZZLE_128B);
    // B[batch, N, K]  batch stride = N * K  (stored transposed)
    init_4d_tma_B_NK(&B_tmap, B, N, K, BN, BK, (uint64_t)batchCount,
                     (uint64_t)N * K, CU_TENSOR_MAP_SWIZZLE_128B, CLUSTER_SIZE);
  } else { // NN
    // A[batch, M, K]  batch stride = M * K
    init_4d_tma_A_MK(&A_tmap, A, M, K, BM, BK, (uint64_t)batchCount,
                     (uint64_t)M * K, CU_TENSOR_MAP_SWIZZLE_128B);
    // B[batch, K, N]  batch stride = K * N
    init_4d_tma_B_KN(&B_tmap, B, K, N, BK, BN, (uint64_t)batchCount,
                     (uint64_t)K * N, CU_TENSOR_MAP_SWIZZLE_128B, CLUSTER_SIZE);
  }

  init_2d_tma_C(&C_tmap, C, (uint64_t)batchCount * M, (uint64_t)N, (uint32_t)BM,
                (uint32_t)STORE_COLS, CU_TENSOR_MAP_SWIZZLE_128B);

  // Ceil-div so shapes smaller than one cluster tile still launch (grid>=1).
  // TMA OOB-fill + clipped 2D store handle partial tiles. Floor-div gave
  // grid=0 → "invalid argument" for shapes < one tile. Ported from fixed
  // _cluster.cu (2026-07-02).
  const int grid_m = (M + CLUSTER_SIZE * BM - 1) / (CLUSTER_SIZE * BM);
  const int grid_n_blocks = (N + BN - 1) / BN; // total BN-blocks along N
  // Each cluster owns CLUSTER_N adjacent N-blocks → count N-CLUSTERS. grid_n
  // (passed to the kernel) is the N-cluster count; decode_mn_tile enumerates
  // clusters and cta_n picks the block within. For CLUSTER_N=1 this ==
  // N-blocks.
  const int grid_n = (grid_n_blocks + CLUSTER_N - 1) / CLUSTER_N;
  const int total_mn_tiles = grid_m * grid_n;

  auto kernel = bgemm_sm103_cluster_kernel<Config, Layout, VerticalWave,
                                           MulticastA, DynamicCluster>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       Config::SMEM_BYTES);

  // Cluster dims via LAUNCH ATTRIBUTES (one kernel serves rigid AND dynamic
  // launches; CUTLASS cluster_launch.hpp pattern, §31.1). Grid is sized in
  // PREFERRED-cluster units (CLUSTER_CTAS) — required divisibility for the
  // preferred-cluster launch, and the unit of the CTA-level tile decode.
  dim3 grid(total_mn_tiles * batchCount * CLUSTER_CTAS);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(Config::NUM_THREADS);
  cfg.dynamicSmemBytes = Config::SMEM_BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  if constexpr (DynamicCluster) {
#if CUDART_VERSION >= 12080
    // "2x2f": fallback = one MMA pair; preferred = the full CLUSTER_CTAS
    // cluster. HW grants per cluster; kernel adapts via %cluster_nctaid.
    attrs[0].val.clusterDim.x = (unsigned)Config::CLUSTER_SIZE; // fallback
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributePreferredClusterDimension;
    attrs[1].val.preferredClusterDim.x = (unsigned)Config::CLUSTER_CTAS;
    attrs[1].val.preferredClusterDim.y = 1;
    attrs[1].val.preferredClusterDim.z = 1;
    cfg.numAttrs = 2;
#else
    fprintf(stderr, "[BluBridge SM103 Cluster] DynamicCluster requires "
                    "CUDA >= 12.8 (preferred-cluster launch)\n");
    return;
#endif
  } else {
    attrs[0].val.clusterDim.x = (unsigned)Config::CLUSTER_CTAS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
  }

  cudaError_t lerr =
      cudaLaunchKernelEx(&cfg, kernel, A, C, C, M, N, K, total_mn_tiles, alpha,
                         beta, A_tmap, B_tmap, C_tmap, grid_n);
  if (lerr != cudaSuccess)
    fprintf(stderr, "[BluBridge SM103 Cluster] LaunchEx error: %s\n",
            cudaGetErrorString(lerr));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "[BluBridge SM103 Cluster] Launch error: %s\n",
            cudaGetErrorString(err));
}
