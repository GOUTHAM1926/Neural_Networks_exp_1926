// Generic (non-Blackwell) GQA forward path + the library-facing dispatch
// wrappers. This file is NOT Blackwell-marked, so the Makefile compiles it for
// the MACHINE arch (sm_86 / sm_90a / sm_103a). That is exactly why gqa_v2 lives
// here and NOT in GQA_fwd_sm103.cu: the fallback WMMA kernel must have an image
// for whatever GPU is actually running (H200/Ampere/etc.), whereas the SM103
// file is compiled only for compute_100a/103a.
//
// The Blackwell causal kernel (gqa_v48_causal, tcgen05/TMA) stays in
// GQA_fwd_sm103.cu and is reached here through the extern entry
// gqa_v48_causal_launch_d64(), taken only when get_arch()==Blackwell_B300.
//
// Since 2026-10-24 gqa_v2 also has a CAUSAL variant (segment-aware: dense rows or
// a GqaChatPlan's segments). Every causal call the dedicated kernels (Hopper v62,
// B300 v48) do not take runs it, on any GPU but a B300.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <stdexcept>

#include "ops/helpers/AttentionKernels.h"
#include "dtype/Types.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/helpers/KernelPolicy.h"
#include "device/CachingCudaAllocator.h"   // v2 design A: the plan's device lists
#include "device/DeviceCore.h"             // getCurrentStream(): the plan's upload
#include <algorithm>
#include <string>
#include <vector>
using namespace nvcuda;

namespace OwnTensor {
namespace cuda {

// Defined in GQA_fwd_sm103.cu (compiled for compute_103a). Only *called* when
// the running GPU is a B300, so a non-Blackwell build that lacks its device
// image never launches it.
void gqa_v48_causal_launch_d64(
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    __nv_bfloat16* dO, float* lse,
    int B, int Nq_heads, int Nkv_heads, int T, int G, float scale);

// Defined in GQA_fwd_sm90.cu (kernel v62, wgmma/TMA, compiled for the machine
// arch). Causal + head_dim 128 only, and the persistent grid is a fixed 132 CTAs
// sized for the H200, so it is taken ONLY when get_arch()==Hopper.
// v2 (2026-10-10): chat_units != nullptr (n_units entries) selects the per-segment variant (design A).
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units);

// v62 needs S % 256 == 0: total_units = B*Hq*(nTiles/4) with nTiles = S/BLOCK_M
// (=S/64), so the /4 pairing requires S/64 to be a multiple of 4.
static inline bool gqa_v62_shape_ok(int T, int hd) {
    return hd == 128 && (T % 256 == 0);
}

namespace {  // internal linkage: fallback kernel + launcher

// =============================================================================
// gqa_v2 -- portable WMMA kernel (sm_80+), head_dim 64/128.
//   CAUSAL = false: full attention over the whole row (the original kernel).
//   CAUSAL = true (added 2026-10-24): causal attention inside one segment. A block is
//     (segment, query head, 16-row query tile counted from the segment start). segs = the
//     plan's {b, t0, L, 0} list (GqaChatPlan::gen_segs), or nullptr for one segment per row
//     (b = blockIdx.x, t0 = 0, L = S: plain causal rows). The key loop stops at the tile's
//     last row, and a key after a row gets score -inf (p = 0). Rows and keys past the segment
//     end are read as zero and never written. Tiles are counted from the segment start, so a
//     chat in a packed row runs the same tiles in the same order as the chat alone at row 0.
// =============================================================================
template<int Br, int Bc, int D, bool CAUSAL>
__global__ void gqa_v2(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale,
  const int32_t *segs
){
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int lane   = threadIdx.x;

  int b = blockIdx.x, t0 = 0, L = S;
  if(CAUSAL && segs != nullptr){
    const int4 sg = reinterpret_cast<const int4*>(segs)[blockIdx.x];
    b = sg.x; t0 = sg.y; L = sg.z;
  }
  if(q_tile * Br >= L) return;                 // the grid covers the longest segment
  const int nq = min(Br, L - q_tile * Br);     // rows of this tile inside the segment (Br when !CAUSAL)

  const int q_row0   = t0 + q_tile * Br;
  const int nKVTiles = CAUSAL ? (q_tile * Br + nq - 1) / Bc + 1   // up to the tile's last row
                              : S / Bc;

  const long qBase  = ((long)(b * Hq  + hq)  * S  + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S  + t0) * D;
  const long lBase  = ((long)(b * Hq  + hq)  * S) + q_row0;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];   // scores for the current KV tile
  __shared__ __align__(16) float         sO[Br * D];    // running (unnormalized) output
  __shared__ __align__(16) float         sPV[Br * D];   // P@V for the current KV tile
  __shared__ float sm[Br];                              // running row max
  __shared__ float sl[Br];                              // running row denominator
  __shared__ float sCorr[Br];                           // per-row rescale factor exp(m_old - m_new)

  for(int i = lane; i < Br * D; i += 32){
    sQ[i] = (!CAUSAL || i / D < nq) ? d_Q[qBase + i] : zero;
    sO[i] = 0.0f;
  }
  if(lane < Br){
    sm[lane] = -INFINITY;
    sl[lane] = 0.0f;
  }
  __syncwarp();

  for(int kc = 0; kc < nKVTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i += 32){
      const bool in = !CAUSAL || kc * Bc + i / D < L;
      sK[i] = in ? d_K[kBase + i] : zero;
      sV[i] = in ? d_V[kBase + i] : zero;
    }
    __syncwarp();

    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

      for(int nt = 0; nt < Bc / 16; ++nt){
        wmma::fill_fragment(acc, 0.0f);
        for(int kt = 0; kt < D / 16; ++kt){
          wmma::load_matrix_sync(qf, sQ + kt * 16, D);
          wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
          wmma::mma_sync(acc, qf, kf, acc);
        }
        for(int t = 0; t < acc.num_elements; ++t){
          acc.x[t] *= scale;
        }
        wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
      }
    }
    __syncwarp();

    if(lane < Br){
      const float m_old = sm[lane];
      const float l_old = sl[lane];

      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j){
        // causal: a key after this row (this also covers every key past the segment end) -> -inf, p = 0.
        // Key 0 of the segment is in tile 0 for every row, so m stays finite.
        if(CAUSAL && kc * Bc + j > q_tile * Br + lane) sS[lane * Bc + j] = -INFINITY;
        tile_max = fmaxf(tile_max, sS[lane * Bc + j]);
      }

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);

      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        const float p = __expf(sS[lane * Bc + j] - m_new);
        sP[lane * Bc + j] = __float2bfloat16(p);
        p_sum += p;
      }

      sm[lane]    = m_new;
      sl[lane]    = l_old * corr + p_sum;
      sCorr[lane] = corr;
    }
    __syncwarp();

    for(int i = lane; i < Br * D; i += 32){
      sO[i] *= sCorr[i / D];
    }
    __syncwarp();

    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D/16];

      for(int nt = 0; nt < D/16; ++nt){
        wmma::fill_fragment(oacc[nt], 0.0f);
        for(int vt = 0; vt < Bc/16; ++vt){
          wmma::load_matrix_sync(pf, sP + vt * 16, Bc);
          wmma::load_matrix_sync(vf, sV + vt * 16 * D + nt * 16, D);
          wmma::mma_sync(oacc[nt], pf, vf, oacc[nt]);
        }
        wmma::store_matrix_sync(sPV + nt * 16, oacc[nt], D, wmma::mem_row_major);
      }
    }
    __syncwarp();

    for(int i = lane; i < Br * D; i += 32){
      sO[i] += sPV[i];
    }
    __syncwarp();
  }

  for(int i = lane; i < Br * D; i += 32){
    if(!CAUSAL || i / D < nq) d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  }
  if(lane < nq){
    d_LSE[lBase + lane] = sm[lane] + logf(sl[lane]);
  }
}

// CAUSAL: grid (segments, Hq, 16-row tiles of the longest segment); segs == nullptr -> one segment per
// row (B segments of length S). !CAUSAL: grid (B, Hq, S/16), the original launch.
template<int Br, int Bc, int D, bool CAUSAL = false>
void launch_gqa_v2(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale,
  const int32_t *segs = nullptr,
  int n_segs = 0,
  int max_len = 0,
  cudaStream_t stream = 0
){
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
  static_assert(Br == 16, "gqa_v2 processes exactly 16 query rows per block");
  static_assert(Bc % Br == 0, "a query tile must not straddle a key tile");

  dim3 GRID = CAUSAL ? dim3(segs ? n_segs : B, Hq, ((segs ? max_len : S) + Br - 1) / Br)
                     : dim3(B, Hq, S / Br);
  dim3 BLOCK(32);
  gqa_v2<Br, Bc, D, CAUSAL><<<GRID, BLOCK, 0, stream>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale, segs);
}

// Generic causal path (2026-10-24): every causal call the dedicated kernels (Hopper v62, B300 v48) do not
// take, on any GPU but a B300. chats == nullptr -> plain causal rows; else the plan's segment list.
void gqa_generic_causal_fwd(__nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
                            __nv_bfloat16* dO, float* lse,
                            int B, int Hq, int Hkv, int G, int T, int hd, float scale,
                            const GqaChatPlan* chats, const char* who)
{
    const int32_t* segs = chats ? chats->gen_segs : nullptr;
    const int n_segs    = chats ? chats->n_gen_segs : 0;
    const int max_len   = chats ? chats->gen_max_len : T;
    cudaStream_t st     = getCurrentStream();
    switch (hd) {
        case 64:
            launch_gqa_v2<16, 32, 64, true>(dQ, dK, dV, dO, lse, B, Hq, Hkv, G, T, scale,
                                            segs, n_segs, max_len, st);
            break;
        case 128:
            launch_gqa_v2<16, 32, 128, true>(dQ, dK, dV, dO, lse, B, Hq, Hkv, G, T, scale,
                                             segs, n_segs, max_len, st);
            break;
        default:
            throw std::runtime_error(std::string(who) + ": only head_dim 64 or 128 are compiled");
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Packed-QKV wrapper. Causal hd==64: B300 -> tcgen05 kernel, else -> gqa_v2.
// ---------------------------------------------------------------------------
void gqa_flash_attn_forward(
    const bfloat16_t* qkv,
    bfloat16_t*       output,
    float*            lse,
    int B, int Nq_heads, int Nkv_heads, int T, int hd,
    bool is_causal)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_forward: Nq_heads must be a positive multiple of Nkv_heads");
    }

    const int   G     = Nq_heads / Nkv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    __nv_bfloat16* base = const_cast<__nv_bfloat16*>(
        reinterpret_cast<const __nv_bfloat16*>(qkv));

    const long qElems = (long)B * Nq_heads  * T * hd;
    const long kElems = (long)B * Nkv_heads * T * hd;

    __nv_bfloat16* dQ = base;
    __nv_bfloat16* dK = base + qElems;
    __nv_bfloat16* dV = base + qElems + kElems;
    __nv_bfloat16* dO = reinterpret_cast<__nv_bfloat16*>(output);

    // Hopper (H100/H200) causal hd==128 -> v62 (wgmma/TMA persistent kernel).
    // BLUTRAIN_NO_SM90_GQA forces the generic causal path below.
    if (is_causal && !disable_sm90_gqa() &&
        get_arch() == ArchFamily::Hopper && gqa_v62_shape_ok(T, hd)) {
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   /*chat_units=*/nullptr, 0);
        return;
    }

    // Every other causal call: B300 has its own hd==64 kernel; any other GPU runs the
    // generic causal WMMA kernel (until 2026-10-24 it ran gqa_v2 without a mask, i.e.
    // full attention, for hd==64 and threw for hd==128).
    if (is_causal && get_arch() != ArchFamily::Blackwell_B300) {
        gqa_generic_causal_fwd(dQ, dK, dV, dO, lse, B, Nq_heads, Nkv_heads, G, T, hd, scale,
                               /*chats=*/nullptr, "gqa_flash_attn_forward");
        return;
    }

    if (is_causal && hd == 64) {
        if (T % 256 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_forward: causal path requires T to be a multiple "
                "of 256 (Br=Bc=128 tiles, two tiles processed per block)");
        }
        gqa_v48_causal_launch_d64(dQ, dK, dV, dO, lse,
                                  B, Nq_heads, Nkv_heads, T, G, scale);
        return;
    }

    if (is_causal) {
        throw std::runtime_error(
            "gqa_flash_attn_forward: causal attention is only implemented for "
            "head_dim 64");
    }

    if (T % 32 != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_forward: T must be a multiple of 32 (tile size)");
    }

    switch (hd) {
        case 64:
            launch_gqa_v2<16, 32, 64>(dQ, dK, dV, dO, lse,
                                      B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        case 128:
            launch_gqa_v2<16, 32, 128>(dQ, dK, dV, dO, lse,
                                       B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        default:
            throw std::runtime_error(
                "gqa_flash_attn_forward: only head_dim 64 or 128 are compiled");
    }
}

// ---------------------------------------------------------------------------
// Unpacked-QKV wrapper. Same dispatch rule as the packed wrapper above.
// ---------------------------------------------------------------------------
void gqa_flash_attn_forward_qkv(
    const bfloat16_t* q,
    const bfloat16_t* k,
    const bfloat16_t* v,
    bfloat16_t*       output,
    float*            lse,
    int B, int Nq_heads, int Nkv_heads, int T, int hd,
    bool is_causal, const GqaChatPlan* chats)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: Nq_heads must be a positive multiple of Nkv_heads");
    }

    const int   G     = Nq_heads / Nkv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    __nv_bfloat16* dQ = const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(q));
    __nv_bfloat16* dK = const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(k));
    __nv_bfloat16* dV = const_cast<__nv_bfloat16*>(reinterpret_cast<const __nv_bfloat16*>(v));
    __nv_bfloat16* dO = reinterpret_cast<__nv_bfloat16*>(output);

    const bool v62_ok = is_causal && !disable_sm90_gqa() &&
                        get_arch() == ArchFamily::Hopper && gqa_v62_shape_ok(T, hd);
    // Generic causal path (2026-10-24): every causal call v62 does not take, on any GPU but a B300.
    const bool generic_ok = is_causal && !v62_ok && get_arch() != ArchFamily::Blackwell_B300;

    // v2 packed rows (design A, a per-chat plan): the Hopper v62 kernel and the generic causal kernel
    // know the plan. Every other path below would IGNORE it and silently let the chats of a row see
    // each other, so a plan that cannot be honoured is refused here instead. A plan is also only valid
    // for the shape it was built for (its units hold row indices, head indices and segment bounds of
    // that shape).
    if (chats != nullptr) {
        if (!v62_ok && !generic_ok)
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: chats (packed rows, design A) needs a causal path that knows "
                "the plan: Hopper v62 or the generic causal kernel (any GPU but a B300)");
        if (chats->B != B || chats->S != T || chats->Hq != Nq_heads || chats->fwd_units == nullptr ||
            chats->gen_segs == nullptr)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: chats plan was built for another shape "
                                     "(B, T, Nq_heads) or is empty");
    }

    // Hopper (H100/H200) causal hd==128 -> v62 (wgmma/TMA persistent kernel).
    // This is the path bluscript's "split" attn_mode reaches.
    // BLUTRAIN_NO_SM90_GQA forces the generic path below; see the note on the
    // packed entry point above about hd==128 throwing instead of falling back.
    if (v62_ok) {
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0);
        return;
    }

    // Any GPU but a B300 (and Hopper calls v62 cannot take): the generic causal WMMA kernel, dense rows
    // or the plan's segments. Until 2026-10-24 hd==64 ran gqa_v2 without a mask (full attention) and
    // hd==128 threw.
    if (generic_ok) {
        gqa_generic_causal_fwd(dQ, dK, dV, dO, lse, B, Nq_heads, Nkv_heads, G, T, hd, scale, chats,
                               "gqa_flash_attn_forward_qkv");
        return;
    }

    if (is_causal && hd == 64) {   // B300
        if (T % 256 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: causal path requires T to be a multiple "
                "of 256 (Br=Bc=128 tiles, two tiles processed per block)");
        }
        gqa_v48_causal_launch_d64(dQ, dK, dV, dO, lse,
                                  B, Nq_heads, Nkv_heads, T, G, scale);
        return;
    }

    if (is_causal) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: causal attention is only implemented for "
            "head_dim 64");
    }

    if (T % 32 != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: T must be a multiple of 32 (tile size)");
    }

    switch (hd) {
        case 64:
            launch_gqa_v2<16, 32, 64>(dQ, dK, dV, dO, lse,
                                      B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        case 128:
            launch_gqa_v2<16, 32, 128>(dQ, dK, dV, dO, lse,
                                       B, Nq_heads, Nkv_heads, G, T, scale);
            break;
        default:
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: only head_dim 64 or 128 are compiled");
    }
}

// ---------------------------------------------------------------------------
// v2 design A: the per-segment plan (see GqaChatPlan in AttentionKernels.h).
// Forward units {b, t0, L, (cpl << 8) | hq}: per segment, the couples of its query pairs (pair cpl
// with pair nP-1-cpl, nP = ceil(L/128)) for every head -- the dense kernel's balancing applied
// inside the segment. Order: longest segment first (LPT, like FA3's varlen scheduler,
// hopper/tile_scheduler.hpp:493-700), and inside a segment head-major then couple, the dense order
// (consecutive units share K/V in L2). Backward tiles {b, t0, L, kt}: one per 64-key tile of each
// segment, most q-tiles to visit first (Vj1d).
// Vj2 (added 2026-10-15): blocks {b, t0, L, kt, hq, dqt0, 0, 0}, one per (128-key tile of a segment, query head),
// in the dense Vj2 grid's order -- sections of gqa_vj2_section_pairs(S) (batch, head) pairs, inside a section the
// blocks with the most 64-row steps first (FA3 SingleTileBwdLPTScheduler); for rows of ONE segment each this is
// exactly the dense order. dq tiles {b, row0, nrows, 0}: the padded 64-row slots of Vj2's LSE / D / fp32 dQ,
// ceil(L/64) per segment in (row, start) order, the same list for every head (FA3 varlen's offset_padded: a chat
// starts at any token, but those buffers need 64-row alignment); dqt0 = hq * n_dq_tiles + the segment's first
// slot + 2 kt. Generic causal kernels (added 2026-10-24, any GPU but Hopper and B300): segments {b, t0, L, 0} in
// (row, start) order, FA2 varlen's cu_seqlens list; each kernel cuts its own tiles from t0. All five lists go up in
// ONE allocation with one asynchronous copy on the current stream (the generic list last, so the four Hopper lists
// keep their place): the plan is built once per micro-batch, and a synchronous copy would first wait for the GPU
// to finish the previous one. The kernels only read the lists.
// ---------------------------------------------------------------------------
int gqa_vj2_section_pairs(int S);   // GQA_bwd_vj2_sm90.cu

void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq)
{
    if (B <= 0 || S <= 0 || Hq <= 0 || Hq > 255 || n_seg <= 0 || !seg_b || !seg_t0 || !seg_len)
        throw std::runtime_error("gqa_chat_plan_build: bad shape or empty segment list (Hq must be 1..255)");
    struct Seg { int b, t0, L; };
    std::vector<Seg> sg((size_t)n_seg);
    for (int i = 0; i < n_seg; ++i) {
        sg[i] = {seg_b[i], seg_t0[i], seg_len[i]};
        if (sg[i].b < 0 || sg[i].b >= B || sg[i].t0 < 0 || sg[i].L <= 0 || (long long)sg[i].t0 + sg[i].L > S)
            throw std::runtime_error("gqa_chat_plan_build: segment " + std::to_string(i) + " is outside its row");
    }
    // Every token of every row must belong to exactly one segment: a row the plan does not cover
    // would keep unwritten (garbage) outputs that the next layers read.
    std::vector<Seg> byrow = sg;
    std::sort(byrow.begin(), byrow.end(), [](const Seg& a, const Seg& c) { return a.b != c.b ? a.b < c.b : a.t0 < c.t0; });
    for (int b = 0, k = 0; b < B; ++b) {
        int t = 0;
        while (k < n_seg && byrow[k].b == b) {
            if (byrow[k].t0 != t) throw std::runtime_error("gqa_chat_plan_build: row " + std::to_string(b) + " has a gap or overlap at token " + std::to_string(t));
            t += byrow[k].L; ++k;
        }
        if (t != S) throw std::runtime_error("gqa_chat_plan_build: row " + std::to_string(b) + " is covered up to token " + std::to_string(t) + " of " + std::to_string(S));
    }
    std::vector<Seg> lpt = byrow;   // longest first; ties keep (row, start) order: deterministic
    std::stable_sort(lpt.begin(), lpt.end(), [](const Seg& a, const Seg& c) { return a.L > c.L; });
    std::vector<int32_t> fu, bt;
    for (const Seg& g : lpt) {
        const int nP = (g.L + 127) / 128, ncpl = (nP + 1) / 2;
        for (int hq = 0; hq < Hq; ++hq)
            for (int cpl = 0; cpl < ncpl; ++cpl) { fu.push_back(g.b); fu.push_back(g.t0); fu.push_back(g.L); fu.push_back((cpl << 8) | hq); }
    }
    struct KT { int b, t0, L, kt, work; };
    std::vector<KT> kts;
    for (const Seg& g : byrow) {
        const int nk = (g.L + 63) / 64;
        for (int kt = 0; kt < nk; ++kt) kts.push_back({g.b, g.t0, g.L, kt, nk - kt});
    }
    std::stable_sort(kts.begin(), kts.end(), [](const KT& a, const KT& c) { return a.work > c.work; });
    for (const KT& k : kts) { bt.push_back(k.b); bt.push_back(k.t0); bt.push_back(k.L); bt.push_back(k.kt); }

    // Vj2: the padded dQ slots (segments in (row, start) order) ...
    std::vector<int32_t> dq;
    std::vector<int> slot0(byrow.size()), rowfirst((size_t)B + 1, 0);
    int nd = 0;
    for (size_t j = 0; j < byrow.size(); ++j) {
        const Seg& g = byrow[j];
        slot0[j] = nd;
        for (int q = 0; 64 * q < g.L; ++q) {
            dq.push_back(g.b); dq.push_back(g.t0 + 64 * q); dq.push_back(std::min(64, g.L - 64 * q)); dq.push_back(0);
            ++nd;
        }
        rowfirst[(size_t)g.b + 1] = (int)j + 1;
    }
    for (int b = 0; b < B; ++b) rowfirst[(size_t)b + 1] = std::max(rowfirst[(size_t)b + 1], rowfirst[b]);
    // ... and the blocks, section by section
    struct VB { int b, t0, L, kt, hq, dqt0, work; };
    std::vector<int32_t> vb;
    std::vector<VB> sec;
    const int sw = gqa_vj2_section_pairs(S), HB = B * Hq;
    for (int h0 = 0; h0 < HB; h0 += sw) {
        sec.clear();
        for (int hb = h0; hb < std::min(HB, h0 + sw); ++hb) {
            const int b = hb / Hq, hq = hb % Hq;
            for (int j = rowfirst[b]; j < rowfirst[(size_t)b + 1]; ++j) {
                const Seg& g = byrow[j];
                const int nq = (g.L + 63) / 64;
                for (int kt = 0; 128 * kt < g.L; ++kt)
                    sec.push_back({g.b, g.t0, g.L, kt, hq, hq * nd + slot0[j] + 2 * kt, nq - 2 * kt});
            }
        }
        std::stable_sort(sec.begin(), sec.end(), [](const VB& a, const VB& c) { return a.work > c.work; });
        for (const VB& v : sec) for (int x : {v.b, v.t0, v.L, v.kt, v.hq, v.dqt0, 0, 0}) vb.push_back(x);
    }

    // Generic causal kernels: one entry per segment, (row, start) order.
    std::vector<int32_t> gs;
    int gmax = 0;
    for (const Seg& g : byrow) {
        for (int x : {g.b, g.t0, g.L, 0}) gs.push_back(x);
        gmax = std::max(gmax, g.L);
    }

    std::vector<int32_t> all;   // every list is a multiple of 4 ints, so each one starts 16-B aligned (int4 loads)
    all.reserve(fu.size() + bt.size() + vb.size() + dq.size() + gs.size());
    for (const std::vector<int32_t>* v : {&fu, &bt, &vb, &dq, &gs}) all.insert(all.end(), v->begin(), v->end());
    gqa_chat_plan_free(plan);
    auto& alloc = CachingCUDAAllocator::instance();
    int32_t* buf = static_cast<int32_t*>(alloc.allocate(all.size() * sizeof(int32_t)));
    // Pageable source: the copy has read it when the call returns, so `all` may go out of scope.
    if (cudaMemcpyAsync(buf, all.data(), all.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        getCurrentStream()) != cudaSuccess) {
        alloc.deallocate(buf);
        throw std::runtime_error("gqa_chat_plan_build: upload failed");
    }
    plan.storage      = buf;
    plan.fwd_units    = buf;
    plan.bwd_tiles    = plan.fwd_units + fu.size();
    plan.vj2_blocks   = plan.bwd_tiles + bt.size();
    plan.dq_tiles     = plan.vj2_blocks + vb.size();
    plan.n_fwd_units  = (int)(fu.size() / 4);
    plan.n_bwd_tiles  = (int)(bt.size() / 4);
    plan.n_vj2_blocks = (int)(vb.size() / 8);
    plan.n_dq_tiles   = nd;
    plan.gen_segs     = plan.dq_tiles + dq.size();
    plan.n_gen_segs   = (int)(gs.size() / 4);
    plan.gen_max_len  = gmax;
    plan.B = B; plan.S = S; plan.Hq = Hq;
}

void gqa_chat_plan_free(GqaChatPlan& plan)
{
    if (plan.storage) CachingCUDAAllocator::instance().deallocate(plan.storage);
    plan = GqaChatPlan{};
}

GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
                                  int n_seg, int B, int S, int Hq)
{
    std::unique_ptr<GqaChatPlan> p(new GqaChatPlan());
    gqa_chat_plan_build(*p, seg_b, seg_t0, seg_len, n_seg, B, S, Hq);
    return GqaChatPlanPtr(p.release(), [](const GqaChatPlan* q) {
        GqaChatPlan* m = const_cast<GqaChatPlan*>(q);
        gqa_chat_plan_free(*m);
        delete m;
    });
}

} // namespace cuda
} // namespace OwnTensor
