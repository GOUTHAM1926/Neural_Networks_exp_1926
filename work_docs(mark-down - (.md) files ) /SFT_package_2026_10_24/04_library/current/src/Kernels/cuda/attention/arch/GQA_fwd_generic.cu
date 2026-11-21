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
#include <cstdlib>   // getenv: BLU_GQA_MX (2026-11-13, mixed units)
#include <string>
#include <vector>
#include <cstdio>          // 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2): the order switch notice
#include <mutex>           // the side table
#include <unordered_map>
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
// 2026-11-13: mixed_units = the plan's list holds mixed units (GqaChatPlan::mx_hkv != 0): the launch then starts the
// kernel instantiation that knows them (MODE_CHATS_MX); false = the kernels of before.
// Old declaration (kept, not deleted):
// void gqa_v62_causal_launch_d128(
//     const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
//     __nv_bfloat16* O, float* LSE,
//     int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
//     const int32_t* chat_units, int n_units);
// win (2026-11-04): sliding window in keys (0 = plain causal), dense rows only; see MODE_WIN there.
// 2026-11-20 step 7 (packed rows + window): and packed rows (a chat plan), the window per chat; see MODE_CHATS_WIN / MODE_CHATS_MX_WIN there.
// Old declaration (kept, not deleted; 2026-11-20 SWA port of Navin's b913f32: int win appended):
// void gqa_v62_causal_launch_d128(..., const int32_t* chat_units, int n_units, bool mixed_units);
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units, bool mixed_units, int win);

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
        // 2026-11-13: the launch takes one more argument (mixed units); this entry has no plan, so false.
        // Old call (kept, not deleted):
        // gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
        //                            B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
        //                            /*chat_units=*/nullptr, 0);
        // Old call (kept, not deleted; 2026-11-20 SWA port of Navin's b913f32: this entry has no window, win = 0):
        // gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
        //                            B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
        //                            /*chat_units=*/nullptr, 0, /*mixed_units=*/false);
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   /*chat_units=*/nullptr, 0, /*mixed_units=*/false, /*win=*/0);
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
    // Old line (kept, not deleted; 2026-11-20 SWA port of Navin's b913f32: int window appended):
    //     bool is_causal, const GqaChatPlan* chats)
    bool is_causal, const GqaChatPlan* chats, int window)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: Nq_heads must be a positive multiple of Nkv_heads");
    }
    // Sliding window (2026-11-04): only the Hopper v62 kernel has it (MODE_WIN). A window at least as long as the
    // row is plain causal attention; a shorter one on any other path would silently be full causal, so it is refused.
    const bool windowed = window > 0 && window < T;
    if (window < 0)
        throw std::runtime_error("gqa_flash_attn_forward_qkv: window must be >= 0 (0 = no window)");

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
        // 2026-11-13 (mixed units): a mixed unit pairs two query heads of ONE KV head, so a plan that holds them is
        // right only for the KV-head count it was built for. With another count the two heads would read different
        // K / V and the result would be wrong without any error, so it is refused here, next to the shape check.
        if (chats->mx_hkv != 0 && chats->mx_hkv != Nkv_heads)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: the chats plan's mixed units were built for " +
                                     std::to_string(chats->mx_hkv) + " KV heads, this call has " +
                                     std::to_string(Nkv_heads));
    }

    if (windowed) {
        if (!v62_ok)
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: a sliding window needs the Hopper v62 path: causal, head_dim 128, "
                "T % 256 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset");
        // Old lines (kept, not deleted; 2026-11-20 step 7 (packed rows + window): v62 runs a window over packed rows now, per chat):
        // if (chats != nullptr)
        //     throw std::runtime_error("gqa_flash_attn_forward_qkv: a sliding window with packed rows (a chats plan) "
        //                              "is not implemented");
        if (window % 128 != 0)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: the window must be a multiple of 128 keys, got " +
                                     std::to_string(window));
    }
    // 2026-11-20 step 7, item A (FlashAttention's rule, commit 0842ec0 "Don't dispatch to local if window size >= seqlen_k", and
    // flash_api.cpp: window_size_left >= max_seqlen_k - 1 -> no window): a window at least as long as the longest segment
    // of the plan hides no key of any chat, so the kernels of before run (modes 1 / 3 and the plain backward copies).
    // The results are the same to the bit (for a chat of L <= W the window modes reduce to exactly the plain code: key
    // range from 0, no edge tile, the same step count and masks; test_swa_packed.cu checks it in a batch that does run the
    // window kernels). Only the KERNEL choice uses the batch: every check above is on the requested window and the same
    // for every batch, so a run never starts fine and stops later at its first long chat. The forward and the backward
    // decide from the same three values (window, T, the plan's gen_max_len), so they always agree. gen_max_len also counts
    // a row's pad tail (a segment like any other): a long pad tail keeps the window kernels, which is still correct.
    const bool win_kernels = windowed && !(chats != nullptr && window >= chats->gen_max_len);

    // Hopper (H100/H200) causal hd==128 -> v62 (wgmma/TMA persistent kernel).
    // This is the path bluscript's "split" attn_mode reaches.
    // BLUTRAIN_NO_SM90_GQA forces the generic path below; see the note on the
    // packed entry point above about hd==128 throwing instead of falling back.
    if (v62_ok) {
        // 2026-11-13: the launch is told whether the plan's list holds mixed units (it then starts MODE_CHATS_MX).
        // Old call (kept, not deleted):
        // gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
        //                            B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
        //                            chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0);
        // Old call (kept, not deleted; 2026-11-20 SWA port of Navin's b913f32: the window goes last, 0 when not windowed):
        // gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
        //                            B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
        //                            chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0,
        //                            /*mixed_units=*/chats != nullptr && chats->mx_hkv != 0);
        // Old call (kept, not deleted; 2026-11-20 step 7, item A: the window goes to the kernels only when it hides a key):
        // gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
        //                            B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
        //                            chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0,
        //                            /*mixed_units=*/chats != nullptr && chats->mx_hkv != 0,
        //                            windowed ? window : 0);
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0,
                                   /*mixed_units=*/chats != nullptr && chats->mx_hkv != 0,
                                   win_kernels ? window : 0);
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

// ---- 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2) (backward study) ----
// The per-KV-head Vj2 list goes up in the plan's ONE allocation, AFTER the generic list (every other list keeps
// its place). Its place, count and group size are kept HERE, keyed by the plan's vj2_blocks pointer, not in
// GqaChatPlan: the test program holds GqaChatPlan by value, so a new field would change the struct that already
// built programs know. gqa_chat_plan_build sets the entry, gqa_chat_plan_free removes it.
namespace {
struct Vj2HsumEntry { const int32_t* list; int n; int G; };
std::mutex& vj2_hsum_mu() { static std::mutex m; return m; }
std::unordered_map<const int32_t*, Vj2HsumEntry>& vj2_hsum_map() {
    static std::unordered_map<const int32_t*, Vj2HsumEntry> m;
    return m;
}
void vj2_hsum_set(const int32_t* key, const int32_t* list, int n, int G) {
    std::lock_guard<std::mutex> lk(vj2_hsum_mu());
    if (key != nullptr && n > 0) vj2_hsum_map()[key] = Vj2HsumEntry{list, n, G};
    else vj2_hsum_map().erase(key);
}
int vj2_hsum_sect() {   // block-order switch BLU_VJ2_HSUM_SECT: 0 = rule 1 (whole rows, ~80 MB), N = N pairs, -1 = one section
    static const int v = [] { const char* e = getenv("BLU_VJ2_HSUM_SECT"); const int x = e ? atoi(e) : 0;
        if (x) fprintf(stderr, "[vj2] head-sum block order: BLU_VJ2_HSUM_SECT=%d\n", x); return x; }();
    return v;
}
}  // namespace
bool gqa_vj2_hsum_list(const int32_t* vj2_blocks, const int32_t** list, int* n, int* G) {
    std::lock_guard<std::mutex> lk(vj2_hsum_mu());
    auto it = vj2_hsum_map().find(vj2_blocks);
    if (it == vj2_hsum_map().end()) return false;
    *list = it->second.list; *n = it->second.n; *G = it->second.G;
    return true;
}

// 2026-11-13: one more argument, Hkv (the KV-head count; 0 = the plan of before). See the forward-unit loop below.
// Old signature (kept, not deleted):
// void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
//                          const int32_t* seg_len, int n_seg, int B, int S, int Hq)
void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq, int Hkv)
{
    if (B <= 0 || S <= 0 || Hq <= 0 || Hq > 255 || n_seg <= 0 || !seg_b || !seg_t0 || !seg_len)
        throw std::runtime_error("gqa_chat_plan_build: bad shape or empty segment list (Hq must be 1..255)");
    if (Hkv < 0 || (Hkv > 0 && Hq % Hkv != 0))
        throw std::runtime_error("gqa_chat_plan_build: Hkv must be 0 (no mixed units) or a divisor of Hq");
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
    // ---- forward units. 2026-11-13: mixed units ("MX"; work doc 45.194, GqaChatPlan::mx_hkv in the header) ----
    // HOW IT WAS (the loop kept below as comments): for every segment and every query head, the couples of its
    // 128-row pairs. A pair is two 64-row query tiles, one for each compute warpgroup. For a segment with an ODD number
    // of 64-row tiles nQ = ceil(L/64), the second tile of the last pair holds no row: that group only waits.
    // WHAT IS NEW: with G = Hq / Hkv >= 2 query heads per KV head, such a segment gets floor(G/2) MIXED units per KV
    // head. A mixed unit gives the last tile (index nQ - 1) of two heads of that KV head to the two groups of one
    // pair. Those two heads' plain units then cover one pair less (bit 30 of the unit word). With an odd G one head is
    // left over and keeps its units of before; a segment with an even nQ keeps all its units of before.
    // ORDER: as before (longest segment first, head-major); a mixed unit comes right after the plain units of its two
    // heads, so consecutive units still share K / V in L2.
    // SWITCHES: Hkv == 0 (the default of the declaration) or G < 2 or BLU_GQA_MX=0 -> exactly the list of before; and
    // the 2% rule below: a plan in which fewer than 2% of the units would share a tile also gets the list of before.
    // MEASURED (H200, 8 rows x 4,096, heads 18/6, forward, 2026-11-13): chats of 32-160: 128-row pairs 7,974 -> 6,798,
    // time -8.0%; chats of 300-1,500: 5,040 -> 4,878, -2.7%; real rows: -0.4..-1.05%; one chat per row: no mixed unit.
    // Old code (kept, not deleted):
    // for (const Seg& g : lpt) {
    //     const int nP = (g.L + 127) / 128, ncpl = (nP + 1) / 2;
    //     for (int hq = 0; hq < Hq; ++hq)
    //         for (int cpl = 0; cpl < ncpl; ++cpl) { fu.push_back(g.b); fu.push_back(g.t0); fu.push_back(g.L); fu.push_back((cpl << 8) | hq); }
    // }
    static const bool mx_env_off = [] { const char* e = std::getenv("BLU_GQA_MX"); return e != nullptr && e[0] == '0'; }();
    int mxG = (Hkv > 0 && !mx_env_off) ? Hq / Hkv : 1;   // query heads per KV head that may share; 1 = no mixing
    // THE 2% RULE (2026-11-13, the chat-length sweep; work doc 45.194). One mixed unit sends the whole plan to the
    // mixed-unit kernel (MODE_CHATS_MX), and its longer unit decode costs 0.5-1.1% of the forward on rows with many
    // units. So mixing pays only when enough tiles are shared. Measured (H200, 8 rows x 4,096, heads 18/6; forward, the
    // mixed plan against the plain one), by "mixed units as a share of the plain list's units":
    //   0.78% (rows of 96-token chats + one 64-token leftover): +0.92%, slower in 3 of 3 rounds
    //   1.75% (rows of 224-token chats + one 64-token leftover): +0.16%, not significant
    //   2.11% (real SFT rows): -0.52%;  3.9%: -1.09%;  5.5-7%: -1.3..-3.0%;  11-18%: -3.9..-8.0%;  33%: -21..-26%
    // So the list is mixed only when that share is at least mx_min_pct (2); below it the list of before is built and
    // the forward runs the kernel of before. BLU_GQA_MX_MIN_PCT=<n> changes the limit for a process (0 = always mix:
    // the test suite uses it so that every small layout still runs the mixed-unit kernel).
    static const int mx_min_pct = [] { const char* e = std::getenv("BLU_GQA_MX_MIN_PCT"); return e != nullptr ? std::atoi(e) : 2; }();
    if (mxG >= 2) {
        long long units_plain = 0, would_mix = 0;                  // counted first, from the segment lengths alone
        for (const Seg& g : lpt) {
            const int nP = (g.L + 127) / 128, nQ = (g.L + 63) / 64;
            units_plain += (long long)Hq * ((nP + 1) / 2);
            if (nQ & 1) would_mix += (long long)(Hq / mxG) * (mxG / 2);
        }
        if (would_mix * 100 < (long long)mx_min_pct * units_plain) mxG = 1;   // too few shared tiles: the list of before
    }
    long long n_mixed = 0;
    for (const Seg& g : lpt) {
        const int nP = (g.L + 127) / 128, nQ = (g.L + 63) / 64;
        const int nmix = (mxG >= 2 && (nQ & 1)) ? mxG / 2 : 0;   // mixed units per KV head for this segment
        auto plain = [&](int hq, int drop) {                      // the couples of one head; drop = 1: without its last pair
            const int np = nP - drop, ncpl = (np + 1) / 2;
            for (int cpl = 0; cpl < ncpl; ++cpl) { fu.push_back(g.b); fu.push_back(g.t0); fu.push_back(g.L); fu.push_back((drop << 30) | (cpl << 8) | hq); }
        };
        for (int h0 = 0; h0 < Hq; h0 += mxG) {                   // one KV head's query heads: h0 .. h0 + mxG - 1
            for (int m = 0; m < nmix; ++m) {
                const int ha = h0 + 2 * m, hb = ha + 1;
                plain(ha, 1); plain(hb, 1);
                fu.push_back(g.b); fu.push_back(g.t0); fu.push_back(g.L);
                fu.push_back((int32_t)(0x80000000u | ((uint32_t)hb << 8) | (uint32_t)ha));
                ++n_mixed;
            }
            for (int hq = h0 + 2 * nmix; hq < h0 + mxG; ++hq) plain(hq, 0);
        }
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
            dq.push_back(g.b); dq.push_back(g.t0 + 64 * q); dq.push_back(std::min(64, g.L - 64 * q));
            // 2026-11-14 idea 2 (direct dQ): 1 = this tile gets dQ from one block only (the chat's first two tiles). Old: dq.push_back(0);
            dq.push_back(q < 2 ? 1 : 0);
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
    // 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2): the same blocks per (128-key tile, KV HEAD) when the plan knows Hkv and G >= 2:
    // {b, t0, L, kt, hq0 = the group's first query head, dqt0 of hq0, n_dq_tiles (slot stride per head), G},
    // sections by rule 1 by default (whole rows that fit ~80 MB; switch BLU_VJ2_HSUM_SECT), LPT inside a section.
    std::vector<int32_t> vh;
    int vh_G = 0;
    if (Hkv > 0 && Hq / Hkv >= 2) {
        const int Gh = Hq / Hkv, HBk = B * Hkv, hs = vj2_hsum_sect();
        // 2026-11-17 FINAL RULE 1: whole rows per section, as many as fit ~80 MB (G x S x 1 KB per pair), power of 2, >= 1.
        // Old line (kept): const int swh = hs > 0 ? hs : (hs < 0 ? HBk : std::max(1, sw / Gh));
        const long pair_bytes = (long)Gh * S * 1024;
        const long rows_fit = std::max(1L, (80L << 20) / ((long)Hkv * pair_bytes));
        int rows2 = 1; while (2L * rows2 <= rows_fit) rows2 *= 2;
        const int swh = hs > 0 ? hs : (hs < 0 ? HBk : rows2 * Hkv);
        vh_G = Gh;
        for (int h0 = 0; h0 < HBk; h0 += swh) {
            sec.clear();
            for (int hb = h0; hb < std::min(HBk, h0 + swh); ++hb) {
                const int b = hb / Hkv, hq0 = (hb % Hkv) * Gh;
                for (int j = rowfirst[b]; j < rowfirst[(size_t)b + 1]; ++j) {
                    const Seg& g = byrow[j];
                    const int nq = (g.L + 63) / 64;
                    for (int kt = 0; 128 * kt < g.L; ++kt)
                        sec.push_back({g.b, g.t0, g.L, kt, hq0, hq0 * nd + slot0[j] + 2 * kt, nq - 2 * kt});
                }
            }
            std::stable_sort(sec.begin(), sec.end(), [](const VB& a, const VB& c) { return a.work > c.work; });
            for (const VB& v : sec) for (int x : {v.b, v.t0, v.L, v.kt, v.hq, v.dqt0, nd, Gh}) vh.push_back(x);
        }
    }

    // Generic causal kernels: one entry per segment, (row, start) order.
    std::vector<int32_t> gs;
    int gmax = 0;
    for (const Seg& g : byrow) {
        for (int x : {g.b, g.t0, g.L, 0}) gs.push_back(x);
        gmax = std::max(gmax, g.L);
    }

    std::vector<int32_t> all;   // every list is a multiple of 4 ints, so each one starts 16-B aligned (int4 loads)
    // 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2): + the per-KV-head list, LAST. Old lines (kept):
    //     all.reserve(fu.size() + bt.size() + vb.size() + dq.size() + gs.size());
    //     for (const std::vector<int32_t>* v : {&fu, &bt, &vb, &dq, &gs}) all.insert(all.end(), v->begin(), v->end());
    all.reserve(fu.size() + bt.size() + vb.size() + dq.size() + gs.size() + vh.size());
    for (const std::vector<int32_t>* v : {&fu, &bt, &vb, &dq, &gs, &vh}) all.insert(all.end(), v->begin(), v->end());
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
    vj2_hsum_set(plan.vj2_blocks, plan.gen_segs + gs.size(), (int)(vh.size() / 8), vh_G);   // 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2)
    plan.mx_hkv = (n_mixed > 0) ? Hkv : 0;   // 2026-11-13: 0 = no mixed unit in the list -> the forward kernel of before
}

void gqa_chat_plan_free(GqaChatPlan& plan)
{
    vj2_hsum_set(plan.vj2_blocks, nullptr, 0, 0);   // 2026-11-17 IDEA 4 (head sums inside the block, on top of ideas 1 + 2): forget this plan's list
    if (plan.storage) CachingCUDAAllocator::instance().deallocate(plan.storage);
    plan = GqaChatPlan{};
}

// 2026-11-13: Hkv passed through to gqa_chat_plan_build (mixed units).
// Old code (kept, not deleted):
// GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
//                                   int n_seg, int B, int S, int Hq)
// {
//     std::unique_ptr<GqaChatPlan> p(new GqaChatPlan());
//     gqa_chat_plan_build(*p, seg_b, seg_t0, seg_len, n_seg, B, S, Hq);
GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
                                  int n_seg, int B, int S, int Hq, int Hkv)
{
    std::unique_ptr<GqaChatPlan> p(new GqaChatPlan());
    gqa_chat_plan_build(*p, seg_b, seg_t0, seg_len, n_seg, B, S, Hq, Hkv);
    return GqaChatPlanPtr(p.release(), [](const GqaChatPlan* q) {
        GqaChatPlan* m = const_cast<GqaChatPlan*>(q);
        gqa_chat_plan_free(*m);
        delete m;
    });
}

} // namespace cuda
} // namespace OwnTensor
