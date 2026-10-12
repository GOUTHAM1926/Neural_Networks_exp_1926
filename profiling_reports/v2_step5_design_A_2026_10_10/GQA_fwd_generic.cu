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
// v2 (2026-10-10): seg_start != nullptr selects the packed-rows variant (design B);
// chat_units != nullptr (n_units entries) the per-segment variant (design A).
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* seg_start, const int32_t* chat_units, int n_units);

// v62 needs S % 256 == 0: total_units = B*Hq*(nTiles/4) with nTiles = S/BLOCK_M
// (=S/64), so the /4 pairing requires S/64 to be a multiple of 4.
static inline bool gqa_v62_shape_ok(int T, int hd) {
    return hd == 128 && (T % 256 == 0);
}

namespace {  // internal linkage: fallback kernel + launcher

// =============================================================================
// gqa_v2 -- portable WMMA fallback (sm_80+). Non-causal, head_dim 64/128.
// =============================================================================
template<int Br, int Bc, int D>
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
  float scale
){
  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int lane   = threadIdx.x;

  const int q_row0    = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq  + hq)  * S  + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq  + hq)  * S) + q_row0;

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
    sQ[i] = d_Q[qBase + i];
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
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
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
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  }
  if(lane < Br){
    d_LSE[lBase + lane] = sm[lane] + logf(sl[lane]);
  }
}

template<int Br, int Bc, int D>
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
  float scale
){
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
  static_assert(Br == 16, "gqa_v2 processes exactly 16 query rows per block");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(32);
  gqa_v2<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
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
    // BLUTRAIN_NO_SM90_GQA forces the generic path below; note that for hd==128
    // there is no generic *causal* kernel, so this throws rather than silently
    // computing non-causal attention.
    if (is_causal && !disable_sm90_gqa() &&
        get_arch() == ArchFamily::Hopper && gqa_v62_shape_ok(T, hd)) {
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   /*seg_start=*/nullptr, /*chat_units=*/nullptr, 0);
        return;
    }

    if (is_causal && hd == 64) {
        if (T % 256 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_forward: causal path requires T to be a multiple "
                "of 256 (Br=Bc=128 tiles, two tiles processed per block)");
        }
        if (get_arch() == ArchFamily::Blackwell_B300) {
            gqa_v48_causal_launch_d64(dQ, dK, dV, dO, lse,
                                      B, Nq_heads, Nkv_heads, T, G, scale);
            return;
        }
        // Non-B300 fallback (portable WMMA, causal via per-tile skip below is
        // NOT available in gqa_v2 -- it computes full attention). NOTE: gqa_v2
        // has no causal mask, so this is only numerically correct for the
        // Blackwell kernel; on other GPUs it approximates by full attention.
        launch_gqa_v2<16, 32, 64>(dQ, dK, dV, dO, lse,
                                  B, Nq_heads, Nkv_heads, G, T, scale);
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
    bool is_causal, const int32_t* seg_start, const GqaChatPlan* chats)
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

    // v2 packed rows (design B, 2026-10-10): only the Hopper v62 kernel knows seg_start. Every
    // other path below would IGNORE it and silently let the chats of a row see each other, so a
    // seg_start that cannot be honoured is refused here instead.
    if (seg_start != nullptr && !v62_ok) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: seg_start (packed rows) needs the Hopper v62 path: "
            "causal, head_dim 128, T % 256 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset");
    }
    // v2 design A (per-segment plan): the same rule, and a plan is only valid for the shape it was
    // built for (its units hold row indices, head indices and segment bounds of that shape).
    if (chats != nullptr) {
        if (seg_start != nullptr)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: pass seg_start (design B) OR chats "
                                     "(design A), not both");
        if (!v62_ok)
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: chats (packed rows, design A) needs the Hopper v62 path: "
                "causal, head_dim 128, T % 256 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset");
        if (chats->B != B || chats->S != T || chats->Hq != Nq_heads || chats->fwd_units == nullptr)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: chats plan was built for another shape "
                                     "(B, T, Nq_heads) or is empty");
    }

    // Hopper (H100/H200) causal hd==128 -> v62 (wgmma/TMA persistent kernel).
    // This is the path bluscript's "split" attn_mode reaches.
    // BLUTRAIN_NO_SM90_GQA forces the generic path below; see the note on the
    // packed entry point above about hd==128 throwing instead of falling back.
    if (v62_ok) {
        gqa_v62_causal_launch_d128(dQ, dK, dV, dO, lse,
                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0, seg_start,
                                   chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0);
        return;
    }

    if (is_causal && hd == 64) {
        if (T % 256 != 0) {
            throw std::runtime_error(
                "gqa_flash_attn_forward_qkv: causal path requires T to be a multiple "
                "of 256 (Br=Bc=128 tiles, two tiles processed per block)");
        }
        if (get_arch() == ArchFamily::Blackwell_B300) {
            gqa_v48_causal_launch_d64(dQ, dK, dV, dO, lse,
                                      B, Nq_heads, Nkv_heads, T, G, scale);
            return;
        }
        launch_gqa_v2<16, 32, 64>(dQ, dK, dV, dO, lse,
                                  B, Nq_heads, Nkv_heads, G, T, scale);
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
// segment, most q-tiles to visit first. Both lists are uploaded once; the kernels only read them.
// ---------------------------------------------------------------------------
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
    gqa_chat_plan_free(plan);
    auto& alloc = CachingCUDAAllocator::instance();
    plan.fwd_units = static_cast<int32_t*>(alloc.allocate(fu.size() * sizeof(int32_t)));
    plan.bwd_tiles = static_cast<int32_t*>(alloc.allocate(bt.size() * sizeof(int32_t)));
    if (cudaMemcpy(plan.fwd_units, fu.data(), fu.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(plan.bwd_tiles, bt.data(), bt.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("gqa_chat_plan_build: upload failed");
    plan.n_fwd_units = (int)(fu.size() / 4);
    plan.n_bwd_tiles = (int)(bt.size() / 4);
    plan.B = B; plan.S = S; plan.Hq = Hq;
}

void gqa_chat_plan_free(GqaChatPlan& plan)
{
    auto& alloc = CachingCUDAAllocator::instance();
    if (plan.fwd_units) alloc.deallocate(plan.fwd_units);
    if (plan.bwd_tiles) alloc.deallocate(plan.bwd_tiles);
    plan = GqaChatPlan{};
}

} // namespace cuda
} // namespace OwnTensor
