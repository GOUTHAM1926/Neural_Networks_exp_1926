// =============================================================================
// FusedRMSRoPE_bf16_opt.cu (2026-10-22): optimized versions of the kernels in FusedRMSRoPE_bf16.cu (Kruthik's bf16
// fused QK-RMSNorm + RoPE with the stage-B positions), in namespace OwnTensor::cuda::ropt so a benchmark can run both.
// Every change below targets a cause proven with ncu on H200 (profiling_reports/rope_bf16_port_2026_10_22,
// rope_bf16_ncu_1022.ncu-rep), and keeps each token's math, the element-to-lane mapping and the dgamma accumulation
// order of the original, so every output is meant to be BIT-IDENTICAL to it (checked by bench_rope_opt.cu):
//
//   F1 forward : the original issues token k+1's loads only after token k's store (the per-token `break` on T keeps
//                the compiler from moving loads up), so each warp waits for memory 4 times. Here all loads of the
//                warp's 4 tokens (q/k row, cos/sin row; positions first when HAS_POS) are issued before any compute.
//   B1 backward: one warp walks its 128 tokens and waits for each token's loads (65% of stalls). Here the loads of
//                token t+32 are issued before token t is computed (software pipeline, depth 1).
//   B2 positions: the cos/sin address needed the position first, a second chained wait per token (+90 us at B=4).
//                Here lane l loads the positions of the warp's tokens 32 at a time (one load per 32 tokens, one batch
//                ahead), and each token takes its position from a shuffle, so the cos/sin load needs no memory wait.
//   R1 dgamma reduce: one block of 128 threads added 2,304 rows one by one (0.0 waves, DRAM 0.3%). Here q and k are
//                reduced in one launch, one warp per 32 columns (8 warps on 8 SMs at hd 128), keeping the SAME
//                sequential order per column (so the sum is bit-identical) but with 32 rows of loads in flight.
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace OwnTensor {
namespace cuda {
namespace ropt {

using bf16 = __nv_bfloat16;
#ifndef BF16_BWD_T_CHUNKS
#define BF16_BWD_T_CHUNKS 32
#endif

template <int W> __device__ __forceinline__ void ld_bf16(const bf16* p, float (&v)[W]);
template <> __device__ __forceinline__ void ld_bf16<4>(const bf16* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
template <> __device__ __forceinline__ void ld_bf16<2>(const bf16* p, float (&v)[2]) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
    v[0] = f.x; v[1] = f.y;
}
template <int W> __device__ __forceinline__ void st_bf16(bf16* p, const float (&v)[W]);
template <> __device__ __forceinline__ void st_bf16<4>(bf16* p, const float (&v)[4]) {
    uint2 u;
    *reinterpret_cast<__nv_bfloat162*>(&u.x) = __float22bfloat162_rn(make_float2(v[0], v[1]));
    *reinterpret_cast<__nv_bfloat162*>(&u.y) = __float22bfloat162_rn(make_float2(v[2], v[3]));
    *reinterpret_cast<uint2*>(p) = u;
}
template <> __device__ __forceinline__ void st_bf16<2>(bf16* p, const float (&v)[2]) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __float22bfloat162_rn(make_float2(v[0], v[1]));
}
template <int W> __device__ __forceinline__ void ld_f32(const float* p, float (&v)[W]) {
    if constexpr (W == 4) { const float4 f = *reinterpret_cast<const float4*>(p); v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w; }
    else                  { const float2 f = *reinterpret_cast<const float2*>(p); v[0] = f.x; v[1] = f.y; }
}

// ---------------------------------------------------------------------------------------------------- F1 ----
// NeoX, hd 64/128 only (the path the trainer uses; the original's generic path is unchanged and not copied here).
// Per token: exactly the original arithmetic (lines 100-134 of FusedRMSRoPE_bf16.cu), in the same order.
template <int HD, int TOK_PER_WARP, int WARPS_PER_BLOCK, bool HAS_POS>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK)
void fwd_neox(const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
              bf16* __restrict__ q_out, bf16* __restrict__ k_out,
              const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
              const float* __restrict__ cos_sin_cache,
              float* __restrict__ q_rstd, float* __restrict__ k_rstd,
              int Hq, int Hkv, int T, int pos_offset, float eps,
              const int32_t* __restrict__ positions) {
    static_assert(HD == 64 || HD == 128, "F1 covers the NeoX hd 64/128 path");
    constexpr int W = HD / 32;
    const int warp_id = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tok_base = blockIdx.x * (WARPS_PER_BLOCK * TOK_PER_WARP) + warp_id * TOK_PER_WARP;
    const int head_ind = blockIdx.y, batch = blockIdx.z;
    const bool is_query = (head_ind < Hq);
    const int h = is_query ? head_ind : (head_ind - Hq);
    const int H = is_query ? Hq : Hkv;
    const bf16*  in_base  = is_query ? q_in    : k_in;
    bf16*        out_base = is_query ? q_out   : k_out;
    const float* gamma    = is_query ? q_gamma : k_gamma;
    float*       rstd_out = is_query ? q_rstd  : k_rstd;
    if (tok_base >= T) return;
    const long out_head  = (((long)batch * H + h) * T + tok_base) * HD;
    const long rstd_base = ((long)batch * H + h) * T + tok_base;
    const bool lo = (lane < 16);

    float g[W], gp[W];
    ld_f32<W>(gamma + lane * W, g);

    // ---- all loads of the warp's tokens first (tokens past T are skipped, never loaded) ----
    int pos[TOK_PER_WARP];
    #pragma unroll
    for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
        const int tok_pos = tok_base + tt;
        pos[tt] = (tok_pos < T) ? (HAS_POS ? positions[(long)batch * T + tok_pos] : tok_pos + pos_offset) : 0;
    }
    float v[TOK_PER_WARP][W], cs0[TOK_PER_WARP][W];
    #pragma unroll
    for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
        const int tok_pos = tok_base + tt;
        if (tok_pos < T) {
            ld_bf16<W>(in_base + (((long)batch * T + tok_pos) * H + h) * HD + lane * W, v[tt]);
            ld_f32<W>(cos_sin_cache + (long)pos[tt] * HD + lane * W, cs0[tt]);
        }
    }
    #pragma unroll
    for (int c = 0; c < W; ++c) gp[c] = __shfl_xor_sync(0xffffffff, g[c], 16);

    // ---- compute and store, token by token, as the original ----
    #pragma unroll
    for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
        const int tok_pos = tok_base + tt;
        if (TOK_PER_WARP > 1 && tok_pos >= T) break;
        bf16* y = out_base + out_head + (long)tt * HD;
        float part = 0.f;
        #pragma unroll
        for (int c = 0; c < W; ++c) part += v[tt][c] * v[tt][c];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
        const float rstd = rsqrtf(part / float(HD) + eps);
        if (lane == 0) rstd_out[rstd_base + tt] = rstd;
        float csp[W], vp[W], out[W];
        #pragma unroll
        for (int c = 0; c < W; ++c) {
            csp[c] = __shfl_xor_sync(0xffffffff, cs0[tt][c], 16);
            vp[c]  = __shfl_xor_sync(0xffffffff, v[tt][c], 16);
        }
        // The original source is lo ? (a*cs - b*sn) : (a*sn + b*cs). Its compiled form (sm_90 SASS of
        // fused_qknorm_rope_fwd_bf16_tm<true,128,4,8,*>) evaluates BOTH halves as fma(a, own, +-(b * partner)), where
        // `own` is this lane's loaded cos (lo) / sin (hi) and `partner` the value shuffled from lane ^ 16. Left to
        // itself, the compiler paired the hi half the other way here (85 of 9.4M outputs 1 bf16 ulp off), so the
        // original's form is written out: __fmul_rn is never fused, __fmaf_rn is exactly one fused multiply-add.
        #pragma unroll
        for (int c = 0; c < W; ++c) {
            const float ys  = v[tt][c] * rstd * g[c];
            const float ysp = vp[c] * rstd * gp[c];
            const float a    = lo ? ys  : ysp;
            const float b    = lo ? ysp : ys;
            const float prod = __fmul_rn(b, csp[c]);
            out[c] = __fmaf_rn(a, cs0[tt][c], lo ? -prod : prod);
        }
        st_bf16<W>(y + lane * W, out);
    }
}

// ----------------------------------------------------------------------------------------------- B1 + B2 ----
// The per-token arithmetic is the original's (lines 251-280 of FusedRMSRoPE_bf16.cu), with the same element-to-lane
// mapping (pair p of lane l = elements l + 32p and l + 32p + HALF for NeoX) and the same dg accumulation order.
template <bool IS_NEOX, int HD>
struct BwdTok {
    static constexpr int PAIRS = (HD / 2 + 31) / 32;
    float dza[PAIRS], dzb[PAIRS], xa[PAIRS], xb[PAIRS], cs[PAIRS], sn[PAIRS];
    float rstd;
};

template <bool IS_NEOX, int HD, bool HAS_POS>
__global__ void bwd(const bf16* __restrict__ dz_q, const bf16* __restrict__ dz_k,
                    const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
                    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
                    const float* __restrict__ cos_sin,
                    const float* __restrict__ q_rstd, const float* __restrict__ k_rstd,
                    bf16* __restrict__ dq_in, bf16* __restrict__ dk_in,
                    float* __restrict__ dq_gpart, float* __restrict__ dk_gpart,
                    int Hq, int Hkv, int T, int pos_offset,
                    const int32_t* __restrict__ positions) {
    constexpr int HALF  = HD / 2;
    constexpr int PAIRS = (HALF + 31) / 32;
    using Tok = BwdTok<IS_NEOX, HD>;

    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.x, hh = blockIdx.y, b = blockIdx.z;
    const bool is_q = (hh < Hq);
    const int  h    = is_q ? hh : (hh - Hq);
    const int  H    = is_q ? Hq : Hkv;
    const bf16*  dz_base = is_q ? dz_q    : dz_k;
    const bf16*  in_base = is_q ? q_in    : k_in;
    const float* gamma   = is_q ? q_gamma : k_gamma;
    const float* rstd_in = is_q ? q_rstd  : k_rstd;
    bf16*        dx_base = is_q ? dq_in   : dk_in;

    int ia_r[PAIRS], ib_r[PAIRS];
    bool act[PAIRS];
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        const int i = lane + p * 32;
        act[p] = (HALF % 32 == 0) || (i < HALF);
        const int ii = act[p] ? i : 0;
        ia_r[p] = IS_NEOX ? ii : (2 * ii);
        ib_r[p] = IS_NEOX ? (ii + HALF) : (2 * ii + 1);
    }
    float dg_a[PAIRS] = {}, dg_b[PAIRS] = {};
    float ga_r[PAIRS], gb_r[PAIRS];
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) { ga_r[p] = gamma[ia_r[p]]; gb_r[p] = gamma[ib_r[p]]; }

    // B2: lane l holds the position of this warp's token number (32*k + l) of batch k; `pnext` is batch k+1,
    // loaded one batch ahead. Token j of the warp is t = t0 + 32*j (BF16_BWD_T_CHUNKS = 32 blocks along t).
    const int n_tok = (T - t0 + BF16_BWD_T_CHUNKS - 1) / BF16_BWD_T_CHUNKS;   // tokens this warp walks
    auto pos_batch = [&](int k) -> int {            // positions of tokens 32k .. 32k+31 of this warp, one per lane
        const int j = 32 * k + lane;
        return (HAS_POS && j < n_tok) ? positions[(long)b * T + t0 + (long)BF16_BWD_T_CHUNKS * j] : 0;
    };
    int pcur = pos_batch(0), pnext = pos_batch(1);
    auto position_of = [&](int j) -> int {          // j in [32k, 32k+31] with k the current batch
        return HAS_POS ? __shfl_sync(0xffffffff, pcur, j & 31) : (t0 + BF16_BWD_T_CHUNKS * j + pos_offset);
    };

    auto load = [&](int j, int pos, Tok& k) {       // all loads of the warp's token j
        const int t = t0 + BF16_BWD_T_CHUNKS * j;
        const bf16* dz = dz_base + (((long)b * H + h) * T + t) * HD;
        const bf16* x  = in_base + (((long)b * T + t) * H + h) * HD;
        const float* cs_row = cos_sin + (long)pos * HD;
        k.rstd = rstd_in[((long)b * H + h) * T + t];
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i = lane + p * 32;
            k.cs[p]  = cs_row[act[p] ? i : 0];
            k.sn[p]  = cs_row[act[p] ? i + HALF : HALF];
            k.dza[p] = __bfloat162float(dz[ia_r[p]]);
            k.dzb[p] = __bfloat162float(dz[ib_r[p]]);
            k.xa[p]  = __bfloat162float(x[ia_r[p]]);
            k.xb[p]  = __bfloat162float(x[ib_r[p]]);
        }
    };

    Tok cur, nxt;
    if (n_tok > 0) load(0, position_of(0), cur);
    for (int j = 0; j < n_tok; ++j) {
        // ---- issue the next token's loads before computing this one ----
        if (j + 1 < n_tok) {
            if (HAS_POS && ((j + 1) & 31) == 0) { pcur = pnext; pnext = pos_batch((j + 1) / 32 + 1); }
            load(j + 1, position_of(j + 1), nxt);
        }
        // ---- the original per-token math ----
        const int t = t0 + BF16_BWD_T_CHUNKS * j;
        bf16* dx = dx_base + (((long)b * T + t) * H + h) * HD;
        const float rstd = cur.rstd;
        float xh_a[PAIRS], xh_b[PAIRS], dy_a[PAIRS], dy_b[PAIRS], g_a[PAIRS], g_b[PAIRS];
        float partial = 0.f;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const float cs = cur.cs[p], sn = cur.sn[p];
            const float dza = cur.dza[p], dzb = cur.dzb[p];
            dy_a[p] =  dza * cs + dzb * sn;
            dy_b[p] = -dza * sn + dzb * cs;
            xh_a[p] = cur.xa[p] * rstd;
            xh_b[p] = cur.xb[p] * rstd;
            g_a[p]  = dy_a[p] * ga_r[p];
            g_b[p]  = dy_b[p] * gb_r[p];
            if (act[p]) partial += g_a[p] * xh_a[p] + g_b[p] * xh_b[p];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, o);
        const float S = partial / (float)HD;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            if (!act[p]) continue;
            dx[ia_r[p]] = __float2bfloat16_rn(rstd * (g_a[p] - xh_a[p] * S));
            dx[ib_r[p]] = __float2bfloat16_rn(rstd * (g_b[p] - xh_b[p] * S));
            dg_a[p] += dy_a[p] * xh_a[p];
            dg_b[p] += dy_b[p] * xh_b[p];
        }
        if (j + 1 < n_tok) cur = nxt;
    }

    const int row = ((b * H) + h) * BF16_BWD_T_CHUNKS + t0;
    float* dst = (is_q ? dq_gpart : dk_gpart) + (long)row * HD;
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        if (!act[p]) continue;
        dst[ia_r[p]] = dg_a[p];
        dst[ib_r[p]] = dg_b[p];
    }
}

// ---------------------------------------------------------------------------------------------------- R1 ----
// blockIdx.y = 0 -> q, 1 -> k; one warp per block, lane = one column. The SAME sequential sum per column as the
// original (acc += part[r][d] for r = 0, 1, 2, ...), but the loads of the next UNROLL rows are issued before the adds.
template <int UNROLL>
__global__ void reduce_dgamma2(const float* __restrict__ qpart, float* __restrict__ dq_gamma, int q_rows,
                               const float* __restrict__ kpart, float* __restrict__ dk_gamma, int k_rows, int hd) {
    const int d = blockIdx.x * 32 + (threadIdx.x & 31);
    if (d >= hd) return;
    const bool is_q = (blockIdx.y == 0);
    const float* part = is_q ? qpart : kpart;
    const int n_rows  = is_q ? q_rows : k_rows;
    float acc = 0.f;
    int r = 0;
    for (; r + UNROLL <= n_rows; r += UNROLL) {
        float v[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = part[(long)(r + u) * hd + d];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) acc += v[u];
    }
    for (; r < n_rows; ++r) acc += part[(long)r * hd + d];
    (is_q ? dq_gamma : dk_gamma)[d] = acc;
}

}  // namespace ropt
}  // namespace cuda
}  // namespace OwnTensor
