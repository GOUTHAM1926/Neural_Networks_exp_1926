#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "ops/helpers/FusedRMSRoPEKernels.h"

// bf16 variant of the fused QK-norm + RoPE kernels: reads Q/K in the projection's token-major
// [B,T,H,hd] layout and writes head-major bf16, with the fp32 math and reduction order of the sm90 kernels.

namespace OwnTensor {
namespace cuda {

using bf16 = __nv_bfloat16;

#ifndef BF16_BWD_T_CHUNKS
#define BF16_BWD_T_CHUNKS 32
#endif
#ifndef BF16_BWD_DEPTH
#define BF16_BWD_DEPTH 3          // tokens in flight per warp in the hd 128 backward (fused_qknorm_rope_bwd_pp_bf16_tm)
#endif
#ifndef BF16_RED_U
#define BF16_RED_U 64             // dgamma reduce: rows per register buffer
#endif
#ifndef BF16_RED_STAGES
#define BF16_RED_STAGES 3         // dgamma reduce: register buffers (STAGES - 1 of them loading while one is added)
#endif

template <int W> __device__ __forceinline__ void ld_bf16(const bf16* p, float (&v)[W]);
template <> __device__ __forceinline__ void ld_bf16<2>(const bf16* p, float (&v)[2]) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
    v[0] = f.x; v[1] = f.y;
}
template <> __device__ __forceinline__ void ld_bf16<4>(const bf16* p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
template <int W> __device__ __forceinline__ void st_bf16(bf16* p, const float (&v)[W]);
template <> __device__ __forceinline__ void st_bf16<2>(bf16* p, const float (&v)[2]) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __float22bfloat162_rn(make_float2(v[0], v[1]));
}
template <> __device__ __forceinline__ void st_bf16<4>(bf16* p, const float (&v)[4]) {
    uint2 u;
    *reinterpret_cast<__nv_bfloat162*>(&u.x) = __float22bfloat162_rn(make_float2(v[0], v[1]));
    *reinterpret_cast<__nv_bfloat162*>(&u.y) = __float22bfloat162_rn(make_float2(v[2], v[3]));
    *reinterpret_cast<uint2*>(p) = u;
}
template <int W> __device__ __forceinline__ void ld_f32(const float* p, float (&v)[W]) {
    if constexpr (W == 4) { const float4 f = *reinterpret_cast<const float4*>(p); v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w; }
    else                  { const float2 f = *reinterpret_cast<const float2*>(p); v[0] = f.x; v[1] = f.y; }
}

// HAS_POS (packed SFT rows, added 2026-10-22, the same scheme as the fp32 kernels in norm/arch/): false = the original
// kernel, position = t + pos_offset; true = per-token positions[b*T + t] (several chats per row, each restarting at 0).
// A template flag, not a runtime branch, so the HAS_POS=false instantiation is the original kernel; positions is
// appended LAST so the existing parameters keep their places.
template <bool IS_NEOX, int HD, int TOK_PER_WARP, int WARPS_PER_BLOCK, bool HAS_POS>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK)
void fused_qknorm_rope_fwd_bf16_tm(
    const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
    bf16* __restrict__ q_out, bf16* __restrict__ k_out,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    const float* __restrict__ cos_sin_cache,
    float* __restrict__ q_rstd, float* __restrict__ k_rstd,
    int Hq, int Hkv, int T, int pos_offset, float eps,
    const int32_t* __restrict__ positions) {   // [B, T], read only when HAS_POS

    constexpr int HALF  = HD / 2;
    constexpr int VECS  = (HD + 31) / 32;
    constexpr int PAIRS = (HALF + 31) / 32;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int tok_base = blockIdx.x * (WARPS_PER_BLOCK * TOK_PER_WARP) + warp_id * TOK_PER_WARP;
    const int head_ind = blockIdx.y;
    const int batch    = blockIdx.z;

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

    if (IS_NEOX && (HD == 64 || HD == 128)) {
        constexpr int W = HD / 32;
        const bool lo = (lane < 16);
        float g[W], gp[W];
        ld_f32<W>(gamma + lane * W, g);
        #pragma unroll
        for (int c = 0; c < W; ++c) gp[c] = __shfl_xor_sync(0xffffffff, g[c], 16);

        #pragma unroll
        for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
            const int tok_pos = tok_base + tt;
            if (TOK_PER_WARP > 1 && tok_pos >= T) break;

            const bf16* x = in_base + (((long)batch * T + tok_pos) * H + h) * HD;
            bf16*       y = out_base + out_head + (long)tt * HD;

            float v[W];
            ld_bf16<W>(x + lane * W, v);

            float part = 0.f;
            #pragma unroll
            for (int c = 0; c < W; ++c) part += v[c] * v[c];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffff, part, o);

            const float rstd = rsqrtf(part / float(HD) + eps);
            if (lane == 0) rstd_out[rstd_base + tt] = rstd;

            // positions: the token's position inside its chat (HAS_POS=false keeps tok_pos + pos_offset)
            const float* cs_row = HAS_POS
                ? cos_sin_cache + (long)positions[(long)batch * T + tok_pos] * HD
                : cos_sin_cache + (long)(tok_pos + pos_offset) * HD;
            float cs0[W], csp[W], vp[W], out[W];
            ld_f32<W>(cs_row + lane * W, cs0);
            #pragma unroll
            for (int c = 0; c < W; ++c) {
                csp[c] = __shfl_xor_sync(0xffffffff, cs0[c], 16);
                vp[c]  = __shfl_xor_sync(0xffffffff, v[c], 16);
            }
            #pragma unroll
            for (int c = 0; c < W; ++c) {
                const float ys  = v[c]  * rstd * g[c];
                const float ysp = vp[c] * rstd * gp[c];
                const float a  = lo ? ys  : ysp;
                const float b  = lo ? ysp : ys;
                const float cs = lo ? cs0[c] : csp[c];
                const float sn = lo ? csp[c] : cs0[c];
                out[c] = lo ? (a * cs - b * sn) : (a * sn + b * cs);
            }
            st_bf16<W>(y + lane * W, out);
        }
    } else {
        float ga_r[PAIRS], gb_r[PAIRS];
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i = lane + p * 32;
            if (HALF % 32 != 0 && i >= HALF) break;
            ga_r[p] = gamma[IS_NEOX ? i : 2 * i];
            gb_r[p] = gamma[IS_NEOX ? i + HALF : 2 * i + 1];
        }
        #pragma unroll
        for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
            const int tok_pos = tok_base + tt;
            if (TOK_PER_WARP > 1 && tok_pos >= T) break;

            const bf16* x = in_base + (((long)batch * T + tok_pos) * H + h) * HD;
            bf16*       y = out_base + out_head + (long)tt * HD;

            float part = 0.0f;
            #pragma unroll
            for (int v = 0; v < VECS; ++v) {
                const int j = lane + v * 32;
                if (HD % 32 != 0 && j >= HD) break;
                const float t = __bfloat162float(x[j]);
                part += t * t;
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffff, part, o);

            const float rstd = rsqrtf(part / float(HD) + eps);
            if (lane == 0) rstd_out[rstd_base + tt] = rstd;

            // positions: same rule as the NeoX path above
            const float* cs_row = HAS_POS
                ? cos_sin_cache + (long)positions[(long)batch * T + tok_pos] * HD
                : cos_sin_cache + (long)(tok_pos + pos_offset) * HD;
            #pragma unroll
            for (int p = 0; p < PAIRS; ++p) {
                const int i = lane + p * 32;
                if (HALF % 32 != 0 && i >= HALF) break;
                const int ia = IS_NEOX ? i : (2 * i);
                const int ib = IS_NEOX ? (i + HALF) : (2 * i + 1);
                const float ya = __bfloat162float(x[ia]) * rstd * ga_r[p];
                const float yb = __bfloat162float(x[ib]) * rstd * gb_r[p];
                const float cs = cs_row[i];
                const float sn = cs_row[i + HALF];
                y[ia] = __float2bfloat16_rn(ya * cs - yb * sn);
                y[ib] = __float2bfloat16_rn(ya * sn + yb * cs);
            }
        }
    }
}

// HAS_POS: as in the forward. The backward undoes the forward's rotation, so it must use the SAME cos/sin row for each
// token. The dgamma partials keep their fixed order, so they stay bit-reproducible with positions too.
// The launcher runs this kernel at hd 64; at hd 128 it runs fused_qknorm_rope_bwd_pp_bf16_tm below, which is meant to
// give the same outputs bit for bit there.
template <bool IS_NEOX, int HD, bool HAS_POS>
__global__ void fused_qknorm_rope_bwd_bf16_tm(
    const bf16* __restrict__ dz_q, const bf16* __restrict__ dz_k,
    const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    const float* __restrict__ cos_sin,
    const float* __restrict__ q_rstd, const float* __restrict__ k_rstd,
    bf16* __restrict__ dq_in, bf16* __restrict__ dk_in,
    float* __restrict__ dq_gpart, float* __restrict__ dk_gpart,
    int Hq, int Hkv, int T, int pos_offset,
    const int32_t* __restrict__ positions) {   // [B, T], read only when HAS_POS

    constexpr int HALF  = HD / 2;
    constexpr int PAIRS = (HALF + 31) / 32;

    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.x;
    const int hh = blockIdx.y;
    const int b  = blockIdx.z;

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
    for (int p = 0; p < PAIRS; ++p) {
        ga_r[p] = gamma[ia_r[p]];
        gb_r[p] = gamma[ib_r[p]];
    }

    for (int t = t0; t < T; t += BF16_BWD_T_CHUNKS) {
        const bf16* dz = dz_base + (((long)b * H + h) * T + t) * HD;
        const long  tm = (((long)b * T + t) * H + h) * HD;
        const bf16* x  = in_base + tm;
        bf16*       dx = dx_base + tm;
        const float  rstd = rstd_in[((long)b * H + h) * T + t];
        // positions: the same row the forward used for this token (HAS_POS=false keeps t + pos_offset)
        const float* cs_row = HAS_POS
            ? cos_sin + (long)positions[(long)b * T + t] * HD
            : cos_sin + (long)(t + pos_offset) * HD;

        float xh_a[PAIRS], xh_b[PAIRS], dy_a[PAIRS], dy_b[PAIRS], g_a[PAIRS], g_b[PAIRS];
        float partial = 0.f;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i  = lane + p * 32;
            const int ia = ia_r[p], ib = ib_r[p];
            const float cs = cs_row[act[p] ? i : 0];
            const float sn = cs_row[act[p] ? i + HALF : HALF];
            const float dza = __bfloat162float(dz[ia]), dzb = __bfloat162float(dz[ib]);
            dy_a[p] =  dza * cs + dzb * sn;
            dy_b[p] = -dza * sn + dzb * cs;
            xh_a[p] = __bfloat162float(x[ia]) * rstd;
            xh_b[p] = __bfloat162float(x[ib]) * rstd;
            g_a[p]  = dy_a[p] * ga_r[p];
            g_b[p]  = dy_b[p] * gb_r[p];
            if (act[p]) partial += g_a[p] * xh_a[p] + g_b[p] * xh_b[p];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, o);
        const float S = partial / (float)HD;

        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            if (!act[p]) continue;
            dx[ia_r[p]] = __float2bfloat16_rn(rstd * (g_a[p] - xh_a[p] * S));
            dx[ib_r[p]] = __float2bfloat16_rn(rstd * (g_b[p] - xh_b[p] * S));
            dg_a[p] += dy_a[p] * xh_a[p];
            dg_b[p] += dy_b[p] * xh_b[p];
        }
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

// The backward at hd 128, with and without positions (2026-10-24). In the kernel above each warp issues a token's loads
// and then waits for them at the first bf16 -> fp32 convert, once per token, 128 times in a row (ncu, H200, B 4: 56% of
// the stall samples on that convert; 82 us at B 1 and 108 us at B 4, 4x the data for 1.3x the time). Here each warp
// keeps DEPTH tokens in flight in a ring of register slots: a slot holds the token's loads AS LOADED (bf16 bits for dz
// and x, fp32 for cos/sin and rstd), a token is converted only when it is computed, and its slot is then refilled with
// token j + DEPTH. (The depth-1 prefetch of 2026-10-22 and the kernel this replaces kept floats, so the converts sat
// right after the next token's loads, inside the "next token exists" branch, and the warp waited there: 84.3 -> 84.3 us.)
// Positions (HAS_POS): a slot also loads the position of the token it will hold next (DEPTH tokens later), so when the
// slot is refilled its cos/sin address needs no wait. (A per-32-token batch of positions rotated by a shuffle, as in the
// kernel this replaces, made ptxas copy the fresh batch at the branch join: one full stall per 32 tokens.)
// The per-token math is written with explicit-rounding intrinsics in the exact form the kernel above compiles to at
// hd 128 (library flags, SASS): dy_a = fma(dza, cs, dzb*sn), dy_b = fma(dzb, cs, -(dza*sn)), the row sum
// 0 + fma(g_a, xh_a, g_b*xh_b) per pair then the xor tree, dx = rstd * fma(-xh, S, g), dgamma = fma(dy, xh, dgamma);
// so neither nvcc nor ptxas can pair the multiply-adds differently when the loop is unrolled. The element-to-lane
// mapping and the dgamma order (token by token, in the warp's token order) are the kernel above's. So every output is
// meant to be bit-identical to it (port test: profiling_reports/rope_bf16_port_2026_10_22/port_1024).
template <int HD>
struct BwdRawBf16 {                       // one token's loads, as loaded
    static constexpr int PAIRS = HD / 64;
    bf16  dza[PAIRS], dzb[PAIRS], xa[PAIRS], xb[PAIRS];
    float cs[PAIRS], sn[PAIRS];
    float rstd;
    int   pos_next;                       // HAS_POS: position of the token this slot holds next
};

template <bool IS_NEOX, int HD, bool HAS_POS, int DEPTH>
__global__ void fused_qknorm_rope_bwd_pp_bf16_tm(
    const bf16* __restrict__ dz_q, const bf16* __restrict__ dz_k,
    const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    const float* __restrict__ cos_sin,
    const float* __restrict__ q_rstd, const float* __restrict__ k_rstd,
    bf16* __restrict__ dq_in, bf16* __restrict__ dk_in,
    float* __restrict__ dq_gpart, float* __restrict__ dk_gpart,
    int Hq, int Hkv, int T, int pos_offset,
    const int32_t* __restrict__ positions) {   // [B, T], read only when HAS_POS

    constexpr int HALF  = HD / 2;
    constexpr int PAIRS = HALF / 32;
    static_assert(HALF % 32 == 0, "every lane must own whole pairs");
    using Raw = BwdRawBf16<HD>;

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
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        const int i = lane + p * 32;
        ia_r[p] = IS_NEOX ? i : (2 * i);
        ib_r[p] = IS_NEOX ? (i + HALF) : (2 * i + 1);
    }
    float dg_a[PAIRS] = {}, dg_b[PAIRS] = {};
    float ga_r[PAIRS], gb_r[PAIRS];
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) { ga_r[p] = gamma[ia_r[p]]; gb_r[p] = gamma[ib_r[p]]; }

    // token j of this warp is t = t0 + 32*j (BF16_BWD_T_CHUNKS blocks along t)
    const int n_tok = (T - t0 + BF16_BWD_T_CHUNKS - 1) / BF16_BWD_T_CHUNKS;
    // HAS_POS: position of token m (same address in every lane). Past the warp's last token it reads the last token's
    // position (never used): an unconditional load lands in the slot directly, while "m < n_tok ? load : 0" made ptxas
    // load into a temporary and copy it into the slot, a copy that waits for the load.
    auto position = [&](int m) -> int {
        return positions[(long)b * T + t0 + (long)BF16_BWD_T_CHUNKS * min(m, n_tok - 1)];
    };

    auto load = [&](int m, int pos_m, Raw& r) {   // token m into slot r; pos_m = its position (HAS_POS)
        const int t = t0 + BF16_BWD_T_CHUNKS * m;
        const int pos = HAS_POS ? pos_m : (t + pos_offset);
        if constexpr (HAS_POS) r.pos_next = position(m + DEPTH);
        const bf16*  dz     = dz_base + (((long)b * H + h) * T + t) * HD;
        const bf16*  x      = in_base + (((long)b * T + t) * H + h) * HD;
        const float* cs_row = cos_sin + (long)pos * HD;
        r.rstd = rstd_in[((long)b * H + h) * T + t];
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i = lane + p * 32;
            r.cs[p]  = cs_row[i];
            r.sn[p]  = cs_row[i + HALF];
            r.dza[p] = dz[ia_r[p]];
            r.dzb[p] = dz[ib_r[p]];
            r.xa[p]  = x[ia_r[p]];
            r.xb[p]  = x[ib_r[p]];
        }
    };

    auto compute = [&](int j, const Raw& r) {     // token j: dx, and its dgamma terms added in token order
        const int t = t0 + BF16_BWD_T_CHUNKS * j;
        bf16* dx = dx_base + (((long)b * T + t) * H + h) * HD;
        const float rstd = r.rstd;
        float xh_a[PAIRS], xh_b[PAIRS], dy_a[PAIRS], dy_b[PAIRS], g_a[PAIRS], g_b[PAIRS];
        float partial = 0.f;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const float dza = __bfloat162float(r.dza[p]), dzb = __bfloat162float(r.dzb[p]);
            dy_a[p] = __fmaf_rn(dza, r.cs[p],  __fmul_rn(dzb, r.sn[p]));      //  dza*cs + dzb*sn
            dy_b[p] = __fmaf_rn(dzb, r.cs[p], -__fmul_rn(dza, r.sn[p]));      // -dza*sn + dzb*cs
            xh_a[p] = __fmul_rn(__bfloat162float(r.xa[p]), rstd);
            xh_b[p] = __fmul_rn(__bfloat162float(r.xb[p]), rstd);
            g_a[p]  = __fmul_rn(dy_a[p], ga_r[p]);
            g_b[p]  = __fmul_rn(dy_b[p], gb_r[p]);
            partial = __fadd_rn(partial, __fmaf_rn(g_a[p], xh_a[p], __fmul_rn(g_b[p], xh_b[p])));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) partial = __fadd_rn(partial, __shfl_xor_sync(0xffffffff, partial, o));
        const float S = __fmul_rn(partial, 1.f / (float)HD);              // = partial / HD exactly (power of two)
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            dx[ia_r[p]] = __float2bfloat16_rn(__fmul_rn(rstd, __fmaf_rn(-xh_a[p], S, g_a[p])));
            dx[ib_r[p]] = __float2bfloat16_rn(__fmul_rn(rstd, __fmaf_rn(-xh_b[p], S, g_b[p])));
            dg_a[p] = __fmaf_rn(dy_a[p], xh_a[p], dg_a[p]);
            dg_b[p] = __fmaf_rn(dy_b[p], xh_b[p], dg_b[p]);
        }
    };

    Raw slot[DEPTH];
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d)
        if (d < n_tok) load(d, HAS_POS ? position(d) : 0, slot[d]);
    for (int j0 = 0; j0 < n_tok; j0 += DEPTH) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int j = j0 + d;
            if (j < n_tok) compute(j, slot[d]);
            if (j + DEPTH < n_tok) load(j + DEPTH, slot[d].pos_next, slot[d]);
        }
    }

    const int row = ((b * H) + h) * BF16_BWD_T_CHUNKS + t0;
    float* dst = (is_q ? dq_gpart : dk_gpart) + (long)row * HD;
    #pragma unroll
    for (int p = 0; p < PAIRS; ++p) {
        dst[ia_r[p]] = dg_a[p];
        dst[ib_r[p]] = dg_b[p];
    }
}

// dgamma = the sum of the backward's partial rows, per column in row order (r = 0, 1, 2, ...): the same sequential sum
// as the single-block reduce of the original, so dgamma is bit-identical. q and k in one launch (blockIdx.y), one warp
// per 32 columns. The 2026-10-23 version loaded a batch of rows, waited for it, then added it, so each batch paid a full
// memory round trip (ncu, H200, B 4: 64 us for 1.6 MB, 8 warps on the GPU). Here the rows go through STAGES register
// buffers of U rows: while one buffer is added, the loads of the next STAGES-1 buffers are in flight, so the loads
// overlap the chain of adds, which is the only part that has to be sequential. Each load is the column pointer plus a
// constant offset (HD is a template parameter) and needs no bound check: the partial buffers carry BF16_RED_PAD_ROWS
// zeroed rows after the last row (ensure_part), and rows past n_rows are loaded but not added. The loads are __ldcg
// (L2): with read-only (.nc) loads ptxas dropped a buffer's first rows and loaded them again right before their adds.
template <int HD, int U, int STAGES>
__global__ __launch_bounds__(32) void reduce_dgamma_qk_bf16_tm(const float* __restrict__ qpart, float* __restrict__ dq_gamma, int q_rows,
                                                               const float* __restrict__ kpart, float* __restrict__ dk_gamma, int k_rows) {
    static_assert(HD % 32 == 0, "one lane per column");
    const int  d      = blockIdx.x * 32 + (threadIdx.x & 31);
    const bool is_q   = (blockIdx.y == 0);
    const int  n_rows = is_q ? q_rows : k_rows;
    const float* col  = (is_q ? qpart : kpart) + d;              // row r of this column is col[r * HD]
    float acc = 0.f;
    float v[STAGES][U];
    auto load = [&](int r0, float (&w)[U]) {
        const float* p = col + (long)r0 * HD;
        #pragma unroll
        for (int u = 0; u < U; ++u) w[u] = __ldcg(p + u * HD);
    };
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) load(s * U, v[s]);
    for (int r = 0; r < n_rows; r += STAGES * U) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            load(r + (s + STAGES - 1) * U, v[(s + STAGES - 1) % STAGES]);
            const int r0 = r + s * U;
            if (r0 + U <= n_rows) {                               // a full buffer: no per-row test
                #pragma unroll
                for (int u = 0; u < U; ++u) acc += v[s][u];
            } else {
                #pragma unroll
                for (int u = 0; u < U; ++u)
                    if (r0 + u < n_rows) acc += v[s][u];
            }
        }
    }
    (is_q ? dq_gamma : dk_gamma)[d] = acc;
}

static float* d_bf16tm_qpart = nullptr; static size_t bf16tm_qpart_cap = 0;
static float* d_bf16tm_kpart = nullptr; static size_t bf16tm_kpart_cap = 0;

// The dgamma reduce loads up to (2*STAGES - 1)*U rows past the last row (never added); the buffers carry that many zeroed
// rows so those loads stay inside the allocation.
#define BF16_RED_PAD_ROWS (2 * BF16_RED_STAGES * BF16_RED_U)

static void ensure_part(float** buf, size_t* cap, size_t need, cudaStream_t stream) {
    if (need > *cap) {
        if (*buf) cudaFree(*buf);
        cudaMalloc(buf, need * sizeof(float));
        cudaMemsetAsync(*buf, 0, need * sizeof(float), stream);   // on the kernels' stream, so it lands before them
        *cap = need;
    }
}

void fused_qk_norm_rope_forward_bf16_tm(
    const bfloat16_t* q_in, const bfloat16_t* k_in,
    bfloat16_t* q_out, bfloat16_t* k_out,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    float* q_rstd, float* k_rstd,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, float eps, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions)
{
    constexpr int TPW = 4, WPB = 8;
    dim3 block(32 * WPB);
    dim3 grid((T + TPW * WPB - 1) / (TPW * WPB), Hq + Hkv, B);
    auto qi = reinterpret_cast<const bf16*>(q_in);  auto ki = reinterpret_cast<const bf16*>(k_in);
    auto qo = reinterpret_cast<bf16*>(q_out);       auto ko = reinterpret_cast<bf16*>(k_out);

    // HASP selects the per-token-positions instantiation (positions != nullptr)
#define LAUNCH_F(NEOX, HDV, HASP)                                                             \
    fused_qknorm_rope_fwd_bf16_tm<NEOX, HDV, TPW, WPB, HASP><<<grid, block, 0, stream>>>(   \
        qi, ki, qo, ko, q_gamma, k_gamma, cos_sin, q_rstd, k_rstd, Hq, Hkv, T, pos_offset, eps, positions)
#define DISPATCH_F(NEOX, HASP)                                                                \
    switch (hd) {                                                                             \
        case  64: LAUNCH_F(NEOX,  64, HASP); break;                                           \
        case 128: LAUNCH_F(NEOX, 128, HASP); break;                                           \
        default : fprintf(stderr, "fused_qk_norm_rope_forward_bf16_tm: unsupported hd=%d\n", hd); abort(); \
    }
    if (positions == nullptr) {   // original path: position = t + pos_offset
        if (!interleaved) { DISPATCH_F(true, false); } else { DISPATCH_F(false, false); }
    } else {                      // packed rows: position = positions[b*T + t]
        if (!interleaved) { DISPATCH_F(true, true); } else { DISPATCH_F(false, true); }
    }
#undef DISPATCH_F
#undef LAUNCH_F
}

void fused_qk_norm_rope_backward_bf16_tm(
    const bfloat16_t* dz_q, const bfloat16_t* dz_k,
    const bfloat16_t* q_in, const bfloat16_t* k_in,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    const float* q_rstd, const float* k_rstd,
    bfloat16_t* dq_in, bfloat16_t* dk_in,
    float* dq_gamma, float* dk_gamma,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions)
{
    ensure_part(&d_bf16tm_qpart, &bf16tm_qpart_cap, ((size_t)B * Hq  * BF16_BWD_T_CHUNKS + BF16_RED_PAD_ROWS) * hd, stream);
    ensure_part(&d_bf16tm_kpart, &bf16tm_kpart_cap, ((size_t)B * Hkv * BF16_BWD_T_CHUNKS + BF16_RED_PAD_ROWS) * hd, stream);
    dim3 block(32);
    dim3 grid(BF16_BWD_T_CHUNKS, Hq + Hkv, B);
    auto dzq = reinterpret_cast<const bf16*>(dz_q); auto dzk = reinterpret_cast<const bf16*>(dz_k);
    auto qi  = reinterpret_cast<const bf16*>(q_in); auto ki  = reinterpret_cast<const bf16*>(k_in);
    auto dqi = reinterpret_cast<bf16*>(dq_in);      auto dki = reinterpret_cast<bf16*>(dk_in);

    // without positions: cache row = t + pos_offset; with positions (packed rows): cache row = positions[b*T + t]
#define LAUNCH_B(NEOX, HDV, HASP)                                                             \
    fused_qknorm_rope_bwd_bf16_tm<NEOX, HDV, HASP><<<grid, block, 0, stream>>>(              \
        dzq, dzk, qi, ki, q_gamma, k_gamma, cos_sin, q_rstd, k_rstd, dqi, dki,               \
        d_bf16tm_qpart, d_bf16tm_kpart, Hq, Hkv, T, pos_offset, positions)
#define LAUNCH_PP(NEOX, HASP)                                                                 \
    fused_qknorm_rope_bwd_pp_bf16_tm<NEOX, 128, HASP, BF16_BWD_DEPTH><<<grid, block, 0, stream>>>( \
        dzq, dzk, qi, ki, q_gamma, k_gamma, cos_sin, q_rstd, k_rstd, dqi, dki,               \
        d_bf16tm_qpart, d_bf16tm_kpart, Hq, Hkv, T, pos_offset, positions)
    if (hd != 64 && hd != 128) {
        fprintf(stderr, "fused_qk_norm_rope_backward_bf16_tm: unsupported hd=%d\n", hd); abort();
    }
    const bool hasp = (positions != nullptr);
    if (hd == 128) {          // the pipelined kernel, with and without positions
        if (!interleaved) { if (hasp) LAUNCH_PP(true,  true); else LAUNCH_PP(true,  false); }
        else              { if (hasp) LAUNCH_PP(false, true); else LAUNCH_PP(false, false); }
    } else {                  // hd 64: the kernel above (the pipelined kernel's math is pinned to the hd 128 form)
        if (!interleaved) { if (hasp) LAUNCH_B(true,  64, true); else LAUNCH_B(true,  64, false); }
        else              { if (hasp) LAUNCH_B(false, 64, true); else LAUNCH_B(false, 64, false); }
    }
#undef LAUNCH_PP
#undef LAUNCH_B

    const int q_rows = B * Hq * BF16_BWD_T_CHUNKS, k_rows = B * Hkv * BF16_BWD_T_CHUNKS;
    const dim3 rgrid(hd / 32, 2);
    if (hd == 128) reduce_dgamma_qk_bf16_tm<128, BF16_RED_U, BF16_RED_STAGES><<<rgrid, 32, 0, stream>>>(
                       d_bf16tm_qpart, dq_gamma, q_rows, d_bf16tm_kpart, dk_gamma, k_rows);
    else           reduce_dgamma_qk_bf16_tm< 64, BF16_RED_U, BF16_RED_STAGES><<<rgrid, 32, 0, stream>>>(
                       d_bf16tm_qpart, dq_gamma, q_rows, d_bf16tm_kpart, dk_gamma, k_rows);
}

} // namespace cuda
} // namespace OwnTensor
