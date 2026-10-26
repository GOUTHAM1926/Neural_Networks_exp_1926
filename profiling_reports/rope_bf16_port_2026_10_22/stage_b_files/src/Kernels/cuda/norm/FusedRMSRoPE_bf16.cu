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

__global__ void reduce_dgamma_bf16_tm(const float* __restrict__ part, float* __restrict__ dgamma, int n_rows, int hd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hd) return;
    float acc = 0.f;
    for (int r = 0; r < n_rows; ++r) acc += part[(long)r * hd + d];
    dgamma[d] = acc;
}

static float* d_bf16tm_qpart = nullptr; static size_t bf16tm_qpart_cap = 0;
static float* d_bf16tm_kpart = nullptr; static size_t bf16tm_kpart_cap = 0;

static void ensure_part(float** buf, size_t* cap, size_t need) {
    if (need > *cap) {
        if (*buf) cudaFree(*buf);
        cudaMalloc(buf, need * sizeof(float));
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
    ensure_part(&d_bf16tm_qpart, &bf16tm_qpart_cap, (size_t)B * Hq  * BF16_BWD_T_CHUNKS * hd);
    ensure_part(&d_bf16tm_kpart, &bf16tm_kpart_cap, (size_t)B * Hkv * BF16_BWD_T_CHUNKS * hd);
    dim3 block(32);
    dim3 grid(BF16_BWD_T_CHUNKS, Hq + Hkv, B);
    auto dzq = reinterpret_cast<const bf16*>(dz_q); auto dzk = reinterpret_cast<const bf16*>(dz_k);
    auto qi  = reinterpret_cast<const bf16*>(q_in); auto ki  = reinterpret_cast<const bf16*>(k_in);
    auto dqi = reinterpret_cast<bf16*>(dq_in);      auto dki = reinterpret_cast<bf16*>(dk_in);

    // HASP selects the per-token-positions instantiation (positions != nullptr)
#define LAUNCH_B(NEOX, HDV, HASP)                                                             \
    fused_qknorm_rope_bwd_bf16_tm<NEOX, HDV, HASP><<<grid, block, 0, stream>>>(              \
        dzq, dzk, qi, ki, q_gamma, k_gamma, cos_sin, q_rstd, k_rstd, dqi, dki,               \
        d_bf16tm_qpart, d_bf16tm_kpart, Hq, Hkv, T, pos_offset, positions)
#define DISPATCH_B(NEOX, HASP)                                                                \
    switch (hd) {                                                                             \
        case  64: LAUNCH_B(NEOX,  64, HASP); break;                                           \
        case 128: LAUNCH_B(NEOX, 128, HASP); break;                                           \
        default : fprintf(stderr, "fused_qk_norm_rope_backward_bf16_tm: unsupported hd=%d\n", hd); abort(); \
    }
    if (positions == nullptr) {   // original path: cache row = t + pos_offset
        if (!interleaved) { DISPATCH_B(true, false); } else { DISPATCH_B(false, false); }
    } else {                      // packed rows: cache row = positions[b*T + t]
        if (!interleaved) { DISPATCH_B(true, true); } else { DISPATCH_B(false, true); }
    }
#undef DISPATCH_B
#undef LAUNCH_B

    const int thr = 128, blk = (hd + thr - 1) / thr;
    reduce_dgamma_bf16_tm<<<blk, thr, 0, stream>>>(d_bf16tm_qpart, dq_gamma, B * Hq  * BF16_BWD_T_CHUNKS, hd);
    reduce_dgamma_bf16_tm<<<blk, thr, 0, stream>>>(d_bf16tm_kpart, dk_gamma, B * Hkv * BF16_BWD_T_CHUNKS, hd);
}

} // namespace cuda
} // namespace OwnTensor
