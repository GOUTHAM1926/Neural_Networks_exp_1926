// =============================================================================
// FusedRMSRoPE_bf16_v2.cu (2026-10-22): second round of optimizations of Kruthik's bf16 fused QK-RMSNorm + RoPE
// (FusedRMSRoPE_bf16.cu), namespace OwnTensor::cuda::ropt2. Aimed at the limit ncu showed after round 1: the L2 cache
// (~70% busy in forward AND backward while DRAM is ~51%), mostly because the 512-byte cos/sin row of a token is read
// once PER HEAD (24 times per token at 18/6 heads, ~200 MB of L2 reads per call at B=4).
// Every kernel keeps, for each (token, head), the original's arithmetic, element-to-lane mapping and order, and in the
// backward the same token order per (batch, head, t0) partial row, so all outputs are meant to be BIT-IDENTICAL to the
// original (checked by bench_rope_v2.cu, which also tries the template variants below):
//
//   fwd2<HD, HPW, HAS_POS>      one warp per (token, chunk of HPW heads). The token's position and cos/sin row are
//                               loaded ONCE into registers (and the partner half shuffled once), then reused for every
//                               head of the chunk. HPW = all heads reads cos/sin once per token.
//   bwd2<.., HPB, STAGE, ..>    one block per (t0, group of HPB heads, batch), one warp per head, each warp walking
//                               the original warp's tokens t0, t0+32, ... in order. The block loads the cos/sin rows of
//                               STAGE tokens at a time into shared memory ONCE (positions resolved by the loading
//                               threads), double-buffered, so the head-warps never touch cos/sin in L2 and never wait
//                               on a position. The next token's dz/x/rstd are prefetched into registers (depth 1).
//   reduce3<UNROLL>             as round 1 (q and k in one launch, one warp per 32 columns, the same sequential order
//                               per column), with UNROLL rows of loads in flight.
// =============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace OwnTensor {
namespace cuda {
namespace ropt2 {

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

// --------------------------------------------------------------------------------------------- forward ----
// NeoX, hd 64/128. Blocks of 8 warps; global warp w handles token w / n_chunks (of all B*T tokens) and heads
// [chunk*HPW, chunk*HPW + HPW) of the Hq query heads followed by the Hkv key heads.
template <int HD, int HPW, bool HAS_POS>
__global__ __launch_bounds__(256)
void fwd2(const bf16* __restrict__ q_in, const bf16* __restrict__ k_in,
          bf16* __restrict__ q_out, bf16* __restrict__ k_out,
          const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
          const float* __restrict__ cos_sin_cache,
          float* __restrict__ q_rstd, float* __restrict__ k_rstd,
          int B, int Hq, int Hkv, int T, int pos_offset, float eps,
          const int32_t* __restrict__ positions) {
    static_assert(HD == 64 || HD == 128, "NeoX hd 64/128 path");
    constexpr int W = HD / 32;
    const int H = Hq + Hkv;
    const int n_chunks = (H + HPW - 1) / HPW;
    const long gw = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const bool lo = (lane < 16);
    const long tok = gw / n_chunks;                 // index over B*T
    if (tok >= (long)B * T) return;
    const int chunk = (int)(gw % n_chunks);
    const int batch = (int)(tok / T), t = (int)(tok % T);

    // the token's cos/sin row: once per warp (the original: once per head)
    const int pos = HAS_POS ? positions[tok] : t + pos_offset;
    float cs0[W], csp[W];
    ld_f32<W>(cos_sin_cache + (long)pos * HD + lane * W, cs0);
    #pragma unroll
    for (int c = 0; c < W; ++c) csp[c] = __shfl_xor_sync(0xffffffff, cs0[c], 16);

    #pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        const int head = chunk * HPW + hh;
        if (head >= H) break;                       // warp-uniform
        const bool isq = head < Hq;
        const int h  = isq ? head : head - Hq;
        const int Hx = isq ? Hq : Hkv;
        const float* gamma = isq ? q_gamma : k_gamma;
        float g[W], gp[W], v[W], vp[W], out[W];
        ld_f32<W>(gamma + lane * W, g);
        ld_bf16<W>((isq ? q_in : k_in) + (((long)batch * T + t) * Hx + h) * HD + lane * W, v);
        #pragma unroll
        for (int c = 0; c < W; ++c) gp[c] = __shfl_xor_sync(0xffffffff, g[c], 16);
        // ---- the original per-token arithmetic (FusedRMSRoPE_bf16.cu lines 103-134), rotation pinned as compiled ----
        float part = 0.f;
        #pragma unroll
        for (int c = 0; c < W; ++c) part += v[c] * v[c];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
        const float rstd = rsqrtf(part / float(HD) + eps);
        if (lane == 0) (isq ? q_rstd : k_rstd)[((long)batch * Hx + h) * T + t] = rstd;
        #pragma unroll
        for (int c = 0; c < W; ++c) vp[c] = __shfl_xor_sync(0xffffffff, v[c], 16);
        #pragma unroll
        for (int c = 0; c < W; ++c) {
            const float ys  = v[c] * rstd * g[c];
            const float ysp = vp[c] * rstd * gp[c];
            const float a    = lo ? ys  : ysp;
            const float b    = lo ? ysp : ys;
            const float prod = __fmul_rn(b, csp[c]);
            out[c] = __fmaf_rn(a, cs0[c], lo ? -prod : prod);
        }
        st_bf16<W>((isq ? q_out : k_out) + (((long)batch * Hx + h) * T + t) * HD + lane * W, out);
    }
}

// -------------------------------------------------------------------------------------------- backward ----
template <int HD>
struct Tok2 {
    static constexpr int PAIRS = (HD / 2 + 31) / 32;
    float dza[PAIRS], dzb[PAIRS], xa[PAIRS], xb[PAIRS];
    float rstd;
};

template <bool IS_NEOX, int HD, int HPB, int STAGE, bool HAS_POS>
__global__ __launch_bounds__(HPB * 32)
void bwd2(const bf16* __restrict__ dz_q, const bf16* __restrict__ dz_k,
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
    constexpr int NT    = HPB * 32;                  // threads per block
    __shared__ __align__(16) float cs_s[2][STAGE][HD];

    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int t0 = blockIdx.x, b = blockIdx.z;
    const int head = blockIdx.y * HPB + wid;
    const int H = Hq + Hkv;
    const bool active = head < H;                    // idle warps still take part in the loads and barriers
    const bool is_q = head < Hq;
    const int  h    = is_q ? head : head - Hq;
    const int  Hx   = is_q ? Hq : Hkv;
    const bf16*  dz_base = is_q ? dz_q    : dz_k;
    const bf16*  in_base = is_q ? q_in    : k_in;
    const float* gamma   = is_q ? q_gamma : k_gamma;
    const float* rstd_in = is_q ? q_rstd  : k_rstd;
    bf16*        dx_base = is_q ? dq_in   : dk_in;
    const int n_tok = (T - t0 + BF16_BWD_T_CHUNKS - 1) / BF16_BWD_T_CHUNKS;
    const int n_stages = (n_tok + STAGE - 1) / STAGE;

    // block-wide load of the cos/sin rows of tokens [s*STAGE, s*STAGE + STAGE) into buffer s & 1 (float4 per thread)
    auto load_stage = [&](int s) {
        constexpr int V4 = STAGE * HD / 4;
        float4* dst = reinterpret_cast<float4*>(&cs_s[s & 1][0][0]);
        for (int i = threadIdx.x; i < V4; i += NT) {
            const int jj = i / (HD / 4), j = s * STAGE + jj;
            if (j >= n_tok) continue;
            const int t = t0 + BF16_BWD_T_CHUNKS * j;
            const int pos = HAS_POS ? positions[(long)b * T + t] : t + pos_offset;
            dst[i] = reinterpret_cast<const float4*>(cos_sin + (long)pos * HD)[i % (HD / 4)];
        }
    };

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
    float ga_r[PAIRS] = {}, gb_r[PAIRS] = {};
    if (active) {
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) { ga_r[p] = gamma[ia_r[p]]; gb_r[p] = gamma[ib_r[p]]; }
    }
    auto load_tok = [&](int j, Tok2<HD>& k) {
        const int t = t0 + BF16_BWD_T_CHUNKS * j;
        const bf16* dz = dz_base + (((long)b * Hx + h) * T + t) * HD;
        const bf16* x  = in_base + (((long)b * T + t) * Hx + h) * HD;
        k.rstd = rstd_in[((long)b * Hx + h) * T + t];
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            k.dza[p] = __bfloat162float(dz[ia_r[p]]);
            k.dzb[p] = __bfloat162float(dz[ib_r[p]]);
            k.xa[p]  = __bfloat162float(x[ia_r[p]]);
            k.xb[p]  = __bfloat162float(x[ib_r[p]]);
        }
    };

    Tok2<HD> cur, nxt;
    load_stage(0);
    if (active && n_tok > 0) load_tok(0, cur);
    __syncthreads();
    for (int s = 0; s < n_stages; ++s) {
        if (s + 1 < n_stages) load_stage(s + 1);     // the other buffer: free since the barrier that ended stage s-1
        const int jend = min(n_tok, (s + 1) * STAGE);
        for (int j = s * STAGE; j < jend; ++j) {
            if (active) {
                if (j + 1 < n_tok) load_tok(j + 1, nxt);
                // ---- the original per-token math (FusedRMSRoPE_bf16.cu lines 251-280), cos/sin from shared memory ----
                const float* cs_row = cs_s[s & 1][j - s * STAGE];
                const int t = t0 + BF16_BWD_T_CHUNKS * j;
                bf16* dx = dx_base + (((long)b * T + t) * Hx + h) * HD;
                const float rstd = cur.rstd;
                float xh_a[PAIRS], xh_b[PAIRS], dy_a[PAIRS], dy_b[PAIRS], g_a[PAIRS], g_b[PAIRS];
                float partial = 0.f;
                #pragma unroll
                for (int p = 0; p < PAIRS; ++p) {
                    const int i = lane + p * 32;
                    const float cs = cs_row[act[p] ? i : 0];
                    const float sn = cs_row[act[p] ? i + HALF : HALF];
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
        }
        __syncthreads();                              // stage s+1 loaded; buffer s & 1 free for stage s+2
    }

    if (active) {
        const int row = ((b * Hx) + h) * BF16_BWD_T_CHUNKS + t0;
        float* dst = (is_q ? dq_gpart : dk_gpart) + (long)row * HD;
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            if (!act[p]) continue;
            dst[ia_r[p]] = dg_a[p];
            dst[ib_r[p]] = dg_b[p];
        }
    }
}

// --------------------------------------------------------------------------------------------- reduce ----
template <int UNROLL>
__global__ void reduce3(const float* __restrict__ qpart, float* __restrict__ dq_gamma, int q_rows,
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

}  // namespace ropt2
}  // namespace cuda
}  // namespace OwnTensor
