#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "ops/helpers/FusedRMSRoPEKernels.h"

// SM90 (Hopper) fused QK-norm + RoPE forward. Gated at runtime behind
// ArchFamily::Hopper by the dispatcher in FusedRMSRoPE_fwd_sm90.cu; non-Hopper
// GPUs use the portable kernel there.

namespace OwnTensor {
namespace cuda {

template <int N> struct vec_of;
template <> struct vec_of<2> { using type = float2; };
template <> struct vec_of<4> { using type = float4; };

__device__ __forceinline__ float2 shfl_xor_vec(float2 v, int m) {
    float2 r;
    r.x = __shfl_xor_sync(0xffffffff, v.x, m);
    r.y = __shfl_xor_sync(0xffffffff, v.y, m);
    return r;
}
__device__ __forceinline__ float4 shfl_xor_vec(float4 v, int m) {
    float4 r;
    r.x = __shfl_xor_sync(0xffffffff, v.x, m);
    r.y = __shfl_xor_sync(0xffffffff, v.y, m);
    r.z = __shfl_xor_sync(0xffffffff, v.z, m);
    r.w = __shfl_xor_sync(0xffffffff, v.w, m);
    return r;
}

// Element accessors so the body can be written once for float2 and float4.
__device__ __forceinline__ float  vget(const float2& v, int i) { return i ? v.y : v.x; }
__device__ __forceinline__ float& vref(float2& v, int i)       { return i ? v.y : v.x; }
__device__ __forceinline__ float  vget(const float4& v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}
__device__ __forceinline__ float& vref(float4& v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

// HAS_POS (v2 packing, added 2026-10-10): false = the original kernel, position = t + pos_offset;
// true = per-token positions[b*T + t] (several chats per row, each restarting at 0).
// It is a template flag, not a runtime branch, so the HAS_POS=false instantiation is the
// pre-change kernel instruction for instruction (verified by a SASS diff, see
// Tests/kernel_tests/test_rope_positions.cu). positions is appended LAST so the existing
// parameters keep their constant-bank offsets.
template <bool IS_NEOX, int HD, int TOK_PER_WARP, int WARPS_PER_BLOCK, bool HAS_POS>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK)
void fused_qknorm_rope_forward_t_sm90(
    const float* __restrict__ q_in,          // [B, Hq,  T, hd]  pre-norm Q
    const float* __restrict__ k_in,          // [B, Hkv, T, hd]  pre-norm K
    float*       __restrict__ q_out,         // [B, Hq,  T, hd]  normed + roped Q
    float*       __restrict__ k_out,         // [B, Hkv, T, hd]  normed + roped K
    const float* __restrict__ q_gamma,       // [hd]
    const float* __restrict__ k_gamma,       // [hd]
    const float* __restrict__ cos_sin_cache, // [cache_seq_len, hd]
    float*       __restrict__ q_rstd,        // [B, Hq,  T]  1/rms to save for backward
    float*       __restrict__ k_rstd,        // [B, Hkv, T]  1/rms to save for backward
    int Hq, int Hkv, int T, int pos_offset, float eps = 1e-5f,
    const int32_t* __restrict__ positions = nullptr) {  // [B, T], read only when HAS_POS

    constexpr int HALF   = HD / 2;
    constexpr int VECS   = (HD   + 31) / 32;   // sum-of-squares steps per lane
    constexpr int PAIRS  = (HALF + 31) / 32;   // pair steps per lane: hd64 -> 1, hd128 -> 2

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    // Each warp owns TOK_PER_WARP CONTIGUOUS tokens, so its slice is
    // TOK_PER_WARP*HD contiguous floats with adjacent cos_sin rows. Do not
    // switch to a grid-stride split: it scatters both.
    const int tok_base = blockIdx.x * (WARPS_PER_BLOCK * TOK_PER_WARP)
                       + warp_id * TOK_PER_WARP;
    const int head_ind = blockIdx.y;
    const int batch    = blockIdx.z;

    const bool is_query = (head_ind < Hq);
    const int h = is_query ? head_ind : (head_ind - Hq);
    const int H = is_query ? Hq : Hkv;

    const float* in_base  = is_query ? q_in    : k_in;
    float*       out_base = is_query ? q_out   : k_out;
    const float* gamma    = is_query ? q_gamma : k_gamma;
    float*       rstd_out = is_query ? q_rstd  : k_rstd;

    // Warp-uniform, so this early-out is not divergence.
    if (tok_base >= T) return;

    // Row stride between consecutive tokens of this head is HD floats.
    const long head_base = (((long)batch * H + h) * T + tok_base) * HD;
    const long rstd_base = ((long)batch * H + h) * T + tok_base;

    if (IS_NEOX && (HD == 64 || HD == 128)) {
        // Vectorized NeoX path.
        constexpr int W  = HD / 32;                 // floats per lane: 2 or 4
        using vec_t = typename vec_of<W>::type;     // float2 or float4
        const bool lo = (lane < 16);                // this lane holds the a-half

        // gp is the partner's gamma: each lane computes BOTH outputs of its
        // pairs and stores only its own, so both gammas are needed.
        const vec_t g  = reinterpret_cast<const vec_t*>(gamma)[lane];
        const vec_t gp = shfl_xor_vec(g, 16);

        #pragma unroll
        for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
            const int tok_pos = tok_base + tt;
            if (TOK_PER_WARP > 1 && tok_pos >= T) break;   // tail guard

            const float* x = in_base  + head_base + (long)tt * HD;
            float*       y = out_base + head_base + (long)tt * HD;

            // One vector load serves BOTH the reduction and the rotate; do not
            // re-read x below.
            const vec_t v = reinterpret_cast<const vec_t*>(x)[lane];

            float part = 0.f;
            #pragma unroll
            for (int c = 0; c < W; ++c) { const float t = vget(v, c); part += t * t; }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffff, part, o);

            const float rstd = rsqrtf(part / float(HD) + eps);
            if (lane == 0) rstd_out[rstd_base + tt] = rstd;

            // a-half lanes read the cos block (0..HALF-1), b-half lanes the sin
            // block (HALF..HD-1), hence the partner exchange.
            // v2: with HAS_POS the cache row is the token's own position in its chat
            // (was always tok_pos + pos_offset; HAS_POS=false keeps exactly that).
            const float* cs_row = HAS_POS
                ? cos_sin_cache + (long)positions[(long)batch * T + tok_pos] * HD
                : cos_sin_cache + (long)(tok_pos + pos_offset) * HD;
            const vec_t cs0 = reinterpret_cast<const vec_t*>(cs_row)[lane];
            const vec_t csp = shfl_xor_vec(cs0, 16);

            const vec_t vp = shfl_xor_vec(v, 16);

            vec_t out;
            #pragma unroll
            for (int c = 0; c < W; ++c) {
                // Scaled value for this lane and for its partner.
                const float ys  = vget(v,  c) * rstd * vget(g,  c);
                const float ysp = vget(vp, c) * rstd * vget(gp, c);
                // a = low-half element of the pair, b = high-half; cos comes
                // from the a-half lane and sin from the b-half lane.
                const float a = lo ? ys  : ysp;
                const float b = lo ? ysp : ys;
                const float cs = lo ? vget(cs0, c) : vget(csp, c);
                const float sn = lo ? vget(csp, c) : vget(cs0, c);
                vref(out, c) = lo ? (a * cs - b * sn) : (a * sn + b * cs);
            }
            reinterpret_cast<vec_t*>(y)[lane] = out;
        }
    } else {
        // Scalar path: GPT-J interleaved, or any other HD. Interleaved pairs
        // (2i, 2i+1) are adjacent, so there is no partner exchange to vectorize
        // away; a vector rewrite here buys nothing.
        float ga_r[PAIRS], gb_r[PAIRS];
        #pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            const int i = lane + p * 32;
            if (HALF % 32 != 0 && i >= HALF) break;
            const int ia = IS_NEOX ? i          : (2 * i);
            const int ib = IS_NEOX ? (i + HALF) : (2 * i + 1);
            ga_r[p] = gamma[ia];
            gb_r[p] = gamma[ib];
        }

        #pragma unroll
        for (int tt = 0; tt < TOK_PER_WARP; ++tt) {
            const int tok_pos = tok_base + tt;
            if (TOK_PER_WARP > 1 && tok_pos >= T) break;   // tail guard

            const float* x = in_base  + head_base + (long)tt * HD;
            float*       y = out_base + head_base + (long)tt * HD;

            // Sum of squares over head dim, accumulated in fp32.
            float part = 0.0f;
            #pragma unroll
            for (int v = 0; v < VECS; ++v) {
                const int j = lane + v * 32;
                if (HD % 32 != 0 && j >= HD) break;
                float t = x[j];
                part += t * t;
            }
            // XOR-shuffle leaves the full sum in EVERY lane -> no broadcast needed.
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                part += __shfl_xor_sync(0xffffffff, part, o);
            }

            const float rstd = rsqrtf(part / float(HD) + eps);
            if (lane == 0) rstd_out[rstd_base + tt] = rstd;

            // Norm + RoPE. One lane owns whole pairs, so both halves stay in one thread.
            // v2: same position rule as the NeoX path above.
            const float* cs_row = HAS_POS
                ? cos_sin_cache + (long)positions[(long)batch * T + tok_pos] * HD
                : cos_sin_cache + (long)(tok_pos + pos_offset) * HD;

            #pragma unroll
            for (int p = 0; p < PAIRS; ++p) {
                const int i = lane + p * 32;
                if (HALF % 32 != 0 && i >= HALF) break;

                const int ia = IS_NEOX ? i          : (2 * i);
                const int ib = IS_NEOX ? (i + HALF) : (2 * i + 1);

                float ya = x[ia] * rstd * ga_r[p];
                float yb = x[ib] * rstd * gb_r[p];

                float cs = cs_row[i];
                float sn = cs_row[i + HALF];

                y[ia] = ya * cs - yb * sn;
                y[ib] = ya * sn + yb * cs;
            }
        }
    }
}

// Do not hoist the `lane == 0` rstd store out of the token loop to coalesce it
// (tried via a register array and via a shared-memory stage, both slower).
// Inside the loop the store hides under the next iteration's load latency;
// hoisting it serializes it behind the loop. The kernel stalls on loads, not stores.

// Tuning knobs, overridable at compile time.
// grid.x = ceil(T / (FWD_TOK_PER_WARP * FWD_WARPS_PER_BLOCK)).
#ifndef FWD_TOK_PER_WARP
#define FWD_TOK_PER_WARP 4
#endif
#ifndef FWD_WARPS_PER_BLOCK
#define FWD_WARPS_PER_BLOCK 8
#endif

void fused_qk_norm_rope_forward_sm90(
    const float* q_in, const float* k_in,
    float*       q_out, float* k_out,
    const float* q_gamma, const float* k_gamma,
    const float* cos_sin,
    float*       q_rstd, float* k_rstd,
    int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, float eps, bool interleaved,
    cudaStream_t stream,
    const int32_t* positions)
{
    constexpr int TPW = FWD_TOK_PER_WARP;
    constexpr int WPB = FWD_WARPS_PER_BLOCK;
    constexpr int TOK_PER_BLOCK = TPW * WPB;

    dim3 block(32 * WPB);
    dim3 grid((T + TOK_PER_BLOCK - 1) / TOK_PER_BLOCK, Hq + Hkv, B);

    // IS_NEOX == !interleaved. hd is specialized to the shipped shapes (64, 128).
    // Keep abort() rather than assert(): assert compiles out under -DNDEBUG and
    // an unsupported hd would then silently leave the output buffers untouched.
    // v2: HASP selects the per-token-positions instantiation (positions != nullptr).
#define LAUNCH_T(NEOX, HDV, HASP)                                              \
    fused_qknorm_rope_forward_t_sm90<NEOX, HDV, TPW, WPB, HASP>                \
        <<<grid, block, 0, stream>>>(                                          \
        q_in, k_in, q_out, k_out, q_gamma, k_gamma, cos_sin,                   \
        q_rstd, k_rstd, Hq, Hkv, T, pos_offset, eps, positions)

#define DISPATCH_HD(NEOX, HASP)                                                \
    switch (hd) {                                                              \
        case  64: LAUNCH_T(NEOX,  64, HASP);  break;                           \
        case 128: LAUNCH_T(NEOX, 128, HASP);  break;                           \
        default :                                                              \
            fprintf(stderr, "fused_qk_norm_rope_forward: unsupported hd=%d "   \
                            "(expected 64 or 128)\n", hd);                     \
            abort();                                                           \
    }

    if (positions == nullptr) {   // original path: position = t + pos_offset
        if (!interleaved) {   // NeoX (bluscript default)
            DISPATCH_HD(true, false);
        } else {              // GPT-J interleaved
            DISPATCH_HD(false, false);
        }
    } else {                      // v2: per-token positions[b*T + t]
        if (!interleaved) {
            DISPATCH_HD(true, true);
        } else {
            DISPATCH_HD(false, true);
        }
    }

#undef DISPATCH_HD
#undef LAUNCH_T
}

} // namespace cuda
} // namespace OwnTensor
