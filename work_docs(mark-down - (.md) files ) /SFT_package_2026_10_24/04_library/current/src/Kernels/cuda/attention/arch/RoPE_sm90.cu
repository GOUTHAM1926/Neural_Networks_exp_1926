// RoPE sm_90 fp32 fast path: head_dim 64/128, NeoX and GPT-J. The generic
// rope_kernel in ../RoPEKernels.cu stays the correctness path (bf16, other
// head_dims, non-Hopper).
//
// NUMERICS WARNING: this path's measured error is ~700x the peer kernels' on
// the same shapes and is not root-caused. On RoPE-adjacent gradient drift,
// disable the Hopper gate in ../RoPEKernels.cu and re-measure against the
// generic kernel first.
//
// No separate backward kernel: RoPE is orthogonal, so R(theta)^T = R(-theta)
// and backward is this forward with sin negated via INVERSE.

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "ops/helpers/RoPEKernels.h"

namespace OwnTensor {
namespace cuda {
namespace sm90 {

// Every launch here is checked (2026-10-31): a refused launch must stop the run, not leave Q / K unrotated.
static inline void check_launch(const char* where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": launch failed: " + cudaGetErrorString(e));
}

template <int VEC> struct rope_vec;
template <> struct rope_vec<1> { using type = float;  };
template <> struct rope_vec<2> { using type = float2; };
template <> struct rope_vec<4> { using type = float4; };

__device__ __forceinline__ float& rope_at(float&  v, int)   { return v; }
__device__ __forceinline__ float& rope_at(float2& v, int j) { return j == 0 ? v.x : v.y; }
__device__ __forceinline__ float& rope_at(float4& v, int j) { return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w)); }

// NeoX (half-split), fp32, vectorized. One block per (b, t); HALF/VEC lanes per
// head x head groups, so a group's head index is warp-uniform and the Q/K plane
// select resolves once. HPT stages that many heads as loads -> rotate -> stores;
// the launcher pins it to 1.
template <int HALF, int VEC, int HPT, bool INVERSE>
__global__ void rope_kernel_neox_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_sin_cache,
    int64_t Tlen,
    int64_t cache_seq_len,
    int nh_q, int nh_k,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t q_batch_stride, int64_t k_batch_stride,
    int64_t pos_offset)
{
    using V = typename rope_vec<VEC>::type;

    constexpr int LANES    = HALF / VEC;   // threads covering one head's half
    constexpr int HEAD_DIM = HALF * 2;

    const int64_t t = blockIdx.x;   // token: grid x (2026-10-31; grid y's limit is 65,535)
    const int64_t b = blockIdx.y;

    const int64_t pos = t + pos_offset;
    if (pos < 0 || pos >= cache_seq_len) return;

    const int nh_total = nh_q + (k != nullptr ? nh_k : 0);

    // 64-bit indexing once here so the loop body stays 32-bit. The batch stride
    // is explicit rather than Tlen*token_stride, so layouts other than
    // [B,T,H,hd] launch once over (B,T) instead of per batch element.
    float* q_bt = q + b * q_batch_stride + t * q_token_stride;
    float* k_bt = (k != nullptr)
                  ? k + b * k_batch_stride + t * k_token_stride
                  : nullptr;

    const float* cache_row = cos_sin_cache + pos * HEAD_DIM;

    const int tid    = threadIdx.x;
    const int lane   = tid % LANES;
    const int group0 = tid / LANES;
    const int groups = blockDim.x / LANES;

    // cos/sin depend on (pos, i) only: hoisted out of the head loop.
    V c_v = *reinterpret_cast<const V*>(cache_row + lane * VEC);
    V s_v = *reinterpret_cast<const V*>(cache_row + HALF + lane * VEC);

    const int head_stride_iter = groups * HPT;

    for (int base_h = group0 * HPT; base_h < nh_total; base_h += head_stride_iter) {
        V*   pa[HPT];
        V*   pb[HPT];
        V    xa[HPT];
        V    xb[HPT];
        bool live[HPT];

        // All loads issue before any consumer so they overlap; do not merge
        // this loop with the rotate loop below.
        #pragma unroll
        for (int u = 0; u < HPT; ++u) {
            const int h = base_h + u;
            live[u] = (h < nh_total);
            if (!live[u]) { pa[u] = nullptr; pb[u] = nullptr; continue; }

            float*  plane;
            int64_t hstride;
            int     local_h;
            if (h < nh_q) {
                plane = q_bt; hstride = q_head_stride; local_h = h;
            } else {
                plane = k_bt; hstride = k_head_stride; local_h = h - nh_q;
            }
            float* head_ptr = plane + local_h * hstride;

            pa[u] = reinterpret_cast<V*>(head_ptr + lane * VEC);
            pb[u] = reinterpret_cast<V*>(head_ptr + HALF + lane * VEC);

            xa[u] = *pa[u];
            xb[u] = *pb[u];
        }

        V ya[HPT], yb[HPT];
        #pragma unroll
        for (int u = 0; u < HPT; ++u) {
            if (!live[u]) continue;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float x = rope_at(xa[u], j);
                const float y = rope_at(xb[u], j);
                const float c = rope_at(c_v, j);
                const float s = INVERSE ? -rope_at(s_v, j) : rope_at(s_v, j);
                rope_at(ya[u], j) = __fmaf_rn(-y, s, x * c);
                rope_at(yb[u], j) = __fmaf_rn( y, c, x * s);
            }
        }

        #pragma unroll
        for (int u = 0; u < HPT; ++u) {
            if (!live[u]) continue;
            *pa[u] = ya[u];
            *pb[u] = yb[u];
        }
    }
}

// GPT-J (interleaved), fp32. The pair (2i, 2i+1) is adjacent, so one float2
// load fetches both halves; cos/sin still come from the half-split table.
// Never benchmarked: the RoPE sweep to date is NeoX fp32 only.
template <int HALF, bool INVERSE>
__global__ void rope_kernel_gptj_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_sin_cache,
    int64_t Tlen,
    int64_t cache_seq_len,
    int nh_q, int nh_k,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset)
{
    constexpr int LANES    = HALF;         // one thread per pair
    constexpr int HEAD_DIM = HALF * 2;

    const int64_t t = blockIdx.x;   // token: grid x (2026-10-31; grid y's limit is 65,535)
    const int64_t b = blockIdx.y;

    const int64_t pos = t + pos_offset;
    if (pos < 0 || pos >= cache_seq_len) return;

    const int nh_total = nh_q + (k != nullptr ? nh_k : 0);

    float* q_bt = q + b * Tlen * q_token_stride + t * q_token_stride;
    float* k_bt = (k != nullptr)
                  ? k + b * Tlen * k_token_stride + t * k_token_stride
                  : nullptr;

    const float* cache_row = cos_sin_cache + pos * HEAD_DIM;

    const int tid       = threadIdx.x;
    const int lane      = tid % LANES;
    const int head0     = tid / LANES;
    const int head_step = blockDim.x / LANES;

    const float c = cache_row[lane];
    const float s = INVERSE ? -cache_row[HALF + lane] : cache_row[HALF + lane];

    for (int h = head0; h < nh_total; h += head_step) {
        float*  plane;
        int64_t head_stride;
        int     local_h;
        if (h < nh_q) {
            plane = q_bt; head_stride = q_head_stride; local_h = h;
        } else {
            plane = k_bt; head_stride = k_head_stride; local_h = h - nh_q;
        }

        // (2*lane, 2*lane+1) is 8-byte aligned whenever the head is; the
        // launcher's f2_ok gate guarantees that.
        float2* p = reinterpret_cast<float2*>(plane + local_h * head_stride) + lane;
        const float2 v = *p;

        float2 o;
        o.x = __fmaf_rn(-v.y, s, v.x * c);
        o.y = __fmaf_rn( v.y, c, v.x * s);
        *p = o;
    }
}

// Launch helpers.

template <int HALF, int VEC, int HPT, bool INVERSE>
static void launch_neox_hpt(
    float* q, float* k, const float* cache,
    int64_t B, int64_t Tlen, int64_t cache_seq_len,
    int nh_q, int nh_k,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t q_batch_stride, int64_t k_batch_stride,
    int64_t pos_offset, cudaStream_t stream)
{
    constexpr int LANES = HALF / VEC;
    const int nh_total = nh_q + (k != nullptr ? nh_k : 0);

    int groups  = (nh_total + HPT - 1) / HPT;
    int threads = LANES * groups;
    // Halving groups past the block limit is safe: the kernel's outer head loop
    // picks up the remainder.
    while (threads > 1024) { groups >>= 1; threads = LANES * groups; }
    if (threads < LANES) threads = LANES;

    dim3 grid(static_cast<unsigned>(Tlen), static_cast<unsigned>(B));
    rope_kernel_neox_f32<HALF, VEC, HPT, INVERSE><<<grid, dim3(threads), 0, stream>>>(
        q, k, cache, Tlen, cache_seq_len, nh_q, nh_k,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        q_batch_stride, k_batch_stride, pos_offset);
    check_launch("rope sm90 neox");
}

// HPT pinned to 1. Deeper staging (HPT=2, 4) was measured and lost both times,
// tracking the occupancy given up; do not re-raise it. The machinery stays for
// a future variant with a different register budget.
template <int HALF, int VEC, bool INVERSE>
static void launch_neox_f32(
    float* q, float* k, const float* cache,
    int64_t B, int64_t Tlen, int64_t cache_seq_len,
    int nh_q, int nh_k,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t q_batch_stride, int64_t k_batch_stride,
    int64_t pos_offset, cudaStream_t stream)
{
    launch_neox_hpt<HALF, VEC, 1, INVERSE>(
        q, k, cache, B, Tlen, cache_seq_len, nh_q, nh_k,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        q_batch_stride, k_batch_stride, pos_offset, stream);
}

template <int HALF, bool INVERSE>
static void launch_gptj_f32(
    float* q, float* k, const float* cache,
    int64_t B, int64_t Tlen, int64_t cache_seq_len,
    int nh_q, int nh_k,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset, cudaStream_t stream)
{
    constexpr int LANES = HALF;
    const int nh_total = nh_q + (k != nullptr ? nh_k : 0);

    int heads_in_flight = 1024 / LANES;
    if (heads_in_flight > nh_total) heads_in_flight = nh_total;
    if (heads_in_flight < 1) heads_in_flight = 1;
    const int threads = LANES * heads_in_flight;

    dim3 grid(static_cast<unsigned>(Tlen), static_cast<unsigned>(B));
    rope_kernel_gptj_f32<HALF, INVERSE><<<grid, dim3(threads), 0, stream>>>(
        q, k, cache, Tlen, cache_seq_len, nh_q, nh_k,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        pos_offset);
    check_launch("rope sm90 gptj");
}

} // namespace sm90

// Returns true if a specialized kernel was launched. Every rejection is a
// correctness gate (unsupported head_dim, or alignment the vector loads
// require), so a false return is always safe to fall back on.
bool try_launch_rope_f32_sm90(
    float* q, float* k, const float* cache,
    int64_t B, int64_t Tlen, int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t q_batch_stride, int64_t k_batch_stride,
    int64_t pos_offset, bool interleaved, bool inverse,
    cudaStream_t stream)
{
    const int nq = static_cast<int>(nh_q);
    const int nk = static_cast<int>(nh_k);

    // Vector loads need 16-byte aligned base pointers and per-head strides.
    auto aligned16 = [&](int64_t elems) { return (elems % 4) == 0; };
    const bool ptr_ok =
        (reinterpret_cast<uintptr_t>(q) % 16 == 0) &&
        (k == nullptr || reinterpret_cast<uintptr_t>(k) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(cache) % 16 == 0);
    const bool stride_ok =
        aligned16(q_head_stride) && aligned16(q_token_stride) &&
        (k == nullptr || (aligned16(k_head_stride) && aligned16(k_token_stride)));
    const bool vec4_ok = ptr_ok && stride_ok;

    if (!interleaved) {
        // VEC=4 when alignment allows, else scalar, which is still specialized.
        #define NEOX_CASE(HD_HALF)                                                    \
            if (head_dim == (HD_HALF) * 2) {                                          \
                if (vec4_ok) {                                                        \
                    if (inverse) sm90::launch_neox_f32<HD_HALF, 4, true >(            \
                        q, k, cache, B, Tlen, cache_seq_len, nq, nk,                  \
                        q_token_stride, q_head_stride, k_token_stride,                \
                        k_head_stride, q_batch_stride, k_batch_stride,                \
                        pos_offset, stream);                                          \
                    else         sm90::launch_neox_f32<HD_HALF, 4, false>(            \
                        q, k, cache, B, Tlen, cache_seq_len, nq, nk,                  \
                        q_token_stride, q_head_stride, k_token_stride,                \
                        k_head_stride, q_batch_stride, k_batch_stride,                \
                        pos_offset, stream);                                          \
                } else {                                                              \
                    if (inverse) sm90::launch_neox_f32<HD_HALF, 1, true >(            \
                        q, k, cache, B, Tlen, cache_seq_len, nq, nk,                  \
                        q_token_stride, q_head_stride, k_token_stride,                \
                        k_head_stride, q_batch_stride, k_batch_stride,                \
                        pos_offset, stream);                                          \
                    else         sm90::launch_neox_f32<HD_HALF, 1, false>(            \
                        q, k, cache, B, Tlen, cache_seq_len, nq, nk,                  \
                        q_token_stride, q_head_stride, k_token_stride,                \
                        k_head_stride, q_batch_stride, k_batch_stride,                \
                        pos_offset, stream);                                          \
                }                                                                     \
                return true;                                                          \
            }
        NEOX_CASE(32)    // head_dim 64
        NEOX_CASE(64)    // head_dim 128
        #undef NEOX_CASE
        return false;
    }

    // GPT-J. The float2 pair load needs 8-byte alignment.
    const bool f2_ok =
        (reinterpret_cast<uintptr_t>(q) % 8 == 0) &&
        (k == nullptr || reinterpret_cast<uintptr_t>(k) % 8 == 0) &&
        (q_head_stride % 2 == 0) && (q_token_stride % 2 == 0) &&
        (k == nullptr || ((k_head_stride % 2 == 0) && (k_token_stride % 2 == 0)));
    if (!f2_ok) return false;

    #define GPTJ_CASE(HD_HALF)                                                        \
        if (head_dim == (HD_HALF) * 2) {                                              \
            if (inverse) sm90::launch_gptj_f32<HD_HALF, true >(                       \
                q, k, cache, B, Tlen, cache_seq_len, nq, nk,                          \
                q_token_stride, q_head_stride, k_token_stride, k_head_stride,         \
                pos_offset, stream);                                                  \
            else         sm90::launch_gptj_f32<HD_HALF, false>(                       \
                q, k, cache, B, Tlen, cache_seq_len, nq, nk,                          \
                q_token_stride, q_head_stride, k_token_stride, k_head_stride,         \
                pos_offset, stream);                                                  \
            return true;                                                              \
        }
    GPTJ_CASE(32)    // head_dim 64
    GPTJ_CASE(64)    // head_dim 128
    #undef GPTJ_CASE
    return false;
}

} // namespace cuda
} // namespace OwnTensor
