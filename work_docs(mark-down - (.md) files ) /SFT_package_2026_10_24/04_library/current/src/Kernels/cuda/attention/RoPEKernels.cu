// =============================================================================
// RoPE (Rotary Position Embedding) — CUDA kernels
//
// Two kernels live here:
//   1. rope_build_cache_kernel : fills a [seq_len, head_dim] cos/sin table.
//                                 columns [0       .. hd/2) = cos
//                                 columns [hd/2    .. hd)   = sin
//                                 inv_freq[i] = 1 / (base ^ (2i / hd))
//
//   2. rope_kernel<IS_NEOX, T>  : rotates Q (and optionally K) in place using
//                                 strided pointers. One block per token.
//                                 IS_NEOX=true  -> pairs (i, i + hd/2)
//                                 IS_NEOX=false -> pairs (2i, 2i + 1)
//                                 inverse=true  -> backward (sin negated;
//                                                  R(theta)^T = R(-theta)).
//                                 T = float (fp32) or __nv_bfloat16 (bf16).
//                                 bf16 loads upcast to fp32, rotate in fp32,
//                                 downcast on store. The cos/sin cache is ALWAYS
//                                 fp32 (dtype-independent, accurate, tiny).
//
// Layout contract (caller supplies strides, in elements):
//   q[b, t, h, d] is at: q + b * (Tlen * q_token_stride? no) — caller passes
//   the stride per token and per head; batch stride is computed as
//   Tlen * q_token_stride. This is enough to support both:
//     - separate Q tensor [B, T, nh_q, hd] contiguous
//         q_token_stride = nh_q * hd, q_head_stride = hd
//     - Q slice of packed QKV [B, T, (nh_q+nh_k+nh_v)*hd]
//         q_token_stride = (nh_q+nh_k+nh_v) * hd, q_head_stride = hd
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include "ops/helpers/RoPEKernels.h"
#include "ops/helpers/KernelDispatch.h"

namespace OwnTensor {
namespace cuda {

// Every launch in this file is checked (2026-10-31): a launch that the GPU refuses (for example a grid dimension over
// its limit) must stop the run, not leave its output unwritten.
static inline void rope_check_launch(const char* where) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": launch failed: " + cudaGetErrorString(e));
}

// Defined in arch/RoPE_sm90.cu; returns false when the shape or alignment is
// outside what it covers.
bool try_launch_rope_f32_sm90(
    float* q, float* k, const float* cache,
    int64_t B, int64_t Tlen, int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t q_batch_stride, int64_t k_batch_stride,
    int64_t pos_offset, bool interleaved, bool inverse,
    cudaStream_t stream);

// ---- load/store helpers: rotate in fp32 regardless of storage dtype ----
template <typename T> __device__ __forceinline__ float rope_to_f(T v);
template <> __device__ __forceinline__ float rope_to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float rope_to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T rope_from_f(float v);
template <> __device__ __forceinline__ float rope_from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 rope_from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---------------------------------------------------------------------------
// Cache builder kernel: one thread per (pos, i) where i in [0, hd/2).
// Writes cos at column i, sin at column i + hd/2.
// ---------------------------------------------------------------------------
// One thread per (pos, i) of the whole table, over a flat grid (2026-10-31: the old grid put pos in grid y, whose limit
// is 65,535, so a table of 65,536+ rows was never launched, and with head_dim 128 half of each 128-thread block was
// idle). Each entry is computed exactly as before.
__global__ void rope_build_cache_kernel(
    float* __restrict__ cache,
    int64_t seq_len, int64_t head_dim, float base)
{
    const int64_t half = head_dim / 2, n = seq_len * half;
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t pos = e / half, i = e % half;

        // inv_freq = 1 / (base ^ (2i / hd))
        float exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim);
        float inv_freq = 1.0f / powf(base, exponent);
        float angle    = static_cast<float>(pos) * inv_freq;

        float c = cosf(angle);
        float s = sinf(angle);

        float* row = cache + pos * head_dim;
        row[i]        = c;
        row[i + half] = s;
    }
}

// Blocks for a flat grid over n entries (256 threads each); the kernels loop, so the cap only bounds the grid.
static inline unsigned rope_flat_blocks(int64_t n) {
    const int64_t b = (n + 255) / 256;
    return static_cast<unsigned>(b < 1 ? 1 : (b > (1 << 20) ? (1 << 20) : b));
}

void rope_build_cache_cuda(float* cos_sin_cache,
                           int64_t seq_len, int64_t head_dim, float base)
{
    if (seq_len <= 0 || head_dim < 2) return;
    rope_build_cache_kernel<<<rope_flat_blocks(seq_len * (head_dim / 2)), 256>>>(cos_sin_cache, seq_len, head_dim, base);
    rope_check_launch("rope_build_cache_cuda");
}

// ---------------------------------------------------------------------------
// LeRoPE cache builder (arXiv:2607.10134): theta_hat[i] = exp(alpha[i]) * theta[i].
// Same grid/layout as rope_build_cache_kernel. One thread per (pos, i).
// alpha[i] == 0 makes this bit-for-bit equivalent to the fixed builder.
// ---------------------------------------------------------------------------
__global__ void rope_build_cache_learnable_kernel(
    float* __restrict__ cache,
    float* __restrict__ theta_hat_out,
    const float* __restrict__ alpha,
    int64_t seq_len, int64_t head_dim, float base)
{
    const int64_t half = head_dim / 2, n = seq_len * half;   // flat grid, as rope_build_cache_kernel (2026-10-31)
    for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < n;
         e += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t pos = e / half, i = e % half;

        float exponent  = (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim);
        float inv_freq  = 1.0f / powf(base, exponent);
        float theta_hat = __expf(alpha[i]) * inv_freq;
        float angle     = static_cast<float>(pos) * theta_hat;

        // theta_hat is per-frequency, not per-position: one row writes it.
        if (theta_hat_out != nullptr && pos == 0) theta_hat_out[i] = theta_hat;

        float* row = cache + pos * head_dim;
        row[i]        = cosf(angle);
        row[i + half] = sinf(angle);
    }
}

void rope_build_cache_learnable_cuda(float* cos_sin_cache,
                                     float* theta_hat_out,
                                     const float* alpha,
                                     int64_t seq_len, int64_t head_dim, float base)
{
    if (seq_len <= 0 || head_dim < 2) return;
    rope_build_cache_learnable_kernel<<<rope_flat_blocks(seq_len * (head_dim / 2)), 256>>>(
        cos_sin_cache, theta_hat_out, alpha, seq_len, head_dim, base);
    rope_check_launch("rope_build_cache_learnable_cuda");
}

// ---------------------------------------------------------------------------
// LeRoPE alpha gradient.
//
// For a pair rotated by phi = pos * theta_hat[i], with forward output (y0, y1):
//   d/dphi (y0, y1) = (-y1, y0)        [rotating the output by another 90 deg]
// so   dL/dphi = go0 * (-y1) + go1 * y0
// and  dL/dalpha[i] = sum dL/dphi * dphi/dalpha[i],  dphi/dalpha[i] = pos * theta_hat[i]
// (because theta_hat = exp(alpha) * theta, so d theta_hat / d alpha = theta_hat).
//
// Grid: (Tlen, B) — one block per token, matching rope_kernel (2026-10-31: the token is in grid x; grid y's limit
// is 65,535, so (B, Tlen) failed for Tlen >= 65,536). Each block
// reduces its own pairs into shared memory, then one atomicAdd per frequency
// per block. That keeps global atomic traffic at B*T*half instead of
// B*T*nh_total*half.
// ---------------------------------------------------------------------------
template <bool IS_NEOX>
__global__ void rope_alpha_backward_kernel(
    float* __restrict__ grad_alpha,
    const float* __restrict__ y_q,  const float* __restrict__ grad_y_q,
    const float* __restrict__ y_k,  const float* __restrict__ grad_y_k,
    const float* __restrict__ theta_hat,
    int64_t Tlen,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset)
{
    extern __shared__ float s_acc[];   // half floats

    int64_t t = blockIdx.x;   // token: grid x (no 65,535 limit)
    int64_t b = blockIdx.y;   // batch row: grid y

    int64_t half     = head_dim / 2;
    int64_t nh_total = nh_q + (y_k != nullptr ? nh_k : 0);
    int64_t pairs    = nh_total * half;

    for (int64_t i = threadIdx.x; i < half; i += blockDim.x) s_acc[i] = 0.0f;
    __syncthreads();

    const float pos = static_cast<float>(t + pos_offset);

    const float* yq_bt = y_q      + b * Tlen * q_token_stride + t * q_token_stride;
    const float* gq_bt = grad_y_q + b * Tlen * q_token_stride + t * q_token_stride;
    const float* yk_bt = (y_k != nullptr)
                       ? y_k      + b * Tlen * k_token_stride + t * k_token_stride : nullptr;
    const float* gk_bt = (grad_y_k != nullptr)
                       ? grad_y_k + b * Tlen * k_token_stride + t * k_token_stride : nullptr;

    for (int64_t idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
        int64_t h = idx / half;
        int64_t i = idx - h * half;

        const float* yp;
        const float* gp;
        int64_t head_stride, local_h;
        if (h < nh_q) {
            yp = yq_bt; gp = gq_bt; head_stride = q_head_stride; local_h = h;
        } else {
            yp = yk_bt; gp = gk_bt; head_stride = k_head_stride; local_h = h - nh_q;
        }

        const float* y_head = yp + local_h * head_stride;
        const float* g_head = gp + local_h * head_stride;

        int64_t a_idx, b_idx;
        if (IS_NEOX) { a_idx = i;         b_idx = i + half;     }
        else         { a_idx = 2 * i;     b_idx = 2 * i + 1;    }

        float y0 = y_head[a_idx], y1 = y_head[b_idx];
        float g0 = g_head[a_idx], g1 = g_head[b_idx];

        // dL/dphi for this pair.
        float dphi = g0 * (-y1) + g1 * y0;
        atomicAdd(&s_acc[i], dphi);
    }
    __syncthreads();

    // One global atomic per frequency per block.
    for (int64_t i = threadIdx.x; i < half; i += blockDim.x) {
        float v = s_acc[i] * pos * theta_hat[i];
        if (v != 0.0f) atomicAdd(&grad_alpha[i], v);
    }
}

void rope_alpha_backward_cuda(
    float* grad_alpha,
    const float* y_q,  const float* grad_y_q,
    const float* y_k,  const float* grad_y_k,
    const float* theta_hat,
    int64_t B, int64_t Tlen,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset,
    bool interleaved)
{
    int64_t half     = head_dim / 2;
    int64_t nh_total = nh_q + (y_k != nullptr ? nh_k : 0);
    int64_t pairs    = nh_total * half;

    int threads = static_cast<int>(pairs < 1024 ? pairs : 1024);
    if (threads <= 0) return;

    dim3 grid(static_cast<unsigned>(Tlen), static_cast<unsigned>(B));
    dim3 block(threads);
    size_t shmem = static_cast<size_t>(half) * sizeof(float);

    if (interleaved) {
        rope_alpha_backward_kernel<false><<<grid, block, shmem>>>(
            grad_alpha, y_q, grad_y_q, y_k, grad_y_k, theta_hat,
            Tlen, nh_q, nh_k, head_dim,
            q_token_stride, q_head_stride, k_token_stride, k_head_stride,
            pos_offset);
    } else {
        rope_alpha_backward_kernel<true><<<grid, block, shmem>>>(
            grad_alpha, y_q, grad_y_q, y_k, grad_y_k, theta_hat,
            Tlen, nh_q, nh_k, head_dim,
            q_token_stride, q_head_stride, k_token_stride, k_head_stride,
            pos_offset);
    }
    rope_check_launch("rope_alpha_backward");
}

// ---------------------------------------------------------------------------
// Core RoPE kernel.
//
// Grid:  (Tlen, B)   (the token in grid x since 2026-10-31; grid y's limit is 65,535)
// Block: up to 1024 threads, striding over (head, pair_index).
//
// One block handles ONE (b, t) pair. Within the block, threads are flattened
// over (h, i) where h in [0, nh_total) and i in [0, hd/2). We let nh_total =
// nh_q + nh_k so heads [0..nh_q) point into Q's plane and heads [nh_q..nh_total)
// point into K's plane (K may be nullptr -> Q-only).
// ---------------------------------------------------------------------------
template <bool IS_NEOX, typename T>
__global__ void rope_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const float* __restrict__ cos_sin_cache,
    int64_t Tlen,
    int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset,
    bool inverse)
{
    int64_t t = blockIdx.x;   // token: grid x (no 65,535 limit)
    int64_t b = blockIdx.y;   // batch row: grid y

    int64_t half     = head_dim / 2;
    int64_t nh_total = nh_q + (k != nullptr ? nh_k : 0);
    int64_t pairs    = nh_total * half;

    // Cache row for this token's absolute position.
    int64_t pos = t + pos_offset;

    // Bounds guard: the cache holds only `cache_seq_len` rows. A pos beyond it
    // (sequence longer than the built cache, or a pos_offset overshoot) would be
    // an out-of-bounds device read. The host validates and throws before launch;
    // this is the in-kernel backstop. pos is identical for every thread in the
    // block (one block per token), so the early return is warp-uniform.
    if (pos < 0 || pos >= cache_seq_len) return;

    const float* cache_row = cos_sin_cache + pos * head_dim;

    // Pointers to this (b, t) plane's first element of Q and K.
    // Batch stride = Tlen * token_stride.
    T* q_bt = q + b * Tlen * q_token_stride + t * q_token_stride;
    T* k_bt = (k != nullptr)
              ? k + b * Tlen * k_token_stride + t * k_token_stride
              : nullptr;

    for (int64_t idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
        int64_t h = idx / half;
        int64_t i = idx - h * half;

        // Pick the right plane (Q or K) and the right stride.
        T* plane;
        int64_t head_stride;
        int64_t local_h;
        if (h < nh_q) {
            plane       = q_bt;
            head_stride = q_head_stride;
            local_h     = h;
        } else {
            plane       = k_bt;
            head_stride = k_head_stride;
            local_h     = h - nh_q;
        }

        T* head_ptr = plane + local_h * head_stride;

        // Pair indices depend on layout style.
        int64_t a_idx, b_idx;
        if (IS_NEOX) {
            a_idx = i;
            b_idx = i + half;
        } else {
            a_idx = 2 * i;
            b_idx = 2 * i + 1;
        }

        // Load -> fp32 (bf16 upcast), rotate in fp32, store -> T (bf16 downcast).
        float x = rope_to_f<T>(head_ptr[a_idx]);
        float y = rope_to_f<T>(head_ptr[b_idx]);

        float c = cache_row[i];
        float s = cache_row[i + half];
        if (inverse) s = -s;

        // Standard 2D rotation:
        //   x' = x*cos - y*sin
        //   y' = x*sin + y*cos
        float x_rot = x * c - y * s;
        float y_rot = x * s + y * c;

        head_ptr[a_idx] = rope_from_f<T>(x_rot);
        head_ptr[b_idx] = rope_from_f<T>(y_rot);
    }
}

// ---------------------------------------------------------------------------
// Shared launch helper (dtype-generic). Both the fp32 and bf16 public
// launchers funnel through this so the grid/block/dispatch logic lives once.
// ---------------------------------------------------------------------------
template <typename T>
static void launch_rope(
    T* q, T* k,
    const float* cos_sin_cache,
    int64_t B, int64_t Tlen,
    int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset,
    bool interleaved,
    bool inverse)
{
    int64_t half     = head_dim / 2;
    int64_t nh_total = nh_q + (k != nullptr ? nh_k : 0);
    int64_t pairs    = nh_total * half;

    int threads = static_cast<int>(pairs < 1024 ? pairs : 1024);
    if (threads <= 0) return;

    dim3 grid(static_cast<unsigned>(Tlen), static_cast<unsigned>(B));
    dim3 block(threads);

    if (interleaved) {
        rope_kernel<false, T><<<grid, block>>>(
            q, k, cos_sin_cache,
            Tlen, cache_seq_len, nh_q, nh_k, head_dim,
            q_token_stride, q_head_stride,
            k_token_stride, k_head_stride,
            pos_offset, inverse);
    } else {
        rope_kernel<true, T><<<grid, block>>>(
            q, k, cos_sin_cache,
            Tlen, cache_seq_len, nh_q, nh_k, head_dim,
            q_token_stride, q_head_stride,
            k_token_stride, k_head_stride,
            pos_offset, inverse);
    }
    rope_check_launch("rope_forward");
}

// ---------------------------------------------------------------------------
// Public launchers.
// ---------------------------------------------------------------------------
void rope_forward_cuda(
    float* q, float* k,
    const float* cos_sin_cache,
    int64_t B, int64_t Tlen,
    int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset,
    bool interleaved,
    bool inverse)
{
    // Fast path returns false for shapes or alignments it does not cover, so
    // falling through is always safe. Carries a numerics caveat: see the header
    // of arch/RoPE_sm90.cu before trusting gradients through it.
    if (get_arch() == ArchFamily::Hopper) {
        if (try_launch_rope_f32_sm90(
                q, k, cos_sin_cache, B, Tlen, cache_seq_len, nh_q, nh_k, head_dim,
                q_token_stride, q_head_stride, k_token_stride, k_head_stride,
                /*q_batch_stride=*/Tlen * q_token_stride,
                /*k_batch_stride=*/Tlen * k_token_stride,
                pos_offset, interleaved, inverse, /*stream=*/0)) {
            return;
        }
    }

    launch_rope<float>(q, k, cos_sin_cache, B, Tlen, cache_seq_len, nh_q, nh_k,
                       head_dim, q_token_stride, q_head_stride,
                       k_token_stride, k_head_stride, pos_offset, interleaved, inverse);
}

void rope_forward_cuda(
    bfloat16_t* q, bfloat16_t* k,
    const float* cos_sin_cache,
    int64_t B, int64_t Tlen,
    int64_t cache_seq_len,
    int64_t nh_q, int64_t nh_k, int64_t head_dim,
    int64_t q_token_stride, int64_t q_head_stride,
    int64_t k_token_stride, int64_t k_head_stride,
    int64_t pos_offset,
    bool interleaved,
    bool inverse)
{
    launch_rope<__nv_bfloat16>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        cos_sin_cache, B, Tlen, cache_seq_len, nh_q, nh_k, head_dim,
        q_token_stride, q_head_stride, k_token_stride, k_head_stride,
        pos_offset, interleaved, inverse);
}

} // namespace cuda
} // namespace OwnTensor
