#include <cuda_runtime.h>
#include "headtail_kernel.cuh"

// ---------------------------------------------------------------------------
// HeadTail loadbalance kernel
//
// For a tensor [outer, T, inner] (outer = B*H, inner = D), permutes along
// the T dimension:
//   dst[outer_idx, 2k,   inner_idx] = src[outer_idx, k,         inner_idx]
//   dst[outer_idx, 2k+1, inner_idx] = src[outer_idx, T - 1 - k, inner_idx]
//
// One thread per element in the output tensor.
// ---------------------------------------------------------------------------
__global__ void headtail_loadbalance_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose flat index into (outer, seq, inner)
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t seq_idx = temp % seq_len;
    int64_t outer_idx = temp / seq_len;

    // Map output seq_idx to source seq_idx
    int64_t src_seq;
    if (seq_idx % 2 == 0) {
        // Even output position: head element
        src_seq = seq_idx / 2;
    } else {
        // Odd output position: tail element (reversed)
        src_seq = seq_len - 1 - (seq_idx / 2);
    }

    int64_t src_idx = outer_idx * (seq_len * inner_size)
                    + src_seq * inner_size
                    + inner_idx;

    dst[idx] = src[src_idx];
}

// ---------------------------------------------------------------------------
// HeadTail unloadbalance kernel (inverse permutation)
//
// Reverses the loadbalance mapping:
//   dst[outer_idx, k,         inner_idx] = src[outer_idx, 2k,   inner_idx]
//   dst[outer_idx, T - 1 - k, inner_idx] = src[outer_idx, 2k+1, inner_idx]
//
// Equivalently, for each output position dst_seq:
//   if dst_seq < T/2:  src_seq = 2 * dst_seq           (was a head element)
//   else:              src_seq = 2 * (T - 1 - dst_seq) + 1  (was a tail element)
// ---------------------------------------------------------------------------
__global__ void headtail_unloadbalance_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose flat index into (outer, seq, inner)
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t dst_seq = temp % seq_len;
    int64_t outer_idx = temp / seq_len;

    // Map output dst_seq back to the loadbalanced source position
    int64_t src_seq;
    if (dst_seq < seq_len / 2) {
        // This position was placed at even index 2*dst_seq during loadbalance
        src_seq = 2 * dst_seq;
    } else {
        // This position was placed at odd index 2*(T-1-dst_seq)+1 during loadbalance
        src_seq = 2 * (seq_len - 1 - dst_seq) + 1;
    }

    int64_t src_idx = outer_idx * (seq_len * inner_size)
                    + src_seq * inner_size
                    + inner_idx;

    dst[idx] = src[src_idx];
}

// ---------------------------------------------------------------------------
// Launcher functions
// ---------------------------------------------------------------------------
void launch_headtail_loadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    cudaStream_t stream)
{
    int64_t total = outer_size * seq_len * inner_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    headtail_loadbalance_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, outer_size, seq_len, inner_size, total);
}

void launch_headtail_unloadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    cudaStream_t stream)
{
    int64_t total = outer_size * seq_len * inner_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    headtail_unloadbalance_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, outer_size, seq_len, inner_size, total);
}