#pragma once

#include <cuda_runtime.h>

// HeadTail permutation kernel launcher declarations.
//
// HeadTail interleaves head and reversed-tail of the sequence:
//   output[2k]   = input[k]           k = 0, ..., T/2-1
//   output[2k+1] = input[T - 1 - k]  k = 0, ..., T/2-1
//
// Result for T=8: [0, 7, 1, 6, 2, 5, 3, 4]
//
// The inverse (unloadbalance) reverses this mapping:
//   output[k]       = input[2k]       k = 0, ..., T/2-1
//   output[T-1-k]   = input[2k+1]    k = 0, ..., T/2-1

void launch_headtail_loadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,   // product of dims before chunkdim (B * H for 4D with chunkdim=2)
    int64_t seq_len,      // size of chunkdim (T)
    int64_t inner_size,   // product of dims after chunkdim (D for 4D with chunkdim=2)
    cudaStream_t stream);

void launch_headtail_unloadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    cudaStream_t stream);