#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "fused_transpose_kernel.cuh"

// =============================================================================
// CUDA Kernels (use raw int64_t — std::vector is host-only)
// =============================================================================

// Kernel for dim 0 sharding: [D0, D1, D2] -> [D0_local, D1, D2]
// For 2D: treat as [D0, D1, 1] -> [D0_local, D1, 1]
__global__ void shard_dim0_kernel(
    float* src,
    float* dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D0_local,
    int rank,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Destination coordinates [d0_local, d1, d2] in output [D0_local, D1, D2]
        int64_t d2 = idx % D2;
        int64_t temp = idx / D2;
        int64_t d1 = temp % D1;
        int64_t d0_local = temp / D1;
        
        // Global d0 coordinate
        int64_t d0_global = rank * D0_local + d0_local;
        
        // Source index in [D0, D1, D2] layout
        int64_t src_idx = d0_global * (D1 * D2) + d1 * D2 + d2;
        
        dst[idx] = src[src_idx];
    }
}

// Kernel for dim 1 sharding: [D0, D1, D2] -> [D0, D1_local, D2]
// For 2D: treat as [1, D0, D1] -> [1, D0_local, D1], i.e. shard dim 0 of 2D
__global__ void shard_dim1_kernel(
    float* src,
    float* dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D1_local,
    int rank,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Destination coordinates [d0, d1_local, d2] in output [D0, D1_local, D2]
        int64_t d2 = idx % D2;
        int64_t temp = idx / D2;
        int64_t d1_local = temp % D1_local;
        int64_t d0 = temp / D1_local;
        
        // Global d1 coordinate
        int64_t d1_global = rank * D1_local + d1_local;
        
        // Source index in [D0, D1, D2] layout
        int64_t src_idx = d0 * (D1 * D2) + d1_global * D2 + d2;
        
        dst[idx] = src[src_idx];
    }
}

// Kernel for dim 2 sharding: [D0, D1, D2] -> [D0, D1, D2_local]
// For 2D: treat as [1, D0, D1] -> [1, D0, D1_local], i.e. shard dim 1 of 2D
__global__ void shard_dim2_kernel(
    float* src,
    float* dst,
    int64_t D0, int64_t D1, int64_t D2,
    int64_t D2_local,
    int rank,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Destination coordinates [d0, d1, d2_local] in output [D0, D1, D2_local]
        int64_t d2_local = idx % D2_local;
        int64_t temp = idx / D2_local;
        int64_t d1 = temp % D1;
        int64_t d0 = temp / D1;
        
        // Global d2 coordinate
        int64_t d2_global = rank * D2_local + d2_local;
        
        // Source index in [D0, D1, D2] layout
        int64_t src_idx = d0 * (D1 * D2) + d1 * D2 + d2_global;
        
        dst[idx] = src[src_idx];
    }
}

// =============================================================================
// Launch Wrappers (host code — can use std::vector)
// For 2D shapes, prepend D0=1 to normalize to 3D
// =============================================================================

static void normalize_shape_3d(const std::vector<int64_t>& shape, int64_t& D0, int64_t& D1, int64_t& D2) {
    if (shape.size() == 2) {
        // 2D [R, C] -> 3D [R, C, 1] so dim indices 0,1 stay the same
        D0 = shape[0];
        D1 = shape[1];
        D2 = 1;
    } else if (shape.size() == 3) {
        D0 = shape[0];
        D1 = shape[1];
        D2 = shape[2];
    } else {
        throw std::runtime_error("launch_shard_kernel: shape must be 2D or 3D, got " + std::to_string(shape.size()) + "D");
    }
}

void launch_shard_dim0_kernel(
    float* d_src,
    float* d_dst,
    const std::vector<int64_t>& shape,
    int64_t D0_local,
    int rank,
    int64_t total_elements,
    cudaStream_t stream
) {
    int64_t D0, D1, D2;
    normalize_shape_3d(shape, D0, D1, D2);
    
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    shard_dim0_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, D0, D1, D2, D0_local, rank, total_elements
    );
}

void launch_shard_dim1_kernel(
    float* d_src,
    float* d_dst,
    const std::vector<int64_t>& shape,
    int64_t D1_local,
    int rank,
    int64_t total_elements,
    cudaStream_t stream
) {
    int64_t D0, D1, D2;
    normalize_shape_3d(shape, D0, D1, D2);
    
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    shard_dim1_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, D0, D1, D2, D1_local, rank, total_elements
    );
}

void launch_shard_dim2_kernel(
    float* d_src,
    float* d_dst,
    const std::vector<int64_t>& shape,
    int64_t D2_local,
    int rank,
    int64_t total_elements,
    cudaStream_t stream
) {
    int64_t D0, D1, D2;
    normalize_shape_3d(shape, D0, D1, D2);
    
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    
    shard_dim2_kernel<<<numBlocks, blockSize, 0, stream>>>(
        d_src, d_dst, D0, D1, D2, D2_local, rank, total_elements
    );
}
