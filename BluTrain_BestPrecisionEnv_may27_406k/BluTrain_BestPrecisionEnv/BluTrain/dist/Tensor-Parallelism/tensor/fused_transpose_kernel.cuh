#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Launch kernel for dim 0 sharding: [D0, D1] -> [D0_local, D1]
// Also works for 3D: [D0, D1, D2] -> [D0_local, D1, D2]
void launch_shard_dim0_kernel(
    float* d_src,           // Source data (full tensor)
    float* d_dst,           // Destination (sharded tensor)
    const std::vector<int64_t>& shape,  // Full shape (2D or 3D)
    int64_t D0_local,       // D0 / world_size
    int rank,
    int64_t total_elements,
    cudaStream_t stream
);

// Launch kernel for dim 1 sharding: [D0, D1] -> [D0, D1_local]
// Also works for 3D: [D0, D1, D2] -> [D0, D1_local, D2]
void launch_shard_dim1_kernel(
    float* d_src,           // Source data (full tensor)
    float* d_dst,           // Destination (sharded tensor)
    const std::vector<int64_t>& shape,  // Full shape (2D or 3D)
    int64_t D1_local,       // D1 / world_size
    int rank,
    int64_t total_elements,
    cudaStream_t stream
);

// Launch kernel for dim 2 sharding (3D only): [D0, D1, D2] -> [D0, D1, D2_local]
void launch_shard_dim2_kernel(
    float* d_src,           // Source data (full tensor)
    float* d_dst,           // Destination (sharded tensor)
    const std::vector<int64_t>& shape,  // Full shape (3D)
    int64_t D2_local,       // D2 / world_size
    int rank,
    int64_t total_elements,
    cudaStream_t stream
);
