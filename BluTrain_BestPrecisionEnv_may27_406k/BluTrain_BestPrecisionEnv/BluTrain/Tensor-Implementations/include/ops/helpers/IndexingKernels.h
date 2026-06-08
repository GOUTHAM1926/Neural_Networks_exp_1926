#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace OwnTensor {
namespace cuda {

template <typename T, typename T_idx>
__global__ void gather_kernel(
    const T* __restrict__ input,
    const T_idx* __restrict__ indices,
    T* __restrict__ output,
    int64_t dim,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ idx_strides,
    const int64_t* __restrict__ in_dims,
    const int64_t* __restrict__ idx_dims,
    int ndim,
    int64_t numel
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    T_idx gathered_idx = indices[i];
    
    if (gathered_idx < 0 || (int64_t)gathered_idx >= in_dims[dim]) {
        return; 
    }

    int64_t in_offset = 0;
    int64_t temp_i = i;
    for (int j = ndim - 1; j >= 0; --j) {
        int64_t coord = temp_i % idx_dims[j];
        temp_i /= idx_dims[j];
        
        if (j == dim) {
            in_offset += (int64_t)gathered_idx * in_strides[j];
        } else {
            in_offset += coord * in_strides[j];
        }
    }

    output[i] = input[in_offset];
}

template <typename T, typename T_idx>
void gather_cuda(
    const T* input,
    const T_idx* indices,
    T* output,
    int64_t dim,
    const int64_t* in_strides,
    const int64_t* idx_strides,
    const int64_t* out_strides,
    const int64_t* in_dims,
    const int64_t* idx_dims,
    int ndim,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    gather_kernel<T, T_idx><<<blocks, threads, 0, stream>>>(
        input, indices, output, dim, in_strides, idx_strides, in_dims, idx_dims, ndim, numel
    );
}

} // namespace cuda
} // namespace OwnTensor
