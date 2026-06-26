#pragma once
// =============================================================================
// fill_cuda_launch<T>
//
// GPU-side fill that replaces the CPU-vector + H->D memcpy pattern previously
// used by Tensor::fill / Tensor::fill_grad / Tensor::ones / Tensor::full.
//
// Semantics: writes `value` to every element of `ptr[0..numel)` on `stream`.
// Fast path: when `value` has all-zero bit representation, calls
// cudaMemsetAsync (DMA engine, no SM occupancy).
// Main path: vectorized 16-byte stores (STG.128 on sm_89) via a uint4-packed
// kernel. Tail elements use a scalar kernel.
//
// Modelled after PyTorch's FillKernel.cu + launch_vectorized_kernel in
// aten/src/ATen/native/cuda/CUDALoops.cuh.
// =============================================================================

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor { namespace cuda {

template <typename T>
void fill_cuda_launch(T* ptr, T value, int64_t numel, cudaStream_t stream);

} } // namespace OwnTensor::cuda
