// ============================================================================
// Create new file: include/core/ConversionKernels.cuh
// ============================================================================

#pragma once

// #ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include "dtype/Dtype.h"

namespace OwnTensor {


// Forward declarations
template<typename T>
void convert_to_bool_cuda(const T* input, bool* output, int64_t n, cudaStream_t stream = 0);

template<typename Src, typename Dst>
void convert_type_cuda(const Src* input, Dst* output, int64_t n, cudaStream_t stream = 0);

void convert_type_cuda_generic(const void* input, Dtype src_dtype, void* output, Dtype dst_dtype, int64_t n, cudaStream_t stream = 0);


// #endif // WITH_CUDA
} // namespace OwnTensor
