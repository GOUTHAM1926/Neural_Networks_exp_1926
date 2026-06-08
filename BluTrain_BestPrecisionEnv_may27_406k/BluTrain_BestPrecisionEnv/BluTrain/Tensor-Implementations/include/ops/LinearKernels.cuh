#pragma once

#ifdef WITH_CUDA

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {

// Fused Linear Forward: Output = Input @ Weight.T + Bias
// Uses cuBLAS for Matmul and a custom kernel for Bias addition (or Epilogue if possible)
void cuda_linear_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    cudaStream_t stream = 0);

// Optimized Bias Backward: grad_bias = sum(grad_output, dim=0..N-2)
void cuda_linear_bias_backward(
    const Tensor& grad_output,
    Tensor& grad_bias,
    cudaStream_t stream = 0);

} // namespace OwnTensor

#endif // WITH_CUDA
