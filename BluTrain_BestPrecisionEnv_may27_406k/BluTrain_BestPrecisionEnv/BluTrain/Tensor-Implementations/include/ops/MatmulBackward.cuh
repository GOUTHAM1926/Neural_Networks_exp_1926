#pragma once

#ifdef WITH_CUDA

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {

/**
 * @brief Optimized CUDA backward matmul kernel
 * 
 * Computes gradients for C = A @ B:
 *   grad_A = grad_output @ B^T
 *   grad_B = A^T @ grad_output
 * 
 * Uses implicit transpose (no explicit transpose operation) with:
 * - Tiled matrix multiplication with register blocking
 * - Double buffering for memory latency hiding
 * - WMMA for FP16/BF16 (Tensor Core acceleration)
 * 
 * @param grad_output Gradient of the output tensor
 * @param A First input tensor from forward pass
 * @param B Second input tensor from forward pass
 * @param grad_A Output: gradient for A (must be pre-allocated)
 * @param grad_B Output: gradient for B (must be pre-allocated)
 * @param stream CUDA stream for async execution
 */
void cuda_matmul_backward(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream = 0);

} // namespace OwnTensor

#endif // WITH_CUDA
