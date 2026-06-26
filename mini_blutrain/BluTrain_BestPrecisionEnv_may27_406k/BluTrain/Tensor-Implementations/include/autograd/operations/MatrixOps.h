#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware linear transformation: x @ W + b
 * Fuse matmul and bias add for better performance.
 */
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

/**
 * @brief Forward-only addmm: beta * input + alpha * (mat1 @ mat2)
 */
Tensor addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha = 1.0f, float beta = 1.0f);



} // namespace autograd
} // namespace OwnTensor