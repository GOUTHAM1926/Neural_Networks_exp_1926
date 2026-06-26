#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware ReLU
 */
Tensor relu(const Tensor &x);

/**
 * @brief Autograd-aware GeLU
 */
Tensor gelu(const Tensor &x);

/**
 * @brief Autograd-aware sigmoid
 */
Tensor sigmoid(const Tensor &x);

/**
 * @brief Autograd-aware softmax
 */
Tensor softmax(const Tensor &x, int64_t dim = -1);

/**
 * @brief Autograd-aware swiglu
 */
Tensor swiglu(const Tensor &x);

/**
 * @brief Fused tril mask + softmax (single kernel launch).
 *
 * Equivalent to softmax(tril(x, diagonal, value)) but avoids the
 * intermediate tril allocation and runs everything in shared memory.
 */
Tensor fused_tril_softmax(const Tensor& x, int64_t diagonal = 0, double value = 0.0);

/**
 * @brief Autograd-aware dropout
 */
Tensor dropout(const Tensor& x, float p, bool training = true);

/**
 * @brief Fused bias + GELU: output = gelu(input + bias). CUDA float32 only.
 */
Tensor fused_bias_gelu(const Tensor& input, const Tensor& bias);

} // namespace autograd
} // namespace OwnTensor
