#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware addition
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware in-place addition: a += b
 * Safe because AddBackward only saves shapes, not input values.
 * Use for residual connections where the input is not needed after the add.
 */
Tensor& add_(Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware multiplication
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware subtraction
 */
Tensor sub(const Tensor& a, const Tensor& b);

/**
 * @brief Autograd-aware division
 */
Tensor div(const Tensor& a, const Tensor& b);

} // namespace autograd
} // namespace OwnTensor
