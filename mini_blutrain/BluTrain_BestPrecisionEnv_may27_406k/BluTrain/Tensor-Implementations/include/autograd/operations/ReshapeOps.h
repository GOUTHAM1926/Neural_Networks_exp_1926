#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware reshape operation
 */
Tensor reshape(const Tensor& input, Shape new_shape);

/**
 * @brief Autograd-aware view operation
 */
Tensor view(const Tensor& input, Shape new_shape);

/**
 * @brief Autograd-aware transpose operation
 */
Tensor transpose(const Tensor& input, int dim0, int dim1);


Tensor contiguous(const Tensor& input);

} // namespace autograd
} // namespace OwnTensor
