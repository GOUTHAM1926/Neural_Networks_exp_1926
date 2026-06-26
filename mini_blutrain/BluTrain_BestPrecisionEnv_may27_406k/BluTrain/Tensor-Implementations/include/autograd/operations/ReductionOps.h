#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware sum
 */
Tensor sum(const Tensor& x);

/**
 * @brief Autograd-aware mean
 */
Tensor mean(const Tensor& x);

} // namespace autograd
} // namespace OwnTensor
