#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {


Tensor tril(const Tensor& input, int64_t diagonal = 0, double value = 0.0);

} // namespace autograd
} // namespace OwnTensor
