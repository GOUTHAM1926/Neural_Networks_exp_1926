#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

Tensor square(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor neg(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor reciprocal(const Tensor& input);
Tensor pow(const Tensor& input, float exponent);

} // namespace autograd
} // namespace OwnTensor
