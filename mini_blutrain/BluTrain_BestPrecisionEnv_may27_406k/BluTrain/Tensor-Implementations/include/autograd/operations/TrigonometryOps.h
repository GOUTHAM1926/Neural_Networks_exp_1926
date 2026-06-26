#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

Tensor sin(const Tensor& input);
Tensor cos(const Tensor& input);
Tensor tan(const Tensor& input);
Tensor asin(const Tensor& input);
Tensor acos(const Tensor& input);
Tensor atan(const Tensor& input);
Tensor sinh(const Tensor& input);
Tensor cosh(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor asinh(const Tensor& input);
Tensor acosh(const Tensor& input);
Tensor atanh(const Tensor& input);

} // namespace autograd
} // namespace OwnTensor
