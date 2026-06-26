#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor exp2(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor log10(const Tensor& input);

} // namespace autograd
} // namespace OwnTensor
