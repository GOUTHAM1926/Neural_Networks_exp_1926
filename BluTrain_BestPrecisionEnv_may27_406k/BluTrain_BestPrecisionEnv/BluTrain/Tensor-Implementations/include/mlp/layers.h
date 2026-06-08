#pragma once

#include "core/Tensor.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include <vector>
#include "ops/helpers/ConditionalOps.h"
#include <numeric>

namespace OwnTensor
{
    namespace mlp_forward
    {
        Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias);
        Tensor flatten(const Tensor& input);
        Tensor dropout(const Tensor& input, float p);
    }
}