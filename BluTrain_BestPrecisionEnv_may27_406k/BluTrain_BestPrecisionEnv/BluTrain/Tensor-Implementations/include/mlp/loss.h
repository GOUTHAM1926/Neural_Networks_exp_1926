#pragma once
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Exponents.h"   
#include "ops/UnaryOps/Arithmetics.h" 
#include <vector>   
#include "ops/UnaryOps/Reduction.h"
#include "ops/helpers/ConditionalOps.h"

#include "core/Tensor.h"

namespace OwnTensor
{
    namespace mlp_forward
    {

        Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

        Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

        Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

        Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets);

    }
}