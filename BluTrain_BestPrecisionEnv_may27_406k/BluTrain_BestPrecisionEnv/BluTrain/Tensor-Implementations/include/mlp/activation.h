#pragma once

#include "core/Tensor.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include <stdexcept>
#include <vector>

namespace OwnTensor
{
    
        Tensor softmax(const Tensor& input, int64_t dim = -1);

        // Tensor tanh(const Tensor& input);

        Tensor sigmoid(const Tensor& input);

        Tensor ReLU(const Tensor& input);

        Tensor GeLU(const Tensor& input);
    
}
