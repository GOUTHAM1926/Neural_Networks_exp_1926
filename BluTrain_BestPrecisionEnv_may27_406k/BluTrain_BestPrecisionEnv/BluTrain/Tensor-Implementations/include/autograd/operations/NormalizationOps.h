#pragma once
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief LayerNorm with autograd support.
 *        Delegates to layer_norm_forward/backward in Normalizations.h.
 */
Tensor layer_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int normalized_shape,
    float eps = 1e-5f);

/**
 * @brief RMSNorm with autograd support.
 *        Delegates to rms_norm_forward/backward in Normalizations.h.
 */
Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    int normalized_shape,
    float eps = 1e-5f);

} // namespace autograd
} // namespace OwnTensor
