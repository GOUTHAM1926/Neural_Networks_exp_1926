#include "autograd/backward/NormalizationBackward.h"
#include "ops/UnaryOps/Normalizations.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// LayerNormBackward — thin wrapper with grad_input_mask (like PyTorch)
// ============================================================================
std::vector<Tensor> LayerNormBackward::apply(std::vector<Tensor>&& grads) {
    Tensor grad_output = grads[0];
    Tensor input  = input_.unpack(shared_from_this());
    Tensor mean   = mean_.unpack(shared_from_this());
    Tensor rstd   = rstd_.unpack(shared_from_this());
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());

    // Check which gradients are actually needed via edge validity
    bool need_input  = (num_inputs() > 0 && next_edge(0).is_valid());
    bool need_weight = (num_inputs() > 1 && next_edge(1).is_valid());
    bool need_bias   = (num_inputs() > 2 && next_edge(2).is_valid());

    auto result = layer_norm_backward(
        grad_output, input, mean, rstd, weight, normalized_shape_, eps_,
        need_input, need_weight, need_bias);

    return {result.grad_input, result.grad_weight, result.grad_bias};
}

// ============================================================================
// RMSNormBackward — thin wrapper with grad_input_mask
// ============================================================================
std::vector<Tensor> RMSNormBackward::apply(std::vector<Tensor>&& grads) {
    Tensor grad_output = grads[0];
    Tensor input = input_.unpack(shared_from_this());
    Tensor rstd  = rstd_.unpack(shared_from_this());
    Tensor weight;
    if (weight_.defined()) weight = weight_.unpack(shared_from_this());

    bool need_input  = (num_inputs() > 0 && next_edge(0).is_valid());
    bool need_weight = (num_inputs() > 1 && next_edge(1).is_valid());

    auto result = rms_norm_backward(
        grad_output, input, rstd, weight, normalized_shape_, eps_,
        need_input, need_weight);

    return {result.grad_input, result.grad_weight};
}

} // namespace autograd
} // namespace OwnTensor
