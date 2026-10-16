#include "autograd/backward/NormalizationBackward.h"
#include "ops/UnaryOps/Normalizations.h"
#include <stdexcept>
#include "profiler/RecordFunction.h"

namespace OwnTensor {
namespace autograd {

// ============================================================================
// LayerNormBackward — thin wrapper with grad_input_mask (like PyTorch)
// ============================================================================
std::vector<Tensor> LayerNormBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("LayerNormBackward::apply");
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
    PROFILER_SCOPE("RMSNormBackward::apply");
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

// ============================================================================
// FusedRMSRoPEBackward — autograd backward implementation
// ============================================================================
std::vector<Tensor> FusedRMSRoPEBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("FusedRMSRoPEBackward::apply");
    Tensor q = q_.unpack(shared_from_this());
    Tensor k = k_.unpack(shared_from_this());
    Tensor q_gamma = q_gamma_.unpack(shared_from_this());
    Tensor k_gamma = k_gamma_.unpack(shared_from_this());
    Tensor cos_sin_cache = cos_sin_cache_.unpack(shared_from_this());
    Tensor q_rstd = q_rstd_.unpack(shared_from_this());
    Tensor k_rstd = k_rstd_.unpack(shared_from_this());

    Tensor dz_q = (grads.size() > 0 && grads[0].is_valid()) ? grads[0] : Tensor::zeros(q.shape(), q.opts());
    Tensor dz_k = (grads.size() > 1 && grads[1].is_valid()) ? grads[1] : Tensor::zeros(k.shape(), k.opts());

    // v2: positions_ is the forward's per-token positions (invalid Tensor = original path).
    auto result = fused_rms_rope_backward(
        dz_q, dz_k, q, k, q_gamma, k_gamma, cos_sin_cache,
        q_rstd, k_rstd, pos_offset_, interleaved_, positions_);

    return {result.grad_q, result.grad_k, result.grad_q_gamma, result.grad_k_gamma};
}

} // namespace autograd
} // namespace OwnTensor

