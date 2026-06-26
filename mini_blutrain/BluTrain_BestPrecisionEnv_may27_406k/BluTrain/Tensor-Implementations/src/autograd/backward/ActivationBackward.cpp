#include "autograd/backward/ActivationBackward.h"
#include "ops/UnaryOps/Activations.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// ReluBackward
// ============================================================================
ReluBackward::ReluBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> ReluBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("ReluBackward: no gradients provided");
    return {relu_backward(grads[0], saved_input_)};
}

// ============================================================================
// GeLUBackward
// ============================================================================
GeLUBackward::GeLUBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> GeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("GeLUBackward: no gradients provided");
    return {gelu_backward(grads[0], saved_input_)};
}

// ============================================================================
// SigmoidBackward
// ============================================================================
SigmoidBackward::SigmoidBackward(const Tensor& output)
    : Node(1), saved_output_(output) {}

std::vector<Tensor> SigmoidBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SigmoidBackward: no gradients provided");
    return {sigmoid_backward(grads[0], saved_output_)};
}

// ============================================================================
// SoftmaxBackward
// ============================================================================
SoftmaxBackward::SoftmaxBackward(const Tensor& output, int64_t dim)
    : Node(1), saved_output_(output), dim_(dim) {}

std::vector<Tensor> SoftmaxBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SoftmaxBackward: no gradients provided");
    return {softmax_backward(grads[0], saved_output_, dim_)};
}

// ============================================================================
// DropoutBackward
// ============================================================================
DropoutBackward::DropoutBackward(const Tensor& mask, float scale)
    : Node(1), saved_mask_(mask), scale_(scale) {}

std::vector<Tensor> DropoutBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("DropoutBackward: no gradients provided");
    return {dropout_backward(grads[0], saved_mask_, scale_)};
}

// ============================================================================
// SwiGLUBackward
// ============================================================================
SwiGLUBackward::SwiGLUBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> SwiGLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SwiGLU Backward: no gradient provided");
    return {swiglu_backward(grads[0], saved_input_)};
}

// ============================================================================
// FusedBiasGeLUBackward
// ============================================================================
FusedBiasGeLUBackward::FusedBiasGeLUBackward(const Tensor& input, const Tensor& bias)
    : Node(2), saved_input_(input), saved_bias_(bias) {}

std::vector<Tensor> FusedBiasGeLUBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("FusedBiasGeLUBackward: no gradients provided");
    auto result = fused_bias_gelu_backward(grads[0], saved_input_, saved_bias_);
    return {result.grad_input, result.grad_bias};
}

} // namespace autograd
} // namespace OwnTensor
