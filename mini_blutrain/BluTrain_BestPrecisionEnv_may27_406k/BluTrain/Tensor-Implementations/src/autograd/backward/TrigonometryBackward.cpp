#include "autograd/backward/TrigonometryBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Trigonometry.h"
#include "ops/UnaryOps/Arithmetics.h" // For sqrt, pow, etc.
#include "ops/helpers/ConditionalOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// SinBackward
// ============================================================================
SinBackward::SinBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> SinBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SinBackward: no grads");
    // grad_x = grad_output * cos(x)
    return {grads[0] * OwnTensor::cos(saved_input_)};
}

// ============================================================================
// CosBackward
// ============================================================================
CosBackward::CosBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> CosBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("CosBackward: no grads");
    // grad_x = -grad_output * sin(x)
    return {grads[0] * -1.0f * OwnTensor::sin(saved_input_)};
}

// ============================================================================
// TanBackward
// ============================================================================
TanBackward::TanBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> TanBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("TanBackward: no grads");
    // grad_x = grad_output / cos^2(x)
    Tensor cos_x = OwnTensor::cos(saved_input_);
    Tensor cos_sq = cos_x * cos_x;
    return {grads[0] / cos_sq};
}

// ...

// ============================================================================
// TanhBackward
// ============================================================================
TanhBackward::TanhBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> TanhBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("TanhBackward: no grads");
    // grad_x = grad_output * (1 - tanh^2(x))
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor tanh_x = OwnTensor::tanh(saved_input_);
    Tensor tanh_sq = tanh_x * tanh_x;
    return {grads[0] * (one - tanh_sq)};
}

// ============================================================================
// AsinBackward
// ============================================================================
AsinBackward::AsinBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AsinBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AsinBackward: no grads");
    // grad_x = grad_output / sqrt(1 - x^2)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    Tensor denom = OwnTensor::sqrt(one - x_sq); // Assuming 1-x^2 works? 
    // Wait, 1 is tensor? Yes created ones.
    return {grads[0] / denom};
}

// ============================================================================
// AcosBackward
// ============================================================================
AcosBackward::AcosBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AcosBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AcosBackward: no grads");
    // grad_x = -grad_output / sqrt(1 - x^2)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    Tensor denom = OwnTensor::sqrt(one - x_sq);
    return {grads[0] * -1.0f / denom};
}

// ============================================================================
// AtanBackward
// ============================================================================
AtanBackward::AtanBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AtanBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AtanBackward: no grads");
    // grad_x = grad_output / (1 + x^2)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    return {grads[0] / (one + x_sq)};
}

// ============================================================================
// SinhBackward
// ============================================================================
SinhBackward::SinhBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> SinhBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SinhBackward: no grads");
    // grad_x = grad_output * cosh(x)
    return {grads[0] * OwnTensor::cosh(saved_input_)};
}

// ============================================================================
// CoshBackward
// ============================================================================
CoshBackward::CoshBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> CoshBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("CoshBackward: no grads");
    // grad_x = grad_output * sinh(x)
    return {grads[0] * OwnTensor::sinh(saved_input_)};
}

// (empty)

// ============================================================================
// AsinhBackward
// ============================================================================
AsinhBackward::AsinhBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AsinhBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AsinhBackward: no grads");
    // grad_x = grad_output / sqrt(x^2 + 1)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    Tensor denom = OwnTensor::sqrt(x_sq + one);
    return {grads[0] / denom};
}

// ============================================================================
// AcoshBackward
// ============================================================================
AcoshBackward::AcoshBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AcoshBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AcoshBackward: no grads");
    // grad_x = grad_output / sqrt(x^2 - 1)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    Tensor denom = OwnTensor::sqrt(x_sq - one);
    return {grads[0] / denom};
}

// ============================================================================
// AtanhBackward
// ============================================================================
AtanhBackward::AtanhBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> AtanhBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AtanhBackward: no grads");
    // grad_x = grad_output / (1 - x^2)
    Tensor one = Tensor::ones(saved_input_.shape(), 
        TensorOptions().with_device(saved_input_.device()).with_dtype(saved_input_.dtype()));
    Tensor x_sq = saved_input_ * saved_input_;
    return {grads[0] / (one - x_sq)};
}

} // namespace autograd
} // namespace OwnTensor
