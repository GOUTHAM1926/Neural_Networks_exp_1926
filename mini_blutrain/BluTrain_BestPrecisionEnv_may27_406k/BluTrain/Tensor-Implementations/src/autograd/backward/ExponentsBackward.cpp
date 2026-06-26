#include "autograd/backward/ExponentsBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include <stdexcept>
#include <cmath>

namespace OwnTensor {
namespace autograd {

// ===================================
// ExpBackward
// ===================================
ExpBackward::ExpBackward(const Tensor& output) : Node(1), saved_output_(output) {}

std::vector<Tensor> ExpBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("ExpBackward: no grads");
    // grad_x = grad_output * exp(x) = grad_output * y
    return {grads[0] * saved_output_};
}

// ===================================
// LogBackward
// ===================================
LogBackward::LogBackward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> LogBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("LogBackward: no grads");
    // grad_x = grad_output / x
    return {grads[0] / saved_input_};
}

// ===================================
// Exp2Backward
// ===================================
Exp2Backward::Exp2Backward(const Tensor& output) : Node(1), saved_output_(output) {}

std::vector<Tensor> Exp2Backward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("Exp2Backward: no grads");
    // grad_x = grad_output * y * ln(2)
    return {grads[0] * saved_output_ * M_LN2};
}

// ===================================
// Log2Backward
// ===================================
Log2Backward::Log2Backward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> Log2Backward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("Log2Backward: no grads");
    // grad_x = grad_output / (x * ln(2))
    return {grads[0] / (saved_input_ * M_LN2)};
}

// ===================================
// Log10Backward
// ===================================
Log10Backward::Log10Backward(const Tensor& input) : Node(1), saved_input_(input) {}

std::vector<Tensor> Log10Backward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("Log10Backward: no grads");
    // grad_x = grad_output / (x * ln(10))
    return {grads[0] / (saved_input_ * M_LN10)};
}

} // namespace autograd
} // namespace OwnTensor
