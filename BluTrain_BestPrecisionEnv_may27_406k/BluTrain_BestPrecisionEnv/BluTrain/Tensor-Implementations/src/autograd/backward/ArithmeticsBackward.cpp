#include "autograd/backward/ArithmeticsBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/helpers/ConditionalOps.h" // For where, if needed
#include <stdexcept>
#include <cmath>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// SquareBackward
// ============================================================================

SquareBackward::SquareBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> SquareBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SquareBackward: no grads");
    
    // grad_x = grad_output * 2 * input
    return {grads[0] * 2.0 * saved_input_};
}

// ============================================================================
// SqrtBackward
// ============================================================================

SqrtBackward::SqrtBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> SqrtBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("SqrtBackward: no grads");
    
    // grad_x = grad_output / (2 * sqrt(input))
    Tensor sqrt_x = OwnTensor::sqrt(saved_input_);
    return {grads[0] / (2.0 * sqrt_x)};
}

// ============================================================================
// NegBackward
// ============================================================================

std::vector<Tensor> NegBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("NegBackward: no grads");
    
    return {grads[0] * -1.0};
}

// ============================================================================
// AbsBackward
// ============================================================================

AbsBackward::AbsBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> AbsBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("AbsBackward: no grads");
    
    // grad_x = grad_output * sign(input)
    Tensor s = OwnTensor::sign(saved_input_);
    return {grads[0] * s};
}

// ============================================================================
// ReciprocalBackward
// ============================================================================

ReciprocalBackward::ReciprocalBackward(const Tensor& input)
    : Node(1), saved_input_(input) {}

std::vector<Tensor> ReciprocalBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("ReciprocalBackward: no grads");
    
    // grad_x = -grad_output / input^2
    Tensor input_sq = saved_input_ * saved_input_;
    return {grads[0] * -1.0 / input_sq};
}

// ============================================================================
// PowBackward
// ============================================================================

PowBackward::PowBackward(const Tensor& input, float exponent)
    : Node(1), saved_input_(input), exponent_(exponent) {}

std::vector<Tensor> PowBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) throw std::runtime_error("PowBackward: no grads");
    
    // grad_x = grad_output * exponent * input^(exponent-1)
    Tensor derived = OwnTensor::pow(saved_input_, static_cast<double>(exponent_) - 1.0);
    return {grads[0] * static_cast<double>(exponent_) * derived};
}

} // namespace autograd
} // namespace OwnTensor
