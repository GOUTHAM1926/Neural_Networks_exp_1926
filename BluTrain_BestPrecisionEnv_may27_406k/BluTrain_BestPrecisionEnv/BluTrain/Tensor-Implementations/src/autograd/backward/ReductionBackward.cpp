#include "autograd/backward/ReductionBackward.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// SumBackward
// ============================================================================

SumBackward::SumBackward(const Shape& input_shape)
    : Node(1), input_shape_(input_shape) {}

std::vector<Tensor> SumBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("SumBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // Broadcast grad_output to input_shape
    Tensor grad_input = Tensor::ones(input_shape_, 
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()));
    
    // Scale by grad_output value
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        grad_input = grad_input * grad_output;
    }
    
    return {grad_input};
}

// ============================================================================
// MeanBackward
// ============================================================================

MeanBackward::MeanBackward(const Shape& input_shape, int64_t numel)
    : Node(1), input_shape_(input_shape), numel_(numel) {}

std::vector<Tensor> MeanBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("MeanBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = grad_output / numel
    Tensor grad_input = Tensor::ones(input_shape_,
        TensorOptions()
            .with_dtype(grad_output.dtype())
            .with_device(grad_output.device()));
    
    // Scale by grad_output / numel
    if (grad_output.ndim() == 0 || grad_output.numel() == 1) {
        grad_input = grad_input * (grad_output * (1.0 / static_cast<double>(numel_)));
    }
    
    return {grad_input};
}

} // namespace autograd
} // namespace OwnTensor