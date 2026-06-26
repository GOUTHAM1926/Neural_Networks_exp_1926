#include "autograd/backward/TrilBackward.h"
#include "ops/TensorOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

std::vector<Tensor> TrilBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("TrilBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];
    
    // grad_input = tril(grad_output, diagonal, 0.0)
    Tensor grad_input = OwnTensor::tril(grad_output, diagonal_, 0.0);
    
    return {grad_input};
}

} // namespace autograd
} // namespace OwnTensor
