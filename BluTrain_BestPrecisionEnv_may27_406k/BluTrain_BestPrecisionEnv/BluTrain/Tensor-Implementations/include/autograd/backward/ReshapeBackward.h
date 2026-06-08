#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for reshape/view operations.
 */
class ReshapeBackward : public Node {
private:
    Shape input_shape_;
    
public:
    ReshapeBackward(const Shape& shape) : input_shape_(shape) {}
    
    const char* name() const override { return "ReshapeBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
