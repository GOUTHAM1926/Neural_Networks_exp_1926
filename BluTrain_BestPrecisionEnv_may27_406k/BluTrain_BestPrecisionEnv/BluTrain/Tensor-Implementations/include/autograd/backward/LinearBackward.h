#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for linear: out = x @ W + b
 * 
 * This fuses matmul and bias add for efficiency.
 * Backward: grad_x = grad_out @ W.T
 *           grad_W = x.T @ grad_out (summed over batch)
 *           grad_b = sum(grad_out, dim=0...-1)
 */
class LinearBackward : public Node {
private:
    Tensor saved_input_;
    Tensor saved_weight_;
    
    // Original shapes for reshaping if needed (though usually not for linear)
    
public:
    LinearBackward(const Tensor& input, const Tensor& weight);
    
    const char* name() const override { return "LinearBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    void release_saved_variables() override {
        saved_input_ = Tensor();
        saved_weight_ = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor
