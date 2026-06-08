#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for sum reduction
 * 
 * Forward: out = sum(x)
 * Backward: grad_x = grad_out (broadcasted to x.shape)
 */
class SumBackward : public Node {
private:
    Shape input_shape_;
    
public:
    explicit SumBackward(const Shape& input_shape);
    
    const char* name() const override { return "SumBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for mean reduction
 * 
 * Forward: out = mean(x)
 * Backward: grad_x = grad_out / numel(x)
 */
class MeanBackward : public Node {
private:
    Shape input_shape_;
    int64_t numel_;
    
public:
    MeanBackward(const Shape& input_shape, int64_t numel);
    
    const char* name() const override { return "MeanBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
