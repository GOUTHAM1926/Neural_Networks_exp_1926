#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for matrix multiplication: a @ b
 * 
 * Forward: out = a @ b
 * Backward: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out
 */
class MatmulBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    MatmulBackward(const Tensor& a, const Tensor& b);
    
    const char* name() const override { return "MatmulBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    void release_saved_variables() override {
        saved_a_ = Tensor();  // Release reference
        saved_b_ = Tensor();
    }
};

/**
 * @brief Backward function for addmm: beta * input + alpha * (mat1 @ mat2)
 */
class AddmmBackward : public Node {
private:
    Tensor saved_mat1_;
    Tensor saved_mat2_;
    Shape saved_input_shape_;
    float alpha_;
    float beta_;
    
public:
    AddmmBackward(const Tensor& input, const Tensor& mat1, const Tensor& mat2, float alpha, float beta);
    
    const char* name() const override { return "AddmmBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    void release_saved_variables() override {
        saved_mat1_ = Tensor();
        saved_mat2_ = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor