#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for addition: a + b
 * 
 * Forward: out = a + b
 * Backward: grad_a = grad_out, grad_b = grad_out
 */
class AddBackward : public Node {
private:
    Shape shape_a_;
    Shape shape_b_;
public:
    AddBackward(const Shape& a_shape, const Shape& b_shape);
    
    const char* name() const override { return "AddBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {}
};

/**
 * @brief Backward function for multiplication: a * b
 * 
 * Forward: out = a * b
 * Backward: grad_a = grad_out * b, grad_b = grad_out * a
 */
class MulBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    MulBackward(const Tensor& a, const Tensor& b);
    
    const char* name() const override { return "MulBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_a_ = Tensor(); saved_b_ = Tensor(); }
};

/**
 * @brief Backward function for subtraction: a - b
 * 
 * Forward: out = a - b
 * Backward: grad_a = grad_out, grad_b = -grad_out
 */
class SubBackward : public Node {
private:
    Shape shape_a_;
    Shape shape_b_;
public:
    SubBackward(const Shape& a_shape, const Shape& b_shape);
    
    const char* name() const override { return "SubBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {}
};

/**
 * @brief Backward function for division: a / b
 * 
 * Forward: out = a / b
 * Backward: grad_a = grad_out / b, grad_b = -grad_out * a / b^2
 */
class DivBackward : public Node {
private:
    Tensor saved_a_;
    Tensor saved_b_;
    
public:
    DivBackward(const Tensor& a, const Tensor& b);
    
    const char* name() const override { return "DivBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_a_ = Tensor(); saved_b_ = Tensor(); }
};

} // namespace autograd
} // namespace OwnTensor
