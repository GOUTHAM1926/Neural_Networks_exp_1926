#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for square(x)
 * Forward: y = x^2
 * Backward: grad_x = grad_y * 2x
 */
class SquareBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    SquareBackward(const Tensor& input);
    
    const char* name() const override { return "SquareBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for sqrt(x)
 * Forward: y = sqrt(x)
 * Backward: grad_x = grad_y / (2 * sqrt(x))
 */
class SqrtBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    SqrtBackward(const Tensor& input);
    
    const char* name() const override { return "SqrtBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};
// ... (NegBackward) ...
class NegBackward : public Node {
public:
    NegBackward() : Node(1) {}
    
    const char* name() const override { return "NegBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for abs(x)
// ...
 */
class AbsBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    AbsBackward(const Tensor& input);
    
    const char* name() const override { return "AbsBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for reciprocal(x)
 * Forward: y = 1/x
 * Backward: grad_x = -grad_y / x^2
 */
class ReciprocalBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    ReciprocalBackward(const Tensor& input);
    
    const char* name() const override { return "ReciprocalBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for pow(x, exponent)
 * Forward: y = x^exponent
 * Backward: grad_x = grad_y * exponent * x^(exponent-1)
 */
class PowBackward : public Node {
private:
    Tensor saved_input_;
    float exponent_;
    
public:
    PowBackward(const Tensor& input, float exponent);
    
    const char* name() const override { return "PowBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
