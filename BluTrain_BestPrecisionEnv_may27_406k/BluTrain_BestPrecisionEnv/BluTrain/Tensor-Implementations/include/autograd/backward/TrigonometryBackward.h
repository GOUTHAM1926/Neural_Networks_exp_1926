#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for sin(x)
 * Forward: y = sin(x)
 * Backward: grad_x = grad_y * cos(x)
 */
class SinBackward : public Node {
private:
    Tensor saved_input_;
public:
    SinBackward(const Tensor& input);
    const char* name() const override { return "SinBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for cos(x)
 * Forward: y = cos(x)
 * Backward: grad_x = -grad_y * sin(x)
 */
class CosBackward : public Node {
private:
    Tensor saved_input_;
public:
    CosBackward(const Tensor& input);
    const char* name() const override { return "CosBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for tan(x)
 * Forward: y = tan(x)
 * Backward: grad_x = grad_y / cos^2(x)
 */
class TanBackward : public Node {
private:
    Tensor saved_input_;
public:
    TanBackward(const Tensor& input);
    const char* name() const override { return "TanBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for asin(x)
 * Forward: y = asin(x)
 * Backward: grad_x = grad_y / sqrt(1 - x^2)
 */
class AsinBackward : public Node {
private:
    Tensor saved_input_;
public:
    AsinBackward(const Tensor& input);
    const char* name() const override { return "AsinBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for acos(x)
 * Forward: y = acos(x)
 * Backward: grad_x = -grad_y / sqrt(1 - x^2)
 */
class AcosBackward : public Node {
private:
    Tensor saved_input_;
public:
    AcosBackward(const Tensor& input);
    const char* name() const override { return "AcosBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for atan(x)
 * Forward: y = atan(x)
 * Backward: grad_x = grad_y / (1 + x^2)
 */
class AtanBackward : public Node {
private:
    Tensor saved_input_;
public:
    AtanBackward(const Tensor& input);
    const char* name() const override { return "AtanBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for sinh(x)
 * Forward: y = sinh(x)
 * Backward: grad_x = grad_y * cosh(x)
 */
class SinhBackward : public Node {
private:
    Tensor saved_input_;
public:
    SinhBackward(const Tensor& input);
    const char* name() const override { return "SinhBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for cosh(x)
 * Forward: y = cosh(x)
 * Backward: grad_x = grad_y * sinh(x)
 */
class CoshBackward : public Node {
private:
    Tensor saved_input_;
public:
    CoshBackward(const Tensor& input);
    const char* name() const override { return "CoshBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for tanh(x)
 * Forward: y = tanh(x)
 * Backward: grad_x = grad_y * (1 - tanh^2(x))
 */
class TanhBackward : public Node {
private:
    Tensor saved_input_;
public:
    TanhBackward(const Tensor& input);
    const char* name() const override { return "TanhBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

// Ignoring inverse hyperbolic for now to verify basics quickly, but can add if requested.
// User said "all that available there". So I should probably add them.

/**
 * @brief Backward function for asinh(x)
 * Forward: y = asinh(x)
 * Backward: grad_x = grad_y / sqrt(x^2 + 1)
 */
class AsinhBackward : public Node {
private:
    Tensor saved_input_;
public:
    AsinhBackward(const Tensor& input);
    const char* name() const override { return "AsinhBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for acosh(x)
 * Forward: y = acosh(x)
 * Backward: grad_x = grad_y / sqrt(x^2 - 1)
 */
class AcoshBackward : public Node {
private:
    Tensor saved_input_;
public:
    AcoshBackward(const Tensor& input);
    const char* name() const override { return "AcoshBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward function for atanh(x)
 * Forward: y = atanh(x)
 * Backward: grad_x = grad_y / (1 - x^2)
 */
class AtanhBackward : public Node {
private:
    Tensor saved_input_;
public:
    AtanhBackward(const Tensor& input);
    const char* name() const override { return "AtanhBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
