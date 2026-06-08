#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward for exp(x)
 * y = exp(x), dy/dx = y
 */
class ExpBackward : public Node {
private:
    Tensor saved_output_;
public:
    ExpBackward(const Tensor& output);
    const char* name() const override { return "ExpBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward for log(x) (ln(x))
 * y = ln(x), dy/dx = 1/x
 */
class LogBackward : public Node {
private:
    Tensor saved_input_;
public:
    LogBackward(const Tensor& input);
    const char* name() const override { return "LogBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward for exp2(x)
 * y = 2^x, dy/dx = 2^x * ln(2) = y * ln(2)
 */
class Exp2Backward : public Node {
private:
    Tensor saved_output_;
public:
    Exp2Backward(const Tensor& output);
    const char* name() const override { return "Exp2Backward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward for log2(x)
 * y = log2(x), dy/dx = 1 / (x * ln(2))
 */
class Log2Backward : public Node {
private:
    Tensor saved_input_;
public:
    Log2Backward(const Tensor& input);
    const char* name() const override { return "Log2Backward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

/**
 * @brief Backward for log10(x)
 * y = log10(x), dy/dx = 1 / (x * ln(10))
 */
class Log10Backward : public Node {
private:
    Tensor saved_input_;
public:
    Log10Backward(const Tensor& input);
    const char* name() const override { return "Log10Backward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
