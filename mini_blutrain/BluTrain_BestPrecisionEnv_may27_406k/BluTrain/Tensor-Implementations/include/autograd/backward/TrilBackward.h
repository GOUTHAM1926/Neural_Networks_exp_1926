#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for tril(input, diagonal)
 * 
 * Forward: result = tril(input, diagonal)
 * Backward: grad_input = tril(grad_output, diagonal)
 */
class TrilBackward : public Node {
private:
    int64_t diagonal_;
    double value_;
    Tensor saved_input_;  // Original pre-tril input (for fusion detection)

public:
    TrilBackward(int64_t diagonal, double value = 0.0)
        : diagonal_(diagonal), value_(value) {}

    TrilBackward(const Tensor& input, int64_t diagonal, double value = 0.0)
        : diagonal_(diagonal), value_(value), saved_input_(input) {}

    const char* name() const override { return "TrilBackward"; }

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override { saved_input_ = Tensor(); }

    // Accessors for fusion detection
    int64_t diagonal() const { return diagonal_; }
    double value() const { return value_; }
    const Tensor& saved_input() const { return saved_input_; }
    bool has_saved_input() const { return saved_input_.is_valid(); }
};

} // namespace autograd
} // namespace OwnTensor
