#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for fused tril mask + softmax.
 *
 * Forward:
 *   1. Tril mask: y_i = x_i  if col_i <= local_row + trilDiag, else -INF
 *   2. Softmax:   out = softmax(y)
 *
 * Backward:
 *   grad_input[i] = out[i] * (grad_output[i] - dot)
 *   where dot = sum_j(grad_output[j] * out[j])  (per row)
 *
 * The tril mask does not need to be re-applied in the backward pass because
 * masked positions have out[i] = 0, so grad_input[i] = 0 * (...) = 0
 * automatically through the softmax Jacobian.
 *
 * Saves the forward output (not the input) to avoid storing the larger
 * pre-mask tensor — identical to SoftmaxBackward.
 */
class FusedTrilSoftmaxBackward : public Node {
private:
    Tensor saved_output_;
    int64_t trilDiag_;
    double   value_;

public:
    FusedTrilSoftmaxBackward(const Tensor& output, int64_t trilDiag, double value);

    const char* name() const override { return "FusedTrilSoftmaxBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_output_ = Tensor(); }
};

} // namespace autograd
} // namespace OwnTensor
