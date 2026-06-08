#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Fused backward for tril_softmax + matmul in attention.
 *
 * Forward:
 *   attn_probs = fused_tril_softmax(attn_weights, diag, val)
 *   attn_out   = matmul(attn_probs, v)
 *
 * Backward (produces grad_attn_weights and grad_v):
 *   grad_attn_probs  = grad_out @ v.T
 *   grad_attn_weights = softmax_bwd(grad_attn_probs, attn_probs)
 *   grad_v            = attn_probs.T @ grad_out
 *
 * Saves attn_probs and v once instead of attn_probs being saved
 * by both FusedTrilSoftmaxBackward and MatmulBackward.
 *
 * Edge 0: attn_weights (input to fused_tril_softmax)
 * Edge 1: v            (second input to matmul)
 */
class FusedAttnSoftmaxMatmulBackward : public Node {
private:
    Tensor saved_attn_probs_;
    Tensor saved_v_;

public:
    FusedAttnSoftmaxMatmulBackward(const Tensor& attn_probs, const Tensor& v);

    const char* name() const override { return "FusedAttnSoftmaxMatmulBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        saved_attn_probs_ = Tensor();
        saved_v_ = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor
