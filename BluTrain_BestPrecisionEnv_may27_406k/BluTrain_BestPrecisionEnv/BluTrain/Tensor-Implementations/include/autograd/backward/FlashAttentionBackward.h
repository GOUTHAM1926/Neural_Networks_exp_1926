#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward node for Flash Attention (Algorithm 4, Dao et al.)
 *
 * Forward : O = softmax(scale · Q @ K^T, causal) @ V
 * Saved   : Q, K, V, O, L (log-sum-exp per row)
 *
 * Backward uses tiling + recomputation:
 *   - Recomputes P = exp(scale·Q·K^T − L) per tile  (no O(T²) storage)
 *   - D_i = Σ_k dO[i,k]·O[i,k]                     (precomputed)
 *   - dS   = P ⊙ (dO @ V^T − D)                     (softmax backward)
 *   - dQ   = scale · dS @ K
 *   - dK   = scale · dS^T @ Q
 *   - dV   = P^T @ dO
 */
class FlashAttentionBackward : public Node {
private:
    Tensor saved_Q_, saved_K_, saved_V_, saved_O_, saved_L_;
    int B_, n_heads_, T_, head_dim_;
    float scale_;

public:
    FlashAttentionBackward(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& O, const Tensor& L,
                           int B, int n_heads, int T, int head_dim, float scale);

    const char* name() const override { return "FlashAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

    void release_saved_variables() override {
        saved_Q_ = Tensor();
        saved_K_ = Tensor();
        saved_V_ = Tensor();
        saved_O_ = Tensor();
        saved_L_ = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor
