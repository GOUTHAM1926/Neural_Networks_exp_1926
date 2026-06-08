#include "autograd/backward/FlashAttentionBackward.h"
#include <stdexcept>
#include <utility>
#include <vector>

namespace OwnTensor {

// Forward declaration — implemented in src/Kernels/cuda/FlashAttention.cu
std::vector<Tensor> dispatch_flash_attention_bwd(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& O, const Tensor& dO, const Tensor& L,
    int B, int n_heads, int T, int head_dim, float scale);

namespace autograd {

FlashAttentionBackward::FlashAttentionBackward(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const Tensor& O, const Tensor& L,
    int B, int n_heads, int T, int head_dim, float scale)
    : Node(3)              // 3 inputs: Q, K, V
    , saved_Q_(Q), saved_K_(K), saved_V_(V)
    , saved_O_(O), saved_L_(L)
    , B_(B), n_heads_(n_heads), T_(T), head_dim_(head_dim)
    , scale_(scale)
{}

std::vector<Tensor> FlashAttentionBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty())
        throw std::runtime_error("FlashAttentionBackward: no gradients provided");

    const Tensor& dO = grads[0];   // [BH, T, d]  or [B, n_heads, T, head_dim]

    // dispatch_flash_attention_bwd returns {dQ, dK, dV}
    std::vector<Tensor> result = dispatch_flash_attention_bwd(
        saved_Q_, saved_K_, saved_V_, saved_O_, dO, saved_L_,
        B_, n_heads_, T_, head_dim_, scale_);

    // Reshape outputs to match the 3-D [BH, T, d] shape expected by the graph
    // (dispatch returns [B, n_heads, T, head_dim] == [BH, 1, T, d])
    Tensor dQ = result[0].reshape(saved_Q_.shape());
    Tensor dK = result[1].reshape(saved_K_.shape());
    Tensor dV = result[2].reshape(saved_V_.shape());

    return {dQ, dK, dV};
}

} // namespace autograd
} // namespace OwnTensor
