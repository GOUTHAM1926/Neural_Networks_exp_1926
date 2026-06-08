
// #pragma once

// #include "nn/NN.h"
// #include "autograd/operations/AttentionOps.h"

// namespace OwnTensor {
// namespace nn {

// /**
//  * @brief Multi-Head Self-Attention module.
//  *
//  * Implements the standard transformer attention block:
//  *   1. Project input to Q, K, V via a single linear layer
//  *   2. Split into num_heads parallel heads
//  *   3. Compute scaled_dot_product_attention (backend-selectable)
//  *   4. Concatenate heads and project output
//  *
//  * LayerNorm and residual connections are NOT included —
//  * they are the caller's responsibility (matching PyTorch convention).
//  *
//  * Usage:
//  *   MultiHeadAttention mha(384, 6, true, autograd::SDPBackend::Math);
//  *   mha.to(device);
//  *   Tensor out = mha.forward(x);  // x: [B, T, C]  ->  out: [B, T, C]
//  */
// class MultiHeadAttention : public Module {
// public:
//     int embed_dim;
//     int num_heads;
//     int head_dim;
//     bool is_causal;
//     autograd::SDPBackend backend;

//     Linear qkv_proj;   // (embed_dim) -> (3 * embed_dim)
//     Linear out_proj;    // (embed_dim) -> (embed_dim)

//     /**
//      * @param embed_dim  Total embedding dimension (must be divisible by num_heads)
//      * @param num_heads  Number of attention heads
//      * @param is_causal  Whether to apply causal (autoregressive) mask
//      * @param backend    Attention computation backend
//      * @param bias       Whether linear layers have bias
//      */
//     MultiHeadAttention(int embed_dim, int num_heads,
//                        bool is_causal = true,
//                        autograd::SDPBackend backend = autograd::SDPBackend::Math,
//                        bool bias = true);

//     /// Self-attention: Q = K = V derived from the same input
//     Tensor forward(const Tensor& x) override;

//     /// Pre-projected attention: caller provides Q, K, V already shaped (B, nh, T, hd)
//     Tensor forward_qkv(const Tensor& q, const Tensor& k, const Tensor& v);
// };

// } // namespace nn
// } // namespace OwnTensor


#pragma once

#include "nn/NN.h"
#include "autograd/operations/AttentionOps.h"

namespace OwnTensor {
namespace nn {

/**
 * @brief Multi-Head Self-Attention module.
 *
 * Implements the standard transformer attention block:
 *   1. Project input to Q, K, V via a single linear layer
 *   2. Split into num_heads parallel heads
 *   3. Compute scaled_dot_product_attention (backend-selectable)
 *   4. Concatenate heads and project output
 *
 * LayerNorm and residual connections are NOT included —
 * they are the caller's responsibility (matching PyTorch convention).
 *
 * Usage:
 *   MultiHeadAttention mha(384, 6, true, autograd::SDPBackend::Math);
 *   mha.to(device);
 *   Tensor out = mha.forward(x);  // x: [B, T, C]  ->  out: [B, T, C]
 */
class MultiHeadAttention : public Module {
public:
    int embed_dim;
    int num_heads;
    int head_dim;
    bool is_causal;
    autograd::SDPBackend backend;

    Linear qkv_proj;   // (embed_dim) -> (3 * embed_dim)
    Linear out_proj;    // (embed_dim) -> (embed_dim)

    /**
     * @param embed_dim  Total embedding dimension (must be divisible by num_heads)
     * @param num_heads  Number of attention heads
     * @param is_causal  Whether to apply causal (autoregressive) mask
     * @param backend    Attention computation backend
     * @param bias       Whether linear layers have bias
     */
    MultiHeadAttention(int embed_dim, int num_heads,
                       bool is_causal = true,
                       autograd::SDPBackend backend = autograd::SDPBackend::Math,
                       bool bias = true);

    /// Self-attention: Q = K = V derived from the same input
    Tensor forward(const Tensor& x) override;

    /// Pre-projected attention: caller provides Q, K, V already shaped (B, nh, T, hd)
    Tensor forward_qkv(const Tensor& q, const Tensor& k, const Tensor& v);
};

} // namespace nn
} // namespace OwnTensor