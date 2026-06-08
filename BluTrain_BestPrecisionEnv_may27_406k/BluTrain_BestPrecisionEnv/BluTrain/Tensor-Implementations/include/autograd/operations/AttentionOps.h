
// // #pragma once

// // #include "core/Tensor.h"

// // namespace OwnTensor {
// // namespace autograd {

// // /**
// //  * @brief Backend selection for scaled_dot_product_attention.
// //  *
// //  * Math:             Composes existing ops (matmul + fused_tril_softmax).
// //  *                   Materializes the full T×T attention matrix.
// //  * MemoryEfficient:  Custom CUDA kernel using online softmax.
// //  *                   Avoids T×T allocation — O(T) extra memory.
// //  */
// // enum class SDPBackend {
// //     Math,
// //     MemoryEfficient,
// //     // FlashAttention,  // Future
// // };

// // /**
// //  * @brief Autograd-aware scaled dot-product attention.
// //  *
// //  * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
// //  * with optional causal mask.
// //  *
// //  * @param query     (B, nh, T, hd) query tensor
// //  * @param key       (B, nh, T, hd) key tensor (S instead of T for cross-attention)
// //  * @param value     (B, nh, T, hd) value tensor
// //  * @param is_causal If true, applies lower-triangular causal mask
// //  * @param backend   Which implementation to use
// //  * @return          (B, nh, T, hd) attention output
// //  */
// // Tensor scaled_dot_product_attention(
// //     const Tensor& query,
// //     const Tensor& key,
// //     const Tensor& value,
// //     bool is_causal = true,
// //     SDPBackend backend = SDPBackend::Math
// // );

// // } // namespace autograd
// // } // namespace OwnTensor

// #pragma once

// #include "core/Tensor.h"

// namespace OwnTensor {
// namespace autograd {

// /**
//  * @brief Backend selection for scaled_dot_product_attention.
//  *
//  * Math:             Composes existing ops (matmul + fused_tril_softmax).
//  *                   Materializes the full T×T attention matrix.
//  * MemoryEfficient:  Custom CUDA kernel using online softmax.
//  *                   Avoids T×T allocation — O(T) extra memory.
//  */
// enum class SDPBackend {
//     Math,
//     MemoryEfficient,
//     // FlashAttention,  // Future
// };

// /**
//  * @brief Autograd-aware scaled dot-product attention.
//  *
//  * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
//  * with optional causal mask.
//  *
//  * @param query     (B, nh, T, hd) query tensor
//  * @param key       (B, nh, T, hd) key tensor (S instead of T for cross-attention)
//  * @param value     (B, nh, T, hd) value tensor
//  * @param is_causal If true, applies lower-triangular causal mask
//  * @param backend   Which implementation to use
//  * @return          (B, nh, T, hd) attention output
//  */
// Tensor scaled_dot_product_attention(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal = true,
//     float dropout_p = 0.0f,
//     SDPBackend backend = SDPBackend::Math
// );

// /**
//  * @brief Fused shard-aware attention over a packed QKV tensor.
//  *
//  * Replaces the [shard → reshape → transpose → SDPA → transpose → reshape]
//  * chain in transformer attention blocks. Reads Q/K/V directly from the
//  * packed `qkv` via strided pointers, writes a packed output [B, T, C], and
//  * — in the backward — writes dQ/dK/dV directly into a single packed dqkv
//  * tensor (no Tensor::cat needed). Eliminates `cat_batched_kernel` calls and
//  * the bulk of `generic_strided_copy_kernel` calls that the unfused chain
//  * incurs around SDPA on each side.
//  *
//  * @param qkv         (B, T, 3 * n_heads * head_dim) packed projection output
//  * @param n_heads     number of attention heads
//  * @param is_causal   if true, applies lower-triangular causal mask
//  * @param dropout_p   dropout probability (currently must be 0.0)
//  * @param backend     which implementation to use (only MemoryEfficient is fused)
//  * @return            (B, T, n_heads * head_dim) attention output, packed
//  */
// Tensor scaled_dot_product_attention_packed(
//     const Tensor& qkv,
//     int64_t n_heads,
//     bool is_causal = true,
//     float dropout_p = 0.0f,
//     SDPBackend backend = SDPBackend::MemoryEfficient
// );

// /**
//  * @brief Fused tril_softmax + matmul for attention: avoids duplicate attn_probs storage.
//  *
//  * Computes:
//  *   attn_probs = fused_tril_softmax(attn_weights, diagonal, value)
//  *   attn_out   = attn_probs @ v
//  *
//  * Same numerics as the two separate ops, but the backward node stores
//  * attn_probs only once (instead of once in FusedTrilSoftmaxBackward and
//  * once in MatmulBackward::saved_a_).
//  *
//  * @param attn_weights  [B, H, T, T] raw attention scores
//  * @param v             [B, H, T, hd] value tensor
//  * @param diagonal      tril diagonal (usually 0 for causal)
//  * @param value         fill value for masked positions (usually -inf)
//  * @return              [B, H, T, hd] attention output
//  */
// Tensor fused_attn_softmax_matmul(
//     const Tensor& attn_weights,
//     const Tensor& v,
//     int64_t diagonal = 0,
//     double fill_value = 0.0);

// } // namespace autograd
// } // namespace OwnTensor


























// #pragma once

// #include "core/Tensor.h"

// namespace OwnTensor {
// namespace autograd {

// /**
//  * @brief Backend selection for scaled_dot_product_attention.
//  *
//  * Math:             Composes existing ops (matmul + fused_tril_softmax).
//  *                   Materializes the full T×T attention matrix.
//  * MemoryEfficient:  Custom CUDA kernel using online softmax.
//  *                   Avoids T×T allocation — O(T) extra memory.
//  */
// enum class SDPBackend {
//     Math,
//     MemoryEfficient,
//     // FlashAttention,  // Future
// };

// /**
//  * @brief Autograd-aware scaled dot-product attention.
//  *
//  * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
//  * with optional causal mask.
//  *
//  * @param query     (B, nh, T, hd) query tensor
//  * @param key       (B, nh, T, hd) key tensor (S instead of T for cross-attention)
//  * @param value     (B, nh, T, hd) value tensor
//  * @param is_causal If true, applies lower-triangular causal mask
//  * @param backend   Which implementation to use
//  * @return          (B, nh, T, hd) attention output
//  */
// Tensor scaled_dot_product_attention(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal = true,
//     SDPBackend backend = SDPBackend::Math
// );

// } // namespace autograd
// } // namespace OwnTensor

#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backend selection for scaled_dot_product_attention.
 *
 * Math:             Composes existing ops (matmul + fused_tril_softmax).
 *                   Materializes the full T×T attention matrix.
 * MemoryEfficient:  Custom CUDA kernel using online softmax.
 *                   Avoids T×T allocation — O(T) extra memory.
 */
enum class SDPBackend {
    Math,
    MemoryEfficient,
    // FlashAttention,  // Future
};

/**
 * @brief Autograd-aware scaled dot-product attention.
 *
 * Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
 * with optional causal mask.
 *
 * @param query     (B, nh, T, hd) query tensor
 * @param key       (B, nh, T, hd) key tensor (S instead of T for cross-attention)
 * @param value     (B, nh, T, hd) value tensor
 * @param is_causal If true, applies lower-triangular causal mask
 * @param backend   Which implementation to use
 * @return          (B, nh, T, hd) attention output
 */
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal = true,
    float dropout_p = 0.0f,
    SDPBackend backend = SDPBackend::Math
);

/**
 * @brief Fully-fused packed-QKV attention (zero copies).
 *
 * Reads Q/K/V directly from packed `qkv [B,T,3C]` via strided pointers
 * (no shard, no reshape, no transpose). Writes output as packed `[B,T,C]`
 * contiguous — feeds straight into c_proj. Backward writes dQ/dK/dV into
 * one packed dqkv [B,T,3C] (no Tensor::cat), feeds straight into c_attn
 * backward. Eliminates all attention-driven copy/cat traffic.
 *
 * @param qkv         [B, T, 3*n_heads*head_dim] packed projection output
 * @param n_heads     number of attention heads
 * @param is_causal   apply causal mask
 * @param dropout_p   currently must be 0.0
 * @param backend     only MemoryEfficient is supported
 * @return            [B, T, n_heads*head_dim] packed contiguous output
 */
Tensor scaled_dot_product_attention_packed(
    const Tensor& qkv,
    int64_t n_heads,
    bool is_causal = true,
    float dropout_p = 0.0f,
    SDPBackend backend = SDPBackend::MemoryEfficient
);

/**
 * @brief Fused tril_softmax + matmul for attention: avoids duplicate attn_probs storage.
 *
 * Computes:
 *   attn_probs = fused_tril_softmax(attn_weights, diagonal, value)
 *   attn_out   = attn_probs @ v
 *
 * Same numerics as the two separate ops, but the backward node stores
 * attn_probs only once (instead of once in FusedTrilSoftmaxBackward and
 * once in MatmulBackward::saved_a_).
 *
 * @param attn_weights  [B, H, T, T] raw attention scores
 * @param v             [B, H, T, hd] value tensor
 * @param diagonal      tril diagonal (usually 0 for causal)
 * @param value         fill value for masked positions (usually -inf)
 * @return              [B, H, T, hd] attention output
 */
Tensor fused_attn_softmax_matmul(
    const Tensor& attn_weights,
    const Tensor& v,
    int64_t diagonal = 0,
    double fill_value = 0.0);

} // namespace autograd
} // namespace OwnTensor