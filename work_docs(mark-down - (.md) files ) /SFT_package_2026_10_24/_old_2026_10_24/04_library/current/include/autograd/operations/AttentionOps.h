#pragma once

#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"   // cuda::GqaChatPlanPtr (v2 packed rows)

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
    FlashAttention,
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
    SDPBackend backend = SDPBackend::Math,
    // Optional [nh] device buffer; when non-null AND backend==MemoryEfficient,
    // the fused kernel atomicMaxes each head's max pre-softmax logit (S^h_max)
    // into it for QK-Clip (MuonClip). Default nullptr = no capture.
    float* qk_max_out = nullptr
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
    SDPBackend backend = SDPBackend::MemoryEfficient,
    // Optional [nh] device buffer for per-head S^h_max capture (QK-Clip).
    float* qk_max_out = nullptr
);

/**
 * @brief GQA high-level API (grouped-query attention).
 *
 * Reads Q/K/V directly from a single packed projection output via strided
 * pointers (no shard/reshape/transpose). The packed last dim is laid out as
 *   [ Q: Nq_heads*hd | K: Nkv_heads*hd | V: Nkv_heads*hd ]
 * so the total width is (Nq_heads + 2*Nkv_heads)*hd. head_dim is inferred.
 * Nq_heads must be a multiple of Nkv_heads; query head h reads KV head
 * h / (Nq_heads / Nkv_heads).
 *
 * @param qkv        [B, T, (Nq_heads + 2*Nkv_heads)*head_dim] packed tensor
 * @param Nq_heads   number of query heads
 * @param Nkv_heads  number of key/value heads (groups)
 * @param is_causal  apply causal mask
 * @param dropout_p  currently must be 0.0
 * @param backend    only FlashAttention is supported
 * @return           [B, T, Nq_heads*head_dim] packed contiguous output
 */
Tensor scaled_dot_product_attention(
  const Tensor& qkv,
  const int Nq_heads,
  const int Nkv_heads,
  bool is_causal = true,
  float dropout_p = 0.0f,
  SDPBackend backend = SDPBackend::FlashAttention
);

/**
 * @brief GQA high-level API (grouped-query attention), unpacked Q/K/V.
 *
 * Same fused bf16 WMMA kernel as the packed-qkv overload above, but Q/K/V are
 * passed as separate tensors instead of one packed buffer. They are
 * concatenated into the required block-concatenated [Q|K|V] bf16 layout
 * internally before the kernel call (one extra copy vs. the packed overload,
 * which is already laid out that way by construction).
 *
 * @param query      [B, Nq_heads,  T, head_dim] bf16
 * @param key        [B, Nkv_heads, T, head_dim] bf16
 * @param value      [B, Nkv_heads, T, head_dim] bf16
 * @param Nq_heads   number of query heads
 * @param Nkv_heads  number of key/value heads (groups)
 * @param is_causal  apply causal mask
 * @param dropout_p  currently must be 0.0
 * @param backend    only FlashAttention is supported
 * @return           [B, Nq_heads, T, head_dim] attention output
 */
Tensor scaled_dot_product_attention(
  const Tensor& query,
  const Tensor& key,
  const Tensor& value,
  const int Nq_heads,
  const int Nkv_heads,
  bool is_causal = true,
  float dropout_p = 0.0f,
  SDPBackend backend = SDPBackend::FlashAttention,
  // v2 packed rows (2026-10-15): optional per-chat plan (design A, the cu_seqlens way; GqaChatPlan in
  // ops/helpers/AttentionKernels.h) built for this (B, T, Nq_heads). nullptr (default) = plain causal over the whole
  // row, as before. The backward node keeps the plan: the backward must use the same one.
  const cuda::GqaChatPlanPtr& chats = nullptr
);

/**
 * @brief FUSED grouped-query attention with QK-norm + RoPE (autograd-aware).
 *
 * Single high-level entry point over cuda::gqa_fused_flash_attn_forward /
 * gqa_fused_flash_attn_backward. Per (token, head) over head_dim it applies, in
 * one kernel: (1) RMSNorm of Q and K (the "QK-norm", scales q_gamma / k_gamma),
 * (2) RoPE rotation of the normalized Q and K, (3) grouped-query flash attention
 * (optionally causal) against the raw V.
 *
 * Precision bridge: the model stays fp32 while the kernel runs bf16 — Q/K/V are
 * cast to bf16 and packed into one block-concatenated buffer internally, and the
 * fp32 attention output is returned. The QK-norm scales stay fp32. Gradients
 * flow back to Q, K, V (fp32) and q_gamma / k_gamma (fp32).
 *
 * @param query          [B, Nq_heads,  T, hd] fp32 (head-major, contiguous-able)
 * @param key            [B, Nkv_heads, T, hd] fp32
 * @param value          [B, Nkv_heads, T, hd] fp32
 * @param q_gamma        [hd] fp32 QK-norm scale for Q (invalid Tensor => skip Q QK-norm)
 * @param k_gamma        [hd] fp32 QK-norm scale for K (invalid Tensor => skip K QK-norm)
 * @param cos_sin_cache  [cache_seq_len, hd] fp32 RoPE cache (build_rope_cache)
 * @param Nq_heads       number of query heads
 * @param Nkv_heads      number of key/value heads; Nq_heads % Nkv_heads == 0
 * @param is_causal      apply causal mask (key index <= query index)
 * @param interleaved    false = NeoX half-split pairs (default), true = GPT-J
 * @param eps            RMSNorm epsilon for the QK-norm
 * @param pos_offset     absolute RoPE start position (0 for from-scratch training)
 * @param backend        only FlashAttention is supported
 * @return               [B, Nq_heads, T, hd] fp32 attention output
 */
Tensor scaled_dot_product_attention(
  const Tensor& query,
  const Tensor& key,
  const Tensor& value,
  const Tensor& q_gamma,
  const Tensor& k_gamma,
  const Tensor& cos_sin_cache,
  int Nq_heads,
  int Nkv_heads,
  bool is_causal = true,
  bool interleaved = false,
  float eps = 1e-6f,
  int pos_offset = 0,
  SDPBackend backend = SDPBackend::FlashAttention
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