#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"   // cuda::GqaChatPlanPtr (v2 packed rows)

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for memory-efficient scaled dot-product attention.
 *
 * Forward:
 *   O = softmax(Q @ K^T / sqrt(hd)) @ V
 *   (computed without materializing the full T×T attention matrix)
 *
 * Backward:
 *   Recomputes attention weights block-by-block from saved Q, K, V, and LSE.
 *   Uses separate kernels for dQ and dK/dV.
 *
 *   dS[i,j] = dO[i] @ V[j]^T
 *   D[i]    = dO[i] . O[i]
 *   dP[i,j] = p[i,j] * (dS[i,j] - D[i])       where p[i,j] = exp(s[i,j] - LSE[i])
 *   dQ[i]   = scale * sum_j dP[i,j] * K[j]
 *   dK[j]   = scale * sum_i dP[i,j] * Q[i]
 *   dV[j]   = sum_i p[i,j] * dO[i]
 *
 * Saves: Q, K, V (detached), O (detached), LSE (log-sum-exp per row)
 * Has 3 inputs: Q (edge 0), K (edge 1), V (edge 2)
 */
class MemEfficientAttentionBackward : public Node {
private:
    Tensor saved_query_;
    Tensor saved_key_;
    Tensor saved_value_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, nh_, T_, hd_;
    bool is_causal_;

public:
    MemEfficientAttentionBackward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t nh, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "MemEfficientAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_query_  = Tensor();
        saved_key_    = Tensor();
        saved_value_  = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

/**
 * @brief Backward for scaled_dot_product_attention_packed.
 *
 * Forward saved a packed `qkv [B, T, 3*C]` and a packed output `O [B, T, C]`.
 * Backward allocates one zeroed `dqkv [B, T, 3*C]`, passes strided views
 * (Q@0, K@C, V@2C — same stride layout as forward) into the unified
 * mem-efficient backward kernel with skip_grad_zero=true, and returns the
 * single packed dqkv. No Tensor::cat needed downstream.
 */
class PackedSDPABackward : public Node {
private:
    Tensor saved_qkv_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, nh_, T_, hd_, C_;
    bool is_causal_;

public:
    PackedSDPABackward(
        const Tensor& qkv,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t nh, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "PackedSDPABackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_qkv_    = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

/**
 * @brief Backward for the packed grouped-query (GQA) flash-attention forward.
 *
 * Forward saved a packed qkv [B, T, (Nq + 2*Nkv)*hd] and a packed output
 * O [B, T, Nq*hd]. Backward allocates one zeroed dqkv with the SAME packed
 * layout, passes strided views (Q@0, K@Nq*hd, V@(Nq+Nkv)*hd) into the GQA
 * backward kernel with skip_grad_zero=true, and returns the single packed
 * dqkv. dK/dV gradients are accumulated across each group of query heads
 * inside the kernel. Single input edge: qkv.
 */
class GQAFlashAttentionBackward : public Node {
private:
    Tensor saved_qkv_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, Nq_heads_, Nkv_heads_, T_, hd_;
    bool is_causal_;

public:
    GQAFlashAttentionBackward(
        const Tensor& qkv,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
        bool is_causal);

    const char* name() const override { return "GQAFlashAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_qkv_    = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
    }
};

/**
 * @brief Backward for the unpacked-QKV grouped-query (GQA) flash-attention
 *        forward (Q, K, V passed as three independent device buffers instead
 *        of one packed buffer).
 *
 * Runs cuda::gqa_flash_attn_backward_qkv -- the same kernels as
 * GQAFlashAttentionBackward, but reading/writing Q/K/V and dQ/dK/dV as
 * separate buffers, so no pack/split copy is needed on either side. dK/dV
 * accumulation across each group of query heads happens inside the kernel.
 *
 * 3 input edges: Q (0), K (1), V (2).
 */
class GQAFlashAttentionQKVBackward : public Node {
private:
    Tensor saved_query_;
    Tensor saved_key_;
    Tensor saved_value_;
    Tensor saved_output_;
    Tensor saved_lse_;
    int64_t B_, Nq_heads_, Nkv_heads_, T_, hd_;
    bool is_causal_;
    cuda::GqaChatPlanPtr chats_;   // v2 packed rows (design A); null = dense. The SAME plan as the forward.
    int window_;                   // sliding window in keys (2026-11-04); 0 = plain causal. The SAME as the forward.

public:
    GQAFlashAttentionQKVBackward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& output, const Tensor& lse,
        int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
        bool is_causal, cuda::GqaChatPlanPtr chats = nullptr, int window = 0);

    const char* name() const override { return "GQAFlashAttentionQKVBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_query_  = Tensor();
        saved_key_    = Tensor();
        saved_value_  = Tensor();
        saved_output_ = Tensor();
        saved_lse_    = Tensor();
        chats_.reset();
    }
};

/**
 * @brief Backward for the FUSED grouped-query flash-attention forward
 *        (QK-norm + RoPE + GQA causal attention).
 *
 * Bridges the fp32 model <-> bf16 kernel. The forward saved: the raw bf16 packed
 * qkv [ Q:[B,Nq,T,hd] | K:[B,Nkv,T,hd] | V:[B,Nkv,T,hd] ], the bf16 output
 * [B,Nq,T,hd], fp32 LSE [B,Nq,T], and fp32 q_rstd [B,Nq,T] / k_rstd [B,Nkv,T]
 * (the per-row RMSNorm reciprocal-std needed to replay QK-norm backward), plus
 * the fp32 cos/sin cache and the fp32 QK-norm scales q_gamma/k_gamma [hd].
 *
 * apply() casts the incoming fp32 output-grad to bf16, runs
 * cuda::gqa_fused_flash_attn_backward (attention-bwd -> RoPE-bwd -> RMSNorm-bwd),
 * then splits the block-concatenated bf16 grad_qkv and casts each block back to
 * fp32, returning dQ [B,Nq,T,hd], dK [B,Nkv,T,hd], dV [B,Nkv,T,hd] and the fp32
 * dq_gamma / dk_gamma [hd].
 *
 * 5 input edges: Q (0), K (1), V (2), q_gamma (3), k_gamma (4). Edges 3/4 are
 * only set when QK-norm is enabled and the gamma requires grad.
 */
class GQAFusedFlashAttentionBackward : public Node {
private:
    Tensor saved_qkv_;      // bf16 packed [Q | K | V] (raw, pre-norm, pre-rope)
    Tensor saved_output_;   // bf16 [B, Nq, T, hd]
    Tensor saved_lse_;      // fp32 [B, Nq, T]
    Tensor saved_q_rstd_;   // fp32 [B, Nq, T]
    Tensor saved_k_rstd_;   // fp32 [B, Nkv, T]
    Tensor saved_q_gamma_;  // fp32 [hd] or invalid (QK-norm off for Q)
    Tensor saved_k_gamma_;  // fp32 [hd] or invalid (QK-norm off for K)
    Tensor saved_cache_;    // fp32 [cache_seq_len, hd]
    int64_t B_, Nq_heads_, Nkv_heads_, T_, hd_;
    int cache_seq_len_, pos_offset_;
    float eps_;
    bool interleaved_, is_causal_;
    bool has_q_gamma_, has_k_gamma_;

public:
    GQAFusedFlashAttentionBackward(
        const Tensor& qkv, const Tensor& output, const Tensor& lse,
        const Tensor& q_rstd, const Tensor& k_rstd,
        const Tensor& q_gamma, const Tensor& k_gamma, const Tensor& cos_sin_cache,
        int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
        int cache_seq_len, int pos_offset, float eps,
        bool interleaved, bool is_causal);

    const char* name() const override { return "GQAFusedFlashAttentionBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override {
        saved_qkv_     = Tensor();
        saved_output_  = Tensor();
        saved_lse_     = Tensor();
        saved_q_rstd_  = Tensor();
        saved_k_rstd_  = Tensor();
        saved_q_gamma_ = Tensor();
        saved_k_gamma_ = Tensor();
        saved_cache_   = Tensor();
    }
};

} // namespace autograd
} // namespace OwnTensor