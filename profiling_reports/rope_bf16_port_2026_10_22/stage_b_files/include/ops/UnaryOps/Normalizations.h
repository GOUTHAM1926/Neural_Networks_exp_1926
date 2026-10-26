#pragma once

#ifndef OWNTENSOR_NORMALIZATIONS_H
#define OWNTENSOR_NORMALIZATIONS_H

#include "core/Tensor.h"

namespace OwnTensor {

// =================================================================
// Pure math normalization functions (no autograd, no graph recording).
// Handle full CPU/GPU dispatching and dtype dispatch internally.
// The autograd wrappers in NormalizationOps.cpp just call these
// and then record the graph — identical to how Activations.h works.
// =================================================================

// ── Forward results ─────────────────────────────────────────────

struct LayerNormForwardResult {
    Tensor output;
    Tensor mean;   // [rows], always float32
    Tensor rstd;   // [rows], always float32
};

struct RMSNormForwardResult {
    Tensor output;
    Tensor rstd;   // [rows], always float32 (no mean for RMSNorm)
};

// ── Backward results ────────────────────────────────────────────

struct LayerNormBackwardResult {
    Tensor grad_input;
    Tensor grad_weight;  // always float32
    Tensor grad_bias;    // always float32
};

struct RMSNormBackwardResult {
    Tensor grad_input;
    Tensor grad_weight;  // always float32
};

struct FusedRMSRoPEForwardResult {
    Tensor q_out;   // [B, Hq, T, hd]
    Tensor k_out;   // [B, Hkv, T, hd]
    Tensor q_rstd;  // [B, Hq, T], always float32
    Tensor k_rstd;  // [B, Hkv, T], always float32
};

struct FusedRMSRoPEBackwardResult {
    Tensor grad_q;
    Tensor grad_k;
    Tensor grad_q_gamma;
    Tensor grad_k_gamma;
};

// ── LayerNorm ───────────────────────────────────────────────────

/**
 * @brief LayerNorm forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *        CPU: Welford one-pass stats + AVX2 vectorized normalize.
 *        GPU: fused Welford + vectorized float4 kernel.
 *        All dtypes (fp32/fp16/bf16), stats always in fp32.
 */
LayerNormForwardResult layer_norm_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int normalized_shape,
    float eps = 1e-5f);

/**
 * @brief LayerNorm backward: grad_input, grad_weight, grad_bias.
 *        CPU: AVX2 vectorized + OpenMP parallel reduction for gamma/beta grads.
 *        GPU: 2-kernel approach (gamma/beta column-reduce + input row-reduce).
 */
LayerNormBackwardResult layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int normalized_shape,
    float eps = 1e-5f,
    bool need_grad_input = true,
    bool need_grad_weight = true,
    bool need_grad_bias = true);

// ── RMSNorm ─────────────────────────────────────────────────────

/**
 * @brief RMSNorm forward: y = gamma * x / sqrt(mean(x^2) + eps)
 *        Same kernel as LayerNorm with rms_norm=true (no mean subtraction, no beta).
 *        CPU: one-pass sum-of-squares + AVX2 normalize.
 *        GPU: fused kernel with rms_norm template flag.
 */
RMSNormForwardResult rms_norm_forward(
    const Tensor& input,
    const Tensor& weight,
    int normalized_shape,
    float eps = 1e-5f);

/**
 * @brief RMSNorm backward: grad_input, grad_weight (no grad_bias).
 */
RMSNormBackwardResult rms_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& rstd,
    const Tensor& weight,
    int normalized_shape,
    float eps = 1e-5f,
    bool need_grad_input = true,
    bool need_grad_weight = true);

// ── Fused RMSNorm + RoPE ────────────────────────────────────────

/**
 * @brief Fused QK-norm (RMSNorm) + RoPE forward on Query and Key tensors.
 *        Executes fused CUDA kernel fusing RMS scale + RoPE rotation.
 *
 * @param positions  (v2 packing, added 2026-10-10) optional int32 CUDA tensor [B, T]:
 *                   the RoPE position of every token, so packed chats can each restart
 *                   at 0. Default (invalid Tensor) = unchanged behaviour, position =
 *                   t + pos_offset. When given, pos_offset must be 0 and every value must
 *                   be < the cos_sin_cache length (the caller guarantees that; the loader
 *                   checks it on the host).
 */
FusedRMSRoPEForwardResult fused_rms_rope_forward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset = 0,
    float eps = 1e-5f,
    bool interleaved = false,
    const Tensor& positions = Tensor());

/**
 * @brief Fused QK-norm (RMSNorm) + RoPE backward on Query and Key tensors.
 *        positions must be the same tensor the forward used (see fused_rms_rope_forward).
 */
FusedRMSRoPEBackwardResult fused_rms_rope_backward(
    const Tensor& grad_q,
    const Tensor& grad_k,
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    const Tensor& q_rstd,
    const Tensor& k_rstd,
    int64_t pos_offset = 0,
    bool interleaved = false,
    const Tensor& positions = Tensor());

FusedRMSRoPEForwardResult fused_rms_rope_forward_bf16_tm(
    const Tensor& q,
    const Tensor& k,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int64_t pos_offset = 0,
    float eps = 1e-5f,
    bool interleaved = false,
    const Tensor& positions = Tensor());   // optional int32 [B, T], as in fused_rms_rope_forward (2026-10-22)

FusedRMSRoPEBackwardResult fused_rms_rope_backward_bf16_tm(
    const Tensor& grad_q,
    const Tensor& grad_k,
    const Tensor& q,
    const Tensor& k,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    const Tensor& q_rstd,
    const Tensor& k_rstd,
    int64_t pos_offset = 0,
    bool interleaved = false,
    const Tensor& positions = Tensor());   // the SAME positions the forward used (2026-10-22)

} // namespace OwnTensor

#endif // OWNTENSOR_NORMALIZATIONS_H

