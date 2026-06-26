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

} // namespace OwnTensor

#endif // OWNTENSOR_NORMALIZATIONS_H
