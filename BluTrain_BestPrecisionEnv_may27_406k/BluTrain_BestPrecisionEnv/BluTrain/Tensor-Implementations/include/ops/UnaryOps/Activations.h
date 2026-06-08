#pragma once

#ifndef OWNTENSOR_ACTIVATIONS_H
#define OWNTENSOR_ACTIVATIONS_H

#include "core/Tensor.h"

namespace OwnTensor {

// =================================================================
// Pure math activation functions (no autograd, no graph recording).
// Handle full CPU/GPU dispatching and dtype dispatch internally,
// analogous to how reduce_sum / reduce_mean work in Reduction.h.
// The autograd wrappers in ActivationOps.cpp just call these
// and then record the graph.
// =================================================================

// ── Forward pass ─────────────────────────────────────────────────

/**
 * @brief ReLU forward: max(0, x)
 *        Dispatches to CUDA kernel (float32) or CPU tensor ops.
 */
Tensor relu_forward(const Tensor& input);

/**
 * @brief GeLU forward (tanh approximation):
 *        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *        Dispatches to fused CUDA kernel (float32/fp16/bf16) or CPU tensor ops.
 */
Tensor gelu_forward(const Tensor& input);

/**
 * @brief Sigmoid forward: 1 / (1 + exp(-x))
 *        Dispatches to CUDA kernel (float32/fp16/bf16) or CPU tensor ops.
 */
Tensor sigmoid_forward(const Tensor& input);

/**
 * @brief Softmax forward: exp(x - max) / sum(exp(x - max))
 *        Dispatches to CUDA kernel (float32/fp16/bf16, last-dim only) or CPU tensor ops.
 * @param dim  The dimension along which softmax is computed
 */
Tensor softmax_forward(const Tensor& input, int64_t dim = -1);

/**
 * @brief SwiGLU forward: swish(A) * B where input is [... , 2*hidden]
 *        Dispatches to CUDA kernel (float32/fp16/bf16) or CPU tensor ops.
 */
Tensor swiglu_forward(const Tensor& input);

/**
 * @brief Fused bias + GeLU: output = gelu(input + bias)
 *        Dispatches to fused CUDA kernel (float32) or CPU tensor ops.
 * @param bias  Bias vector broadcast along the last dimension
 */
Tensor fused_bias_gelu_forward(const Tensor& input, const Tensor& bias);

// ── Dropout ──────────────────────────────────────────────────────

struct DropoutForwardResult {
    Tensor output;
    Tensor mask;
};

/**
 * @brief Dropout forward: randomly zeros elements with probability p,
 *        scales remaining by 1/(1-p). Returns output + mask (for backward).
 *        Dispatches to CUDA kernel or CPU implementation.
 */
DropoutForwardResult dropout_forward(const Tensor& input, float p);

// ── Fused tril + softmax ────────────────────────────────────────

/**
 * @brief Fused tril mask + softmax: equivalent to softmax(tril(x, diagonal, value))
 *        GPU: uses fused kernel (single pass in shared memory).
 *        CPU: composes tril() + softmax_forward() separately.
 */
Tensor fused_tril_softmax_forward(const Tensor& input, int64_t diagonal = 0, double value = 0.0);

// =================================================================
// Backward pass — CPU/GPU dispatch + optimized CPU kernels
// =================================================================

Tensor relu_backward(const Tensor& grad_output, const Tensor& input);
Tensor gelu_backward(const Tensor& grad_output, const Tensor& input);
Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output);
Tensor softmax_backward(const Tensor& grad_output, const Tensor& output, int64_t dim);
Tensor dropout_backward(const Tensor& grad_output, const Tensor& mask, float scale);
Tensor swiglu_backward(const Tensor& grad_output, const Tensor& input);

struct FusedBiasGeLUBackwardResult {
    Tensor grad_input;
    Tensor grad_bias;
};
FusedBiasGeLUBackwardResult fused_bias_gelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& bias);

} // namespace OwnTensor

#endif // OWNTENSOR_ACTIVATIONS_H
