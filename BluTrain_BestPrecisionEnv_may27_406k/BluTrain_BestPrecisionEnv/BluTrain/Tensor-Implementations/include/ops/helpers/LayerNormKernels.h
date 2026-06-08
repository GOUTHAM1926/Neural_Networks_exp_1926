#pragma once
#include <cuda_runtime.h>
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// ── LayerNorm Forward (mean/rstd always float32) ──
void layer_norm_forward_cuda(
    const float* x, const float* gamma, const float* beta,
    float* y, float* mean, float* rstd, int rows, int cols, float eps);
void layer_norm_forward_cuda(
    const float16_t* x, const float16_t* gamma, const float16_t* beta,
    float16_t* y, float* mean, float* rstd, int rows, int cols, float eps);
void layer_norm_forward_cuda(
    const bfloat16_t* x, const bfloat16_t* gamma, const bfloat16_t* beta,
    bfloat16_t* y, float* mean, float* rstd, int rows, int cols, float eps);

// ── LayerNorm Backward (grad_gamma/grad_beta always float32) ──
void layer_norm_backward_cuda(
    const float* grad_y, const float* x, const float* mean, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);
void layer_norm_backward_cuda(
    const float16_t* grad_y, const float16_t* x, const float* mean, const float* rstd, const float16_t* gamma,
    float16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);
void layer_norm_backward_cuda(
    const bfloat16_t* grad_y, const bfloat16_t* x, const float* mean, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* grad_x, float* grad_gamma, float* grad_beta, int rows, int cols);

// ── RMSNorm Forward (no mean, no beta — same fused kernel with rms_norm=true) ──
void rms_norm_forward_cuda(
    const float* x, const float* gamma,
    float* y, float* rstd, int rows, int cols, float eps);
void rms_norm_forward_cuda(
    const float16_t* x, const float16_t* gamma,
    float16_t* y, float* rstd, int rows, int cols, float eps);
void rms_norm_forward_cuda(
    const bfloat16_t* x, const bfloat16_t* gamma,
    bfloat16_t* y, float* rstd, int rows, int cols, float eps);

// ── RMSNorm Backward (no grad_beta — only grad_gamma + grad_input) ──
void rms_norm_backward_cuda(
    const float* grad_y, const float* x, const float* rstd, const float* gamma,
    float* grad_x, float* grad_gamma, int rows, int cols);
void rms_norm_backward_cuda(
    const float16_t* grad_y, const float16_t* x, const float* rstd, const float16_t* gamma,
    float16_t* grad_x, float* grad_gamma, int rows, int cols);
void rms_norm_backward_cuda(
    const bfloat16_t* grad_y, const bfloat16_t* x, const float* rstd, const bfloat16_t* gamma,
    bfloat16_t* grad_x, float* grad_gamma, int rows, int cols);

} // namespace cuda
} // namespace OwnTensor
