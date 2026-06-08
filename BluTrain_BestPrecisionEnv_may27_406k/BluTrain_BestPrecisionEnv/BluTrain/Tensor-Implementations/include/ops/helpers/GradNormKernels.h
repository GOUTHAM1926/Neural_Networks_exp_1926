#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute sum of squared elements for a gradient tensor
 * Adds result to an accumulator (atomically)
 */
void grad_norm_squared_cuda(
    const float* grad,
    float* norm_sq_accumulator,  // Single float on GPU to accumulate into
    int64_t numel
);

/**
 * @brief Compute infinity norm (max abs value) for a gradient tensor
 * Updates accumulator with max value (atomically)
 */
void grad_norm_inf_cuda(
    const float* grad,
    float* norm_inf_accumulator,  // Single float on GPU
    int64_t numel
);

/**
 * @brief Scale all gradients by a coefficient
 */
void scale_gradients_cuda(
    float* grad,
    float scale,
    int64_t numel
);

/**
 * @brief Compute clip coefficient on GPU: clip_coef = max_norm / (sqrt(norm_sq) + eps)
 * @param norm_sq_or_inf  Single float on GPU containing norm² (L2) or max_abs (Inf)
 * @param clip_coef_out   Output: clip coefficient (clamped to max 1.0) 
 * @param max_norm        Maximum allowed norm
 * @param is_inf_norm     If true, norm_sq_or_inf is already the inf norm (no sqrt)
 */
void compute_clip_coef_cuda(
    float* norm_sq_or_inf,
    float* clip_coef_out,
    float max_norm,
    bool is_inf_norm
);

/**
 * @brief Scale gradients using clip coefficient stored on GPU
 */
void scale_gradients_with_gpu_coef_cuda(
    float* grad,
    const float* clip_coef,  // Single float on GPU
    int64_t numel
);

} // namespace cuda
} // namespace OwnTensor
