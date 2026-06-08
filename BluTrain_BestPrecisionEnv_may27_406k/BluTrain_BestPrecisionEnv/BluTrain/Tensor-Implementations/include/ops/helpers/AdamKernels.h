#pragma once
#include <cstdint>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Fused Adam optimizer kernel
 * 
 * Performs the complete Adam update in a single kernel launch:
 * - m = beta1 * m + (1 - beta1) * grad
 * - v = beta2 * v + (1 - beta2) * grad^2
 * - m_hat = m / (1 - beta1^t)
 * - v_hat = v / (1 - beta2^t) 
 * - param = param - lr * m_hat / (sqrt(v_hat) + eps)
 * - Optionally applies weight decay (AdamW style)
 */
void fused_adam_cuda(
    float* param,           // Parameter tensor (in-place update)
    const float* grad,      // Gradient tensor
    float* m,               // First moment (in-place update)
    float* v,               // Second moment (in-place update)
    int64_t numel,          // Number of elements
    float lr,               // Learning rate
    float beta1,            // First moment decay
    float beta2,            // Second moment decay
    float eps,              // Epsilon for numerical stability
    float weight_decay,     // Weight decay coefficient
    float bias_correction1, // 1 - beta1^t
    float bias_correction2, // 1 - beta2^t
    bool is_adamw  
);

} // namespace cuda
} // namespace OwnTensor
