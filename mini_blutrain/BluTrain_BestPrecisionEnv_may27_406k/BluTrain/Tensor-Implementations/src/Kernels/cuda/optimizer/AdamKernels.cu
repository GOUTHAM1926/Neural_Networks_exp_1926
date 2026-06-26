#include "ops/helpers/AdamKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace OwnTensor {
namespace cuda {

// Vectorized load/store structure for aligned memory access
struct alignas(16) float4_aligned {
    float x, y, z, w;
};

// Warp-level reduction for better performance
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized Adam kernel with vectorization and ILP
template <int VecSize = 4>
__global__ void fused_adam_kernel_optimized(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    bool is_adamw
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Check if we can use vectorized access
    bool use_vectorized = (numel >= VecSize) &&
                         (reinterpret_cast<uintptr_t>(param) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(grad) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(m) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(v) % 16 == 0);

    if (use_vectorized) {
        // Vectorized path for better memory bandwidth
        int64_t vec_numel = numel / VecSize;
        float4_aligned* param_vec = reinterpret_cast<float4_aligned*>(param);
        const float4_aligned* grad_vec = reinterpret_cast<const float4_aligned*>(grad);
        float4_aligned* m_vec = reinterpret_cast<float4_aligned*>(m);
        float4_aligned* v_vec = reinterpret_cast<float4_aligned*>(v);

        for (int64_t i = idx; i < vec_numel; i += stride) {
            // Load vectorized data
            float4_aligned p = param_vec[i];
            float4_aligned g = grad_vec[i];
            float4_aligned m_val = m_vec[i];
            float4_aligned v_val = v_vec[i];

            // Process each element in the vector
            #pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                float* p_ptr = &p.x + j;
                float g_val = (&g.x)[j];
                float* m_ptr = &m_val.x + j;
                float* v_ptr = &v_val.x + j;

                // Weight decay handling
                if (weight_decay > 0.0f) {
                    if (is_adamw) {
                       // AdamW: Decoupled weight decay (directly on params)
                        *p_ptr *= (1.0f - lr * weight_decay);
                    } else {
                       // Adam: Coupled L2 regularization (add to gradient)
                        g_val += weight_decay * *p_ptr;
                    }
                }

                // Update moments
                float m_new = beta1 * *m_ptr + (1.0f - beta1) * g_val;
                float v_new = beta2 * *v_ptr + (1.0f - beta2) * g_val * g_val;

                // Bias correction and parameter update
                float m_hat = m_new / bias_correction1;
                float v_hat = v_new / bias_correction2;
                *p_ptr -= lr * m_hat / (sqrtf(v_hat) + eps);

                *m_ptr = m_new;
                *v_ptr = v_new;
            }

            // Store vectorized results
            param_vec[i] = p;
            m_vec[i] = m_val;
            v_vec[i] = v_val;
        }

        // Handle remaining elements for vectorized path
        int64_t start = vec_numel * VecSize;
        #pragma unroll 4
        for (int64_t i = start + idx; i < numel; i += stride) {
            float g = grad[i];
            float p = param[i];

            // Weight decay handling
            if (weight_decay > 0.0f) {
                if (is_adamw) {
                   // AdamW: Decoupled weight decay
                    p *= (1.0f - lr * weight_decay);
                } else {
                   // Adam: Coupled L2 regularization
                    g += weight_decay * p;
                }
            }

            float m_new = beta1 * m[i] + (1.0f - beta1) * g;
            float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;

            float m_hat = m_new / bias_correction1;
            float v_hat = v_new / bias_correction2;
            p -= lr * m_hat / (sqrtf(v_hat) + eps);

            param[i] = p;
            m[i] = m_new;
            v[i] = v_new;
        }
    } else {
        // Full scalar fallback path (Unaligned or too small)
        #pragma unroll 4
        for (int64_t i = idx; i < numel; i += stride) {
            float g = grad[i];
            float p = param[i];

            // Weight decay handling
            if (weight_decay > 0.0f) {
                if (is_adamw) {
                   // AdamW: Decoupled weight decay
                    p *= (1.0f - lr * weight_decay);
                } else {
                   // Adam: Coupled L2 regularization
                    g += weight_decay * p;
               }
            }

            float m_new = beta1 * m[i] + (1.0f - beta1) * g;
            float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;

            float m_hat = m_new / bias_correction1;
            float v_hat = v_new / bias_correction2;
            p -= lr * m_hat / (sqrtf(v_hat) + eps);

            param[i] = p;
            m[i] = m_new;
            v[i] = v_new;
        }
    }
}

// Optimized launcher with better block calculation
void fused_adam_cuda(
    float* param,
    const float* grad,
    float* m,
    float* v,
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    bool is_adamw
) {
    if (numel == 0) return;

    const int threads_per_block = 256;

    // Calculate optimal number of blocks for better occupancy
    int num_blocks = (numel + threads_per_block - 1) / threads_per_block;

    // Ensure we have enough blocks for good GPU utilization
    const int min_blocks = 16;
    num_blocks = max(num_blocks, min_blocks);

    // Cap at maximum blocks
    const int max_blocks = 65535;
    num_blocks = min(num_blocks, max_blocks);

    // Launch optimized kernel
    fused_adam_kernel_optimized<4><<<num_blocks, threads_per_block>>>(
        param, grad, m, v, numel,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, is_adamw
    );
}

} // namespace cuda
} // namespace OwnTensor
