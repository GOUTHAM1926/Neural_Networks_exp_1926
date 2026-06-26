#include "ops/helpers/GradNormKernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
namespace OwnTensor {
namespace cuda {

// =============================================================================
// FUSED GRADIENT NORM KERNEL
// Computes sum of squares and atomically adds to accumulator
// =============================================================================

__global__ void grad_norm_squared_kernel(
    const float* __restrict__ grad,
    float* __restrict__ norm_sq_accumulator,
    int64_t numel
) {
    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Each thread computes sum of squares for its elements
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = grad[i];
        thread_sum += val * val;
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // First thread atomically adds block result to global accumulator
    if (tid == 0) {
        atomicAdd(norm_sq_accumulator, sdata[0]);
    }
}

void grad_norm_squared_cuda(
    const float* grad,
    float* norm_sq_accumulator,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)1024);
    size_t smem_size = threads * sizeof(float);
    
    grad_norm_squared_kernel<<<blocks, threads, smem_size>>>(
        grad, norm_sq_accumulator, numel
    );
}

// =============================================================================
// INF NORM KERNEL (MAX ABS)
// =============================================================================

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(old);
        float max_val = fmaxf(val, old_val);
        if (max_val == old_val) break; // Optimization: if val is smaller, no need to swap
        old = atomicCAS(address_as_int, assumed, __float_as_int(max_val));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void grad_norm_inf_kernel(
    const float* __restrict__ grad,
    float* __restrict__ norm_inf_accumulator,
    int64_t numel
) {
    extern __shared__ float sdata[];
    
    int64_t tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    float thread_max = 0.0f;
    #pragma unroll 4
    for (int64_t i = idx; i < numel; i += stride) {
        float val = fabsf(grad[i]);
        if (val > thread_max) thread_max = val;
    }
    
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMaxFloat(norm_inf_accumulator, sdata[0]);
    }
}

void grad_norm_inf_cuda(
    const float* grad,
    float* norm_inf_accumulator,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel + threads - 1) / threads, (int64_t)1024);
    size_t smem_size = threads * sizeof(float);
    
    grad_norm_inf_kernel<<<blocks, threads, smem_size>>>(
        grad, norm_inf_accumulator, numel
    );
}

// =============================================================================
// SCALE GRADIENTS KERNEL
// =============================================================================

__global__ void scale_gradients_kernel(
    float* __restrict__ grad,
    float scale,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Vectorized path using float4
    int64_t numel4 = numel / 4;
    #pragma unroll 4
    for (int64_t i = idx; i < numel4; i += stride) {
        float4* ptr = reinterpret_cast<float4*>(grad) + i;
        float4 val = *ptr;
        val.x *= scale;
        val.y *= scale;
        val.z *= scale;
        val.w *= scale;
        *ptr = val;
    }
    
    // Handle remaining elements
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        grad[i] *= scale;
    }
}

void scale_gradients_cuda(
    float* grad,
    float scale,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
    if (blocks == 0) blocks = 1;
    
    scale_gradients_kernel<<<blocks, threads>>>(grad, scale, numel);
}

// =============================================================================
// COMPUTE CLIP COEFFICIENT ON GPU
// =============================================================================

__global__ void compute_clip_coef_kernel(
    float* __restrict__ norm_sq_or_inf,
    float* __restrict__ clip_coef_out,
    float max_norm,
    bool is_inf_norm
) {
    float norm_val = *norm_sq_or_inf;
    float total_norm = is_inf_norm ? norm_val : sqrtf(norm_val);
    float clip_coef = max_norm / (total_norm + 1e-6f);
    // Clamp to max 1.0 (don't scale up)
    *clip_coef_out = (clip_coef < 1.0f) ? clip_coef : 1.0f;
    // Also store the total_norm back for potential return value
    *norm_sq_or_inf = total_norm;
}

void compute_clip_coef_cuda(
    float* norm_sq_or_inf,
    float* clip_coef_out,
    float max_norm,
    bool is_inf_norm
) {
    compute_clip_coef_kernel<<<1, 1>>>(norm_sq_or_inf, clip_coef_out, max_norm, is_inf_norm);
}

// =============================================================================
// SCALE GRADIENTS WITH GPU COEFFICIENT
// =============================================================================

__global__ void scale_gradients_with_gpu_coef_kernel(
    float* __restrict__ grad,
    const float* __restrict__ clip_coef,
    int64_t numel
) {
    float scale = *clip_coef;
    if (scale >= 1.0f) return;  // No scaling needed
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    // Vectorized path using float4
    int64_t numel4 = numel / 4;
    #pragma unroll 4
    for (int64_t i = idx; i < numel4; i += stride) {
        float4* ptr = reinterpret_cast<float4*>(grad) + i;
        float4 val = *ptr;
        val.x *= scale;
        val.y *= scale;
        val.z *= scale;
        val.w *= scale;
        *ptr = val;
    }
    
    // Handle remaining elements
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        grad[i] *= scale;
    }
}

void scale_gradients_with_gpu_coef_cuda(
    float* grad,
    const float* clip_coef,
    int64_t numel
) {
    int threads = 256;
    int blocks = std::min((numel / 4 + threads - 1) / threads, (int64_t)65535);
    if (blocks == 0) blocks = 1;
    
    scale_gradients_with_gpu_coef_kernel<<<blocks, threads>>>(grad, clip_coef, numel);
}

} // namespace cuda
} // namespace OwnTensor
