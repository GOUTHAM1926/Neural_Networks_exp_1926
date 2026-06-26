#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dist_grad_norm_kernels.h"

// 1. HELPERS FIRST (Must be at the top)
__device__ static float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// 2. KERNELS SECOND
__global__ void grad_norm_sq_into_buffer(const float* grad, float* buffer_slot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0;
    while (i < n) {
        local_sum += grad[i] * grad[i];
        i += blockDim.x * gridDim.x;
    }
    atomicAdd(buffer_slot, local_sum);
}

__global__ void grad_norm_inf_into_buffer(const float* grad, float* buffer_slot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = 0;
    while (i < n) {
        local_max = fmaxf(local_max, fabsf(grad[i]));
        i += blockDim.x * gridDim.x;
    }
    atomicMaxFloat(buffer_slot, local_max);
}

__global__ void buffer_reduce_kernel(float* d_buffer, float* d_result, int n, bool is_inf_norm) {
    int i = threadIdx.x;
    float res = is_inf_norm ? 0.0f : 0.0f;
    for (int j = i; j < n; j += blockDim.x) {
        if (is_inf_norm) res = fmaxf(res, d_buffer[j]);
        else res += d_buffer[j];
    }
    if (is_inf_norm) atomicMaxFloat(d_result, res);
    else atomicAdd(d_result, res);
}

__global__ void apply_clip_nccl_kernel(float* grad, const float* d_global_stat, float max_norm, int n, bool is_inf_norm) {
    float global_val = *d_global_stat;
    float global_norm = is_inf_norm ? global_val : sqrtf(global_val);
    if (global_norm <= max_norm) return;
    float clip_coef = max_norm / (global_norm + 1e-6f);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= clip_coef;
}

// 3. LAUNCH WRAPPERS LAST
void launch_grad_norm_sq(const float* grad, float* buffer_slot, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    grad_norm_sq_into_buffer<<<blocks, threads, 0, stream>>>(grad, buffer_slot, n);
}

void launch_grad_norm_inf(const float* grad, float* buffer_slot, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    grad_norm_inf_into_buffer<<<blocks, threads, 0, stream>>>(grad, buffer_slot, n);
}

void launch_buffer_reduce(float* d_buffer, float* d_result, int n, bool is_inf_norm, cudaStream_t stream) {
    buffer_reduce_kernel<<<1, 256, 0, stream>>>(d_buffer, d_result, n, is_inf_norm);
}

void launch_apply_clip(float* grad, const float* d_global_stat, float max_norm, int n, bool is_inf_norm, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    apply_clip_nccl_kernel<<<blocks, threads, 0, stream>>>(grad, d_global_stat, max_norm, n, is_inf_norm);
}
