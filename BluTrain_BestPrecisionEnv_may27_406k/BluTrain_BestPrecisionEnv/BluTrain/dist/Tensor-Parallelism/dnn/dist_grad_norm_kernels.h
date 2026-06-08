#pragma once

#ifdef __CUDACC__
#define KERNEL_ARGS(...) <<< __VA_ARGS__ >>>
#else
#define KERNEL_ARGS(...)
#endif

// This allows the .cpp file to see the kernels without getting confused by CUDA syntax
#ifdef __cplusplus
extern "C" {
#endif


void launch_grad_norm_sq(const float* grad, float* buffer_slot, int n, cudaStream_t stream = 0);

void launch_grad_norm_inf(const float* grad, float* buffer_slot, int n, cudaStream_t stream = 0);

void launch_buffer_reduce(float* d_buffer, float* d_result, int n, bool is_inf_norm, cudaStream_t stream = 0);

void launch_apply_clip(float* grad, const float* d_global_stat, float max_norm, int n, bool is_inf_norm, cudaStream_t stream = 0);


#ifdef __cplusplus
}
#endif
