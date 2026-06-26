#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace OwnTensor {
namespace cuda {

// ---- Type conversion helpers ----
template<typename T> __device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float(float val) { return val; }
template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float(float val) { return val; }
template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

// ---- Constants ----
__device__ constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
__device__ constexpr float GELU_COEF      = 0.044715f;

// ---- Fast Math Helpers (Inline PTX) ----
__device__ __forceinline__ float fast_tanh(float x) {
    float res;
    asm("tanh.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

__device__ __forceinline__ float fast_exp(float x) {
    float res;
    float x_log2e = x * 1.44269504089f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(res) : "f"(x_log2e));
    return res;
}

__device__ __forceinline__ float fast_rcp(float x) {
    float res;
    asm("rcp.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return fast_rcp(1.0f + fast_exp(-x));
}

} // namespace cuda
} // namespace OwnTensor
