#pragma once

#include "dtype/Types.h"
#include <utility>

namespace OwnTensor {

// ==================================================================================
// FP4 (Unpacked) Conversions
// ==================================================================================

// FP4 -> FP32
__device__ __host__ inline float fp4_to_fp32(float4_e2m1_t val) {
    return static_cast<float>(val);
}

// FP4 -> FP64
__device__ __host__ inline double fp4_to_fp64(float4_e2m1_t val) {
    return static_cast<double>(static_cast<float>(val));
}

// FP4 -> FP16
__device__ __host__ inline float16_t fp4_to_fp16(float4_e2m1_t val) {
    return float16_t(static_cast<float>(val));
}

// FP4 -> BF16
__device__ __host__ inline bfloat16_t fp4_to_bf16(float4_e2m1_t val) {
    return bfloat16_t(static_cast<float>(val));
}

// FP32 -> FP4
__device__ __host__ inline float4_e2m1_t fp32_to_fp4(float val) {
    return float4_e2m1_t(val);
}

// FP64 -> FP4
__device__ __host__ inline float4_e2m1_t fp64_to_fp4(double val) {
    return float4_e2m1_t(val);
}

// FP16 -> FP4
__device__ __host__ inline float4_e2m1_t fp16_to_fp4(float16_t val) {
    return float4_e2m1_t(static_cast<float>(val));
}

// BF16 -> FP4
__device__ __host__ inline float4_e2m1_t bf16_to_fp4(bfloat16_t val) {
    return float4_e2m1_t(static_cast<float>(val));
}

// FP4 -> FP4 (Identity)
__device__ __host__ inline float4_e2m1_t fp4_to_fp4(float4_e2m1_t val) {
    return val;
}

// ==================================================================================
// FP4 (Packed) Conversions
// ==================================================================================

// Packed FP4 -> FP32 (Writes 2 values)
__device__ __host__ inline void packed_fp4_to_fp32(float4_e2m1_2x_t val, float* out) {
    out[0] = static_cast<float>(val.get_low());
    out[1] = static_cast<float>(val.get_high());
}

// Packed FP4 -> FP16 (Writes 2 values)
__device__ __host__ inline void packed_fp4_to_fp16(float4_e2m1_2x_t val, float16_t* out) {
    out[0] = float16_t(static_cast<float>(val.get_low()));
    out[1] = float16_t(static_cast<float>(val.get_high()));
}

// Packed FP4 -> FP4 (Unpacks to 2 values)
__device__ __host__ inline void packed_fp4_to_fp4(float4_e2m1_2x_t val, float4_e2m1_t* out) {
    out[0] = val.get_low();
    out[1] = val.get_high();
}

} // namespace OwnTensor
