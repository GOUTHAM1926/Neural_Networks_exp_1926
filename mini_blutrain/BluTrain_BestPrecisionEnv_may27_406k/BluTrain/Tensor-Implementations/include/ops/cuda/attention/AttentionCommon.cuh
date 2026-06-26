#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <stdio.h>

namespace OwnTensor {

static constexpr int MAX_HD = 256; 

// --- Reduction Primitives ---
static __inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

static __inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __inline__ __device__ float blockReduceMax(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceMax(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bmax = -INFINITY;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bmax = fmaxf(bmax, warp_vals[w]);
        warp_vals[0] = bmax;
    }
    __syncthreads();
    return warp_vals[0];
}

static __inline__ __device__ float blockReduceSum(float val) {
    __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    val = warpReduceSum(val);
    if (laneId == 0) warp_vals[warpId] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bsum = 0.0f;
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bsum += warp_vals[w];
        warp_vals[0] = bsum;
    }
    __syncthreads();
    return warp_vals[0];
}

// Strided params for memory-efficient attention kernels.
// Last-dim (HeadDim) stride is always 1, matching PyTorch's
// CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA requirement. Strides are in elements,
// not bytes. B is batch size, nh is number of heads (needed to split
// blockIdx.y = b*nh + h into (b, h)).
struct MemEfficientBwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    const float* __restrict__ O;        // saved attention output (for D = sum(dO * O))
    const float* __restrict__ dO;
    const float* __restrict__ LSE;
    float* __restrict__ D;   // precompute_D writes, main kernels read
    float* __restrict__ dQ;
    float* __restrict__ dK;
    float* __restrict__ dV;
    int B;
    int nh;
    int T;
    float scale;
    bool is_causal;
    // Tensor strides (elements). Shape is [B, nh, T, HeadDim] for Q/K/V/O/dO/dQ/dK/dV
    // and [B, nh, T] for LSE/D. Last-dim stride is implicitly 1.
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t do_strideB, do_strideM, do_strideH;
    int64_t dq_strideB, dq_strideM, dq_strideH;
    int64_t dk_strideB, dk_strideM, dk_strideH;
    int64_t dv_strideB, dv_strideM, dv_strideH;
    int64_t lse_strideB, lse_strideH;
    int64_t d_strideB,   d_strideH;
};

struct MemEfficientFwdParams {
    const float* __restrict__ Q;
    const float* __restrict__ K;
    const float* __restrict__ V;
    float* __restrict__ O;
    float* __restrict__ LSE;
    int B;
    int nh;
    int T;
    float scale;
    bool is_causal;
    float dropout_p;
    const float* __restrict__ dropout_mask;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t lse_strideB, lse_strideH;
};

} // namespace OwnTensor