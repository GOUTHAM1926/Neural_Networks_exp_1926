//======================================================================
// ZeroKernels.cu — see ZeroKernels.h.
//======================================================================

#include "ZeroKernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace owt {
namespace zero {
namespace kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxCastBlocks = 4096;   // grid-stride covers the rest

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }

// The tensor library is built with --use_fast_math, so its `x *= s` is a
// flush-to-zero multiply. Spell that out rather than inherit it from whatever
// flags this file happens to be compiled with.
__device__ __forceinline__ float mul_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

template <typename T> __host__ __device__ constexpr int vec_elems() {
    return 16 / static_cast<int>(sizeof(T));
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// ---------------------------------------------------------------------------
// cast + scale
// ---------------------------------------------------------------------------
template <typename T, bool SCALED>
__global__ void cast_scaled_vec_kernel(const T* __restrict__ src, float* __restrict__ dst,
                                       int64_t nvec, float scale) {
    constexpr int V = vec_elems<T>();
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nvec;
         i += stride) {
        const uint4 raw = reinterpret_cast<const uint4*>(src)[i];
        const T* e = reinterpret_cast<const T*>(&raw);
        float o[V];
#pragma unroll
        for (int k = 0; k < V; ++k) o[k] = SCALED ? mul_ftz(to_f32(e[k]), scale) : to_f32(e[k]);
        float4* d = reinterpret_cast<float4*>(dst + i * V);
#pragma unroll
        for (int k = 0; k < V / 4; ++k) d[k] = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
    }
}

template <typename T, bool SCALED>
__global__ void cast_scaled_scalar_kernel(const T* __restrict__ src, float* __restrict__ dst,
                                          int64_t n, float scale) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const float v = to_f32(src[i]);
        dst[i] = SCALED ? mul_ftz(v, scale) : v;
    }
}

inline int blocks_for(int64_t work, int cap) {
    const int64_t b = (work + kThreads - 1) / kThreads;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(b, cap)));
}

template <typename T, bool SCALED>
void cast_scaled_typed(const void* src_v, float* dst, int64_t n, float scale, cudaStream_t s) {
    const T* src = static_cast<const T*>(src_v);
    constexpr int V = vec_elems<T>();
    int64_t done = 0;
    if (aligned16(src) && aligned16(dst)) {
        const int64_t nvec = n / V;
        if (nvec > 0) {
            cast_scaled_vec_kernel<T, SCALED><<<blocks_for(nvec, kMaxCastBlocks), kThreads, 0, s>>>(
                src, dst, nvec, scale);
        }
        done = nvec * V;
    }
    const int64_t rem = n - done;
    if (rem > 0) {
        cast_scaled_scalar_kernel<T, SCALED><<<blocks_for(rem, kMaxCastBlocks), kThreads, 0, s>>>(
            src + done, dst + done, rem, scale);
    }
}

template <typename T>
void cast_scaled_dispatch(const void* src, float* dst, int64_t n, float scale, cudaStream_t s) {
    if (scale == 1.0f) cast_scaled_typed<T, false>(src, dst, n, scale, s);
    else               cast_scaled_typed<T, true >(src, dst, n, scale, s);
}

// ---------------------------------------------------------------------------
// sum of squares
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_part[kThreads / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_part[warp] = v;
    __syncthreads();
    v = (threadIdx.x < kThreads / 32) ? warp_part[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    }
    return v;   // valid in thread 0
}

// Work is split into LOGICAL chunks of V elements counted from src, so which
// thread adds which element depends on n alone -- never on the address. The
// same tensor therefore sums to the same bits wherever it sits in memory (the
// RS/AG and reduce/broadcast layouts place it differently). An aligned full
// chunk is read with one 16-byte load, anything else element by element; the
// arithmetic and its order are the same either way.
template <typename T>
__global__ void sumsq_partial_kernel(const T* __restrict__ src, int64_t n, bool vec_ok,
                                     float scale, float* __restrict__ partials) {
    constexpr int V = vec_elems<T>();
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t nchunks = (n + V - 1) / V;
    const int64_t full    = n / V;
    const bool scaled = (scale != 1.0f);
    float acc = 0.0f;
    for (int64_t j = tid; j < nchunks; j += stride) {
        if (vec_ok && j < full) {
            const uint4 raw = reinterpret_cast<const uint4*>(src)[j];
            const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
            for (int k = 0; k < V; ++k) {
                const float x = to_f32(e[k]);
                const float v = scaled ? mul_ftz(x, scale) : x;
                acc = __fmaf_rn(v, v, acc);
            }
        } else {
            const int64_t base = j * V;
#pragma unroll
            for (int k = 0; k < V; ++k) {
                if (base + k < n) {
                    const float x = to_f32(src[base + k]);
                    const float v = scaled ? mul_ftz(x, scale) : x;
                    acc = __fmaf_rn(v, v, acc);
                }
            }
        }
    }
    const float b = block_sum(acc);
    if (threadIdx.x == 0) partials[blockIdx.x] = b;
}

__global__ void sumsq_finalize_kernel(const float* __restrict__ partials, int nparts,
                                      float* __restrict__ acc_out) {
    float v = 0.0f;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) v += partials[i];
    const float b = block_sum(v);
    if (threadIdx.x == 0) *acc_out += b;
}

template <typename T>
void sumsq_typed(const void* src_v, int64_t n, float scale, float* d_acc, float* d_partials,
                 cudaStream_t s) {
    const T* src = static_cast<const T*>(src_v);
    constexpr int V = vec_elems<T>();
    const int64_t nchunks = (n + V - 1) / V;
    const int blocks = blocks_for(nchunks, kSumsqPartials);   // a function of n only
    sumsq_partial_kernel<T><<<blocks, kThreads, 0, s>>>(src, n, aligned16(src), scale, d_partials);
    sumsq_finalize_kernel<<<1, kThreads, 0, s>>>(d_partials, blocks, d_acc);
}

} // namespace

void cast_to_f32_scaled(const void* src, ElemKind kind, float* dst, int64_t n, float scale,
                        cudaStream_t stream) {
    if (n <= 0) return;
    switch (kind) {
        case ElemKind::F32:  cast_scaled_dispatch<float>(src, dst, n, scale, stream);         break;
        case ElemKind::BF16: cast_scaled_dispatch<__nv_bfloat16>(src, dst, n, scale, stream); break;
        case ElemKind::F16:  cast_scaled_dispatch<__half>(src, dst, n, scale, stream);        break;
        default: break;   // callers check for UNSUPPORTED before calling
    }
}

void sumsq_accumulate(const void* src, ElemKind kind, int64_t n, float scale, float* d_acc,
                      float* d_partials, cudaStream_t stream) {
    if (n <= 0) return;
    switch (kind) {
        case ElemKind::F32:  sumsq_typed<float>(src, n, scale, d_acc, d_partials, stream);         break;
        case ElemKind::BF16: sumsq_typed<__nv_bfloat16>(src, n, scale, d_acc, d_partials, stream); break;
        case ElemKind::F16:  sumsq_typed<__half>(src, n, scale, d_acc, d_partials, stream);        break;
        default: break;
    }
}

} // namespace kernels
} // namespace zero
} // namespace owt
