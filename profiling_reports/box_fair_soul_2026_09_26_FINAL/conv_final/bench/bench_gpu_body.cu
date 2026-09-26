// bench_gpu_body.cu -- compiled TWICE with the library's GPU flags:
//   OLD: -DOwnTensor=OwnTensorOld -DBENCH_NS=gbench_old -I<original include>
//   NEW:                          -DBENCH_NS=gbench_new -I<patched include>
// Each copy has its own kernels (in its own namespace), built from its own Types.h.
#include "dtype/Types.h"
#include "bench_gpu_api.h"

namespace BENCH_NS {
using namespace ::OwnTensor;   // becomes ::OwnTensorOld in the old copy

uint16_t probe_f2bf(float f) { return bfloat16_t(f).raw_bits; }

// keeps the compiler from folding or hoisting the round trips
#define KEEP_F(x) asm volatile("" : "+f"(x))
#define KEEP_D(x) asm volatile("" : "+d"(x))
#define KEEP_I(x) asm volatile("" : "+r"(x))
#define GRID_LOOP for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)

__global__ void k_f32_bf16(const float* a, uint16_t* o, size_t n)   { GRID_LOOP o[i] = bfloat16_t(a[i]).raw_bits; }
__global__ void k_bf16_f32(const uint16_t* a, float* o, size_t n)   { GRID_LOOP { bfloat16_t x; x.raw_bits = a[i]; o[i] = static_cast<float>(x); } }
__global__ void k_f32_f16(const float* a, uint16_t* o, size_t n)    { GRID_LOOP o[i] = float16_t(a[i]).raw_bits; }
__global__ void k_f16_f32(const uint16_t* a, float* o, size_t n)    { GRID_LOOP { float16_t x; x.raw_bits = a[i]; o[i] = static_cast<float>(x); } }
__global__ void k_f64_bf16(const double* a, uint16_t* o, size_t n)  { GRID_LOOP o[i] = bfloat16_t(a[i]).raw_bits; }
__global__ void k_f64_f16(const double* a, uint16_t* o, size_t n)   { GRID_LOOP o[i] = float16_t(a[i]).raw_bits; }
__global__ void k_i32_bf16(const int32_t* a, uint16_t* o, size_t n) { GRID_LOOP o[i] = bfloat16_t(a[i]).raw_bits; }
__global__ void k_i32_f16(const int32_t* a, uint16_t* o, size_t n)  { GRID_LOOP o[i] = float16_t(a[i]).raw_bits; }
__global__ void k_i64_bf16(const int64_t* a, uint16_t* o, size_t n) { GRID_LOOP o[i] = bfloat16_t(a[i]).raw_bits; }
__global__ void k_bf16_mul(const uint16_t* a, uint16_t* o, size_t n) {
    GRID_LOOP { bfloat16_t x, y; x.raw_bits = a[2 * i]; y.raw_bits = a[2 * i + 1]; o[i] = (x * y).raw_bits; }
}
__global__ void k_f16_mul(const uint16_t* a, uint16_t* o, size_t n) {
    GRID_LOOP { float16_t x, y; x.raw_bits = a[2 * i]; y.raw_bits = a[2 * i + 1]; o[i] = (x * y).raw_bits; }
}
__global__ void k_bf16_to_f16(const uint16_t* a, uint16_t* o, size_t n) {
    GRID_LOOP { bfloat16_t x; x.raw_bits = a[i]; o[i] = float16_t(static_cast<float>(x)).raw_bits; }
}
__global__ void k_rt_bf16(const float* a, uint16_t* o, size_t n) {
    GRID_LOOP { float x = a[i]; bfloat16_t b(x);
        for (int k = 0; k < RT_K; ++k) { x = static_cast<float>(b); KEEP_F(x); b = bfloat16_t(x); }
        o[i] = b.raw_bits; }
}
__global__ void k_rt_f16(const float* a, uint16_t* o, size_t n) {
    GRID_LOOP { float x = a[i]; float16_t h(x);
        for (int k = 0; k < RT_K; ++k) { x = static_cast<float>(h); KEEP_F(x); h = float16_t(x); }
        o[i] = h.raw_bits; }
}
__global__ void k_rt_f64_bf16(const double* a, uint16_t* o, size_t n) {
    GRID_LOOP { double d = a[i]; bfloat16_t b(d);
        for (int k = 0; k < RT_K; ++k) { d = static_cast<double>(static_cast<float>(b)); KEEP_D(d); b = bfloat16_t(d); }
        o[i] = b.raw_bits; }
}
__global__ void k_rt_i32_bf16(const int32_t* a, uint16_t* o, size_t n) {
    GRID_LOOP { int v = a[i]; bfloat16_t b(v);
        for (int k = 0; k < RT_K; ++k) { v = static_cast<int>(static_cast<float>(b)); KEEP_I(v); b = bfloat16_t(v); }
        o[i] = b.raw_bits; }
}
__global__ void k_chain_bf16(const uint16_t* a, uint16_t* o, size_t n) {   // a = triples (x, b, c)
    GRID_LOOP { bfloat16_t x, b, c; x.raw_bits = a[3 * i]; b.raw_bits = a[3 * i + 1]; c.raw_bits = a[3 * i + 2];
        for (int k = 0; k < RT_K; ++k) { x = x * b + c; unsigned short r = x.raw_bits; asm volatile("" : "+h"(r)); x.raw_bits = r; }
        o[i] = x.raw_bits; }
}

float run(int fn, const void* in, void* out, size_t n, int reps, cudaStream_t s) {
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads < 132 * 32 ? (n + threads - 1) / threads : 132 * 32);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int r = 0; r < reps; ++r) {
        switch (fn) {
        case G_F32_BF16:    k_f32_bf16<<<blocks, threads, 0, s>>>((const float*)in, (uint16_t*)out, n); break;
        case G_BF16_F32:    k_bf16_f32<<<blocks, threads, 0, s>>>((const uint16_t*)in, (float*)out, n); break;
        case G_F32_F16:     k_f32_f16<<<blocks, threads, 0, s>>>((const float*)in, (uint16_t*)out, n); break;
        case G_F16_F32:     k_f16_f32<<<blocks, threads, 0, s>>>((const uint16_t*)in, (float*)out, n); break;
        case G_F64_BF16:    k_f64_bf16<<<blocks, threads, 0, s>>>((const double*)in, (uint16_t*)out, n); break;
        case G_F64_F16:     k_f64_f16<<<blocks, threads, 0, s>>>((const double*)in, (uint16_t*)out, n); break;
        case G_I32_BF16:    k_i32_bf16<<<blocks, threads, 0, s>>>((const int32_t*)in, (uint16_t*)out, n); break;
        case G_I32_F16:     k_i32_f16<<<blocks, threads, 0, s>>>((const int32_t*)in, (uint16_t*)out, n); break;
        case G_I64_BF16:    k_i64_bf16<<<blocks, threads, 0, s>>>((const int64_t*)in, (uint16_t*)out, n); break;
        case G_BF16_MUL:    k_bf16_mul<<<blocks, threads, 0, s>>>((const uint16_t*)in, (uint16_t*)out, n); break;
        case G_F16_MUL:     k_f16_mul<<<blocks, threads, 0, s>>>((const uint16_t*)in, (uint16_t*)out, n); break;
        case G_BF16_TO_F16: k_bf16_to_f16<<<blocks, threads, 0, s>>>((const uint16_t*)in, (uint16_t*)out, n); break;
        case G_RT_BF16:     k_rt_bf16<<<blocks, threads, 0, s>>>((const float*)in, (uint16_t*)out, n); break;
        case G_RT_F16:      k_rt_f16<<<blocks, threads, 0, s>>>((const float*)in, (uint16_t*)out, n); break;
        case G_RT_F64_BF16: k_rt_f64_bf16<<<blocks, threads, 0, s>>>((const double*)in, (uint16_t*)out, n); break;
        case G_RT_I32_BF16: k_rt_i32_bf16<<<blocks, threads, 0, s>>>((const int32_t*)in, (uint16_t*)out, n); break;
        case G_CHAIN_BF16:  k_chain_bf16<<<blocks, threads, 0, s>>>((const uint16_t*)in, (uint16_t*)out, n); break;
        default: break;
        }
    }
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}
}  // namespace BENCH_NS
