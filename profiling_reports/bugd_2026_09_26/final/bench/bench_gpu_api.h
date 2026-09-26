// bench_gpu_api.h -- shared by the GPU micro-benchmark main and its two bodies (old / new code).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

enum GpuFn : int {
    // memory-shaped: one conversion per element (load, convert, store) -- like real ops
    G_F32_BF16, G_BF16_F32, G_F32_F16, G_F16_F32, G_F64_BF16, G_F64_F16,
    G_I32_BF16, G_I32_F16, G_I64_BF16, G_BF16_MUL, G_F16_MUL, G_BF16_TO_F16,
    // conversion-heavy: 64 dependent round trips per element, in registers
    G_RT_BF16,        // x = float(bfloat16_t(x))      x64
    G_RT_F16,         // x = float(float16_t(x))       x64
    G_RT_F64_BF16,    // d = double(float(bfloat16_t(d))) x64
    G_RT_I32_BF16,    // v = int(float(bfloat16_t(v)))  x64
    G_CHAIN_BF16,     // a = a * b + c (bfloat16_t operators) x64
    G_COUNT
};
constexpr int RT_K = 64;

namespace gbench_old {
float run(int fn, const void* d_in, void* d_out, size_t n, int reps, cudaStream_t s);   // total ms for reps launches
uint16_t probe_f2bf(float f);   // host-side fingerprint of this copy
}
namespace gbench_new {
float run(int fn, const void* d_in, void* d_out, size_t n, int reps, cudaStream_t s);
uint16_t probe_f2bf(float f);
}
