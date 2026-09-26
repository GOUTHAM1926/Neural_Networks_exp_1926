// bench_cpu_api.h -- shared by the CPU micro-benchmark main and its two bodies (old / new code).
#pragma once
#include <cstddef>
#include <cstdint>

// One entry per conversion path in include/dtype/Types.h (and the AVX2 store in Vectorized.h).
enum BenchFn : int {
    F_F32_BF16,      // detail::float_to_bfloat16                 (Bug D clamp in the old code)
    F_BF16_F32,      // detail::bfloat16_to_float                 (3 branches -> 1 shift)
    F_F32_F16,       // detail::float_to_float16                  (rewritten)
    F_F16_F32,       // detail::float16_to_float                  (unchanged on the CPU)
    F_F64_BF16,      // bfloat16_t(double)                        (old: through float, two roundings)
    F_F64_F16,       // float16_t(double)
    F_I32_BF16,      // bfloat16_t(int32_t)
    F_I32_F16,       // float16_t(int32_t)
    F_I64_BF16,      // bfloat16_t(int64_t)                       (new: round to odd for |v| >= 2^53)
    F_I8_BF16,       // bfloat16_t(int8_t)
    F_BF16_MUL,      // bfloat16_t * bfloat16_t                   (float math + one fp32 -> bf16)
    F_F16_MUL,       // float16_t * float16_t
    F_BF16_TO_F16,   // float16_t(float(bfloat16_t))              (the library's bf16 -> fp16 path)
    F_VEC_F32_BF16,  // vec::store_float_as_bf16, 8 lanes (AVX2)  (old: truncation)
    F_COUNT
};

namespace bench_old {
void run(int fn, const void* in, void* out, size_t n, int threads);
uint16_t probe_f2bf(float f);   // fingerprint: which code this copy really contains
uint16_t probe_f2h(float f);
}
namespace bench_new {
void run(int fn, const void* in, void* out, size_t n, int threads);
uint16_t probe_f2bf(float f);
uint16_t probe_f2h(float f);
}
