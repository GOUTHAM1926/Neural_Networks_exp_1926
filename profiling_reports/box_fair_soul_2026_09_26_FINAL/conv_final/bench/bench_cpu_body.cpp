// bench_cpu_body.cpp -- compiled TWICE with the library's CPU flags:
//   OLD: -DOwnTensor=OwnTensorOld -DBENCH_NS=bench_old -I<original include>
//   NEW:                          -DBENCH_NS=bench_new -I<patched include>
// Renaming the namespace keeps the two copies of every inline function apart, so one
// program holds both versions and the main can time them back to back.
#include "dtype/Types.h"
#include "ops/helpers/Vectorized.h"
#include "bench_cpu_api.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>

namespace BENCH_NS {
using namespace ::OwnTensor;   // becomes ::OwnTensorOld in the old copy

uint16_t probe_f2bf(float f) { return detail::float_to_bfloat16(f); }
uint16_t probe_f2h(float f) { return detail::float_to_float16(f); }

#define LOOP(...) _Pragma("omp parallel for num_threads(threads) schedule(static)") \
    for (long long i = 0; i < (long long)n; ++i) { __VA_ARGS__; }

void run(int fn, const void* in, void* out, size_t n, int threads) {
    switch (fn) {
    case F_F32_BF16: { auto a = (const float*)in; auto o = (uint16_t*)out; LOOP(o[i] = detail::float_to_bfloat16(a[i])) } break;
    case F_BF16_F32: { auto a = (const uint16_t*)in; auto o = (float*)out; LOOP(o[i] = detail::bfloat16_to_float(a[i])) } break;
    case F_F32_F16:  { auto a = (const float*)in; auto o = (uint16_t*)out; LOOP(o[i] = detail::float_to_float16(a[i])) } break;
    case F_F16_F32:  { auto a = (const uint16_t*)in; auto o = (float*)out; LOOP(o[i] = detail::float16_to_float(a[i])) } break;
    case F_F64_BF16: { auto a = (const double*)in; auto o = (uint16_t*)out; LOOP(o[i] = bfloat16_t(a[i]).raw_bits) } break;
    case F_F64_F16:  { auto a = (const double*)in; auto o = (uint16_t*)out; LOOP(o[i] = float16_t(a[i]).raw_bits) } break;
    case F_I32_BF16: { auto a = (const int32_t*)in; auto o = (uint16_t*)out; LOOP(o[i] = bfloat16_t(a[i]).raw_bits) } break;
    case F_I32_F16:  { auto a = (const int32_t*)in; auto o = (uint16_t*)out; LOOP(o[i] = float16_t(a[i]).raw_bits) } break;
    case F_I64_BF16: { auto a = (const int64_t*)in; auto o = (uint16_t*)out; LOOP(o[i] = bfloat16_t(a[i]).raw_bits) } break;
    case F_I8_BF16:  { auto a = (const int8_t*)in; auto o = (uint16_t*)out; LOOP(o[i] = bfloat16_t(a[i]).raw_bits) } break;
    case F_BF16_MUL: {       // in = pairs (a, b) of bf16 bits
        auto a = (const uint16_t*)in; auto o = (uint16_t*)out;
        LOOP(bfloat16_t x; x.raw_bits = a[2 * i]; bfloat16_t y; y.raw_bits = a[2 * i + 1]; o[i] = (x * y).raw_bits)
    } break;
    case F_F16_MUL: {
        auto a = (const uint16_t*)in; auto o = (uint16_t*)out;
        LOOP(float16_t x; x.raw_bits = a[2 * i]; float16_t y; y.raw_bits = a[2 * i + 1]; o[i] = (x * y).raw_bits)
    } break;
    case F_BF16_TO_F16: {
        auto a = (const uint16_t*)in; auto o = (uint16_t*)out;
        LOOP(bfloat16_t x; x.raw_bits = a[i]; o[i] = float16_t(static_cast<float>(x)).raw_bits)
    } break;
    case F_VEC_F32_BF16: {   // n is a multiple of 8
        auto a = (const float*)in; auto o = (uint16_t*)out;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (long long i = 0; i < (long long)n; i += 8)
            vec::store_float_as_bf16(o + i, vec::Vectorized<float>(_mm256_loadu_ps(a + i)));
    } break;
    default: break;
    }
}
#undef LOOP
}  // namespace BENCH_NS
