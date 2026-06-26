#pragma once

#ifndef OWNTENSOR_VECTORIZED_H
#define OWNTENSOR_VECTORIZED_H

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace OwnTensor {
namespace vec {

// =================================================================
// Vectorized<T> — Minimal AVX2 SIMD wrapper for CPU reductions
//
// Wraps AVX2 256-bit intrinsics. Provides:
//   loadu/storeu, +, *, min, max, reduce_add, reduce_max, reduce_min
//
// Width per type (256-bit register):
//   float:  8 elements
//   double: 4 elements
//   int32:  8 elements
//   int64:  4 elements
// =================================================================

// Default: no specialization
template <typename T>
struct Vectorized {
    static constexpr int size() { return 1; }
};

// =================================================================
// Vectorized<float> — 8-wide AVX2
// =================================================================
template <>
struct Vectorized<float> {
    __m256 values;

    static constexpr int size() { return 8; }

    Vectorized() : values(_mm256_setzero_ps()) {}
    Vectorized(__m256 v) : values(v) {}
    Vectorized(float val) : values(_mm256_set1_ps(val)) {}

    static Vectorized loadu(const float* ptr) {
        return _mm256_loadu_ps(ptr);
    }
    static Vectorized loadu(const float* ptr, int count) {
        if (count == size()) return _mm256_loadu_ps(ptr);
        float tmp[8] = {0};
        std::memcpy(tmp, ptr, count * sizeof(float));
        return _mm256_loadu_ps(tmp);
    }
    void storeu(float* ptr) const {
        _mm256_storeu_ps(ptr, values);
    }
    void storeu(float* ptr, int count) const {
        if (count == size()) { _mm256_storeu_ps(ptr, values); return; }
        float tmp[8];
        _mm256_storeu_ps(tmp, values);
        std::memcpy(ptr, tmp, count * sizeof(float));
    }

    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_ps(values, other.values);
    }
    Vectorized operator*(const Vectorized& other) const {
        return _mm256_mul_ps(values, other.values);
    }
    Vectorized operator/(const Vectorized& other) const {
        return _mm256_div_ps(values, other.values);
    }
    Vectorized operator-(const Vectorized& other) const {
        return _mm256_sub_ps(values, other.values);
    }

    // Fused multiply-add: a * b + c
    static Vectorized fmadd(const Vectorized& a, const Vectorized& b, const Vectorized& c) {
        return _mm256_fmadd_ps(a.values, b.values, c.values);
    }

    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_ps(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_ps(a.values, b.values);
    }

    // Horizontal reduction: sum all 8 lanes → single float
    float reduce_add() const {
        // [a0 a1 a2 a3 | a4 a5 a6 a7]
        auto v = values;
        auto v1 = _mm256_permute2f128_ps(v, v, 0x1);   // swap 128-bit halves
        v = _mm256_add_ps(v, v1);                        // [a0+a4 a1+a5 a2+a6 a3+a7 | ...]
        v1 = _mm256_shuffle_ps(v, v, 0x4E);              // swap 64-bit pairs
        v = _mm256_add_ps(v, v1);                         // [a0+a2+a4+a6 a1+a3+a5+a7 | ...]
        v1 = _mm256_shuffle_ps(v, v, 0xB1);              // swap 32-bit pairs
        v = _mm256_add_ps(v, v1);                         // final sum in lane 0
        return _mm256_cvtss_f32(v);
    }
    float reduce_max() const {
        auto v = values;
        auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
        v = _mm256_max_ps(v, v1);
        v1 = _mm256_shuffle_ps(v, v, 0x4E);
        v = _mm256_max_ps(v, v1);
        v1 = _mm256_shuffle_ps(v, v, 0xB1);
        v = _mm256_max_ps(v, v1);
        return _mm256_cvtss_f32(v);
    }
    float reduce_min() const {
        auto v = values;
        auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
        v = _mm256_min_ps(v, v1);
        v1 = _mm256_shuffle_ps(v, v, 0x4E);
        v = _mm256_min_ps(v, v1);
        v1 = _mm256_shuffle_ps(v, v, 0xB1);
        v = _mm256_min_ps(v, v1);
        return _mm256_cvtss_f32(v);
    }

    // Bitwise AND (for boolean/mask operations)
    Vectorized operator&(const Vectorized& other) const {
        return _mm256_and_ps(values, other.values);
    }
    Vectorized operator|(const Vectorized& other) const {
        return _mm256_or_ps(values, other.values);
    }

    // Absolute value: clear sign bit
    Vectorized abs() const {
        const __m256 sign_mask = _mm256_set1_ps(-0.0f); // 0x80000000
        return _mm256_andnot_ps(sign_mask, values);
    }

    // NaN check: returns mask where NaN lanes are all-1s
    Vectorized isnan() const {
        return _mm256_cmp_ps(values, values, _CMP_UNORD_Q);
    }

    // Blend: select from b where mask is set, else from a
    static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
        return _mm256_blendv_ps(a.values, b.values, mask.values);
    }

    // ─────────────────────────────────────────────────────────────
    // Vectorized tanh() — Rational polynomial approximation (Pade)
    //
    // For |x| <= 3.97:  tanh(x) ≈ x * P(x²) / Q(x²)   (~20-bit accuracy)
    // For |x| > 3.97:   tanh(x) = ±1  (saturated)
    //
    // Coefficients from Cephes library (public domain):
    //   P(z) = -0.8237728127 + z*(-0.003831010665 + z*(-5.70498872e-6))
    //   Q(z) =  2.4713006234 + z*( 1.0            + z*( 0.0004946110 + z*0.0))
    //   tanh(x) = x + x * z * P(z) / Q(z)  where z = x²
    //
    // All done with FMA — zero branching in the hot path.
    // ─────────────────────────────────────────────────────────────
    Vectorized tanh() const {
        // Constants
        const __m256 one      = _mm256_set1_ps(1.0f);
        const __m256 neg_one  = _mm256_set1_ps(-1.0f);
        const __m256 clamp_hi = _mm256_set1_ps(3.97f);

        // Polynomial coefficients (Cephes single-precision)
        const __m256 p0 = _mm256_set1_ps(-8.237728127e-1f);
        const __m256 p1 = _mm256_set1_ps(-3.831010665e-3f);
        const __m256 p2 = _mm256_set1_ps(-5.70498872e-6f);

        const __m256 q0 = _mm256_set1_ps(2.4713006234e0f);
        const __m256 q1 = _mm256_set1_ps(1.0f);
        const __m256 q2 = _mm256_set1_ps(4.946110e-4f);

        __m256 x = values;

        // z = x * x
        __m256 z = _mm256_mul_ps(x, x);

        // Numerator: P(z) = p0 + z*(p1 + z*p2)  (Horner's method with FMA)
        __m256 num = _mm256_fmadd_ps(p2, z, p1);   // p2*z + p1
        num = _mm256_fmadd_ps(num, z, p0);          // (p2*z + p1)*z + p0

        // Denominator: Q(z) = q0 + z*(q1 + z*q2)
        __m256 den = _mm256_fmadd_ps(q2, z, q1);   // q2*z + q1
        den = _mm256_fmadd_ps(den, z, q0);          // (q2*z + q1)*z + q0

        // tanh(x) = x + x * z * P(z) / Q(z)
        __m256 result = _mm256_mul_ps(z, num);       // z * P(z)
        result = _mm256_div_ps(result, den);         // z * P(z) / Q(z)
        result = _mm256_fmadd_ps(result, x, x);     // x + x * z*P(z)/Q(z)

        // Clamp: for |x| > 3.97, return ±1
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
        __m256 saturated = _mm256_cmp_ps(abs_x, clamp_hi, _CMP_GT_OQ);
        __m256 sign = _mm256_and_ps(x, sign_mask);            // extract sign
        __m256 clamped = _mm256_or_ps(sign, one);             // ±1 with sign of x
        // But clamped should be sign(x)*1, which needs: if x<0 then -1 else 1
        clamped = _mm256_blendv_ps(one, neg_one, sign);       // -1 where x<0, +1 otherwise

        // Blend: use polynomial result where |x| <= 3.97, clamped where |x| > 3.97
        return _mm256_blendv_ps(result, clamped, saturated);
    }
};

// =================================================================
// Vectorized<double> — 4-wide AVX2
// =================================================================
template <>
struct Vectorized<double> {
    __m256d values;

    static constexpr int size() { return 4; }

    Vectorized() : values(_mm256_setzero_pd()) {}
    Vectorized(__m256d v) : values(v) {}
    Vectorized(double val) : values(_mm256_set1_pd(val)) {}

    static Vectorized loadu(const double* ptr) {
        return _mm256_loadu_pd(ptr);
    }
    static Vectorized loadu(const double* ptr, int count) {
        if (count == size()) return _mm256_loadu_pd(ptr);
        double tmp[4] = {0};
        std::memcpy(tmp, ptr, count * sizeof(double));
        return _mm256_loadu_pd(tmp);
    }
    void storeu(double* ptr) const {
        _mm256_storeu_pd(ptr, values);
    }
    void storeu(double* ptr, int count) const {
        if (count == size()) { _mm256_storeu_pd(ptr, values); return; }
        double tmp[4];
        _mm256_storeu_pd(tmp, values);
        std::memcpy(ptr, tmp, count * sizeof(double));
    }

    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_pd(values, other.values);
    }
    Vectorized operator*(const Vectorized& other) const {
        return _mm256_mul_pd(values, other.values);
    }
    Vectorized operator/(const Vectorized& other) const {
        return _mm256_div_pd(values, other.values);
    }
    Vectorized operator-(const Vectorized& other) const {
        return _mm256_sub_pd(values, other.values);
    }

    static Vectorized fmadd(const Vectorized& a, const Vectorized& b, const Vectorized& c) {
        return _mm256_fmadd_pd(a.values, b.values, c.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_pd(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_pd(a.values, b.values);
    }

    double reduce_add() const {
        auto v = values;
        auto v1 = _mm256_permute2f128_pd(v, v, 0x1);   // swap 128-bit halves
        v = _mm256_add_pd(v, v1);                        // [a0+a2, a1+a3, ...]
        v1 = _mm256_shuffle_pd(v, v, 0x5);               // swap 64-bit pairs
        v = _mm256_add_pd(v, v1);
        return _mm256_cvtsd_f64(v);
    }
    double reduce_max() const {
        auto v = values;
        auto v1 = _mm256_permute2f128_pd(v, v, 0x1);
        v = _mm256_max_pd(v, v1);
        v1 = _mm256_shuffle_pd(v, v, 0x5);
        v = _mm256_max_pd(v, v1);
        return _mm256_cvtsd_f64(v);
    }
    double reduce_min() const {
        auto v = values;
        auto v1 = _mm256_permute2f128_pd(v, v, 0x1);
        v = _mm256_min_pd(v, v1);
        v1 = _mm256_shuffle_pd(v, v, 0x5);
        v = _mm256_min_pd(v, v1);
        return _mm256_cvtsd_f64(v);
    }

    Vectorized isnan() const {
        return _mm256_cmp_pd(values, values, _CMP_UNORD_Q);
    }
    static Vectorized blendv(const Vectorized& a, const Vectorized& b, const Vectorized& mask) {
        return _mm256_blendv_pd(a.values, b.values, mask.values);
    }
};

// =================================================================
// Vectorized<int32_t> — 8-wide AVX2
// =================================================================
template <>
struct Vectorized<int32_t> {
    __m256i values;

    static constexpr int size() { return 8; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(int32_t val) : values(_mm256_set1_epi32(val)) {}

    static Vectorized loadu(const int32_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(int32_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }

    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi32(values, other.values);
    }
    Vectorized operator*(const Vectorized& other) const {
        return _mm256_mullo_epi32(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epi32(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epi32(a.values, b.values);
    }

    int32_t reduce_add() const {
        // Extract 128-bit halves and add
        __m128i lo = _mm256_castsi256_si128(values);
        __m128i hi = _mm256_extracti128_si256(values, 1);
        lo = _mm_add_epi32(lo, hi);
        // Horizontal add within 128-bit
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        return _mm_cvtsi128_si32(lo);
    }
};

// =================================================================
// Vectorized<int64_t> — 4-wide AVX2
// =================================================================
template <>
struct Vectorized<int64_t> {
    __m256i values;

    static constexpr int size() { return 4; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(int64_t val) : values(_mm256_set1_epi64x(val)) {}

    static Vectorized loadu(const int64_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(int64_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }

    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi64(values, other.values);
    }

    int64_t reduce_add() const {
        __m128i lo = _mm256_castsi256_si128(values);
        __m128i hi = _mm256_extracti128_si256(values, 1);
        lo = _mm_add_epi64(lo, hi);           // [a0+a2, a1+a3]
        __m128i hi64 = _mm_unpackhi_epi64(lo, lo);
        lo = _mm_add_epi64(lo, hi64);
        return _mm_cvtsi128_si64(lo);
    }
};

// =================================================================
// Vectorized<uint8_t> — 32-wide AVX2
// =================================================================
template <>
struct Vectorized<uint8_t> {
    __m256i values;
    static constexpr int size() { return 32; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(uint8_t val) : values(_mm256_set1_epi8(val)) {}

    static Vectorized loadu(const uint8_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(uint8_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi8(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epu8(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epu8(a.values, b.values);
    }
    // AND/OR for bool-like operations
    Vectorized operator&(const Vectorized& other) const {
        return _mm256_and_si256(values, other.values);
    }
    Vectorized operator|(const Vectorized& other) const {
        return _mm256_or_si256(values, other.values);
    }
};

// =================================================================
// Vectorized<int8_t> — 32-wide AVX2
// =================================================================
template <>
struct Vectorized<int8_t> {
    __m256i values;
    static constexpr int size() { return 32; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(int8_t val) : values(_mm256_set1_epi8(val)) {}

    static Vectorized loadu(const int8_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(int8_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi8(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epi8(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epi8(a.values, b.values);
    }
};

// =================================================================
// Vectorized<int16_t> — 16-wide AVX2
// =================================================================
template <>
struct Vectorized<int16_t> {
    __m256i values;
    static constexpr int size() { return 16; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(int16_t val) : values(_mm256_set1_epi16(val)) {}

    static Vectorized loadu(const int16_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(int16_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi16(values, other.values);
    }
    Vectorized operator*(const Vectorized& other) const {
        return _mm256_mullo_epi16(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epi16(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epi16(a.values, b.values);
    }
};

// =================================================================
// Vectorized<uint16_t> — 16-wide AVX2
// =================================================================
template <>
struct Vectorized<uint16_t> {
    __m256i values;
    static constexpr int size() { return 16; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(uint16_t val) : values(_mm256_set1_epi16(val)) {}

    static Vectorized loadu(const uint16_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(uint16_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi16(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epu16(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epu16(a.values, b.values);
    }
};

// =================================================================
// Vectorized<uint32_t> — 8-wide AVX2
// =================================================================
template <>
struct Vectorized<uint32_t> {
    __m256i values;
    static constexpr int size() { return 8; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(uint32_t val) : values(_mm256_set1_epi32(static_cast<int32_t>(val))) {}

    static Vectorized loadu(const uint32_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(uint32_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi32(values, other.values);
    }
    Vectorized operator*(const Vectorized& other) const {
        return _mm256_mullo_epi32(values, other.values);
    }
    static Vectorized min(const Vectorized& a, const Vectorized& b) {
        return _mm256_min_epu32(a.values, b.values);
    }
    static Vectorized max(const Vectorized& a, const Vectorized& b) {
        return _mm256_max_epu32(a.values, b.values);
    }
};

// =================================================================
// Vectorized<uint64_t> — 4-wide AVX2
// =================================================================
template <>
struct Vectorized<uint64_t> {
    __m256i values;
    static constexpr int size() { return 4; }

    Vectorized() : values(_mm256_setzero_si256()) {}
    Vectorized(__m256i v) : values(v) {}
    Vectorized(uint64_t val) : values(_mm256_set1_epi64x(static_cast<int64_t>(val))) {}

    static Vectorized loadu(const uint64_t* ptr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }
    void storeu(uint64_t* ptr) const {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    }
    Vectorized operator+(const Vectorized& other) const {
        return _mm256_add_epi64(values, other.values);
    }
    uint64_t reduce_add() const {
        __m128i lo = _mm256_castsi256_si128(values);
        __m128i hi = _mm256_extracti128_si256(values, 1);
        lo = _mm_add_epi64(lo, hi);
        __m128i hi64 = _mm_unpackhi_epi64(lo, lo);
        lo = _mm_add_epi64(lo, hi64);
        return static_cast<uint64_t>(_mm_cvtsi128_si64(lo));
    }
};

// =================================================================
// LOAD-CONVERT HELPERS for fp16/bf16
//
// These types have NO native AVX2 compute. Strategy:
//   Load fp16/bf16 → convert to float → compute in Vectorized<float>
//   Convert float → fp16/bf16 → store
//
// Uses F16C hardware instructions (_mm256_cvtph_ps / _mm256_cvtps_ph)
// for fp16, and bit-shift for bf16.
// =================================================================

// Load 8 x fp16 → 8 x float (F16C hardware instruction)
inline Vectorized<float> load_fp16_as_float(const void* ptr) {
    __m128i fp16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return Vectorized<float>(_mm256_cvtph_ps(fp16_vals));
}

// Store 8 x float → 8 x fp16 (F16C hardware instruction)
// _MM_FROUND_TO_NEAREST_INT = round to nearest even
inline void store_float_as_fp16(void* ptr, const Vectorized<float>& v) {
    __m128i fp16_vals = _mm256_cvtps_ph(v.values, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), fp16_vals);
}

// Load 8 x bf16 → 8 x float (bit-shift: bf16 is upper 16 bits of float)
inline Vectorized<float> load_bf16_as_float(const void* ptr) {
    // Load 8 x uint16 into lower 128 bits
    __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    // Zero-extend to 32-bit integers
    __m256i int32_vals = _mm256_cvtepu16_epi32(bf16_vals);
    // Shift left by 16 to get float bit pattern
    int32_vals = _mm256_slli_epi32(int32_vals, 16);
    // Reinterpret as float
    return Vectorized<float>(_mm256_castsi256_ps(int32_vals));
}

// Store 8 x float → 8 x bf16 (truncate lower 16 bits)
inline void store_float_as_bf16(void* ptr, const Vectorized<float>& v) {
    // Reinterpret float as int32
    __m256i int32_vals = _mm256_castps_si256(v.values);
    // Shift right by 16 to get bf16 bits
    int32_vals = _mm256_srli_epi32(int32_vals, 16);
    // Pack 32-bit to 16-bit (truncate)
    // Extract 128-bit halves, pack with signed saturation (values are positive small)
    __m128i lo = _mm256_castsi256_si128(int32_vals);
    __m128i hi = _mm256_extracti128_si256(int32_vals, 1);
    __m128i packed = _mm_packus_epi32(lo, hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
}

// =================================================================
// LOAD-CONVERT HELPERS for complex32_t (2 × fp16)
//
// complex32_t is stored as [real_fp16, imag_fp16] pairs.
// For addition: (a+bi)+(c+di) = (a+c)+(b+d)i — real and imag add
// independently, so we just treat them as interleaved fp16 values.
//
// 8 complex32_t = 16 fp16 values → need 2 × F16C converts
// Result: 2 × Vectorized<float> (8 floats each = 16 total)
// =================================================================

// Load 4 x complex32_t (= 8 fp16) → Vectorized<float> (8 interleaved floats)
// This processes 4 complex numbers at a time
inline Vectorized<float> load_complex32_as_float(const void* ptr) {
    // 4 complex32_t = 8 fp16 values = 16 bytes = 128 bits
    __m128i fp16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return Vectorized<float>(_mm256_cvtph_ps(fp16_vals));
}

// Store Vectorized<float> (8 interleaved floats) → 4 x complex32_t (= 8 fp16)
inline void store_float_as_complex32(void* ptr, const Vectorized<float>& v) {
    __m128i fp16_vals = _mm256_cvtps_ph(v.values, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), fp16_vals);
}

// =================================================================
// COMPLEX64/128 SIMD STRATEGY (no special helpers needed!)
//
// complex64_t  = [float real, float imag]  = 8 bytes
//   → 4 complex64_t  = 8 floats  → Vectorized<float> directly
//   → For sum: just add interleaved floats, reals add with reals,
//     imags add with imags because memory layout is consistent
//
// complex128_t = [double real, double imag] = 16 bytes
//   → 2 complex128_t = 4 doubles → Vectorized<double> directly
//
// No conversion needed! Just reinterpret_cast the pointer.
// =================================================================

// Load 4 x complex64_t as 8 interleaved floats
inline Vectorized<float> load_complex64_as_float(const void* ptr) {
    return Vectorized<float>::loadu(reinterpret_cast<const float*>(ptr));
}

// Store 8 interleaved floats as 4 x complex64_t
inline void store_float_as_complex64(void* ptr, const Vectorized<float>& v) {
    v.storeu(reinterpret_cast<float*>(ptr));
}

// Load 2 x complex128_t as 4 interleaved doubles
inline Vectorized<double> load_complex128_as_double(const void* ptr) {
    return Vectorized<double>::loadu(reinterpret_cast<const double*>(ptr));
}

// Store 4 interleaved doubles as 2 x complex128_t
inline void store_double_as_complex128(void* ptr, const Vectorized<double>& v) {
    v.storeu(reinterpret_cast<double*>(ptr));
}

} // namespace vec
} // namespace OwnTensor

#endif // OWNTENSOR_VECTORIZED_H
