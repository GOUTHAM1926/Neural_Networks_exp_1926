#pragma once

#include <cstdint>
#include <limits>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <complex>
// ==================================================================================
// CUDA COMPATIBILITY MACROS
// ==================================================================================
#ifndef __CUDACC__
    // When NOT compiling with nvcc, these macros do nothing
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __host__
        #define __host__
    #endif
#endif
namespace OwnTensor {

//  ALWAYS define custom structs (both CPU and GPU compilation)
// These are the ONLY types we use in all code paths

// ==================================================================================
// LOW-LEVEL CONVERSION FUNCTIONS - NOW DEVICE-COMPATIBLE
// ==================================================================================

namespace detail {

// ---- BFloat16 Conversions (Google Brain Float) ----
// __device__ __host__ inline float bfloat16_to_float(uint16_t b) {
//     // BFloat16: Sign(1) + Exp(8) + Mantissa(7) -> Just shift left 16 bits
//     uint32_t u = static_cast<uint32_t>(b) << 16;
//     float f;
//     ::std::memcpy(&f, &u, sizeof(f));
//     return f;

__device__ __host__ inline float bfloat16_to_float(uint16_t b) {
    uint32_t sign = (b & 0x8000u) << 16;      // Sign bit to float position
    uint32_t exp = (b & 0x7F80u) >> 7;        // Extract exponent (8 bits)
    uint32_t frac = (b & 0x007Fu);             // Extract mantissa (7 bits)
    uint32_t u;

    if (exp == 0xFF) {
        // Inf or NaN
        if (frac == 0) {
            // Infinity
            u = sign | 0x7F800000u; // float infinity pattern
        } else {
            // NaN - set quiet NaN with some mantissa bits preserved
            u = sign | 0x7F800000u | (frac << 16);
        }
    } else {
        // Normal or subnormal number: shift bfloat16 bits to float position
        u = (static_cast<uint32_t>(b)) << 16;
    }

    float f;
    ::std::memcpy(&f, &u, sizeof(f));
    return f;
}


__device__ __host__ inline uint16_t float_to_bfloat16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = u & 0x80000000u;
    uint32_t exponent = (u >> 23) & 0xFF;
    uint32_t mantissa = u & 0x7FFFFFu;

    // Handle NaN and Infinity explicitly
    if (exponent == 0xFF) {
        if (mantissa == 0) {
            // Infinity
            return static_cast<uint16_t>((sign >> 16) | 0x7F80u);
        } else {
            // NaN: keep quiet NaN pattern
            return static_cast<uint16_t>((sign >> 16) | 0x7FC1u);
        }
    }

    // Overflow, clamp to Infinity
    if (exponent > 0x8E) {  // exponent > 142 decimal (127+15)
        return static_cast<uint16_t>((sign >> 16) | 0x7F80u);
    }

    // Normal rounding and truncation
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu;
    if (lsb) rounding_bias = 0x8000u;
    u += rounding_bias;

    return static_cast<uint16_t>(u >> 16);
}

//     uint32_t u;
//     ::std::memcpy(&u, &f, sizeof(u));
//     // Round-to-nearest-even (RNE) logic
//     uint32_t lsb = (u >> 16) & 1u;
//     uint32_t rounding_bias = 0x7FFFu + lsb;
//     u += rounding_bias;
//     return static_cast<uint16_t>(u >> 16);
// }

// ---- Float16 Conversions (IEEE 754 Half-Precision) ----
__device__ __host__ inline float float16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = (h & 0x03FFu);
    uint32_t u;

    if (exp == 0) {
        if (frac == 0) {
            // Zero (preserve sign)
            u = sign;
        } else {
            // Denormal number
            float f = static_cast<float>(frac) / 1024.0f;
            #ifdef __CUDA_ARCH__
            f = ldexpf(f, -14);
            #else
            f = ::std::ldexp(f, -14);
            #endif
            if (sign) f = -f;
            ::std::memcpy(&u, &f, sizeof(f));
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        u = sign | 0x7F800000u | (frac << 13);
    } else {
        // Normal number
        uint32_t exp32 = exp + (127 - 15);
        u = sign | (exp32 << 23) | (frac << 13);
    }
    
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

__device__ __host__ inline uint16_t float_to_float16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp_32 = (x >> 23) & 0xFF;
    uint32_t mant_32 = x & 0x007FFFFFu;

    // 1. Handle NaN and Infinity
    if (exp_32 == 0xFF) {
        if (mant_32 == 0) {
            // Infinity
            return static_cast<uint16_t>(sign | 0x7C00u);
        } else {
            // NaN
            uint32_t qnan_mant = mant_32 >> 13;
            if (qnan_mant == 0) qnan_mant = 1;
            return static_cast<uint16_t>(sign | 0x7C00u | qnan_mant);
        }
    }

    int32_t exp_16 = static_cast<int32_t>(exp_32) - 127 + 15;

    if (exp_16 <= 0) {
        if (exp_16 < -10) return static_cast<uint16_t>(sign);
        mant_32 |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(1 - exp_16);
        uint32_t half_mant = mant_32 >> (shift + 13);
        return static_cast<uint16_t>(sign | half_mant);
    } else if (exp_16 >= 31) {
        // Overflow to infinity
        return static_cast<uint16_t>(sign | 0x7C00u);
    } else {
        uint16_t half_exp = static_cast<uint16_t>(exp_16);
        uint32_t half_mant = mant_32 + 0x00001000u;
        return static_cast<uint16_t>(sign | (half_exp << 10) | (half_mant >> 13));
    }
}

} // namespace detail

// ==================================================================================
// CUSTOM STRUCT DEFINITIONS FOR BF16/FP16 - FULLY DEVICE-COMPATIBLE
// ==================================================================================

/**
 * @brief BFloat16 (Brain Floating Point 16)
 * Format: Sign(1) + Exponent(8) + Mantissa(7)
 * Range: Same as Float32, reduced precision
 * Use case: Deep learning, faster than FP32, wider range than FP16
 */
struct bfloat16_t {
    uint16_t raw_bits;

    // ---- Constructors ----
    __device__ __host__ bfloat16_t() : raw_bits(0) {}
    __device__ __host__ explicit bfloat16_t(float val) { raw_bits = detail::float_to_bfloat16(val); }
    __device__ __host__ bfloat16_t(const bfloat16_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>    
    __device__ __host__ explicit bfloat16_t(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
    }

    __device__ __host__ operator float() const { return detail::bfloat16_to_float(raw_bits); }

    // ---- Assignment Operators ----
    __device__ __host__ bfloat16_t& operator=(float val) {
        raw_bits = detail::float_to_bfloat16(val);
        return *this;
    }
    
    __device__ __host__ bfloat16_t& operator=(const bfloat16_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>
    __device__ __host__ bfloat16_t& operator=(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
        return *this;
    }

    // ---- Comparison Operators ----
    __device__ __host__ bool operator>(const bfloat16_t& other) const {
        return static_cast<float>(*this) > static_cast<float>(other);
    }
    __device__ __host__ bool operator<(const bfloat16_t& other) const {
        return static_cast<float>(*this) < static_cast<float>(other);
    }
    __device__ __host__ bool operator>=(const bfloat16_t& other) const {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }
    __device__ __host__ bool operator<=(const bfloat16_t& other) const {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }
    __device__ __host__ bool operator==(const bfloat16_t& other) const {
        return raw_bits == other.raw_bits;
    }
    __device__ __host__ bool operator!=(const bfloat16_t& other) const {
        return raw_bits != other.raw_bits;
    }

    // ---- Arithmetic Operators ----
    __device__ __host__ bfloat16_t operator+(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) + static_cast<float>(other));
    }
    
    __device__ __host__ bfloat16_t operator-(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) - static_cast<float>(other));
    }
    
    __device__ __host__ bfloat16_t operator*(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) * static_cast<float>(other));
    }
    
    __device__ __host__ bfloat16_t operator/(const bfloat16_t& other) const {
        return bfloat16_t(static_cast<float>(*this) / static_cast<float>(other));
    }

    // ---- Compound Assignment Operators ----
    __device__ __host__ bfloat16_t& operator+=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) + static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ bfloat16_t& operator-=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) - static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ bfloat16_t& operator*=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) * static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ bfloat16_t& operator/=(const bfloat16_t& other) {
        *this = bfloat16_t(static_cast<float>(*this) / static_cast<float>(other));
        return *this;
    }
};

/**
 * @brief Float16 (IEEE 754 Half Precision)
 * Format: Sign(1) + Exponent(5) + Mantissa(10)
 * Range: ±65504, limited but higher precision than BF16
 * Use case: Graphics, mobile AI, memory-constrained scenarios
 */
struct float16_t {
    uint16_t raw_bits;

    // ---- Constructors ----
    __device__ __host__ float16_t() : raw_bits(0) {}
    __device__ __host__ explicit float16_t(float val) { raw_bits = detail::float_to_float16(val); }
    __device__ __host__ float16_t(const float16_t& other) : raw_bits(other.raw_bits) {}

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>
    __device__ __host__ explicit float16_t(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
    }

    __device__ __host__ operator float() const { return detail::float16_to_float(raw_bits); }

    // ---- Assignment Operators ----
    __device__ __host__ float16_t& operator=(float val) {
        raw_bits = detail::float_to_float16(val);
        return *this;
    }
    
    __device__ __host__ float16_t& operator=(const float16_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    template <typename U, typename = ::std::enable_if_t<
        ::std::is_arithmetic_v<U> && !::std::is_same_v<std::decay_t<U>, float>
    >>
    __device__ __host__ float16_t& operator=(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
        return *this;
    }

    // ---- Comparison Operators ----
    __device__ __host__ bool operator>(const float16_t& other) const {
        return static_cast<float>(*this) > static_cast<float>(other);
    }
    __device__ __host__ bool operator<(const float16_t& other) const {
        return static_cast<float>(*this) < static_cast<float>(other);
    }
    __device__ __host__ bool operator>=(const float16_t& other) const {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }
    __device__ __host__ bool operator<=(const float16_t& other) const {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }
    __device__ __host__ bool operator==(const float16_t& other) const {
        return raw_bits == other.raw_bits;
    }
    __device__ __host__ bool operator!=(const float16_t& other) const {
        return raw_bits != other.raw_bits;
    }

    // ---- Arithmetic Operators ----
    __device__ __host__ float16_t operator+(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) + static_cast<float>(other));
    }
    
    __device__ __host__ float16_t operator-(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) - static_cast<float>(other));
    }
    
    __device__ __host__ float16_t operator*(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) * static_cast<float>(other));
    }
    
    __device__ __host__ float16_t operator/(const float16_t& other) const {
        return float16_t(static_cast<float>(*this) / static_cast<float>(other));
    }

    // ---- Compound Assignment Operators ----
    __device__ __host__ float16_t& operator+=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) + static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ float16_t& operator-=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) - static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ float16_t& operator*=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) * static_cast<float>(other));
        return *this;
    }
    
    __device__ __host__ float16_t& operator/=(const float16_t& other) {
        *this = float16_t(static_cast<float>(*this) / static_cast<float>(other));
        return *this;
    }
};


// ==================================================================================
// MATH OVERLOADS FOR CUSTOM TYPES
// ==================================================================================

    // ---- Float16 Math ----
    inline float16_t abs(float16_t a) { return float16_t(std::abs(static_cast<float>(a))); }
    inline float16_t sqrt(float16_t a) { return float16_t(std::sqrt(static_cast<float>(a))); }
    inline float16_t exp(float16_t a) { return float16_t(std::exp(static_cast<float>(a))); }
    inline float16_t sin(float16_t a) { return float16_t(std::sin(static_cast<float>(a))); }
    inline float16_t cos(float16_t a) { return float16_t(std::cos(static_cast<float>(a))); }
    inline float16_t tan(float16_t a) { return float16_t(std::tan(static_cast<float>(a))); }
    inline float16_t tanh(float16_t a) { return float16_t(std::tanh(static_cast<float>(a))); }
    inline float16_t floor(float16_t a) { return float16_t(std::floor(static_cast<float>(a))); }
    inline float16_t ceil(float16_t a) { return float16_t(std::ceil(static_cast<float>(a))); }
    inline float16_t round(float16_t a) { return float16_t(std::round(static_cast<float>(a))); }
    inline float16_t pow(float16_t a, float16_t b) { return float16_t(std::pow(static_cast<float>(a), static_cast<float>(b))); }
    inline float16_t hypot(float16_t a, float16_t b) { return float16_t(std::hypot(static_cast<float>(a), static_cast<float>(b))); }

    // ---- BFloat16 Math ----
    inline bfloat16_t abs(bfloat16_t a) { return bfloat16_t(std::abs(static_cast<float>(a))); }
    inline bfloat16_t sqrt(bfloat16_t a) { return bfloat16_t(std::sqrt(static_cast<float>(a))); }
    inline bfloat16_t exp(bfloat16_t a) { return bfloat16_t(std::exp(static_cast<float>(a))); }
    inline bfloat16_t log(bfloat16_t a) { return bfloat16_t(std::log(static_cast<float>(a))); }
    inline bfloat16_t sin(bfloat16_t a) { return bfloat16_t(std::sin(static_cast<float>(a))); }
    inline bfloat16_t cos(bfloat16_t a) { return bfloat16_t(std::cos(static_cast<float>(a))); }
    inline bfloat16_t tan(bfloat16_t a) { return bfloat16_t(std::tan(static_cast<float>(a))); }
    inline bfloat16_t tanh(bfloat16_t a) { return bfloat16_t(std::tanh(static_cast<float>(a))); }
    inline bfloat16_t floor(bfloat16_t a) { return bfloat16_t(std::floor(static_cast<float>(a))); }
    inline bfloat16_t ceil(bfloat16_t a) { return bfloat16_t(std::ceil(static_cast<float>(a))); }
    inline bfloat16_t round(bfloat16_t a) { return bfloat16_t(std::round(static_cast<float>(a))); }
    inline bfloat16_t pow(bfloat16_t a, bfloat16_t b) { return bfloat16_t(std::pow(static_cast<float>(a), static_cast<float>(b))); }
    inline bfloat16_t hypot(bfloat16_t a, bfloat16_t b) { return bfloat16_t(std::hypot(static_cast<float>(a), static_cast<float>(b))); }

} // namespace OwnTensor

// ==================================================================================
// std::numeric_limits SPECIALIZATIONS
// ==================================================================================

namespace std {

/**
 * @brief Helper template for numeric_limits of 16-bit float types.
 */
template <typename T>
struct numeric_limits_fp16_helper {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    
    static T lowest() noexcept;
    static T max() noexcept;
    static T min() noexcept { return T(1.0e-8f); }
    static T epsilon() noexcept { return T(0.0001f); }
    static T infinity() noexcept { return T(std::numeric_limits<float>::infinity()); }
    static T quiet_NaN() noexcept { return T(std::numeric_limits<float>::quiet_NaN()); }
};

// ---- BFloat16 Limits ----
template <>
struct numeric_limits<OwnTensor::bfloat16_t> : public numeric_limits_fp16_helper<OwnTensor::bfloat16_t> {};

template <>
inline OwnTensor::bfloat16_t numeric_limits_fp16_helper<OwnTensor::bfloat16_t>::lowest() noexcept {
    return OwnTensor::bfloat16_t(-3.38953e38f);
}

template <>
inline OwnTensor::bfloat16_t numeric_limits_fp16_helper<OwnTensor::bfloat16_t>::max() noexcept {
    return OwnTensor::bfloat16_t(3.38953e38f);
}

// ---- Float16 Limits ----
template <>
struct numeric_limits<OwnTensor::float16_t> : public numeric_limits_fp16_helper<OwnTensor::float16_t> {};

template <>
inline OwnTensor::float16_t numeric_limits_fp16_helper<OwnTensor::float16_t>::lowest() noexcept {
    return OwnTensor::float16_t(-65504.0f);
}

template <>
inline OwnTensor::float16_t numeric_limits_fp16_helper<OwnTensor::float16_t>::max() noexcept {
    return OwnTensor::float16_t(65504.0f);
}

} // namespace std

namespace OwnTensor {
    // ==================================================================================
    // COMPLEX TYPES
    // ==================================================================================

    /**
     * @brief Complex32 (Half-Precision Complex Number)
     * Components: real(float16_t) + imag(float16_t)
     * Use case: Memory-efficient complex computations
     */
      struct complex32_t {
        float16_t real_;
        float16_t imag_;

        // ---- Constructors ----
        __device__ __host__ complex32_t() : real_(0.0f), imag_(0.0f) {}
        
        __device__ __host__ explicit complex32_t(float16_t r, float16_t i = float16_t(0.0f))
            : real_(r), imag_(i) {}
        
        __device__ __host__ complex32_t(const complex32_t& other) 
            : real_(other.real_), imag_(other.imag_) {}

        // Removed template constructor for scalar types to avoid ambiguity
        // template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
        // __device__ __host__ complex32_t(T real_val, T imag_val = T(0))
        //     : real_(float16_t(static_cast<float>(real_val))), 
        //       imag_(float16_t(static_cast<float>(imag_val))) {}

        // Constructor from std::complex<float> for convenience
        __host__ explicit complex32_t(const std::complex<float>& c) 
            : real_(float16_t(c.real())), imag_(float16_t(c.imag())) {}

        // Constructor from std::complex<double> for convenience
        __host__ explicit complex32_t(const std::complex<double>& c) 
            : real_(float16_t(static_cast<float>(c.real()))), 
              imag_(float16_t(static_cast<float>(c.imag()))) {}
    
        // Conversion from other custom complex types (defined out-of-line after complex64_t/complex128_t)
        __device__ __host__ explicit complex32_t(const struct complex64_t& c);
        __device__ __host__ explicit complex32_t(const struct complex128_t& c);

        // Generic scalar constructors for any type convertible to float (resolves ambiguity)
        template <typename T, typename = std::enable_if_t<std::is_convertible_v<T, float> && !std::is_same_v<std::decay_t<T>, float16_t> && !std::is_same_v<std::decay_t<T>, complex32_t>>>
        __device__ __host__ explicit complex32_t(T r)
            : real_(float16_t(static_cast<float>(r))), imag_(float16_t(0.0f)) {}

        template <typename T, typename U, typename = std::enable_if_t<std::is_convertible_v<T, float> && std::is_convertible_v<U, float> && !(std::is_same_v<std::decay_t<T>, float16_t> && std::is_same_v<std::decay_t<U>, float16_t>)>>
        __device__ __host__ explicit complex32_t(T r, U i)
            : real_(float16_t(static_cast<float>(r))), imag_(float16_t(static_cast<float>(i))) {}

        // ---- Accessor Methods ----
        __device__ __host__ float16_t real() const { return real_; }
        __device__ __host__ float16_t imag() const { return imag_; }
        __device__ __host__ void real(float16_t val) { real_ = val; }
        __device__ __host__ void imag(float16_t val) { imag_ = val; }

        // ---- Conversion Operators ----
        __device__ __host__ operator std::complex<float>() const {
            return std::complex<float>(static_cast<float>(real_), static_cast<float>(imag_));
        }

        // ---- Assignment Operators ----
        __device__ __host__ complex32_t& operator=(const complex32_t& other) {
            real_ = other.real_;
            imag_ = other.imag_;
            return *this;
        }

        // ---- Arithmetic Operators ----
        __device__ __host__ complex32_t operator+(const complex32_t& other) const {
            return complex32_t(real_ + other.real_, imag_ + other.imag_);
        }

        __device__ __host__ complex32_t operator-(const complex32_t& other) const {
            return complex32_t(real_ - other.real_, imag_ - other.imag_);
        }

        __device__ __host__ complex32_t operator*(const complex32_t& other) const {
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            float16_t r = real_ * other.real_ - imag_ * other.imag_;
            float16_t i = real_ * other.imag_ + imag_ * other.real_;
            return complex32_t(r, i);
        }

        __device__ __host__ complex32_t operator/(const complex32_t& other) const {
            // (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
            float16_t denom = other.real_ * other.real_ + other.imag_ * other.imag_;
            float16_t r = (real_ * other.real_ + imag_ * other.imag_) / denom;
            float16_t i = (imag_ * other.real_ - real_ * other.imag_) / denom;
            return complex32_t(r, i);
        }

        __device__ __host__ complex32_t operator-() const {
            return complex32_t(float16_t(0.0f) - real_, float16_t(0.0f) - imag_);
        }

        // ---- Compound Assignment Operators ----
        __device__ __host__ complex32_t& operator+=(const complex32_t& other) {
            real_ += other.real_;
            imag_ += other.imag_;
            return *this;
        }

        __device__ __host__ complex32_t& operator-=(const complex32_t& other) {
            real_ -= other.real_;
            imag_ -= other.imag_;
            return *this;
        }

        __device__ __host__ complex32_t& operator*=(const complex32_t& other) {
            *this = *this * other;
            return *this;
        }

        __device__ __host__ complex32_t& operator/=(const complex32_t& other) {
            *this = *this / other;
            return *this;
        }

        // ---- Comparison Operators ----
        __device__ __host__ bool operator==(const complex32_t& other) const {
            return (real_ == other.real_) && (imag_ == other.imag_);
        }

        __device__ __host__ bool operator!=(const complex32_t& other) const {
            return !(*this == other);
        }
    };

    // Helper functions for complex32_t to match std::complex interface for float
    // These are global functions, not member functions.
    __device__ __host__ inline float real(const complex32_t& c) { return static_cast<float>(c.real()); }
    __device__ __host__ inline float imag(const complex32_t& c) { return static_cast<float>(c.imag()); }
    __device__ __host__ inline void real(complex32_t& c, float v) { c.real(float16_t(v)); }
    __device__ __host__ inline void imag(complex32_t& c, float v) { c.imag(float16_t(v)); }

    // ---- Math Functions for complex32_t ----
    
    // Absolute value (magnitude)
    inline float16_t abs(const complex32_t& z) {
        float r = static_cast<float>(z.real());
        float i = static_cast<float>(z.imag());
        return float16_t(std::sqrt(r * r + i * i));
    }

    // Conjugate
    inline complex32_t conj(const complex32_t& z) {
        return complex32_t(z.real(), float16_t(0.0f) - z.imag());
    }

    // Argument (phase angle)
    inline float16_t arg(const complex32_t& z) {
        return float16_t(std::atan2(static_cast<float>(z.imag()), 
                                     static_cast<float>(z.real())));
    }

    // Norm (squared magnitude)
    inline float16_t norm(const complex32_t& z) {
        float16_t r = z.real();
        float16_t i = z.imag();
        return r * r + i * i;
    }

    // Polar form constructor
    inline complex32_t polar(float16_t rho, float16_t theta) {
        float r = static_cast<float>(rho);
        float t = static_cast<float>(theta);
        return complex32_t(float16_t(r * std::cos(t)), 
                          float16_t(r * std::sin(t)));
    }

    // ==================================================================================
    // STANDARD COMPLEX TYPE ALIASES
    // ==================================================================================
    // ------------------------------------------------------------------
// COMPLEX64 – single‑precision complex number
// ------------------------------------------------------------------
struct complex64_t {
    float real_;
    float imag_;

    // Constructors
    __device__ __host__ complex64_t() : real_(0.0f), imag_(0.0f) {}
    __device__ __host__ explicit complex64_t(float r, float i = 0.0f) : real_(r), imag_(i) {}

    // Generic scalar constructor for any type convertible to float (resolves ambiguity)
    template <typename T, typename = std::enable_if_t<std::is_convertible_v<T, float> && !std::is_same_v<std::decay_t<T>, float> && !std::is_same_v<std::decay_t<T>, complex64_t>>>
    __device__ __host__ explicit complex64_t(T r)
        : real_(static_cast<float>(r)), imag_(0.0f) {}
    
    __host__ explicit complex64_t(const std::complex<float>& c)
        : real_(c.real()), imag_(c.imag()) {}
    __device__ __host__ complex64_t(const complex64_t& other)
        : real_(other.real_), imag_(other.imag_) {}
    
    // Copy assignment operator
    __device__ __host__ complex64_t& operator=(const complex64_t& other) {
        if (this != &other) {
            real_ = other.real_;
            imag_ = other.imag_;
        }
        return *this;
    }

    // Accessors
    __device__ __host__ float real() const { return real_; }
    __device__ __host__ float imag() const { return imag_; }
    __device__ __host__ void real(float v) { real_ = v; }
    __device__ __host__ void imag(float v) { imag_ = v; }

    // Conversion to std::complex
    __device__ __host__ operator std::complex<float>() const {
        return std::complex<float>(real_, imag_);
    }

    // Conversion from complex32_t
    __device__ __host__ explicit complex64_t(const complex32_t& c)
        : real_(static_cast<float>(c.real())), imag_(static_cast<float>(c.imag())) {}

    // Conversion from complex128_t (defined out-of-line after complex128_t)
    __device__ __host__ explicit complex64_t(const struct complex128_t& c);

    // Arithmetic operators
    __device__ __host__ complex64_t operator+(const complex64_t& o) const {
        return complex64_t(real_ + o.real_, imag_ + o.imag_);
    }
    __device__ __host__ complex64_t operator-(const complex64_t& o) const {
        return complex64_t(real_ - o.real_, imag_ - o.imag_);
    }
    __device__ __host__ complex64_t operator*(const complex64_t& o) const {
        return complex64_t(real_ * o.real_ - imag_ * o.imag_,
                real_ * o.imag_ + imag_ * o.real_);
    }
    __device__ __host__ complex64_t operator/(const complex64_t& o) const {
        float denom = o.real_ * o.real_ + o.imag_ * o.imag_;
        return complex64_t((real_ * o.real_ + imag_ * o.imag_) / denom,
                (imag_ * o.real_ - real_ * o.imag_) / denom);
    }
    __device__ __host__ complex64_t operator-() const {
        return complex64_t(-real_, -imag_);
    }

    // Compound assignment
    __device__ __host__ complex64_t& operator+=(const complex64_t& o) { real_ += o.real_; imag_ += o.imag_; return *this; }
    __device__ __host__ complex64_t& operator-=(const complex64_t& o) { real_ -= o.real_; imag_ -= o.imag_; return *this; }
    __device__ __host__ complex64_t& operator*=(const complex64_t& o) { *this = *this * o; return *this; }
    __device__ __host__ complex64_t& operator/=(const complex64_t& o) { *this = *this / o; return *this; }

    // Comparison (exact)
    __device__ __host__ bool operator==(const complex64_t& o) const { return real_ == o.real_ && imag_ == o.imag_; }
    __device__ __host__ bool operator!=(const complex64_t& o) const { return !(*this == o); }
};

// Helper functions for complex64_t to match std::complex interface for float
__device__ __host__ inline float real(const complex64_t& c) { return c.real(); }
__device__ __host__ inline float imag(const complex64_t& c) { return c.imag(); }
__device__ __host__ inline void real(complex64_t& c, float v) { c.real(v); }
__device__ __host__ inline void imag(complex64_t& c, float v) { c.imag(v); }

// ---- Math Functions for complex64_t ----

// Absolute value (magnitude)
inline float abs(const complex64_t& z) {
    float r = z.real();
    float i = z.imag();
    return std::sqrt(r * r + i * i);
}

// Conjugate
inline complex64_t conj(const complex64_t& z) {
    return complex64_t(z.real(), -z.imag());
}

// Argument (phase angle)
inline float arg(const complex64_t& z) {
    return std::atan2(z.imag(), z.real());
}

// Norm (squared magnitude)
inline float norm(const complex64_t& z) {
    float r = z.real();
    float i = z.imag();
    return r * r + i * i;
}

// Polar form constructor
inline complex64_t polar(float rho, float theta) {
    return complex64_t(rho * std::cos(theta), rho * std::sin(theta));
}

// ------------------------------------------------------------------
// COMPLEX128 – double‑precision complex number
// ------------------------------------------------------------------
struct complex128_t {
    double real_;
    double imag_;

    // Constructors
    __device__ __host__ complex128_t() : real_(0.0), imag_(0.0) {}
    __device__ __host__ explicit complex128_t(double r, double i = 0.0) : real_(r), imag_(i) {}
    
    // Generic scalar constructor for any type convertible to double
    template <typename T, typename = std::enable_if_t<std::is_convertible_v<T, double> && !std::is_same_v<std::decay_t<T>, double> && !std::is_same_v<std::decay_t<T>, complex128_t>>>
    __device__ __host__ explicit complex128_t(T r)
        : real_(static_cast<double>(r)), imag_(0.0) {}

    __host__ explicit complex128_t(const std::complex<double>& c)
        : real_(c.real()), imag_(c.imag()) {}
    __device__ __host__ complex128_t(const complex128_t& other)
        : real_(other.real_), imag_(other.imag_) {}
    
    // Copy assignment operator
    __device__ __host__ complex128_t& operator=(const complex128_t& other) {
        if (this != &other) {
            real_ = other.real_;
            imag_ = other.imag_;
        }
        return *this;
    }
    
    // Conversion from other custom complex types
    __device__ __host__ explicit complex128_t(const complex32_t& c)
        : real_(static_cast<double>(static_cast<float>(c.real()))), 
          imag_(static_cast<double>(static_cast<float>(c.imag()))) {}
    __device__ __host__ explicit complex128_t(const complex64_t& c)
        : real_(static_cast<double>(c.real())), imag_(static_cast<double>(c.imag())) {}

    // Accessors
    __device__ __host__ double real() const { return real_; }
    __device__ __host__ double imag() const { return imag_; }
    __device__ __host__ void real(double v) { real_ = v; }
    __device__ __host__ void imag(double v) { imag_ = v; }

    // Conversion to std::complex
    __device__ __host__ operator std::complex<double>() const {
        return std::complex<double>(real_, imag_);
    }

    // Arithmetic operators
    __device__ __host__ complex128_t operator+(const complex128_t& o) const {
        return complex128_t(real_ + o.real_, imag_ + o.imag_);
    }
    __device__ __host__ complex128_t operator-(const complex128_t& o) const {
        return complex128_t(real_ - o.real_, imag_ - o.imag_);
    }
    __device__ __host__ complex128_t operator*(const complex128_t& o) const {
        return complex128_t(real_ * o.real_ - imag_ * o.imag_,
                real_ * o.imag_ + imag_ * o.real_);
    }
    __device__ __host__ complex128_t operator/(const complex128_t& o) const {
        double denom = o.real_ * o.real_ + o.imag_ * o.imag_;
        return complex128_t((real_ * o.real_ + imag_ * o.imag_) / denom,
                (imag_ * o.real_ - real_ * o.imag_) / denom);
    }
    __device__ __host__ complex128_t operator-() const {
        return complex128_t(-real_, -imag_);
    }

    // Compound assignment
    __device__ __host__ complex128_t& operator+=(const complex128_t& o) { real_ += o.real_; imag_ += o.imag_; return *this; }
    __device__ __host__ complex128_t& operator-=(const complex128_t& o) { real_ -= o.real_; imag_ -= o.imag_; return *this; }
    __device__ __host__ complex128_t& operator*=(const complex128_t& o) { *this = *this * o; return *this; }
    __device__ __host__ complex128_t& operator/=(const complex128_t& o) { *this = *this / o; return *this; }

    // Comparison (exact)
    __device__ __host__ bool operator==(const complex128_t& o) const { return real_ == o.real_ && imag_ == o.imag_; }
    __device__ __host__ bool operator!=(const complex128_t& o) const { return !(*this == o); }
};

// Helper functions for complex128_t to match std::complex interface
__device__ __host__ inline double real(const complex128_t& c) { return c.real(); }
__device__ __host__ inline double imag(const complex128_t& c) { return c.imag(); }
__device__ __host__ inline void real(complex128_t& c, double v) { c.real(v); }
__device__ __host__ inline void imag(complex128_t& c, double v) { c.imag(v); }

// ---- Math Functions for complex128_t ----

// Absolute value (magnitude)
inline double abs(const complex128_t& z) {
    double r = z.real();
    double i = z.imag();
    return std::sqrt(r * r + i * i);
}

// Conjugate
inline complex128_t conj(const complex128_t& z) {
    return complex128_t(z.real(), -z.imag());
}

// Argument (phase angle)
inline double arg(const complex128_t& z) {
    return std::atan2(z.imag(), z.real());
}

// Norm (squared magnitude)
inline double norm(const complex128_t& z) {
    double r = z.real();
    double i = z.imag();
    return r * r + i * i;
}

// Polar form constructor
inline complex128_t polar(double rho, double theta) {
    return complex128_t(rho * std::cos(theta), rho * std::sin(theta));
}

// ------------------------------------------------------------------
// Cross-type conversions (defined after all three complex types are declared)
// ------------------------------------------------------------------

// Out-of-line constructor definitions for cross-complex conversions
inline __device__ __host__ complex32_t::complex32_t(const complex64_t& c)
    : real_(float16_t(c.real())), imag_(float16_t(c.imag())) {}

inline __device__ __host__ complex32_t::complex32_t(const complex128_t& c)
    : real_(float16_t(static_cast<float>(c.real()))), imag_(float16_t(static_cast<float>(c.imag()))) {}

inline __device__ __host__ complex64_t::complex64_t(const complex128_t& c)
    : real_(static_cast<float>(c.real())), imag_(static_cast<float>(c.imag())) {}

// Helper conversion functions
inline complex64_t to_complex64(const complex128_t& c) {
    return complex64_t(static_cast<float>(c.real()), static_cast<float>(c.imag()));
}

inline complex128_t to_complex128(const complex64_t& c) {
    return complex128_t(static_cast<double>(c.real()), static_cast<double>(c.imag()));
}

 // namespace OwnTensor
    // ==================================================================================
    // GLOBAL ARITHMETIC OPERATORS FOR MIXED SCALAR-COMPLEX TYPES
    // ==================================================================================

    // ---- complex32_t ----
    __device__ __host__ inline complex32_t operator+(float s, const complex32_t& c) { return complex32_t(s) + c; }
    __device__ __host__ inline complex32_t operator+(const complex32_t& c, float s) { return c + complex32_t(s); }
    __device__ __host__ inline complex32_t operator-(float s, const complex32_t& c) { return complex32_t(s) - c; }
    __device__ __host__ inline complex32_t operator-(const complex32_t& c, float s) { return c - complex32_t(s); }
    __device__ __host__ inline complex32_t operator*(float s, const complex32_t& c) { return complex32_t(s) * c; }
    __device__ __host__ inline complex32_t operator*(const complex32_t& c, float s) { return c * complex32_t(s); }
    __device__ __host__ inline complex32_t operator/(float s, const complex32_t& c) { return complex32_t(s) / c; }
    __device__ __host__ inline complex32_t operator/(const complex32_t& c, float s) { return c / complex32_t(s); }

    __device__ __host__ inline complex32_t operator+(double s, const complex32_t& c) { return complex32_t(s) + c; }
    __device__ __host__ inline complex32_t operator+(const complex32_t& c, double s) { return c + complex32_t(s); }
    __device__ __host__ inline complex32_t operator-(double s, const complex32_t& c) { return complex32_t(s) - c; }
    __device__ __host__ inline complex32_t operator-(const complex32_t& c, double s) { return c - complex32_t(s); }
    __device__ __host__ inline complex32_t operator*(double s, const complex32_t& c) { return complex32_t(s) * c; }
    __device__ __host__ inline complex32_t operator*(const complex32_t& c, double s) { return c * complex32_t(s); }
    __device__ __host__ inline complex32_t operator/(double s, const complex32_t& c) { return complex32_t(s) / c; }
    __device__ __host__ inline complex32_t operator/(const complex32_t& c, double s) { return c / complex32_t(s); }

    // integer overloads for complex32_t
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator+(T s, const complex32_t& c) { return complex32_t(static_cast<float>(s)) + c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator+(const complex32_t& c, T s) { return c + complex32_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator-(T s, const complex32_t& c) { return complex32_t(static_cast<float>(s)) - c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator-(const complex32_t& c, T s) { return c - complex32_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator*(T s, const complex32_t& c) { return complex32_t(static_cast<float>(s)) * c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator*(const complex32_t& c, T s) { return c * complex32_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator/(T s, const complex32_t& c) { return complex32_t(static_cast<float>(s)) / c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex32_t operator/(const complex32_t& c, T s) { return c / complex32_t(static_cast<float>(s)); }

    // ---- complex64_t ----
    __device__ __host__ inline complex64_t operator+(float s, const complex64_t& c) { return complex64_t(s) + c; }
    __device__ __host__ inline complex64_t operator+(const complex64_t& c, float s) { return c + complex64_t(s); }
    __device__ __host__ inline complex64_t operator-(float s, const complex64_t& c) { return complex64_t(s) - c; }
    __device__ __host__ inline complex64_t operator-(const complex64_t& c, float s) { return c - complex64_t(s); }
    __device__ __host__ inline complex64_t operator*(float s, const complex64_t& c) { return complex64_t(s) * c; }
    __device__ __host__ inline complex64_t operator*(const complex64_t& c, float s) { return c * complex64_t(s); }
    __device__ __host__ inline complex64_t operator/(float s, const complex64_t& c) { return complex64_t(s) / c; }
    __device__ __host__ inline complex64_t operator/(const complex64_t& c, float s) { return c / complex64_t(s); }

    __device__ __host__ inline complex64_t operator+(double s, const complex64_t& c) { return complex64_t(s) + c; }
    __device__ __host__ inline complex64_t operator+(const complex64_t& c, double s) { return c + complex64_t(s); }
    __device__ __host__ inline complex64_t operator-(double s, const complex64_t& c) { return complex64_t(s) - c; }
    __device__ __host__ inline complex64_t operator-(const complex64_t& c, double s) { return c - complex64_t(s); }
    __device__ __host__ inline complex64_t operator*(double s, const complex64_t& c) { return complex64_t(s) * c; }
    __device__ __host__ inline complex64_t operator*(const complex64_t& c, double s) { return c * complex64_t(s); }
    __device__ __host__ inline complex64_t operator/(double s, const complex64_t& c) { return complex64_t(s) / c; }
    __device__ __host__ inline complex64_t operator/(const complex64_t& c, double s) { return c / complex64_t(s); }

    // integer overloads for complex64_t
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator+(T s, const complex64_t& c) { return complex64_t(static_cast<float>(s)) + c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator+(const complex64_t& c, T s) { return c + complex64_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator-(T s, const complex64_t& c) { return complex64_t(static_cast<float>(s)) - c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator-(const complex64_t& c, T s) { return c - complex64_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator*(T s, const complex64_t& c) { return complex64_t(static_cast<float>(s)) * c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator*(const complex64_t& c, T s) { return c * complex64_t(static_cast<float>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator/(T s, const complex64_t& c) { return complex64_t(static_cast<float>(s)) / c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex64_t operator/(const complex64_t& c, T s) { return c / complex64_t(static_cast<float>(s)); }

    // ---- complex128_t ----
    __device__ __host__ inline complex128_t operator+(float s, const complex128_t& c) { return complex128_t(s) + c; }
    __device__ __host__ inline complex128_t operator+(const complex128_t& c, float s) { return c + complex128_t(s); }
    __device__ __host__ inline complex128_t operator-(float s, const complex128_t& c) { return complex128_t(s) - c; }
    __device__ __host__ inline complex128_t operator-(const complex128_t& c, float s) { return c - complex128_t(s); }
    __device__ __host__ inline complex128_t operator*(float s, const complex128_t& c) { return complex128_t(s) * c; }
    __device__ __host__ inline complex128_t operator*(const complex128_t& c, float s) { return c * complex128_t(s); }
    __device__ __host__ inline complex128_t operator/(float s, const complex128_t& c) { return complex128_t(s) / c; }
    __device__ __host__ inline complex128_t operator/(const complex128_t& c, float s) { return c / complex128_t(s); }

    __device__ __host__ inline complex128_t operator+(double s, const complex128_t& c) { return complex128_t(s) + c; }
    __device__ __host__ inline complex128_t operator+(const complex128_t& c, double s) { return c + complex128_t(s); }
    __device__ __host__ inline complex128_t operator-(double s, const complex128_t& c) { return complex128_t(s) - c; }
    __device__ __host__ inline complex128_t operator-(const complex128_t& c, double s) { return c - complex128_t(s); }
    __device__ __host__ inline complex128_t operator*(double s, const complex128_t& c) { return complex128_t(s) * c; }
    __device__ __host__ inline complex128_t operator*(const complex128_t& c, double s) { return c * complex128_t(s); }
    __device__ __host__ inline complex128_t operator/(double s, const complex128_t& c) { return complex128_t(s) / c; }
    __device__ __host__ inline complex128_t operator/(const complex128_t& c, double s) { return c / complex128_t(s); }

    // integer overloads for complex128_t
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator+(T s, const complex128_t& c) { return complex128_t(static_cast<double>(s)) + c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator+(const complex128_t& c, T s) { return c + complex128_t(static_cast<double>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator-(T s, const complex128_t& c) { return complex128_t(static_cast<double>(s)) - c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator-(const complex128_t& c, T s) { return c - complex128_t(static_cast<double>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator*(T s, const complex128_t& c) { return complex128_t(static_cast<double>(s)) * c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator*(const complex128_t& c, T s) { return c * complex128_t(static_cast<double>(s)); }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator/(T s, const complex128_t& c) { return complex128_t(static_cast<double>(s)) / c; }
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __device__ __host__ inline complex128_t operator/(const complex128_t& c, T s) { return c / complex128_t(static_cast<double>(s)); }

    // ==================================================================================
    // GLOBAL COMPARISON OPERATORS FOR MIXED SCALAR-COMPLEX TYPES
    // ==================================================================================

    // ---- complex32_t ----
    __device__ __host__ inline bool operator==(const complex32_t& c, float s) { return c == complex32_t(s); }
    __device__ __host__ inline bool operator==(float s, const complex32_t& c) { return complex32_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex32_t& c, float s) { return c != complex32_t(s); }
    __device__ __host__ inline bool operator!=(float s, const complex32_t& c) { return complex32_t(s) != c; }

    __device__ __host__ inline bool operator==(const complex32_t& c, double s) { return c == complex32_t(s); }
    __device__ __host__ inline bool operator==(double s, const complex32_t& c) { return complex32_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex32_t& c, double s) { return c != complex32_t(s); }
    __device__ __host__ inline bool operator!=(double s, const complex32_t& c) { return complex32_t(s) != c; }

    // ---- complex64_t ----
    __device__ __host__ inline bool operator==(const complex64_t& c, float s) { return c == complex64_t(s); }
    __device__ __host__ inline bool operator==(float s, const complex64_t& c) { return complex64_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex64_t& c, float s) { return c != complex64_t(s); }
    __device__ __host__ inline bool operator!=(float s, const complex64_t& c) { return complex64_t(s) != c; }

    __device__ __host__ inline bool operator==(const complex64_t& c, double s) { return c == complex64_t(s); }
    __device__ __host__ inline bool operator==(double s, const complex64_t& c) { return complex64_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex64_t& c, double s) { return c != complex64_t(s); }
    __device__ __host__ inline bool operator!=(double s, const complex64_t& c) { return complex64_t(s) != c; }

    // ---- complex128_t ----
    __device__ __host__ inline bool operator==(const complex128_t& c, float s) { return c == complex128_t(s); }
    __device__ __host__ inline bool operator==(float s, const complex128_t& c) { return complex128_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex128_t& c, float s) { return c != complex128_t(s); }
    __device__ __host__ inline bool operator!=(float s, const complex128_t& c) { return complex128_t(s) != c; }

    __device__ __host__ inline bool operator==(const complex128_t& c, double s) { return c == complex128_t(s); }
    __device__ __host__ inline bool operator==(double s, const complex128_t& c) { return complex128_t(s) == c; }
    __device__ __host__ inline bool operator!=(const complex128_t& c, double s) { return c != complex128_t(s); }
    __device__ __host__ inline bool operator!=(double s, const complex128_t& c) { return complex128_t(s) != c; }
    
    // ==================================================================================
    // MATH FUNCTIONS
    // ==================================================================================
    
    __device__ __host__ inline bool isnan(const complex32_t& c) {
        return std::isnan(static_cast<float>(c.real_)) || std::isnan(static_cast<float>(c.imag_));
    }

    __device__ __host__ inline bool isnan(const complex64_t& c) {
        return std::isnan(c.real_) || std::isnan(c.imag_);
    }

    __device__ __host__ inline bool isnan(const complex128_t& c) {
        return std::isnan(c.real_) || std::isnan(c.imag_);
    }
    }
 // namespace OwnTensor
