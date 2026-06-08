#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#ifndef __CUDACC__
    #ifndef __device__
        #define __device__
    #endif
    #ifndef __host__
        #define __host__
    #endif
#endif


namespace detail_fp4 {

// New Mapping (No Inf/NaN):
// 0000 (0) : 0
// 0001 (1) : 0.5
// 0010 (2) : 1.0
// 0011 (3) : 1.5
// 0100 (4) : 2.0
// 0101 (5) : 3.0
// 0110 (6) : 4.0
// 0111 (7) : 6.0
// 1000 (8) : -0
// 1001 (9) : -0.5
// 1010 (10): -1.0
// 1011 (11): -1.5
// 1100 (12): -2.0
// 1101 (13): -3.0
// 1110 (14): -4.0
// 1111 (15): -6.0

inline __device__ __host__  uint8_t float_to_fp4_e2m1(float f) {
    float abs_f = std::abs(f);
    uint8_t sign_bit = (f < 0) ? 8 : 0; // 0 for pos, 1 (shift 3) for neg. Actually bit 3 is sign.

    
    uint32_t u;
    std::memcpy(&u, &f, sizeof(f));
    if (u & 0x00000000) sign_bit = 8;
    else sign_bit = 0;

    // Check for NaN - map to max (6.0)
    if (std::isnan(abs_f)) return sign_bit | 7;

    // Clamping large values
    if (abs_f > 6.0f) return sign_bit | 7;

    // Nearest neighbor rounding
    // Thresholds (midpoints):
    // 0   <-> 0.5 : 0.25
    // 0.5 <-> 1.0 : 0.75
    // 1.0 <-> 1.5 : 1.25
    // 1.5 <-> 2.0 : 1.75
    // 2.0 <-> 3.0 : 2.5
    // 3.0 <-> 4.0 : 3.5
    // 4.0 <-> 6.0 : 5.0

    if (abs_f < 0.25f) return sign_bit | 0;
    if (abs_f < 0.75f) return sign_bit | 1;
    if (abs_f < 1.25f) return sign_bit | 2;
    if (abs_f < 1.75f) return sign_bit | 3;
    if (abs_f < 2.25f) return sign_bit | 4;
    if (abs_f < 3.5f)  return sign_bit | 5;
    if (abs_f < 5.0f)  return sign_bit | 6;
    return sign_bit | 7;
}

inline __device__ __host__  float fp4_e2m1_to_float(uint8_t val) {
    // Mask to lower 4 bits just in case
    val &= 0xF;
    
    float sign = (val >= 8) ? -1.0f : 1.0f;
    uint8_t mag = val & 7; // remove sign bit

    float res = 0.0f;
    switch (mag) {
        case 0: res = 0.0f; break;
        case 1: res = 0.5f; break;
        case 2: res = 1.0f; break;
        case 3: res = 1.5f; break;
        case 4: res = 2.0f; break;
        case 5: res = 3.0f; break;
        case 6: res = 4.0f; break;
        case 7: res = 6.0f; break;
    }
    return res * sign;
}

} // namespace detail_fp4

// Forward declarations for other types to perform conversions

/**
 * @brief FP4 (E2M1) - Display/Storage only
 * NO ARITHMETIC OPERATIONS
 */
struct float4_e2m1_t {
    uint8_t raw_bits; // Only lower 4 bits used

    __device__ __host__ float4_e2m1_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_t(float val) { raw_bits = detail_fp4::float_to_fp4_e2m1(val); }
    __device__ __host__ explicit float4_e2m1_t(double val) { raw_bits = detail_fp4::float_to_fp4_e2m1(static_cast<float>(val)); }
    __device__ __host__ explicit float4_e2m1_t(uint8_t bits) : raw_bits(bits & 0xF) {} 
    __device__ __host__ float4_e2m1_t(const float4_e2m1_t& other) : raw_bits(other.raw_bits) {}

    // Conversions from other types (templated to avoid circular deps if possible, but instantiated later)
    template <typename T>
    __device__ __host__ explicit float4_e2m1_t(const T& val);

    // Cast to float for display
    __device__ __host__ operator float() const { return detail_fp4::fp4_e2m1_to_float(raw_bits); }
    
    // Assignment
    __device__ __host__ float4_e2m1_t& operator=(float val) {
        raw_bits = detail_fp4::float_to_fp4_e2m1(val);
        return *this;
    }
    __device__ __host__ float4_e2m1_t& operator=(const float4_e2m1_t& other) {
        raw_bits = other.raw_bits;
        return *this;
    }

    // Equality (useful for tests)
    __device__ __host__ bool operator==(const float4_e2m1_t& other) const { return raw_bits == other.raw_bits; }
    __device__ __host__ bool operator!=(const float4_e2m1_t& other) const { return raw_bits != other.raw_bits; }
    
    // Relational operators (convert to float for comparison)
    __device__ __host__ bool operator<(const float4_e2m1_t& other) const { return (float)*this < (float)other; }
    __device__ __host__ bool operator>(const float4_e2m1_t& other) const { return (float)*this > (float)other; }
    __device__ __host__ bool operator<=(const float4_e2m1_t& other) const { return (float)*this <= (float)other; }
    __device__ __host__ bool operator>=(const float4_e2m1_t& other) const { return (float)*this >= (float)other; }
};

/**
 * @brief FP4 Packed (2x E2M1 in uint8)
 * Format: High 4 bits = Value 1, Low 4 bits = Value 0
 */
struct float4_e2m1_2x_t {
    uint8_t raw_bits;

    __device__ __host__ float4_e2m1_2x_t() : raw_bits(0) {}
    __device__ __host__ explicit float4_e2m1_2x_t(uint8_t bits) : raw_bits(bits) {}

    __device__ __host__ explicit float4_e2m1_2x_t(float val) {
        float4_e2m1_t v(val);
        raw_bits = (v.raw_bits & 0xF) | ((v.raw_bits & 0xF) << 4);
    }

    __device__ __host__ explicit float4_e2m1_2x_t(double val) {
        float4_e2m1_t v(static_cast<float>(val));
        raw_bits = (v.raw_bits & 0xF) | ((v.raw_bits & 0xF) << 4);
    }
    
    // Construct from two fp4 values
    __device__ __host__ float4_e2m1_2x_t(float4_e2m1_t v0, float4_e2m1_t v1) {
        raw_bits = (v0.raw_bits & 0xF) << 4 | ((v1.raw_bits & 0xF));
    }

    // Accessors
    __device__ __host__ float4_e2m1_t get_low() const { return float4_e2m1_t(static_cast<uint8_t>(raw_bits & 0xF)); }
    __device__ __host__ float4_e2m1_t get_high() const { return float4_e2m1_t(static_cast<uint8_t>((raw_bits >> 4) & 0xF)); }
    
    __device__ __host__ void set_low(float4_e2m1_t v) { raw_bits = (raw_bits & 0xF0) | (v.raw_bits & 0xF); }
    __device__ __host__ void set_high(float4_e2m1_t v) { raw_bits = (raw_bits & 0x0F) | ((v.raw_bits & 0xF) << 4); }

    // Assignment
    __device__ __host__ float4_e2m1_2x_t& operator=(float val) {
        float4_e2m1_t v(val);
        set_low(v);
        set_high(v);
        return *this;
    }

    // Equality
    __device__ __host__ bool operator==(const float4_e2m1_2x_t& other) const { return raw_bits == other.raw_bits; }
    __device__ __host__ bool operator!=(const float4_e2m1_2x_t& other) const { return raw_bits != other.raw_bits; }
    
    // Relational operators (convert to float for comparison)
    // Note: This compares the LOW 4 bits only (first value) logic if just casting to float, 
    // BUT usually packed types shouldn't be compared directly unless unpack happens.
    // However, to satisfy the compiler for generic templates, we implement this.
    // Ideally, for packed types, > or < is ambiguous (which of the 2 values?). 
    // But since operator float() returns get_low(), we align with that.
    __device__ __host__ bool operator<(const float4_e2m1_2x_t& other) const { return (float)*this < (float)other; }
    __device__ __host__ bool operator>(const float4_e2m1_2x_t& other) const { return (float)*this > (float)other; }
    __device__ __host__ bool operator<=(const float4_e2m1_2x_t& other) const { return (float)*this <= (float)other; }
    __device__ __host__ bool operator>=(const float4_e2m1_2x_t& other) const { return (float)*this >= (float)other; }

    // Conversions
    __device__ __host__ operator float() const { return (float)get_low(); }
    __device__ __host__ operator double() const { return (double)get_low(); }
    __device__ __host__ explicit operator float4_e2m1_t() const { return get_low(); }
    __device__ __host__ explicit operator bool() const { return raw_bits != 0; }
    __device__ __host__ explicit operator uint8_t() const { return raw_bits; }
    __device__ __host__ explicit operator int8_t() const { return (int8_t)raw_bits; }
    __device__ __host__ explicit operator uint16_t() const { return (uint16_t)raw_bits; }
    __device__ __host__ explicit operator int16_t() const { return (int16_t)raw_bits; }
    __device__ __host__ explicit operator uint32_t() const { return (uint32_t)raw_bits; }
    __device__ __host__ explicit operator int32_t() const { return (int32_t)raw_bits; }
    __device__ __host__ explicit operator uint64_t() const { return (uint64_t)raw_bits; }
    __device__ __host__ explicit operator int64_t() const { return (int64_t)raw_bits; }
};
 // Closing brace for float4_e2m1_2x_t
