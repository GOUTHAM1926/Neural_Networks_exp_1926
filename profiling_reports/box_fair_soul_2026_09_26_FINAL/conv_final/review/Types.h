#pragma once

#include <cstdint>
#include <limits>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <complex>
// nvcc only: the GPU paths of the conversions below call NVIDIA's intrinsics
// (__float2bfloat16_rn, __float2half_rn, __half2float, __double2bfloat16, ...).
// A plain host compiler (g++) never sees these headers and uses the manual code.
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
// CPU builds with -mf16c (the library's CPU flags): the fp16 <-> fp32 conversions below
// use the x86 F16C instructions (_cvtss_sh / _cvtsh_ss).
#if defined(__F16C__) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
#endif
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

// ==================================================================================
// bf16 / fp16  <->  fp32 / fp64 / integer conversions
// ----------------------------------------------------------------------------------
// Every conversion here rounds to nearest, ties to even (the IEEE 754 default for
// binary formats), and rounds only ONCE.
//
// GPU (nvcc device pass, __CUDA_ARCH__ defined): NVIDIA's intrinsics. Each intrinsic
//   picks the instruction for the target GPU at compile time (NV_IF_ELSE_TARGET inside
//   cuda_bf16.hpp / cuda_fp16.hpp). Example: __float2bfloat16_rn is one F2FP.BF16
//   instruction on sm_80 and newer (F2FP.BF16.PACK_AB on sm_80/86,
//   F2FP.BF16.F32.PACK_AB from sm_89), and NVIDIA's own round-to-nearest-even software
//   on older GPUs.
// CPU (nvcc host pass, or a plain host compiler): fp16 <-> fp32 use the x86 F16C
//   instructions when the file is compiled with -mf16c (the library's CPU flags), and
//   the manual code below otherwise. Every other CPU conversion is the manual code; it
//   uses integer operations for every step that rounds, so it does not depend on the
//   FPU rounding mode or on flush-to-zero. The exception is double (and 32/64-bit
//   integer) input: it is rounded to odd into float with the hardware conversion and
//   then rounded once into bf16/fp16, NVIDIA's host method; that needs flush-to-zero
//   off (true for the library's CPU code, which is not built with fast-math).
//
// NaN: the CPU code returns the canonical NaN 0x7FFF for both bf16 and fp16, the same
// value CUDA's own software conversion returns. The sign and payload of a NaN are not
// kept (IEEE 754 does not require it).
//
// Modified 2026-09-26 (this block):
//   - float_to_bfloat16: an `exponent > 0x8E` clamp sent every |x| >= 65,536 to Inf.
//     0x8E = 127 + 15 is the fp16 exponent limit, not the bf16 one. Removed.
//   - float_to_float16: rewritten; the old code was wrong for 100,815,872 of the 2^32
//     fp32 inputs (details at the function).
//   - bfloat16_to_float: three branches that all computed the same shift -> one shift.
//   - double_to_bfloat16, double_to_float16, int64_to_float_rto,
//     arithmetic_to_bfloat16, arithmetic_to_float16: new. The struct constructors used
//     to send double and integer inputs through float first, which rounds twice (bf16:
//     double and integers above 2^24; fp16: double).
//   - GPU paths now call the intrinsics instead of the manual code. Compiled for
//     sm_90a (H100/H200), one fp32 -> bf16 conversion went from 17 instructions with
//     a branch to 1.
//   - CPU method choices were measured old vs new (Intel i7-14700K, 1 thread, ns per
//     value, 4M values, 12 alternating rounds, 2026-09-26); the numbers and the rejected
//     alternatives are noted at each function. Summary: fp32 -> bf16 2.5x faster, bf16 ->
//     fp32 1.6x, the fp16 paths 1.5-21x (F16C), bf16 / fp16 operators 1.4-4x. Slower,
//     because the old code was wrong there: int32 -> bf16 (0.35 -> 0.46), int64 -> bf16
//     above 2^24 (0.69 -> 2.1), double -> fp16 on normal values (0.96 -> 1.11), and the
//     AVX2 bf16 store in Vectorized.h (0.11 -> 0.14, it now rounds instead of truncating).
// Checked on the CPU, bit for bit for every non-NaN input (a NaN input must give a NaN;
// the libraries differ in which NaN bits they return): all 2^32 fp32 inputs and all 2^16
// bf16/fp16 inputs against PyTorch (c10), Eigen and the FP16 library (the versions
// TensorFlow pins), NumPy, x86 F16C hardware, GCC _Float16 / __bf16 and CUDA's host code.
// ==================================================================================

// ---- bf16 -> fp32 ----
// bf16 is exactly the top 16 bits of an fp32, so widening is a 16-bit shift. It is
// exact for all 65,536 inputs, NaN payloads included.
// Modified: the old body had three branches (Inf / NaN / everything else) that all
// produced this same shift. The shift is kept on the GPU as well: it compiles to one
// SHF.L.U32 on every GPU (checked sm_80 to sm_120), and the PTX ISA documents
// cvt.f32.bf16 (what __bfloat162float uses from sm_90) as giving an "unspecified NaN"
// for a NaN input, while the shift keeps the bits.
__device__ __host__ inline float bfloat16_to_float(uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    ::std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---- fp32 -> bf16 ----
__device__ __host__ inline uint16_t float_to_bfloat16(float f) {
#if defined(__CUDA_ARCH__)
    // Modified: the GPU calls the intrinsic instead of the manual code below.
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
#else
    uint32_t u;
    ::std::memcpy(&u, &f, sizeof(u));
    // Modified: NaN used to come out as sign | 0x7FC1.
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return 0x7FFFu;
    }
    // Round to nearest even on the 16 dropped bits: add 0x7FFF plus the lowest kept
    // bit. A carry runs into the exponent by itself. A value at or above the bf16 maximum
    // plus half a step (fp32 bits 0x7F7F8000) lands exactly on Inf (0x7F80); anything
    // below that rounds to at most the maximum 0x7F7F. +-Inf stays +-Inf.
    // Modified: an `if (exponent > 0x8E) return +-Inf;` clamp used to sit here.
    // 0x8E = 127 + 15 is the fp16 exponent limit; bf16 has the fp32 exponent range and
    // needs no clamp. The clamp turned every finite |x| >= 65,536 into Inf:
    // 1,878,982,656 of the 2^32 fp32 inputs were wrong, 0 are now.
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
#endif
}

// ---- fp16 -> fp32 ----
// Always exact: every fp16 value is also an fp32 value.
__device__ __host__ inline float float16_to_float(uint16_t h) {
#if defined(__CUDA_ARCH__)
    // Modified: the GPU calls the intrinsic (one HADD2.F32 on every GPU from sm_75)
    // instead of the branching manual code below.
    return __half2float(__ushort_as_half(h));
#elif defined(__F16C__)
    // Modified: a CPU build with F16C (the library's uses -mf16c) does one VCVTPH2PS,
    // which is exact. Measured 0.25 ns per value for every input; the manual code below
    // takes 0.65 ns, and 5.0 ns for subnormal inputs (float-division branch).
    return _cvtsh_ss(h);
#else
    // Unchanged, and correct for all 65,536 inputs. The subnormal branch uses float
    // math, but frac / 1024 and ldexp(., -14) are exact and give normal fp32 results,
    // so the rounding mode and flush-to-zero cannot change the answer.
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
            f = ::std::ldexp(f, -14);
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
    ::std::memcpy(&f, &u, sizeof(f));
    return f;
#endif
}

// ---- fp32 -> fp16 ----
__device__ __host__ inline uint16_t float_to_float16(float f) {
#if defined(__CUDA_ARCH__)
    // Modified: the GPU calls the intrinsic (one F2F / F2FP instruction on every GPU
    // from sm_75) instead of the manual code below.
    return __half_as_ushort(__float2half_rn(f));
#elif defined(__F16C__)
    // Modified: a CPU build with F16C (the library's uses -mf16c) does one VCVTPS2PH:
    // round to nearest even with correct subnormals, the same bits as the manual code
    // below for every non-NaN input (checked over all 2^32). NaN is mapped to 0x7FFF
    // first, so the CPU returns the same NaN bits as the GPU and as the manual code.
    // Measured 0.38 ns per value; the manual code below takes 0.67 ns, and 4.0 ns when
    // the result is an fp16 subnormal.
    uint32_t x;
    ::std::memcpy(&x, &f, sizeof(x));
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {
        return 0x7FFFu;
    }
    return static_cast<uint16_t>(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT));
#else
    // Used when F16C is not available (e.g. the nvcc host pass, which is not given -mf16c).
    // Modified: rewritten. The old code was wrong for 100,815,872 of the 2^32 fp32
    // inputs, in three ways:
    //   1. results in the fp16 subnormal range were cut off instead of rounded
    //      (100,662,272 inputs; e.g. 2.98023259e-08 gave 0 instead of 2^-24);
    //   2. exact ties were rounded up instead of to the even neighbour (30,720);
    //   3. a rounding carry was joined to the exponent with `|` instead of `+`, so when
    //      the exponent was odd the carry was lost and the value HALVED (122,880;
    //      e.g. 0.00012204051 gave 0.000061035).
    // The normal-range step is the one Eigen uses (float_to_half_rtne); the subnormal
    // step rounds by value like NumPy (npy_floatbits_to_halfbits), with integers only.
    uint32_t x;
    ::std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax   = x & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) {
        return 0x7FFFu;                                    // NaN
    }
    if (ax >= 0x477FF000u) {
        return static_cast<uint16_t>(sign | 0x7C00u);      // >= 65520 (max + half a step), or Inf -> Inf
    }
    if (ax >= 0x38800000u) {                               // normal fp16 result (>= 2^-14)
        uint32_t r = ax - 0x38000000u;                     // exponent bias 127 -> 15
        r += 0x0FFFu + ((r >> 13) & 1u);                   // RNE on the 13 dropped bits; a carry ADDS into the exponent
        return static_cast<uint16_t>(sign | (r >> 13));
    }
    if (ax <= 0x33000000u) {
        return static_cast<uint16_t>(sign);                // <= 2^-25 (half the smallest subnormal) -> +-0
    }
    // Subnormal fp16 result: count in steps of 2^-24 and round by value.
    const uint32_t e     = ax >> 23;                       // 102..112
    const uint32_t m     = (ax & 0x007FFFFFu) | 0x00800000u;
    const uint32_t shift = 126u - e;                       // 14..24 bits dropped
    uint32_t q = m >> shift;
    const uint32_t rem  = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u))) {
        ++q;                                               // q may become 0x400 = smallest normal; still correct
    }
    return static_cast<uint16_t>(sign | q);
#endif
}

// ---- fp64 -> bf16 (new) ----
// One rounding, straight from the double. Going through float first with round-to-nearest
// rounds twice and can be one step off: 1 + 2^-8 + 2^-30 must give 0x3F81, but float rounds
// it onto the tie 1 + 2^-8, and the tie then goes to even, 0x3F80.
__device__ __host__ inline uint16_t double_to_bfloat16(double d) {
#if defined(__CUDA_ARCH__)
    // One cvt.rn.bf16.f64 (F2F.BF16.F64) from sm_90; NVIDIA's round-to-odd software
    // (still one correct rounding) on older GPUs.
    return __bfloat16_as_ushort(__double2bfloat16(d));
#else
    // Round to odd into float, then round to nearest even into bf16. Float keeps 16 more
    // bits than bf16 (at least 2 are needed), so the two steps give exactly one correct
    // rounding (Boldo & Melquiond, IEEE Trans. Computers 57(4), 2008). It is the method of
    // NVIDIA's own host code for __double2bfloat16 (cuda_bf16.hpp), and it has no branches.
    // The result does not depend on the FPU rounding mode (the step below always lands on
    // the truncated neighbour); it needs flush-to-zero off for results in the subnormal
    // range, which holds for the library's CPU code (not built with fast-math).
    // Measured 0.52 ns per value for every kind of input. Rejected: rounding directly
    // from the double bits with integers (0.7-3.9 ns, its branches stop vectorization).
    // The old two-rounding code took 0.50 ns.
    const float f = static_cast<float>(d);                 // one of the two float neighbours of d
    uint32_t u;
    ::std::memcpy(&u, &f, sizeof(u));
    const double back = static_cast<double>(f);
    u -= (::std::fabs(back) > ::std::fabs(d)) ? 1u : 0u;   // rounded away from zero: step to the truncated neighbour
    u |= (back != d) ? 1u : 0u;                            // inexact (or NaN): force the last bit to 1
    float g;
    ::std::memcpy(&g, &u, sizeof(g));
    return float_to_bfloat16(g);                           // NaN stays NaN -> 0x7FFF; +-Inf stays +-Inf
#endif
}

// ---- fp64 -> fp16 (new) ----
// One rounding, straight from the double (same reason as above:
// 1 + 2^-11 + 2^-40 must give 0x3C01, going through float gives 0x3C00).
__device__ __host__ inline uint16_t double_to_float16(double d) {
#if defined(__CUDA_ARCH__)
    // One cvt.rn.f16.f64 on every GPU.
    return __half_as_ushort(__double2half(d));
#else
    // Same round-to-odd method as double_to_bfloat16: float keeps 13 more bits than fp16.
    // Measured 1.1 ns per value for every kind of input (old two-rounding code: 0.96 ns on
    // normal values, 1.4-4.5 ns on large, tiny and random ones).
    const float f = static_cast<float>(d);
    uint32_t u;
    ::std::memcpy(&u, &f, sizeof(u));
    const double back = static_cast<double>(f);
    u -= (::std::fabs(back) > ::std::fabs(d)) ? 1u : 0u;
    u |= (back != d) ? 1u : 0u;
    float g;
    ::std::memcpy(&g, &u, sizeof(g));
    return float_to_float16(g);
#endif
}

// ---- 64-bit integer -> float, rounded to odd at 24 bits (new, CPU path) ----
// Keeps the top 24 significant bits of |v| and forces the last kept bit to 1 if any lower
// bit was 1 (round to odd). Every float step here is exact: the kept value is below 2^24
// and the scale is a power of two, so the result does not depend on the FPU rounding mode
// or on flush-to-zero. Rounding to odd with at least 2 extra bits, then to nearest even,
// equals one direct round to nearest even (Boldo & Melquiond, IEEE Trans. Computers 57(4),
// 2008); float keeps 24 bits and bf16 needs 8, so float_to_bfloat16 of this value is one
// correct rounding of v. |v| <= 2^24 comes out exact (nothing is cut off).
// Chosen over rounding to odd through double (measured on the CPU, see arithmetic_to_bfloat16).
template <typename I>
__device__ __host__ inline float int64_to_float_rto(I v) {
    static_assert(::std::is_integral_v<I> && sizeof(I) == 8, "int64_to_float_rto: 64-bit integers");
    // No data-dependent branch anywhere (a sign test mispredicts on mixed-sign data).
    uint64_t a = static_cast<uint64_t>(v);
    uint32_t sign = 0;
    if constexpr (::std::is_signed_v<I>) {
        const uint64_t m = static_cast<uint64_t>(static_cast<int64_t>(v) >> 63);   // all ones when v < 0
        a = (a ^ m) - m;                                   // |v|, correct for INT64_MIN too
        sign = static_cast<uint32_t>(m) & 0x80000000u;
    }
#if defined(__CUDA_ARCH__)
    const int bits = 64 - __clzll(static_cast<long long>(a | 1u));
#else
    const int bits = 64 - __builtin_clzll(a | 1u);         // significant bits (1 for a == 0)
#endif
    const int s = bits > 24 ? bits - 24 : 0;               // bits below the top 24 (0..40)
    const uint32_t sticky = (a & ((1ull << s) - 1u)) != 0u ? 1u : 0u;
    const uint32_t kept = static_cast<uint32_t>(a >> s) | sticky;   // < 2^24: exact in float
    const uint32_t scale_bits = (127u + static_cast<uint32_t>(s)) << 23;
    float scale;
    ::std::memcpy(&scale, &scale_bits, sizeof(scale));     // 2^s, exact
    const float f = static_cast<float>(kept) * scale;      // exact: at most 24 bits times a power of two
    uint32_t fb;
    ::std::memcpy(&fb, &f, sizeof(fb));
    fb |= sign;                                            // |f| never has the sign bit set
    float out;
    ::std::memcpy(&out, &fb, sizeof(out));
    return out;
}

// ---- any arithmetic type -> bf16 / fp16 with ONE rounding (new) ----
// Used by the templated constructors and operator= of bfloat16_t and float16_t
// (float itself has its own overload and does not come here).
// GPU: NVIDIA's intrinsics per type (one cvt from sm_90 for bf16 and on every GPU for
// fp16; NVIDIA's round-to-odd software for bf16 before sm_90).
// CPU: floating inputs go through double (exact for float and double). bf16: 8/16-bit
// integers go through float (exact); 32-bit ones through double (exact) and round to odd;
// 64-bit ones through int64_to_float_rto (exact integer round to odd into float). fp16: every integer
// goes through float (see arithmetic_to_float16). long double is first rounded to double;
// that is the one case with two roundings left (host only, not used by the library).
template <typename U>
__device__ __host__ inline uint16_t arithmetic_to_bfloat16(U v) {
    if constexpr (::std::is_floating_point_v<U>) {
        return double_to_bfloat16(static_cast<double>(v));
    } else {
        static_assert(::std::is_integral_v<U> && sizeof(U) <= 8, "arithmetic_to_bfloat16: integers up to 64 bits");
#if defined(__CUDA_ARCH__)
        if constexpr (sizeof(U) <= 4) {
            if constexpr (::std::is_signed_v<U>) {
                return __bfloat16_as_ushort(__int2bfloat16_rn(static_cast<int>(v)));
            } else {
                return __bfloat16_as_ushort(__uint2bfloat16_rn(static_cast<unsigned int>(v)));
            }
        } else {
            if constexpr (::std::is_signed_v<U>) {
                return __bfloat16_as_ushort(__ll2bfloat16_rn(static_cast<long long>(v)));
            } else {
                return __bfloat16_as_ushort(__ull2bfloat16_rn(static_cast<unsigned long long>(v)));
            }
        }
#else
        if constexpr (sizeof(U) <= 2) {
            return float_to_bfloat16(static_cast<float>(v));        // exact in float: one rounding
        } else if constexpr (sizeof(U) == 4) {
            // Exact in double, then round to odd into float (branch-free, so loops vectorize).
            // Measured 0.46 ns per value. Rejected: a |v| <= 2^24 float fast path (0.58 ns
            // small, 1.30 ns large: the branch stops vectorization). Old code: 0.35 ns, wrong
            // for every |v| >= 65,536.
            return double_to_bfloat16(static_cast<double>(v));
        } else {
            // 64-bit. |v| <= 2^24 (typical data: indices, counts) is exact in float, so a plain
            // conversion is one rounding and the fastest; the test is one unsigned compare,
            // and it predicts well on real data. Larger values take the exact integer round to
            // odd into float. Both steps end in one rounding into bf16.
            // Measured 0.82 ns per value for |v| < 256 and 2.1 ns above 2^24. Rejected: round
            // to odd through double (3.6-4.4 ns); the same round to odd without this fast path
            // (2.1 ns for every value); a sign branch instead of the bit trick in
            // int64_to_float_rto (3.7 ns, it mispredicts on mixed signs). Old code: 0.69-0.97
            // ns, wrong for every |v| >= 65,536.
            bool fits24;
            if constexpr (::std::is_signed_v<U>) {
                fits24 = static_cast<uint64_t>(v) + (1ull << 24) <= (1ull << 25);   // -2^24 <= v <= 2^24
            } else {
                fits24 = v <= (1ull << 24);
            }
            return float_to_bfloat16(fits24 ? static_cast<float>(v) : int64_to_float_rto(v));
        }
#endif
    }
}

template <typename U>
__device__ __host__ inline uint16_t arithmetic_to_float16(U v) {
    if constexpr (::std::is_floating_point_v<U>) {
        return double_to_float16(static_cast<double>(v));
    } else {
        static_assert(::std::is_integral_v<U> && sizeof(U) <= 8, "arithmetic_to_float16: integers up to 64 bits");
#if defined(__CUDA_ARCH__)
        if constexpr (sizeof(U) <= 4) {
            if constexpr (::std::is_signed_v<U>) {
                return __half_as_ushort(__int2half_rn(static_cast<int>(v)));
            } else {
                return __half_as_ushort(__uint2half_rn(static_cast<unsigned int>(v)));
            }
        } else {
            if constexpr (::std::is_signed_v<U>) {
                return __half_as_ushort(__ll2half_rn(static_cast<long long>(v)));
            } else {
                return __half_as_ushort(__ull2half_rn(static_cast<unsigned long long>(v)));
            }
        }
#else
        // Every integer can go through float for fp16: |v| < 2^24 is exact in float, and any
        // |v| >= 65520 gives +-Inf both directly and through float (float rounding is
        // monotonic and float(v) >= 65520 there), so this is always one rounding. NVIDIA's
        // host code for __int2half_rn / __ll2half_rn makes the same argument.
        // Measured 0.45 ns per value (int32); going through double took 1.24 ns; old code
        // 0.68-0.93 ns.
        return float_to_float16(static_cast<float>(v));
#endif
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
    // Modified: double and integer inputs now convert with one rounding. They used to
    // go through static_cast<float> first, which rounds twice (1 + 2^-8 + 2^-30 gave
    // 0x3F80 instead of 0x3F81; the integer 2^24 + 2^16 + 1 gave 0x4B80 instead of 0x4B81).
    // Together with the old clamp in float_to_bfloat16, every int32 with |x| >= 65,536
    // came out as Inf (4,294,836,225 of the 2^32 int32 values were wrong; 0 are now).
    __device__ __host__ explicit bfloat16_t(U val) {
        raw_bits = detail::arithmetic_to_bfloat16(val);
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
    // Modified: one rounding for double and integer inputs (see the constructor above).
    __device__ __host__ bfloat16_t& operator=(U val) {
        raw_bits = detail::arithmetic_to_bfloat16(val);
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
    // Modified: double inputs now convert with one rounding. They used to go through
    // static_cast<float> first, which rounds twice (1 + 2^-11 + 2^-40 gave 0x3C00 instead
    // of 0x3C01). For integers, going through float was already a single rounding (every
    // integer below the fp16 overflow point is exact in float); they were wrong only
    // through the old float_to_float16 rounding bug (5,140 of the 2^32 int32 values, the
    // first being 2049). On the GPU they now take NVIDIA's integer intrinsics.
    __device__ __host__ explicit float16_t(U val) {
        raw_bits = detail::arithmetic_to_float16(val);
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
    // Modified: one rounding for double inputs (see the constructor above).
    __device__ __host__ float16_t& operator=(U val) {
        raw_bits = detail::arithmetic_to_float16(val);
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

    namespace detail {
    // New (2026-09-26): scalar -> float16_t for complex32_t with one rounding.
    // Arithmetic types (double, int, ...) use float16_t's templated constructor, which
    // rounds once. Other types that convert to float (e.g. bfloat16_t) are exact in
    // float, so going through float is still one rounding.
    template <typename T>
    __device__ __host__ inline float16_t scalar_to_float16(const T& v) {
        if constexpr (::std::is_arithmetic_v<T>) {
            return float16_t(v);
        } else {
            return float16_t(static_cast<float>(v));
        }
    }
    } // namespace detail

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
        // Modified: the double parts convert with one rounding (they went through
        // static_cast<float> first, which rounds twice).
        __host__ explicit complex32_t(const std::complex<double>& c)
            : real_(float16_t(c.real())),
              imag_(float16_t(c.imag())) {}
    
        // Conversion from other custom complex types (defined out-of-line after complex64_t/complex128_t)
        __device__ __host__ explicit complex32_t(const struct complex64_t& c);
        __device__ __host__ explicit complex32_t(const struct complex128_t& c);

        // Generic scalar constructors for any type convertible to float (resolves ambiguity)
        template <typename T, typename = std::enable_if_t<std::is_convertible_v<T, float> && !std::is_same_v<std::decay_t<T>, float16_t> && !std::is_same_v<std::decay_t<T>, complex32_t>>>
        __device__ __host__ explicit complex32_t(T r)
            // Modified: one rounding for double/integer r (was static_cast<float> first).
            : real_(detail::scalar_to_float16(r)), imag_(float16_t(0.0f)) {}

        template <typename T, typename U, typename = std::enable_if_t<std::is_convertible_v<T, float> && std::is_convertible_v<U, float> && !(std::is_same_v<std::decay_t<T>, float16_t> && std::is_same_v<std::decay_t<U>, float16_t>)>>
        __device__ __host__ explicit complex32_t(T r, U i)
            // Modified: one rounding for double/integer r and i (was static_cast<float> first).
            : real_(detail::scalar_to_float16(r)), imag_(detail::scalar_to_float16(i)) {}

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

// Modified: the double parts convert with one rounding (they went through
// static_cast<float> first, which rounds twice).
inline __device__ __host__ complex32_t::complex32_t(const complex128_t& c)
    : real_(float16_t(c.real())), imag_(float16_t(c.imag())) {}

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
