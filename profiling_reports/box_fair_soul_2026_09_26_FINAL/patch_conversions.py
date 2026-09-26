#!/usr/bin/env python3
"""Fix the bf16/fp16 conversion bugs in Tensor-Implementations (2026-09-26).

Run from the Tensor-Implementations root:   python3 patch_conversions.py
Every edit must match exactly once; nothing is written unless ALL edits match.
A second run fails on purpose (the old text is gone).

Files:
  include/dtype/Types.h               conversion functions, struct constructors, complex32_t
  include/ops/helpers/Vectorized.h    AVX2 float -> bf16 store (truncated, now rounds)
  src/UnaryOps/cpu/Activations.cpp    3 scalar float -> bf16 stores (truncated, now round)
  include/ops/helpers/ReductionImpl.h 7 CPU mean outputs: double -> float -> 16 bit (two roundings) -> one

Types.h is patched the same way whether or not the clamp line was already removed by
hand: the whole `namespace detail { ... }` block is replaced.
"""
import sys

TYPES = "include/dtype/Types.h"
VEC = "include/ops/helpers/Vectorized.h"
ACT = "src/UnaryOps/cpu/Activations.cpp"
RED = "include/ops/helpers/ReductionImpl.h"


def replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT (nothing written): {what}: expected 1 match, found {n}")
    return text.replace(old, new)


# ----------------------------------------------------------------------------------
# Types.h
# ----------------------------------------------------------------------------------
INCLUDE_OLD = "#include <complex>\n"
INCLUDE_NEW = """#include <complex>
// nvcc only: the GPU paths of the conversions below call NVIDIA's intrinsics
// (__float2bfloat16_rn, __float2half_rn, __half2float, __double2bfloat16, ...).
// A plain host compiler (g++) never sees these headers and uses the manual code.
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
"""

DETAIL_NEW = r"""namespace detail {

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
// CPU (nvcc host pass, or a plain host compiler): the manual code below. It uses
//   integer operations for every step that rounds, so it does not depend on the FPU
//   rounding mode or on flush-to-zero.
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
//   - double_to_bfloat16, double_to_float16, integer_to_double_rto,
//     arithmetic_to_bfloat16, arithmetic_to_float16: new. The struct constructors used
//     to send double and integer inputs through float first, which rounds twice (bf16:
//     double and integers above 2^24; fp16: double).
//   - GPU paths now call the intrinsics instead of the manual code. Compiled for
//     sm_90a (H100/H200), one fp32 -> bf16 conversion went from 17 instructions with
//     a branch to 1.
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
#else
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
// One rounding, straight from the double. Going through float first rounds twice and
// can be one step off: 1 + 2^-8 + 2^-30 must give 0x3F81, but float rounds it onto the
// tie 1 + 2^-8, and the tie then goes to even, 0x3F80.
__device__ __host__ inline uint16_t double_to_bfloat16(double d) {
#if defined(__CUDA_ARCH__)
    // One cvt.rn.bf16.f64 (F2F.BF16.F64) from sm_90; NVIDIA's round-to-odd software
    // (still one correct rounding) on older GPUs.
    return __bfloat16_as_ushort(__double2bfloat16(d));
#else
    uint64_t x;
    ::std::memcpy(&x, &d, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 48) & 0x8000u);
    const uint64_t ax   = x & 0x7FFFFFFFFFFFFFFFull;
    if (ax > 0x7FF0000000000000ull) {
        return 0x7FFFu;                                    // NaN
    }
    if (ax >= 0x47EFF00000000000ull) {
        return static_cast<uint16_t>(sign | 0x7F80u);      // >= bf16 max + half a step, or Inf -> Inf
    }
    if (ax >= 0x3810000000000000ull) {                     // normal bf16 result (>= 2^-126)
        uint64_t r = ax - 0x3800000000000000ull;           // exponent bias 1023 -> 127
        r += 0x00000FFFFFFFFFFFull + ((r >> 45) & 1u);     // RNE on the 45 dropped bits
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(r >> 45));
    }
    if (ax <= 0x3790000000000000ull) {
        return sign;                                       // <= 2^-134 (half the smallest subnormal) -> +-0
    }
    // Subnormal bf16 result: count in steps of 2^-133 and round by value.
    const uint32_t e     = static_cast<uint32_t>(ax >> 52);   // 889..896
    const uint64_t m     = (ax & 0x000FFFFFFFFFFFFFull) | 0x0010000000000000ull;
    const uint32_t shift = 942u - e;                           // 46..53 bits dropped
    uint64_t q = m >> shift;
    const uint64_t rem  = m & ((1ull << shift) - 1u);
    const uint64_t half = 1ull << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u))) {
        ++q;                                                   // may become 0x80 = smallest normal; still correct
    }
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(q));
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
    uint64_t x;
    ::std::memcpy(&x, &d, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 48) & 0x8000u);
    const uint64_t ax   = x & 0x7FFFFFFFFFFFFFFFull;
    if (ax > 0x7FF0000000000000ull) {
        return 0x7FFFu;                                    // NaN
    }
    if (ax >= 0x40EFFE0000000000ull) {
        return static_cast<uint16_t>(sign | 0x7C00u);      // >= 65520 (max + half a step), or Inf -> Inf
    }
    if (ax >= 0x3F10000000000000ull) {                     // normal fp16 result (>= 2^-14)
        uint64_t r = ax - 0x3F00000000000000ull;           // exponent bias 1023 -> 15
        r += 0x000001FFFFFFFFFFull + ((r >> 42) & 1u);     // RNE on the 42 dropped bits
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(r >> 42));
    }
    if (ax <= 0x3E60000000000000ull) {
        return sign;                                       // <= 2^-25 (half the smallest subnormal) -> +-0
    }
    // Subnormal fp16 result: count in steps of 2^-24 and round by value.
    const uint32_t e     = static_cast<uint32_t>(ax >> 52);   // 998..1008
    const uint64_t m     = (ax & 0x000FFFFFFFFFFFFFull) | 0x0010000000000000ull;
    const uint32_t shift = 1051u - e;                          // 43..53 bits dropped
    uint64_t q = m >> shift;
    const uint64_t rem  = m & ((1ull << shift) - 1u);
    const uint64_t half = 1ull << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u))) {
        ++q;                                                   // may become 0x400 = smallest normal; still correct
    }
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(q));
#endif
}

// ---- integer -> double, prepared for ONE later rounding (new, CPU path) ----
// Integers below 2^53 in magnitude convert to double exactly. Larger 64-bit values are
// rounded to odd: keep the top 53 bits and force the last kept bit to 1 if any dropped
// bit was 1. Rounding to odd with at least 2 extra bits, then to nearest even, equals
// one direct round to nearest even (Boldo & Melquiond, IEEE Trans. Computers 57(4),
// 2008). Here the double keeps 45 (bf16) or 42 (fp16) extra bits. NVIDIA's host code
// for __ll2bfloat16_rn uses the same idea with float as the middle format.
template <typename I>
__device__ __host__ inline double integer_to_double_rto(I v) {
    static_assert(::std::is_integral_v<I> && sizeof(I) <= 8, "integer_to_double_rto: integers up to 64 bits");
    if constexpr (sizeof(I) < 8) {
        return static_cast<double>(v);                     // |v| < 2^32: exact
    } else {
        bool neg = false;
        uint64_t a = static_cast<uint64_t>(v);
        if constexpr (::std::is_signed_v<I>) {
            if (v < 0) {
                neg = true;
                a = 0ull - a;                              // |v|, correct for INT64_MIN too
            }
        }
        if (a >= (1ull << 53)) {
            int s = 0;
            while ((a >> s) >= (1ull << 53)) ++s;          // 1..11
            const uint64_t sticky = (a & ((1ull << s) - 1u)) != 0u ? 1u : 0u;
            a = ((a >> s) | sticky) << s;                  // round to odd at 53 bits; exact in double
        }
        const double d = static_cast<double>(a);           // exact: at most 53 significant bits
        return neg ? -d : d;
    }
}

// ---- any arithmetic type -> bf16 / fp16 with ONE rounding (new) ----
// Used by the templated constructors and operator= of bfloat16_t and float16_t
// (float itself has its own overload and does not come here).
// GPU: NVIDIA's intrinsics per type (one cvt from sm_90 for bf16 and on every GPU for
// fp16; NVIDIA's round-to-odd software for bf16 before sm_90).
// CPU: floating inputs go through double (exact for float and double); integers go
// through integer_to_double_rto. long double is first rounded to double; that is the
// one case with two roundings left (host only, not used by the library).
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
        return double_to_bfloat16(integer_to_double_rto(v));
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
        return double_to_float16(integer_to_double_rto(v));
#endif
    }
}

} // namespace detail
"""

BF16_CTOR_OLD = """    __device__ __host__ explicit bfloat16_t(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
    }"""
BF16_CTOR_NEW = """    // Modified: double and integer inputs now convert with one rounding. They used to
    // go through static_cast<float> first, which rounds twice (1 + 2^-8 + 2^-30 gave
    // 0x3F80 instead of 0x3F81; the integer 2^24 + 2^16 + 1 gave 0x4B80 instead of 0x4B81).
    // Together with the old clamp in float_to_bfloat16, every int32 with |x| >= 65,536
    // came out as Inf (4,294,836,225 of the 2^32 int32 values were wrong; 0 are now).
    __device__ __host__ explicit bfloat16_t(U val) {
        raw_bits = detail::arithmetic_to_bfloat16(val);
    }"""
BF16_ASSIGN_OLD = """    __device__ __host__ bfloat16_t& operator=(U val) {
        raw_bits = detail::float_to_bfloat16(static_cast<float>(val));
        return *this;
    }"""
BF16_ASSIGN_NEW = """    // Modified: one rounding for double and integer inputs (see the constructor above).
    __device__ __host__ bfloat16_t& operator=(U val) {
        raw_bits = detail::arithmetic_to_bfloat16(val);
        return *this;
    }"""
FP16_CTOR_OLD = """    __device__ __host__ explicit float16_t(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
    }"""
FP16_CTOR_NEW = """    // Modified: double inputs now convert with one rounding. They used to go through
    // static_cast<float> first, which rounds twice (1 + 2^-11 + 2^-40 gave 0x3C00 instead
    // of 0x3C01). For integers, going through float was already a single rounding (every
    // integer below the fp16 overflow point is exact in float); they were wrong only
    // through the old float_to_float16 rounding bug (5,140 of the 2^32 int32 values, the
    // first being 2049). On the GPU they now take NVIDIA's integer intrinsics.
    __device__ __host__ explicit float16_t(U val) {
        raw_bits = detail::arithmetic_to_float16(val);
    }"""
FP16_ASSIGN_OLD = """    __device__ __host__ float16_t& operator=(U val) {
        raw_bits = detail::float_to_float16(static_cast<float>(val));
        return *this;
    }"""
FP16_ASSIGN_NEW = """    // Modified: one rounding for double inputs (see the constructor above).
    __device__ __host__ float16_t& operator=(U val) {
        raw_bits = detail::arithmetic_to_float16(val);
        return *this;
    }"""

HELPER_ANCHOR = """    /**
     * @brief Complex32 (Half-Precision Complex Number)"""
HELPER_NEW = """    namespace detail {
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

""" + HELPER_ANCHOR

# The original lines end in a trailing space; spelled out so no editor strips them.
C32_STDDOUBLE_OLD = ("        __host__ explicit complex32_t(const std::complex<double>& c) \n"
                     "            : real_(float16_t(static_cast<float>(c.real()))), \n"
                     "              imag_(float16_t(static_cast<float>(c.imag()))) {}")
C32_STDDOUBLE_NEW = """        // Modified: the double parts convert with one rounding (they went through
        // static_cast<float> first, which rounds twice).
        __host__ explicit complex32_t(const std::complex<double>& c)
            : real_(float16_t(c.real())),
              imag_(float16_t(c.imag())) {}"""
C32_T1_OLD = """            : real_(float16_t(static_cast<float>(r))), imag_(float16_t(0.0f)) {}"""
C32_T1_NEW = """            // Modified: one rounding for double/integer r (was static_cast<float> first).
            : real_(detail::scalar_to_float16(r)), imag_(float16_t(0.0f)) {}"""
C32_T2_OLD = """            : real_(float16_t(static_cast<float>(r))), imag_(float16_t(static_cast<float>(i))) {}"""
C32_T2_NEW = """            // Modified: one rounding for double/integer r and i (was static_cast<float> first).
            : real_(detail::scalar_to_float16(r)), imag_(detail::scalar_to_float16(i)) {}"""
C32_C128_OLD = """inline __device__ __host__ complex32_t::complex32_t(const complex128_t& c)
    : real_(float16_t(static_cast<float>(c.real()))), imag_(float16_t(static_cast<float>(c.imag()))) {}"""
C32_C128_NEW = """// Modified: the double parts convert with one rounding (they went through
// static_cast<float> first, which rounds twice).
inline __device__ __host__ complex32_t::complex32_t(const complex128_t& c)
    : real_(float16_t(c.real())), imag_(float16_t(c.imag())) {}"""

# ----------------------------------------------------------------------------------
# Vectorized.h
# ----------------------------------------------------------------------------------
VEC_HDR_OLD = """// Uses F16C hardware instructions (_mm256_cvtph_ps / _mm256_cvtps_ph)
// for fp16, and bit-shift for bf16."""
VEC_HDR_NEW = """// Uses F16C hardware instructions (_mm256_cvtph_ps / _mm256_cvtps_ph)
// for fp16. bf16 loads are a bit-shift (exact); bf16 stores round to nearest even
// (store_float_as_bf16 below; modified 2026-09-26, it used to truncate)."""

VEC_OLD = """// Store 8 x float → 8 x bf16 (truncate lower 16 bits)
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
}"""
VEC_NEW = """// Store 8 x float → 8 x bf16, round to nearest even (IEEE 754 default).
// Modified 2026-09-26: this used to cut the low 16 bits off (round toward zero). That
// made results up to one bf16 step too small in magnitude, and turned NaNs whose
// payload sat only in the low 16 bits into Inf (0x7F800001 -> 0x7F80). It now does
// what the scalar OwnTensor::detail::float_to_bfloat16 does, lane by lane: add
// 0x7FFF plus the lowest kept bit, shift right by 16 (a carry runs into the exponent,
// overflow lands on Inf), and NaN lanes become 0x7FFF. The NaN test is an integer
// compare, so it cannot be optimised away by fast-math flags. The rounding is the same
// as PyTorch's AVX2 cvtfp32_bf16 (aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h);
// the NaN handling differs (PyTorch uses a float compare and writes 0xFFFF; here an
// integer compare writes 0x7FFF, the same value as the scalar function).
// Checked bit-identical to the scalar function for all 2^32 fp32 inputs.
inline void store_float_as_bf16(void* ptr, const Vectorized<float>& v) {
    const __m256i bits = _mm256_castps_si256(v.values);
    const __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    __m256i rounded = _mm256_add_epi32(bits, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7FFF)));
    rounded = _mm256_srli_epi32(rounded, 16);
    // NaN lanes: |bits| > 0x7F800000. After the mask both sides are non-negative, so the
    // signed 32-bit compare is exact.
    const __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i is_nan   = _mm256_cmpgt_epi32(abs_bits, _mm256_set1_epi32(0x7F800000));
    rounded = _mm256_blendv_epi8(rounded, _mm256_set1_epi32(0x7FFF), is_nan);
    // Every lane now holds a value <= 0xFFFF, so the unsigned-saturating pack keeps it.
    __m128i lo = _mm256_castsi256_si128(rounded);
    __m128i hi = _mm256_extracti128_si256(rounded, 1);
    __m128i packed = _mm_packus_epi32(lo, hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), packed);
}"""

# ----------------------------------------------------------------------------------
# Activations.cpp (scalar tails of GELU forward, bias+GELU forward, GELU backward)
# ----------------------------------------------------------------------------------
ACT_A_OLD = """                    uint32_t bits; std::memcpy(&bits, &result, 4);
                    uint16_t out = static_cast<uint16_t>(bits >> 16);"""
ACT_A_NEW = """                    // Modified 2026-09-26: round to nearest even through the shared scalar
                    // converter. This used to be a truncating `bits >> 16`, unlike the fp16
                    // branch above (which rounds) and the GPU kernels (which round).
                    uint16_t out = ::OwnTensor::detail::float_to_bfloat16(result);"""
ACT_B_OLD = """                    uint32_t bits; std::memcpy(&bits, &result, 4);
                    uint16_t out = (uint16_t)(bits >> 16);"""
ACT_B_NEW = ACT_A_NEW


# ----------------------------------------------------------------------------------
# ReductionImpl.h (CPU mean): the mean is computed in double; bf16/fp16 outputs went
# through float first (two roundings). The 16-bit constructors now take the double.
# ----------------------------------------------------------------------------------
RED_WHY = """// Modified 2026-09-26: one rounding for bf16/fp16 as well. This used to be
{i}// static_cast<T>(static_cast<float>(...)), i.e. double -> float -> 16 bit, which rounds
{i}// twice; the bf16/fp16 constructors now take the double directly and round once."""

def red_block(indent, lhs, var):
    i = " " * indent
    old = (f"{i}if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>)\n"
           f"{i}    {lhs} = static_cast<T>(static_cast<float>({var}));\n"
           f"{i}else\n"
           f"{i}    {lhs} = static_cast<T>({var});")
    new = f"{i}" + RED_WHY.format(i=i) + f"\n{i}{lhs} = static_cast<T>({var});"
    return old, new

RED_A_OLD, RED_A_NEW = red_block(20, "out_data[o]", "mv")        # 2 sites (per-slot mean, two strategies)
RED_B_OLD, RED_B_NEW = red_block(16, "out_data[0]", "mv")        # full reduction
RED_C_OLD, RED_C_NEW = red_block(12, "out_data[0]", "mv")        # full reduction, threaded
RED_D_OLD = """                        out_data[o] = c > 0.0 ? static_cast<T>(static_cast<float>(s / c)) : static_cast<T>(std::nanf(""));"""
RED_D_NEW = ("                        " + RED_WHY.format(i="                        ") + "\n"
             """                        out_data[o] = c > 0.0 ? static_cast<T>(s / c) : static_cast<T>(std::nanf(""));""")
RED_E_OLD = """            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>)
                out_data[o] = static_cast<T>(static_cast<float>(mean_val));"""
RED_E_NEW = """            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>)
                // Modified 2026-09-26: one rounding (was static_cast<float> first, which
                // rounds twice; the bf16/fp16 constructors now take the double directly).
                out_data[o] = static_cast<T>(mean_val);"""
RED_F_OLD = """                out_data[o] = T((float)(csum.real() / (double)reduced_count), (float)(csum.imag() / (double)reduced_count));"""
RED_F_NEW = """                // Modified 2026-09-26: the double parts go straight to T. The (float) casts made
                // complex32 round twice (double -> float -> fp16) and cut complex128 down to
                // float precision. complex64 still rounds once to float, as before.
                out_data[o] = T(csum.real() / (double)reduced_count, csum.imag() / (double)reduced_count);"""


def main():
    types = open(TYPES).read()
    vec = open(VEC).read()
    act = open(ACT).read()
    red = open(RED).read()

    # --- Types.h ---
    types = replace_once(types, INCLUDE_OLD, INCLUDE_NEW, "Types.h #include <complex>")
    start_tok, end_tok = "namespace detail {\n", "} // namespace detail\n"
    if types.count(start_tok) != 1 or types.count(end_tok) != 1:
        sys.exit(f"ABORT (nothing written): Types.h detail block markers: "
                 f"{types.count(start_tok)} starts, {types.count(end_tok)} ends (expected 1 and 1)")
    s = types.index(start_tok)
    e = types.index(end_tok) + len(end_tok)
    block = types[s:e]
    # Count only live definitions (a line that starts with the signature), not the
    # commented-out old copy that also sits in this block.
    live = [l for l in block.splitlines() if l.startswith("__device__ __host__ inline")]
    for sig in ("inline float bfloat16_to_float(uint16_t b)", "inline uint16_t float_to_bfloat16(float f)",
                "inline float float16_to_float(uint16_t h)", "inline uint16_t float_to_float16(float f)"):
        n = sum(sig in l for l in live)
        if n != 1:
            sys.exit(f"ABORT (nothing written): Types.h detail block: live '{sig}' found {n} times")
    if len(live) != 4:
        sys.exit(f"ABORT (nothing written): Types.h detail block has {len(live)} live functions, expected 4")
    if "double_to_bfloat16" in block or "arithmetic_to_bfloat16" in block:
        sys.exit("ABORT (nothing written): Types.h already patched")
    types = types[:s] + DETAIL_NEW + types[e:]
    types = replace_once(types, BF16_CTOR_OLD, BF16_CTOR_NEW, "bfloat16_t templated constructor")
    types = replace_once(types, BF16_ASSIGN_OLD, BF16_ASSIGN_NEW, "bfloat16_t templated operator=")
    types = replace_once(types, FP16_CTOR_OLD, FP16_CTOR_NEW, "float16_t templated constructor")
    types = replace_once(types, FP16_ASSIGN_OLD, FP16_ASSIGN_NEW, "float16_t templated operator=")
    types = replace_once(types, HELPER_ANCHOR, HELPER_NEW, "complex32_t doc comment (helper anchor)")
    types = replace_once(types, C32_STDDOUBLE_OLD, C32_STDDOUBLE_NEW, "complex32_t(std::complex<double>)")
    types = replace_once(types, C32_T1_OLD, C32_T1_NEW, "complex32_t(T r)")
    types = replace_once(types, C32_T2_OLD, C32_T2_NEW, "complex32_t(T r, U i)")
    types = replace_once(types, C32_C128_OLD, C32_C128_NEW, "complex32_t(const complex128_t&)")

    # --- Vectorized.h ---
    vec = replace_once(vec, VEC_HDR_OLD, VEC_HDR_NEW, "Vectorized.h fp16/bf16 helper section comment")
    vec = replace_once(vec, VEC_OLD, VEC_NEW, "Vectorized.h store_float_as_bf16")

    # --- Activations.cpp ---
    if act.count(ACT_A_OLD) != 2:
        sys.exit(f"ABORT (nothing written): Activations.cpp static_cast truncation: expected 2, found {act.count(ACT_A_OLD)}")
    act = act.replace(ACT_A_OLD, ACT_A_NEW)
    act = replace_once(act, ACT_B_OLD, ACT_B_NEW, "Activations.cpp (uint16_t) truncation")
    # No code line (comments excluded) may still build bf16 bits with a truncating shift.
    left = [i + 1 for i, l in enumerate(act.splitlines())
            if ">> 16" in l and not l.lstrip().startswith("//")]
    if left:
        sys.exit(f"ABORT (nothing written): Activations.cpp still has '>> 16' at lines {left}")

    # --- ReductionImpl.h ---
    if red.count(RED_A_OLD) != 2:
        sys.exit(f"ABORT (nothing written): ReductionImpl.h per-slot mean block: expected 2, found {red.count(RED_A_OLD)}")
    red = red.replace(RED_A_OLD, RED_A_NEW)
    red = replace_once(red, RED_B_OLD, RED_B_NEW, "ReductionImpl.h full-reduction mean")
    red = replace_once(red, RED_C_OLD, RED_C_NEW, "ReductionImpl.h threaded full-reduction mean")
    red = replace_once(red, RED_D_OLD, RED_D_NEW, "ReductionImpl.h bf16/fp16 vector-path tail mean")
    red = replace_once(red, RED_E_OLD, RED_E_NEW, "ReductionImpl.h mean_val (16-bit branch)")
    red = replace_once(red, RED_F_OLD, RED_F_NEW, "ReductionImpl.h complex mean")

    open(TYPES, "w").write(types)
    open(VEC, "w").write(vec)
    open(ACT, "w").write(act)
    open(RED, "w").write(red)
    print("patched: include/dtype/Types.h (include, detail block, 4 struct templates, complex32_t x4 + helper)")
    print("patched: include/ops/helpers/Vectorized.h (store_float_as_bf16)")
    print("patched: src/UnaryOps/cpu/Activations.cpp (3 scalar bf16 stores)")
    print("patched: include/ops/helpers/ReductionImpl.h (7 CPU mean outputs, one rounding)")
    print("next: `git diff --stat`, then a CLEAN rebuild (make does not track header changes)")


if __name__ == "__main__":
    main()
