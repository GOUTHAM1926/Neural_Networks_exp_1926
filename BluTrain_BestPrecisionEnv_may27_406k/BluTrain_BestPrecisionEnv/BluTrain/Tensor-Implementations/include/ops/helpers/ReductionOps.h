// include/ops/helpers/ReductionOps.h - FIXED: Uses GPU intrinsics
#pragma once

#ifndef OWNTENSOR_REDUCTION_OPS_H
#define OWNTENSOR_REDUCTION_OPS_H

// ═══════════════════════════════════════════════════════════
// COMPILATION CONTEXT SETUP
// ═══════════════════════════════════════════════════════════

#ifdef __CUDACC__
// GPU COMPILATION (nvcc)
#define DEVICE_HOST __device__ __host__
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif
#ifndef CUDART_INF
#define CUDART_INF __longlong_as_double(0x7ff0000000000000LL)
#endif
#else
// CPU COMPILATION (g++)
#define DEVICE_HOST
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F __builtin_huge_valf()
#endif
#ifndef CUDART_INF
#define CUDART_INF __builtin_huge_val()
#endif
#endif

#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace OwnTensor {
namespace detail {

// ═══════════════════════════════════════════════════════════
// GPU INTRINSIC HELPERS (FORWARD DECLARATIONS)
// ═══════════════════════════════════════════════════════════

#ifdef __CUDA_ARCH__
//  GPU device code - use intrinsics
template <typename T> __device__ inline T gpu_add(T a, T b) { return a + b; }
template <> __device__ inline __half gpu_add(__half a, __half b) {
  return __hadd(a, b);
}
template <>
__device__ inline __nv_bfloat16 gpu_add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hadd(a, b);
}

template <typename T> __device__ inline T gpu_mul(T a, T b) { return a * b; }
template <> __device__ inline __half gpu_mul(__half a, __half b) {
  return __hmul(a, b);
}
template <>
__device__ inline __nv_bfloat16 gpu_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hmul(a, b);
}

template <typename T> __device__ inline bool gpu_lt(T a, T b) { return a < b; }
template <> __device__ inline bool gpu_lt(__half a, __half b) {
  return __hlt(a, b);
}
template <> __device__ inline bool gpu_lt(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hlt(a, b);
}

template <typename T> __device__ inline bool gpu_gt(T a, T b) { return a > b; }
template <> __device__ inline bool gpu_gt(__half a, __half b) {
  return __hgt(a, b);
}
template <> __device__ inline bool gpu_gt(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hgt(a, b);
}

template <typename T> __device__ inline bool gpu_isnan(T val) {
  if constexpr (std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t>) {
    return isnan(val); // Use OwnTensor::isnan for complex types
  } else {
    return std::isnan(val); // Use std::isnan for standard types
  }
}
template <> __device__ inline bool gpu_isnan(__half val) {
  return __hisnan(val);
}
template <> __device__ inline bool gpu_isnan(__nv_bfloat16 val) {
  return __hisnan(val);
}

// #else
// //  CPU host code - use regular operations
// template<typename T> inline T gpu_add(T a, T b) { return a + b; }
// template<typename T> inline T gpu_mul(T a, T b) { return a * b; }
// template<typename T> inline bool gpu_lt(T a, T b) { return a < b; }
// template<typename T> inline bool gpu_gt(T a, T b) { return a > b; }
// template<typename T> inline bool gpu_isnan(T val) {
//     if constexpr (std::is_floating_point_v<T>) {
//         return std::isnan(val);
//     }
//     return false;
// }
#endif

// ═══════════════════════════════════════════════════════════
// HELPER TRAITS
// ═══════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_half_float_v =
    std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>;

#ifdef __CUDACC__
template <typename T>
constexpr bool is_native_half_v =
    std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
#else
template <typename T> constexpr bool is_native_half_v = false;
#endif

template <typename T>
constexpr bool is_any_float_v =
    std::is_floating_point_v<T> || is_half_float_v<T> || is_native_half_v<T>;

// ═══════════════════════════════════════════════════════════
// VALUE-INDEX PAIR FOR ARG REDUCTIONS
// ═══════════════════════════════════════════════════════════

template <typename T> struct ValueIndex {
  T value;
  int64_t index;

  DEVICE_HOST ValueIndex() : value(T{}), index(-1) {}
  DEVICE_HOST ValueIndex(T val, int64_t idx) : value(val), index(idx) {}

  DEVICE_HOST bool operator>(const ValueIndex<T> &other) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return false; // Should never be called due to dispatcher checks
    } else {
#ifdef __CUDA_ARCH__
      return gpu_gt(value, other.value);
#else
      return value > other.value;
#endif
    }
  }

  DEVICE_HOST bool operator<(const ValueIndex<T> &other) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return false; // Should never be called due to dispatcher checks
    } else {
#ifdef __CUDA_ARCH__
      return gpu_lt(value, other.value);
#else
      return value < other.value;
#endif
    }
  }
};

// ═══════════════════════════════════════════════════════════
// DEVICE-SAFE HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

template <typename T> DEVICE_HOST constexpr T get_lowest_value() {
  if constexpr (std::is_same_v<T, float16_t>) {
    return T(-65504.0f);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return T(-3.38953e38f);
  }
#ifdef __CUDACC__
  else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(-65504.0f);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(-3.38953e38f);
  }
#endif
  else if constexpr (std::is_same_v<T, float>) {
    return -3.402823466e+38f;
  } else if constexpr (std::is_same_v<T, double>) {
    return -1.7976931348623158e+308;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return -32768;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return -2147483648;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return -9223372036854775807LL - 1LL;
  } else if constexpr (std::is_same_v<T, float4_e2m1_t>) {
    return T(-3.0f); // Min representable value
  } else if constexpr (std::is_same_v<T, float4_e2m1_2x_t>) {
    return T(-3.0f); // Min representable value
  }
  return T{};
}

template <typename T> DEVICE_HOST constexpr T get_max_value() {
  if constexpr (std::is_same_v<T, float16_t>) {
    return T(65504.0f);
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return T(3.38953e38f);
  }
#ifdef __CUDACC__
  else if constexpr (std::is_same_v<T, __half>) {
    return __float2half(65504.0f);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(3.38953e38f);
  }
#endif
  else if constexpr (std::is_same_v<T, float>) {
    return 3.402823466e+38f;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1.7976931348623158e+308;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return 32767;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return 2147483647;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return 9223372036854775807LL;
  } else if constexpr (std::is_same_v<T, float4_e2m1_t>) {
    return T(3.0f); // Max representable value
  } else if constexpr (std::is_same_v<T, float4_e2m1_2x_t>) {
    return T(3.0f); // Max representable value (sets both to max?)
    // Actually, for packed type, identity should probably be max in both slots?
    // But get_max_value returns a scalar T.
    // If T is float4_e2m1_2x_t, it constructs from float.
    // The constructor float4_e2m1_2x_t(float) sets both low and high to that
    // value. So T(3.0f) sets both to 3.0f. Correct.
  }
  return T{};
}

template <typename T> DEVICE_HOST inline bool is_nan_check(T val) {
#ifdef __CUDA_ARCH__
  return gpu_isnan(val);
#else
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(val);
  } else if constexpr (is_half_float_v<T>) {
    float f_val = static_cast<float>(val);
    return std::isnan(f_val);
  } else if constexpr (std::is_same_v<T, float4_e2m1_t> ||
                       std::is_same_v<T, float4_e2m1_2x_t>) {
    float f_val = static_cast<float>(val);
    return std::isnan(f_val);
  }
  return false;
#endif
}

// Specializations for gpu_isnan
#ifdef __CUDA_ARCH__
template <> __device__ inline bool gpu_isnan(float4_e2m1_t val) {
  return std::isnan(static_cast<float>(val));
}
template <> __device__ inline bool gpu_isnan(float4_e2m1_2x_t val) {
  return std::isnan(static_cast<float>(val));
}
#endif

// ═══════════════════════════════════════════════════════════
// ACCUMULATOR TYPE SELECTOR  (IsGPU=false → CPU, IsGPU=true → GPU)
// Pure compile-time trait — zero runtime cost.
// ═══════════════════════════════════════════════════════════

template <typename T, bool IsGPU = false> struct AccumulatorTypeSelector {
  using type = T; // default: no promotion
};

// ── Signed integers: int64_t on both CPU and GPU
// ──────────────────────────────
template <bool IsGPU> struct AccumulatorTypeSelector<int8_t, IsGPU> {
  using type = int64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<int16_t, IsGPU> {
  using type = int64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<int32_t, IsGPU> {
  using type = int64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<int64_t, IsGPU> {
  using type = int64_t;
};

// ── Unsigned integers: uint64_t on both CPU and GPU ──────────────────────────
template <bool IsGPU> struct AccumulatorTypeSelector<uint8_t, IsGPU> {
  using type = uint64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<uint16_t, IsGPU> {
  using type = uint64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<uint32_t, IsGPU> {
  using type = uint64_t;
};
template <bool IsGPU> struct AccumulatorTypeSelector<uint64_t, IsGPU> {
  using type = uint64_t;
};

// ── Bool: int64_t on both (sum = count-of-true, in-kernel cast, no pre-cast)
// ──
template <bool IsGPU> struct AccumulatorTypeSelector<bool, IsGPU> {
  using type = int64_t;
};

// ── Half precision: float on both CPU and GPU
// ─────────────────────────────────
template <bool IsGPU> struct AccumulatorTypeSelector<float16_t, IsGPU> {
  using type = float;
};
template <bool IsGPU> struct AccumulatorTypeSelector<bfloat16_t, IsGPU> {
  using type = float;
};

// ── float: CPU → double (precision), GPU → float (double is 32x slower on GPU)
template <> struct AccumulatorTypeSelector<float, false> {
  using type = double;
};
template <> struct AccumulatorTypeSelector<float, true> {
  using type = float;
};

// ── FP4 types: float on both (no native math ops on FP4) ─────────────────────
template <bool IsGPU> struct AccumulatorTypeSelector<float4_e2m1_t, IsGPU> {
  using type = float;
};
template <bool IsGPU> struct AccumulatorTypeSelector<float4_e2m1_2x_t, IsGPU> {
  using type = float;
};

// ── Complex types: promote component precision, mirroring scalar float rules
// ──
//
//   complex32_t  = 2 × float16_t  →  complex64_t  = 2 × float   (on BOTH CPU
//   and GPU)
//     Rationale: float16 component needs float promotion, same as scalar
//     float16_t → float.
//
//   complex64_t  = 2 × float      →  complex128_t = 2 × double  (CPU only)
//                                 →  complex64_t              (GPU)
//     Rationale: mirrors scalar float rule exactly.
//       CPU: float → double (better precision, no cost).
//       GPU: float → float  (FP64 is 32× slower on consumer GPUs — same reason
//       as scalar float).
//
//   complex128_t = 2 × double     →  complex128_t (no promotion on either
//   device)
//     Rationale: already at maximum precision.
//
// This matches PyTorch's acc_type for complex on CPU and CUDA respectively.
template <bool IsGPU> struct AccumulatorTypeSelector<complex32_t, IsGPU> {
  using type = complex64_t;
};
template <> struct AccumulatorTypeSelector<complex64_t, false> {
  using type = complex128_t;
}; // CPU
template <> struct AccumulatorTypeSelector<complex64_t, true> {
  using type = complex64_t;
}; // GPU
template <bool IsGPU> struct AccumulatorTypeSelector<complex128_t, IsGPU> {
  using type = complex128_t;
};

// ── GPU-native half types (nvcc only) ────────────────────────────────────────
#ifdef __CUDACC__
template <bool IsGPU> struct AccumulatorTypeSelector<__half, IsGPU> {
  using type = float;
};
template <bool IsGPU> struct AccumulatorTypeSelector<__nv_bfloat16, IsGPU> {
  using type = float;
};
#endif

// ── Convenience alias
// ───────────────────────────────────────────────────────── Default
// auto-selects: nvcc (.cu files) → GPU types, g++ (.cpp files) → CPU types.
// Explicit AccumulatorType<T, true/false> overrides when needed.
#ifdef __CUDACC__
template <typename T, bool IsGPU = true>
#else
template <typename T, bool IsGPU = false>
#endif
using AccumulatorType = typename AccumulatorTypeSelector<T, IsGPU>::type;

// ── Product accumulator: NO float→double promotion
// ──────────────────────────── Multiplication doesn't suffer from catastrophic
// cancellation (unlike addition). It only scales exponents — no mantissa digits
// are destroyed. Promoting to double doesn't help: if result > FLOAT_MAX,
// casting double back still gives Infinity. So product stays in native
// precision. Only fp16/bf16 still promote to float32 (no native CPU math), and
// integers still promote to int64 (overflow protection).
template <typename T> struct ProductAccumulatorSelector {
  // Default: same as AccumulatorType (handles fp16→float, int→int64, etc.)
  using type = AccumulatorType<T>;
};
// Override: float stays float on CPU (NOT double)
template <> struct ProductAccumulatorSelector<float> {
  using type = float;
};
// Override: complex64 stays complex64 on CPU (NOT complex128)
template <> struct ProductAccumulatorSelector<complex64_t> {
  using type = complex64_t;
};

template <typename T>
using ProductAccumType = typename ProductAccumulatorSelector<T>::type;

// ═══════════════════════════════════════════════════════════
//  CORE REDUCTION OPERATIONS (NOW USES GPU INTRINSICS!)
// ═══════════════════════════════════════════════════════════

// ---- Generic shfl_down_sync handling complex types ----

#ifdef __CUDACC__
// Robust bit-cast shuffle to handle structs (Complex, etc.) via 32-bit
// registers
template <typename T> __device__ inline T generic_shfl_down(T val, int offset) {
  if constexpr (sizeof(T) == 4) {
    unsigned int tmp;
    memcpy(&tmp, &val, 4);
    tmp = ::__shfl_down_sync(0xffffffff, tmp, offset, 32);
    memcpy(&val, &tmp, 4);
  } else if constexpr (sizeof(T) == 8) {
    unsigned long long tmp;
    memcpy(&tmp, &val, 8);
    unsigned int lo = (unsigned int)(tmp & 0xffffffff);
    unsigned int hi = (unsigned int)(tmp >> 32);
    lo = ::__shfl_down_sync(0xffffffff, lo, offset, 32);
    hi = ::__shfl_down_sync(0xffffffff, hi, offset, 32);
    tmp = ((unsigned long long)hi << 32) | lo;
    memcpy(&val, &tmp, 8);
  } else if constexpr (sizeof(T) == 2) {
    unsigned short tmp;
    memcpy(&tmp, &val, 2);
    unsigned int itmp = tmp;
    itmp = ::__shfl_down_sync(0xffffffff, itmp, offset, 32);
    tmp = (unsigned short)itmp;
    memcpy(&val, &tmp, 2);
  } else {
    val = ::__shfl_down_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

__device__ inline complex32_t generic_shfl_down(complex32_t val, int offset) {
  return complex32_t(generic_shfl_down(val.real(), offset),
                     generic_shfl_down(val.imag(), offset));
}
__device__ inline complex64_t generic_shfl_down(complex64_t val, int offset) {
  return complex64_t(generic_shfl_down(val.real(), offset),
                     generic_shfl_down(val.imag(), offset));
}
__device__ inline complex128_t generic_shfl_down(complex128_t val, int offset) {
  return complex128_t(generic_shfl_down(val.real(), offset),
                      generic_shfl_down(val.imag(), offset));
}
#endif

// Greater-than helper (used by MinOp, MaxOp, WelfordOps::project)
template <typename T> DEVICE_HOST inline bool generic_gt(T a, T b) {
  return a > b;
}

template <typename T> struct SumOp {
  using AccT = AccumulatorType<T>;

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
// edits: Gautam_Reddy_1926 — dropped the isnan early-return branch on both
//   GPU and CPU paths. IEEE 754 already propagates NaN through plain a+b
//   (NaN + x = NaN), and pytorch's sum_functor at
//   pytorch_source/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu does
//   the same plain a+b with zero NaN check.
//
//   CPU side: the isnan block was DEFENSIVE CODE — not strictly dead, but
//   never reached in practice. dispatch in ReductionImpl.h:2408 routes all
//   non-integral / non-bool float types (float, double, half, bfloat16,
//   complex) to cascade_sum_kernel BEFORE this functor is touched. Only
//   integers (int*/uint*) and bool fall through to reduce_kernel<SumOp>,
//   and for those AccT = int64_t/uint64_t/int64_t so is_any_float_v<AccT>
//   is false at compile time → the if constexpr block was eliminated by
//   the compiler. Defensive guard that only matters if someone bypasses
//   cascade_sum_kernel and routes float sums through reduce_kernel
//   directly. Removed here for code clarity.
//
//   GPU side: the isnan block WAS executing for every float add — every
//   float/half/bfloat16 sum on GPU paid 2 gpu_isnan + 2 branches per
//   element. That's the real perf win from this removal.
#ifdef __CUDA_ARCH__
    return gpu_add(a, b);
#else
    return a + b;
#endif
  }

  // 3-arg overload for unified kernel (idx unused for value ops)
  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  // Merge two partial results (same as reduce for sum)
  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  // Final output projection (identity for sum)
  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  // Warp shuffle for GPU reduction pipeline
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct ProductOp {
  using AccT = ProductAccumType<T>;

  DEVICE_HOST AccT identity() const { return AccT(1.0f); }
  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
// edits: Gautam_Reddy_1926 — dropped the isnan early-return branch on both
//   GPU and CPU paths. IEEE 754 already propagates NaN through plain a*b
//   (NaN * x = NaN), and pytorch's prod_functor in
//   pytorch_source/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu does
//   the same plain a*b with zero NaN check.
//
//   CPU side: same story as SumOp — the isnan block was DEFENSIVE CODE,
//   not strictly dead. ProductOp is dispatched via reduce_kernel for ALL
//   types (no cascade equivalent for product), but for integer/bool T
//   the AccT is int64/uint64/int64 so is_any_float_v<AccT> is false →
//   the if constexpr block is compiled out. For float T the AccT IS
//   float-type and the branch WOULD run, BUT in practice product
//   reductions over float tensors are rare in our workloads. Keeping
//   it as defensive guard adds cost on the float path, so we drop it
//   to match pytorch's IEEE-relying behavior. If a numerical issue
//   surfaces on a float product later, revert this hunk first.
//
//   GPU side: the isnan block WAS executing for every float multiply —
//   2 gpu_isnan + 2 branches per element removed.
#ifdef __CUDA_ARCH__
    return gpu_mul(a, b);
#else
    return a * b;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct MinOp {
  using AccT =
      T; // compare op: result always within input range, no widening needed

  DEVICE_HOST AccT identity() const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<AccT>(get_max_value<T>());
    } else if constexpr (is_half_float_v<T>) {
      return static_cast<AccT>(get_max_value<T>());
    }
#ifdef __CUDACC__
    else if constexpr (is_native_half_v<T>) {
      return get_max_value<T>();
    }
#endif
    else {
      return get_max_value<AccT>();
    }
  }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<AccT, complex32_t> ||
                  std::is_same_v<AccT, complex64_t> ||
                  std::is_same_v<AccT, complex128_t>) {
      return a; // Placeholder - should never be called
    } else {
#ifdef __CUDA_ARCH__
      //  GPU: Use intrinsics
      if constexpr (is_any_float_v<T>) {
        if (gpu_isnan(a))
          return a;
        if (gpu_isnan(b))
          return b;
      }
      return gpu_lt(a, b) ? a : b;
#else
      // CPU path
      if constexpr (is_any_float_v<T>) {
        if (is_nan_check(a))
          return a;
        if (is_nan_check(b))
          return b;
      }
      return (a < b) ? a : b;
#endif
    }
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct MaxOp {
  using AccT =
      T; // compare op: result always within input range, no widening needed

  DEVICE_HOST AccT identity() const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<AccT>(get_lowest_value<T>());
    } else if constexpr (is_half_float_v<T>) {
      return static_cast<AccT>(get_lowest_value<T>());
    }
#ifdef __CUDACC__
    else if constexpr (is_native_half_v<T>) {
      return get_lowest_value<T>();
    }
#endif
    else {
      return get_lowest_value<AccT>();
    }
  }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<AccT, complex32_t> ||
                  std::is_same_v<AccT, complex64_t> ||
                  std::is_same_v<AccT, complex128_t>) {
      return a; // Placeholder - should never be called
    } else {
#ifdef __CUDA_ARCH__
      //  GPU: Use intrinsics
      if constexpr (is_any_float_v<T>) {
        if (gpu_isnan(a))
          return a;
        if (gpu_isnan(b))
          return b;
      }
      return gpu_gt(a, b) ? a : b;
#else
      // CPU path
      if constexpr (is_any_float_v<T>) {
        if (is_nan_check(a))
          return a;
        if (is_nan_check(b))
          return b;
      }
      return (a > b) ? a : b;
#endif
    }
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};
// ═══════════════════════════════════════════════════════════
// VARIANCE OPERATION (Two-pass algorithm for numerical stability)
// ═══════════════════════════════════════════════════════════

template <typename T> struct VarianceOp {
  using AccT = AccumulatorType<T>;
  int64_t correction; // Bessel's correction
  AccT mean_value;    // Pre-computed mean

  DEVICE_HOST explicit VarianceOp(int64_t corr = 1, AccT mean = AccT(0.0f))
      : correction(corr), mean_value(mean) {}

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &acc, const AccT &val) const {
#ifdef __CUDA_ARCH__
    //  GPU: Propagate NaN immediately
    if constexpr (is_any_float_v<AccT>) {
      if (gpu_isnan(acc))
        return acc; // Already NaN, propagate it
      if (gpu_isnan(val))
        return val; // New NaN, propagate it
    }
    AccT diff = val - mean_value;
    return gpu_add(acc, gpu_mul(diff, diff));
#else
    //  CPU: Propagate NaN immediately
    if constexpr (is_any_float_v<AccT>) {
      if (is_nan_check(acc))
        return acc; // Already NaN, propagate it
      if (is_nan_check(val))
        return val; // New NaN, propagate it
    }
    AccT diff = val - mean_value;
    return acc + diff * diff;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
// Combining two partial sums of squared deviations
#ifdef __CUDA_ARCH__
    return gpu_add(a, b);
#else
    return a + b;
#endif
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};
// ═══════════════════════════════════════════════════════════
// NaN-AWARE OPERATIONS (ALSO USE GPU INTRINSICS)
// ═══════════════════════════════════════════════════════════

template <typename T> struct NanSumOp {
  using AccT = AccumulatorType<T>;

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
      if (gpu_isnan(a))
        return b;
      if (gpu_isnan(b))
        return a;
    }
    return gpu_add(a, b);
#else
    if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
      if (is_nan_check(a))
        return b;
      if (is_nan_check(b))
        return a;
    }
    return a + b;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct NanProductOp {
  using AccT = ProductAccumType<T>;

  DEVICE_HOST AccT identity() const { return AccT(1.0f); }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
      if (gpu_isnan(a))
        return b;
      if (gpu_isnan(b))
        return a;
    }
    return gpu_mul(a, b);
#else
    if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
      if (is_nan_check(a))
        return b;
      if (is_nan_check(b))
        return a;
    }
    return a * b;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct NanMinOp {
  using AccT =
      T; // compare op: result always within input range, no widening needed

  DEVICE_HOST AccT identity() const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<AccT>(get_max_value<T>());
    } else if constexpr (is_half_float_v<T>) {
      return static_cast<AccT>(get_max_value<T>());
    }
#ifdef __CUDACC__
    else if constexpr (is_native_half_v<T>) {
      return get_max_value<T>();
    }
#endif
    else {
      return get_max_value<AccT>();
    }
  }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<AccT, complex32_t> ||
                  std::is_same_v<AccT, complex64_t> ||
                  std::is_same_v<AccT, complex128_t>) {
      return a; // Placeholder - should never be called
    } else {
#ifdef __CUDA_ARCH__
      if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
        if (gpu_isnan(a))
          return b;
        if (gpu_isnan(b))
          return a;
      }
      return gpu_lt(a, b) ? a : b;
#else
      if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
        if (is_nan_check(a))
          return b;
        if (is_nan_check(b))
          return a;
      }
      return (a < b) ? a : b;
#endif
    }
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

template <typename T> struct NanMaxOp {
  using AccT =
      T; // compare op: result always within input range, no widening needed

  DEVICE_HOST AccT identity() const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<AccT>(get_lowest_value<T>());
    } else if constexpr (is_half_float_v<T>) {
      return static_cast<AccT>(get_lowest_value<T>());
    }
#ifdef __CUDACC__
    else if constexpr (is_native_half_v<T>) {
      return get_lowest_value<T>();
    }
#endif
    else {
      return get_lowest_value<AccT>();
    }
  }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
    // Complex types don't support ordering operations
    if constexpr (std::is_same_v<AccT, complex32_t> ||
                  std::is_same_v<AccT, complex64_t> ||
                  std::is_same_v<AccT, complex128_t>) {
      // This branch will never be reached but prevents compilation errors
      return a; // Placeholder - should never be called
    } else {
#ifdef __CUDA_ARCH__
      if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
        if (gpu_isnan(a))
          return b;
        if (gpu_isnan(b))
          return a;
      }
      return gpu_gt(a, b) ? a : b;
#else
      if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
        if (is_nan_check(a))
          return b;
        if (is_nan_check(b))
          return a;
      }
      return (a > b) ? a : b;
#endif
    }
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
}; // ═══════════════════════════════════════════════════════════
// NaN-aware variance (IGNORES NaNs, doesn't propagate them)
// ═══════════════════════════════════════════════════════════

template <typename T> struct NanVarianceOp {
  using AccT = AccumulatorType<T>;
  int64_t correction;
  AccT mean_value;

  DEVICE_HOST explicit NanVarianceOp(int64_t corr = 1, AccT mean = AccT(0.0f))
      : correction(corr), mean_value(mean) {}

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &acc, const AccT &val) const {
#ifdef __CUDA_ARCH__
    //  GPU: Skip NaN values (don't propagate them)
    if (gpu_isnan(val))
      return acc; // Ignore NaN, return accumulator unchanged
    AccT diff = val - mean_value;
    return gpu_add(acc, gpu_mul(diff, diff));
#else
    //  CPU: Skip NaN values (don't propagate them)
    if (is_nan_check(val))
      return acc; // Ignore NaN, return accumulator unchanged
    AccT diff = val - mean_value;
    return acc + diff * diff;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
#ifdef __CUDA_ARCH__
    return gpu_add(a, b);
#else
    return a + b;
#endif
  }

  DEVICE_HOST AccT project(const AccT &a) const { return a; }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};
// ═══════════════════════════════════════════════════════════
// INDEX REDUCTIONS (ArgMin/ArgMax) - ALSO USE GPU INTRINSICS
// ═══════════════════════════════════════════════════════════

template <typename T> struct ArgMinOp {
  using AccumulatorType = ValueIndex<T>;

  DEVICE_HOST ValueIndex<T> identity() const {
    return ValueIndex<T>(get_max_value<T>(), -1);
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &a,
                                   const ValueIndex<T> &b) const {
    // Complex types are blocked at dispatcher level but prevent compilation
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return a; // Should never be called
    } else {
#ifdef __CUDA_ARCH__
      if constexpr (is_any_float_v<T>) {
        if (gpu_isnan(a.value) && gpu_isnan(b.value))
          return (a.index < b.index) ? a : b;
        if (gpu_isnan(a.value))
          return a;
        if (gpu_isnan(b.value))
          return b;
      }
      if (gpu_lt(a.value, b.value)) {
        return a;
      } else if (gpu_gt(a.value, b.value)) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#else
      if constexpr (is_any_float_v<T>) {
        if (is_nan_check(a.value) && is_nan_check(b.value))
          return (a.index < b.index) ? a : b;
        if (is_nan_check(a.value))
          return a;
        if (is_nan_check(b.value))
          return b;
      }
      if (a.value < b.value) {
        return a;
      } else if (b.value < a.value) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#endif
    }
  }

  // 3-arg: wrap val+idx into ValueIndex, delegate to 2-arg
  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &acc, const T &val,
                                   int64_t idx) const {
    return reduce(acc, ValueIndex<T>(val, idx));
  }

  DEVICE_HOST ValueIndex<T> combine(const ValueIndex<T> &a,
                                    const ValueIndex<T> &b) const {
    return reduce(a, b);
  }

  // Project: extract index for output
  DEVICE_HOST int64_t project(const ValueIndex<T> &a) const { return a.index; }

#ifdef __CUDACC__
  __device__ ValueIndex<T> warp_shfl_down(ValueIndex<T> val, int offset) const {
    ValueIndex<T> result;
    result.value = generic_shfl_down(val.value, offset);
    result.index = generic_shfl_down(val.index, offset);
    return result;
  }
#endif
};

template <typename T> struct ArgMaxOp {
  using AccumulatorType = ValueIndex<T>;

  DEVICE_HOST ValueIndex<T> identity() const {
    return ValueIndex<T>(get_lowest_value<T>(), -1);
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &a,
                                   const ValueIndex<T> &b) const {
    // Complex types are blocked at dispatcher level but prevent compilation
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return a; // Should never be called
    } else {
#ifdef __CUDA_ARCH__
      if constexpr (is_any_float_v<T>) {
        if (gpu_isnan(a.value) && gpu_isnan(b.value))
          return (a.index < b.index) ? a : b;
        if (gpu_isnan(a.value))
          return a;
        if (gpu_isnan(b.value))
          return b;
      }
      if (gpu_gt(a.value, b.value)) {
        return a;
      } else if (gpu_lt(a.value, b.value)) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#else
      if constexpr (is_any_float_v<T>) {
        if (is_nan_check(a.value) && is_nan_check(b.value))
          return (a.index < b.index) ? a : b;
        if (is_nan_check(a.value))
          return a;
        if (is_nan_check(b.value))
          return b;
      }
      if (a.value > b.value) {
        return a;
      } else if (b.value > a.value) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#endif
    }
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &acc, const T &val,
                                   int64_t idx) const {
    return reduce(acc, ValueIndex<T>(val, idx));
  }

  DEVICE_HOST ValueIndex<T> combine(const ValueIndex<T> &a,
                                    const ValueIndex<T> &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST int64_t project(const ValueIndex<T> &a) const { return a.index; }

#ifdef __CUDACC__
  __device__ ValueIndex<T> warp_shfl_down(ValueIndex<T> val, int offset) const {
    ValueIndex<T> result;
    result.value = generic_shfl_down(val.value, offset);
    result.index = generic_shfl_down(val.index, offset);
    return result;
  }
#endif
};

template <typename T> struct NanArgMinOp {
  using AccumulatorType = ValueIndex<T>;

  DEVICE_HOST ValueIndex<T> identity() const {
    return ValueIndex<T>(get_max_value<T>(), -1);
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &a,
                                   const ValueIndex<T> &b) const {
    // Complex types are blocked at dispatcher level but prevent compilation
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return a; // Should never be called
    } else {
#ifdef __CUDA_ARCH__
      const bool a_is_nan = gpu_isnan(a.value);
      const bool b_is_nan = gpu_isnan(b.value);
      if (a_is_nan && b_is_nan)
        return (a.index < b.index) ? a : b;
      if (a_is_nan)
        return b;
      if (b_is_nan)
        return a;

      if (gpu_lt(a.value, b.value)) {
        return a;
      } else if (gpu_gt(a.value, b.value)) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#else
      const bool a_is_nan = is_nan_check(a.value);
      const bool b_is_nan = is_nan_check(b.value);
      if (a_is_nan && b_is_nan)
        return (a.index < b.index) ? a : b;
      if (a_is_nan)
        return b;
      if (b_is_nan)
        return a;

      if (a.value < b.value) {
        return a;
      } else if (b.value < a.value) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#endif
    }
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &acc, const T &val,
                                   int64_t idx) const {
    return reduce(acc, ValueIndex<T>(val, idx));
  }

  DEVICE_HOST ValueIndex<T> combine(const ValueIndex<T> &a,
                                    const ValueIndex<T> &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST int64_t project(const ValueIndex<T> &a) const { return a.index; }

#ifdef __CUDACC__
  __device__ ValueIndex<T> warp_shfl_down(ValueIndex<T> val, int offset) const {
    ValueIndex<T> result;
    result.value = generic_shfl_down(val.value, offset);
    result.index = generic_shfl_down(val.index, offset);
    return result;
  }
#endif
};

template <typename T> struct NanArgMaxOp {
  using AccumulatorType = ValueIndex<T>;

  DEVICE_HOST ValueIndex<T> identity() const {
    return ValueIndex<T>(get_lowest_value<T>(), -1);
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &a,
                                   const ValueIndex<T> &b) const {
    // Complex types are blocked at dispatcher level but prevent compilation
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      return a; // Should never be called
    } else {
#ifdef __CUDA_ARCH__
      const bool a_is_nan = gpu_isnan(a.value);
      const bool b_is_nan = gpu_isnan(b.value);
      if (a_is_nan && b_is_nan)
        return (a.index < b.index) ? a : b;
      if (a_is_nan)
        return b;
      if (b_is_nan)
        return a;

      if (gpu_gt(a.value, b.value)) {
        return a;
      } else if (gpu_lt(a.value, b.value)) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#else
      const bool a_is_nan = is_nan_check(a.value);
      const bool b_is_nan = is_nan_check(b.value);
      if (a_is_nan && b_is_nan)
        return (a.index < b.index) ? a : b;
      if (a_is_nan)
        return b;
      if (b_is_nan)
        return a;

      if (a.value > b.value) {
        return a;
      } else if (b.value > a.value) {
        return b;
      } else {
        return (a.index < b.index) ? a : b;
      }
#endif
    }
  }

  DEVICE_HOST ValueIndex<T> reduce(const ValueIndex<T> &acc, const T &val,
                                   int64_t idx) const {
    return reduce(acc, ValueIndex<T>(val, idx));
  }

  DEVICE_HOST ValueIndex<T> combine(const ValueIndex<T> &a,
                                    const ValueIndex<T> &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST int64_t project(const ValueIndex<T> &a) const { return a.index; }

#ifdef __CUDACC__
  __device__ ValueIndex<T> warp_shfl_down(ValueIndex<T> val, int offset) const {
    ValueIndex<T> result;
    result.value = generic_shfl_down(val.value, offset);
    result.index = generic_shfl_down(val.index, offset);
    return result;
  }
#endif
};

// ═══════════════════════════════════════════════════════════
// BOOLEAN REDUCTION OPERATIONS (Bool dtype only)
// ═══════════════════════════════════════════════════════════

template <typename T> struct AllOp {
  using AccT = bool; // Always accumulate as bool

  DEVICE_HOST bool identity() const {
    return true; // Neutral element for AND operation
  }

  DEVICE_HOST bool reduce(const bool &a, const bool &b) const { return a && b; }

  DEVICE_HOST bool reduce(const bool &acc, const bool &val,
                          int64_t /*idx*/) const {
    return reduce(acc, val);
  }

  DEVICE_HOST bool combine(const bool &a, const bool &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST bool project(const bool &a) const { return a; }

#ifdef __CUDACC__
  __device__ bool warp_shfl_down(bool val, int offset) const {
    return generic_shfl_down(static_cast<int>(val), offset) != 0;
  }
#endif
};

template <typename T> struct AnyOp {
  using AccT = bool; // Always accumulate as bool

  DEVICE_HOST bool identity() const {
    return false; // Neutral element for OR operation
  }

  DEVICE_HOST bool reduce(const bool &a, const bool &b) const { return a || b; }

  DEVICE_HOST bool reduce(const bool &acc, const bool &val,
                          int64_t /*idx*/) const {
    return reduce(acc, val);
  }

  DEVICE_HOST bool combine(const bool &a, const bool &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST bool project(const bool &a) const { return a; }

#ifdef __CUDACC__
  __device__ bool warp_shfl_down(bool val, int offset) const {
    return generic_shfl_down(static_cast<int>(val), offset) != 0;
  }
#endif
};
// ═══════════════════════════════════════════════════════════
// WELFORD DATA & OPS (Single-pass variance, PyTorch-compatible)
// ═══════════════════════════════════════════════════════════

template <typename acc_t, typename index_t = int64_t> struct WelfordData {
  acc_t mean;
  acc_t m2;
  index_t n;
  acc_t nf;

  DEVICE_HOST WelfordData()
      : mean(acc_t(0)), m2(acc_t(0)), n(0), nf(acc_t(0)) {}
  DEVICE_HOST WelfordData(acc_t mean, acc_t m2, index_t n, acc_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename T> struct WelfordOps {
  using AccScalar = AccumulatorType<T>;
  using acc_t = WelfordData<AccScalar>;
  AccScalar correction;
  bool take_sqrt;

  DEVICE_HOST WelfordOps(AccScalar corr = AccScalar(1), bool sqrt = false)
      : correction(corr), take_sqrt(sqrt) {}

  DEVICE_HOST acc_t identity() const { return acc_t{}; }

  // Per-element Welford update
  DEVICE_HOST acc_t reduce(acc_t acc, T data, int64_t /*idx*/) const {
    int64_t new_n = acc.n + 1;
    AccScalar s_data = static_cast<AccScalar>(data);
    AccScalar new_nf = static_cast<AccScalar>(new_n);
    AccScalar delta = s_data - acc.mean;
    AccScalar new_mean = acc.mean + delta / new_nf;
    AccScalar new_delta = s_data - new_mean;
    return acc_t(new_mean, acc.m2 + delta * new_delta, new_n, new_nf);
  }

  // 2-arg overload for backward compat (delegates to 3-arg)
  DEVICE_HOST acc_t reduce(const acc_t &a, const acc_t &b) const {
    return combine(a, b);
  }

  // Merge two Welford partial aggregates (parallel reduction)
  DEVICE_HOST acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == AccScalar(0.0f))
      return b;
    if (b.nf == AccScalar(0.0f))
      return a;
    AccScalar delta = b.mean - a.mean;
    AccScalar new_count = a.nf + b.nf;
    AccScalar nb_over_n = b.nf / new_count;
    return acc_t(a.mean + delta * nb_over_n,
                 a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
                 -1, // n is unreliable after combine (may overflow int32)
                 new_count);
  }

  // Final projection: variance (or std if take_sqrt)
  DEVICE_HOST AccScalar project(acc_t acc) const {
    AccScalar divisor =
        generic_gt(acc.nf, correction) ? acc.nf - correction : AccScalar(0.0f);
    AccScalar var = acc.m2 / divisor;
    // take_sqrt: for std deviation
    // Note: sqrt not available in all constexpr contexts, handled at call site
    return var;
  }

#ifdef __CUDACC__
  __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return acc_t(
        generic_shfl_down(acc.mean, offset), generic_shfl_down(acc.m2, offset),
        generic_shfl_down(acc.n, offset), generic_shfl_down(acc.nf, offset));
  }
#endif
};

// ═══════════════════════════════════════════════════════════
// MEAN OPS (Unified kernel compatible: sum + project=divide)
// ═══════════════════════════════════════════════════════════

template <typename T> struct MeanOps {
  using AccT = AccumulatorType<T>;
  AccT factor; // = 1.0 / reduced_count (pre-computed on host)

  DEVICE_HOST MeanOps(AccT f = AccT(1.0f)) : factor(f) {}

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
#ifdef __CUDA_ARCH__
    return gpu_add(a, b);
#else
    return a + b;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  // Project: multiply by 1/count to get mean
  DEVICE_HOST AccT project(const AccT &a) const {
#ifdef __CUDA_ARCH__
    return gpu_mul(a, factor);
#else
    return a * factor;
#endif
  }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

// NaN-aware mean: skips NaN values, tracks count
template <typename T> struct NanMeanOps {
  using AccT = AccumulatorType<T>;
  // No pre-computed factor — must divide by valid_count at end

  DEVICE_HOST NanMeanOps() {}

  DEVICE_HOST AccT identity() const { return AccT(0.0f); }

  DEVICE_HOST AccT reduce(const AccT &a, const AccT &b) const {
#ifdef __CUDA_ARCH__
    if constexpr (std::is_floating_point_v<AccT> || is_native_half_v<AccT>) {
      if (gpu_isnan(a))
        return b;
      if (gpu_isnan(b))
        return a;
    }
    return gpu_add(a, b);
#else
    if constexpr (std::is_floating_point_v<AccT> || is_half_float_v<AccT>) {
      if (is_nan_check(a))
        return b;
      if (is_nan_check(b))
        return a;
    }
    return a + b;
#endif
  }

  DEVICE_HOST AccT reduce(const AccT &acc, const T &val,
                          int64_t /*idx*/) const {
    return reduce(acc, static_cast<AccT>(val));
  }

  DEVICE_HOST AccT combine(const AccT &a, const AccT &b) const {
    return reduce(a, b);
  }

  DEVICE_HOST AccT project(const AccT &a) const {
    return a; // Division by valid_count handled externally
  }

#ifdef __CUDACC__
  __device__ AccT warp_shfl_down(AccT val, int offset) const {
    return generic_shfl_down(val, offset);
  }
#endif
};

// ═══════════════════════════════════════════════════════════
// REDUCTION TYPE DISPATCHER
// ═══════════════════════════════════════════════════════════

enum class ReductionType {
  SUM,
  PRODUCT,
  MIN,
  MAX,
  NANSUM,
  NANPRODUCT,
  NANMIN,
  NANMAX,
  ARGMIN,
  ARGMAX,
  NANARGMIN,
  NANARGMAX,
  ALL,
  ANY
};

template <ReductionType R, typename T> struct ReductionOpSelector;

template <typename T> struct ReductionOpSelector<ReductionType::SUM, T> {
  using type = SumOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::PRODUCT, T> {
  using type = ProductOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::MIN, T> {
  using type = MinOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::MAX, T> {
  using type = MaxOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANSUM, T> {
  using type = NanSumOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANPRODUCT, T> {
  using type = NanProductOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANMIN, T> {
  using type = NanMinOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANMAX, T> {
  using type = NanMaxOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::ARGMIN, T> {
  using type = ArgMinOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::ARGMAX, T> {
  using type = ArgMaxOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANARGMIN, T> {
  using type = NanArgMinOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::NANARGMAX, T> {
  using type = NanArgMaxOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::ALL, T> {
  using type = AllOp<T>;
};
template <typename T> struct ReductionOpSelector<ReductionType::ANY, T> {
  using type = AnyOp<T>;
};
} // namespace detail
} // namespace OwnTensor

#endif // OWNTENSOR_REDUCTION_OPS_H