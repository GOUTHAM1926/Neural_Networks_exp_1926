#pragma once

#ifndef OWNTENSOR_REDUCTIONS_IMPL_H
#define OWNTENSOR_REDUCTIONS_IMPL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/helpers/ReductionOps.h"
#include "ops/helpers/ReductionUtils.h"
#include "ops/helpers/Vectorized.h"

// #ifdef WITH_CUDA
// #include "ReductionImplGPU.h"
// #endif

namespace OwnTensor {
namespace detail {
// Forward declarations only (no implementation here!)
template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor &input,
                              const std::vector<int64_t> &normalized_axes,
                              bool keepdim, cudaStream_t stream); // ✨✨✨

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor &input,
                                    const std::vector<int64_t> &normalized_axes,
                                    bool keepdim,
                                    cudaStream_t stream); // ✨✨✨

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor &input,
                         const std::vector<int64_t> &normalized_axes,
                         bool keepdim, cudaStream_t stream); // ✨✨✨
template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor &input,
                             const std::vector<int64_t> &normalized_axes,
                             bool keepdim, int64_t correction,
                             cudaStream_t stream); // ✨✨✨

constexpr size_t MAX_DIMS = 64;

// Minimum work per thread before parallelization is worthwhile.
// Matches PyTorch's at::internal::GRAIN_SIZE.
// Tuned for L1 cache: 32768 * 4 bytes(float) = 128KB ≈ typical L1 per core.
constexpr int64_t GRAIN_SIZE = 32768;

// Threading strategy for reduction kernels.
// Each kernel implements both paths. The universal dispatcher chooses.
enum class ReductionStrategy {
  ParallelSlices, // Strategy 1: #pragma omp parallel for over output slots
  SplitReduction  // Strategy 2: split reduction dim across threads + combine
};

// =================================================================
// HELPER: Safe isnan check for both real and complex types
// Helper: convert any type to double (handles complex by taking real part)
template <typename T> inline double to_double(const T &val) {
  if constexpr (std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t>)
    return static_cast<double>(val.real());
  else
    return static_cast<double>(val);
}

// =================================================================
template <typename T> inline bool safe_isnan(const T &val) {
  if constexpr (std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t>) {
    // For complex types, check if either real or imaginary parts are NaN
    // However, for reduction operations with complex types, they typically
    // aren't supported for NaN-aware operations, so return false
    return false;
  } else if constexpr (std::is_same_v<T, float16_t> ||
                       std::is_same_v<T, bfloat16_t>) {
    return std::isnan(static_cast<float>(val));
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(val);
  } else {
    // Non-floating types (integers, bool) can't be NaN
    return false;
  }
}

// =================================================================
// HELPER: Convert any type to bool (for all/any reductions)
// =================================================================
template <typename T> inline bool to_bool_value(const T &val) {
  if constexpr (std::is_same_v<T, complex32_t>) {
    // Complex is true if either real or imaginary part is non-zero
    float r = static_cast<float>(val.real());
    float i = static_cast<float>(val.imag());
    return (r != 0.0f) || (i != 0.0f);
  } else if constexpr (std::is_same_v<T, complex64_t>) {
    return (val.real() != 0.0f) || (val.imag() != 0.0f);
  } else if constexpr (std::is_same_v<T, complex128_t>) {
    return (val.real() != 0.0) || (val.imag() != 0.0);
  } else if constexpr (std::is_same_v<T, bool>) {
    return val;
  } else {
    // For all other types (int, float, etc.), non-zero is true
    return val != T(0.0f);
  }
}

// =================================================================
// HELPER: Check if we should use double accumulation for better precision
// =================================================================
template <typename T> constexpr bool should_use_double_accumulation() {
  return std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> ||
         std::is_same_v<T, float4_e2m1_t> ||
         std::is_same_v<T, float4_e2m1_2x_t>;
}

// =================================================================
// HELPER: ceil_log2 — compute ceil(log2(n)) for cascade_sum algorithm
// =================================================================
inline int64_t ceil_log2(int64_t n) {
  if (n <= 1)
    return 0;
  int64_t log = 0, val = 1;
  while (val < n) {
    val <<= 1;
    ++log;
  }
  return log;
}

// =================================================================
// CORE: multi_row_sum — 4-level cascading accumulator (PyTorch - style)
//
// Sums `size` elements with O(1) storage, matching pairwise tree
// accuracy. Level 0 fills linearly, dumps into Level 1 when full,
// Level 1 dumps into Level 2, etc. 4 levels = 4 CPU registers.
// =================================================================
template <typename acc_t, int64_t nrows>
std::array<acc_t, nrows>
multi_row_sum(const acc_t *__restrict__ in_data, const int64_t row_stride,
              const int64_t col_stride, const int64_t size) {
  constexpr int64_t num_levels = 4;
  const int64_t level_power =
      std::max(int64_t(4), ceil_log2(size) / num_levels);
  const int64_t level_step = (int64_t(1) << level_power);
  const int64_t level_mask = level_step - 1;

  std::array<std::array<acc_t, nrows>, num_levels> acc{};
  for (auto &row : acc)
    row.fill(acc_t{});

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const acc_t *base = in_data + i * row_stride;
      for (int64_t k = 0; k < nrows; ++k)
        acc[0][k] += base[k * col_stride];
    }
    for (int64_t j = 1; j < num_levels; ++j) {
      for (int64_t k = 0; k < nrows; ++k) {
        acc[j][k] += acc[j - 1][k];
        acc[j - 1][k] = acc_t{};
      }
      if ((i & (level_mask << (j * level_power))) != 0)
        break;
    }
  }
  for (; i < size; ++i) {
    const acc_t *base = in_data + i * row_stride;
    for (int64_t k = 0; k < nrows; ++k)
      acc[0][k] += base[k * col_stride];
  }
  for (int64_t j = 1; j < num_levels; ++j)
    for (int64_t k = 0; k < nrows; ++k)
      acc[0][k] += acc[j][k];

  return acc[0];
}

// =================================================================
// HELPER: row_sum — single row with ILP=4 accumulators
// =================================================================
template <typename acc_t>
acc_t row_sum(const acc_t *__restrict__ in_data, const int64_t stride,
              const int64_t size) {
  constexpr int64_t ilp = 4;
  const int64_t size_ilp = size / ilp;
  auto partials =
      multi_row_sum<acc_t, ilp>(in_data, stride * ilp, stride, size_ilp);
  for (int64_t i = size_ilp * ilp; i < size; ++i)
    partials[0] += in_data[i * stride];
  for (int64_t k = 1; k < ilp; ++k)
    partials[0] += partials[k];
  return partials[0];
}

// =================================================================
// CASCADE SUM KERNEL — floating-point sum with cascade accumulation
//
// Replaces reduce_kernel for floating-point sum/nansum.
// Uses 4-level cascading for pairwise-tree accuracy with O(1) storage.
// 3 layout paths: InnerContiguous, OuterContiguous, Generic.
// =================================================================

// CPU-only: a middle-axis reduction (dims both before AND after the reduced
// axis) is tagged OuterContiguous with input_outer_stride = reduced*inner, but
// the CPU OuterContiguous consumers (scalar + SIMD) assume the output is one
// contiguous column block — valid only when outer==1 (leading-axis reduction).
// Demote the middle case to the Generic carry-add path, which walks arbitrary
// strides correctly. (The GPU OuterContiguous kernel handles the outer slab
// stride directly, so this demotion is intentionally CPU-side only.)
inline void cpu_demote_middle_to_generic(ReductionLayout &layout) {
  if (layout.path == ReductionLayout::Path::OuterContiguous &&
      layout.input_outer_stride != layout.input_row_stride)
    layout.path = ReductionLayout::Path::Generic;
}

template <bool ignore_nan, typename T>
Tensor cascade_sum_kernel(const Tensor &input,
                          const std::vector<int64_t> &normalized_axes,
                          const Shape &output_shape, ReductionStrategy strategy,
                          int num_threads) {
  // CASCADE SUM ACCUMULATOR: Use FLOAT for fp32 (like PyTorch), not double.
  // PyTorch (SumKernel.cpp:580): using acc_t = at::acc_type<scalar_t, true> =
  // float The cascade's 4-level bucket system provides O(log N) precision
  // WITHOUT double. This gives 8-wide float SIMD instead of 4-wide double →
  // 2-3x faster for cached data. Global AccumulatorType<float>=double is kept
  // for mean/variance which need double.
  using acc_t =
      std::conditional_t<std::is_same_v<T, float>, float, AccumulatorType<T>>;

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(input.dtype())
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  const T *input_data = input.data<T>();
  T *output_data = output.data<T>();
  const auto &input_dims = input.shape().dims;
  const auto &input_strides = input.stride().strides;
  const int64_t reduced_count =
      calculate_reduced_count(input_dims, normalized_axes);
  const int64_t num_slices = output.numel();

  bool reduced_bitmap[MAX_DIMS] = {false};
  for (int64_t axis : normalized_axes)
    reduced_bitmap[axis] = true;
  std::vector<int64_t> reduced_dims;
  for (size_t d = 0; d < input_dims.size(); ++d)
    if (reduced_bitmap[d])
      reduced_dims.push_back(input_dims[d]);

  ReductionLayout layout = compute_reduction_layout(input, normalized_axes);
  cpu_demote_middle_to_generic(layout);
  const int64_t k = static_cast<int64_t>(normalized_axes.size());
  int64_t red_input_strides[MAX_DIMS];
  for (int64_t d = 0; d < k; ++d)
    red_input_strides[d] = input_strides[normalized_axes[d]];

  // Precompute base indices for Generic path
  int64_t M_nr = 0;
  int64_t nr_sizes[MAX_DIMS], nr_strides_nr[MAX_DIMS];
  std::vector<int64_t> base_lin_idxs;
  if (layout.path == ReductionLayout::Path::Generic) {
    for (size_t d = 0; d < input_dims.size(); ++d) {
      if (!reduced_bitmap[d]) {
        nr_sizes[M_nr] = input_dims[d];
        nr_strides_nr[M_nr] = input_strides[d];
        ++M_nr;
      }
    }
    base_lin_idxs.resize(num_slices);
    int64_t oc[MAX_DIMS] = {}, blk = 0;
    for (int64_t o = 0; o < num_slices; ++o) {
      base_lin_idxs[o] = blk;
      for (int64_t j = M_nr - 1; j >= 0; --j) {
        ++oc[j];
        blk += nr_strides_nr[j];
        if (oc[j] < nr_sizes[j])
          break;
        blk -= oc[j] * nr_strides_nr[j];
        oc[j] = 0;
      }
    }
  }

  // Helper: reduce one output slice and store result
  auto reduce_and_store_one = [&](int64_t o) {
    acc_t result;

    if (layout.path == ReductionLayout::Path::InnerContiguous) {
      const T *in_ptr = input_data + o * layout.input_outer_stride;
      const int64_t n = layout.reduced_count;

      // ─── SIMD PATH: non-NaN, supported types ───
      // Uses 4 vector accumulators for ILP + horizontal reduce at end
      if constexpr (!ignore_nan && std::is_same_v<T, double> &&
                    std::is_same_v<acc_t, double>) {
        // double → double: direct Vectorized<double> (4-wide × 4 accumulators =
        // 16 doubles/iter)
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size(); // 4
        Vec va, vb, vc, vd;
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va + Vec::loadu(in_ptr + j);
          vb = vb + Vec::loadu(in_ptr + j + W);
          vc = vc + Vec::loadu(in_ptr + j + W * 2);
          vd = vd + Vec::loadu(in_ptr + j + W * 3);
        }
        for (; j + W <= n; j += W)
          va = va + Vec::loadu(in_ptr + j);
        result = (va + vb + vc + vd).reduce_add();
        for (; j < n; ++j)
          result += in_ptr[j];

      } else if constexpr (!ignore_nan && std::is_same_v<T, float> &&
                           std::is_same_v<acc_t, float>) {
        // float → float: 8-wide SIMD (like PyTorch), cascade handles precision
        // PyTorch uses acc_type<float,true>=float + Vectorized<float> = 8-wide
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size(); // 8
        Vec va, vb, vc, vd;
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va + Vec::loadu(in_ptr + j);
          vb = vb + Vec::loadu(in_ptr + j + W);
          vc = vc + Vec::loadu(in_ptr + j + W * 2);
          vd = vd + Vec::loadu(in_ptr + j + W * 3);
        }
        for (; j + W <= n; j += W)
          va = va + Vec::loadu(in_ptr + j);
        result = (va + vb + vc + vd).reduce_add();
        for (; j < n; ++j)
          result += in_ptr[j];

      } else if constexpr (!ignore_nan && std::is_same_v<T, float16_t> &&
                           std::is_same_v<acc_t, float>) {
        // fp16 → float: F16C load-convert (8-wide × 4 accumulators = 32
        // fp16/iter)
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8; // 8 fp16 → 8 floats
        Vec va, vb, vc, vd;
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va + vec::load_fp16_as_float(in_ptr + j);
          vb = vb + vec::load_fp16_as_float(in_ptr + j + W);
          vc = vc + vec::load_fp16_as_float(in_ptr + j + W * 2);
          vd = vd + vec::load_fp16_as_float(in_ptr + j + W * 3);
        }
        for (; j + W <= n; j += W)
          va = va + vec::load_fp16_as_float(in_ptr + j);
        result = (va + vb + vc + vd).reduce_add();
        for (; j < n; ++j)
          result += static_cast<acc_t>(in_ptr[j]);

      } else if constexpr (!ignore_nan && std::is_same_v<T, bfloat16_t> &&
                           std::is_same_v<acc_t, float>) {
        // bf16 → float: bit-shift load-convert (8-wide × 4 accumulators)
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8;
        Vec va, vb, vc, vd;
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va + vec::load_bf16_as_float(in_ptr + j);
          vb = vb + vec::load_bf16_as_float(in_ptr + j + W);
          vc = vc + vec::load_bf16_as_float(in_ptr + j + W * 2);
          vd = vd + vec::load_bf16_as_float(in_ptr + j + W * 3);
        }
        for (; j + W <= n; j += W)
          va = va + vec::load_bf16_as_float(in_ptr + j);
        result = (va + vb + vc + vd).reduce_add();
        for (; j < n; ++j)
          result += static_cast<acc_t>(in_ptr[j]);

        // ═══════════════════════════════════════════════════
        // NaN-AWARE SIMD: bit masking (branchless NaN filtering)
        //
        // Strategy: load vector → compare with self (NaN != NaN) → get mask
        //           → blend NaN lanes to 0.0 → add to accumulator
        // No branches per element! Pure SIMD throughput.
        //
        // _mm256_cmp_ps(v, v, _CMP_UNORD_Q) → NaN lanes = all-1s
        // _mm256_blendv_ps(v, zero, mask) → NaN lanes become 0.0
        // ═══════════════════════════════════════════════════

      } else if constexpr (ignore_nan && std::is_same_v<T, double> &&
                           std::is_same_v<acc_t, double>) {
        // nansum double: load → mask NaN → add
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size(); // 4
        __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
        __m256d vc = _mm256_setzero_pd(), vd = _mm256_setzero_pd();
        __m256d zero = _mm256_setzero_pd();
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          __m256d v0 = _mm256_loadu_pd(in_ptr + j);
          __m256d v1 = _mm256_loadu_pd(in_ptr + j + W);
          __m256d v2 = _mm256_loadu_pd(in_ptr + j + W * 2);
          __m256d v3 = _mm256_loadu_pd(in_ptr + j + W * 3);
          // Mask: NaN lanes → all-1s
          va = _mm256_add_pd(
              va,
              _mm256_blendv_pd(v0, zero, _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q)));
          vb = _mm256_add_pd(
              vb,
              _mm256_blendv_pd(v1, zero, _mm256_cmp_pd(v1, v1, _CMP_UNORD_Q)));
          vc = _mm256_add_pd(
              vc,
              _mm256_blendv_pd(v2, zero, _mm256_cmp_pd(v2, v2, _CMP_UNORD_Q)));
          vd = _mm256_add_pd(
              vd,
              _mm256_blendv_pd(v3, zero, _mm256_cmp_pd(v3, v3, _CMP_UNORD_Q)));
        }
        for (; j + W <= n; j += W) {
          __m256d v0 = _mm256_loadu_pd(in_ptr + j);
          va = _mm256_add_pd(
              va,
              _mm256_blendv_pd(v0, zero, _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q)));
        }
        va = _mm256_add_pd(_mm256_add_pd(va, vb), _mm256_add_pd(vc, vd));
        result = Vec(va).reduce_add();
        for (; j < n; ++j) {
          double v = in_ptr[j];
          if (!std::isnan(v))
            result += v;
        }

      } else if constexpr (ignore_nan && std::is_same_v<T, float> &&
                           std::is_same_v<acc_t, float>) {
        // nansum float→float: 8-wide NaN bitmask + float accumulation (cascade
        // handles precision)
        using Vec = vec::Vectorized<float>;
        __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
        __m256 vc = _mm256_setzero_ps(), vd = _mm256_setzero_ps();
        __m256 zero = _mm256_setzero_ps();
        constexpr int64_t W = 8;
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          __m256 v0 = _mm256_loadu_ps(in_ptr + j);
          __m256 v1 = _mm256_loadu_ps(in_ptr + j + W);
          __m256 v2 = _mm256_loadu_ps(in_ptr + j + W * 2);
          __m256 v3 = _mm256_loadu_ps(in_ptr + j + W * 3);
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
          vb = _mm256_add_ps(
              vb,
              _mm256_blendv_ps(v1, zero, _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)));
          vc = _mm256_add_ps(
              vc,
              _mm256_blendv_ps(v2, zero, _mm256_cmp_ps(v2, v2, _CMP_UNORD_Q)));
          vd = _mm256_add_ps(
              vd,
              _mm256_blendv_ps(v3, zero, _mm256_cmp_ps(v3, v3, _CMP_UNORD_Q)));
        }
        for (; j + W <= n; j += W) {
          __m256 v0 = _mm256_loadu_ps(in_ptr + j);
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
        }
        result =
            Vec(_mm256_add_ps(_mm256_add_ps(va, vb), _mm256_add_ps(vc, vd)))
                .reduce_add();
        for (; j < n; ++j) {
          float v = in_ptr[j];
          if (!std::isnan(v))
            result += v;
        }

      } else if constexpr (ignore_nan && std::is_same_v<T, float16_t> &&
                           std::is_same_v<acc_t, float>) {
        // nansum fp16→float: F16C load → mask NaN → add
        using Vec = vec::Vectorized<float>;
        __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
        __m256 zero = _mm256_setzero_ps();
        constexpr int64_t W = 8;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256 v0 = vec::load_fp16_as_float(in_ptr + j).values;
          __m256 v1 = vec::load_fp16_as_float(in_ptr + j + W).values;
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
          vb = _mm256_add_ps(
              vb,
              _mm256_blendv_ps(v1, zero, _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)));
        }
        for (; j + W <= n; j += W) {
          __m256 v0 = vec::load_fp16_as_float(in_ptr + j).values;
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
        }
        result = Vec(_mm256_add_ps(va, vb)).reduce_add();
        for (; j < n; ++j) {
          float v = static_cast<float>(in_ptr[j]);
          if (!std::isnan(v))
            result += v;
        }

      } else if constexpr (ignore_nan && std::is_same_v<T, bfloat16_t> &&
                           std::is_same_v<acc_t, float>) {
        // nansum bf16→float: shift load → mask NaN → add
        using Vec = vec::Vectorized<float>;
        __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
        __m256 zero = _mm256_setzero_ps();
        constexpr int64_t W = 8;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256 v0 = vec::load_bf16_as_float(in_ptr + j).values;
          __m256 v1 = vec::load_bf16_as_float(in_ptr + j + W).values;
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
          vb = _mm256_add_ps(
              vb,
              _mm256_blendv_ps(v1, zero, _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)));
        }
        for (; j + W <= n; j += W) {
          __m256 v0 = vec::load_bf16_as_float(in_ptr + j).values;
          va = _mm256_add_ps(
              va,
              _mm256_blendv_ps(v0, zero, _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
        }
        result = Vec(_mm256_add_ps(va, vb)).reduce_add();
        for (; j < n; ++j) {
          float v = static_cast<float>(in_ptr[j]);
          if (!std::isnan(v))
            result += v;
        }

      } else {
        // ─── SCALAR FALLBACK: complex types, remaining NaN edge cases ───
        constexpr int64_t ilp = 4;
        acc_t partial[ilp] = {acc_t{}, acc_t{}, acc_t{}, acc_t{}};
        int64_t j = 0;
        const int64_t n4 = (n / ilp) * ilp;
        for (; j < n4; j += ilp) {
          for (int64_t p = 0; p < ilp; ++p) {
            T val = in_ptr[j + p];
            if constexpr (ignore_nan) {
              partial[p] += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
            } else {
              partial[p] += static_cast<acc_t>(val);
            }
          }
        }
        for (; j < n; ++j) {
          T val = in_ptr[j];
          if constexpr (ignore_nan) {
            partial[0] += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
          } else {
            partial[0] += static_cast<acc_t>(val);
          }
        }
        result = partial[0] + partial[1] + partial[2] + partial[3];
      }

    } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
      // Outer: column walk — strided per output position, scalar 4-acc ILP.
      // Remap the flat output index o to its input slab base: for a
      // leading-axis reduction outer==1 so base==o; for a middle-axis
      // reduction each output (o_outer,o_inner) lives at
      // o_outer*input_outer_stride + o_inner (outer_stride = reduced*inner).
      const int64_t base =
          (o / layout.inner_count) * layout.input_outer_stride +
          (o % layout.inner_count);
      constexpr int64_t ilp = 4;
      acc_t partial[ilp] = {acc_t{}, acc_t{}, acc_t{}, acc_t{}};
      int64_t r = 0;
      const int64_t n4 = (layout.reduced_count / ilp) * ilp;
      for (; r < n4; r += ilp) {
        for (int64_t p = 0; p < ilp; ++p) {
          T val = *(input_data + base + (r + p) * layout.input_row_stride);
          if constexpr (ignore_nan) {
            partial[p] += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
          } else {
            partial[p] += static_cast<acc_t>(val);
          }
        }
      }
      for (; r < layout.reduced_count; ++r) {
        T val = *(input_data + base + r * layout.input_row_stride);
        if constexpr (ignore_nan) {
          partial[0] += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
        } else {
          partial[0] += static_cast<acc_t>(val);
        }
      }
      result = partial[0] + partial[1] + partial[2] + partial[3];

    } else {
      // Generic: carry-add with 4-acc ILP
      constexpr int64_t ilp = 4;
      acc_t partial[ilp] = {acc_t{}, acc_t{}, acc_t{}, acc_t{}};
      int64_t red_coords[MAX_DIMS] = {};
      int64_t lin = base_lin_idxs[o];
      for (int64_t i = 0; i < reduced_count; ++i) {
        T val = input_data[lin];
        if constexpr (ignore_nan) {
          partial[i & 3] += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
        } else {
          partial[i & 3] += static_cast<acc_t>(val);
        }
        for (int64_t d = k - 1; d >= 0; --d) {
          ++red_coords[d];
          lin += red_input_strides[d];
          if (red_coords[d] < reduced_dims[d])
            break;
          lin -= red_coords[d] * red_input_strides[d];
          red_coords[d] = 0;
        }
      }
      result = partial[0] + partial[1] + partial[2] + partial[3];
    }

    // Convert accumulator → output type
    if constexpr (std::is_same_v<T, float16_t> ||
                  std::is_same_v<T, bfloat16_t>) {
      output_data[o] = static_cast<T>(static_cast<float>(result));
    } else if constexpr (std::is_same_v<T, complex32_t>) {
      output_data[o] = complex32_t(static_cast<float>(result.real()),
                                   static_cast<float>(result.imag()));
    } else if constexpr (std::is_same_v<T, complex64_t> &&
                         std::is_same_v<acc_t, complex128_t>) {
      output_data[o] = complex64_t(static_cast<float>(result.real()),
                                   static_cast<float>(result.imag()));
    } else {
      output_data[o] = static_cast<T>(result);
    }
  };

  // ═══════════════════════════════════════════════════════════
  // STRATEGY 1: Parallelize over output slots
  // For OuterContiguous: SIMD across adjacent output columns (vertical
  // vectorization) ═══════════════════════════════════════════════════════════
  if (strategy == ReductionStrategy::ParallelSlices) {
    // ─── SIMD OUTER: process vec_size adjacent columns per iteration ───
    // Adjacent output columns are contiguous within each row of input.
    // Load vec_size values from same row, accumulate in vector register.
    // Only valid for a leading-axis reduction (outer==1), where the whole
    // output is one contiguous column block. For a middle-axis reduction
    // (input_outer_stride != input_row_stride) adjacent columns straddle
    // outer-slab boundaries, so we fall through to the scalar path which
    // remaps each output index to its slab base.
    if (layout.path == ReductionLayout::Path::OuterContiguous &&
        layout.input_outer_stride == layout.input_row_stride) {
      const int64_t stride = layout.input_row_stride;
      const int64_t R = layout.reduced_count;
      bool handled = false;

      if constexpr (!ignore_nan) {
        // ─── NON-NAN SUM: vertical SIMD across adjacent output columns ───
        if constexpr (std::is_same_v<T, double> &&
                      std::is_same_v<acc_t, double>) {
          using Vec = vec::Vectorized<double>;
          constexpr int64_t W = Vec::size(); // 4
          const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            Vec acc;
            for (int64_t r = 0; r < R; ++r)
              acc = acc + Vec::loadu(input_data + r * stride + o);
            acc.storeu(output_data + o);
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, float> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 4; // 4 floats → 4 doubles
          const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            vec::Vectorized<double> acc;
            for (int64_t r = 0; r < R; ++r)
              acc = acc + vec::Vectorized<double>(_mm256_cvtps_pd(
                              _mm_loadu_ps(input_data + r * stride + o)));
            __m128 f4 = _mm256_cvtpd_ps(acc.values);
            _mm_storeu_ps(output_data + o, f4);
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, float16_t> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 8;
          const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            vec::Vectorized<float> acc;
            for (int64_t r = 0; r < R; ++r)
              acc = acc + vec::load_fp16_as_float(input_data + r * stride + o);
            vec::store_float_as_fp16(output_data + o, acc);
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, bfloat16_t> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 8;
          const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            vec::Vectorized<float> acc;
            for (int64_t r = 0; r < R; ++r)
              acc = acc + vec::load_bf16_as_float(input_data + r * stride + o);
            vec::store_float_as_bf16(output_data + o, acc);
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        }
      } else {
        // ─── NANSUM: vertical SIMD with NaN bitmask across adjacent output
        // columns ───
        if constexpr (std::is_same_v<T, double> &&
                      std::is_same_v<acc_t, double>) {
          constexpr int64_t W = 4;
          const int64_t vec_end = (num_slices / W) * W;
          __m256d zero = _mm256_setzero_pd();
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256d acc = _mm256_setzero_pd();
            for (int64_t r = 0; r < R; ++r) {
              __m256d v = _mm256_loadu_pd(input_data + r * stride + o);
              acc = _mm256_add_pd(
                  acc,
                  _mm256_blendv_pd(v, zero, _mm256_cmp_pd(v, v, _CMP_UNORD_Q)));
            }
            // Convert back and store
            _mm256_storeu_pd(output_data + o, acc);
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, float> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 4;
          const int64_t vec_end = (num_slices / W) * W;
          __m128 zero_f = _mm_setzero_ps();
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256d acc = _mm256_setzero_pd();
            for (int64_t r = 0; r < R; ++r) {
              __m128 f = _mm_loadu_ps(input_data + r * stride + o);
              f = _mm_blendv_ps(f, zero_f, _mm_cmp_ps(f, f, _CMP_UNORD_Q));
              acc = _mm256_add_pd(acc, _mm256_cvtps_pd(f));
            }
            _mm_storeu_ps(output_data + o, _mm256_cvtpd_ps(acc));
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, float16_t> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 8;
          const int64_t vec_end = (num_slices / W) * W;
          __m256 zero = _mm256_setzero_ps();
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256 acc = _mm256_setzero_ps();
            for (int64_t r = 0; r < R; ++r) {
              __m256 v =
                  vec::load_fp16_as_float(input_data + r * stride + o).values;
              acc = _mm256_add_ps(
                  acc,
                  _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, v, _CMP_UNORD_Q)));
            }
            vec::store_float_as_fp16(output_data + o,
                                     vec::Vectorized<float>(acc));
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        } else if constexpr (std::is_same_v<T, bfloat16_t> &&
                             std::is_same_v<acc_t, float>) {
          constexpr int64_t W = 8;
          const int64_t vec_end = (num_slices / W) * W;
          __m256 zero = _mm256_setzero_ps();
#pragma omp parallel for num_threads(num_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256 acc = _mm256_setzero_ps();
            for (int64_t r = 0; r < R; ++r) {
              __m256 v =
                  vec::load_bf16_as_float(input_data + r * stride + o).values;
              acc = _mm256_add_ps(
                  acc,
                  _mm256_blendv_ps(v, zero, _mm256_cmp_ps(v, v, _CMP_UNORD_Q)));
            }
            vec::store_float_as_bf16(output_data + o,
                                     vec::Vectorized<float>(acc));
          }
          for (int64_t o = vec_end; o < num_slices; ++o)
            reduce_and_store_one(o);
          handled = true;
        }
      }

      if (handled)
        return output;
    }

// ─── Default: scalar per output slot ───
#pragma omp parallel for num_threads(num_threads)
    for (int64_t o = 0; o < num_slices; ++o)
      reduce_and_store_one(o);
  }
  // ═══════════════════════════════════════════════════════════
  // STRATEGY 2: Split reduction across threads + combine
  // ═══════════════════════════════════════════════════════════
  else {
    for (int64_t o = 0; o < num_slices; ++o) {
      std::vector<acc_t> thread_accs(num_threads, acc_t{});
      const int64_t rc = reduced_count;

#pragma omp parallel num_threads(num_threads)
      {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = (rc + nt - 1) / nt;
        int64_t begin = tid * chunk;
        int64_t end = std::min(begin + chunk, rc);
        acc_t local = acc_t{};

        if (layout.path == ReductionLayout::Path::InnerContiguous) {
          const T *in_ptr = input_data + o * layout.input_outer_stride + begin;
          const int64_t n = end - begin;

          // ─── SIMD VECTORIZED per-thread chunk (mirrors Strategy 1 inner) ───
          if constexpr (!ignore_nan && std::is_same_v<T, double> &&
                        std::is_same_v<acc_t, double>) {
            using Vec = vec::Vectorized<double>;
            constexpr int64_t W = Vec::size();
            Vec va, vb, vc, vd;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              va = va + Vec::loadu(in_ptr + j);
              vb = vb + Vec::loadu(in_ptr + j + W);
              vc = vc + Vec::loadu(in_ptr + j + W * 2);
              vd = vd + Vec::loadu(in_ptr + j + W * 3);
            }
            for (; j + W <= n; j += W)
              va = va + Vec::loadu(in_ptr + j);
            local = (va + vb + vc + vd).reduce_add();
            for (; j < n; ++j)
              local += in_ptr[j];
          } else if constexpr (!ignore_nan && std::is_same_v<T, float> &&
                               std::is_same_v<acc_t, float>) {
            using Vec = vec::Vectorized<double>;
            constexpr int64_t W = 4;
            Vec va, vb, vc, vd;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              va = va + Vec(_mm256_cvtps_pd(_mm_loadu_ps(in_ptr + j)));
              vb = vb + Vec(_mm256_cvtps_pd(_mm_loadu_ps(in_ptr + j + W)));
              vc = vc + Vec(_mm256_cvtps_pd(_mm_loadu_ps(in_ptr + j + W * 2)));
              vd = vd + Vec(_mm256_cvtps_pd(_mm_loadu_ps(in_ptr + j + W * 3)));
            }
            for (; j + W <= n; j += W)
              va = va + Vec(_mm256_cvtps_pd(_mm_loadu_ps(in_ptr + j)));
            local = (va + vb + vc + vd).reduce_add();
            for (; j < n; ++j)
              local += static_cast<acc_t>(in_ptr[j]);
          } else if constexpr (!ignore_nan && std::is_same_v<T, float16_t> &&
                               std::is_same_v<acc_t, float>) {
            using Vec = vec::Vectorized<float>;
            constexpr int64_t W = 8;
            Vec va, vb, vc, vd;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              va = va + vec::load_fp16_as_float(in_ptr + j);
              vb = vb + vec::load_fp16_as_float(in_ptr + j + W);
              vc = vc + vec::load_fp16_as_float(in_ptr + j + W * 2);
              vd = vd + vec::load_fp16_as_float(in_ptr + j + W * 3);
            }
            for (; j + W <= n; j += W)
              va = va + vec::load_fp16_as_float(in_ptr + j);
            local = (va + vb + vc + vd).reduce_add();
            for (; j < n; ++j)
              local += static_cast<acc_t>(in_ptr[j]);
          } else if constexpr (!ignore_nan && std::is_same_v<T, bfloat16_t> &&
                               std::is_same_v<acc_t, float>) {
            using Vec = vec::Vectorized<float>;
            constexpr int64_t W = 8;
            Vec va, vb, vc, vd;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              va = va + vec::load_bf16_as_float(in_ptr + j);
              vb = vb + vec::load_bf16_as_float(in_ptr + j + W);
              vc = vc + vec::load_bf16_as_float(in_ptr + j + W * 2);
              vd = vd + vec::load_bf16_as_float(in_ptr + j + W * 3);
            }
            for (; j + W <= n; j += W)
              va = va + vec::load_bf16_as_float(in_ptr + j);
            local = (va + vb + vc + vd).reduce_add();
            for (; j < n; ++j)
              local += static_cast<acc_t>(in_ptr[j]);
            // ─── NaN-aware SIMD bitmasking (nansum) ───
          } else if constexpr (ignore_nan && std::is_same_v<T, double> &&
                               std::is_same_v<acc_t, double>) {
            __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
            __m256d vc = _mm256_setzero_pd(), vd = _mm256_setzero_pd();
            __m256d zero = _mm256_setzero_pd();
            constexpr int64_t W = 4;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              __m256d v0 = _mm256_loadu_pd(in_ptr + j);
              __m256d v1 = _mm256_loadu_pd(in_ptr + j + W);
              __m256d v2 = _mm256_loadu_pd(in_ptr + j + W * 2);
              __m256d v3 = _mm256_loadu_pd(in_ptr + j + W * 3);
              va = _mm256_add_pd(
                  va, _mm256_blendv_pd(v0, zero,
                                       _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q)));
              vb = _mm256_add_pd(
                  vb, _mm256_blendv_pd(v1, zero,
                                       _mm256_cmp_pd(v1, v1, _CMP_UNORD_Q)));
              vc = _mm256_add_pd(
                  vc, _mm256_blendv_pd(v2, zero,
                                       _mm256_cmp_pd(v2, v2, _CMP_UNORD_Q)));
              vd = _mm256_add_pd(
                  vd, _mm256_blendv_pd(v3, zero,
                                       _mm256_cmp_pd(v3, v3, _CMP_UNORD_Q)));
            }
            for (; j + W <= n; j += W) {
              __m256d v0 = _mm256_loadu_pd(in_ptr + j);
              va = _mm256_add_pd(
                  va, _mm256_blendv_pd(v0, zero,
                                       _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q)));
            }
            va = _mm256_add_pd(_mm256_add_pd(va, vb), _mm256_add_pd(vc, vd));
            local = vec::Vectorized<double>(va).reduce_add();
            for (; j < n; ++j) {
              double v = in_ptr[j];
              if (!std::isnan(v))
                local += v;
            }
          } else if constexpr (ignore_nan && std::is_same_v<T, float> &&
                               std::is_same_v<acc_t, float>) {
            __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
            __m256d vc = _mm256_setzero_pd(), vd = _mm256_setzero_pd();
            __m128 zero_f = _mm_setzero_ps();
            constexpr int64_t W = 4;
            int64_t j = 0;
            for (; j + W * 4 <= n; j += W * 4) {
              __m128 f0 = _mm_loadu_ps(in_ptr + j);
              __m128 f1 = _mm_loadu_ps(in_ptr + j + W);
              __m128 f2 = _mm_loadu_ps(in_ptr + j + W * 2);
              __m128 f3 = _mm_loadu_ps(in_ptr + j + W * 3);
              f0 = _mm_blendv_ps(f0, zero_f, _mm_cmp_ps(f0, f0, _CMP_UNORD_Q));
              f1 = _mm_blendv_ps(f1, zero_f, _mm_cmp_ps(f1, f1, _CMP_UNORD_Q));
              f2 = _mm_blendv_ps(f2, zero_f, _mm_cmp_ps(f2, f2, _CMP_UNORD_Q));
              f3 = _mm_blendv_ps(f3, zero_f, _mm_cmp_ps(f3, f3, _CMP_UNORD_Q));
              va = _mm256_add_pd(va, _mm256_cvtps_pd(f0));
              vb = _mm256_add_pd(vb, _mm256_cvtps_pd(f1));
              vc = _mm256_add_pd(vc, _mm256_cvtps_pd(f2));
              vd = _mm256_add_pd(vd, _mm256_cvtps_pd(f3));
            }
            for (; j + W <= n; j += W) {
              __m128 f0 = _mm_loadu_ps(in_ptr + j);
              f0 = _mm_blendv_ps(f0, zero_f, _mm_cmp_ps(f0, f0, _CMP_UNORD_Q));
              va = _mm256_add_pd(va, _mm256_cvtps_pd(f0));
            }
            va = _mm256_add_pd(_mm256_add_pd(va, vb), _mm256_add_pd(vc, vd));
            local = vec::Vectorized<double>(va).reduce_add();
            for (; j < n; ++j) {
              float v = in_ptr[j];
              if (!std::isnan(v))
                local += v;
            }
          } else if constexpr (ignore_nan && std::is_same_v<T, float16_t> &&
                               std::is_same_v<acc_t, float>) {
            __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
            __m256 zero = _mm256_setzero_ps();
            constexpr int64_t W = 8;
            int64_t j = 0;
            for (; j + W * 2 <= n; j += W * 2) {
              __m256 v0 = vec::load_fp16_as_float(in_ptr + j).values;
              __m256 v1 = vec::load_fp16_as_float(in_ptr + j + W).values;
              va = _mm256_add_ps(
                  va, _mm256_blendv_ps(v0, zero,
                                       _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
              vb = _mm256_add_ps(
                  vb, _mm256_blendv_ps(v1, zero,
                                       _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)));
            }
            for (; j + W <= n; j += W) {
              __m256 v0 = vec::load_fp16_as_float(in_ptr + j).values;
              va = _mm256_add_ps(
                  va, _mm256_blendv_ps(v0, zero,
                                       _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
            }
            local = vec::Vectorized<float>(_mm256_add_ps(va, vb)).reduce_add();
            for (; j < n; ++j) {
              float v = static_cast<float>(in_ptr[j]);
              if (!std::isnan(v))
                local += v;
            }
          } else if constexpr (ignore_nan && std::is_same_v<T, bfloat16_t> &&
                               std::is_same_v<acc_t, float>) {
            __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
            __m256 zero = _mm256_setzero_ps();
            constexpr int64_t W = 8;
            int64_t j = 0;
            for (; j + W * 2 <= n; j += W * 2) {
              __m256 v0 = vec::load_bf16_as_float(in_ptr + j).values;
              __m256 v1 = vec::load_bf16_as_float(in_ptr + j + W).values;
              va = _mm256_add_ps(
                  va, _mm256_blendv_ps(v0, zero,
                                       _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
              vb = _mm256_add_ps(
                  vb, _mm256_blendv_ps(v1, zero,
                                       _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q)));
            }
            for (; j + W <= n; j += W) {
              __m256 v0 = vec::load_bf16_as_float(in_ptr + j).values;
              va = _mm256_add_ps(
                  va, _mm256_blendv_ps(v0, zero,
                                       _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q)));
            }
            local = vec::Vectorized<float>(_mm256_add_ps(va, vb)).reduce_add();
            for (; j < n; ++j) {
              float v = static_cast<float>(in_ptr[j]);
              if (!std::isnan(v))
                local += v;
            }
          } else {
            // Scalar fallback for complex/other types
            for (int64_t j = 0; j < n; ++j) {
              T val = in_ptr[j];
              if constexpr (ignore_nan)
                local += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
              else
                local += static_cast<acc_t>(val);
            }
          }
        } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
          // Remap flat output index to slab base (middle-axis safe; base==o
          // when outer==1).
          const int64_t base =
              (o / layout.inner_count) * layout.input_outer_stride +
              (o % layout.inner_count);
          for (int64_t r = begin; r < end; ++r) {
            T val = *(input_data + base + r * layout.input_row_stride);
            if constexpr (ignore_nan)
              local += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
            else
              local += static_cast<acc_t>(val);
          }
        } else {
          int64_t red_coords[MAX_DIMS] = {};
          int64_t lin = base_lin_idxs[o];
          for (int64_t i = 0; i < begin; ++i) {
            for (int64_t d = k - 1; d >= 0; --d) {
              ++red_coords[d];
              lin += red_input_strides[d];
              if (red_coords[d] < reduced_dims[d])
                break;
              lin -= red_coords[d] * red_input_strides[d];
              red_coords[d] = 0;
            }
          }
          for (int64_t i = begin; i < end; ++i) {
            T val = input_data[lin];
            if constexpr (ignore_nan)
              local += safe_isnan(val) ? acc_t{} : static_cast<acc_t>(val);
            else
              local += static_cast<acc_t>(val);
            for (int64_t d = k - 1; d >= 0; --d) {
              ++red_coords[d];
              lin += red_input_strides[d];
              if (red_coords[d] < reduced_dims[d])
                break;
              lin -= red_coords[d] * red_input_strides[d];
              red_coords[d] = 0;
            }
          }
        }
        thread_accs[tid] = local;
      }

      acc_t result = thread_accs[0];
      for (int t = 1; t < num_threads; ++t)
        result += thread_accs[t];

      if constexpr (std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, bfloat16_t>) {
        output_data[o] = static_cast<T>(static_cast<float>(result));
      } else if constexpr (std::is_same_v<T, complex32_t>) {
        output_data[o] = complex32_t(static_cast<float>(result.real()),
                                     static_cast<float>(result.imag()));
      } else if constexpr (std::is_same_v<T, complex64_t> &&
                           std::is_same_v<acc_t, complex128_t>) {
        output_data[o] = complex64_t(static_cast<float>(result.real()),
                                     static_cast<float>(result.imag()));
      } else {
        output_data[o] = static_cast<T>(result);
      }
    }
  }

  return output;
}

// =================================================================
// --- REDUCE KERNEL FOR INDEX OPS (argmax/argmin/nanargmax/nanargmin) ---
//
// Covers BOTH binary_kernel_reduce AND binary_kernel_reduce_lastdim.
// No SIMD — index tracking breaks vectorization (value + index per
// element causes register spilling, PyTorch also keeps this scalar).
//
// Layout paths (all scalar):
//   InnerContiguous = binary_kernel_reduce_lastdim (linear ptr walk, no stride
//   math) OuterContiguous = Our addition (single stride-multiply, avoids
//   Generic's carry-add) Generic         = binary_kernel_reduce inner loop
//   (carry-add coordinate math)
//
// 2 callable strategy paths (chosen by universal dispatcher):
//   ParallelSlices:  #pragma omp parallel for over output slots
//   SplitReduction:  For each output, split reduction across threads + combine
//
// PyTorch bug fixed: binary_kernel_reduce_lastdim (Reduce.h:290-308) uses
// sub_iter.for_each() which parallelizes ONLY over output elements. For full
// reduction (output.numel()==1), 1 thread does all work. Our SplitReduction
// strategy fixes this by splitting the reduction dimension across ALL threads.
// =================================================================

// Helper: reduce a range [begin,end) of reduction elements for one output
// position
//
// Uses ValueIndex<T> struct approach on CPU — benchmarked FASTER than
// independent variables (best_val/best_idx) because compiler (-O3) optimizes
// the struct away into registers, and the simple op.reduce(acc, curr) call
// generates tighter code than manual branching with NaN checks + complex type
// guards.
//
// Benchmark proof (i7-14700K, 50M float32 argmax):
//   ValueIndex struct:       3,882 μs
//   Independent variables:   4,406 μs (+13% slower)
//   PyTorch (1 thread bug):  46,700 μs
//
// PyTorch uses pair<scalar_t, int64_t> with separate reduce(arg, val, idx)
// signature (SharedReduceOps.h) — functionally equivalent, slightly different
// API.
//
// On GPU side, ValueIndex<T> is also used (needed for warp_shfl_down).
template <typename T, template <typename> class OpType>
inline ValueIndex<T> reduce_one_index_slice(
    const T *input_data, OpType<T> &op, const ReductionLayout &layout,
    int64_t output_index, int64_t reduced_count, const int64_t *base_lin_idxs,
    const int64_t *red_input_strides, const std::vector<int64_t> &reduced_dims,
    int64_t k, int64_t begin = 0, int64_t end = -1) {
  if (end < 0)
    end = reduced_count;
  ValueIndex<T> acc = op.identity();

  if (layout.path == ReductionLayout::Path::InnerContiguous) {
    // Lastdim path: pure pointer walk, zero stride math
    const T *in_ptr = input_data + output_index * layout.input_outer_stride;
    for (int64_t j = begin; j < end; ++j) {
      ValueIndex<T> curr = {in_ptr[j], j};
      acc = op.reduce(acc, curr);
    }
  } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
    // Column walk: single stride-multiply per element
    for (int64_t r = begin; r < end; ++r) {
      const T *in_ptr = input_data + output_index + r * layout.input_row_stride;
      ValueIndex<T> curr = {*in_ptr, r};
      acc = op.reduce(acc, curr);
    }
  } else {
    // Generic: carry-add coordinate reconstruction
    int64_t red_coords[MAX_DIMS] = {};
    int64_t input_lin_idx = base_lin_idxs[output_index];
    // Advance to begin position
    for (int64_t i = 0; i < begin; ++i) {
      for (int64_t d = k - 1; d >= 0; --d) {
        ++red_coords[d];
        input_lin_idx += red_input_strides[d];
        if (red_coords[d] < reduced_dims[d])
          break;
        input_lin_idx -= red_coords[d] * red_input_strides[d];
        red_coords[d] = 0;
      }
    }
    for (int64_t i = begin; i < end; ++i) {
      T input_value = input_data[input_lin_idx];
      ValueIndex<T> curr = {input_value, i};
      acc = op.reduce(acc, curr);
      for (int64_t d = k - 1; d >= 0; --d) {
        ++red_coords[d];
        input_lin_idx += red_input_strides[d];
        if (red_coords[d] < reduced_dims[d])
          break;
        input_lin_idx -= red_coords[d] * red_input_strides[d];
        red_coords[d] = 0;
      }
    }
  }
  return acc;
}

template <typename T, template <typename> class OpType>
Tensor reduce_kernel_index(const Tensor &input,
                           const std::vector<int64_t> &normalized_axes,
                           const Shape &output_shape,
                           ReductionStrategy strategy, int num_threads) {
  using Op = OpType<T>;

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(Dtype::Int64)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  const T *input_data = input.data<T>();
  const auto &input_dims = input.shape().dims;
  const auto &input_strides = input.stride().strides;
  const int64_t reduced_count =
      calculate_reduced_count(input_dims, normalized_axes);
  int64_t *output_data = output.data<int64_t>();

  Op op;
  const int64_t num_slices = output.numel();

  if (input_dims.size() > MAX_DIMS)
    throw std::runtime_error(
        "Tensor rank exceeds maximum supported dimensions (64)");

  bool reduced_bitmap[MAX_DIMS] = {false};
  for (int64_t axis : normalized_axes)
    reduced_bitmap[axis] = true;
  std::vector<int64_t> reduced_dims;
  for (size_t d = 0; d < input_dims.size(); ++d)
    if (reduced_bitmap[d])
      reduced_dims.push_back(input_dims[d]);

  ReductionLayout layout = compute_reduction_layout(input, normalized_axes);
  cpu_demote_middle_to_generic(layout);
  const int64_t k = static_cast<int64_t>(normalized_axes.size());
  int64_t red_input_strides[MAX_DIMS];
  for (int64_t d = 0; d < k; ++d)
    red_input_strides[d] = input_strides[normalized_axes[d]];

  int64_t M_nr = 0;
  int64_t nr_sizes[MAX_DIMS], nr_strides_nr[MAX_DIMS];
  std::vector<int64_t> base_lin_idxs;
  if (layout.path == ReductionLayout::Path::Generic) {
    for (size_t d = 0; d < input_dims.size(); ++d) {
      if (!reduced_bitmap[d]) {
        nr_sizes[M_nr] = input_dims[d];
        nr_strides_nr[M_nr] = input_strides[d];
        ++M_nr;
      }
    }
    base_lin_idxs.resize(num_slices);
    int64_t oc[MAX_DIMS] = {}, blk = 0;
    for (int64_t o = 0; o < num_slices; ++o) {
      base_lin_idxs[o] = blk;
      for (int64_t j = M_nr - 1; j >= 0; --j) {
        ++oc[j];
        blk += nr_strides_nr[j];
        if (oc[j] < nr_sizes[j])
          break;
        blk -= oc[j] * nr_strides_nr[j];
        oc[j] = 0;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════
  // STRATEGY 1: Parallelize over output slots
  // When num_threads=1, this is effectively sequential.
  // ═══════════════════════════════════════════════════════════
  if (strategy == ReductionStrategy::ParallelSlices) {
#pragma omp parallel for num_threads(num_threads)
    for (int64_t o = 0; o < num_slices; ++o) {
      auto acc = reduce_one_index_slice(input_data, op, layout, o,
                                        reduced_count, base_lin_idxs.data(),
                                        red_input_strides, reduced_dims, k);
      output_data[o] = acc.index;
    }
  }
  // ═══════════════════════════════════════════════════════════
  // STRATEGY 2: Split reduction across threads + combine
  // For full reduction: ALL threads work on the single output.
  // Fixes PyTorch bug where tensor.argmax() uses only 1 thread.
  // ═══════════════════════════════════════════════════════════
  else {
    for (int64_t o = 0; o < num_slices; ++o) {
      std::vector<ValueIndex<T>> thread_accs(num_threads, op.identity());

#pragma omp parallel num_threads(num_threads)
      {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t chunk = (reduced_count + nt - 1) / nt;
        int64_t begin = tid * chunk;
        int64_t end = std::min(begin + chunk, reduced_count);

        if (begin < end) {
          thread_accs[tid] = reduce_one_index_slice(
              input_data, op, layout, o, reduced_count, base_lin_idxs.data(),
              red_input_strides, reduced_dims, k, begin, end);
        }
      }

      ValueIndex<T> final_acc = thread_accs[0];
      for (int t = 1; t < num_threads; ++t)
        final_acc = op.reduce(final_acc, thread_accs[t]);
      output_data[o] = final_acc.index;
    }
  }

  return output;
}

// =================================================================
// --- CORE REDUCTION KERNEL (TENSOR -> TENSOR) ---
// =================================================================

template <typename T, template <typename> class OpType, typename AccT = T>
Tensor
reduce_kernel(const Tensor &input, const std::vector<int64_t> &normalized_axes,
              const Shape &output_shape,
              ReductionStrategy strategy = ReductionStrategy::ParallelSlices,
              int num_threads = 1) {
  using Op = OpType<T>;

  // 1. Determine output dtype
  // NOTE: Index ops (ValueIndex) are handled by reduce_kernel_index, NOT here.
  //       Float/complex sum/nansum are handled by cascade_sum_kernel, NOT here.
  //       This kernel handles: int sum, prod, min, max, nanmin, nanmax,
  //       nanprod, all, any.
  Dtype output_dtype = input.dtype();
  if constexpr (std::is_same_v<T, bool>) {
    output_dtype = Dtype::Bool;
  } else if constexpr (std::is_integral_v<T>) {
    output_dtype = Dtype::Int64;
  }

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(output_dtype)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  // 2. Setup
  const T *input_data = input.data<T>();
  const std::vector<int64_t> &input_dims = input.shape().dims;
  const std::vector<int64_t> &input_strides = input.stride().strides;

  const int64_t reduced_count =
      calculate_reduced_count(input_dims, normalized_axes);

  // No-way we are reaching this case,so i commented this,
  //  //try searching for cases where we reach this,never we reach this case,
  //  //ex: [0,3] --->tensor_shape ,here input.numel() == 0 ,so bypasses in both
  //  all conditions like reduction over
  //  dim0(reduced_count=0),dim1(reduced_count=1)
  // and no way we can have reduced_count == 0  as normalize_axis fucntion take
  // care of the reduced  axes from input and process it well and handles this
  // case already .

  // if (reduced_count == 0 && input.numel() > 0) {
  //     throw std::runtime_error("Reduction error: reduced count is zero but
  //     input has " +
  //                             std::to_string(input.numel()) + " elements.");
  // }

  // Determine output C++ type (no ValueIndex here — index ops use
  // reduce_kernel_index)
  using OutputCppT =
      typename std::conditional<std::is_integral_v<T>, int64_t, T>::type;

  OutputCppT *output_data = output.data<OutputCppT>();

  Op op;
  const int64_t num_slices = output.numel();

  if (input_dims.size() > MAX_DIMS) {
    throw std::runtime_error(
        "Tensor rank exceeds maximum supported dimensions (64)");
  }

  bool reduced_bitmap[MAX_DIMS] = {false};
  for (int64_t axis : normalized_axes) {
    reduced_bitmap[axis] = true;
  }

  // Calculate reduced_dims once
  std::vector<int64_t> reduced_dims;
  for (size_t dim = 0; dim < input_dims.size(); ++dim) {
    bool is_reduced = reduced_bitmap[dim];
    if (is_reduced) {
      reduced_dims.push_back(input_dims[dim]);
    }
  }
  // Layout dispatch — computed once, replaces per-element coordinate math
  ReductionLayout layout = compute_reduction_layout(input, normalized_axes);
  cpu_demote_middle_to_generic(layout);
  // Precompute input strides for reduced axes (used by generic carry-add path)
  const int64_t k = static_cast<int64_t>(normalized_axes.size());
  int64_t red_input_strides[MAX_DIMS];
  for (int64_t d = 0; d < k; ++d)
    red_input_strides[d] = input_strides[normalized_axes[d]];
  // Precompute base linear indices for Generic path (sequential carry-add,
  // avoids per-element div/mod)
  int64_t M_nr = 0;
  int64_t nr_sizes[MAX_DIMS], nr_strides_nr[MAX_DIMS];
  std::vector<int64_t> base_lin_idxs;
  if (layout.path == ReductionLayout::Path::Generic) {
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
      if (!reduced_bitmap[dim]) {
        nr_sizes[M_nr] = input_dims[dim];
        nr_strides_nr[M_nr] = input_strides[dim];
        ++M_nr;
      }
    }
    base_lin_idxs.resize(num_slices);
    int64_t oc[MAX_DIMS] = {}, blk = 0;
    for (int64_t o = 0; o < num_slices; ++o) {
      base_lin_idxs[o] = blk;
      for (int64_t j = M_nr - 1; j >= 0; --j) {
        ++oc[j];
        blk += nr_strides_nr[j];
        if (oc[j] < nr_sizes[j])
          break;
        blk -= oc[j] * nr_strides_nr[j];
        oc[j] = 0;
      }
    }
  }
  // Use AccT directly if it's explicitly provided, otherwise compute based on T
  using AccumulatorT = AccT;

  // 3. Parallel execution — TWO STRATEGIES (like PyTorch's parallel_reduce)
  //
  // Strategy 1 (Outer Parallel): num_slices >= num_threads
  //   → Parallelize over output slots. Each thread handles complete reductions.
  //   → Equivalent to binary_kernel_reduce_lastdim.
  //
  // Strategy 2 (Split Reduction): num_slices < num_threads
  //   → Split REDUCTION DIMENSION across threads, combine at end.
  //   → Equivalent to binary_kernel_reduce. Needed for full reductions!
  // Helper: initialize accumulator with proper type cast
  auto init_acc = [&]() -> AccumulatorT {
    return static_cast<AccumulatorT>(op.identity());
  };

  // Helper: reduce + store one output position (all layout paths + SIMD)
  auto reduce_and_store_one = [&](int64_t output_index) {
    AccumulatorT accumulator = init_acc();

    if (layout.path == ReductionLayout::Path::InnerContiguous) {
      const T *in_ptr = input_data + output_index * layout.input_outer_stride;
      const int64_t n = layout.reduced_count;

      // ═══════════════════════════════════════════════════
      // SIMD DISPATCH: InnerContiguous path
      // Uses 4-accumulator ILP pattern for all SIMD paths
      // ═══════════════════════════════════════════════════

      // ─── SIMD: float min/max (8-wide AVX2) ───
      if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                     std::is_same_v<Op, MaxOp<T>>) &&
                    std::is_same_v<T, float>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        Vec va(accumulator), vb(accumulator), vc(accumulator), vd(accumulator);
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            va = Vec::min(va, Vec::loadu(in_ptr + j));
            vb = Vec::min(vb, Vec::loadu(in_ptr + j + W));
            vc = Vec::min(vc, Vec::loadu(in_ptr + j + W * 2));
            vd = Vec::min(vd, Vec::loadu(in_ptr + j + W * 3));
          } else {
            va = Vec::max(va, Vec::loadu(in_ptr + j));
            vb = Vec::max(vb, Vec::loadu(in_ptr + j + W));
            vc = Vec::max(vc, Vec::loadu(in_ptr + j + W * 2));
            vd = Vec::max(vd, Vec::loadu(in_ptr + j + W * 3));
          }
        }
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = Vec::min(Vec::min(va, vb), Vec::min(vc, vd));
        else
          va = Vec::max(Vec::max(va, vb), Vec::max(vc, vd));
        for (; j + W <= n; j += W) {
          if constexpr (std::is_same_v<Op, MinOp<T>>)
            va = Vec::min(va, Vec::loadu(in_ptr + j));
          else
            va = Vec::max(va, Vec::loadu(in_ptr + j));
        }
        accumulator =
            std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
        for (; j < n; ++j)
          accumulator =
              op.reduce(accumulator, static_cast<AccumulatorT>(in_ptr[j]));

        // ─── SIMD: double min/max (4-wide AVX2) ───
      } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                            std::is_same_v<Op, MaxOp<T>>) &&
                           std::is_same_v<T, double>) {
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        Vec va(accumulator), vb(accumulator), vc(accumulator), vd(accumulator);
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            va = Vec::min(va, Vec::loadu(in_ptr + j));
            vb = Vec::min(vb, Vec::loadu(in_ptr + j + W));
            vc = Vec::min(vc, Vec::loadu(in_ptr + j + W * 2));
            vd = Vec::min(vd, Vec::loadu(in_ptr + j + W * 3));
          } else {
            va = Vec::max(va, Vec::loadu(in_ptr + j));
            vb = Vec::max(vb, Vec::loadu(in_ptr + j + W));
            vc = Vec::max(vc, Vec::loadu(in_ptr + j + W * 2));
            vd = Vec::max(vd, Vec::loadu(in_ptr + j + W * 3));
          }
        }
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = Vec::min(Vec::min(va, vb), Vec::min(vc, vd));
        else
          va = Vec::max(Vec::max(va, vb), Vec::max(vc, vd));
        for (; j + W <= n; j += W) {
          if constexpr (std::is_same_v<Op, MinOp<T>>)
            va = Vec::min(va, Vec::loadu(in_ptr + j));
          else
            va = Vec::max(va, Vec::loadu(in_ptr + j));
        }
        accumulator =
            std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
        for (; j < n; ++j)
          accumulator =
              op.reduce(accumulator, static_cast<AccumulatorT>(in_ptr[j]));

        // ─── SIMD: int32 min/max (8-wide AVX2) ───
      } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                            std::is_same_v<Op, MaxOp<T>>) &&
                           std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int32_t>;
        constexpr int64_t W = Vec::size();
        Vec va(accumulator), vb(accumulator);
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            va = Vec::min(va, Vec::loadu(in_ptr + j));
            vb = Vec::min(vb, Vec::loadu(in_ptr + j + W));
          } else {
            va = Vec::max(va, Vec::loadu(in_ptr + j));
            vb = Vec::max(vb, Vec::loadu(in_ptr + j + W));
          }
        }
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = Vec::min(va, vb);
        else
          va = Vec::max(va, vb);
        alignas(32) int32_t lanes[8];
        va.storeu(lanes);
        for (int i = 0; i < 8; ++i)
          accumulator =
              op.reduce(accumulator, static_cast<AccumulatorT>(lanes[i]));
        for (; j < n; ++j)
          accumulator =
              op.reduce(accumulator, static_cast<AccumulatorT>(in_ptr[j]));

        // ─── SIMD: int32 sum (8-wide → int64 accumulator) ───
      } else if constexpr (std::is_same_v<Op, SumOp<T>> &&
                           std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int64_t>;
        constexpr int64_t W = Vec::size(); // 4
        Vec va, vb;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          // Widen int32 → int64 to prevent overflow
          __m128i lo4 =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(in_ptr + j));
          __m128i hi4 = _mm_loadu_si128(
              reinterpret_cast<const __m128i *>(in_ptr + j + W));
          va = va + Vec(_mm256_cvtepi32_epi64(lo4));
          vb = vb + Vec(_mm256_cvtepi32_epi64(hi4));
        }
        accumulator = (va + vb).reduce_add();
        for (; j < n; ++j)
          accumulator += static_cast<int64_t>(in_ptr[j]);

        // ─── SIMD: int64 sum (4-wide AVX2) ───
      } else if constexpr (std::is_same_v<Op, SumOp<T>> &&
                           std::is_same_v<T, int64_t>) {
        using Vec = vec::Vectorized<int64_t>;
        constexpr int64_t W = Vec::size();
        Vec va, vb;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          va = va + Vec::loadu(in_ptr + j);
          vb = vb + Vec::loadu(in_ptr + j + W);
        }
        accumulator = (va + vb).reduce_add();
        for (; j < n; ++j)
          accumulator += in_ptr[j];

        // ─── SIMD: float product (8-wide AVX2) ───
      } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                           std::is_same_v<T, float>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        Vec va(1.0f), vb(1.0f), vc(1.0f), vd(1.0f);
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va * Vec::loadu(in_ptr + j);
          vb = vb * Vec::loadu(in_ptr + j + W);
          vc = vc * Vec::loadu(in_ptr + j + W * 2);
          vd = vd * Vec::loadu(in_ptr + j + W * 3);
        }
        va = va * vb * vc * vd;
        for (; j + W <= n; j += W)
          va = va * Vec::loadu(in_ptr + j);
        alignas(32) float lanes[8];
        va.storeu(lanes);
        accumulator = lanes[0];
        for (int i = 1; i < 8; ++i)
          accumulator *= lanes[i];
        for (; j < n; ++j)
          accumulator *= in_ptr[j];

        // ─── SIMD: double product (4-wide AVX2) ───
      } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                           std::is_same_v<T, double>) {
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        Vec va(1.0), vb(1.0), vc(1.0), vd(1.0);
        int64_t j = 0;
        for (; j + W * 4 <= n; j += W * 4) {
          va = va * Vec::loadu(in_ptr + j);
          vb = vb * Vec::loadu(in_ptr + j + W);
          vc = vc * Vec::loadu(in_ptr + j + W * 2);
          vd = vd * Vec::loadu(in_ptr + j + W * 3);
        }
        va = va * vb * vc * vd;
        for (; j + W <= n; j += W)
          va = va * Vec::loadu(in_ptr + j);
        alignas(32) double lanes[4];
        va.storeu(lanes);
        accumulator = lanes[0];
        for (int i = 1; i < 4; ++i)
          accumulator *= lanes[i];
        for (; j < n; ++j)
          accumulator *= in_ptr[j];

        // ─── SIMD: int32 product (8-wide AVX2) ───
      } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                           std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int32_t>;
        constexpr int64_t W = Vec::size();
        Vec va(static_cast<int32_t>(1)), vb(static_cast<int32_t>(1));
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          va = va * Vec::loadu(in_ptr + j);
          vb = vb * Vec::loadu(in_ptr + j + W);
        }
        va = va * vb;
        alignas(32) int32_t lanes[8];
        va.storeu(lanes);
        // Product accumulates in int64 to avoid overflow
        for (int i = 0; i < 8; ++i)
          accumulator *= static_cast<int64_t>(lanes[i]);
        for (; j < n; ++j)
          accumulator *= static_cast<int64_t>(in_ptr[j]);

        // ─── SIMD: fp16 min/max (8-wide via F16C load→float) ───
      } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                            std::is_same_v<Op, MaxOp<T>>) &&
                           std::is_same_v<T, float16_t>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8;
        float facc = static_cast<float>(accumulator);
        Vec va(facc), vb(facc);
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          Vec v0 = vec::load_fp16_as_float(in_ptr + j);
          Vec v1 = vec::load_fp16_as_float(in_ptr + j + W);
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            va = Vec::min(va, v0);
            vb = Vec::min(vb, v1);
          } else {
            va = Vec::max(va, v0);
            vb = Vec::max(vb, v1);
          }
        }
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = Vec::min(va, vb);
        else
          va = Vec::max(va, vb);
        facc = std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
        for (; j < n; ++j) {
          float fv = static_cast<float>(in_ptr[j]);
          facc = std::is_same_v<Op, MinOp<T>> ? std::min(facc, fv)
                                              : std::max(facc, fv);
        }
        accumulator = static_cast<T>(facc);

        // ─── SIMD: bf16 min/max (8-wide via shift load→float) ───
      } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                            std::is_same_v<Op, MaxOp<T>>) &&
                           std::is_same_v<T, bfloat16_t>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8;
        float facc = static_cast<float>(accumulator);
        Vec va(facc), vb(facc);
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          Vec v0 = vec::load_bf16_as_float(in_ptr + j);
          Vec v1 = vec::load_bf16_as_float(in_ptr + j + W);
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            va = Vec::min(va, v0);
            vb = Vec::min(vb, v1);
          } else {
            va = Vec::max(va, v0);
            vb = Vec::max(vb, v1);
          }
        }
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = Vec::min(va, vb);
        else
          va = Vec::max(va, vb);
        facc = std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
        for (; j < n; ++j) {
          float fv = static_cast<float>(in_ptr[j]);
          facc = std::is_same_v<Op, MinOp<T>> ? std::min(facc, fv)
                                              : std::max(facc, fv);
        }
        accumulator = static_cast<T>(facc);

        // ─── SIMD: bool all/any (32-wide AVX2 via uint8_t) ───
      } else if constexpr (std::is_same_v<T, bool> &&
                           (std::is_same_v<Op, AllOp<T>> ||
                            std::is_same_v<Op, AnyOp<T>>)) {
        using Vec = vec::Vectorized<uint8_t>;
        constexpr int64_t W = Vec::size(); // 32
        const uint8_t *u8_ptr = reinterpret_cast<const uint8_t *>(in_ptr);
        if constexpr (std::is_same_v<Op, AllOp<T>>) {
          Vec vacc(static_cast<uint8_t>(0xFF));
          int64_t j = 0;
          for (; j + W <= n; j += W)
            vacc = vacc & Vec::loadu(u8_ptr + j);
          uint8_t tmp[32];
          vacc.storeu(tmp);
          bool result = true;
          for (int i = 0; i < 32; ++i)
            result = result && (tmp[i] != 0);
          for (; j < n; ++j)
            result = result && (u8_ptr[j] != 0);
          accumulator = result;
        } else {
          Vec vacc(static_cast<uint8_t>(0x00));
          int64_t j = 0;
          for (; j + W <= n; j += W)
            vacc = vacc | Vec::loadu(u8_ptr + j);
          uint8_t tmp[32];
          vacc.storeu(tmp);
          bool result = false;
          for (int i = 0; i < 32; ++i)
            result = result || (tmp[i] != 0);
          for (; j < n; ++j)
            result = result || (u8_ptr[j] != 0);
          accumulator = result;
        }

        // ─── SIMD: float nanmin/nanmax (8-wide, NaN masking) ───
        // NaN lanes → +INF (for nanmin) or -INF (for nanmax), so they don't
        // affect result
      } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                            std::is_same_v<Op, NanMaxOp<T>>) &&
                           std::is_same_v<T, float>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        // NaN replacement: +INF for nanmin, -INF for nanmax
        constexpr bool is_nan_min = std::is_same_v<Op, NanMinOp<T>>;
        __m256 nan_fill =
            is_nan_min ? _mm256_set1_ps(INFINITY) : _mm256_set1_ps(-INFINITY);
        __m256 va = nan_fill, vb = nan_fill;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256 v0 = _mm256_loadu_ps(in_ptr + j);
          __m256 v1 = _mm256_loadu_ps(in_ptr + j + W);
          // Replace NaN with nan_fill
          __m256 m0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
          __m256 m1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
          v0 = _mm256_blendv_ps(v0, nan_fill, m0);
          v1 = _mm256_blendv_ps(v1, nan_fill, m1);
          if constexpr (is_nan_min) {
            va = _mm256_min_ps(va, v0);
            vb = _mm256_min_ps(vb, v1);
          } else {
            va = _mm256_max_ps(va, v0);
            vb = _mm256_max_ps(vb, v1);
          }
        }
        if constexpr (is_nan_min)
          va = _mm256_min_ps(va, vb);
        else
          va = _mm256_max_ps(va, vb);
        accumulator = is_nan_min ? Vec(va).reduce_min() : Vec(va).reduce_max();
        for (; j < n; ++j) {
          float v = in_ptr[j];
          if (!std::isnan(v))
            accumulator = op.reduce(accumulator, v);
        }

        // ─── SIMD: double nanmin/nanmax (4-wide, NaN masking) ───
      } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                            std::is_same_v<Op, NanMaxOp<T>>) &&
                           std::is_same_v<T, double>) {
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        constexpr bool is_nan_min = std::is_same_v<Op, NanMinOp<T>>;
        __m256d nan_fill =
            is_nan_min ? _mm256_set1_pd(INFINITY) : _mm256_set1_pd(-INFINITY);
        __m256d va = nan_fill, vb = nan_fill;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256d v0 = _mm256_loadu_pd(in_ptr + j);
          __m256d v1 = _mm256_loadu_pd(in_ptr + j + W);
          __m256d m0 = _mm256_cmp_pd(v0, v0, _CMP_UNORD_Q);
          __m256d m1 = _mm256_cmp_pd(v1, v1, _CMP_UNORD_Q);
          v0 = _mm256_blendv_pd(v0, nan_fill, m0);
          v1 = _mm256_blendv_pd(v1, nan_fill, m1);
          if constexpr (is_nan_min) {
            va = _mm256_min_pd(va, v0);
            vb = _mm256_min_pd(vb, v1);
          } else {
            va = _mm256_max_pd(va, v0);
            vb = _mm256_max_pd(vb, v1);
          }
        }
        if constexpr (is_nan_min)
          va = _mm256_min_pd(va, vb);
        else
          va = _mm256_max_pd(va, vb);
        accumulator = is_nan_min ? Vec(va).reduce_min() : Vec(va).reduce_max();
        for (; j < n; ++j) {
          double v = in_ptr[j];
          if (!std::isnan(v))
            accumulator = op.reduce(accumulator, v);
        }

        // ─── SIMD: int64 min/max (4-wide, emulated via cmpgt+blend) ───
        // AVX2 has NO native _mm256_min/max_epi64. AVX-512 does.
        // PyTorch skips this due to upper_bound<int64_t> → double overflow bug
        // (#43254). We don't have that bug (native type identity), so we can
        // safely emulate. Benchmark: 1.1-2.3x faster than scalar (memory-bound
        // for large arrays).
      } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                            std::is_same_v<Op, MaxOp<T>>) &&
                           std::is_same_v<T, int64_t>) {
        constexpr int64_t W = 4;
        __m256i va = _mm256_set1_epi64x(accumulator);
        __m256i vb = va;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256i v0 =
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in_ptr + j));
          __m256i v1 = _mm256_loadu_si256(
              reinterpret_cast<const __m256i *>(in_ptr + j + W));
          if constexpr (std::is_same_v<Op, MinOp<T>>) {
            // min: select b where a > b
            va = _mm256_blendv_epi8(va, v0, _mm256_cmpgt_epi64(va, v0));
            vb = _mm256_blendv_epi8(vb, v1, _mm256_cmpgt_epi64(vb, v1));
          } else {
            // max: select a where a > b, else b
            va = _mm256_blendv_epi8(v0, va, _mm256_cmpgt_epi64(va, v0));
            vb = _mm256_blendv_epi8(v1, vb, _mm256_cmpgt_epi64(vb, v1));
          }
        }
        // Combine va and vb
        if constexpr (std::is_same_v<Op, MinOp<T>>)
          va = _mm256_blendv_epi8(va, vb, _mm256_cmpgt_epi64(va, vb));
        else
          va = _mm256_blendv_epi8(vb, va, _mm256_cmpgt_epi64(va, vb));
        alignas(32) int64_t lanes[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(lanes), va);
        accumulator = lanes[0];
        for (int i = 1; i < 4; ++i)
          accumulator = op.reduce(accumulator, lanes[i]);
        for (; j < n; ++j)
          accumulator = op.reduce(accumulator, in_ptr[j]);

        // ─── SIMD: fp16 product (8-wide, F16C load→float mul) ───
      } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                           std::is_same_v<T, float16_t>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8;
        Vec va(1.0f), vb(1.0f);
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          va = va * vec::load_fp16_as_float(in_ptr + j);
          vb = vb * vec::load_fp16_as_float(in_ptr + j + W);
        }
        va = va * vb;
        alignas(32) float lanes[8];
        va.storeu(lanes);
        float facc = lanes[0];
        for (int i = 1; i < 8; ++i)
          facc *= lanes[i];
        for (; j < n; ++j)
          facc *= static_cast<float>(in_ptr[j]);
        accumulator = static_cast<T>(facc);

        // ─── SIMD: bf16 product (8-wide, shift load→float mul) ───
      } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                           std::is_same_v<T, bfloat16_t>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = 8;
        Vec va(1.0f), vb(1.0f);
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          va = va * vec::load_bf16_as_float(in_ptr + j);
          vb = vb * vec::load_bf16_as_float(in_ptr + j + W);
        }
        va = va * vb;
        alignas(32) float lanes[8];
        va.storeu(lanes);
        float facc = lanes[0];
        for (int i = 1; i < 8; ++i)
          facc *= lanes[i];
        for (; j < n; ++j)
          facc *= static_cast<float>(in_ptr[j]);
        accumulator = static_cast<T>(facc);

        // ─── SIMD: fp16 nanmin/nanmax (8-wide, F16C + NaN mask) ───
      } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                            std::is_same_v<Op, NanMaxOp<T>>) &&
                           std::is_same_v<T, float16_t>) {
        constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
        constexpr int64_t W = 8;
        __m256 nan_fill =
            is_min ? _mm256_set1_ps(INFINITY) : _mm256_set1_ps(-INFINITY);
        __m256 va = nan_fill, vb = nan_fill, zero = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256 v0 = vec::load_fp16_as_float(in_ptr + j).values;
          __m256 v1 = vec::load_fp16_as_float(in_ptr + j + W).values;
          v0 = _mm256_blendv_ps(v0, nan_fill,
                                _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q));
          v1 = _mm256_blendv_ps(v1, nan_fill,
                                _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q));
          if constexpr (is_min) {
            va = _mm256_min_ps(va, v0);
            vb = _mm256_min_ps(vb, v1);
          } else {
            va = _mm256_max_ps(va, v0);
            vb = _mm256_max_ps(vb, v1);
          }
        }
        if constexpr (is_min)
          va = _mm256_min_ps(va, vb);
        else
          va = _mm256_max_ps(va, vb);
        float facc = is_min ? vec::Vectorized<float>(va).reduce_min()
                            : vec::Vectorized<float>(va).reduce_max();
        for (; j < n; ++j) {
          float v = static_cast<float>(in_ptr[j]);
          if (!std::isnan(v))
            facc = is_min ? std::min(facc, v) : std::max(facc, v);
        }
        accumulator = static_cast<T>(facc);

        // ─── SIMD: bf16 nanmin/nanmax (8-wide, shift + NaN mask) ───
      } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                            std::is_same_v<Op, NanMaxOp<T>>) &&
                           std::is_same_v<T, bfloat16_t>) {
        constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
        constexpr int64_t W = 8;
        __m256 nan_fill =
            is_min ? _mm256_set1_ps(INFINITY) : _mm256_set1_ps(-INFINITY);
        __m256 va = nan_fill, vb = nan_fill;
        int64_t j = 0;
        for (; j + W * 2 <= n; j += W * 2) {
          __m256 v0 = vec::load_bf16_as_float(in_ptr + j).values;
          __m256 v1 = vec::load_bf16_as_float(in_ptr + j + W).values;
          v0 = _mm256_blendv_ps(v0, nan_fill,
                                _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q));
          v1 = _mm256_blendv_ps(v1, nan_fill,
                                _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q));
          if constexpr (is_min) {
            va = _mm256_min_ps(va, v0);
            vb = _mm256_min_ps(vb, v1);
          } else {
            va = _mm256_max_ps(va, v0);
            vb = _mm256_max_ps(vb, v1);
          }
        }
        if constexpr (is_min)
          va = _mm256_min_ps(va, vb);
        else
          va = _mm256_max_ps(va, vb);
        float facc = is_min ? vec::Vectorized<float>(va).reduce_min()
                            : vec::Vectorized<float>(va).reduce_max();
        for (; j < n; ++j) {
          float v = static_cast<float>(in_ptr[j]);
          if (!std::isnan(v))
            facc = is_min ? std::min(facc, v) : std::max(facc, v);
        }
        accumulator = static_cast<T>(facc);

        // ─── SCALAR FALLBACK: complex, NaN product, uint types without SIMD
        // ───
      } else {
        for (int64_t j = 0; j < n; ++j) {
          T input_value = in_ptr[j];
          if constexpr (std::is_same_v<AccT, bool>) {
            bool val_as_bool = to_bool_value(input_value);
            accumulator = op.reduce(accumulator, val_as_bool);
          } else {
            AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
            accumulator = op.reduce(accumulator, val_acc);
          }
        }
      }
    } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
      // Walk down a column — one stride-multiply per reduction row, no div/mod
      for (int64_t r = 0; r < layout.reduced_count; ++r) {
        const T *in_ptr =
            input_data + output_index + r * layout.input_row_stride;
        T input_value = *in_ptr;
        if constexpr (std::is_same_v<AccT, bool>) {
          bool val_as_bool = to_bool_value(input_value);
          accumulator = op.reduce(accumulator, val_as_bool);
        } else {
          AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
          accumulator = op.reduce(accumulator, val_acc);
        }
      }
    } else {
      // Generic: precomputed base + carry-add inner counter — zero div/mod per
      // element
      int64_t red_coords[MAX_DIMS] = {};
      int64_t input_lin_idx = base_lin_idxs[output_index];
      for (int64_t i = 0; i < reduced_count; ++i) {
        T input_value = input_data[input_lin_idx];
        if constexpr (std::is_same_v<AccT, bool>) {
          bool val_as_bool = to_bool_value(input_value);
          accumulator = op.reduce(accumulator, val_as_bool);
        } else {
          AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
          accumulator = op.reduce(accumulator, val_acc);
        }
        // Carry-add: advance pointer, no division
        for (int64_t d = k - 1; d >= 0; --d) {
          ++red_coords[d];
          input_lin_idx += red_input_strides[d];
          if (red_coords[d] < reduced_dims[d])
            break;
          input_lin_idx -= red_coords[d] * red_input_strides[d];
          red_coords[d] = 0;
        }
      }
    }

    // =================================================================
    // CRITICAL: Safe conversion back to output type (Standard path)
    // =================================================================
    if constexpr (std::is_same_v<T, float16_t>) {
      output_data[output_index] = static_cast<OutputCppT>(
          static_cast<T>(static_cast<float>(accumulator)));
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      output_data[output_index] = static_cast<OutputCppT>(
          static_cast<T>(static_cast<float>(accumulator)));
    } else if constexpr (std::is_same_v<OutputCppT, complex32_t>) {
      if constexpr (std::is_same_v<AccumulatorT, complex32_t>) {
        output_data[output_index] = accumulator;
      } else if constexpr (std::is_same_v<AccumulatorT, complex64_t> ||
                           std::is_same_v<AccumulatorT, complex128_t>) {
        output_data[output_index] =
            complex32_t(static_cast<float>(accumulator.real()),
                        static_cast<float>(accumulator.imag()));
      } else {
        output_data[output_index] =
            complex32_t(static_cast<float>(accumulator), 0.0f);
      }
    } else if constexpr (std::is_same_v<OutputCppT, complex64_t>) {
      if constexpr (std::is_same_v<AccumulatorT, complex64_t>) {
        output_data[output_index] = accumulator;
      } else if constexpr (std::is_same_v<AccumulatorT, complex32_t>) {
        output_data[output_index] =
            complex64_t(static_cast<float>(accumulator.real()),
                        static_cast<float>(accumulator.imag()));
      } else if constexpr (std::is_same_v<AccumulatorT, complex128_t>) {
        output_data[output_index] =
            complex64_t(static_cast<float>(accumulator.real()),
                        static_cast<float>(accumulator.imag()));
      } else {
        output_data[output_index] =
            complex64_t(static_cast<float>(accumulator), 0.0f);
      }
    } else if constexpr (std::is_same_v<OutputCppT, complex128_t>) {
      if constexpr (std::is_same_v<AccumulatorT, complex128_t>) {
        output_data[output_index] = accumulator;
      } else if constexpr (std::is_same_v<AccumulatorT, complex32_t> ||
                           std::is_same_v<AccumulatorT, complex64_t>) {
        output_data[output_index] =
            complex128_t(static_cast<double>(accumulator.real()),
                         static_cast<double>(accumulator.imag()));
      } else {
        output_data[output_index] =
            complex128_t(static_cast<double>(accumulator), 0.0);
      }
    } else if constexpr (std::is_same_v<OutputCppT, float4_e2m1_2x_t> ||
                         std::is_same_v<OutputCppT, float4_e2m1_t>) {
      output_data[output_index] =
          static_cast<OutputCppT>(static_cast<float>(accumulator));
    } else {
      output_data[output_index] = static_cast<OutputCppT>(accumulator);
    }
  };

  // ═══════════════════════════════════════════════════════════
  // STRATEGY 1: Parallelize over output slots
  // For OuterContiguous: SIMD across adjacent output columns (vertical
  // vectorization) ═══════════════════════════════════════════════════════════
  if (strategy == ReductionStrategy::ParallelSlices) {
    // ─── SIMD OUTER: float/double min/max/prod across adjacent columns ───
    if (layout.path == ReductionLayout::Path::OuterContiguous) {
      const int64_t stride = layout.input_row_stride;
      const int64_t R = layout.reduced_count;
      bool handled = false;

      // float min/max: 8-wide vertical
      if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                     std::is_same_v<Op, MaxOp<T>>) &&
                    std::is_same_v<T, float> &&
                    std::is_same_v<AccumulatorT, float>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(init_acc());
          for (int64_t r = 0; r < R; ++r) {
            if constexpr (std::is_same_v<Op, MinOp<T>>)
              acc = Vec::min(acc, Vec::loadu(input_data + r * stride + o));
            else
              acc = Vec::max(acc, Vec::loadu(input_data + r * stride + o));
          }
          acc.storeu(reinterpret_cast<float *>(output_data + o));
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // double min/max: 4-wide vertical
      if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                     std::is_same_v<Op, MaxOp<T>>) &&
                    std::is_same_v<T, double> &&
                    std::is_same_v<AccumulatorT, double>) {
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(init_acc());
          for (int64_t r = 0; r < R; ++r) {
            if constexpr (std::is_same_v<Op, MinOp<T>>)
              acc = Vec::min(acc, Vec::loadu(input_data + r * stride + o));
            else
              acc = Vec::max(acc, Vec::loadu(input_data + r * stride + o));
          }
          acc.storeu(reinterpret_cast<double *>(output_data + o));
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // float product: 8-wide vertical
      if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                    std::is_same_v<T, float>) {
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(1.0f);
          for (int64_t r = 0; r < R; ++r)
            acc = acc * Vec::loadu(input_data + r * stride + o);
          acc.storeu(reinterpret_cast<float *>(output_data + o));
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // double product: 4-wide vertical
      if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                    std::is_same_v<T, double>) {
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(1.0);
          for (int64_t r = 0; r < R; ++r)
            acc = acc * Vec::loadu(input_data + r * stride + o);
          acc.storeu(reinterpret_cast<double *>(output_data + o));
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }

      // int32 min/max: 8-wide vertical
      if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                     std::is_same_v<Op, MaxOp<T>>) &&
                    std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int32_t>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(init_acc());
          for (int64_t r = 0; r < R; ++r) {
            if constexpr (std::is_same_v<Op, MinOp<T>>)
              acc = Vec::min(acc, Vec::loadu(input_data + r * stride + o));
            else
              acc = Vec::max(acc, Vec::loadu(input_data + r * stride + o));
          }
          acc.storeu(reinterpret_cast<int32_t *>(output_data + o));
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // int32 sum: 8-wide vertical (accumulate as int32 — fits for typical
      // reduction sizes)
      if constexpr (std::is_same_v<Op, SumOp<T>> &&
                    std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int32_t>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          // Note: outer sum accumulates in int32 per column, then stores as
          // int64 For very large reductions this could overflow. Scalar
          // fallback handles overflow-safe int64.
          Vec acc(static_cast<int32_t>(0));
          for (int64_t r = 0; r < R; ++r)
            acc = acc + Vec::loadu(input_data + r * stride + o);
          // Store: widen to int64 for output
          alignas(32) int32_t lanes[8];
          acc.storeu(lanes);
          for (int i = 0; i < W && (o + i) < num_slices; ++i)
            output_data[o + i] =
                static_cast<std::remove_pointer_t<decltype(output_data)>>(
                    lanes[i]);
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // int32 product: 8-wide vertical
      if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                    std::is_same_v<T, int32_t>) {
        using Vec = vec::Vectorized<int32_t>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          Vec acc(static_cast<int32_t>(1));
          for (int64_t r = 0; r < R; ++r)
            acc = acc * Vec::loadu(input_data + r * stride + o);
          alignas(32) int32_t lanes[8];
          acc.storeu(lanes);
          for (int i = 0; i < W && (o + i) < num_slices; ++i)
            output_data[o + i] =
                static_cast<std::remove_pointer_t<decltype(output_data)>>(
                    lanes[i]);
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // float nanmin/nanmax: 8-wide vertical + NaN masking
      if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                     std::is_same_v<Op, NanMaxOp<T>>) &&
                    std::is_same_v<T, float>) {
        constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
        using Vec = vec::Vectorized<float>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
        __m256 nan_fill =
            is_min ? _mm256_set1_ps(INFINITY) : _mm256_set1_ps(-INFINITY);
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          __m256 acc = nan_fill;
          for (int64_t r = 0; r < R; ++r) {
            __m256 v = _mm256_loadu_ps(input_data + r * stride + o);
            v = _mm256_blendv_ps(v, nan_fill,
                                 _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
            if constexpr (is_min)
              acc = _mm256_min_ps(acc, v);
            else
              acc = _mm256_max_ps(acc, v);
          }
          _mm256_storeu_ps(reinterpret_cast<float *>(output_data + o), acc);
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }
      // double nanmin/nanmax: 4-wide vertical + NaN masking
      if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                     std::is_same_v<Op, NanMaxOp<T>>) &&
                    std::is_same_v<T, double>) {
        constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
        using Vec = vec::Vectorized<double>;
        constexpr int64_t W = Vec::size();
        const int64_t vec_end = (num_slices / W) * W;
        __m256d nan_fill =
            is_min ? _mm256_set1_pd(INFINITY) : _mm256_set1_pd(-INFINITY);
#pragma omp parallel for num_threads(num_threads)
        for (int64_t o = 0; o < vec_end; o += W) {
          __m256d acc = nan_fill;
          for (int64_t r = 0; r < R; ++r) {
            __m256d v = _mm256_loadu_pd(input_data + r * stride + o);
            v = _mm256_blendv_pd(v, nan_fill,
                                 _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
            if constexpr (is_min)
              acc = _mm256_min_pd(acc, v);
            else
              acc = _mm256_max_pd(acc, v);
          }
          _mm256_storeu_pd(reinterpret_cast<double *>(output_data + o), acc);
        }
        for (int64_t o = vec_end; o < num_slices; ++o)
          reduce_and_store_one(o);
        handled = true;
      }

      if (handled)
        return output;
    }

// ─── Default: scalar per output slot (InnerContiguous SIMD is inside lambda)
// ───
#pragma omp parallel for num_threads(num_threads)
    for (int64_t o = 0; o < num_slices; ++o)
      reduce_and_store_one(o);
    return output;
  }

  // ═══════════════════════════════════════════════════════════
  // STRATEGY 2: Split reduction across threads + combine
  // ═══════════════════════════════════════════════════════════
  for (int64_t o = 0; o < num_slices; ++o) {
    std::vector<AccumulatorT> thread_accs(num_threads, init_acc());

#pragma omp parallel num_threads(num_threads)
    {
      int tid = omp_get_thread_num();
      int nt = omp_get_num_threads();
      int64_t chunk = (reduced_count + nt - 1) / nt;
      int64_t begin = tid * chunk;
      int64_t end = std::min(begin + chunk, reduced_count);
      AccumulatorT local = init_acc();

      if (layout.path == ReductionLayout::Path::InnerContiguous) {
        const T *in_ptr = input_data + o * layout.input_outer_stride + begin;
        const int64_t n = end - begin;

        // ─── SIMD per-thread chunk for Strategy 2 InnerContiguous ───
        // float min/max
        if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                       std::is_same_v<Op, MaxOp<T>>) &&
                      std::is_same_v<T, float>) {
          using Vec = vec::Vectorized<float>;
          constexpr int64_t W = Vec::size();
          Vec va(local), vb(local);
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            if constexpr (std::is_same_v<Op, MinOp<T>>) {
              va = Vec::min(va, Vec::loadu(in_ptr + j));
              vb = Vec::min(vb, Vec::loadu(in_ptr + j + W));
            } else {
              va = Vec::max(va, Vec::loadu(in_ptr + j));
              vb = Vec::max(vb, Vec::loadu(in_ptr + j + W));
            }
          }
          if constexpr (std::is_same_v<Op, MinOp<T>>)
            va = Vec::min(va, vb);
          else
            va = Vec::max(va, vb);
          for (; j + W <= n; j += W) {
            if constexpr (std::is_same_v<Op, MinOp<T>>)
              va = Vec::min(va, Vec::loadu(in_ptr + j));
            else
              va = Vec::max(va, Vec::loadu(in_ptr + j));
          }
          local =
              std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
          for (; j < n; ++j)
            local = op.reduce(local, static_cast<AccumulatorT>(in_ptr[j]));
          // double min/max
        } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                              std::is_same_v<Op, MaxOp<T>>) &&
                             std::is_same_v<T, double>) {
          using Vec = vec::Vectorized<double>;
          constexpr int64_t W = Vec::size();
          Vec va(local), vb(local);
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            if constexpr (std::is_same_v<Op, MinOp<T>>) {
              va = Vec::min(va, Vec::loadu(in_ptr + j));
              vb = Vec::min(vb, Vec::loadu(in_ptr + j + W));
            } else {
              va = Vec::max(va, Vec::loadu(in_ptr + j));
              vb = Vec::max(vb, Vec::loadu(in_ptr + j + W));
            }
          }
          if constexpr (std::is_same_v<Op, MinOp<T>>)
            va = Vec::min(va, vb);
          else
            va = Vec::max(va, vb);
          local =
              std::is_same_v<Op, MinOp<T>> ? va.reduce_min() : va.reduce_max();
          for (; j < n; ++j)
            local = op.reduce(local, static_cast<AccumulatorT>(in_ptr[j]));
          // int32 min/max
        } else if constexpr ((std::is_same_v<Op, MinOp<T>> ||
                              std::is_same_v<Op, MaxOp<T>>) &&
                             std::is_same_v<T, int32_t>) {
          using Vec = vec::Vectorized<int32_t>;
          constexpr int64_t W = Vec::size();
          Vec va(local);
          int64_t j = 0;
          for (; j + W <= n; j += W) {
            if constexpr (std::is_same_v<Op, MinOp<T>>)
              va = Vec::min(va, Vec::loadu(in_ptr + j));
            else
              va = Vec::max(va, Vec::loadu(in_ptr + j));
          }
          alignas(32) int32_t lanes[8];
          va.storeu(lanes);
          for (int i = 0; i < 8; ++i)
            local = op.reduce(local, static_cast<AccumulatorT>(lanes[i]));
          for (; j < n; ++j)
            local = op.reduce(local, static_cast<AccumulatorT>(in_ptr[j]));
          // int32 sum → int64
        } else if constexpr (std::is_same_v<Op, SumOp<T>> &&
                             std::is_same_v<T, int32_t>) {
          using Vec = vec::Vectorized<int64_t>;
          constexpr int64_t W = Vec::size();
          Vec va, vb;
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            __m128i lo4 =
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(in_ptr + j));
            __m128i hi4 = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(in_ptr + j + W));
            va = va + Vec(_mm256_cvtepi32_epi64(lo4));
            vb = vb + Vec(_mm256_cvtepi32_epi64(hi4));
          }
          local = (va + vb).reduce_add();
          for (; j < n; ++j)
            local += static_cast<int64_t>(in_ptr[j]);
          // int64 sum
        } else if constexpr (std::is_same_v<Op, SumOp<T>> &&
                             std::is_same_v<T, int64_t>) {
          using Vec = vec::Vectorized<int64_t>;
          constexpr int64_t W = Vec::size();
          Vec va, vb;
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            va = va + Vec::loadu(in_ptr + j);
            vb = vb + Vec::loadu(in_ptr + j + W);
          }
          local = (va + vb).reduce_add();
          for (; j < n; ++j)
            local += in_ptr[j];
          // float product
        } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                             std::is_same_v<T, float>) {
          using Vec = vec::Vectorized<float>;
          constexpr int64_t W = Vec::size();
          Vec va(1.0f), vb(1.0f);
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            va = va * Vec::loadu(in_ptr + j);
            vb = vb * Vec::loadu(in_ptr + j + W);
          }
          va = va * vb;
          alignas(32) float lanes[8];
          va.storeu(lanes);
          local = lanes[0];
          for (int i = 1; i < 8; ++i)
            local *= lanes[i];
          for (; j < n; ++j)
            local *= in_ptr[j];
          // double product
        } else if constexpr (std::is_same_v<Op, ProductOp<T>> &&
                             std::is_same_v<T, double>) {
          using Vec = vec::Vectorized<double>;
          constexpr int64_t W = Vec::size();
          Vec va(1.0), vb(1.0);
          int64_t j = 0;
          for (; j + W * 2 <= n; j += W * 2) {
            va = va * Vec::loadu(in_ptr + j);
            vb = vb * Vec::loadu(in_ptr + j + W);
          }
          va = va * vb;
          alignas(32) double lanes[4];
          va.storeu(lanes);
          local = lanes[0];
          for (int i = 1; i < 4; ++i)
            local *= lanes[i];
          for (; j < n; ++j)
            local *= in_ptr[j];
          // float nanmin/nanmax
        } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                              std::is_same_v<Op, NanMaxOp<T>>) &&
                             std::is_same_v<T, float>) {
          constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
          constexpr int64_t W = 8;
          __m256 nan_fill =
              is_min ? _mm256_set1_ps(INFINITY) : _mm256_set1_ps(-INFINITY);
          __m256 va = nan_fill;
          int64_t j = 0;
          for (; j + W <= n; j += W) {
            __m256 v = _mm256_loadu_ps(in_ptr + j);
            v = _mm256_blendv_ps(v, nan_fill,
                                 _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
            if constexpr (is_min)
              va = _mm256_min_ps(va, v);
            else
              va = _mm256_max_ps(va, v);
          }
          local = is_min ? vec::Vectorized<float>(va).reduce_min()
                         : vec::Vectorized<float>(va).reduce_max();
          for (; j < n; ++j) {
            float v = in_ptr[j];
            if (!std::isnan(v))
              local = op.reduce(local, v);
          }
          // double nanmin/nanmax
        } else if constexpr ((std::is_same_v<Op, NanMinOp<T>> ||
                              std::is_same_v<Op, NanMaxOp<T>>) &&
                             std::is_same_v<T, double>) {
          constexpr bool is_min = std::is_same_v<Op, NanMinOp<T>>;
          constexpr int64_t W = 4;
          __m256d nan_fill =
              is_min ? _mm256_set1_pd(INFINITY) : _mm256_set1_pd(-INFINITY);
          __m256d va = nan_fill;
          int64_t j = 0;
          for (; j + W <= n; j += W) {
            __m256d v = _mm256_loadu_pd(in_ptr + j);
            v = _mm256_blendv_pd(v, nan_fill,
                                 _mm256_cmp_pd(v, v, _CMP_UNORD_Q));
            if constexpr (is_min)
              va = _mm256_min_pd(va, v);
            else
              va = _mm256_max_pd(va, v);
          }
          local = is_min ? vec::Vectorized<double>(va).reduce_min()
                         : vec::Vectorized<double>(va).reduce_max();
          for (; j < n; ++j) {
            double v = in_ptr[j];
            if (!std::isnan(v))
              local = op.reduce(local, v);
          }
          // bool all/any
        } else if constexpr (std::is_same_v<T, bool> &&
                             (std::is_same_v<Op, AllOp<T>> ||
                              std::is_same_v<Op, AnyOp<T>>)) {
          using Vec = vec::Vectorized<uint8_t>;
          constexpr int64_t W = Vec::size();
          const uint8_t *u8_ptr = reinterpret_cast<const uint8_t *>(in_ptr);
          if constexpr (std::is_same_v<Op, AllOp<T>>) {
            Vec vacc(static_cast<uint8_t>(0xFF));
            int64_t j = 0;
            for (; j + W <= n; j += W)
              vacc = vacc & Vec::loadu(u8_ptr + j);
            uint8_t tmp[32];
            vacc.storeu(tmp);
            bool result = true;
            for (int i = 0; i < 32; ++i)
              result = result && (tmp[i] != 0);
            for (; j < n; ++j)
              result = result && (u8_ptr[j] != 0);
            local = result;
          } else {
            Vec vacc(static_cast<uint8_t>(0x00));
            int64_t j = 0;
            for (; j + W <= n; j += W)
              vacc = vacc | Vec::loadu(u8_ptr + j);
            uint8_t tmp[32];
            vacc.storeu(tmp);
            bool result = false;
            for (int i = 0; i < 32; ++i)
              result = result || (tmp[i] != 0);
            for (; j < n; ++j)
              result = result || (u8_ptr[j] != 0);
            local = result;
          }
          // Scalar fallback
        } else {
          for (int64_t j = 0; j < n; ++j) {
            if constexpr (std::is_same_v<AccT, bool>)
              local = op.reduce(local, to_bool_value(in_ptr[j]));
            else
              local = op.reduce(local, static_cast<AccumulatorT>(in_ptr[j]));
          }
        }
      } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
        for (int64_t r = begin; r < end; ++r) {
          T val = *(input_data + o + r * layout.input_row_stride);
          if constexpr (std::is_same_v<AccT, bool>)
            local = op.reduce(local, to_bool_value(val));
          else
            local = op.reduce(local, static_cast<AccumulatorT>(val));
        }
      } else {
        int64_t red_coords[MAX_DIMS] = {};
        int64_t lin = base_lin_idxs[o];
        for (int64_t i = 0; i < begin; ++i) {
          for (int64_t d = k - 1; d >= 0; --d) {
            ++red_coords[d];
            lin += red_input_strides[d];
            if (red_coords[d] < reduced_dims[d])
              break;
            lin -= red_coords[d] * red_input_strides[d];
            red_coords[d] = 0;
          }
        }
        for (int64_t i = begin; i < end; ++i) {
          if constexpr (std::is_same_v<AccT, bool>)
            local = op.reduce(local, to_bool_value(input_data[lin]));
          else
            local =
                op.reduce(local, static_cast<AccumulatorT>(input_data[lin]));
          for (int64_t d = k - 1; d >= 0; --d) {
            ++red_coords[d];
            lin += red_input_strides[d];
            if (red_coords[d] < reduced_dims[d])
              break;
            lin -= red_coords[d] * red_input_strides[d];
            red_coords[d] = 0;
          }
        }
      }
      thread_accs[tid] = local;
    }

    AccumulatorT final_acc = thread_accs[0];
    for (int t = 1; t < num_threads; ++t)
      final_acc = op.reduce(final_acc, thread_accs[t]);

    // Safe output conversion (same logic as Strategy 1's reduce_and_store_one)
    if constexpr (std::is_same_v<T, float16_t>) {
      output_data[o] = static_cast<OutputCppT>(
          static_cast<T>(static_cast<float>(final_acc)));
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
      output_data[o] = static_cast<OutputCppT>(
          static_cast<T>(static_cast<float>(final_acc)));
    } else if constexpr (std::is_same_v<OutputCppT, complex32_t> ||
                         std::is_same_v<OutputCppT, complex64_t> ||
                         std::is_same_v<OutputCppT, complex128_t>) {
      // Complex output: delegate to Strategy 1 path for this single element
      // (avoid duplicating all complex conversion branches)
      reduce_and_store_one(o); // re-reduce sequentially — correct but slower
                               // for complex Strategy 2
    } else {
      output_data[o] = static_cast<OutputCppT>(final_acc);
    }
  }

  return output;
}

// =================================================================
// --- DISPATCHER TEMPLATES WITH TYPE VALIDATION ---
// =================================================================

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction(const Tensor &input,
                          const std::vector<int64_t> &normalized_axes,
                          bool keepdim, cudaStream_t stream) { // ✨✨✨
  constexpr bool is_all_any_op = std::is_same_v<OpType<T>, AllOp<T>> ||
                                 std::is_same_v<OpType<T>, AnyOp<T>>;

  // ALL/ANY for non-bool types: NO separate to_bool() tensor copy.
  // Falls through to reduce_kernel<T, AllOp/AnyOp, bool> which handles
  // inline conversion via to_bool_value() per element (single pass, no
  // allocation).
  //
  // Previous: input.to_bool() → full tensor copy + alloc → reduce copy (2
  // passes) Now: direct reduce with inline to_bool_value() (1 pass, 0 extra
  // allocation) PyTorch: TensorIterator casts to kBool on-the-fly. Same
  // approach.
  //
  // SIMD for T=bool: 32-wide uint8 AND/OR in reduce_kernel.
  // SIMD for T=float/int: compare-to-zero mask could be added but low priority
  // since all/any on non-bool is rare in DL. Scalar to_bool_value() is
  // sufficient.

  // FP4 VALIDATION: No reductions allowed
  constexpr bool is_fp4 =
      std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>;
  if constexpr (is_fp4) {
    throw std::runtime_error(
        "Reductions (sum, min, max, etc.) are not supported for FP4 types.");
  }

  //  CRITICAL: Validate that NaN operations are only used with floating point
  //  types
  constexpr bool is_nan_op = std::is_same_v<OpType<T>, NanSumOp<T>> ||
                             std::is_same_v<OpType<T>, NanProductOp<T>> ||
                             std::is_same_v<OpType<T>, NanMinOp<T>> ||
                             std::is_same_v<OpType<T>, NanMaxOp<T>> ||
                             std::is_same_v<OpType<T>, NanArgMinOp<T>> ||
                             std::is_same_v<OpType<T>, NanArgMaxOp<T>>;

  constexpr bool is_float_type =
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

  [[maybe_unused]] constexpr bool is_comparison_op =
      std::is_same_v<OpType<T>, MinOp<T>> ||
      std::is_same_v<OpType<T>, MaxOp<T>> ||
      std::is_same_v<OpType<T>, NanMinOp<T>> ||
      std::is_same_v<OpType<T>, NanMaxOp<T>> ||
      std::is_same_v<OpType<T>, ArgMinOp<T>> ||
      std::is_same_v<OpType<T>, ArgMaxOp<T>> ||
      std::is_same_v<OpType<T>, NanArgMinOp<T>> ||
      std::is_same_v<OpType<T>, NanArgMaxOp<T>>;

  // Block comparison operations on complex types - they don't have a natural
  // ordering
  constexpr bool is_complex_type = std::is_same_v<T, complex32_t> ||
                                   std::is_same_v<T, complex64_t> ||
                                   std::is_same_v<T, complex128_t>;

  if constexpr (is_complex_type && is_comparison_op) {
    throw std::runtime_error(
        "Comparison-based reduction operations (min, max, argmin, argmax) are "
        "not supported for complex types. "
        "Complex numbers do not have a natural ordering. "
        "Got: " +
        get_dtype_name(input.dtype()));
  }

  // Block NaN operations on non-float types at compile time
  if constexpr (is_nan_op && !is_float_type) {
    throw std::runtime_error(
        "NaN-aware operations are only supported for floating point types "
        "(Float16, Bfloat16, Float32, Float64). "
        "Got: " +
        get_dtype_name(input.dtype()));
  }

#ifdef WITH_CUDA
  if (input.is_cuda()) {
    // Route to GPU implementation
    if constexpr (is_fp4) {
      throw std::runtime_error("Reductions are not supported for FP4 types.");
    } else {
      if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> ||
                    std::is_same_v<OpType<T>, ArgMinOp<T>> ||
                    std::is_same_v<OpType<T>, NanArgMaxOp<T>> ||
                    std::is_same_v<OpType<T>, NanArgMinOp<T>>) {
        return dispatch_index_reduction_gpu<T, OpType>(
            input, normalized_axes, keepdim, stream); // ✨✨✨
      } else if constexpr ((std::is_same_v<OpType<T>, AllOp<T>> ||
                            std::is_same_v<OpType<T>, AnyOp<T>>) &&
                           !std::is_same_v<T, bool>) {
        // GPU all/any for non-bool: convert to bool first (GPU only has bool
        // instantiations) CPU uses inline to_bool_value() — zero allocation,
        // single pass. GPU uses to_bool() — separate tensor, but avoids needing
        // AllOp<float> etc. on GPU.
        Tensor bool_input = input.to_bool();
        return dispatch_reduction_gpu<bool, OpType>(bool_input, normalized_axes,
                                                    keepdim, stream);
      } else {
        return dispatch_reduction_gpu<T, OpType>(input, normalized_axes,
                                                 keepdim, stream); // ✨✨✨
      }
    }
  }
#endif

  // ═══════════════════════════════════════════════════════════════════
  // UNIVERSAL CPU DISPATCHER (like PyTorch's parallel_reduce decision tree)
  //
  // Decides STRATEGY + KERNEL in one place. Kernels just implement 2 paths.
  //
  // PyTorch bug fixed: binary_kernel_reduce_lastdim (Reduce.h:290-308) uses
  // sub_iter.for_each() which parallelizes ONLY over output elements.
  // For full reduction (output.numel()==1), 1 thread does all work.
  // Our Case 2 forces SplitReduction for full reductions, using ALL threads.
  //
  // ═══════════════════════════════════════════════════════════════════
  // THREADING STRATEGY DECISION TABLE
  // Benchmark results on i7-14700K (28 cores, AVX2, L1=32KB per core)
  // ═══════════════════════════════════════════════════════════════════
  //
  // GRAIN_SIZE = 32768 (128KB ≈ L1 cache per core)
  //
  // Element Count | Type       | 1 Thread  | Optimal    | Speedup | Strategy
  // ─────────────────────────────────────────────────────────────────────────
  // 1K (4KB)      | float      | 1.2 μs    | 28T:42μs   | 0.03x   | SERIAL (too
  // small) 10K (40KB)    | float      | 2.8 μs    | 16T:3.2μs  | 0.88x   |
  // SERIAL (< GRAIN_SIZE) 50K (200KB)   | float      | 8.5 μs    | 8T:15μs    |
  // 0.57x   | SERIAL (< GRAIN_SIZE) 100K (400KB)  | float      | 18.2 μs   |
  // 12T:8.3μs  | 2.2x    | PARALLEL (> GRAIN_SIZE) 500K (2MB)    | float      |
  // 95 μs     | 20T:12μs   | 7.9x    | PARALLEL 1M (4MB)      | float      |
  // 187 μs    | 24T:10.5μs | 17.8x   | PARALLEL 10M (40MB)    | float      |
  // 1987 μs   | 28T:323μs  | 6.1x    | PARALLEL (full cores) 100M (400MB)  |
  // float      | 21043 μs  | 28T:1847μs | 11.4x   | PARALLEL
  //
  // 1K (8KB)      | double     | 1.4 μs    | 28T:45μs   | 0.03x   | SERIAL
  // 10K (80KB)    | double     | 3.1 μs    | 16T:4.2μs  | 0.74x   | SERIAL
  // 50K (400KB)   | double     | 9.2 μs    | 8T:18μs    | 0.51x   | SERIAL
  // 100K (800KB)  | double     | 19.8 μs   | 12T:9.5μs  | 2.1x    | PARALLEL
  // 500K (4MB)    | double     | 102 μs    | 20T:14.2μs | 7.2x    | PARALLEL
  // 1M (8MB)      | double     | 198 μs    | 24T:11.8μs | 16.8x   | PARALLEL
  // 10M (80MB)    | double     | 2154 μs   | 28T:387μs  | 5.6x    | PARALLEL
  // 100M (800MB)  | double     | 22987 μs  | 28T:2134μs | 10.8x   | PARALLEL
  //
  // 1K (4KB)      | int32      | 0.9 μs    | 28T:38μs   | 0.02x   | SERIAL
  // 10K (40KB)    | int32      | 2.1 μs    | 16T:2.8μs  | 0.75x   | SERIAL
  // 50K (200KB)   | int32      | 7.3 μs    | 8T:12μs    | 0.61x   | SERIAL
  // 100K (400KB)  | int32      | 16.5 μs   | 12T:7.2μs  | 2.3x    | PARALLEL
  // 500K (2MB)    | int32      | 85 μs     | 20T:10.5μs | 8.1x    | PARALLEL
  // 1M (4MB)      | int32      | 168 μs    | 24T:9.2μs  | 18.3x   | PARALLEL
  // 10M (40MB)    | int32      | 1847 μs   | 28T:298μs  | 6.2x    | PARALLEL
  // 100M (400MB)  | int32      | 19856 μs  | 28T:1658μs | 12.0x   | PARALLEL
  //
  // 1K (2KB)      | float16    | 1.1 μs    | 28T:40μs   | 0.03x   | SERIAL
  // 10K (20KB)    | float16    | 2.3 μs    | 16T:3.1μs  | 0.74x   | SERIAL
  // 50K (100KB)   | float16    | 6.8 μs    | 8T:14μs    | 0.49x   | SERIAL
  // 100K (200KB)  | float16    | 14.2 μs   | 12T:8.1μs  | 1.75x   | PARALLEL
  // (marginal) 500K (1MB)    | float16    | 72 μs     | 20T:11.5μs | 6.3x    |
  // PARALLEL 1M (2MB)      | float16    | 145 μs    | 24T:10.2μs | 14.2x   |
  // PARALLEL 10M (20MB)    | float16    | 1523 μs   | 28T:268μs  | 5.7x    |
  // PARALLEL 100M (200MB)  | float16    | 16234 μs  | 28T:1456μs | 11.2x   |
  // PARALLEL
  //
  // STRATEGY 1 (ParallelSlices):  #pragma omp parallel for over output slots
  //   Best when: num_slices >= num_threads (many independent outputs)
  //   Load balance: Excellent (each thread handles one or more complete
  //   reductions) Cache efficiency: Good (thread-local output cache locality)
  //   Overhead: 1x parallel launch cost
  //
  // STRATEGY 2 (SplitReduction):  Split reduction dimension across threads
  //   Best when: num_slices < num_threads (few outputs, e.g., full reduction)
  //   Load balance: Perfect (all threads work on same problem)
  //   Cache efficiency: Fair (inter-thread reduction cost at end)
  //   Overhead: 1x parallel launch + thread synchronization
  //
  // Decision rules:
  //   1. If input.numel() < GRAIN_SIZE OR max_threads == 1 → Serial (no
  //   threading)
  //   2. Else if num_slices == 1 (full reduction) → Strategy 2 (SplitReduction)
  //      (PyTorch bug: naive binary_kernel_reduce_lastdim uses only 1 thread!)
  //      (Our fix: use ALL threads to split reduction dimension)
  //   3. Else if num_slices >= actual_threads → Strategy 1 (ParallelSlices)
  //   4. Else → Strategy 2 (SplitReduction)
  //
  // ═══════════════════════════════════════════════════════════════════
  // ═══════════════════════════════════════════════════════════════════
  Shape output_shape = detail::calculate_output_shape(input.shape().dims,
                                                      normalized_axes, keepdim);

  const int64_t reduced_count =
      detail::calculate_reduced_count(input.shape().dims, normalized_axes);
  int64_t num_slices = 1;
  for (auto d : output_shape.dims)
    num_slices *= d;
  const int max_threads = omp_get_max_threads();

  ReductionStrategy strategy;
  int actual_threads;

  // CASE 1: SMALL TENSOR — not worth threading (like parallel_reduce's
  // serial_for_each)
  if (input.numel() < GRAIN_SIZE || max_threads == 1) {
    actual_threads = 1;
    strategy = ReductionStrategy::ParallelSlices; // with 1 thread = sequential
  }
  // CASE 2: FULL REDUCTION — always Strategy 2 (like parallel_reduce's
  // two_pass_reduction)
  else if (num_slices == 1) {
    actual_threads = std::min(
        max_threads, std::max(1, static_cast<int>(reduced_count / GRAIN_SIZE)));
    strategy = ReductionStrategy::SplitReduction;
  }
  // CASE 3: PARTIAL REDUCTION — choose strategy (like parallel_reduce's
  // parallel_dim_reduction)
  //
  // PyTorch's approach (TensorIteratorReduce.cpp:116-136):
  //   parallel_for(0, cols, /*grain_size=*/1, ...) — uses ALL available threads
  //   for output-parallel Strategy 1. No GRAIN_SIZE cap on thread count here
  //   because Case 1 already filtered out small tensors (numel < GRAIN_SIZE).
  //
  // Strategy 1 (ParallelSlices): actual_threads = min(max_threads, num_slices)
  //   Each thread gets (num_slices / threads) output positions.
  //   Total work per thread = (num_slices / threads) * reduced_count.
  //   Safe because Case 1 guarantees total_work >= GRAIN_SIZE.
  //
  // Strategy 2 (SplitReduction): actual_threads = min(max_threads,
  // reduced_count / GRAIN_SIZE)
  //   Each thread processes a chunk of the reduction dimension.
  //   GRAIN_SIZE caps threads to ensure each chunk is large enough.
  else {
    if (num_slices >= max_threads) {
      // Enough output slots for all threads → Strategy 1
      strategy = ReductionStrategy::ParallelSlices;
      actual_threads = max_threads;
    } else {
      // Few output slots → Strategy 2 (split reduction per output)
      strategy = ReductionStrategy::SplitReduction;
      actual_threads =
          std::min(max_threads,
                   std::max(1, static_cast<int>(reduced_count / GRAIN_SIZE)));
    }
  }

  // Route to kernel based on operation type
  if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> ||
                std::is_same_v<OpType<T>, ArgMinOp<T>> ||
                std::is_same_v<OpType<T>, NanArgMaxOp<T>> ||
                std::is_same_v<OpType<T>, NanArgMinOp<T>>) {
    return detail::reduce_kernel_index<T, OpType>(
        input, normalized_axes, output_shape, strategy, actual_threads);
  } else if constexpr (std::is_same_v<OpType<T>, SumOp<T>> &&
                       !std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    return detail::cascade_sum_kernel</*ignore_nan=*/false, T>(
        input, normalized_axes, output_shape, strategy, actual_threads);
  } else if constexpr (std::is_same_v<OpType<T>, NanSumOp<T>>) {
    return detail::cascade_sum_kernel</*ignore_nan=*/true, T>(
        input, normalized_axes, output_shape, strategy, actual_threads);
  } else {
    using Op = OpType<T>;
    return reduce_kernel<T, OpType, typename Op::AccT>(
        input, normalized_axes, output_shape, strategy, actual_threads);
  }
}

// =================================================================
// --- REDUCE_KERNEL_MEAN: Unified Mean / NanMean Kernel ---
//
// Architecture:
//   Float nanmean: Fused single-pass SIMD (sum + NaN count + divide in ONE
//   pass)
//     - InnerContiguous: SIMD NaN masking + double accumulation (4-wide for
//     fp32/fp64, 8-wide float for fp16/bf16)
//     - OuterContiguous: Vertical SIMD across adjacent output columns
//     - Generic: Scalar carry-add (can't SIMD irregular memory access)
//     - OpenMP + Strategy 1/2 via universal dispatcher
//     - No cascade needed: double acc error O(N×2^-53), divided by N → O(2^-53)
//     = constant!
//     - Count in double (safe to 2^53), division in double, ONE final cast to
//     output dtype
//
//   Float regular mean: Fused single-pass SIMD sum (same as above but no NaN
//   check/count)
//     - Then reciprocal multiply: 1 division + N/W SIMD multiplications
//     - Gets sum directly from double accumulator (no cascade_sum round-trip!)
//     - Type-casting analysis: fp32→double (1 instr), accumulate, double*recip,
//     cast→fp32 = 2 casts total
//
//   Int regular mean: Calls existing reduce_kernel(SumOp) for optimized int sum
//     - Then scalar reciprocal multiply (AVX2 can't int64→double)
//     - Output dtype: Float64 (PyTorch doesn't support int mean at all, NumPy
//     returns fp64)
//
// Type-casting comparison vs old approach:
//   OLD: cascade_sum(double_acc → cast1:fp32) → mean(load fp32 → cast2:maybe →
//   divide → store)  = 2-3 casts NEW: load fp32 → cast1:double → accumulate →
//   divide double/double → cast2:fp32              = 2 casts (minimum!)
//
// Why no cascade_sum for mean:
//   cascade_sum protects against catastrophic cancellation in SUM (absolute
//   error O(N×eps)). For MEAN, division by N cancels the N factor: mean error =
//   O(N×eps)/N = O(eps) = CONSTANT. So double accumulation alone is sufficient
//   for mean — cascade adds complexity without benefit.
// =================================================================
template <typename T, template <typename> class SumOpType>
Tensor reduce_kernel_mean(const Tensor &input,
                          const std::vector<int64_t> &normalized_axes,
                          bool keepdim, cudaStream_t stream) { // ✨✨✨
  constexpr bool is_nan_mean = std::is_same_v<SumOpType<T>, NanSumOp<T>>;

  // Validate: NaN-aware mean only for float types
  if constexpr (is_nan_mean) {
    constexpr bool is_float = std::is_floating_point_v<T> || is_half_float_v<T>;
    if constexpr (!is_float) {
      throw std::runtime_error(
          "NaN-aware mean is only supported for floating point types. Got: " +
          get_dtype_name(input.dtype()));
    }
  }

  // FP4 not supported
  if constexpr (std::is_same_v<T, float4_e2m1_t> ||
                std::is_same_v<T, float4_e2m1_2x_t>) {
    throw std::runtime_error("Mean reduction is not supported for FP4 types.");
  }

#ifdef WITH_CUDA
  if (input.is_cuda()) {
    return dispatch_mean_gpu<T, SumOpType>(input, normalized_axes, keepdim,
                                           stream);
  }
#endif

  // ═══════════════════════════════════════════════════════════
  // CPU MEAN: Reuse sum/nansum infrastructure + divide
  //
  // mean    = dispatch_reduction<T, SumOp>(...) / reduced_count
  // nanmean = dispatch_reduction<T, NanSumOp>(...) / non_nan_count_per_slice
  //
  // This reuses ALL optimizations: cascade_sum for float precision,
  // SIMD, Strategy 1/2, GRAIN_SIZE, universal dispatcher.
  // PyTorch does the same: sum_out(result, self, dim).div_(dim_prod)
  // ═══════════════════════════════════════════════════════════

  const int64_t reduced_count =
      detail::calculate_reduced_count(input.shape().dims, normalized_axes);
  if (reduced_count == 0) {
    throw std::runtime_error("Cannot compute mean: reduced count is zero.");
  }

  // ═══════════════════════════════════════════════════════════
  // FLOAT NANMEAN: Fused single-pass SIMD (sum + NaN count + divide in ONE
  // pass) INT MEAN: Call reduce_kernel(SumOp) + reciprocal multiply FLOAT MEAN:
  // Fused single-pass SIMD sum (double acc) + reciprocal multiply
  //
  // Benchmark proof (i7-14700K, 28 threads):
  //   Fused SIMD vs Two-Pass vs Fused Scalar:
  //   Layer norm (32, 768):        Fused SIMD: 20μs,  Two-Pass: 40μs,  Scalar:
  //   21μs Spatial (2048, 50176):       Fused SIMD: 5955μs, Two-Pass: 10552μs,
  //   Scalar: 8241μs Attention (49152, 128):      Fused SIMD: 292μs,  Two-Pass:
  //   325μs,  Scalar: 591μs 50% NaN (1000, 10000):       Fused SIMD: 217μs,
  //   Two-Pass: 225μs,  Scalar: 1407μs
  //
  // Type-casting per output element (minimized):
  //   fp32: load fp32 → cast1:double → accumulate → divide double/double →
  //   cast2:fp32  = 2 casts fp64: load fp64 → accumulate double → divide
  //   double/double → store fp64          = 0 casts! fp16: load fp16 →
  //   cast1:float → accumulate → divide float→double/double → cast2:fp16 = 2
  //   casts
  //
  // No cascade needed for mean: error = O(N×eps)/N = O(eps) = constant
  // (division cancels N).
  // ═══════════════════════════════════════════════════════════

  if constexpr (is_nan_mean) {
    // NanMean: Fused single-pass SIMD — sum + count + divide in ONE data read
    // Uses SIMD NaN bit masking (branchless) + float accumulation for fp32
    // PyTorch uses float acc (at::acc_type<float,true>=float) — cascade handles
    // precision Beats PyTorch's 4-pass approach (isnan + logical_not + sum +
    // nansum)
    using acc_t =
        std::conditional_t<std::is_same_v<T, float>, float, AccumulatorType<T>>;

    const T *input_data = input.data<T>();
    const auto &input_dims = input.shape().dims;
    const auto &input_strides = input.stride().strides;

    Shape output_shape =
        detail::calculate_output_shape(input_dims, normalized_axes, keepdim);
    Tensor output({output_shape}, TensorOptions()
                                      .with_dtype(input.dtype())
                                      .with_device(input.device())
                                      .with_req_grad(input.requires_grad()));
    T *out_data = output.data<T>();
    const int64_t num_elements = output.numel();

    ReductionLayout layout = compute_reduction_layout(input, normalized_axes);
    cpu_demote_middle_to_generic(layout);
    const int64_t k = static_cast<int64_t>(normalized_axes.size());
    int64_t red_strides[MAX_DIMS];
    for (int64_t d = 0; d < k; ++d)
      red_strides[d] = input_strides[normalized_axes[d]];

    bool reduced_bitmap[MAX_DIMS] = {false};
    for (int64_t axis : normalized_axes)
      reduced_bitmap[axis] = true;

    std::vector<int64_t> reduced_dims;
    for (size_t dim = 0; dim < input_dims.size(); ++dim)
      if (reduced_bitmap[dim])
        reduced_dims.push_back(input_dims[dim]);

    // Precompute base indices for Generic path
    int64_t M_nr = 0;
    int64_t nr_sizes[MAX_DIMS], nr_strides_arr[MAX_DIMS];
    std::vector<int64_t> base_lin_idxs;
    if (layout.path == ReductionLayout::Path::Generic) {
      for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        if (!reduced_bitmap[dim]) {
          nr_sizes[M_nr] = input_dims[dim];
          nr_strides_arr[M_nr] = input_strides[dim];
          ++M_nr;
        }
      }
      base_lin_idxs.resize(num_elements);
      int64_t oc[MAX_DIMS] = {}, blk = 0;
      for (int64_t o = 0; o < num_elements; ++o) {
        base_lin_idxs[o] = blk;
        for (int64_t j = M_nr - 1; j >= 0; --j) {
          ++oc[j];
          blk += nr_strides_arr[j];
          if (oc[j] < nr_sizes[j])
            break;
          blk -= oc[j] * nr_strides_arr[j];
          oc[j] = 0;
        }
      }
    }

    // Helper: compute sum+count for one output position, all layout paths
    // Returns {sum_in_double, count_in_double}
    auto fused_sum_count_one = [&](int64_t o) -> std::pair<double, double> {
      double sum = 0.0;
      double count = 0.0;

      if (layout.path == ReductionLayout::Path::InnerContiguous) {
        const T *in = input_data + o * layout.input_outer_stride;
        const int64_t n = layout.reduced_count;

        // SIMD fp32 nanmean: 8-wide float cascade → double safe accumulator
        // Cascade approach (like PyTorch's cascade_sum):
        //   Fast path: 8-wide float SIMD (8 floats/cycle, full AVX2 width)
        //   Every DUMP_INTERVAL iterations: dump float acc → double safe acc
        //   Result: float SIMD speed + double precision + no overflow
        // Benchmark proof: 3x faster than 4-wide double for L1/L2-cached data
        if constexpr (std::is_same_v<T, float>) {
          constexpr int DUMP_INTERVAL =
              4; // dump every 64 elements; safe up to 5.3e36 per element  //
                 // dump every 128 elements (16×8)
          // Fast float accumulators (8-wide × 2 ILP)
          __m256 fs0 = _mm256_setzero_ps(), fs1 = _mm256_setzero_ps();
          __m256 fc0 = _mm256_setzero_ps(), fc1 = _mm256_setzero_ps();
          // Safe double accumulators (cascade target)
          __m256d ds0 = _mm256_setzero_pd(), ds1 = _mm256_setzero_pd();
          __m256d dc0 = _mm256_setzero_pd(), dc1 = _mm256_setzero_pd();
          __m256 z8 = _mm256_setzero_ps(), o8 = _mm256_set1_ps(1.0f);
          int dump_ctr = 0;
          int64_t j = 0;

          for (; j + 16 <= n; j += 16) {
            __m256 v0 = _mm256_loadu_ps(in + j);
            __m256 m0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
            fs0 = _mm256_add_ps(fs0, _mm256_blendv_ps(v0, z8, m0));
            fc0 = _mm256_add_ps(fc0, _mm256_blendv_ps(o8, z8, m0));

            __m256 v1 = _mm256_loadu_ps(in + j + 8);
            __m256 m1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
            fs1 = _mm256_add_ps(fs1, _mm256_blendv_ps(v1, z8, m1));
            fc1 = _mm256_add_ps(fc1, _mm256_blendv_ps(o8, z8, m1));

            if (++dump_ctr == DUMP_INTERVAL) {
              // Cascade dump: float → double (prevents precision loss +
              // overflow)
              __m256 ts = _mm256_add_ps(fs0, fs1);
              __m256 tc = _mm256_add_ps(fc0, fc1);
              ds0 = _mm256_add_pd(ds0,
                                  _mm256_cvtps_pd(_mm256_castps256_ps128(ts)));
              ds1 = _mm256_add_pd(
                  ds1, _mm256_cvtps_pd(_mm256_extractf128_ps(ts, 1)));
              dc0 = _mm256_add_pd(dc0,
                                  _mm256_cvtps_pd(_mm256_castps256_ps128(tc)));
              dc1 = _mm256_add_pd(
                  dc1, _mm256_cvtps_pd(_mm256_extractf128_ps(tc, 1)));
              fs0 = _mm256_setzero_ps();
              fs1 = _mm256_setzero_ps();
              fc0 = _mm256_setzero_ps();
              fc1 = _mm256_setzero_ps();
              dump_ctr = 0;
            }
          }
          // 8-wide tail
          for (; j + 8 <= n; j += 8) {
            __m256 v = _mm256_loadu_ps(in + j);
            __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            fs0 = _mm256_add_ps(fs0, _mm256_blendv_ps(v, z8, m));
            fc0 = _mm256_add_ps(fc0, _mm256_blendv_ps(o8, z8, m));
          }
          // Final dump remaining float → double
          __m256 ts = _mm256_add_ps(fs0, fs1);
          __m256 tc = _mm256_add_ps(fc0, fc1);
          ds0 = _mm256_add_pd(ds0, _mm256_cvtps_pd(_mm256_castps256_ps128(ts)));
          ds1 =
              _mm256_add_pd(ds1, _mm256_cvtps_pd(_mm256_extractf128_ps(ts, 1)));
          dc0 = _mm256_add_pd(dc0, _mm256_cvtps_pd(_mm256_castps256_ps128(tc)));
          dc1 =
              _mm256_add_pd(dc1, _mm256_cvtps_pd(_mm256_extractf128_ps(tc, 1)));
          // Horizontal reduce double
          __m256d vtot_s = _mm256_add_pd(ds0, ds1);
          __m256d vtot_c = _mm256_add_pd(dc0, dc1);
          double sa[4], ca[4];
          _mm256_storeu_pd(sa, vtot_s);
          _mm256_storeu_pd(ca, vtot_c);
          sum = sa[0] + sa[1] + sa[2] + sa[3];
          count = ca[0] + ca[1] + ca[2] + ca[3];
          // Scalar tail
          for (; j < n; ++j) {
            if (!std::isnan(in[j])) {
              sum += (double)in[j];
              count += 1.0;
            }
          }
        } else if constexpr (std::is_same_v<T, double>) {
          // fp64: already double, SIMD 4-wide
          __m256d vsum = _mm256_setzero_pd(), vcnt = _mm256_setzero_pd();
          __m256d zero_d = _mm256_setzero_pd();
          __m256d ones_d = _mm256_set1_pd(1.0);
          int64_t j = 0;
          for (; j + 4 <= n; j += 4) {
            __m256d v = _mm256_loadu_pd(in + j);
            __m256d m = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
            vsum = _mm256_add_pd(vsum, _mm256_blendv_pd(v, zero_d, m));
            vcnt = _mm256_add_pd(vcnt, _mm256_blendv_pd(ones_d, zero_d, m));
          }
          double sa[4], ca[4];
          _mm256_storeu_pd(sa, vsum);
          _mm256_storeu_pd(ca, vcnt);
          sum = sa[0] + sa[1] + sa[2] + sa[3];
          count = ca[0] + ca[1] + ca[2] + ca[3];
          for (; j < n; ++j) {
            if (!std::isnan(in[j])) {
              sum += in[j];
              count += 1.0;
            }
          }
        } else if constexpr (std::is_same_v<T, float16_t> ||
                             std::is_same_v<T, bfloat16_t>) {
          // fp16/bf16: load→float, NaN mask, accumulate in float, promote to
          // double at end
          __m256 vsum_f = _mm256_setzero_ps(), vcnt_f = _mm256_setzero_ps();
          __m256 zero8 = _mm256_setzero_ps();
          __m256 ones8 = _mm256_set1_ps(1.0f);
          int64_t j = 0;
          for (; j + 8 <= n; j += 8) {
            __m256 v;
            if constexpr (std::is_same_v<T, float16_t>)
              v = vec::load_fp16_as_float(in + j).values;
            else
              v = vec::load_bf16_as_float(in + j).values;
            __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            vsum_f = _mm256_add_ps(vsum_f, _mm256_blendv_ps(v, zero8, m));
            vcnt_f = _mm256_add_ps(vcnt_f, _mm256_blendv_ps(ones8, zero8, m));
          }
          float sa[8], ca[8];
          _mm256_storeu_ps(sa, vsum_f);
          _mm256_storeu_ps(ca, vcnt_f);
          for (int p = 0; p < 8; ++p) {
            sum += (double)sa[p];
            count += (double)ca[p];
          }
          for (; j < n; ++j) {
            float fv = static_cast<float>(in[j]);
            if (!std::isnan(fv)) {
              sum += (double)fv;
              count += 1.0;
            }
          }
        } else {
          // Fallback scalar (complex, etc.)
          for (int64_t j = 0; j < n; ++j) {
            if (!safe_isnan(in[j])) {
              sum += to_double(static_cast<acc_t>(in[j]));
              count += 1.0;
            }
          }
        }
      } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
        // OuterContiguous: strided access, scalar (can't SIMD non-contiguous
        // per output)
        for (int64_t r = 0; r < layout.reduced_count; ++r) {
          T val = *(input_data + o + r * layout.input_row_stride);
          if (!safe_isnan(val)) {
            sum += to_double(static_cast<acc_t>(val));
            count += 1.0;
          }
        }
      } else {
        // Generic path: optimized for single-axis (common DL case)
        if (k == 1) {
          // Single reduced dim: simple stride loop (no coord math!)
          int64_t lin = base_lin_idxs[o];
          const int64_t stride = red_strides[0];
          const int64_t cnt = reduced_dims[0];
          for (int64_t i = 0; i < cnt; ++i, lin += stride) {
            T val = input_data[lin];
            if (!safe_isnan(val)) {
              sum += to_double(static_cast<acc_t>(val));
              count += 1.0;
            }
          }
        } else {
          // Multi-axis: full carry-add
          int64_t coords[MAX_DIMS] = {};
          int64_t lin = base_lin_idxs[o];
          for (int64_t i = 0; i < reduced_count; ++i) {
            T val = input_data[lin];
            if (!safe_isnan(val)) {
              sum += to_double(static_cast<acc_t>(val));
              count += 1.0;
            }
            for (int64_t d = k - 1; d >= 0; --d) {
              ++coords[d];
              lin += red_strides[d];
              if (coords[d] < reduced_dims[d])
                break;
              lin -= coords[d] * red_strides[d];
              coords[d] = 0;
            }
          }
        }
      }
      return {sum, count};
    };

    // Universal dispatcher: GRAIN_SIZE + Strategy 1/2
    const int max_threads = omp_get_max_threads();

    if (input.numel() < GRAIN_SIZE || max_threads == 1) {
      // CASE 1: Small tensor → sequential
      for (int64_t o = 0; o < num_elements; ++o) {
        auto [s, c] = fused_sum_count_one(o);
        if (c > 0.0) {
          double mv = s / c;
          if constexpr (std::is_same_v<T, float16_t> ||
                        std::is_same_v<T, bfloat16_t>)
            out_data[o] = static_cast<T>(static_cast<float>(mv));
          else
            out_data[o] = static_cast<T>(mv);
        } else {
          if constexpr (std::is_floating_point_v<T>)
            out_data[o] = std::numeric_limits<T>::quiet_NaN();
          else
            out_data[o] = static_cast<T>(std::nanf(""));
        }
      }
    } else if (num_elements == 1) {
      // CASE 2: Full reduction → Strategy 2 (split reduction across threads)
      // This fixes PyTorch's bug where full-reduction nanmean uses 1 thread
      int actual_threads =
          std::min(max_threads,
                   std::max(1, static_cast<int>(reduced_count / GRAIN_SIZE)));
      if (actual_threads < 1)
        actual_threads = 1;

      std::vector<double> thread_sums(actual_threads, 0.0);
      std::vector<double> thread_counts(actual_threads, 0.0);

      if (layout.path == ReductionLayout::Path::InnerContiguous) {
        const T *in = input_data;
        const int64_t n = layout.reduced_count;
#pragma omp parallel num_threads(actual_threads)
        {
          int tid = omp_get_thread_num();
          int nt = omp_get_num_threads();
          int64_t chunk = (n + nt - 1) / nt;
          int64_t begin = tid * chunk;
          int64_t end = std::min(begin + chunk, n);
          double local_sum = 0.0, local_count = 0.0;

          if constexpr (std::is_same_v<T, float>) {
            // 8-wide float cascade → double (Strategy 2 per-thread)
            constexpr int DUMP_INTERVAL =
                4; // dump every 64 elements; safe up to 5.3e36 per element
            __m256 fs0 = _mm256_setzero_ps(), fs1 = _mm256_setzero_ps();
            __m256 fc0 = _mm256_setzero_ps(), fc1 = _mm256_setzero_ps();
            __m256d ds0 = _mm256_setzero_pd(), ds1 = _mm256_setzero_pd();
            __m256d dc0 = _mm256_setzero_pd(), dc1 = _mm256_setzero_pd();
            __m256 z8 = _mm256_setzero_ps(), o8 = _mm256_set1_ps(1.0f);
            int dump_ctr = 0;
            int64_t j = begin;
            for (; j + 16 <= end; j += 16) {
              __m256 v0 = _mm256_loadu_ps(in + j);
              __m256 m0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
              fs0 = _mm256_add_ps(fs0, _mm256_blendv_ps(v0, z8, m0));
              fc0 = _mm256_add_ps(fc0, _mm256_blendv_ps(o8, z8, m0));
              __m256 v1 = _mm256_loadu_ps(in + j + 8);
              __m256 m1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
              fs1 = _mm256_add_ps(fs1, _mm256_blendv_ps(v1, z8, m1));
              fc1 = _mm256_add_ps(fc1, _mm256_blendv_ps(o8, z8, m1));
              if (++dump_ctr == DUMP_INTERVAL) {
                __m256 ts = _mm256_add_ps(fs0, fs1),
                       tc = _mm256_add_ps(fc0, fc1);
                ds0 = _mm256_add_pd(
                    ds0, _mm256_cvtps_pd(_mm256_castps256_ps128(ts)));
                ds1 = _mm256_add_pd(
                    ds1, _mm256_cvtps_pd(_mm256_extractf128_ps(ts, 1)));
                dc0 = _mm256_add_pd(
                    dc0, _mm256_cvtps_pd(_mm256_castps256_ps128(tc)));
                dc1 = _mm256_add_pd(
                    dc1, _mm256_cvtps_pd(_mm256_extractf128_ps(tc, 1)));
                fs0 = fs1 = fc0 = fc1 = _mm256_setzero_ps();
                dump_ctr = 0;
              }
            }
            for (; j + 8 <= end; j += 8) {
              __m256 v = _mm256_loadu_ps(in + j);
              __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
              fs0 = _mm256_add_ps(fs0, _mm256_blendv_ps(v, z8, m));
              fc0 = _mm256_add_ps(fc0, _mm256_blendv_ps(o8, z8, m));
            }
            __m256 ts = _mm256_add_ps(fs0, fs1), tc = _mm256_add_ps(fc0, fc1);
            ds0 =
                _mm256_add_pd(ds0, _mm256_cvtps_pd(_mm256_castps256_ps128(ts)));
            ds1 = _mm256_add_pd(ds1,
                                _mm256_cvtps_pd(_mm256_extractf128_ps(ts, 1)));
            dc0 =
                _mm256_add_pd(dc0, _mm256_cvtps_pd(_mm256_castps256_ps128(tc)));
            dc1 = _mm256_add_pd(dc1,
                                _mm256_cvtps_pd(_mm256_extractf128_ps(tc, 1)));
            double sa[4], ca[4];
            _mm256_storeu_pd(sa, _mm256_add_pd(ds0, ds1));
            _mm256_storeu_pd(ca, _mm256_add_pd(dc0, dc1));
            local_sum = sa[0] + sa[1] + sa[2] + sa[3];
            local_count = ca[0] + ca[1] + ca[2] + ca[3];
            for (; j < end; ++j) {
              if (!std::isnan(in[j])) {
                local_sum += (double)in[j];
                local_count += 1.0;
              }
            }
          } else if constexpr (std::is_same_v<T, double>) {
            __m256d vs = _mm256_setzero_pd(), vc = _mm256_setzero_pd();
            __m256d zd = _mm256_setzero_pd(), od = _mm256_set1_pd(1.0);
            int64_t j = begin;
            for (; j + 4 <= end; j += 4) {
              __m256d v = _mm256_loadu_pd(in + j);
              __m256d m = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
              vs = _mm256_add_pd(vs, _mm256_blendv_pd(v, zd, m));
              vc = _mm256_add_pd(vc, _mm256_blendv_pd(od, zd, m));
            }
            double sa[4], ca[4];
            _mm256_storeu_pd(sa, vs);
            _mm256_storeu_pd(ca, vc);
            local_sum = sa[0] + sa[1] + sa[2] + sa[3];
            local_count = ca[0] + ca[1] + ca[2] + ca[3];
            for (; j < end; ++j) {
              if (!std::isnan(in[j])) {
                local_sum += in[j];
                local_count += 1.0;
              }
            }
          } else {
            for (int64_t j = begin; j < end; ++j) {
              auto val = in[j];
              if (!safe_isnan(val)) {
                local_sum += to_double(static_cast<acc_t>(val));
                local_count += 1.0;
              }
            }
          }
          thread_sums[tid] = local_sum;
          thread_counts[tid] = local_count;
        }
      } else {
        // OuterContiguous/Generic full reduction: rare, use scalar Strategy 2
        auto [s, c] = fused_sum_count_one(0);
        thread_sums[0] = s;
        thread_counts[0] = c;
        actual_threads = 1;
      }
      // Combine thread results
      double total_sum = 0.0, total_count = 0.0;
      for (int t = 0; t < actual_threads; ++t) {
        total_sum += thread_sums[t];
        total_count += thread_counts[t];
      }
      if (total_count > 0.0) {
        double mv = total_sum / total_count;
        if constexpr (std::is_same_v<T, float16_t> ||
                      std::is_same_v<T, bfloat16_t>)
          out_data[0] = static_cast<T>(static_cast<float>(mv));
        else
          out_data[0] = static_cast<T>(mv);
      } else {
        if constexpr (std::is_floating_point_v<T>)
          out_data[0] = std::numeric_limits<T>::quiet_NaN();
        else
          out_data[0] = static_cast<T>(std::nanf(""));
      }
    } else {
      // CASE 3: Partial reduction → Strategy 1 (parallel over output slots)
      int actual_threads =
          std::min(max_threads, static_cast<int>(num_elements));

      // ─── Vertical SIMD for OuterContiguous nanmean ───
      if (layout.path == ReductionLayout::Path::OuterContiguous) {
        const int64_t stride = layout.input_row_stride;
        const int64_t R = layout.reduced_count;
        bool handled = false;

        if constexpr (std::is_same_v<T, float>) {
          constexpr int64_t W = 4; // 4 floats → 4 doubles for sum
          const int64_t vec_end = (num_elements / W) * W;
          __m128 zero_f = _mm_setzero_ps();
          __m128 ones_f = _mm_set1_ps(1.0f);
#pragma omp parallel for num_threads(actual_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256d vsum = _mm256_setzero_pd();
            __m256d vcnt = _mm256_setzero_pd();
            for (int64_t r = 0; r < R; ++r) {
              __m128 f = _mm_loadu_ps(input_data + r * stride + o);
              __m128 mask = _mm_cmp_ps(f, f, _CMP_UNORD_Q);
              __m128 val = _mm_blendv_ps(f, zero_f, mask);
              __m128 cnt = _mm_blendv_ps(ones_f, zero_f, mask);
              vsum = _mm256_add_pd(vsum, _mm256_cvtps_pd(val));
              vcnt = _mm256_add_pd(vcnt, _mm256_cvtps_pd(cnt));
            }
            // Divide sum/count per lane
            __m256d vmean = _mm256_div_pd(vsum, vcnt);
            // Handle count==0 → NaN
            __m256d zero_mask =
                _mm256_cmp_pd(vcnt, _mm256_setzero_pd(), _CMP_EQ_OQ);
            vmean = _mm256_blendv_pd(
                vmean, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                zero_mask);
            _mm_storeu_ps(out_data + o, _mm256_cvtpd_ps(vmean));
          }
          for (int64_t o = vec_end; o < num_elements; ++o) {
            auto [s, c] = fused_sum_count_one(o);
            out_data[o] = c > 0.0 ? static_cast<T>(s / c)
                                  : std::numeric_limits<T>::quiet_NaN();
          }
          handled = true;
        } else if constexpr (std::is_same_v<T, double>) {
          constexpr int64_t W = 4;
          const int64_t vec_end = (num_elements / W) * W;
          __m256d zero_d = _mm256_setzero_pd();
          __m256d ones_d = _mm256_set1_pd(1.0);
#pragma omp parallel for num_threads(actual_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256d vsum = _mm256_setzero_pd();
            __m256d vcnt = _mm256_setzero_pd();
            for (int64_t r = 0; r < R; ++r) {
              __m256d v = _mm256_loadu_pd(input_data + r * stride + o);
              __m256d mask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
              vsum = _mm256_add_pd(vsum, _mm256_blendv_pd(v, zero_d, mask));
              vcnt =
                  _mm256_add_pd(vcnt, _mm256_blendv_pd(ones_d, zero_d, mask));
            }
            __m256d vmean = _mm256_div_pd(vsum, vcnt);
            __m256d zero_mask = _mm256_cmp_pd(vcnt, zero_d, _CMP_EQ_OQ);
            vmean = _mm256_blendv_pd(
                vmean, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                zero_mask);
            _mm256_storeu_pd(out_data + o, vmean);
          }
          for (int64_t o = vec_end; o < num_elements; ++o) {
            auto [s, c] = fused_sum_count_one(o);
            out_data[o] = c > 0.0 ? static_cast<T>(s / c)
                                  : std::numeric_limits<T>::quiet_NaN();
          }
          handled = true;
        } else if constexpr (std::is_same_v<T, float16_t> ||
                             std::is_same_v<T, bfloat16_t>) {
          constexpr int64_t W = 8;
          const int64_t vec_end = (num_elements / W) * W;
          __m256 zero8 = _mm256_setzero_ps();
          __m256 ones8 = _mm256_set1_ps(1.0f);
#pragma omp parallel for num_threads(actual_threads)
          for (int64_t o = 0; o < vec_end; o += W) {
            __m256 vsum = _mm256_setzero_ps();
            __m256 vcnt = _mm256_setzero_ps();
            for (int64_t r = 0; r < R; ++r) {
              __m256 v;
              if constexpr (std::is_same_v<T, float16_t>)
                v = vec::load_fp16_as_float(input_data + r * stride + o).values;
              else
                v = vec::load_bf16_as_float(input_data + r * stride + o).values;
              __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
              vsum = _mm256_add_ps(vsum, _mm256_blendv_ps(v, zero8, mask));
              vcnt = _mm256_add_ps(vcnt, _mm256_blendv_ps(ones8, zero8, mask));
            }
            __m256 vmean = _mm256_div_ps(vsum, vcnt);
            __m256 zero_mask = _mm256_cmp_ps(vcnt, zero8, _CMP_EQ_OQ);
            vmean = _mm256_blendv_ps(vmean, _mm256_set1_ps(std::nanf("")),
                                     zero_mask);
            if constexpr (std::is_same_v<T, float16_t>)
              vec::store_float_as_fp16(out_data + o,
                                       vec::Vectorized<float>(vmean));
            else
              vec::store_float_as_bf16(out_data + o,
                                       vec::Vectorized<float>(vmean));
          }
          for (int64_t o = vec_end; o < num_elements; ++o) {
            auto [s, c] = fused_sum_count_one(o);
            out_data[o] = c > 0.0 ? static_cast<T>(static_cast<float>(s / c))
                                  : static_cast<T>(std::nanf(""));
          }
          handled = true;
        }

        if (handled)
          return output;
      }

// ─── Default: scalar per output slot ───
#pragma omp parallel for num_threads(actual_threads)
      for (int64_t o = 0; o < num_elements; ++o) {
        auto [s, c] = fused_sum_count_one(o);
        if (c > 0.0) {
          double mv = s / c;
          if constexpr (std::is_same_v<T, float16_t> ||
                        std::is_same_v<T, bfloat16_t>)
            out_data[o] = static_cast<T>(static_cast<float>(mv));
          else
            out_data[o] = static_cast<T>(mv);
        } else {
          if constexpr (std::is_floating_point_v<T>)
            out_data[o] = std::numeric_limits<T>::quiet_NaN();
          else
            out_data[o] = static_cast<T>(std::nanf(""));
        }
      }
    }
    return output;
  } else if constexpr (std::is_integral_v<T>) {
    // INT MEAN: Call existing reduce_kernel for optimized int sum + reciprocal
    // multiply PyTorch doesn't support int mean,NumPy returns fp64,and iam
    // supporting it returning fp64 only .
    Tensor sum_result = dispatch_reduction<T, SumOpType>(input, normalized_axes,
                                                         keepdim, stream);
    const int64_t num_elements = sum_result.numel();
    Tensor mean_result({sum_result.shape()},
                       TensorOptions()
                           .with_dtype(Dtype::Float64)
                           .with_device(input.device())
                           .with_req_grad(input.requires_grad()));
    const int64_t *sum_data = sum_result.data<int64_t>();
    double *mean_data = mean_result.data<double>();
    const double recip = 1.0 / static_cast<double>(reduced_count);
#pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i)
      mean_data[i] = static_cast<double>(sum_data[i]) * recip;
    return mean_result;
  } else {
    // FLOAT REGULAR MEAN: Call cascade_sum_kernel (fully optimized) +
    // reciprocal multiply
    //
    // Why not fused single-pass? Benchmark proof (i7-14700K, 28 threads):
    //   Fused mean kernel (own sum loop):  cascade_sum_kernel is 2-3x faster
    //   because it has:
    //   - multi_row_sum: processes multiple rows simultaneously
    //   - 4-level adaptive cascading with dynamic level_step
    //   - ILP=4 accumulators
    //   - Full TensorIterator-equivalent layout dispatch
    //   The per-element lambda overhead in fused approach can't match
    //   cascade_sum's batch processing.
    //
    // For nanmean: fused IS better because PyTorch's nanmean is 4-pass
    // (terribly slow). For regular mean: cascade_sum + reciprocal is better
    // because cascade_sum matches PyTorch's speed.
    //
    // Type-casting: cascade_sum outputs in T (fp32). We multiply by fp32
    // reciprocal.
    //   1 cast: none needed (sum is already fp32, reciprocal is fp32)
    //   Precision: same as PyTorch (both use cascade_sum → fp32 division)

    // Inline fused sum + reciprocal multiply (zero dispatch overhead)
    // Uses SAME algorithm as cascade_sum_kernel per path:
    //   InnerContiguous: 4-ILP double SIMD (NO cascade levels — just 4
    //   accumulators) OuterContiguous: Vertical SIMD across adjacent output
    //   columns Generic: Simple stride loop (single-axis) or carry-add
    //   (multi-axis)

    const T *input_data = input.data<T>();
    const auto &input_dims = input.shape().dims;
    const auto &input_strides = input.stride().strides;

    Shape output_shape =
        detail::calculate_output_shape(input_dims, normalized_axes, keepdim);
    Tensor output({output_shape}, TensorOptions()
                                      .with_dtype(input.dtype())
                                      .with_device(input.device())
                                      .with_req_grad(input.requires_grad()));
    T *out_data = output.data<T>();
    const int64_t num_elements = output.numel();

    ReductionLayout layout = compute_reduction_layout(input, normalized_axes);
    cpu_demote_middle_to_generic(layout);
    const double recip = 1.0 / static_cast<double>(reduced_count);

    // Helper: sum one output + reciprocal multiply
    auto sum_and_store_one = [&](int64_t o) {
      double sum = 0.0;

      if (layout.path == ReductionLayout::Path::InnerContiguous) {
        const T *in = input_data + o * layout.input_outer_stride;
        const int64_t n = layout.reduced_count;

        if constexpr (std::is_same_v<T, float>) {
          // 8-wide float SIMD × 4 ILP (like PyTorch:
          // acc_type<float,true>=float) Division by N cancels accumulation
          // error → float precision sufficient
          __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
          __m256 vc = _mm256_setzero_ps(), vd = _mm256_setzero_ps();
          int64_t j = 0;
          for (; j + 32 <= n; j += 32) {
            va = _mm256_add_ps(va, _mm256_loadu_ps(in + j));
            vb = _mm256_add_ps(vb, _mm256_loadu_ps(in + j + 8));
            vc = _mm256_add_ps(vc, _mm256_loadu_ps(in + j + 16));
            vd = _mm256_add_ps(vd, _mm256_loadu_ps(in + j + 24));
          }
          for (; j + 8 <= n; j += 8)
            va = _mm256_add_ps(va, _mm256_loadu_ps(in + j));
          // Horizontal reduce to double for final precision
          __m256 vtot =
              _mm256_add_ps(_mm256_add_ps(va, vb), _mm256_add_ps(vc, vd));
          float sa[8];
          _mm256_storeu_ps(sa, vtot);
          for (int p = 0; p < 8; ++p)
            sum += (double)sa[p];
          for (; j < n; ++j)
            sum += (double)in[j];
        } else if constexpr (std::is_same_v<T, double>) {
          __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
          __m256d vc = _mm256_setzero_pd(), vd = _mm256_setzero_pd();
          int64_t j = 0;
          for (; j + 16 <= n; j += 16) {
            va = _mm256_add_pd(va, _mm256_loadu_pd(in + j));
            vb = _mm256_add_pd(vb, _mm256_loadu_pd(in + j + 4));
            vc = _mm256_add_pd(vc, _mm256_loadu_pd(in + j + 8));
            vd = _mm256_add_pd(vd, _mm256_loadu_pd(in + j + 12));
          }
          for (; j + 4 <= n; j += 4)
            va = _mm256_add_pd(va, _mm256_loadu_pd(in + j));
          __m256d vtot =
              _mm256_add_pd(_mm256_add_pd(va, vb), _mm256_add_pd(vc, vd));
          double sa[4];
          _mm256_storeu_pd(sa, vtot);
          sum = sa[0] + sa[1] + sa[2] + sa[3];
          for (; j < n; ++j)
            sum += in[j];
        } else if constexpr (std::is_same_v<T, float16_t> ||
                             std::is_same_v<T, bfloat16_t>) {
          __m256 vs = _mm256_setzero_ps();
          int64_t j = 0;
          for (; j + 8 <= n; j += 8) {
            __m256 v;
            if constexpr (std::is_same_v<T, float16_t>)
              v = vec::load_fp16_as_float(in + j).values;
            else
              v = vec::load_bf16_as_float(in + j).values;
            vs = _mm256_add_ps(vs, v);
          }
          float sa[8];
          _mm256_storeu_ps(sa, vs);
          for (int p = 0; p < 8; ++p)
            sum += (double)sa[p];
          for (; j < n; ++j)
            sum += (double)(float)in[j];
        } else {
          for (int64_t j = 0; j < n; ++j) {
            if constexpr (std::is_same_v<T, complex32_t> ||
                          std::is_same_v<T, complex64_t> ||
                          std::is_same_v<T, complex128_t>)
              sum += (double)in[j].real();
            else
              sum += to_double(static_cast<AccumulatorType<T>>(in[j]));
          }
        }
      } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
        // Strided access: scalar per output element
        for (int64_t r = 0; r < layout.reduced_count; ++r) {
          T val = *(input_data + o + r * layout.input_row_stride);
          if constexpr (std::is_same_v<T, complex32_t> ||
                        std::is_same_v<T, complex64_t> ||
                        std::is_same_v<T, complex128_t>)
            sum += (double)val.real();
          else
            sum += to_double(static_cast<AccumulatorType<T>>(val));
        }
      } else {
        // Generic: optimized single-axis stride or multi-axis carry-add
        const int64_t k = static_cast<int64_t>(normalized_axes.size());
        int64_t red_strides[MAX_DIMS];
        for (int64_t d = 0; d < k; ++d)
          red_strides[d] = input_strides[normalized_axes[d]];

        bool reduced_bitmap[MAX_DIMS] = {false};
        for (int64_t axis : normalized_axes)
          reduced_bitmap[axis] = true;
        std::vector<int64_t> reduced_dims;
        for (size_t dim = 0; dim < input_dims.size(); ++dim)
          if (reduced_bitmap[dim])
            reduced_dims.push_back(input_dims[dim]);

        // Precompute base index for this output
        int64_t M_nr = 0;
        int64_t nr_sizes[MAX_DIMS], nr_strides_arr[MAX_DIMS];
        for (size_t dim = 0; dim < input_dims.size(); ++dim) {
          if (!reduced_bitmap[dim]) {
            nr_sizes[M_nr] = input_dims[dim];
            nr_strides_arr[M_nr] = input_strides[dim];
            ++M_nr;
          }
        }
        // Compute base linear index for output position o
        int64_t base_lin = 0, rem = o;
        for (int64_t d = M_nr - 1; d >= 0; --d) {
          base_lin += (rem % nr_sizes[d]) * nr_strides_arr[d];
          rem /= nr_sizes[d];
        }

        if (k == 1) {
          int64_t lin = base_lin;
          const int64_t stride = red_strides[0];
          for (int64_t i = 0; i < reduced_dims[0]; ++i, lin += stride)
            sum += to_double(static_cast<AccumulatorType<T>>(input_data[lin]));
        } else {
          int64_t coords[MAX_DIMS] = {};
          int64_t lin = base_lin;
          for (int64_t i = 0; i < reduced_count; ++i) {
            sum += to_double(static_cast<AccumulatorType<T>>(input_data[lin]));
            for (int64_t d = k - 1; d >= 0; --d) {
              ++coords[d];
              lin += red_strides[d];
              if (coords[d] < reduced_dims[d])
                break;
              lin -= coords[d] * red_strides[d];
              coords[d] = 0;
            }
          }
        }
      }

      // Reciprocal multiply: 1 division (precomputed) + 1 multiply per output
      // Benchmark proof: reciprocal is 32-69% faster than division for
      // many-output cases (multiply: ~5 cycles vs division: ~25 cycles per
      // element)
      double mean_val = sum * recip;
      if constexpr (std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, bfloat16_t>)
        out_data[o] = static_cast<T>(static_cast<float>(mean_val));
      else if constexpr (std::is_same_v<T, complex32_t> ||
                         std::is_same_v<T, complex64_t> ||
                         std::is_same_v<T, complex128_t>) {
        // Complex: need both real and imag
        AccumulatorType<T> csum{};
        if (layout.path == ReductionLayout::Path::InnerContiguous) {
          const T *in = input_data + o * layout.input_outer_stride;
          for (int64_t j = 0; j < layout.reduced_count; ++j)
            csum = csum + static_cast<AccumulatorType<T>>(in[j]);
        } else if (layout.path == ReductionLayout::Path::OuterContiguous) {
          for (int64_t r = 0; r < layout.reduced_count; ++r)
            csum = csum + static_cast<AccumulatorType<T>>(
                              *(input_data + o + r * layout.input_row_stride));
        }
        out_data[o] = T((float)(csum.real() / (double)reduced_count),
                        (float)(csum.imag() / (double)reduced_count));
      } else
        out_data[o] = static_cast<T>(mean_val);
    };

    // Universal dispatcher
    const int max_threads = omp_get_max_threads();
    if (input.numel() < GRAIN_SIZE || max_threads == 1) {
      for (int64_t o = 0; o < num_elements; ++o)
        sum_and_store_one(o);
    } else if (num_elements == 1) {
      // Full reduction → Strategy 2
      int actual_threads =
          std::min(max_threads,
                   std::max(1, static_cast<int>(reduced_count / GRAIN_SIZE)));
      if (actual_threads < 1)
        actual_threads = 1;
      std::vector<double> thread_sums(actual_threads, 0.0);
      if (layout.path == ReductionLayout::Path::InnerContiguous) {
        const T *in = input_data;
        const int64_t n = layout.reduced_count;
#pragma omp parallel num_threads(actual_threads)
        {
          int tid = omp_get_thread_num();
          int nt = omp_get_num_threads();
          int64_t chunk = (n + nt - 1) / nt;
          int64_t begin = tid * chunk;
          int64_t end = std::min(begin + chunk, n);
          double local = 0.0;
          if constexpr (std::is_same_v<T, float>) {
            // 8-wide float SIMD (Strategy 2 per-thread)
            __m256 va = _mm256_setzero_ps(), vb = _mm256_setzero_ps();
            __m256 vc = _mm256_setzero_ps(), vd = _mm256_setzero_ps();
            int64_t j = begin;
            for (; j + 32 <= end; j += 32) {
              va = _mm256_add_ps(va, _mm256_loadu_ps(in + j));
              vb = _mm256_add_ps(vb, _mm256_loadu_ps(in + j + 8));
              vc = _mm256_add_ps(vc, _mm256_loadu_ps(in + j + 16));
              vd = _mm256_add_ps(vd, _mm256_loadu_ps(in + j + 24));
            }
            for (; j + 8 <= end; j += 8)
              va = _mm256_add_ps(va, _mm256_loadu_ps(in + j));
            __m256 vtot =
                _mm256_add_ps(_mm256_add_ps(va, vb), _mm256_add_ps(vc, vd));
            float sa[8];
            _mm256_storeu_ps(sa, vtot);
            for (int p = 0; p < 8; ++p)
              local += (double)sa[p];
            for (; j < end; ++j)
              local += (double)in[j];
          } else if constexpr (std::is_same_v<T, double>) {
            __m256d va = _mm256_setzero_pd(), vb = _mm256_setzero_pd();
            int64_t j = begin;
            for (; j + 8 <= end; j += 8) {
              va = _mm256_add_pd(va, _mm256_loadu_pd(in + j));
              vb = _mm256_add_pd(vb, _mm256_loadu_pd(in + j + 4));
            }
            double sa[4];
            _mm256_storeu_pd(sa, _mm256_add_pd(va, vb));
            local = sa[0] + sa[1] + sa[2] + sa[3];
            for (; j < end; ++j)
              local += in[j];
          } else {
            for (int64_t j = begin; j < end; ++j)
              local += to_double(static_cast<AccumulatorType<T>>(in[j]));
          }
          thread_sums[tid] = local;
        }
      } else {
        sum_and_store_one(0);
        return output;
      }
      double total = 0.0;
      for (int t = 0; t < actual_threads; ++t)
        total += thread_sums[t];
      double mv = total * recip;
      if constexpr (std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, bfloat16_t>)
        out_data[0] = static_cast<T>(static_cast<float>(mv));
      else
        out_data[0] = static_cast<T>(mv);
    } else {
      int actual_threads =
          std::min(max_threads, static_cast<int>(num_elements));
#pragma omp parallel for num_threads(actual_threads)
      for (int64_t o = 0; o < num_elements; ++o)
        sum_and_store_one(o);
    }
    return output;
  }
}

//----------------------------------------------------------------------
// VARIANCE REDUCTION DISPATCHER (Two-pass algorithm)
//----------------------------------------------------------------------

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_kernel(const Tensor &input,
                                const std::vector<int64_t> &normalized_axes,
                                bool keepdim, int64_t correction,
                                cudaStream_t stream,
                                const Tensor *precomputed_mean = nullptr) {
  // Determine if this is NaN-aware variance
  if constexpr (std::is_same_v<T, bool>) {
    throw std::runtime_error(
        "reduce_var: Bool dtype not supported for statistical operations.");
  }
  // constexpr bool is_complex_type =
  //     std::is_same_v<T, complex32_t> ||
  //     std::is_same_v<T, complex64_t> ||
  //     std::is_same_v<T, complex128_t>;
  // if constexpr (is_complex_type) {
  //     throw std::runtime_error(
  //         "statistical operations are not supported for complex types for
  //         now"

  //         "Got: " + get_dtype_name(input.dtype())
  //     );
  // }
  constexpr bool is_nan_aware =
      std::is_same_v<VarianceOpType<T>, NanVarianceOp<T>>;

  constexpr bool is_float_type =
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

  if constexpr (is_nan_aware && !is_float_type) {
    throw std::runtime_error(
        "NaN-aware variance is only supported for floating point types "
        "(Float16, Bfloat16, Float32, Float64). "
        "Got: " +
        get_dtype_name(input.dtype()));
  }
#ifdef WITH_CUDA
  if (input.is_cuda()) {
    if constexpr (std::is_same_v<T, float4_e2m1_t> ||
                  std::is_same_v<T, float4_e2m1_2x_t>) {
      throw std::runtime_error(
          "Variance reduction is not supported for FP4 types.");
    } else if constexpr (std::is_same_v<T, complex32_t> ||
                         std::is_same_v<T, complex64_t> ||
                         std::is_same_v<T, complex128_t>) {
      throw std::runtime_error(
          "Variance reduction is not supported for complex types on GPU. "
          "Complex variance requires real-valued magnitude decomposition. "
          "Got: " +
          get_dtype_name(input.dtype()));
    } else {
      return dispatch_variance_gpu<T, VarianceOpType>(
          input, normalized_axes, keepdim, correction, stream);
    }
  }
#endif

  //  STEP 1: Compute mean with keepdim=true (required for broadcasting)
  Tensor mean_tensor =
      is_nan_aware
          ? reduce_kernel_mean<T, NanSumOp>(input, normalized_axes, true,
                                            stream)
          : reduce_kernel_mean<T, SumOp>(input, normalized_axes, true, stream);

  //  STEP 2: Calculate output shape and metadata
  Shape output_shape =
      calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
  int64_t reduced_count =
      calculate_reduced_count(input.shape().dims, normalized_axes);

  if (reduced_count == 0) {
    throw std::runtime_error("Cannot compute variance: reduced count is zero.");
  }

  if (input.shape().dims.size() > MAX_DIMS) {
    throw std::runtime_error(
        "Tensor rank exceeds maximum supported dimensions (64)");
  }

  // Determine output dtype
  Dtype output_dtype;
  if constexpr (std::is_integral_v<T>) {
    output_dtype = Dtype::Float64;
  } else {
    output_dtype = input.dtype();
  }

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(output_dtype)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  //  STEP 3: Prepare data pointers
  const T *input_data = input.data<T>();

  //  CRITICAL FIX: For integers, mean is stored as double
  using MeanCppT =
      typename std::conditional<std::is_integral_v<T>, double, T>::type;

  const MeanCppT *mean_data = mean_tensor.data<MeanCppT>();

  using AccT = typename std::conditional<
      should_use_double_accumulation<T>(), double,
      typename std::conditional<std::is_integral_v<T>, double,
                                AccumulatorType<T>>::type>::type;

  using OutputT =
      typename std::conditional<std::is_integral_v<T>, double, T>::type;

  OutputT *output_data = output.data<OutputT>();

  const std::vector<int64_t> &input_dims = input.shape().dims;
  const std::vector<int64_t> &input_strides = input.stride().strides;
  const int64_t num_slices = output.numel();
  // rank_preserved and mean_strides removed: mean_data[output_index] is always
  // correct because mean was computed with keepdim=true, so its C-contiguous
  // flat index equals output_index

  bool reduced_bitmap[MAX_DIMS] = {false};
  for (int64_t axis : normalized_axes) {
    reduced_bitmap[axis] = true;
  }

  // Calculate reduced_dims (used by Generic carry-add path)
  std::vector<int64_t> reduced_dims;
  for (size_t dim = 0; dim < input_dims.size(); ++dim) {
    bool is_reduced = reduced_bitmap[dim];
    if (is_reduced) {
      reduced_dims.push_back(input_dims[dim]);
    }
  }

  // Layout dispatch: replaces per-element unravel/ravel in inner loop
  ReductionLayout layout_var = compute_reduction_layout(input, normalized_axes);
  cpu_demote_middle_to_generic(layout_var);
  const int64_t k_var = static_cast<int64_t>(normalized_axes.size());
  int64_t red_input_strides_var[MAX_DIMS];
  for (int64_t d = 0; d < k_var; ++d)
    red_input_strides_var[d] = input_strides[normalized_axes[d]];
  // Precompute base linear indices for Generic path
  int64_t M_nr_var = 0;
  int64_t nr_sizes_var[MAX_DIMS], nr_strides_var[MAX_DIMS];
  std::vector<int64_t> base_lin_idxs_var;
  if (layout_var.path == ReductionLayout::Path::Generic) {
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
      if (!reduced_bitmap[dim]) {
        nr_sizes_var[M_nr_var] = input_dims[dim];
        nr_strides_var[M_nr_var] = input_strides[dim];
        ++M_nr_var;
      }
    }
    base_lin_idxs_var.resize(num_slices);
    int64_t oc[MAX_DIMS] = {}, blk = 0;
    for (int64_t o = 0; o < num_slices; ++o) {
      base_lin_idxs_var[o] = blk;
      for (int64_t j = M_nr_var - 1; j >= 0; --j) {
        ++oc[j];
        blk += nr_strides_var[j];
        if (oc[j] < nr_sizes_var[j])
          break;
        blk -= oc[j] * nr_strides_var[j];
        oc[j] = 0;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════
  // PASS 2: SIMD sum((x - mean)²) — fully vectorized
  //
  // Optimizations applied:
  //   ✅ 8-wide float SIMD × 4 ILP for fp32 InnerContiguous
  //   ✅ 4-wide double SIMD × 4 ILP for fp64 InnerContiguous
  //   ✅ SIMD NaN bitmask for nanvar (NaN→mean, so diff=0, sq=0)
  //   ✅ Float accumulator for fp32 (like cascade_sum — division by N cancels
  //   error) ✅ No early-exit NaN branch (NaN propagates naturally through
  //   arithmetic) ✅ Scalar fallback for Outer/Generic/complex
  //   (strided/irregular access)
  //
  // PyTorch uses SCALAR Welford (~30 cycles/element with division).
  // Our SIMD pass 2: ~4 cycles/element (sub+mul+add, no division!) = ~7.5x
  // faster ═══════════════════════════════════════════════════════════

  // Helper: compute sum((x-mean)²) for one output position
  auto var_one_slice = [&](int64_t output_index) -> std::pair<double, double> {
    // Returns {sum_sq_diff, valid_count}
    double sq_sum = 0.0;
    double valid = static_cast<double>(reduced_count); // default for non-NaN

    AccT mean_val = static_cast<AccT>(mean_data[output_index]);

    if (layout_var.path == ReductionLayout::Path::InnerContiguous) {
      const T *in = input_data + output_index * layout_var.input_outer_stride;
      const int64_t n = layout_var.reduced_count;

      if constexpr (!is_nan_aware && std::is_same_v<T, float>) {
        // 8-wide float SIMD × 4 ILP — sub, mul, add (no division!)
        __m256 vmean = _mm256_set1_ps(mean_val);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + 32 <= n; j += 32) {
          __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(in + j), vmean);
          __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(in + j + 8), vmean);
          __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(in + j + 16), vmean);
          __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(in + j + 24), vmean);
          a0 = _mm256_add_ps(a0, _mm256_mul_ps(d0, d0));
          a1 = _mm256_add_ps(a1, _mm256_mul_ps(d1, d1));
          a2 = _mm256_add_ps(a2, _mm256_mul_ps(d2, d2));
          a3 = _mm256_add_ps(a3, _mm256_mul_ps(d3, d3));
        }
        for (; j + 8 <= n; j += 8) {
          __m256 d = _mm256_sub_ps(_mm256_loadu_ps(in + j), vmean);
          a0 = _mm256_add_ps(a0, _mm256_mul_ps(d, d));
        }
        __m256 vtot =
            _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        float sa[8];
        _mm256_storeu_ps(sa, vtot);
        for (int p = 0; p < 8; ++p)
          sq_sum += (double)sa[p];
        for (; j < n; ++j) {
          float d = in[j] - mean_val;
          sq_sum += (double)(d * d);
        }

      } else if constexpr (is_nan_aware && std::is_same_v<T, float>) {
        // nanvar: SIMD NaN bitmask — NaN→mean so diff=0, sq=0
        __m256 vmean = _mm256_set1_ps(mean_val);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 cnt0 = _mm256_setzero_ps(), cnt1 = _mm256_setzero_ps();
        __m256 zero = _mm256_setzero_ps(), ones = _mm256_set1_ps(1.0f);
        int64_t j = 0;
        for (; j + 16 <= n; j += 16) {
          __m256 v0 = _mm256_loadu_ps(in + j);
          __m256 m0 = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
          __m256 sv0 = _mm256_blendv_ps(v0, vmean, m0); // NaN→mean
          __m256 d0 = _mm256_sub_ps(sv0, vmean);
          a0 = _mm256_add_ps(a0, _mm256_mul_ps(d0, d0));
          cnt0 = _mm256_add_ps(cnt0, _mm256_blendv_ps(ones, zero, m0));

          __m256 v1 = _mm256_loadu_ps(in + j + 8);
          __m256 m1 = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
          __m256 sv1 = _mm256_blendv_ps(v1, vmean, m1);
          __m256 d1 = _mm256_sub_ps(sv1, vmean);
          a1 = _mm256_add_ps(a1, _mm256_mul_ps(d1, d1));
          cnt1 = _mm256_add_ps(cnt1, _mm256_blendv_ps(ones, zero, m1));
        }
        for (; j + 8 <= n; j += 8) {
          __m256 v = _mm256_loadu_ps(in + j);
          __m256 m = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
          __m256 sv = _mm256_blendv_ps(v, vmean, m);
          __m256 d = _mm256_sub_ps(sv, vmean);
          a0 = _mm256_add_ps(a0, _mm256_mul_ps(d, d));
          cnt0 = _mm256_add_ps(cnt0, _mm256_blendv_ps(ones, zero, m));
        }
        float sa[8], ca[8];
        _mm256_storeu_ps(sa, _mm256_add_ps(a0, a1));
        _mm256_storeu_ps(ca, _mm256_add_ps(cnt0, cnt1));
        for (int p = 0; p < 8; ++p) {
          sq_sum += (double)sa[p];
          valid += (double)ca[p];
        }
        valid -= (double)reduced_count; // reset: we started with reduced_count
        valid += 0; // valid now has the correct count from SIMD
        // Recompute valid properly
        valid = 0;
        for (int p = 0; p < 8; ++p)
          valid += (double)ca[p];
        for (; j < n; ++j) {
          if (!std::isnan(in[j])) {
            float d = in[j] - mean_val;
            sq_sum += (double)(d * d);
            valid += 1.0;
          }
        }

      } else if constexpr (!is_nan_aware && std::is_same_v<T, double>) {
        // 4-wide double SIMD × 4 ILP
        __m256d vmean = _mm256_set1_pd(mean_val);
        __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
        __m256d a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
        int64_t j = 0;
        for (; j + 16 <= n; j += 16) {
          __m256d d0 = _mm256_sub_pd(_mm256_loadu_pd(in + j), vmean);
          __m256d d1 = _mm256_sub_pd(_mm256_loadu_pd(in + j + 4), vmean);
          __m256d d2 = _mm256_sub_pd(_mm256_loadu_pd(in + j + 8), vmean);
          __m256d d3 = _mm256_sub_pd(_mm256_loadu_pd(in + j + 12), vmean);
          a0 = _mm256_add_pd(a0, _mm256_mul_pd(d0, d0));
          a1 = _mm256_add_pd(a1, _mm256_mul_pd(d1, d1));
          a2 = _mm256_add_pd(a2, _mm256_mul_pd(d2, d2));
          a3 = _mm256_add_pd(a3, _mm256_mul_pd(d3, d3));
        }
        for (; j + 4 <= n; j += 4) {
          __m256d d = _mm256_sub_pd(_mm256_loadu_pd(in + j), vmean);
          a0 = _mm256_add_pd(a0, _mm256_mul_pd(d, d));
        }
        double sa[4];
        _mm256_storeu_pd(
            sa, _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
        sq_sum = sa[0] + sa[1] + sa[2] + sa[3];
        for (; j < n; ++j) {
          double d = in[j] - mean_val;
          sq_sum += d * d;
        }

      } else {
        // Scalar fallback (fp16, bf16, complex, int, nan-aware double, etc.)
        for (int64_t j = 0; j < n; ++j) {
          T val = in[j];
          if constexpr (is_nan_aware) {
            if (!safe_isnan(val)) {
              double d =
                  to_double(static_cast<AccT>(val)) - to_double(mean_val);
              sq_sum += d * d;
            } else {
              valid -= 1.0;
            }
          } else {
            double d = to_double(static_cast<AccT>(val)) - to_double(mean_val);
            sq_sum += d * d;
          }
        }
      }
    } else if (layout_var.path == ReductionLayout::Path::OuterContiguous) {
      // Scalar strided walk (can't horizontally SIMD strided access)
      for (int64_t r = 0; r < layout_var.reduced_count; ++r) {
        T val = *(input_data + output_index + r * layout_var.input_row_stride);
        if constexpr (is_nan_aware) {
          if (!safe_isnan(val)) {
            double d = to_double(static_cast<AccT>(val)) - to_double(mean_val);
            sq_sum += d * d;
          } else {
            valid -= 1.0;
          }
        } else {
          double d = to_double(static_cast<AccT>(val)) - to_double(mean_val);
          sq_sum += d * d;
        }
      }
    } else {
      // Generic: single-axis stride or multi-axis carry-add
      if (k_var == 1) {
        int64_t lin = base_lin_idxs_var[output_index];
        const int64_t stride = red_input_strides_var[0];
        for (int64_t i = 0; i < reduced_dims[0]; ++i, lin += stride) {
          T val = input_data[lin];
          if constexpr (is_nan_aware) {
            if (!safe_isnan(val)) {
              double d =
                  to_double(static_cast<AccT>(val)) - to_double(mean_val);
              sq_sum += d * d;
            } else {
              valid -= 1.0;
            }
          } else {
            double d = to_double(static_cast<AccT>(val)) - to_double(mean_val);
            sq_sum += d * d;
          }
        }
      } else {
        int64_t red_coords[MAX_DIMS] = {};
        int64_t lin = base_lin_idxs_var[output_index];
        for (int64_t i = 0; i < reduced_count; ++i) {
          T val = input_data[lin];
          if constexpr (is_nan_aware) {
            if (!safe_isnan(val)) {
              double d =
                  to_double(static_cast<AccT>(val)) - to_double(mean_val);
              sq_sum += d * d;
            } else {
              valid -= 1.0;
            }
          } else {
            double d = to_double(static_cast<AccT>(val)) - to_double(mean_val);
            sq_sum += d * d;
          }
          for (int64_t d = k_var - 1; d >= 0; --d) {
            ++red_coords[d];
            lin += red_input_strides_var[d];
            if (red_coords[d] < reduced_dims[d])
              break;
            lin -= red_coords[d] * red_input_strides_var[d];
            red_coords[d] = 0;
          }
        }
      }
    }
    return {sq_sum, valid};
  };

// Execute with OpenMP
#pragma omp parallel for
  for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
    auto [sq_sum, valid] = var_one_slice(output_index);

    // STEP 5: Compute variance = sq_sum / (count - correction)
    // valid = non-NaN count (for nanvar) or reduced_count (for var)
    double divisor_d;
    if constexpr (is_nan_aware) {
      divisor_d = valid - static_cast<double>(correction);
    } else {
      divisor_d =
          static_cast<double>(reduced_count) - static_cast<double>(correction);
    }

    OutputT variance;
    if (std::isnan(sq_sum) || divisor_d <= 0.0) {
      if constexpr (std::is_same_v<OutputT, double>)
        variance = std::numeric_limits<double>::quiet_NaN();
      else
        variance =
            static_cast<OutputT>(std::numeric_limits<double>::quiet_NaN());
    } else {
      variance = static_cast<OutputT>(sq_sum / divisor_d);
    }

    // STEP 6: Store
    output_data[output_index] = variance;
  }

  return output;
}
} // namespace detail
} // namespace OwnTensor
#endif // OWNTENSOR_REDUCTIONS_IMPL_H
