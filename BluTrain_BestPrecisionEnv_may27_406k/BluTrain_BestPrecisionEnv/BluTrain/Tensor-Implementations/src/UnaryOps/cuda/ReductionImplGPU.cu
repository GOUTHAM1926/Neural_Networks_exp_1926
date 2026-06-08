// src/UnaryOps/ReductionImplGPU.cu - Unified GPU reduction dispatcher
// ReductionKernels.cuh now contains: GpuReduceConfig + OffsetCalculator +
// unified_reduce_kernel
#include "core/Tensor.h"
#include "ops/helpers/ReductionKernels.cuh"
#include "ops/helpers/ReductionUtils.h"
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <memory>
#include <string>
#include <typeinfo>
//  CRITICAL: Include both custom structs AND native CUDA types
#include "device/CachingCudaAllocator.h"
#include "dtype/CudaTraits.h"
#include "dtype/Types.h" // Custom structs (float16_t, bfloat16_t)
#include "dtype/fp4.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h> // Native CUDA types (__half, __nv_bfloat16)
#include <vector>
namespace OwnTensor {
namespace detail {

#ifdef WITH_CUDA

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION TRAITS (Custom Struct → Native CUDA Type)
// ═══════════════════════════════════════════════════════════
// Traits are now defined in dtype/CudaTraits.h

// =================================================================
// GPU DEVICE MEMORY HELPER
// =================================================================

// ═══════════════════════════════════════════════════════════
// UNIFIED LAUNCHER HELPER
// Gets device props (cached after first call) for config solver.
// ═══════════════════════════════════════════════════════════
static void get_device_props(int &num_mp, int &max_tpm) {
  static int cached_num_mp = -1, cached_max_tpm = -1;
  if (cached_num_mp < 0) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    cached_num_mp = props.multiProcessorCount;
    cached_max_tpm = props.maxThreadsPerMultiProcessor;
  }
  num_mp = cached_num_mp;
  max_tpm = cached_max_tpm;
}

// ═══════════════════════════════════════════════════════════
// GENERIC-PATH SETUP (bug-#3 fix)
// For non-consecutive / non-contiguous reductions the layout is Generic but
// carries no offset info. Fill in num_outputs/reduced_count and build the two
// offset calculators the kernel needs: output_calc walks the NON-reduced dims
// (-> base offset per output), input_calc walks the REDUCED dims (-> offset of
// each reduction step). Both use the input's real strides, so out_base+in_off
// addresses the exact element (matches the CPU base_lin_idxs path). No-op for
// Tier 1/2.
// ═══════════════════════════════════════════════════════════
static void
setup_generic_path(detail::ReductionLayout &layout, const Tensor &input,
                   const std::vector<int64_t> &axes,
                   detail::OffsetCalculator<1, uint32_t> &input_calc,
                   detail::OffsetCalculator<1, uint32_t> &output_calc) {
  if (layout.path != detail::ReductionLayout::Path::Generic)
    return;
  const auto &dims = input.shape().dims;
  const auto &strides = input.stride().strides;
  const int nd = static_cast<int>(dims.size());
  bool is_reduced[64] = {false};
  for (int64_t a : axes)
    is_reduced[a] = true;
  std::vector<int64_t> r_sz, r_st, o_sz, o_st;
  int64_t reduced = 1, outputs = 1;
  for (int d = 0; d < nd; ++d) {
    if (is_reduced[d]) {
      r_sz.push_back(dims[d]);
      r_st.push_back(strides[d]);
      reduced *= dims[d];
    } else {
      o_sz.push_back(dims[d]);
      o_st.push_back(strides[d]);
      outputs *= dims[d];
    }
  }
  layout.num_outputs = outputs;
  layout.reduced_count = reduced;
  const int64_t *rsp = r_st.data();
  const int64_t *osp = o_st.data();
  input_calc = detail::OffsetCalculator<1, uint32_t>(
      static_cast<int>(r_sz.size()), r_sz.data(), &rsp);
  output_calc = detail::OffsetCalculator<1, uint32_t>(
      static_cast<int>(o_sz.size()), o_sz.data(), &osp);
}

// ═══════════════════════════════════════════════════════════
// GPU VALUE REDUCTION DISPATCHER
// All 10 scalar ops: sum, product, min, max, nansum, nanproduct,
// nanmin, nanmax, all, any — all route through unified_reduce_kernel.
// ═══════════════════════════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────
// 32-bit indexing guard.
// The GPU reduction kernels index memory with uint32_t and keep the per-launch
// output/reduced counts in `int`. They therefore only support tensors up to:
//   * total elements (numel)         <= 2^32 - 1   (uint32 flat index)
//   * num_outputs AND reduced_count  <= 2^31 - 1   (int config counts)
// Bigger tensors (e.g. attention scores [B,H,T,T] or logits [B,T,V] at large
// model/sequence sizes) would silently overflow these and corrupt results.
// Fail loudly instead, until the 64-bit / split-indexing path exists.
// TODO: implement PyTorch-style output-dim split (64-bit base ptr + 32-bit
// local index) for >2^31 reductions — see
// work_docs/data_accessing_techniques.md.
static inline void ensure_32bit_reduction_indexing(const Tensor &input,
                                                   int64_t reduced_count) {
  constexpr int64_t I32MAX = 2147483647LL; // 2^31 - 1
  constexpr int64_t U32MAX = 4294967295LL; // 2^32 - 1
  const int64_t numel = input.numel();
  const int64_t num_outputs =
      (reduced_count > 0) ? numel / reduced_count : numel;
  if (numel > U32MAX || reduced_count > I32MAX || num_outputs > I32MAX) {
    throw std::runtime_error(
        "GPU reduction: tensor exceeds 32-bit indexing limits (numel > 2^32, "
        "or "
        "output/reduced count > 2^31). The 64-bit / split-indexing reduction "
        "path is not yet implemented.");
  }
}

template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor &input,
                              const std::vector<int64_t> &normalized_axes,
                              bool keepdim, cudaStream_t stream) {
  // ── 1. Layout analysis (replaces all per-element device metadata) ──
  auto layout = compute_reduction_layout(input, normalized_axes);

  // ── 2. Output shape + dtype ──
  Shape output_shape =
      calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
  constexpr bool is_compare_op =
      std::is_same_v<OpType<T>, detail::MinOp<T>> ||
      std::is_same_v<OpType<T>, detail::MaxOp<T>> ||
      std::is_same_v<OpType<T>, detail::NanMinOp<T>> ||
      std::is_same_v<OpType<T>, detail::NanMaxOp<T>>;
  Dtype output_dtype;
  if constexpr (std::is_same_v<T, bool>)
    output_dtype = Dtype::Bool;
  else if constexpr (std::is_integral_v<T> && !is_compare_op)
    output_dtype = Dtype::Int64;
  else
    output_dtype = input.dtype();

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(output_dtype)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  const int64_t reduced_count =
      calculate_reduced_count(input.shape().dims, normalized_axes);
  if (reduced_count == 0)
    throw std::runtime_error("GPU Reduction: reduced count is zero");
  ensure_32bit_reduction_indexing(input, reduced_count);

  // ── 3. Type conversion ──
  using CudaT = CudaNativeType<T>;
  using ops_t = OpType<CudaT>;
  ops_t ops{};
  using arg_t = decltype(ops.identity());
  using OutputCppT =
      std::conditional_t<std::is_integral_v<T> && !is_compare_op, int64_t, T>;
  using OutputCudaT = CudaNativeType<OutputCppT>;

  // ── 4. Config solver ──
  int num_mp, max_tpm;
  get_device_props(num_mp, max_tpm);
  // ── 5. OffsetCalculator: empty for Tier 1/2; populated for Generic (bug #3)
  // ──
  detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
  setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
  auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
  int64_t step_stride =
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.inner_count
          : 1;

  // ── 6. Shared memory = warps × sizeof(arg_t), min 1 warp ──
  int smem = std::max(config.shared_memory_size(),
                      (config.block_width * config.block_height / 32 + 1) *
                          (int)sizeof(arg_t));

  // ── 7. Launch via packed struct + compile-time Path dispatch ──
  const CudaT *src = reinterpret_cast<const CudaT *>(input.data<T>());
  OutputCudaT *dst = reinterpret_cast<OutputCudaT *>(output.data<OutputCppT>());
  cuda::launch_reduce_kernel<CudaT, OutputCudaT, ops_t>(
      ops, config, input_calc, output_calc, src, dst, layout.path, step_stride,
      smem, stream,
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.input_outer_stride
          : 0);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA reduce_kernel failed: ") +
                             cudaGetErrorString(err));
  return output;
}

// ═══════════════════════════════════════════════════════════
// GPU INDEX REDUCTION DISPATCHER
// ArgMin, ArgMax, NanArgMin, NanArgMax — output is always int64.
// project() in the functor extracts .index from ValueIndex<T>.
// ═══════════════════════════════════════════════════════════
template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor &input,
                                    const std::vector<int64_t> &normalized_axes,
                                    bool keepdim, cudaStream_t stream) {
  auto layout = compute_reduction_layout(input, normalized_axes);
  Shape output_shape =
      calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(Dtype::Int64)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  const int64_t reduced_count =
      calculate_reduced_count(input.shape().dims, normalized_axes);
  if (reduced_count == 0)
    throw std::runtime_error("GPU Index Reduction: reduced count is zero");
  ensure_32bit_reduction_indexing(input, reduced_count);

  using CudaT = CudaNativeType<T>;
  using ops_t = OpType<CudaT>;
  ops_t ops{};
  using arg_t = decltype(ops.identity()); // = ValueIndex<CudaT>

  int num_mp, max_tpm;
  get_device_props(num_mp, max_tpm);
  detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
  setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
  auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
  int64_t step_stride =
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.inner_count
          : 1;
  int smem = std::max(config.shared_memory_size(),
                      (config.block_width * config.block_height / 32 + 1) *
                          (int)sizeof(arg_t));

  const CudaT *src = reinterpret_cast<const CudaT *>(input.data<T>());
  int64_t *dst = output.data<int64_t>();
  cuda::launch_reduce_kernel<CudaT, int64_t, ops_t>(
      ops, config, input_calc, output_calc, src, dst, layout.path, step_stride,
      smem, stream,
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.input_outer_stride
          : 0);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA index_kernel failed: ") +
                             cudaGetErrorString(err));
  return output;
}

// ═══════════════════════════════════════════════════════════
// GPU MEAN REDUCTION DISPATCHER
// Uses MeanOps functor: sum then project(a) = a * (1/count).
// NaN-aware path uses NanSumOp which skips NaN in accumulate step.
// ═══════════════════════════════════════════════════════════
template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor &input,
                         const std::vector<int64_t> &normalized_axes,
                         bool keepdim, cudaStream_t stream) {
  auto layout = compute_reduction_layout(input, normalized_axes);
  Shape output_shape =
      calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

  // Mean output is always float32 for integer inputs, or same as input for
  // float
  Dtype output_dtype = (std::is_integral_v<T>) ? Dtype::Float32 : input.dtype();
  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(output_dtype)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  using CudaT = CudaNativeType<T>;
  using OutT = std::conditional_t<std::is_integral_v<T>, float, T>;
  using OutCudaT = CudaNativeType<OutT>;
  using AccT = detail::AccumulatorType<CudaT>;
  const int64_t reduced_count =
      calculate_reduced_count(input.shape().dims, normalized_axes);
  ensure_32bit_reduction_indexing(input, reduced_count);
  double inv_count = 1.0 / static_cast<double>(reduced_count);
  AccT factor;
  if constexpr (std::is_same_v<CudaT, complex32_t> ||
                std::is_same_v<CudaT, complex64_t> ||
                std::is_same_v<CudaT, complex128_t>) {
    factor = AccT(inv_count, 0.0);
  } else {
    factor = static_cast<AccT>(inv_count);
  }
  using ops_t = detail::MeanOps<CudaT>;
  ops_t ops{factor};

  int num_mp, max_tpm;
  get_device_props(num_mp, max_tpm);
  detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
  setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
  auto config = detail::build_reduce_config<CudaT>(layout, num_mp, max_tpm);
  int64_t step_stride =
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.inner_count
          : 1;
  int smem = std::max(config.shared_memory_size(),
                      (config.block_width * config.block_height / 32 + 1) *
                          (int)sizeof(CudaT));

  const CudaT *src = reinterpret_cast<const CudaT *>(input.data<T>());
  OutCudaT *dst = reinterpret_cast<OutCudaT *>(output.data<OutT>());
  cuda::launch_reduce_kernel<CudaT, OutCudaT, ops_t>(
      ops, config, input_calc, output_calc, src, dst, layout.path, step_stride,
      smem, stream,
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.input_outer_stride
          : 0);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA mean kernel failed: ") +
                             cudaGetErrorString(err));
  return output;
}

// ═══════════════════════════════════════════════════════════
// GPU VARIANCE REDUCTION DISPATCHER (COMPLETE IMPLEMENTATION)
// ═══════════════════════════════════════════════════════════

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor &input,
                             const std::vector<int64_t> &normalized_axes,
                             bool keepdim, int64_t correction,
                             cudaStream_t stream) {
  // Single-pass Welford: 1x HBM read vs old 2-pass (2x reads).
  auto layout = compute_reduction_layout(input, normalized_axes);
  Shape output_shape =
      calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
  const int64_t reduced_count =
      calculate_reduced_count(input.shape().dims, normalized_axes);
  if (reduced_count == 0)
    throw std::runtime_error("GPU Variance: reduced count is zero");
  ensure_32bit_reduction_indexing(input, reduced_count);

  Dtype output_dtype = std::is_integral_v<T> ? Dtype::Float32 : input.dtype();
  Tensor output({output_shape}, TensorOptions()
                                    .with_dtype(output_dtype)
                                    .with_device(input.device())
                                    .with_req_grad(input.requires_grad()));

  using CudaT = CudaNativeType<T>;
  using OutputCppT = std::conditional_t<std::is_integral_v<T>, float, T>;
  using OutputCudaT = CudaNativeType<OutputCppT>;
  using AccScalar = detail::AccumulatorType<CudaT>;
  using ops_t = detail::WelfordOps<CudaT>;
  AccScalar corr_val;
  if constexpr (std::is_same_v<CudaT, complex32_t> ||
                std::is_same_v<CudaT, complex64_t> ||
                std::is_same_v<CudaT, complex128_t>) {
    corr_val = AccScalar(static_cast<double>(correction), 0.0);
  } else {
    corr_val = static_cast<AccScalar>(static_cast<double>(correction));
  }
  ops_t ops{corr_val, /*take_sqrt=*/false};
  using arg_t = decltype(ops.identity()); // = WelfordData<AccScalar>

  int num_mp, max_tpm;
  get_device_props(num_mp, max_tpm);
  detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
  setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
  auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
  int64_t step_stride =
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.inner_count
          : 1;
  int smem = std::max(config.shared_memory_size(),
                      (config.block_width * config.block_height / 32 + 1) *
                          (int)sizeof(arg_t));

  const CudaT *src = reinterpret_cast<const CudaT *>(input.data<T>());
  OutputCudaT *dst = reinterpret_cast<OutputCudaT *>(output.data<OutputCppT>());
  cuda::launch_reduce_kernel<CudaT, OutputCudaT, ops_t>(
      ops, config, input_calc, output_calc, src, dst, layout.path, step_stride,
      smem, stream,
      (layout.path == detail::ReductionLayout::Path::OuterContiguous)
          ? layout.input_outer_stride
          : 0);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA variance_kernel failed: ") +
                             cudaGetErrorString(err));
  return output;
}

// =================================================================
//  EXPLICIT TEMPLATE INSTANTIATIONS - Using Custom Structs
// =================================================================
// ===========================================================
// UNSIGNED INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================
// uint8_t (unsigned char) - Basic operations only
template Tensor dispatch_reduction_gpu<uint8_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint8_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint8_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<uint8_t, SumOp>(const Tensor &,
                                                  const std::vector<int64_t> &,
                                                  bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<uint8_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<uint8_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// uint16_t (unsigned short) - Basic operations only
template Tensor dispatch_reduction_gpu<uint16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint16_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint16_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<uint16_t, SumOp>(const Tensor &,
                                                   const std::vector<int64_t> &,
                                                   bool,
                                                   cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<uint16_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<uint16_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// uint32_t (unsigned int) - Basic operations only
template Tensor dispatch_reduction_gpu<uint32_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint32_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint32_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<uint32_t, SumOp>(const Tensor &,
                                                   const std::vector<int64_t> &,
                                                   bool,
                                                   cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<uint32_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<uint32_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// uint64_t (unsigned long long) - Basic operations only
template Tensor dispatch_reduction_gpu<uint64_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint64_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<uint64_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<uint64_t, SumOp>(const Tensor &,
                                                   const std::vector<int64_t> &,
                                                   bool,
                                                   cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<uint64_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<uint64_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// ===========================================================
// INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================

// int8_t (signed char) - Basic operations only
template Tensor dispatch_reduction_gpu<int8_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<int8_t, SumOp>(const Tensor &,
                                                 const std::vector<int64_t> &,
                                                 bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<int8_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<int8_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// int16_t (short) - Basic operations only
template Tensor dispatch_reduction_gpu<int16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<int16_t, SumOp>(const Tensor &,
                                                  const std::vector<int64_t> &,
                                                  bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<int16_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<int16_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// int32_t (int) - Basic operations only
template Tensor dispatch_reduction_gpu<int32_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<int32_t, SumOp>(const Tensor &,
                                                  const std::vector<int64_t> &,
                                                  bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<int32_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<int32_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// int64_t (long) - Basic operations only
template Tensor dispatch_reduction_gpu<int64_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<int64_t, SumOp>(const Tensor &,
                                                  const std::vector<int64_t> &,
                                                  bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<int64_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<int64_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// ===========================================================
//  FLOATING POINT - Using CUSTOM STRUCTS (NOT __half/__nv_bfloat16)
// ===========================================================

// float16_t (custom struct)
template Tensor dispatch_reduction_gpu<float16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<float16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<float16_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<float16_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<float16_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
// bfloat16_t (custom struct)
template Tensor dispatch_reduction_gpu<bfloat16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// float - All operations
template Tensor dispatch_reduction_gpu<float, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<float, SumOp>(const Tensor &,
                                                const std::vector<int64_t> &,
                                                bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_mean_gpu<float, NanSumOp>(const Tensor &,
                                                   const std::vector<int64_t> &,
                                                   bool,
                                                   cudaStream_t); // ✨✨✨
template Tensor dispatch_variance_gpu<float, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<float, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
// double - All operations
template Tensor
dispatch_reduction_gpu<double, SumOp>(const Tensor &,
                                      const std::vector<int64_t> &, bool,
                                      cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, ProductOp>(const Tensor &,
                                          const std::vector<int64_t> &, bool,
                                          cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, MinOp>(const Tensor &,
                                      const std::vector<int64_t> &, bool,
                                      cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, MaxOp>(const Tensor &,
                                      const std::vector<int64_t> &, bool,
                                      cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, NanSumOp>(const Tensor &,
                                         const std::vector<int64_t> &, bool,
                                         cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, NanProductOp>(const Tensor &,
                                             const std::vector<int64_t> &, bool,
                                             cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, NanMinOp>(const Tensor &,
                                         const std::vector<int64_t> &, bool,
                                         cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_reduction_gpu<double, NanMaxOp>(const Tensor &,
                                         const std::vector<int64_t> &, bool,
                                         cudaStream_t); // ✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool,
    cudaStream_t); // ✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool,
    cudaStream_t); // ✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool,
    cudaStream_t); // ✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool,
    cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_mean_gpu<double, SumOp>(const Tensor &, const std::vector<int64_t> &,
                                 bool, cudaStream_t); // ✨✨✨//✨✨✨
template Tensor
dispatch_mean_gpu<double, NanSumOp>(const Tensor &,
                                    const std::vector<int64_t> &, bool,
                                    cudaStream_t); // ✨✨✨//✨✨✨
template Tensor dispatch_variance_gpu<double, VarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨
template Tensor dispatch_variance_gpu<double, NanVarianceOp>(
    const Tensor &input, const std::vector<int64_t> &axes, bool keepdim,
    int64_t correction, cudaStream_t stream); // ✨✨✨

// Boolean type - Basic operations only
template Tensor dispatch_mean_gpu<bool, SumOp>(const Tensor &,
                                               const std::vector<int64_t> &,
                                               bool, cudaStream_t); // ✨✨✨
// template Tensor dispatch_variance_gpu<bool, VarianceOp>(const Tensor& input,
// const std::vector<int64_t>& axes, bool keepdim , int64_t correction,
// cudaStream_t stream); //✨✨✨ template Tensor dispatch_variance_gpu<bool,
// NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool
// keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_reduction_gpu<bool, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bool, MinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bool, MaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_reduction_gpu<bool, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMinOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMaxOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t); // ✨✨✨
// Add to the Bool section at the end of the file

// Boolean-specific reductions
template Tensor dispatch_reduction_gpu<bool, AllOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<bool, AnyOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
#endif // WITH_CUDA

// ===========================================================
// COMPLEX TYPES - Explicit Instantiations
// ===========================================================

#ifdef WITH_CUDA
// complex32_t
template Tensor dispatch_reduction_gpu<complex32_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, MinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex32_t, MaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, NanMinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex32_t, NanMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex32_t, ArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex32_t, ArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex32_t, NanArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex32_t, NanArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex32_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex32_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex32_t, VarianceOp>(const Tensor&
// input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction,
// cudaStream_t stream); template Tensor dispatch_variance_gpu<complex32_t,
// NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool
// keepdim , int64_t correction, cudaStream_t stream);

// complex64_t
template Tensor dispatch_reduction_gpu<complex64_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, MinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex64_t, MaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, NanMinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex64_t, NanMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex64_t, ArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex64_t, ArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex64_t, NanArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex64_t, NanArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex64_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex64_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex64_t, VarianceOp>(const Tensor&
// input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction,
// cudaStream_t stream); template Tensor dispatch_variance_gpu<complex64_t,
// NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool
// keepdim , int64_t correction, cudaStream_t stream);

// complex128_t
template Tensor dispatch_reduction_gpu<complex128_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, ProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, MinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex128_t, MaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, NanProductOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, NanMinOp>(const Tensor&,
// const std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_reduction_gpu<complex128_t, NanMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex128_t, ArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex128_t, ArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex128_t, NanArgMinOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t); template Tensor
// dispatch_index_reduction_gpu<complex128_t, NanArgMaxOp>(const Tensor&, const
// std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex128_t, SumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex128_t, NanSumOp>(
    const Tensor &, const std::vector<int64_t> &, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex128_t, VarianceOp>(const Tensor&
// input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction,
// cudaStream_t stream); template Tensor dispatch_variance_gpu<complex128_t,
// NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool
// keepdim , int64_t correction, cudaStream_t stream);
#endif

} // namespace detail
} // namespace OwnTensor