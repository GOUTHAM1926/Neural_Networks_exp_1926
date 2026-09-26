// src/UnaryOps/ReductionImplGPU.cu - Unified GPU reduction dispatcher
// ReductionKernels.cuh now contains: GpuReduceConfig + OffsetCalculator + unified_reduce_kernel
#include "ops/helpers/ReductionKernels.cuh"
#include "ops/helpers/ReductionUtils.h"
#include "core/Tensor.h"
#include <cuda_runtime.h>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <string>
//  CRITICAL: Include both custom structs AND native CUDA types
#include "dtype/Types.h"        // Custom structs (float16_t, bfloat16_t)
#include "dtype/fp4.h"
#include <cuda_fp16.h>          // Native CUDA types (__half, __nv_bfloat16)
#include <cuda_bf16.h>
#include "dtype/CudaTraits.h"
#include "device/CachingCudaAllocator.h"
#include <vector>
#include "device/CudaKernelScope.h"
#include "profiler/RecordFunction.h"
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
static void get_device_props(int& num_mp, int& max_tpm) {
    PROFILER_SCOPE("get_device_props");
    static int cached_num_mp = -1, cached_max_tpm = -1;
    if (cached_num_mp < 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        cached_num_mp  = props.multiProcessorCount;
        cached_max_tpm = props.maxThreadsPerMultiProcessor;
    }
    num_mp = cached_num_mp; max_tpm = cached_max_tpm;
}

// Modified: new helper. Generic layouts (reduced axes that are not consecutive, e.g. {0, 2}, or a
// non-contiguous input such as a transposed view) arrive from compute_reduction_layout with only
// path = Generic: num_outputs = reduced_count = 0 and no offset information. The dispatchers used
// to launch them with EMPTY offset calculators, and build_reduce_config clamps both counts to 1, so
// the kernel computed a single output from src[0 .. ] -- wrong for every such reduction.
// This fills in the counts and builds the two calculators the Generic kernel path already uses
// (out_base = output_calc.get(output_idx), in_off = input_calc.get(i), value = src[out_base + in_off]):
//   output_calc walks the NON-reduced dims in order -> the output tensor's C order;
//   input_calc  walks the REDUCED dims             -> one reduction step each.
// Both use the input's real strides, so any view is addressed correctly. No-op for Tier 1/2.
// OffsetCalculator holds at most MAX_DIMS dims per side and silently ignores the rest, so more
// dims than that is refused here instead of computed wrongly.
static void setup_generic_path(detail::ReductionLayout& layout, const Tensor& input,
                               const std::vector<int64_t>& axes,
                               detail::OffsetCalculator<1, uint32_t>& input_calc,
                               detail::OffsetCalculator<1, uint32_t>& output_calc) {
    if (layout.path != detail::ReductionLayout::Path::Generic) return;
    const auto& dims    = input.shape().dims;
    const auto& strides = input.stride().strides;
    const int nd = static_cast<int>(dims.size());
    bool is_reduced[64] = {false};
    for (int64_t a : axes) is_reduced[a] = true;

    std::vector<int64_t> r_sz, r_st, o_sz, o_st;
    int64_t reduced = 1, outputs = 1;
    for (int d = 0; d < nd; ++d) {
        if (is_reduced[d]) { r_sz.push_back(dims[d]); r_st.push_back(strides[d]); reduced *= dims[d]; }
        else               { o_sz.push_back(dims[d]); o_st.push_back(strides[d]); outputs *= dims[d]; }
    }
    constexpr int kMaxDims = detail::OffsetCalculator<1, uint32_t>::MAX_DIMS;
    if (static_cast<int>(r_sz.size()) > kMaxDims || static_cast<int>(o_sz.size()) > kMaxDims)
        throw std::runtime_error("GPU reduction: a Generic-path reduction supports at most " +
                                 std::to_string(kMaxDims) + " reduced and " + std::to_string(kMaxDims) +
                                 " kept dims (got " + std::to_string(r_sz.size()) + " and " +
                                 std::to_string(o_sz.size()) + ")");
    layout.num_outputs   = outputs;
    layout.reduced_count = reduced;
    const int64_t* rsp = r_st.data();
    const int64_t* osp = o_st.data();
    input_calc  = detail::OffsetCalculator<1, uint32_t>(static_cast<int>(r_sz.size()), r_sz.data(), &rsp);
    output_calc = detail::OffsetCalculator<1, uint32_t>(static_cast<int>(o_sz.size()), o_sz.data(), &osp);
}

// ═══════════════════════════════════════════════════════════
// GPU VALUE REDUCTION DISPATCHER
// All 10 scalar ops: sum, product, min, max, nansum, nanproduct,
// nanmin, nanmax, all, any — all route through unified_reduce_kernel.
// ═══════════════════════════════════════════════════════════
template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input,
                               const std::vector<int64_t>& normalized_axes,
                               bool keepdim, cudaStream_t stream)
{
    CUDA_KERNEL_SCOPE("dispatch_reduction_gpu", stream);
    // ── 1. Layout analysis (replaces all per-element device metadata) ──
    auto layout = compute_reduction_layout(input, normalized_axes);

    // ── 2. Output shape + dtype ──
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    constexpr bool is_compare_op =
        std::is_same_v<OpType<T>, detail::MinOp<T>>    ||
        std::is_same_v<OpType<T>, detail::MaxOp<T>>    ||
        std::is_same_v<OpType<T>, detail::NanMinOp<T>> ||
        std::is_same_v<OpType<T>, detail::NanMaxOp<T>>;
    Dtype output_dtype;
    if constexpr (std::is_same_v<T, bool>)                      output_dtype = Dtype::Bool;
    else if constexpr (std::is_integral_v<T> && !is_compare_op) output_dtype = Dtype::Int64;
    else                                                          output_dtype = input.dtype();

    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype).with_device(input.device()).with_req_grad(input.requires_grad()));

    const int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    if (reduced_count == 0) throw std::runtime_error("GPU Reduction: reduced count is zero");



    // ── 3. Type conversion ──
    using CudaT = CudaNativeType<T>;
    using ops_t  = OpType<CudaT>;
    ops_t ops{};
    using arg_t      = decltype(ops.identity());
    using OutputCppT = std::conditional_t<std::is_integral_v<T> && !is_compare_op, int64_t, T>;
    using OutputCudaT = CudaNativeType<OutputCppT>;

    // ── 4. Config solver ──
    int num_mp, max_tpm; get_device_props(num_mp, max_tpm);
    // ── 5. OffsetCalculator (empty for Tier 1/2; built for Generic) ──
    // Modified: built BEFORE the config, because setup_generic_path also sets the Generic
    // layout's num_outputs / reduced_count, which the config is sized from.
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous)
                          ? layout.inner_count : 1;

    // ── 6. Shared memory = warps × sizeof(arg_t), min 1 warp ──
    int smem = std::max(config.shared_memory_size(),
                        (config.block_width * config.block_height / 32 + 1) * (int)sizeof(arg_t));

    // ── 7. Launch via packed struct + compile-time Path dispatch ──
    const CudaT*   src = reinterpret_cast<const CudaT*>(input.data<T>());
    OutputCudaT*   dst = reinterpret_cast<OutputCudaT*>(output.data<OutputCppT>());
    cuda::launch_reduce_kernel<CudaT, OutputCudaT, ops_t>(
        ops, config, input_calc, output_calc, src, dst, layout.path, step_stride, smem, stream,
        (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.input_outer_stride : 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA reduce_kernel failed: ") + cudaGetErrorString(err));
    return output;
}

// ═══════════════════════════════════════════════════════════
// GPU INDEX REDUCTION DISPATCHER
// ArgMin, ArgMax, NanArgMin, NanArgMax — output is always int64.
// project() in the functor extracts .index from ValueIndex<T>.
// ═══════════════════════════════════════════════════════════
template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input,
                                     const std::vector<int64_t>& normalized_axes,
                                     bool keepdim, cudaStream_t stream)
{
    CUDA_KERNEL_SCOPE("dispatch_index_reduction_gpu", stream);
    auto layout = compute_reduction_layout(input, normalized_axes);
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

    Tensor output({output_shape}, TensorOptions()
        .with_dtype(Dtype::Int64).with_device(input.device()).with_req_grad(input.requires_grad()));

    const int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    if (reduced_count == 0) throw std::runtime_error("GPU Index Reduction: reduced count is zero");


    using CudaT = CudaNativeType<T>;
    using ops_t  = OpType<CudaT>;
    ops_t ops{};
    using arg_t = decltype(ops.identity()); // = ValueIndex<CudaT>

    int num_mp, max_tpm; get_device_props(num_mp, max_tpm);
    // Modified: calculators built before the config (setup_generic_path sets the Generic counts).
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous)
                          ? layout.inner_count : 1;
    int smem = std::max(config.shared_memory_size(),
                        (config.block_width * config.block_height / 32 + 1) * (int)sizeof(arg_t));

    const CudaT* src = reinterpret_cast<const CudaT*>(input.data<T>());
    int64_t*     dst = output.data<int64_t>();
    cuda::launch_reduce_kernel<CudaT, int64_t, ops_t>(
        ops, config, input_calc, output_calc, src, dst, layout.path, step_stride, smem, stream,
        (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.input_outer_stride : 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA index_kernel failed: ") + cudaGetErrorString(err));
    return output;
}

// ═══════════════════════════════════════════════════════════
// GPU MEAN REDUCTION DISPATCHER
// Uses MeanOps functor: sum then project(a) = a * (1/count).
// NaN-aware path uses NanSumOp which skips NaN in accumulate step.
// ═══════════════════════════════════════════════════════════
template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input,
                         const std::vector<int64_t>& normalized_axes,
                         bool keepdim, cudaStream_t stream)
{
    CUDA_KERNEL_SCOPE("dispatch_mean_gpu", stream);
    auto layout = compute_reduction_layout(input, normalized_axes);
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

    // Modified: integer and bool input now return Float64 (it was Float32), the same dtype as
    // the CPU mean (ReductionImpl.h reduce_kernel_mean: "returning fp64") and as NumPy
    // (_methods.py _mean: int -> float64). With the MeanOps fix below the values are also the
    // CPU's, bit for bit: exact int64 sum, then double(sum) * (1.0 / n), stored with no second
    // rounding. Only the final store changes (8 bytes per output instead of 4); reading and
    // summing the input are the same. Float input keeps its own dtype, as before.
    Dtype output_dtype = (std::is_integral_v<T>) ? Dtype::Float64 : input.dtype();
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype).with_device(input.device()).with_req_grad(input.requires_grad()));

    using CudaT = CudaNativeType<T>;
    using OutT  = std::conditional_t<std::is_integral_v<T>, double, T>;
    using OutCudaT = CudaNativeType<OutT>;
    using ops_t = detail::MeanOps<CudaT>;
    // Modified: the 1/n factor now has MeanOps' own FactorT, which is double for an integer
    // accumulator (int / uint / bool input) and AccT otherwise. It was declared as AccT, and for
    // integer input AccT is int64_t / uint64_t, so static_cast<AccT>(1.0 / n) was 0 for every
    // n >= 2 and every integer mean on the GPU came out 0. Measured before the fix on the H200:
    // mean int32 [512, 4096] over axis 1 -> all 512 outputs wrong by the whole mean (worst error
    // ~4.6 on true means of ~4.5). Float, half and complex means are unchanged (FactorT == AccT).
    using FactorT = typename ops_t::FactorT;
    double inv_count = 1.0 / static_cast<double>(calculate_reduced_count(input.shape().dims, normalized_axes));
    FactorT factor;
    if constexpr (std::is_same_v<CudaT, complex32_t> || std::is_same_v<CudaT, complex64_t> || std::is_same_v<CudaT, complex128_t>) {
        factor = FactorT(inv_count, 0.0);
    } else {
        factor = static_cast<FactorT>(inv_count);
    }
    ops_t ops{factor};

    // Modified: the config and the shared memory are now sized with the ACCUMULATOR type
    // arg_t = decltype(ops.identity()), exactly as the value, index and variance dispatchers
    // in this file already do. They were sized with the INPUT type CudaT:
    //   - build_reduce_config<T> sets element_size_bytes = sizeof(T), and
    //     shared_memory_size() = element_size_bytes * num_threads;
    //   - but block_x_reduce / block_y_reduce store arg_t values (one per thread).
    // For bf16/fp16 input arg_t is float (4 bytes) and CudaT is 2 bytes, so the kernel wrote
    // twice the shared memory it was given. compute-sanitizer on the H200: "Invalid __shared__
    // write of size 4 bytes" in unified_reduce_kernel<__nv_bfloat16, __nv_bfloat16,
    // MeanOps<__nv_bfloat16>, unsigned int, Path 1, 512, 4>, then "illegal memory access".
    // Every bf16/fp16 mean over axis 0 or over all axes crashed, and some last-axis shapes.
    // Integer inputs had the same mismatch (their accumulator is int64). fp32/fp64 were not
    // affected (arg_t == CudaT). Block shape is unchanged: MaxThreads is 512 for every type,
    // so only the shared-memory size changes. PyTorch sizes it the same way:
    // ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output) (aten/src/ATen/native/cuda/
    // Reduce.cuh, setReduceConfig).
    using arg_t = decltype(ops.identity());

    int num_mp, max_tpm; get_device_props(num_mp, max_tpm);
    // Modified: calculators built before the config (setup_generic_path sets the Generic counts).
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.inner_count : 1;
    int smem = std::max(config.shared_memory_size(), (config.block_width * config.block_height / 32 + 1) * (int)sizeof(arg_t));

    const CudaT* src = reinterpret_cast<const CudaT*>(input.data<T>());
    OutCudaT*    dst = reinterpret_cast<OutCudaT*>(output.data<OutT>());
    cuda::launch_reduce_kernel<CudaT, OutCudaT, ops_t>(
        ops, config, input_calc, output_calc, src, dst, layout.path, step_stride, smem, stream,
        (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.input_outer_stride : 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA mean kernel failed: ") + cudaGetErrorString(err));
    return output;
}

// ═══════════════════════════════════════════════════════════
// GPU VARIANCE REDUCTION DISPATCHER (COMPLETE IMPLEMENTATION)
//═══════════════════════════════════════════════════════════

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor& input,
                              const std::vector<int64_t>& normalized_axes,
                              bool keepdim, int64_t correction,
                              cudaStream_t stream)
{
    CUDA_KERNEL_SCOPE("dispatch_variance_gpu", stream);
    // Single-pass Welford: 1x HBM read vs old 2-pass (2x reads).
    auto layout = compute_reduction_layout(input, normalized_axes);
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    const int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);
    if (reduced_count == 0) throw std::runtime_error("GPU Variance: reduced count is zero");


    Dtype output_dtype = std::is_integral_v<T> ? Dtype::Float32 : input.dtype();
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype).with_device(input.device()).with_req_grad(input.requires_grad()));

    using CudaT       = CudaNativeType<T>;
    using OutputCppT  = std::conditional_t<std::is_integral_v<T>, float, T>;
    using OutputCudaT = CudaNativeType<OutputCppT>;
    using AccScalar   = detail::AccumulatorType<CudaT>;
    using ops_t = detail::WelfordOps<CudaT>;
    AccScalar corr_val;
    if constexpr (std::is_same_v<CudaT, complex32_t> || std::is_same_v<CudaT, complex64_t> || std::is_same_v<CudaT, complex128_t>) {
        corr_val = AccScalar(static_cast<double>(correction), 0.0);
    } else {
        corr_val = static_cast<AccScalar>(static_cast<double>(correction));
    }
    ops_t ops{corr_val, /*take_sqrt=*/false};
    using arg_t = decltype(ops.identity()); // = WelfordData<AccScalar>

    int num_mp, max_tpm; get_device_props(num_mp, max_tpm);
    // Modified: calculators built before the config (setup_generic_path sets the Generic counts).
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous)
                          ? layout.inner_count : 1;
    int smem = std::max(config.shared_memory_size(),
                        (config.block_width * config.block_height / 32 + 1) * (int)sizeof(arg_t));

    const CudaT*  src = reinterpret_cast<const CudaT*>(input.data<T>());
    OutputCudaT*  dst = reinterpret_cast<OutputCudaT*>(output.data<OutputCppT>());
    cuda::launch_reduce_kernel<CudaT, OutputCudaT, ops_t>(
        ops, config, input_calc, output_calc, src, dst, layout.path, step_stride, smem, stream,
        (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.input_outer_stride : 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA variance_kernel failed: ") + cudaGetErrorString(err));
    return output;
}



// =================================================================
//  EXPLICIT TEMPLATE INSTANTIATIONS - Using Custom Structs
// =================================================================
// ===========================================================
// UNSIGNED INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================
// uint8_t (unsigned char) - Basic operations only
template Tensor dispatch_reduction_gpu<uint8_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint8_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint8_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint8_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<uint8_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<uint8_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<uint8_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   

// uint16_t (unsigned short) - Basic operations only
template Tensor dispatch_reduction_gpu<uint16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<uint16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<uint16_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<uint16_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   

// uint32_t (unsigned int) - Basic operations only
template Tensor dispatch_reduction_gpu<uint32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint32_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint32_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint32_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<uint32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<uint32_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<uint32_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   

// uint64_t (unsigned long long) - Basic operations only
template Tensor dispatch_reduction_gpu<uint64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<uint64_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint64_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<uint64_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<uint64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<uint64_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<uint64_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   

// ===========================================================
// INTEGER TYPES - BASIC OPERATIONS ONLY (NO NaN)
// ===========================================================

// int8_t (signed char) - Basic operations only
template Tensor dispatch_reduction_gpu<int8_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int8_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int8_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int8_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int8_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<int8_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// int16_t (short) - Basic operations only
template Tensor dispatch_reduction_gpu<int16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int16_t,VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨ 
template Tensor dispatch_variance_gpu<int16_t,NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨   


// int32_t (int) - Basic operations only
template Tensor dispatch_reduction_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int32_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int32_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int32_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<int32_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// int64_t (long) - Basic operations only
template Tensor dispatch_reduction_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<int64_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<int64_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<int64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<int64_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<int64_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// ===========================================================
//  FLOATING POINT - Using CUSTOM STRUCTS (NOT __half/__nv_bfloat16)
// ===========================================================



// float16_t (custom struct)
template Tensor dispatch_reduction_gpu<float16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float16_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float16_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<float16_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction  , cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<float16_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// bfloat16_t (custom struct)
template Tensor dispatch_reduction_gpu<bfloat16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bfloat16_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bfloat16_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<bfloat16_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<bfloat16_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

// float - All operations
template Tensor dispatch_reduction_gpu<float, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<float, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<float, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_mean_gpu<float, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_variance_gpu<float, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<float, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// double - All operations
template Tensor dispatch_reduction_gpu<double, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_reduction_gpu<double, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_index_reduction_gpu<double, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_mean_gpu<double, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_mean_gpu<double, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨//✨✨✨
template Tensor dispatch_variance_gpu<double, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_variance_gpu<double, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨

//Boolean type - Basic operations only
template Tensor dispatch_mean_gpu<bool, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
// template Tensor dispatch_variance_gpu<bool, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
// template Tensor dispatch_variance_gpu<bool, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream); //✨✨✨
template Tensor dispatch_reduction_gpu<bool, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_reduction_gpu<bool, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨
template Tensor dispatch_index_reduction_gpu<bool, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);//✨✨✨   
// Add to the Bool section at the end of the file

// Boolean-specific reductions
template Tensor dispatch_reduction_gpu<bool, AllOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<bool, AnyOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
#endif // WITH_CUDA

// ===========================================================
// COMPLEX TYPES - Explicit Instantiations
// ===========================================================

#ifdef WITH_CUDA
// complex32_t
template Tensor dispatch_reduction_gpu<complex32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex32_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex32_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex32_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex32_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex32_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex32_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex32_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex32_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex32_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);
// template Tensor dispatch_variance_gpu<complex32_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);

// complex64_t
template Tensor dispatch_reduction_gpu<complex64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex64_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex64_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex64_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex64_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex64_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex64_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex64_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex64_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex64_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);
// template Tensor dispatch_variance_gpu<complex64_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);

// complex128_t
template Tensor dispatch_reduction_gpu<complex128_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, ProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, MinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, MaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_reduction_gpu<complex128_t, NanProductOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, NanMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_reduction_gpu<complex128_t, NanMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex128_t, ArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex128_t, ArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex128_t, NanArgMinOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_index_reduction_gpu<complex128_t, NanArgMaxOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex128_t, SumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
template Tensor dispatch_mean_gpu<complex128_t, NanSumOp>(const Tensor&, const std::vector<int64_t>&, bool, cudaStream_t);
// template Tensor dispatch_variance_gpu<complex128_t, VarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);
// template Tensor dispatch_variance_gpu<complex128_t, NanVarianceOp>(const Tensor& input, const std::vector<int64_t>& axes, bool keepdim , int64_t correction, cudaStream_t stream);
#endif

} // namespace detail
} // namespace OwnTensor