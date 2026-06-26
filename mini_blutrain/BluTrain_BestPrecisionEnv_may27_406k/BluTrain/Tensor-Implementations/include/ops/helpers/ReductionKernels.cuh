// include/ops/helpers/ReductionKernels.cuh
// Unified GPU reduction kernel + config solver + offset calculator
// All in one file — these helpers are only used here, no need for separate
// headers.
#pragma once

#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Constants for occupancy calculation based on architecture limits
// sm_75 (Turing): 1024 threads/SM
// sm_86/87/89 (Ampere): 1536 threads/SM
// sm_70/80 (Volta/A100): 2048 threads/SM
#if __CUDA_ARCH__ == 750
#define CUDA_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 ||  \
    __CUDA_ARCH__ == 1200
#define CUDA_MAX_THREADS_PER_SM 1536
#else
#define CUDA_MAX_THREADS_PER_SM 2048
#endif

#define GAU_MAX_THREADS_PER_BLOCK 1024

// Clipping macro to ensure (threads * blocks) <= hardware limit
#define GAU_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)                \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM)         \
        ? (blocks_per_sm)                                                      \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /               \
           (threads_per_block))))
#endif

// gpu_isnan, gpu_add, gpu_mul, gpu_lt, gpu_gt are ALL defined in ReductionOps.h
// under #ifdef __CUDA_ARCH__ — no need to re-declare them here.
#include "ReductionOps.h"
#include "ReductionUtils.h"           // For ReductionLayout
#include "device/AllocatorRegistry.h" // multi-CTA staging buffer alloc

namespace OwnTensor {
namespace cuda {

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSIONS (wrappers for __half / __nv_bfloat16)
// ═══════════════════════════════════════════════════════════

template <typename T> __device__ float to_float(T val) {
  return static_cast<float>(val);
}
template <> __device__ float to_float(__half val) { return __half2float(val); }
template <> __device__ float to_float(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T> __device__ T from_float(float val) {
  return static_cast<T>(val);
}
template <> __device__ __half from_float(float val) {
  return __float2half(val);
}
template <> __device__ __nv_bfloat16 from_float(float val) {
  return __float2bfloat16(val);
}

} // namespace cuda

namespace detail {

// ═══════════════════════════════════════════════════════════
// SECTION 1: GPU REDUCE CONFIG
// Mirrors PyTorch's ReduceConfig (Reduce.cuh:73-217)
// Decides block/grid dimensions based on the problem shape.
// ═══════════════════════════════════════════════════════════

// Max live threads per block — conservative so we don't miss occupancy
template <typename T> struct MaxThreads {
  static constexpr int VALUE = 512;
};

// Helpers: round to powers of 2
inline int next_pow2(int x) {
  if (x <= 0)
    return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}
inline int last_pow2(int x) { return next_pow2(x + 1) / 2; }
inline int div_up(int a, int b) { return (a + b - 1) / b; }

struct GpuReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  int element_size_bytes;
  int num_inputs;  // reduced_count: elements per output
  int num_outputs; // independent outputs
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};
  int block_width = 1;
  int block_height = 1;
  int num_threads = 1;
  bool vectorize_input = false;
  int output_vec_size = 1;

  GpuReduceConfig() = default;
  GpuReduceConfig(int esz, int nout, int nin)
      : element_size_bytes(esz), num_inputs(nin), num_outputs(nout) {}

  // Set block_width × block_height so it fits max_num_threads
  template <typename T> void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_t = MaxThreads<T>::VALUE / output_vec_size;
    int d0 = dim0 < max_t ? static_cast<int>(last_pow2(dim0)) : max_t;
    int d1 = dim1 < max_t ? static_cast<int>(last_pow2(dim1)) : max_t;
    block_width = std::min(d0, 32);
    block_height = std::min(d1, max_t / block_width);
    block_width = std::min(d0, max_t / block_height);
    num_threads = block_width * block_height;
  }

  int split_input(int p) {
    int s = step_input;
    step_input *= p;
    return s;
  }
  int split_output(int p) {
    int s = step_output;
    step_output *= p;
    return s;
  }

  dim3 block() const { return dim3(block_width, block_height); }
  dim3 grid() const {
    return dim3(div_up(num_outputs / output_vec_size, step_output),
                ctas_per_output);
  }

  __host__ __device__ bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }
  __host__ __device__ bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }
  __host__ __device__ bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= 32))
      return 0;
    return element_size_bytes * num_threads * output_vec_size;
  }
  int values_per_thread() const { return div_up(num_inputs, step_input); }
  int semaphore_size() const {
    return should_global_reduce() ? (int)(sizeof(int) * grid().x) : 0;
  }
};

// Host-side solver — mirrors PyTorch's setReduceConfig()
// Call this ONCE on the host before launching the kernel.
template <typename acc_t>
GpuReduceConfig build_reduce_config(const ReductionLayout &layout, int num_mp,
                                    int max_threads_per_mp) {
  int num_outputs = static_cast<int>(layout.num_outputs);
  int inputs_per_output = static_cast<int>(layout.reduced_count);

  // OuterContiguous: the output dim (num_outputs) is the "fast" axis,
  // reduced_count is how many strided rows we reduce. For a middle-axis
  // reduction num_outputs = outer*inner (NOT just inner) — the kernel remaps
  // each output index to its slab base via outer_stride.
  if (layout.path == ReductionLayout::Path::OuterContiguous) {
    num_outputs = static_cast<int>(layout.num_outputs);
    inputs_per_output = static_cast<int>(layout.reduced_count);
  } else if (layout.path == ReductionLayout::Path::Generic) {
    if (num_outputs == 0)
      num_outputs = 1;
    if (inputs_per_output == 0)
      inputs_per_output = 1;
  }

  auto config = GpuReduceConfig(sizeof(acc_t), num_outputs, inputs_per_output);
  bool inner = (layout.path != ReductionLayout::Path::OuterContiguous);

  int64_t dim0 = inner ? inputs_per_output : num_outputs;
  int64_t dim1 = inner ? num_outputs : inputs_per_output;

  // Output vectorisation — pytorch Reduce.cuh:1097-1109 port
  // When the output dim is the fast (contiguous) dim — i.e. OuterContiguous —
  // contiguous output writes can be coalesced into 16-byte (float4) stores.
  // Eligibility: fp32 accumulator, num_outputs % 4 == 0, dim0 (= num_outputs)
  // >= 128. Setting output_vec_size=4 halves max_num_threads inside
  // set_block_dimension (because mnt_wrapper-style cap is `max / vec`), which
  // collapses our 32x16 (=512) block down to 32x4 (=128) — matching pytorch's
  // geometry for c_attn/c_proj/c_fc bias-grad sums. The 4x smaller block lets
  // ~4x more CTAs live per SM and grants a much higher ctas_per_output in the
  // multi-CTA branch below, closing the 1.75x per-call gap measured against
  // pytorch.
  //
  // Gate is now on: unified_reduce_kernel_outer_vec4 + launcher dispatch +
  // staging sizing landed in Part 2. Disable only as an emergency revert path
  // (set to false to restore the scalar-store behaviour without touching the
  // rest of the file).
  // Restrict vec4 to a leading-axis reduction (outer==1), where the output is
  // one contiguous column block from base 0. For a middle-axis reduction
  // (input_outer_stride != input_row_stride) the float4 store would straddle
  // outer-slab boundaries, so the scalar OuterContiguous path (with slab remap)
  // handles it instead.
  constexpr bool kEnableOutputVec = true;
  if (kEnableOutputVec && !inner && sizeof(acc_t) == sizeof(float) &&
      layout.input_outer_stride == layout.input_row_stride &&
      (num_outputs % 4) == 0 && dim0 >= 128) {
    config.output_vec_size = 4;
    dim0 /= 4;
  }

  config.set_block_dimension<acc_t>(dim0, dim1);

  int bw = config.block_width, bh = config.block_height;

  if (inner)
    config.input_mult[0] = config.split_input(bw);
  else
    config.output_mult[0] = config.split_output(bw);

  constexpr int min_vpt = 16, max_vpt = 256;
  bool split_warps =
      config.values_per_thread() >= std::min<int>(bh * 16, max_vpt);
  if (split_warps)
    config.input_mult[1] = config.split_input(bh);
  else
    config.output_mult[1] = config.split_output(bh);

  // Multi-CTA global reduce?
  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid = num_mp * blocks_per_sm;
  int gx = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_vpt &&
      gx <= target_grid) {
    int cpo =
        std::max(std::min<int>(div_up(target_grid, gx),
                               div_up(config.values_per_thread(), min_vpt)),
                 div_up(config.values_per_thread(), max_vpt));
    config.ctas_per_output = cpo;
    if (cpo > 1)
      config.input_mult[2] = config.split_input(cpo);
  }
  return config;
}

// ═══════════════════════════════════════════════════════════
// SECTION 2: OFFSET CALCULATOR
// Generic N-D stride-based index → byte offset.
// Used only by the Generic fallback tier (~5% of DL cases).
// ═══════════════════════════════════════════════════════════

template <int NARGS = 1, typename index_t = uint32_t> struct OffsetCalculator {
  static constexpr int MAX_DIMS = 10;
  int dims;
  index_t sizes[MAX_DIMS];
  index_t magic[MAX_DIMS];  // Granlund-Montgomery magic multiplier per dim
  index_t shift1[MAX_DIMS]; // min(shift, 1) to prevent 32-bit overflow
  index_t shift2[MAX_DIMS]; // max(shift - 1, 0)
  index_t strides[NARGS][MAX_DIMS];

  __host__ __device__ OffsetCalculator() : dims(0) {}

  __host__ __device__ OffsetCalculator(int dims, const int64_t *shape,
                                       const int64_t *const *strides_arr)
      : dims(dims) {
    for (int i = 0; i < dims && i < MAX_DIMS; i++) {
      uint32_t d = static_cast<uint32_t>(shape[i]);
      this->sizes[i] = d;
      // Granlund-Montgomery 1994 / Hacker's Delight ch.10 magic-number
      // precompute. Replaces ~30-cycle hw integer divide with ~6-cycle __umulhi
      // + shift in get().
      uint32_t s;
      for (s = 0; s < 32; s++)
        if ((1U << s) >= d)
          break;
      this->shift1[i] = (s > 0) ? 1 : 0;
      this->shift2[i] = (s > 0) ? (s - 1) : 0;
      uint64_t m = ((uint64_t(1) << 32) * ((uint64_t(1) << s) - d)) / d + 1;
      this->magic[i] = static_cast<uint32_t>(m);
      for (int j = 0; j < NARGS; j++)
        this->strides[j][i] = static_cast<index_t>(strides_arr[j][i]);
    }
  }

  // O(ndim) fast div/mod — Granlund-Montgomery. Only called on the Generic (5%)
  // path.
  __host__ __device__ index_t get(index_t linear_idx, int arg = 0) const {
    index_t offset = 0;
    for (int d = dims - 1; d >= 0; d--) {
#if defined(__CUDA_ARCH__)
      uint32_t t = __umulhi(linear_idx, magic[d]);
#else
      uint32_t t = (uint64_t(linear_idx) * magic[d]) >> 32;
#endif
      uint32_t q = (t + ((linear_idx - t) >> shift1[d])) >>
                   shift2[d];                 // linear_idx / sizes[d]
      uint32_t r = linear_idx - q * sizes[d]; // linear_idx % sizes[d]
      offset += r * strides[arg][d];
      linear_idx = q;
    }
    return offset;
  }
};

// ═══════════════════════════════════════════════════════════
// SECTION 3: PACKED REDUCE OP STRUCT
// Packs all kernel arguments into a single struct to minimize
// cudaLaunchKernel parameter overhead (like PyTorch's ReduceOp).
// ═══════════════════════════════════════════════════════════

template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t = uint32_t>
struct ReduceOp {
  ops_t ops;
  GpuReduceConfig config;
  OffsetCalculator<1, index_t> input_calc;
  OffsetCalculator<1, index_t> output_calc;
  const scalar_t *__restrict__ src;
  out_scalar_t *__restrict__ dst;
  int64_t step_stride;
  int64_t outer_stride; // OuterContiguous middle-axis: stride between outer
                        // slabs (reduced*inner); 0 otherwise
  // Multi-CTA staging (populated only when config.should_global_reduce()).
  // cta_buf: num_outputs * ctas_per_output * sizeof(arg_t), uninitialised.
  // semaphores: num_outputs ints, zero-initialised by host before launch.
  void *cta_buf = nullptr;
  int *semaphores = nullptr;
};

} // namespace detail

namespace cuda {

// ═══════════════════════════════════════════════════════════
// SECTION 4: WARP / BLOCK REDUCE STAGES
// ═══════════════════════════════════════════════════════════

// Block-X reduce: all threads in a row collaborate to reduce input
// Uses warp shuffles first (register), then shared memory for cross-warp.
template <typename arg_t, typename ops_t>
__device__ arg_t block_x_reduce(arg_t value, const ops_t &ops,
                                char *shared_memory) {
  // Block is 2-D: threadIdx.x walks the reduced dim, threadIdx.y indexes
  // independent outputs. The reduction must stay WITHIN a threadIdx.y row.
  // (Mirrors PyTorch ATen Reduce.cuh::block_x_reduce.)
  int dim_x = blockDim.x;
  arg_t *smem = reinterpret_cast<arg_t *>(shared_memory);

  // Cross-warp stage — only when a row spans more than one warp. Indexed by the
  // full 2-D thread id so rows never clobber each other.
  if (dim_x > 32) {
    int addr = threadIdx.x + threadIdx.y * blockDim.x;
    smem[addr] = value;
    for (int offset = dim_x / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
        value = ops.combine(value, smem[addr + offset]);
        smem[addr] = value;
      }
    }
    dim_x = 32;
  }

  __syncthreads();
  // Intra-warp shuffle bounded by the row width: with block_width < 32 a
  // physical warp holds several rows, so offsets must stay < dim_x to avoid
  // summing across neighbouring outputs. Result lands in threadIdx.x == 0.
  for (int offset = dim_x >> 1; offset > 0; offset >>= 1)
    value = ops.combine(value, ops.warp_shfl_down(value, offset));
  return value;
}

// Block-Y reduce: threads in the same column (same threadIdx.x) reduce
// across all Y rows using shared memory.
template <typename arg_t, typename ops_t>
__device__ arg_t block_y_reduce(arg_t value, const ops_t &ops,
                                char *shared_memory) {
  arg_t *smem = reinterpret_cast<arg_t *>(shared_memory);
  smem[threadIdx.x + threadIdx.y * blockDim.x] = value;
  __syncthreads();

  if (threadIdx.y == 0) {
    value = smem[threadIdx.x];
    for (int i = 1; i < blockDim.y; i++)
      value = ops.combine(value, smem[threadIdx.x + i * blockDim.x]);
  }
  // Trailing barrier: the vec4 kernel calls this once per output lane reusing
  // the SAME shared buffer. Without this sync, ty!=0 threads race ahead to the
  // next lane and overwrite smem while ty==0 is still reading the current
  // lane's partials — corrupting every lane except the last. (PyTorch avoids
  // this by reducing all lanes in one vec-wide pass.)
  __syncthreads();
  return value;
}

// ═══════════════════════════════════════════════════════════
// SECTION 5: UNIFIED REDUCTION KERNEL
// Templated on Path to eliminate runtime branching.
// Uses vectorized loads (vt0 elements per iteration) and
// multiple independent accumulators for ILP.
//
// Template params:
//   scalar_t      — input element type (after CudaNativeType conversion)
//   out_scalar_t  — output element type
//   ops_t         — functor (SumOp, MinOp, ArgMinOp, MeanOps, WelfordOps, ...)
//   index_t       — indexing width (uint32_t for tensors < 4GB)
//   PATH          — compile-time path selection (no runtime branching)
//   NT            — num_threads for __launch_bounds__
//   VT0           — values per thread for ILP (default 4)
// ═══════════════════════════════════════════════════════════

// Vectorized load helper: load VT0 elements as a single wide transaction
template <typename T, int N> struct alignas(sizeof(T) * N) VecLoad {
  T val[N];
};

// Compile-time min-blocks-per-SM based on num_threads:
//   512 threads → 4 blocks/SM (2048 threads/SM)
//   256 threads → 8 blocks/SM
//   128 threads → 4 blocks/SM (conservative — high register kernels)
//   64  threads → 4 blocks/SM
//   32  threads → 4 blocks/SM
// Capped at 4 to avoid ptxas spilling registers trying to satisfy
// an unreachable occupancy target for complex functors (WelfordOps, etc).
// The compiler will still achieve higher occupancy when registers allow.
// Capped at hardware limit to ensure valid occupancy and avoid warnings (sm_86:
// 1536 threads/SM).
template <int NT> struct MinBlocksPerSM {
  static constexpr int requested = (NT >= 256) ? (2048 / NT) : 4;
  static constexpr int VALUE = GAU_MIN_BLOCKS_PER_SM(NT, requested);
};

template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t, detail::ReductionLayout::Path PATH, int NT,
          int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE) __global__
    void unified_reduce_kernel(
        detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
  extern __shared__ char shared_memory[];

  // accumulator type is whatever the functor's identity() returns
  using arg_t = decltype(op.ops.identity());
  const auto &config = op.config;

  // ── 1. Map this thread to an output index ──
  index_t output_idx =
      (index_t)threadIdx.x *
          config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.x * config.step_output;

  if (output_idx >= (index_t)config.num_outputs)
    return;

  // ── 2. Map this thread to its starting input index ──
  // input_mult[CTA] term: when ctas_per_output > 1, each block along grid.y
  // owns a disjoint slice of the reduced axis. Without this term, all CTAs
  // for a given output_idx would walk the same range (the multi-CTA bug).
  index_t idx =
      (index_t)threadIdx.x *
          config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.y * config.input_mult[detail::GpuReduceConfig::CTA];

  const index_t end = (index_t)config.num_inputs;
  const index_t stride = (index_t)config.step_input;

  // ── 3. Thread-local reduction with ILP (VT0 independent accumulators) ──
  // Multiple accumulators hide the latency of dependent reduce operations.
  arg_t acc[VT0];
#pragma unroll
  for (int i = 0; i < VT0; i++)
    acc[i] = op.ops.identity();

  if constexpr (PATH == detail::ReductionLayout::Path::InnerContiguous) {
    // TIER 1: Contiguous – stride=1, output × reduced_count laid out linearly.
    // Covers ~70% of DL workloads (e.g. sum(axis=-1) on row-major tensors).
    const scalar_t *__restrict__ base_ptr =
        op.src + output_idx * (index_t)config.num_inputs;

    // Vectorized main loop: load VT0 elements per iteration
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        acc[i] = op.ops.reduce(acc[i], base_ptr[idx + i * stride],
                               (int64_t)(idx + i * stride));
      }
      idx += stride * VT0;
    }
    // Scalar tail
    while (idx < end) {
      acc[0] = op.ops.reduce(acc[0], base_ptr[idx], (int64_t)idx);
      idx += stride;
    }
  } else if constexpr (PATH == detail::ReductionLayout::Path::OuterContiguous) {
    // TIER 2: Single-stride – output dimension is innermost, reduced dim is
    // outer. Covers ~25% of DL workloads (e.g. sum(axis=0) on NCHW → C).
    // Remap the flat output index to its input slab base: leading-axis
    // reduction has outer==1 so o_outer==0 → base == output_idx; a middle-axis
    // reduction (outer>1) lives at o_outer*outer_stride + o_inner.
    const index_t inner_count = (index_t)op.step_stride;
    const index_t o_outer = output_idx / inner_count;
    const index_t o_inner = output_idx - o_outer * inner_count;
    const scalar_t *__restrict__ base_ptr =
        op.src + o_outer * (index_t)op.outer_stride + o_inner;
    const index_t row_stride = (index_t)op.step_stride;

    // ILP main loop: VT0 independent accumulators, strided access
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        acc[i] =
            op.ops.reduce(acc[i], base_ptr[(idx + i * stride) * row_stride],
                          (int64_t)(idx + i * stride));
      }
      idx += stride * VT0;
    }
    // Scalar tail
    while (idx < end) {
      acc[0] = op.ops.reduce(acc[0], base_ptr[idx * row_stride], (int64_t)idx);
      idx += stride;
    }
  } else {
    // TIER 3: Generic fallback — uses OffsetCalculator (O(ndim) div/mod).
    // Covers ~5% of cases: non-standard axes, non-contiguous or permuted
    // tensors.
    index_t out_base =
        (op.output_calc.dims > 0) ? op.output_calc.get(output_idx) : output_idx;

    // ILP main loop for generic path
    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (int i = 0; i < VT0; i++) {
        index_t cur = idx + i * stride;
        index_t in_off =
            (op.input_calc.dims > 0) ? op.input_calc.get(cur) : cur;
        acc[i] = op.ops.reduce(acc[i], op.src[out_base + in_off], (int64_t)cur);
      }
      idx += stride * VT0;
    }
    // Scalar tail
    while (idx < end) {
      index_t in_off = (op.input_calc.dims > 0) ? op.input_calc.get(idx) : idx;
      acc[0] = op.ops.reduce(acc[0], op.src[out_base + in_off], (int64_t)idx);
      idx += stride;
    }
  }

// ── 4. Combine VT0 accumulators into acc[0] ──
#pragma unroll
  for (int i = 1; i < VT0; i++) {
    acc[0] = op.ops.combine(acc[0], acc[i]);
  }

  arg_t value = acc[0];

  // ── 5. Block-level map-reduce stages ──
  if (config.should_block_x_reduce())
    value = block_x_reduce(value, op.ops, shared_memory);
  if (config.should_block_y_reduce())
    value = block_y_reduce(value, op.ops, shared_memory);

  // ── 6. Write output ──
  bool is_leader = (!config.should_block_x_reduce() || threadIdx.x == 0) &&
                   (!config.should_block_y_reduce() || threadIdx.y == 0);

  // ── 6a. Multi-CTA path — line-by-line port of PyTorch
  // ReduceOp::global_reduce
  //                       (aten/src/ATen/native/cuda/Reduce.cuh:786-870).
  //
  // Invariant entering this block: `value` is the per-CTA block-reduced partial
  // (post block_x_reduce AND block_y_reduce). For threads where !is_leader the
  // contents of `value` are stale/garbage — that is OK, only is_leader threads
  // stage and only the last CTA reads back from staging.
  //
  // Staging layout (matches PT's staging_memory_offset):
  //   block_x_reduce ON  : slot = blockIdx.y + blockIdx.x*gridDim.y
  //                        (1 slot per CTA)
  //   block_x_reduce OFF : slot = threadIdx.x + (blockIdx.y +
  //   blockIdx.x*gridDim.y) * blockDim.x
  //                        (blockDim.x slots per CTA — one per output column
  //                        held by lane tx)
  //
  // Semaphores: one int per blockIdx.x tile, zero-initialised by the host.
  if (config.should_global_reduce()) {
    arg_t *cta_buf = reinterpret_cast<arg_t *>(op.cta_buf);

    auto staging_offset = [&](index_t cta2) -> index_t {
      index_t off = cta2 + (index_t)blockIdx.x * (index_t)gridDim.y;
      if (!config.should_block_x_reduce())
        off = (index_t)threadIdx.x + off * (index_t)blockDim.x;
      return off;
    };

    // (1) Stage: every is_leader thread writes its (block-reduced) partial
    //     for this CTA at slot staging_offset(blockIdx.y).
    if (is_leader) {
      cta_buf[staging_offset((index_t)blockIdx.y)] = value;
    }
    __threadfence(); // store must be globally visible before atomicAdd
    __syncthreads();

    // (2) Last-CTA detection — one atomic per output-column tile (blockIdx.x).
    __shared__ bool s_is_last;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev = atomicAdd(&op.semaphores[blockIdx.x], 1);
      s_is_last = (prev == config.ctas_per_output - 1);
    }
    __syncthreads();
    if (!s_is_last)
      return;

    __threadfence(); // acquire pattern after the atomic: pair with the
                     // producer's __threadfence above.

    // (3) Every thread in the last CTA resets to identity and re-reads its
    //     slice of the staging buffer, combining cpo partials in parallel.
    //     Iteration pattern mirrors PT's global_reduce loop.
    value = op.ops.identity();
    if (config.should_block_x_reduce()) {
      // Each thread covers a strided subset of [0, cpo).
      index_t input_offset =
          (index_t)threadIdx.x + (index_t)threadIdx.y * (index_t)blockDim.x;
      index_t step = (index_t)blockDim.x * (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        value = op.ops.combine(value, cta_buf[staging_offset(input_offset)]);
      }
    } else {
      // Each (tx, ty) reads slots {tx + (ty + bx*gd.y)*bd.x, tx + (ty+bd.y +
      // bx*gd.y)*bd.x, …} i.e. each lane reconstructs its own output column
      // from cpo partials, distributed across the y-dim threads of the last
      // CTA.
      index_t input_offset = (index_t)threadIdx.y;
      index_t step = (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        value = op.ops.combine(value, cta_buf[staging_offset(input_offset)]);
      }
    }

    // (4) Final block-level reductions over the y (and possibly x) dim so
    //     only the is_leader thread holds the full sum for its output_idx.
    value = block_y_reduce(value, op.ops, shared_memory);
    if (config.should_block_x_reduce())
      value = block_x_reduce(value, op.ops, shared_memory);

    // (5) Same write rule as the single-CTA path.
    if (is_leader && output_idx < (index_t)config.num_outputs) {
      auto final_val = op.ops.project(value);
      if constexpr (std::is_same_v<out_scalar_t, __half>)
        op.dst[output_idx] =
            from_float<out_scalar_t>(static_cast<float>(final_val));
      else
        op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
    }
    return;
  }

  // ── 6b. Single-CTA path (unchanged): leader writes dst directly ──
  if (is_leader) {
    auto final_val = op.ops.project(value);

    if constexpr (std::is_same_v<out_scalar_t, __half>)
      op.dst[output_idx] =
          from_float<out_scalar_t>(static_cast<float>(final_val));
    else
      op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
  }
}

// ═══════════════════════════════════════════════════════════
// SECTION 5b: VECTORISED OUTPUT REDUCTION KERNEL (OUTPUT_VEC=4)
// Specialised for OuterContiguous + fp32 + num_outputs % 4 == 0.
// Each thread handles 4 consecutive output columns simultaneously:
//   • float4 LDG per input row (16-byte coalesced load).
//   • 4 independent column accumulators.
//   • float4 STG per output (16-byte coalesced store).
//
// Matches PyTorch's output_vec_size=4 codepath in Reduce.cuh. Triggered by
// the launcher when config.output_vec_size == 4 AND path == OuterContiguous.
// The host config halves max_num_threads (mnt_wrapper / vec_size) so
// block_height collapses from 16 → 4, freeing register budget for the 4×
// accumulator state and quadrupling blocks-per-SM occupancy.
// ═══════════════════════════════════════════════════════════
template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t, int NT, int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE) __global__
    void unified_reduce_kernel_outer_vec4(
        detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
  extern __shared__ char shared_memory[];
  using arg_t = decltype(op.ops.identity());
  const auto &config = op.config;

  constexpr int OUTPUT_VEC = 4;

  // ── 1. Each thread maps to a 4-wide vec block of consecutive output columns
  // ── output_mult/step_output are computed by host as if 1 output per thread;
  // we multiply by OUTPUT_VEC to get the base scalar index of the 4-wide block.
  index_t output_idx_base =
      ((index_t)threadIdx.x *
           config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
       (index_t)threadIdx.y *
           config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
       (index_t)blockIdx.x * config.step_output) *
      OUTPUT_VEC;

  if (output_idx_base >= (index_t)config.num_outputs)
    return;

  // ── 2. Starting input-row index for this thread ──
  index_t idx =
      (index_t)threadIdx.x *
          config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
      (index_t)threadIdx.y *
          config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
      (index_t)blockIdx.y * config.input_mult[detail::GpuReduceConfig::CTA];

  const index_t end = (index_t)config.num_inputs;
  const index_t stride = (index_t)config.step_input;
  const index_t row_stride = (index_t)op.step_stride;

  // ── 3. Per-output-column accumulators (one per vec lane) ──
  arg_t acc[OUTPUT_VEC];
#pragma unroll
  for (int v = 0; v < OUTPUT_VEC; v++)
    acc[v] = op.ops.identity();

  // ── 4. OuterContiguous strided read; one float4 LDG per row ──
  const scalar_t *__restrict__ base_ptr = op.src + output_idx_base;
  while (idx < end) {
    const scalar_t *row_ptr = base_ptr + (index_t)idx * row_stride;
    if constexpr (std::is_same_v<scalar_t, float>) {
      float4 v4 = *reinterpret_cast<const float4 *>(row_ptr);
      acc[0] = op.ops.reduce(acc[0], v4.x, (int64_t)idx);
      acc[1] = op.ops.reduce(acc[1], v4.y, (int64_t)idx);
      acc[2] = op.ops.reduce(acc[2], v4.z, (int64_t)idx);
      acc[3] = op.ops.reduce(acc[3], v4.w, (int64_t)idx);
    } else {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        acc[v] = op.ops.reduce(acc[v], row_ptr[v], (int64_t)idx);
    }
    idx += stride;
  }

  // ── 5. Block-level reductions (4 sequential calls — one per output column)
  // ── The extra __syncthreads inside block_x/y_reduce are unavoidable given
  // the scalar shared-mem layout. A future optimisation could batch the 4
  // reductions into a single set of syncs with an interleaved smem layout.
  if (config.should_block_x_reduce()) {
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_x_reduce(acc[v], op.ops, shared_memory);
  }
  if (config.should_block_y_reduce()) {
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_y_reduce(acc[v], op.ops, shared_memory);
  }

  bool is_leader = (!config.should_block_x_reduce() || threadIdx.x == 0) &&
                   (!config.should_block_y_reduce() || threadIdx.y == 0);

  // ── 6a. Multi-CTA staging path — same protocol as scalar kernel but
  //        each is_leader thread stages OUTPUT_VEC values consecutively. ──
  if (config.should_global_reduce()) {
    arg_t *cta_buf = reinterpret_cast<arg_t *>(op.cta_buf);

    auto staging_offset = [&](index_t cta2) -> index_t {
      index_t off = cta2 + (index_t)blockIdx.x * (index_t)gridDim.y;
      if (!config.should_block_x_reduce())
        off = (index_t)threadIdx.x + off * (index_t)blockDim.x;
      return off;
    };

    // (1) Stage 4 values per slot — host launcher sized cta_buf accordingly.
    if (is_leader) {
      index_t base_off = staging_offset((index_t)blockIdx.y) * OUTPUT_VEC;
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        cta_buf[base_off + v] = acc[v];
    }
    __threadfence();
    __syncthreads();

    // (2) Last-CTA detection (single semaphore per blockIdx.x tile).
    __shared__ bool s_is_last;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev = atomicAdd(&op.semaphores[blockIdx.x], 1);
      s_is_last = (prev == config.ctas_per_output - 1);
    }
    __syncthreads();
    if (!s_is_last)
      return;

    __threadfence();

// (3) Last CTA reads OUTPUT_VEC values per slot.
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = op.ops.identity();

    if (config.should_block_x_reduce()) {
      index_t input_offset =
          (index_t)threadIdx.x + (index_t)threadIdx.y * (index_t)blockDim.x;
      index_t step = (index_t)blockDim.x * (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        index_t base_off = staging_offset(input_offset) * OUTPUT_VEC;
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++)
          acc[v] = op.ops.combine(acc[v], cta_buf[base_off + v]);
      }
    } else {
      index_t input_offset = (index_t)threadIdx.y;
      index_t step = (index_t)blockDim.y;
      for (; input_offset < (index_t)config.ctas_per_output;
           input_offset += step) {
        index_t base_off = staging_offset(input_offset) * OUTPUT_VEC;
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++)
          acc[v] = op.ops.combine(acc[v], cta_buf[base_off + v]);
      }
    }

// (4) Final block-level reductions per output column.
#pragma unroll
    for (int v = 0; v < OUTPUT_VEC; v++)
      acc[v] = block_y_reduce(acc[v], op.ops, shared_memory);
    if (config.should_block_x_reduce()) {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++)
        acc[v] = block_x_reduce(acc[v], op.ops, shared_memory);
    }

    // (5) Vector store — single float4 STG when out_scalar_t is fp32.
    if (is_leader && output_idx_base < (index_t)config.num_outputs) {
      if constexpr (std::is_same_v<out_scalar_t, float>) {
        float4 r;
        r.x = static_cast<float>(op.ops.project(acc[0]));
        r.y = static_cast<float>(op.ops.project(acc[1]));
        r.z = static_cast<float>(op.ops.project(acc[2]));
        r.w = static_cast<float>(op.ops.project(acc[3]));
        *reinterpret_cast<float4 *>(op.dst + output_idx_base) = r;
      } else {
#pragma unroll
        for (int v = 0; v < OUTPUT_VEC; v++) {
          auto final_val = op.ops.project(acc[v]);
          if constexpr (std::is_same_v<out_scalar_t, __half>)
            op.dst[output_idx_base + v] =
                from_float<out_scalar_t>(static_cast<float>(final_val));
          else
            op.dst[output_idx_base + v] = static_cast<out_scalar_t>(final_val);
        }
      }
    }
    return;
  }

  // ── 6b. Single-CTA path: leader does the vector store directly. ──
  if (is_leader) {
    if constexpr (std::is_same_v<out_scalar_t, float>) {
      float4 r;
      r.x = static_cast<float>(op.ops.project(acc[0]));
      r.y = static_cast<float>(op.ops.project(acc[1]));
      r.z = static_cast<float>(op.ops.project(acc[2]));
      r.w = static_cast<float>(op.ops.project(acc[3]));
      *reinterpret_cast<float4 *>(op.dst + output_idx_base) = r;
    } else {
#pragma unroll
      for (int v = 0; v < OUTPUT_VEC; v++) {
        auto final_val = op.ops.project(acc[v]);
        if constexpr (std::is_same_v<out_scalar_t, __half>)
          op.dst[output_idx_base + v] =
              from_float<out_scalar_t>(static_cast<float>(final_val));
        else
          op.dst[output_idx_base + v] = static_cast<out_scalar_t>(final_val);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════
// SECTION 6: HOST-SIDE KERNEL LAUNCHER
// Resolves Path at compile time and dispatches to the correct
// kernel specialization. Packs args into ReduceOp struct.
// ═══════════════════════════════════════════════════════════

template <typename scalar_t, typename out_scalar_t, typename ops_t,
          typename index_t = uint32_t, int VT0 = 4>
inline void
launch_reduce_kernel(const ops_t &ops, const detail::GpuReduceConfig &config,
                     const detail::OffsetCalculator<1, index_t> &input_calc,
                     const detail::OffsetCalculator<1, index_t> &output_calc,
                     const scalar_t *src, out_scalar_t *dst,
                     detail::ReductionLayout::Path path, int64_t step_stride,
                     int smem, cudaStream_t stream, int64_t outer_stride = 0) {
  // Pack into single struct for minimal launch parameter overhead
  detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
  op.ops = ops;
  op.config = config;
  op.input_calc = input_calc;
  op.output_calc = output_calc;
  op.src = src;
  op.dst = dst;
  op.step_stride = step_stride;
  op.outer_stride = outer_stride;

  // Multi-CTA staging: allocate cta_buf (uninit) and semaphores (zero-init).
  // Sized exactly like PyTorch's: num_outputs*ctas_per_output accumulators and
  // num_outputs ints. Uses the caching allocator so repeated launches reuse
  // the same blocks — no per-call cudaMalloc/cudaFree.
  //
  // OUTPUT_VEC>1: each is_leader thread stages `output_vec_size` consecutive
  // arg_t values per slot, so multiply the per-slot footprint by
  // output_vec_size. This matches pytorch's global_memory_size() formula in
  // Reduce.cuh:192-200.
  using arg_t = decltype(ops.identity());
  Allocator *scratch = nullptr;
  if (config.should_global_reduce()) {
    scratch = AllocatorRegistry::get_caching_allocator();
    const dim3 g = config.grid();
    // cta_buf sizing (matches PyTorch's global_memory_size):
    //   1 slot per (CTA-x, CTA-y) when block_x_reduce is active
    //   blockDim.x slots per (CTA-x, CTA-y) otherwise (each lane reconstructs
    //   its own column)
    size_t slots = static_cast<size_t>(g.x) * static_cast<size_t>(g.y);
    if (!config.should_block_x_reduce()) {
      slots *= static_cast<size_t>(config.block_width);
    }
    size_t cta_bytes =
        sizeof(arg_t) * slots * static_cast<size_t>(config.output_vec_size);
    size_t sem_bytes = sizeof(int) * static_cast<size_t>(g.x);
    op.cta_buf = scratch->allocate(cta_bytes);
    op.semaphores = reinterpret_cast<int *>(scratch->allocate(sem_bytes));
    cudaMemsetAsync(op.semaphores, 0, sem_bytes, stream);
  }

  const int nt = config.num_threads;

// Dispatch on Path at compile time + num_threads for launch_bounds.
// Using a lambda to avoid duplicating the grid/block/smem logic.
#define LAUNCH_KERNEL(PATH_ENUM, NT_VAL)                                       \
  unified_reduce_kernel<scalar_t, out_scalar_t, ops_t, index_t, PATH_ENUM,     \
                        NT_VAL, VT0>                                           \
      <<<config.grid(), config.block(), smem, stream>>>(op)

// Vec-4 dispatch — only enabled when the host config opted in
// (build_reduce_config gates this on path == OuterContiguous, sizeof(acc_t)==4,
// num_outputs % 4 == 0). Uses the specialised kernel below that emits float4
// loads/stores.
#define LAUNCH_VEC4(NT_VAL)                                                    \
  unified_reduce_kernel_outer_vec4<scalar_t, out_scalar_t, ops_t, index_t,     \
                                   NT_VAL, VT0>                                \
      <<<config.grid(), config.block(), smem, stream>>>(op)

  if (config.output_vec_size == 4 &&
      path == detail::ReductionLayout::Path::OuterContiguous) {
    if (nt <= 32) {
      LAUNCH_VEC4(32);
    } else if (nt <= 64) {
      LAUNCH_VEC4(64);
    } else if (nt <= 128) {
      LAUNCH_VEC4(128);
    } else if (nt <= 256) {
      LAUNCH_VEC4(256);
    } else {
      LAUNCH_VEC4(512);
    }
  } else if (path == detail::ReductionLayout::Path::InnerContiguous) {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 512);
    }
  } else if (path == detail::ReductionLayout::Path::OuterContiguous) {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 512);
    }
  } else {
    if (nt <= 32) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 32);
    } else if (nt <= 64) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 64);
    } else if (nt <= 128) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 128);
    } else if (nt <= 256) {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 256);
    } else {
      LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 512);
    }
  }

#undef LAUNCH_KERNEL
#undef LAUNCH_VEC4

  // Return scratch blocks to the caching allocator. The allocator pools them
  // for reuse on the next launch, so this is essentially free.
  if (scratch) {
    scratch->deallocate(op.cta_buf);
    scratch->deallocate(op.semaphores);
  }
}

} // namespace cuda
} // namespace OwnTensor

#endif // REDUCTION_KERNELS_CUH
