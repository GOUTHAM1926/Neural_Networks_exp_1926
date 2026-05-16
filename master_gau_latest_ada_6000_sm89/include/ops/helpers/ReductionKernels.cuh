// include/ops/helpers/ReductionKernels.cuh
// Unified GPU reduction kernel + config solver + offset calculator
// All in one file — these helpers are only used here, no need for separate headers.
#pragma once

#ifndef REDUCTION_KERNELS_CUH
#define REDUCTION_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <algorithm>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Constants for occupancy calculation based on architecture limits
// sm_75 (Turing): 1024 threads/SM
// sm_86/87/89 (Ampere): 1536 threads/SM
// sm_70/80 (Volta/A100): 2048 threads/SM
#if __CUDA_ARCH__ == 750
  #define CUDA_MAX_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
  #define CUDA_MAX_THREADS_PER_SM 1536
#else
  #define CUDA_MAX_THREADS_PER_SM 2048
#endif

#define GAU_MAX_THREADS_PER_BLOCK 1024

// Clipping macro to ensure (threads * blocks) <= hardware limit
#define GAU_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)        \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /       \
           (threads_per_block))))
#endif

// gpu_isnan, gpu_add, gpu_mul, gpu_lt, gpu_gt are ALL defined in ReductionOps.h
// under #ifdef __CUDA_ARCH__ — no need to re-declare them here.
#include "ReductionOps.h"
#include "ReductionUtils.h"  // For ReductionLayout
#include "device/AllocatorRegistry.h"  // multi-CTA staging buffer alloc

namespace OwnTensor {
namespace cuda {

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSIONS (wrappers for __half / __nv_bfloat16)
// ═══════════════════════════════════════════════════════════

template<typename T> __device__ float to_float(T val) { return static_cast<float>(val); }
template<> __device__ float to_float(__half val)        { return __half2float(val); }
template<> __device__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T> __device__ T from_float(float val) { return static_cast<T>(val); }
template<> __device__ __half       from_float(float val) { return __float2half(val); }
template<> __device__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

} // namespace cuda

namespace detail {

// ═══════════════════════════════════════════════════════════
// SECTION 1: GPU REDUCE CONFIG
// Mirrors PyTorch's ReduceConfig (Reduce.cuh:73-217)
// Decides block/grid dimensions based on the problem shape.
// ═══════════════════════════════════════════════════════════

// Max live threads per block — conservative so we don't miss occupancy
template<typename T>
struct MaxThreads { static constexpr int VALUE = 512; };

// Helpers: round to powers of 2
inline int next_pow2(int x) {
    if (x <= 0) return 1;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
inline int last_pow2(int x)   { return next_pow2(x + 1) / 2; }
inline int div_up(int a, int b){ return (a + b - 1) / b; }

struct GpuReduceConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA     = 2;

    int element_size_bytes;
    int num_inputs;           // reduced_count: elements per output
    int num_outputs;          // independent outputs
    int step_input  = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3]  = {0, 0, 0};
    int output_mult[2] = {0, 0};
    int block_width  = 1;
    int block_height = 1;
    int num_threads  = 1;
    bool vectorize_input = false;
    int  output_vec_size = 1;

    GpuReduceConfig() = default;
    GpuReduceConfig(int esz, int nout, int nin)
        : element_size_bytes(esz), num_inputs(nin), num_outputs(nout) {}

    // Set block_width × block_height so it fits max_num_threads
    template<typename T>
    void set_block_dimension(int64_t dim0, int64_t dim1) {
        const int max_t = MaxThreads<T>::VALUE / output_vec_size;
        int d0 = dim0 < max_t ? static_cast<int>(last_pow2(dim0)) : max_t;
        int d1 = dim1 < max_t ? static_cast<int>(last_pow2(dim1)) : max_t;
        block_width  = std::min(d0, 32);
        block_height = std::min(d1, max_t / block_width);
        block_width  = std::min(d0, max_t / block_height);
        num_threads  = block_width * block_height;
    }

    int split_input (int p) { int s = step_input;  step_input  *= p; return s; }
    int split_output(int p) { int s = step_output; step_output *= p; return s; }

    dim3 block() const { return dim3(block_width, block_height); }
    dim3 grid()  const { return dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output); }

    __host__ __device__ bool should_block_x_reduce() const { return input_mult[BLOCK_X] != 0; }
    __host__ __device__ bool should_block_y_reduce() const { return input_mult[BLOCK_Y] != 0; }
    __host__ __device__ bool should_global_reduce()  const { return input_mult[CTA]     != 0; }

    int shared_memory_size() const {
        if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= 32))
            return 0;
        return element_size_bytes * num_threads * output_vec_size;
    }
    int values_per_thread() const { return div_up(num_inputs, step_input); }
    int semaphore_size()    const {
        return should_global_reduce() ? (int)(sizeof(int) * grid().x) : 0;
    }
};

// Host-side solver — mirrors PyTorch's setReduceConfig()
// Call this ONCE on the host before launching the kernel.
template<typename acc_t>
GpuReduceConfig build_reduce_config(const ReductionLayout& layout, int num_mp, int max_threads_per_mp) {
    int num_outputs       = static_cast<int>(layout.num_outputs);
    int inputs_per_output = static_cast<int>(layout.reduced_count);

    // OuterContiguous: inner_count is the "fast" dimension (outputs),
    // num_outputs is how many rows we reduce over.
    if (layout.path == ReductionLayout::Path::OuterContiguous) {
        num_outputs       = static_cast<int>(layout.inner_count);
        inputs_per_output = static_cast<int>(layout.reduced_count);
    } else if (layout.path == ReductionLayout::Path::Generic) {
        if (num_outputs       == 0) num_outputs       = 1;
        if (inputs_per_output == 0) inputs_per_output = 1;
    }

    auto config = GpuReduceConfig(sizeof(acc_t), num_outputs, inputs_per_output);
    bool inner = (layout.path != ReductionLayout::Path::OuterContiguous);

    int64_t dim0 = inner ? inputs_per_output : num_outputs;
    int64_t dim1 = inner ? num_outputs       : inputs_per_output;
    config.set_block_dimension<acc_t>(dim0, dim1);

    int bw = config.block_width, bh = config.block_height;

    if (inner) config.input_mult[0]  = config.split_input(bw);
    else       config.output_mult[0] = config.split_output(bw);

    constexpr int min_vpt = 16, max_vpt = 256;
    bool split_warps = config.values_per_thread() >= std::min<int>(bh * 16, max_vpt);
    if (split_warps) config.input_mult[1]  = config.split_input(bh);
    else             config.output_mult[1] = config.split_output(bh);

    // Multi-CTA global reduce?
    const int blocks_per_sm = max_threads_per_mp / config.num_threads;
    const int target_grid   = num_mp * blocks_per_sm;
    int gx = config.grid().x;
    if (config.input_mult[1] != 0 && config.values_per_thread() >= max_vpt && gx <= target_grid) {
        int cpo = std::max(
            std::min<int>(div_up(target_grid, gx), div_up(config.values_per_thread(), min_vpt)),
            div_up(config.values_per_thread(), max_vpt));
        config.ctas_per_output = cpo;
        if (cpo > 1) config.input_mult[2] = config.split_input(cpo);
    }
    return config;
}

// ═══════════════════════════════════════════════════════════
// SECTION 2: OFFSET CALCULATOR
// Generic N-D stride-based index → byte offset.
// Used only by the Generic fallback tier (~5% of DL cases).
// ═══════════════════════════════════════════════════════════

template<int NARGS = 1, typename index_t = uint32_t>
struct OffsetCalculator {
    static constexpr int MAX_DIMS = 10;
    int     dims;
    index_t sizes[MAX_DIMS];
    index_t strides[NARGS][MAX_DIMS];

    __host__ __device__ OffsetCalculator() : dims(0) {}

    __host__ __device__ OffsetCalculator(int dims, const int64_t* shape,
                                          const int64_t* const* strides_arr) : dims(dims) {
        for (int i = 0; i < dims && i < MAX_DIMS; i++) {
            this->sizes[i] = static_cast<index_t>(shape[i]);
            for (int j = 0; j < NARGS; j++)
                this->strides[j][i] = static_cast<index_t>(strides_arr[j][i]);
        }
    }

    // O(ndim) div/mod — only called on the Generic (5%) path
    __host__ __device__ index_t get(index_t linear_idx, int arg = 0) const {
        index_t offset = 0;
        for (int d = dims - 1; d >= 0; d--) {
            index_t coord  = linear_idx % sizes[d];
            linear_idx    /= sizes[d];
            offset        += coord * strides[arg][d];
        }
        return offset;
    }
};

// ═══════════════════════════════════════════════════════════
// SECTION 3: PACKED REDUCE OP STRUCT
// Packs all kernel arguments into a single struct to minimize
// cudaLaunchKernel parameter overhead (like PyTorch's ReduceOp).
// ═══════════════════════════════════════════════════════════

template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t = uint32_t>
struct ReduceOp {
    ops_t ops;
    GpuReduceConfig config;
    OffsetCalculator<1, index_t> input_calc;
    OffsetCalculator<1, index_t> output_calc;
    const scalar_t*  __restrict__ src;
    out_scalar_t*    __restrict__ dst;
    int64_t step_stride;
    // Multi-CTA staging (populated only when config.should_global_reduce()).
    // cta_buf: num_outputs * ctas_per_output * sizeof(arg_t), uninitialised.
    // semaphores: num_outputs ints, zero-initialised by host before launch.
    void* cta_buf    = nullptr;
    int*  semaphores = nullptr;
};

} // namespace detail

namespace cuda {

// ═══════════════════════════════════════════════════════════
// SECTION 4: WARP / BLOCK REDUCE STAGES
// ═══════════════════════════════════════════════════════════

// Block-X reduce: all threads in a row collaborate to reduce input
// Uses warp shuffles first (register), then shared memory for cross-warp.
template<typename arg_t, typename ops_t>
__device__ arg_t block_x_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    // Warp-level shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value = ops.combine(value, ops.warp_shfl_down(value, offset));

    // Write warp leader to shared memory
    arg_t* smem = reinterpret_cast<arg_t*>(shared_memory);
    if (lane == 0) smem[wid] = value;
    __syncthreads();

    // First warp reduces across all warp results
    if (wid == 0) {
        value = (threadIdx.x < blockDim.x / 32) ? smem[lane] : ops.identity();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            value = ops.combine(value, ops.warp_shfl_down(value, offset));
    }
    return value;
}

// Block-Y reduce: threads in the same column (same threadIdx.x) reduce
// across all Y rows using shared memory.
template<typename arg_t, typename ops_t>
__device__ arg_t block_y_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
    arg_t* smem = reinterpret_cast<arg_t*>(shared_memory);
    smem[threadIdx.x + threadIdx.y * blockDim.x] = value;
    __syncthreads();

    if (threadIdx.y == 0) {
        value = smem[threadIdx.x];
        for (int i = 1; i < blockDim.y; i++)
            value = ops.combine(value, smem[threadIdx.x + i * blockDim.x]);
    }
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
template<typename T, int N>
struct alignas(sizeof(T) * N) VecLoad {
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
// Capped at hardware limit to ensure valid occupancy and avoid warnings (sm_86: 1536 threads/SM).
template<int NT>
struct MinBlocksPerSM {
    static constexpr int requested = (NT >= 256) ? (2048 / NT) : 4;
    static constexpr int VALUE = GAU_MIN_BLOCKS_PER_SM(NT, requested);
};

template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t,
         detail::ReductionLayout::Path PATH, int NT, int VT0 = 4>
__launch_bounds__(NT, MinBlocksPerSM<NT>::VALUE)
__global__ void unified_reduce_kernel(detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op) {
    extern __shared__ char shared_memory[];

    // accumulator type is whatever the functor's identity() returns
    using arg_t = decltype(op.ops.identity());
    const auto& config = op.config;

    // ── 1. Map this thread to an output index ──
    index_t output_idx =
        (index_t)threadIdx.x * config.output_mult[detail::GpuReduceConfig::BLOCK_X] +
        (index_t)threadIdx.y * config.output_mult[detail::GpuReduceConfig::BLOCK_Y] +
        (index_t)blockIdx.x  * config.step_output;

    if (output_idx >= (index_t)config.num_outputs) return;

    // ── 2. Map this thread to its starting input index ──
    // input_mult[CTA] term: when ctas_per_output > 1, each block along grid.y
    // owns a disjoint slice of the reduced axis. Without this term, all CTAs
    // for a given output_idx would walk the same range (the multi-CTA bug).
    index_t idx =
        (index_t)threadIdx.x * config.input_mult[detail::GpuReduceConfig::BLOCK_X] +
        (index_t)threadIdx.y * config.input_mult[detail::GpuReduceConfig::BLOCK_Y] +
        (index_t)blockIdx.y  * config.input_mult[detail::GpuReduceConfig::CTA];

    const index_t end    = (index_t)config.num_inputs;
    const index_t stride = (index_t)config.step_input;

    // ── 3. Thread-local reduction with ILP (VT0 independent accumulators) ──
    // Multiple accumulators hide the latency of dependent reduce operations.
    arg_t acc[VT0];
    #pragma unroll
    for (int i = 0; i < VT0; i++) acc[i] = op.ops.identity();

    if constexpr (PATH == detail::ReductionLayout::Path::InnerContiguous) {
        // TIER 1: Contiguous – stride=1, output × reduced_count laid out linearly.
        // Covers ~70% of DL workloads (e.g. sum(axis=-1) on row-major tensors).
        const scalar_t* __restrict__ base_ptr = op.src + output_idx * (index_t)config.num_inputs;

        // Vectorized main loop: load VT0 elements per iteration
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                acc[i] = op.ops.reduce(acc[i], base_ptr[idx + i * stride], (int64_t)(idx + i * stride));
            }
            idx += stride * VT0;
        }
        // Scalar tail
        while (idx < end) {
            acc[0] = op.ops.reduce(acc[0], base_ptr[idx], (int64_t)idx);
            idx += stride;
        }
    }
    else if constexpr (PATH == detail::ReductionLayout::Path::OuterContiguous) {
        // TIER 2: Single-stride – output dimension is innermost, reduced dim is outer.
        // Covers ~25% of DL workloads (e.g. sum(axis=0) on NCHW → C).
        const scalar_t* __restrict__ base_ptr = op.src + output_idx;
        const index_t row_stride = (index_t)op.step_stride;

        // ILP main loop: VT0 independent accumulators, strided access
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                acc[i] = op.ops.reduce(acc[i], base_ptr[(idx + i * stride) * row_stride], (int64_t)(idx + i * stride));
            }
            idx += stride * VT0;
        }
        // Scalar tail
        while (idx < end) {
            acc[0] = op.ops.reduce(acc[0], base_ptr[idx * row_stride], (int64_t)idx);
            idx += stride;
        }
    }
    else {
        // TIER 3: Generic fallback — uses OffsetCalculator (O(ndim) div/mod).
        // Covers ~5% of cases: non-standard axes, non-contiguous or permuted tensors.
        index_t out_base = (op.output_calc.dims > 0)
            ? op.output_calc.get(output_idx)
            : output_idx;

        // ILP main loop for generic path
        while (idx + (VT0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < VT0; i++) {
                index_t cur = idx + i * stride;
                index_t in_off = (op.input_calc.dims > 0) ? op.input_calc.get(cur) : cur;
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

    // ── 6a. Multi-CTA path: stage partial → atomic last-CTA detect → combine + store ──
    // Mirrors PyTorch's ReduceOp::global_reduce (aten/src/ATen/native/cuda/Reduce.cuh).
    // Layout (matches PT's staging_memory_offset when block_x_reduce is active):
    //   cta_buf slot      = blockIdx.y + blockIdx.x * gridDim.y     (size: gridDim.x * gridDim.y)
    //   semaphores slot   = blockIdx.x                              (size: gridDim.x, zero-init)
    if (config.should_global_reduce()) {
        arg_t* cta_buf = reinterpret_cast<arg_t*>(op.cta_buf);

        // (1) per-CTA leader writes its partial into the staging buffer
        if (is_leader) {
            index_t slot = (index_t)blockIdx.y + (index_t)blockIdx.x * (index_t)gridDim.y;
            cta_buf[slot] = value;
        }
        __threadfence();   // partial must be globally visible before semaphore bump
        __syncthreads();

        // (2) last-CTA detection: one atomicAdd per output-column tile (blockIdx.x)
        __shared__ bool s_is_last;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            int prev = atomicAdd(&op.semaphores[blockIdx.x], 1);
            s_is_last = (prev == config.ctas_per_output - 1);
        }
        __syncthreads();
        if (!s_is_last) return;

        // (3) last CTA reduces the cpo partials sequentially and writes dst.
        // One thread does the combine — cpo is small (typically <= 64).
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            __threadfence();   // acquire pattern after the atomic
            arg_t combined = op.ops.identity();
            for (int i = 0; i < config.ctas_per_output; i++) {
                index_t slot = (index_t)i + (index_t)blockIdx.x * (index_t)gridDim.y;
                combined = op.ops.combine(combined, cta_buf[slot]);
            }
            auto final_val = op.ops.project(combined);
            if constexpr (std::is_same_v<out_scalar_t, __half>)
                op.dst[output_idx] = from_float<out_scalar_t>(static_cast<float>(final_val));
            else
                op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
        }
        return;
    }

    // ── 6b. Single-CTA path (unchanged): leader writes dst directly ──
    if (is_leader) {
        auto final_val = op.ops.project(value);

        if constexpr (std::is_same_v<out_scalar_t, __half>)
            op.dst[output_idx] = from_float<out_scalar_t>(static_cast<float>(final_val));
        else
            op.dst[output_idx] = static_cast<out_scalar_t>(final_val);
    }
}

// ═══════════════════════════════════════════════════════════
// SECTION 6: HOST-SIDE KERNEL LAUNCHER
// Resolves Path at compile time and dispatches to the correct
// kernel specialization. Packs args into ReduceOp struct.
// ═══════════════════════════════════════════════════════════

template<typename scalar_t, typename out_scalar_t, typename ops_t, typename index_t = uint32_t, int VT0 = 4>
inline void launch_reduce_kernel(
    const ops_t& ops,
    const detail::GpuReduceConfig& config,
    const detail::OffsetCalculator<1, index_t>& input_calc,
    const detail::OffsetCalculator<1, index_t>& output_calc,
    const scalar_t* src,
    out_scalar_t* dst,
    detail::ReductionLayout::Path path,
    int64_t step_stride,
    int smem,
    cudaStream_t stream)
{
    // Pack into single struct for minimal launch parameter overhead
    detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
    op.ops = ops;
    op.config = config;
    op.input_calc = input_calc;
    op.output_calc = output_calc;
    op.src = src;
    op.dst = dst;
    op.step_stride = step_stride;

    // Multi-CTA staging: allocate cta_buf (uninit) and semaphores (zero-init).
    // Sized exactly like PyTorch's: num_outputs*ctas_per_output accumulators and
    // num_outputs ints. Uses the caching allocator so repeated launches reuse
    // the same blocks — no per-call cudaMalloc/cudaFree.
    using arg_t = decltype(ops.identity());
    Allocator* scratch = nullptr;
    if (config.should_global_reduce()) {
        scratch = AllocatorRegistry::get_caching_allocator();
        const dim3 g = config.grid();
        size_t cta_bytes = sizeof(arg_t) * static_cast<size_t>(g.x) * static_cast<size_t>(g.y);
        size_t sem_bytes = sizeof(int)   * static_cast<size_t>(g.x);
        op.cta_buf    = scratch->allocate(cta_bytes);
        op.semaphores = reinterpret_cast<int*>(scratch->allocate(sem_bytes));
        cudaMemsetAsync(op.semaphores, 0, sem_bytes, stream);
    }

    const int nt = config.num_threads;

    // Dispatch on Path at compile time + num_threads for launch_bounds.
    // Using a lambda to avoid duplicating the grid/block/smem logic.
    #define LAUNCH_KERNEL(PATH_ENUM, NT_VAL) \
        unified_reduce_kernel<scalar_t, out_scalar_t, ops_t, index_t, PATH_ENUM, NT_VAL, VT0> \
            <<<config.grid(), config.block(), smem, stream>>>(op)

    if (path == detail::ReductionLayout::Path::InnerContiguous) {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::InnerContiguous, 512); }
    }
    else if (path == detail::ReductionLayout::Path::OuterContiguous) {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::OuterContiguous, 512); }
    }
    else {
        if      (nt <= 32)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 32);  }
        else if (nt <= 64)  { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 64);  }
        else if (nt <= 128) { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 128); }
        else if (nt <= 256) { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 256); }
        else                { LAUNCH_KERNEL(detail::ReductionLayout::Path::Generic, 512); }
    }

    #undef LAUNCH_KERNEL

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
