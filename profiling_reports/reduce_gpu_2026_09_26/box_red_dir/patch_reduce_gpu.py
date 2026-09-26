#!/usr/bin/env python3
"""GPU reduction fixes (Tensor-Implementations dist_side_merge 7810aea).

  Bug 1  dispatch_mean_gpu sized the config / shared memory with the INPUT type, not the
         accumulator -> shared-memory overflow -> crash (bf16/fp16/integer means).
  Bug 2  block_x_reduce treated the 2-D block as 1-D -> rows overwrote each other in shared
         memory and the warp shuffle summed across rows -> WRONG last-axis / middle-axis results
         for every dtype and op (sum, mean, max, ...).
  (Not changed: block_y_reduce has no trailing barrier. In the multi-CTA combine it is followed
  by block_x_reduce, whose only pre-barrier shared write stores the SAME value to the SAME slot
  each thread already wrote, so no reader can see a different value; PyTorch's global_reduce,
  Reduce.cuh:868-870, has the same sequence. The June copy needed a barrier only for its vec4
  kernel, which this tree does not have.)
  Bug 4  every INTEGER mean on the GPU returned 0: the 1/n factor was stored in the int64
         accumulator type, and static_cast<int64_t>(1.0 / n) == 0 for n >= 2.
         The integer mean now also returns Float64, like the CPU mean and NumPy (was Float32).

Run from the Tensor-Implementations root:  python3 patch_reduce_gpu.py
Refuses to write anything unless both files have the expected md5 and every pattern matches once.
"""
import hashlib, sys

EXPECT = {
    "src/UnaryOps/cuda/ReductionImplGPU.cu":   "babaf273967919a74b61b3109f2bcf38",
    "include/ops/helpers/ReductionKernels.cuh": "f4735ea2a02b1d80f0f54e629b50484c",
    "include/ops/helpers/ReductionOps.h":       "78d059a608adf95cdf38179b8ed5de3c",
}

EDITS = {
"src/UnaryOps/cuda/ReductionImplGPU.cu": [
("""    // Mean output is always float32 for integer inputs, or same as input for float
    Dtype output_dtype = (std::is_integral_v<T>) ? Dtype::Float32 : input.dtype();
""",
"""    // Modified: integer and bool input now return Float64 (it was Float32), the same dtype as
    // the CPU mean (ReductionImpl.h reduce_kernel_mean: "returning fp64") and as NumPy
    // (_methods.py _mean: int -> float64). With the MeanOps fix below the values are also the
    // CPU's, bit for bit: exact int64 sum, then double(sum) * (1.0 / n), stored with no second
    // rounding. Only the final store changes (8 bytes per output instead of 4); reading and
    // summing the input are the same. Float input keeps its own dtype, as before.
    Dtype output_dtype = (std::is_integral_v<T>) ? Dtype::Float64 : input.dtype();
"""),
("""    using OutT  = std::conditional_t<std::is_integral_v<T>, float, T>;
""",
"""    using OutT  = std::conditional_t<std::is_integral_v<T>, double, T>;
"""),
("""    using AccT  = detail::AccumulatorType<CudaT>;
    double inv_count = 1.0 / static_cast<double>(calculate_reduced_count(input.shape().dims, normalized_axes));
    AccT factor;
    if constexpr (std::is_same_v<CudaT, complex32_t> || std::is_same_v<CudaT, complex64_t> || std::is_same_v<CudaT, complex128_t>) {
        factor = AccT(inv_count, 0.0);
    } else {
        factor = static_cast<AccT>(inv_count);
    }
    using ops_t = detail::MeanOps<CudaT>;
    ops_t ops{factor};
""",
"""    using ops_t = detail::MeanOps<CudaT>;
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
"""),
("""    int num_mp, max_tpm; get_device_props(num_mp, max_tpm);
    auto config = detail::build_reduce_config<CudaT>(layout, num_mp, max_tpm);
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.inner_count : 1;
    int smem = std::max(config.shared_memory_size(), (config.block_width * config.block_height / 32 + 1) * (int)sizeof(CudaT));
""",
"""    // Modified: the config and the shared memory are now sized with the ACCUMULATOR type
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
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    int64_t step_stride = (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.inner_count : 1;
    int smem = std::max(config.shared_memory_size(), (config.block_width * config.block_height / 32 + 1) * (int)sizeof(arg_t));
"""),
],
"include/ops/helpers/ReductionKernels.cuh": [
("""__device__ arg_t block_x_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
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
""",
"""__device__ arg_t block_x_reduce(arg_t value, const ops_t& ops, char* shared_memory) {
    // Modified: rewritten for the 2-D block. Same algorithm as PyTorch's block_x_reduce
    // (aten/src/ATen/native/cuda/Reduce.cuh, checked against the source at 5f76830b728,
    // 2025-12-05), including the decreasing shuffle offsets PyTorch uses on CUDA. threadIdx.x walks the reduced dim; threadIdx.y indexes INDEPENDENT
    // outputs, so the reduction must stay inside one threadIdx.y row.
    // The old body treated the block as 1-D, which is wrong whenever block_height > 1:
    //   - it wrote the warp partials to smem[threadIdx.x / 32], ignoring threadIdx.y, so every
    //     row wrote the same slots and the rows overwrote each other;
    //   - it always shuffled over the full 32 lanes, so with block_width < 32 (several rows in
    //     one warp) it added neighbouring rows' values together;
    //   - its second stage read only blockDim.x / 32 slots, i.e. whichever row wrote last.
    // Measured before this fix on the H200 (outputs wrong by more than 1 ulp):
    //   last axis [512, 4096] (block 32 x 16): mean fp32 480 / 512, sum fp32 479 / 512,
    //     max fp32 476 / 512, mean fp64 480 / 512;
    //   block_width 16 / 8 (two / four rows per warp): sum [4096, 16] and max [4096, 8]
    //     4096 / 4096 each;
    //   block_width > 32: sum [2, 65536] 2 / 2; multi-CTA sum [8, 1 << 20] 8 / 8;
    //   middle axis: sum fp32 [7, 300, 5] 30 / 35.
    // Reductions without block_x_reduce (axis 0 and full reductions, block_y only) were right.
    // Shared memory: only when a row spans more than one warp, indexed by the full 2-D thread
    // id, num_threads slots -- exactly config.shared_memory_size(), which is 0 when
    // block_width <= 32 and block_y is off (then no shared memory is touched here).
    // block_width is a power of two (set_block_dimension uses last_pow2), so the halving loop
    // below covers every lane.
    int dim_x = blockDim.x;
    arg_t* smem = reinterpret_cast<arg_t*>(shared_memory);

    // Cross-warp stage -- only when a row spans more than one warp.
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
    // Intra-warp shuffle bounded by the row width: with block_width < 32 one warp holds several
    // rows, so the offsets stay below dim_x and no row adds in its neighbour. The result lands
    // in threadIdx.x == 0 of each row, which is the is_leader thread the caller writes from.
    for (int offset = dim_x >> 1; offset > 0; offset >>= 1)
        value = ops.combine(value, ops.warp_shfl_down(value, offset));
    return value;
}
"""),
],
"include/ops/helpers/ReductionOps.h": [
("""template <typename T>
struct MeanOps {
    using AccT = AccumulatorType<T>;
    AccT factor;  // = 1.0 / reduced_count (pre-computed on host)

    DEVICE_HOST MeanOps(AccT f = AccT(1.0f)) : factor(f) {}
""",
"""template <typename T>
struct MeanOps {
    using AccT = AccumulatorType<T>;
    // Modified: the factor type is separate from the accumulator. For an integer accumulator
    // (int8..int64, uint8..uint64, bool -> int64_t / uint64_t) the factor 1/n cannot be stored in
    // AccT: static_cast<int64_t>(1.0 / n) is 0 for n >= 2, so project() returned sum * 0 and every
    // integer mean on the GPU was 0. The sum stays EXACT in int64 / uint64; only the factor and
    // the result are double, and the kernel's output write converts that double once to the
    // output type (static_cast<out_scalar_t>). This is the same arithmetic as the CPU integer
    // mean (ReductionImpl.h reduce_kernel_mean: static_cast<double>(sum) * (1.0 / n)). For
    // float / half / complex accumulators FactorT == AccT, so those means are bit-identical.
    using FactorT = std::conditional_t<std::is_integral_v<AccT>, double, AccT>;
    FactorT factor;  // = 1.0 / reduced_count (pre-computed on host)

    DEVICE_HOST MeanOps(FactorT f = FactorT(1.0f)) : factor(f) {}
"""),
("""    // Project: multiply by 1/count to get mean
    DEVICE_HOST AccT project(const AccT& a) const {
        #ifdef __CUDA_ARCH__
        return gpu_mul(a, factor);
        #else
        return a * factor;
        #endif
    }
""",
"""    // Project: multiply by 1/count to get mean
    // Modified: integer accumulators return double (see FactorT); every other type is unchanged.
    // The callers take the result with `auto final_val = op.ops.project(value)`.
    DEVICE_HOST auto project(const AccT& a) const {
        if constexpr (std::is_integral_v<AccT>) {
            return static_cast<double>(a) * factor;
        } else {
        #ifdef __CUDA_ARCH__
            return gpu_mul(a, factor);
        #else
            return a * factor;
        #endif
        }
    }
"""),
],
}

def main():
    out = {}
    for path, want in EXPECT.items():
        data = open(path, "rb").read()
        got = hashlib.md5(data).hexdigest()
        if got != want:
            sys.exit(f"ABORT: {path} md5 {got}, expected {want}. Nothing written.")
        text = data.decode()
        for i, (old, new) in enumerate(EDITS[path], 1):
            n = text.count(old)
            if n != 1:
                sys.exit(f"ABORT: {path} edit {i}: pattern found {n} times (need exactly 1). Nothing written.")
            text = text.replace(old, new)
        out[path] = text
    for path, text in out.items():
        open(path, "w").write(text)
        print(f"patched {path}: {len(EDITS[path])} edit(s)")
    print("done: 3 files, 7 edits")

if __name__ == "__main__":
    main()
