#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include <random>
#include <cstring>
#include "device/DeviceCore.h"
#include "device/DeviceTransfer.h"



#include <cuda_runtime.h>
#include <cuda.h>          // driver API: cuMemsetD8/D16/D32Async
#include <curand.h>
#include "ops/helpers/ConversionKernels.cuh"
#include "ops/helpers/FillKernels.h"

// Forward declaration of the fused batched-cat CUDA launcher
// (defined in src/Kernels/cuda/misc/BatchedCat.cu).
extern "C" void cat_batched_cuda(
    void* output_ptr,
    int32_t elem_size,
    int64_t total_dim_size,
    int64_t inner_size,
    const void* const* input_ptrs,
    const int64_t* input_nelements,
    const int64_t* input_dim_sizes,
    const int64_t* input_d_offsets,
    const int64_t* input_outer_strides,
    const int64_t* input_dim_strides,
    const int32_t* input_is_contig,
    int64_t num_inputs,
    cudaStream_t stream);

namespace OwnTensor
{
    // ---------------------------------------------------------------------
    // fill_constant_gpu — edits: Gautam_Reddy_1926
    //
    // Single entry point used by ones()/full() on GPU. Dispatches by
    // sizeof(T) to the fastest correct API per the bench_fill_apis.cpp
    // measurements:
    //   1 B  → cudaMemsetAsync          (DMA engine)
    //   2 B  → cuMemsetD16Async          (driver, repeats 16-bit pattern)
    //   4 B  → cuMemsetD32Async          (driver, repeats 32-bit pattern)
    //   8+ B → fill_cuda_launch<T>       (no D64 driver API exists)
    //
    // The driver APIs accept ANY repeated pattern, not just 1 — so this
    // path handles full(value) too, not only ones().
    // ---------------------------------------------------------------------
    template <typename T>
    static void fill_constant_gpu(T* ptr, T value, int64_t numel, cudaStream_t stream)
    {
        const size_t nbytes = static_cast<size_t>(numel) * sizeof(T);
        const CUdeviceptr dp = reinterpret_cast<CUdeviceptr>(ptr);
        const CUstream    cs = reinterpret_cast<CUstream>(stream);

        if constexpr (sizeof(T) == 1) {
            uint8_t pat;
            std::memcpy(&pat, &value, 1);
            cudaError_t err = cudaMemsetAsync(ptr, pat, nbytes, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("fill_constant_gpu cudaMemsetAsync: ") +
                                         cudaGetErrorString(err));
            }
        } else if constexpr (sizeof(T) == 2) {
            uint16_t pat;
            std::memcpy(&pat, &value, 2);
            CUresult r = cuMemsetD16Async(dp, pat, numel, cs);
            if (r != CUDA_SUCCESS) {
                const char* s = nullptr; cuGetErrorString(r, &s);
                throw std::runtime_error(std::string("fill_constant_gpu cuMemsetD16Async: ") +
                                         (s ? s : "?"));
            }
        } else if constexpr (sizeof(T) == 4) {
            uint32_t pat;
            std::memcpy(&pat, &value, 4);
            CUresult r = cuMemsetD32Async(dp, pat, numel, cs);
            if (r != CUDA_SUCCESS) {
                const char* s = nullptr; cuGetErrorString(r, &s);
                throw std::runtime_error(std::string("fill_constant_gpu cuMemsetD32Async: ") +
                                         (s ? s : "?"));
            }
        } else {
            // 8 B (double, int64) and any wider type → kernel fallback.
            OwnTensor::cuda::fill_cuda_launch<T>(ptr, value, numel, stream);
        }
    }

    // Helper for CUDA RNG
    void cuda_rand_uniform(float* data, size_t count, unsigned long seed, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateUniform(gen, data, count);
        curandDestroyGenerator(gen);
    }

    void cuda_rand_uniform(double* data, size_t count, unsigned long seed, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateUniformDouble(gen, data, count);
        curandDestroyGenerator(gen);
    }

    void cuda_rand_normal(float* data, size_t count, unsigned long seed, float sd, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateNormal(gen, data, count, 0.0f, float(sd));
        curandDestroyGenerator(gen);
    }

    void cuda_rand_normal(double* data, size_t count, unsigned long seed, double sd, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateNormalDouble(gen, data, count, 0.0, sd);
        curandDestroyGenerator(gen);
    }


    // -----------------------------------------------------------------------
    // Tensor::zeros — edits: Gautam_Reddy_1926
    //
    // Both CPU and GPU paths are now memset-based. Rationale below.
    //
    // CPU path: std::memset(ptr, 0, nbytes)
    //   - Zero is the one case where a byte-wise memset is correct for every
    //     numeric dtype: bit-pattern 0 == numeric 0 for float/double/half/
    //     bf16/int*/complex/bool. No per-dtype dispatch needed.
    //   - glibc's memset is SIMD-tuned (AVX2/AVX-512 internally) and beats
    //     our per-element fill loop by 2–3x on large buffers.
    //   - We previously had a custom fill_cpu_simd path with OpenMP. OpenMP
    //     was REMOVED because GCC's libgomp pays ~6–7 ms of per-region setup
    //     cost on the first parallel section per thread team, which dwarfs
    //     the actual fill cost. PyTorch uses TBB-backed at::parallel_for
    //     (~100 ns dispatch), so they get parallelism cheaply; we don't
    //    TO-DO (Gautam_Reddy_1926) : investigate if TBB can be used here and-
    //    c-implementation of openmp in cpu path of fill function ,
    //   - The fallback fill path (kept for non-zero values in ones()/full())
    //     uses manual AVX2 (_mm256_storeu_si256, 32 B/store) and AVX-512
    //     (_mm512_storeu_si512, 64 B/store) intrinsics with a scalar fallback.
    //     See detail::fill_cpu_simd<T> in include/core/TensorDataManip.h.
    //
    // GPU path: cudaMemsetAsync(ptr, 0, nbytes, stream)
    //   - Driver schedules this on the copy/DMA engine: NO SM occupancy,
    //     overlaps with concurrent kernels on the compute engine.
    //   - For non-zero patterns we still need our fill_cuda_launch<T> kernel
    //     (vectorized uint4/uint2 stores; see src/Kernels/cuda/misc/FillKernel.cu).
    //   - Loud-fail on error — no silent fallback to a slow path.
    //
    // TODO (Gautam_Reddy_1926): research whether the CUDA Driver APIs
    //   cuMemsetD8Async / cuMemsetD16Async / cuMemsetD32Async beat the runtime
    //   cudaMemsetAsync for the zero case (they shouldn't — both go through
    //   the same DMA engine — but worth measuring). See bench_fill_apis.cpp.
    //
    // vs PyTorch:
    //   at::zeros() = at::empty() + tensor.zero_(). zero_() has a contiguous
    //   fast path that calls cudaMemsetAsync (aten/src/ATen/native/cuda/
    //   Resize.cpp); strided tensors fall through to TensorIterator + a CUDA
    //   functor. We only need the contiguous path because zeros() always
    //   creates a fresh contiguous tensor. Same end result.
    // -----------------------------------------------------------------------
    Tensor Tensor::zeros(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            std::memset(tensor.data(), 0, tensor.nbytes());
        }
        else
        {
#ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cudaError_t err = cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("Tensor::zeros cudaMemsetAsync: ") +
                                         cudaGetErrorString(err));
            }
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::empty(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

//         if (opts.device.is_cpu())
//         {
//             // CPU implementation - handles all 7 types automatically
//             dispatch_by_dtype(opts.dtype, [&](auto [[maybe_unused]] dummy)
//                 {
//                     // using T = decltype(dummy);
//                     // tensor.fill(T(0.0f));
//                 });
//         }
//         else
//         {
//             // GPU implementation - optimized with cudaMemset
// #ifdef WITH_CUDA
//             // cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
//             // cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
// #else
//             throw std::runtime_error("CUDA not available");
// #endif
//         }
        return tensor;
    }

    // -----------------------------------------------------------------------
    // Tensor::ones — edits: Gautam_Reddy_1926
    //
    // WHY THIS REWRITE EXISTS:
    //   ones() used to be a slow toy path. On GPU it built a host vector of
    //   numel*sizeof(T), filled it on CPU, then H->D memcpy'd it across PCIe
    //   every single call — that's a host alloc + host fill + PCIe transfer
    //   on the training critical path. On CPU it was a plain scalar loop
    //   (`for i: ptr[i] = 1`), no SIMD, no parallelism. We now match the
    //   "fast and right" approach that PyTorch uses, and in many cases beat
    //   it by going through the CUDA Driver memset APIs.
    //
    // CPU path (NOW):
    //   tensor.fill<T> routes (via traits in TensorDataManip.h) to
    //   detail::fill_cpu_simd<T>:
    //     - AVX-512 (_mm512_storeu_si512, 64 B/store) when __AVX512F__ set
    //     - AVX2    (_mm256_storeu_si256, 32 B/store) otherwise
    //     - scalar tail for the leftover elements
    //   OpenMP is intentionally omitted (libgomp's per-region setup cost
    //   dwarfs the fill itself — see the zeros() comment block).
    //
    // GPU path (NOW): per-sizeof(T) dispatch to the API that bench_fill_apis.cpp
    //   measured as fastest-AND-correct for that element width:
    //     - bool                    → cudaMemsetAsync(0x01)        (byte-backed)
    //     - 1 B int8/uint8          → cudaMemsetAsync(byte pattern of 1)
    //     - 2 B half/bf16/int16     → cuMemsetD16Async(bit_rep_of_1)
    //     - 4 B float/int32         → cuMemsetD32Async(0x3F800000 / 0x1)
    //     - 8 B double/int64        → fill_cuda_launch<T>           (no D64 API)
    //   The driver memsets run on the DMA engine — zero SM occupancy, free
    //   to overlap with concurrent compute kernels. They beat our custom
    //   fill kernel by ~5–10% on the dtypes where they're applicable.
    //
    //   IMPORTANT: cudaMemsetAsync only writes a byte pattern, so it CANNOT
    //   produce a correct float-1.0 (0x3F800000) by itself — that's why we
    //   use the wider driver APIs. The byte/word/dword variants exist
    //   precisely for this multi-byte case.
    //
    //   No host-side allocation. No PCIe crossing. No host->device copy.
    //
    // vs PyTorch:
    //   at::ones()  = at::empty() + tensor.fill_(1).
    //   at::full()  = at::empty() + tensor.fill_(value).
    //   PyTorch's fill_() ALWAYS goes through TensorIterator + a CUDA
    //   functor (gpu_kernel(iter, FillFunctor<T>{value}), see
    //   aten/src/ATen/native/cuda/Fill.cu) — i.e. a kernel launch. They
    //   never use cuMemsetD16/D32Async because TensorIterator must also
    //   handle non-contiguous / strided / broadcast cases through the same
    //   path. We split the responsibility:
    //     factories (zeros/ones/full) → driver memsets    [contiguous only]
    //     in-place fill / fill_grad   → fill_cuda_launch  [handles strides]
    //   Net effect: factories beat PyTorch by ~5–10% on 2 B / 4 B dtypes
    //   while in-place fill matches PyTorch's TensorIterator-functor approach.
    // -----------------------------------------------------------------------
    Tensor Tensor::ones(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // Special handling for bool: ones = true
                        tensor.fill(true);
                    }
                    else
                    {
                        tensor.fill(T(1.0f));
                    }
                });
        }
        else
        {
            // GPU implementation — native fill kernel. Replaces the old
            // CPU-vector + H->D memcpy path (which alloc'd numel*sizeof(T) of
            // host memory and paid a PCIe crossing every call).
#ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // Bool is byte-backed; bit pattern 0x01 == true.
                        cudaError_t err = cudaMemsetAsync(tensor.data(), 1, tensor.numel(), stream);
                        if (err != cudaSuccess) {
                            throw std::runtime_error(std::string("Tensor::ones cudaMemsetAsync: ") +
                                                     cudaGetErrorString(err));
                        }
                    }
                    else
                    {
                        // Per-sizeof dispatch: cuMemsetD16/D32Async for 2/4 B,
                        // fill kernel fallback for 8 B (no D64 API exists).
                        fill_constant_gpu<T>(
                            reinterpret_cast<T*>(tensor.data()),
                            T(1.0f),
                            static_cast<int64_t>(tensor.numel()),
                            stream);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    // -----------------------------------------------------------------------
    // Tensor::full — edits: Gautam_Reddy_1926
    //
    // WHY: full(value) is the generalised form of ones() — same hot path,
    //   same problem (was H->D copy of a host-filled buffer). The fix is
    //   the same per-sizeof(T) dispatch. The driver memset APIs accept ANY
    //   repeated 1/2/4-byte pattern, not just 0 or 1, so the bit-rep of
    //   `value` (cast to T) goes straight into cuMemsetD16/D32Async.
    //   Examples: full(3.14f) → cuMemsetD32Async(0x4048F5C3), full(half(2.5))
    //   → cuMemsetD16Async(0x4100). No SMs used, DMA engine only.
    //
    //   The 8 B path (double, int64) still falls back to fill_cuda_launch<T>
    //   because the CUDA driver does not expose a cuMemsetD64Async. That
    //   fallback is the same uint4 STG.128 vectorized kernel used elsewhere.
    //
    //   bool is special: byte-backed, any nonzero value collapses to true,
    //   so cudaMemsetAsync with 0 or 1 is exactly correct.
    //
    // vs PyTorch:
    //   at::full() = at::empty() + tensor.fill_(value). PyTorch's fill_()
    //   always uses a TensorIterator + CUDA functor kernel (Fill.cu —
    //   gpu_kernel(iter, FillFunctor<T>{value})), never cuMemsetD*Async,
    //   because TensorIterator must support strided/broadcast cases
    //   uniformly. We diverge at the factory layer: full() is guaranteed
    //   to allocate a fresh contiguous tensor, so we can safely use the
    //   driver memset shortcut here. In-place fill() still matches their
    //   approach (kernel-based, see Tensor::fill in TensorDataManip.h).
    // -----------------------------------------------------------------------
    Tensor Tensor::full(Shape shape, TensorOptions opts, float value)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // For bool: any nonzero value = true
                        tensor.fill(value != 0.0f);
                    }
                    else
                    {
                        tensor.fill(static_cast<T>(value));
                    }
                });
        }
        else
        {
#ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        uint8_t bool_val = (value != 0.0f) ? 1 : 0;
                        cudaError_t err = cudaMemsetAsync(tensor.data(), bool_val, tensor.numel(), stream);
                        if (err != cudaSuccess) {
                            throw std::runtime_error(std::string("Tensor::full cudaMemsetAsync: ") +
                                                     cudaGetErrorString(err));
                        }
                    }
                    else
                    {
                        // Per-sizeof dispatch (Gautam_Reddy_1926):
                        //   1 B → cudaMemsetAsync, 2 B → cuMemsetD16Async,
                        //   4 B → cuMemsetD32Async, 8 B → fill_cuda_launch.
                        // Driver memsets accept ANY repeated pattern, not
                        // just 0/1 — so full(value) benefits same as ones().
                        fill_constant_gpu<T>(
                            reinterpret_cast<T*>(tensor.data()),
                            static_cast<T>(value),
                            static_cast<int64_t>(tensor.numel()),
                            stream);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::rand(Shape shape, TensorOptions opts,unsigned long seed, U lower, U upper)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            //std::random_device rd;
            std::mt19937 gen(seed);

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::uniform_real_distribution<T> dist(lower, upper);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("rand only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            // std::random_device rd;
            // unsigned long seed = rd();
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_uniform(static_cast<float*>(tensor.data()), tensor.numel(), seed, stream);
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_uniform(static_cast<double*>(tensor.data()), tensor.numel(), seed, stream);
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_uniform(temp_data, tensor.numel(), seed, stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU rand only supports float/double/half/bfloat16");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::randn(Shape shape, TensorOptions opts,unsigned long seed , U sd)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            //std::random_device rd;
            std::mt19937 gen(seed);

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::normal_distribution<T> dist(0.0, sd);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::normal_distribution<float> dist(0.0f, float(sd));
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("randn only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            //std::random_device rd;
            //unsigned long seed = rd();
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_normal(static_cast<float*>(tensor.data()), tensor.numel(), seed, sd, stream);//✨✨✨
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_normal(static_cast<double*>(tensor.data()), tensor.numel(), seed, sd, stream);//✨✨✨
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_normal(temp_data, tensor.numel(), seed, float(sd), stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU randn only supports float/double");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template Tensor Tensor::rand<float>(Shape shape, TensorOptions opts,unsigned long seed, float lower, float upper);
    template Tensor Tensor::rand<double>(Shape shape, TensorOptions opts,unsigned long seed, double lower, double upper);

    template Tensor Tensor::randn<float>(Shape shape, TensorOptions opts,unsigned long seed, float sd);
    template Tensor Tensor::randn<double>(Shape shape, TensorOptions opts,unsigned long seed, double sd);

    // ======================================================================
    // multinomial — sample indices from a probability distribution
// ======================================================================
    Tensor Tensor::multinomial(const Tensor& input, int64_t num_samples,
                               bool replacement, unsigned long seed)
    {
        // If seed is 0 (default), use random_device for non-deterministic sampling.
        // Otherwise use the caller-provided seed (e.g. 42 + ddp_rank for DDP).
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        // --- Validation ---
        const auto& sh = input.shape();
        if (sh.dims.size() != 1 && sh.dims.size() != 2) {
            throw std::runtime_error("multinomial: input must be 1-D or 2-D tensor");
        }

        bool is_1d   = (sh.dims.size() == 1);
        int64_t nrows = is_1d ? 1 : sh.dims[0];
        int64_t ncols = is_1d ? sh.dims[0] : sh.dims[1];

        if (num_samples <= 0) {
            throw std::runtime_error("multinomial: num_samples must be > 0");
        }

        // --- Bring input to CPU as float for sampling ---
        Tensor cpu_input = input;
        if (cpu_input.device().is_cuda()) {
            cpu_input = cpu_input.to_cpu();
        }
        if (cpu_input.dtype() != Dtype::Float32 && cpu_input.dtype() != Dtype::Float64) {
            cpu_input = cpu_input.as_type(Dtype::Float32);
        }

        // --- Allocate output on CPU (Int64) ---
        Shape out_shape = is_1d ? Shape{{num_samples}} : Shape{{nrows, num_samples}};
        Tensor output(out_shape, Dtype::Int64, DeviceIndex(Device::CPU), false);
        int64_t* out_ptr = output.data<int64_t>();

        // --- Sample per row ---
        std::mt19937 gen(seed);

        for (int64_t row = 0; row < nrows; ++row) {
            // Build weight vector for this row
            std::vector<double> weights(ncols);

            dispatch_by_dtype(cpu_input.dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                if constexpr (std::is_floating_point_v<T>) {
                    const T* row_data = cpu_input.data<T>() + row * ncols;
                    for (int64_t c = 0; c < ncols; ++c) {
                        double w = static_cast<double>(row_data[c]);
                        if (w < 0.0 || !std::isfinite(w)) {
                            throw std::runtime_error(
                                "multinomial: input must be non-negative and finite");
                        }
                        weights[c] = w;
                    }
                } else {
                    throw std::runtime_error(
                        "multinomial: input must be a floating-point tensor");
                }
            });

            // Check non-zero sum
            double total = 0.0;
            int64_t non_zero_count = 0;
            for (int64_t c = 0; c < ncols; ++c) {
                total += weights[c];
                if (weights[c] > 0.0) ++non_zero_count;
            }
            if (total == 0.0) {
                throw std::runtime_error(
                    "multinomial: rows must have a non-zero sum");
            }

            if (!replacement && num_samples > non_zero_count) {
                throw std::runtime_error(
                    "multinomial: cannot sample " + std::to_string(num_samples) +
                    " without replacement from row with " +
                    std::to_string(non_zero_count) + " non-zero elements");
            }

            // Sample indices
            int64_t* row_out = out_ptr + row * num_samples;

            if (replacement) {
                // With replacement: single distribution, draw num_samples times
                std::discrete_distribution<int64_t> dist(weights.begin(), weights.end());
                for (int64_t s = 0; s < num_samples; ++s) {
                    row_out[s] = dist(gen);
                }
            } else {
                // Without replacement: draw one, zero-out, rebuild distribution
                std::vector<double> w_copy = weights;
                for (int64_t s = 0; s < num_samples; ++s) {
                    std::discrete_distribution<int64_t> dist(w_copy.begin(), w_copy.end());
                    int64_t idx = dist(gen);
                    row_out[s] = idx;
                    w_copy[idx] = 0.0;  // prevent re-sampling
                }
            }
        }

        // --- Move output to same device as input ---
        if (input.device().is_cuda()) {
#ifdef WITH_CUDA
            output = output.to_cuda(input.device().index);
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return output;
    }

    Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
        if (tensors.empty()) {
            throw std::runtime_error("Tensor::cat expects a non-empty list of tensors");
        }

        // 1. Validate inputs and calculate output shape
        const Tensor& t0 = tensors[0];
        int64_t ndim = t0.ndim();

        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Tensor::cat: invalid dimension " + std::to_string(dim));
        }

        Shape out_shape = t0.shape();
        int64_t total_dim_size = 0;

        for (const auto& t : tensors) {
            if (t.ndim() != ndim) {
                throw std::runtime_error("Tensor::cat: all tensors must have same number of dimensions");
            }
            if (t.dtype() != t0.dtype()) {
                throw std::runtime_error("Tensor::cat: all tensors must have same dtype");
            }
            if (t.device().device != t0.device().device || t.device().index != t0.device().index) {
                // strict device check for now
                throw std::runtime_error("Tensor::cat: all tensors must be on same device");
            }

            for (int64_t i = 0; i < ndim; ++i) {
                if (i != dim && t.shape().dims[i] != t0.shape().dims[i]) {
                    throw std::runtime_error("Tensor::cat: sizes do not match except at dimension " + std::to_string(dim));
                }
            }
            total_dim_size += t.shape().dims[dim];
        }

        out_shape.dims[dim] = total_dim_size;

        // 2. Allocate output tensor
        Tensor result(out_shape, t0.dtype(), t0.device(), t0.requires_grad());

        // 3. Copy data
        // Optimization: If dim=0 and all inputs contiguous, simple memcpy.
        bool all_contiguous = true;
        for(const auto& t : tensors) if(!t.is_contiguous()) all_contiguous = false;

        size_t offset_bytes = 0;
        size_t element_size = t0.dtype_size(t0.dtype());

        if (dim == 0 && all_contiguous) {
             uint8_t* out_ptr = static_cast<uint8_t*>(result.data());

             for (const auto& t : tensors) {
                 size_t bytes = t.numel() * element_size;
                 device::copy_memory(out_ptr + offset_bytes, result.device().device,
                                     t.data(), t.device().device,
                                     bytes);
                 offset_bytes += bytes;
             }
        } else {
            // General case: cat along arbitrary dim. Compute the flattened
            // 3D view [outer_size, dim_size, inner_size] of the output,
            // where outer_size = prod(dims[0..dim-1]) and
            // inner_size = prod(dims[dim+1..]).
            int64_t prob_outer_size = 1;
            for(int64_t i=0; i<dim; ++i) prob_outer_size *= out_shape.dims[i];

            int64_t prob_inner_size = 1;
            for(int64_t i=dim+1; i<ndim; ++i) prob_inner_size *= out_shape.dims[i];

#ifdef WITH_CUDA
            if (result.device().is_cuda()) {
                // Fused batched cat kernel (PyTorch-style) — ONE launch for
                // all inputs, supports strided sources, eliminates the
                // per-input .contiguous() + cudaMemcpy2DAsync loop.
                std::vector<const void*> in_ptrs;
                std::vector<int64_t>     in_nelem;
                std::vector<int64_t>     in_dim_sizes;
                std::vector<int64_t>     in_d_offsets;
                std::vector<int64_t>     in_outer_strides;
                std::vector<int64_t>     in_dim_strides;
                std::vector<int32_t>     in_is_contig;
                in_ptrs.reserve(tensors.size());
                in_nelem.reserve(tensors.size());
                in_dim_sizes.reserve(tensors.size());
                in_d_offsets.reserve(tensors.size());
                in_outer_strides.reserve(tensors.size());
                in_dim_strides.reserve(tensors.size());
                in_is_contig.reserve(tensors.size());

                int64_t dim_offset = 0;
                for (const auto& t : tensors) {
                    const int64_t dim_size = t.shape().dims[dim];
                    const int64_t nelem    = t.numel();
                    const bool contig      = t.is_contiguous() && t.storage_offset() == 0;

                    in_ptrs.push_back(t.data());
                    in_nelem.push_back(nelem);
                    in_dim_sizes.push_back(dim_size);
                    in_d_offsets.push_back(dim_offset);

                    // Strides (in ELEMENTS, not bytes) for the 3D-flattened
                    // view. For a contiguous tensor these are unused by the
                    // kernel's fast path.
                    const auto& s = t.stride().strides;
                    int64_t dim_stride   = (dim < (int64_t)s.size()) ? s[dim] : 1;
                    int64_t outer_stride = (dim > 0 && dim - 1 < (int64_t)s.size())
                                             ? s[dim - 1]
                                             : (dim_size * prob_inner_size);
                    in_outer_strides.push_back(outer_stride);
                    in_dim_strides.push_back(dim_stride);
                    in_is_contig.push_back(contig ? 1 : 0);

                    dim_offset += dim_size;
                }

                cudaStream_t stream = 0;
                cat_batched_cuda(
                    result.data(),
                    (int32_t)element_size,
                    total_dim_size,
                    prob_inner_size,
                    in_ptrs.data(),
                    in_nelem.data(),
                    in_dim_sizes.data(),
                    in_d_offsets.data(),
                    in_outer_strides.data(),
                    in_dim_strides.data(),
                    in_is_contig.data(),
                    (int64_t)tensors.size(),
                    stream);
            } else
#endif
            {
                // CPU fallback: per-tensor strided copy (original loop).
                size_t inner_bytes = prob_inner_size * element_size;
                size_t row_pitch_out = total_dim_size * inner_bytes;
                uint8_t* out_ptr = static_cast<uint8_t*>(result.data());
                int64_t dim_offset = 0;

                for (const auto& t : tensors) {
                    Tensor t_cont = t.contiguous();
                    const uint8_t* in_ptr = static_cast<const uint8_t*>(t_cont.data());
                    int64_t dim_size = t.shape().dims[dim];
                    size_t chunk_bytes = dim_size * inner_bytes;

                    for(int64_t i=0; i<prob_outer_size; ++i) {
                        size_t out_idx_bytes = i * row_pitch_out + (dim_offset * inner_bytes);
                        size_t in_idx_bytes  = i * chunk_bytes;
                        device::copy_memory(out_ptr + out_idx_bytes, result.device().device,
                                            in_ptr + in_idx_bytes, t_cont.device().device,
                                            chunk_bytes);
                    }
                    dim_offset += dim_size;
                }
            }
        }

        return result;
    }

}
