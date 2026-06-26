#pragma once

#ifndef TENSOR_DATAMANIP_H
#define TENSOR_DATAMANIP_H

#include "core/Tensor.h"
#include "device/DeviceTransfer.h"
#include <iostream>
#include <cstring>
#include <vector>
#include "dtype/DtypeTraits.h"
#include "dtype/fp4.h"
#ifdef WITH_CUDA
#include "ops/helpers/FillKernels.h"
#include "device/DeviceCore.h"
#include "dtype/fp4.h"
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace OwnTensor { namespace detail {
// Types that have an explicit instantiation of fill_cuda_launch<T> in FillKernel.cu.
// fp4 / bool fall back to the legacy CPU-vector path because packing fp4 into a
// uint4 would require quantization-aware handling, and bool has its own
// out-of-line specialization in Tensor.cpp.
template <typename T>
constexpr bool fill_cuda_supported =
    std::is_same_v<T, float>        || std::is_same_v<T, double>      ||
    std::is_same_v<T, float16_t>    || std::is_same_v<T, bfloat16_t>  ||
    std::is_same_v<T, int8_t>       || std::is_same_v<T, int16_t>     ||
    std::is_same_v<T, int32_t>      || std::is_same_v<T, int64_t>     ||
    std::is_same_v<T, uint8_t>      || std::is_same_v<T, uint16_t>    ||
    std::is_same_v<T, uint32_t>     || std::is_same_v<T, uint64_t>    ||
    std::is_same_v<T, complex32_t>  || std::is_same_v<T, complex64_t> ||
    std::is_same_v<T, complex128_t>;

// Types that benefit from the AVX2 + OpenMP CPU fill fast path. Requirements:
//   - trivially copyable (we bit-copy the value into the SIMD pattern)
//   - sizeof(T) divides 32 (fits cleanly in a __m256i)
// Everything in fill_cuda_supported meets both requirements, so we reuse it.
// bool / fp4 deliberately excluded (they have their own handling paths).
template <typename T>
constexpr bool fill_cpu_simd_supported = fill_cuda_supported<T>;

// AVX2 + OpenMP fill for trivially copyable types with sizeof(T) | 32.
// Threshold for OMP parallelism mirrors PyTorch's internal::GRAIN_SIZE.
// AVX2/AVX512 SIMD fill, single-threaded.
// OpenMP was tested and removed: GCC's per-parallel-region overhead was
// ~6-7 ms per call at our typical tensor sizes (> 1000x the actual work).
// PyTorch avoids this by backing at::parallel_for with TBB (sub-µs overhead);
// we'd need to link TBB to get the same benefit. Not worth it here since
// CPU fill isn't on any hot path of GPU training — GPU fill (FillKernel.cu)
// is what matters, and that's separate.
template <typename T>
inline void fill_cpu_simd(T* __restrict__ ptr, T value, size_t numel) {
#if defined(__AVX512F__)
    constexpr size_t VBYTES = 64;
    static_assert(sizeof(T) <= VBYTES && (VBYTES % sizeof(T) == 0),
                  "fill_cpu_simd: sizeof(T) must divide 64");
    constexpr size_t VEC_ELEMS = VBYTES / sizeof(T);
    alignas(VBYTES) unsigned char pattern[VBYTES];
    for (size_t k = 0; k < VEC_ELEMS; ++k)
        std::memcpy(pattern + k * sizeof(T), &value, sizeof(T));
    __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pattern));
    const size_t n_vec = numel / VEC_ELEMS;
    unsigned char* p = reinterpret_cast<unsigned char*>(ptr);
    for (size_t i = 0; i < n_vec; ++i)
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(p + i * VBYTES), vec);
    for (size_t i = n_vec * VEC_ELEMS; i < numel; ++i) ptr[i] = value;

#elif defined(__AVX2__)
    constexpr size_t VBYTES = 32;
    static_assert(sizeof(T) <= VBYTES && (VBYTES % sizeof(T) == 0),
                  "fill_cpu_simd: sizeof(T) must divide 32");
    constexpr size_t VEC_ELEMS = VBYTES / sizeof(T);
    alignas(VBYTES) unsigned char pattern[VBYTES];
    for (size_t k = 0; k < VEC_ELEMS; ++k)
        std::memcpy(pattern + k * sizeof(T), &value, sizeof(T));
    __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pattern));
    const size_t n_vec = numel / VEC_ELEMS;
    unsigned char* p = reinterpret_cast<unsigned char*>(ptr);
    for (size_t i = 0; i < n_vec; ++i)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i * VBYTES), vec);
    for (size_t i = n_vec * VEC_ELEMS; i < numel; ++i) ptr[i] = value;

#else
    for (size_t i = 0; i < numel; ++i) ptr[i] = value;
#endif
}
} } // namespace OwnTensor::detail

namespace OwnTensor {
// Forward declaration for is_same_type
    // template<typename T>
    // bool is_same_type(Dtype dtype);

    // =========================================================================
    // GENERIC IMPLEMENTATIONS (Used for standard types: float, double, int32, etc.)
    // =========================================================================

    template <typename T>
    inline void Tensor::set_data(const T* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<T>(dtype()))
        {
            throw std::runtime_error("Datatype mismatch");
        }

        // Use device-aware copy for standard types
        device::copy_memory(data(), device(),
                           source_data, DeviceIndex(Device::CPU),
                           count * sizeof(T));
    }

    template <typename T>
    inline void Tensor::set_grad(const T* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<T>(dtype()))
        {
            throw std::runtime_error("Datatype mismatch");
        }

        // Use device-aware copy for standard types
        device::copy_memory(grad(), device(),
                           source_data, DeviceIndex(Device::CPU),
                           count * sizeof(T));
    }

    template <typename T>
    inline void Tensor::set_data(const std::vector<T>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <typename T>
    inline void Tensor::set_grad(const std::vector<T>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    // -----------------------------------------------------------------------
    // Tensor::fill — edits: Gautam_Reddy_1926
    //
    // fill() vs full() — IMPORTANT distinction:
    //   - full(shape, value)  → factory; ALLOCATES a fresh, always-contiguous
    //     tensor. Because the buffer is guaranteed flat & contiguous, full()
    //     can use the CUDA Driver memset APIs (cuMemsetD16/D32Async) which
    //     run on the DMA engine and skip the SMs entirely (~5–10% faster
    //     than our kernel on 2 B / 4 B dtypes).
    //   - fill(value) (this method) → IN-PLACE on an EXISTING tensor that
    //     may be a strided view, slice, or transposed alias. cuMemsetD*Async
    //     writes a contiguous flat range; on a strided view it would corrupt
    //     elements that aren't part of the logical view. So fill() MUST stay
    //     on our custom fill_cuda_launch<T> kernel, which iterates by element
    //     and respects whatever pointer + numel we hand it. PyTorch makes the
    //     same call (fill_ goes through TensorIterator + a CUDA functor,
    //     never cuMemset).
    //
    // Paths:
    //   CPU: detail::fill_cpu_simd<T>  (AVX-512 / AVX2 stores; OpenMP off
    //        because libgomp setup cost dwarfs the work — see zeros() doc)
    //   GPU: cuda::fill_cuda_launch<T> (uint4 STG.128 vectorized stores)
    //   GPU fallback (fp4 / other quantized types): old CPU-vector + H->D
    //        memcpy via set_data(); kept because fill_cuda_launch is only
    //        instantiated for the supported dtype set.
    // -----------------------------------------------------------------------
    template <typename T>
    inline void Tensor::fill(T value)
    {
        if (!is_same_type<T>(dtype())) {
            throw std::runtime_error("Fill: Datatype mismatch - input type must match tensor dtype");
        }

        if (device().is_cpu()) {
            if constexpr (detail::fill_cpu_simd_supported<T>) {
                detail::fill_cpu_simd<T>(
                    reinterpret_cast<T*>(this->data()), value, numel());
            } else {
                T* data = reinterpret_cast<T*>(this->data());
                for (size_t i = 0; i < numel(); ++i) data[i] = value;
            }
        } else {
#ifdef WITH_CUDA
            if constexpr (detail::fill_cuda_supported<T>) {
                // Native GPU fill — cudaMemsetAsync for zero, vectorized uint4
                // stores for non-zero. Replaces the old CPU-vector + H->D
                // memcpy that alloc'd numel*sizeof(T) of host memory per call.
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                OwnTensor::cuda::fill_cuda_launch<T>(
                    reinterpret_cast<T*>(this->data()),
                    value,
                    static_cast<int64_t>(numel()),
                    stream);
            } else {
                // Fallback for fp4 / quantized types (Gautam_Reddy_1926):
                // direct cudaMemcpyAsync, skipping the set_data() wrapper.
                // set_data() would just re-validate dtype (we already checked
                // above) and call device::copy_memory == cudaMemcpyAsync.
                // For fp4 the struct layout IS its raw byte (1 B), so a
                // contiguous std::vector<T> is the correct H->D byte stream.
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                std::vector<T> temp_data(numel(), value);
                cudaError_t err = cudaMemcpyAsync(
                    this->data(), temp_data.data(),
                    numel() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("Tensor::fill cudaMemcpyAsync: ") +
                                             cudaGetErrorString(err));
                }
                cudaStreamSynchronize(stream);  // host vec dies at end of scope
            }
#else
            throw std::runtime_error("Tensor::fill: CUDA not available");
#endif
        }
    }

    // -----------------------------------------------------------------------
    // Tensor::fill_grad — edits: Gautam_Reddy_1926
    //
    // Same as Tensor::fill, but writes into the GRAD buffer instead of the
    // data buffer. Same reasoning applies for the API choice:
    //   - This is in-place on an existing grad buffer; the buffer may have
    //     been allocated for a tensor that's a strided view, so we MUST use
    //     the custom fill_cuda_launch<T> kernel — NOT cuMemsetD16/D32Async
    //     (those would corrupt non-contiguous layouts).
    //   - The factory routines (zeros / ones / full) are the ones allowed to
    //     use the driver memset shortcut, because they create fresh
    //     contiguous tensors. Anything that operates in-place (fill,
    //     fill_grad, zero_grad on a single tensor) stays on the kernel path.
    //
    // Paths (identical to fill()):
    //   CPU: detail::fill_cpu_simd<T>   (AVX-512 / AVX2)
    //   GPU: cuda::fill_cuda_launch<T>  (uint4 STG.128)
    //   GPU fallback for fp4 / quantized: CPU-vector + H->D memcpy.
    //
    // NOTE: Optimizer::zero_grad() does NOT call fill_grad in a loop — it
    //   batches all GPU grads into ONE multi_tensor_zero kernel launch
    //   (see src/Kernels/cuda/optimizer/arch/MultiTensorKernels_sm89.cu).
    //   That batching (148 launches → 1) is the optimizer-side win, separate
    //   from any per-tensor speed comparison.
    // -----------------------------------------------------------------------
    template <typename T>
    inline void Tensor::fill_grad(T value)
    {
        if (!is_same_type<T>(dtype())) {
            throw std::runtime_error("Fill: Datatype mismatch - input type must match tensor dtype");
        }

        if (device().is_cpu()) {
            if constexpr (detail::fill_cpu_simd_supported<T>) {
                detail::fill_cpu_simd<T>(
                    reinterpret_cast<T*>(grad()), value, numel());
            } else {
                T* data = reinterpret_cast<T*>(grad());
                for (size_t i = 0; i < numel(); ++i) data[i] = value;
            }
        } else {
#ifdef WITH_CUDA
            if constexpr (detail::fill_cuda_supported<T>) {
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                OwnTensor::cuda::fill_cuda_launch<T>(
                    reinterpret_cast<T*>(grad()),
                    value,
                    static_cast<int64_t>(numel()),
                    stream);
            } else if constexpr (std::is_same_v<T, bool>) {
                // Bool is byte-backed; std::vector<bool> is bitset-packed and
                // has no .data() — so we cannot use the std::vector<T> fallback
                // below. Use cudaMemsetAsync directly with byte 0x00 / 0x01.
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cudaError_t err = cudaMemsetAsync(grad(), value ? 1 : 0,
                                                  numel() * sizeof(bool), stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("Tensor::fill_grad bool cudaMemsetAsync: ") +
                                             cudaGetErrorString(err));
                }
            } else {
                // Fallback for fp4 / quantized types — direct H->D memcpy.
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                std::vector<T> temp_data(numel(), value);
                cudaError_t err = cudaMemcpyAsync(
                    grad(), temp_data.data(),
                    numel() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("Tensor::fill_grad cudaMemcpyAsync: ") +
                                             cudaGetErrorString(err));
                }
                cudaStreamSynchronize(stream);  // host vec dies at end of scope
            }
#else
            throw std::runtime_error("Tensor::fill_grad: CUDA not available");
#endif
        }
    }

    template <typename T>
    inline void Tensor::set_data(std::initializer_list<T> values) {
        set_data(values.begin(), values.size());
    }


    template <typename T>
    inline void Tensor::set_grad(std::initializer_list<T> values) {
        set_grad(values.begin(), values.size());
    }

    // =========================================================================
    // SPECIALIZED IMPLEMENTATIONS (CRITICAL FIX for custom 16-bit types)
    // =========================================================================

    // --- Specialization for float16_t ---
    // edits: Gautam_Reddy_1926 — dropped per-element raw_bits extraction loop.
    // float16_t is `struct { uint16_t raw_bits; }`, standard-layout, sizeof==2,
    // so a `const float16_t*` is byte-identical to `const uint16_t*`. We just
    // reinterpret_cast and let copy_memory memcpy the bytes directly. Saves
    // an O(N) host loop + O(N) host vector alloc per call (matches PyTorch's
    // at::Half handling; their Half is also layout-compatible with uint16_t).
    template <>
    inline void Tensor::set_data<float16_t>(const float16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<float16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }
        device::copy_memory(data(), device(),
                            reinterpret_cast<const uint16_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint16_t));
    }

    template <>
    inline void Tensor::set_grad<float16_t>(const float16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<float16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }
        device::copy_memory(grad(), device(),
                            reinterpret_cast<const uint16_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint16_t));
    }

    // Specialization for vector<float16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<float16_t>(const std::vector<float16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <>
    inline void Tensor::set_grad<float16_t>(const std::vector<float16_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    // --- Specialization for bfloat16_t ---
    // edits: Gautam_Reddy_1926 — same reinterpret_cast trick as float16_t.
    // bfloat16_t is layout-compatible with uint16_t (single raw_bits field).
    template <>
    inline void Tensor::set_data<bfloat16_t>(const bfloat16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bfloat16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }
        device::copy_memory(data(), device(),
                            reinterpret_cast<const uint16_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint16_t));
    }

    template <>
    inline void Tensor::set_grad<bfloat16_t>(const bfloat16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bfloat16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }
        device::copy_memory(grad(), device(),
                            reinterpret_cast<const uint16_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint16_t));
    }

    // Specialization for vector<bfloat16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<bfloat16_t>(const std::vector<bfloat16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <>
    inline void Tensor::set_grad<bfloat16_t>(const std::vector<bfloat16_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    // Functions for Boolean

    // edits: Gautam_Reddy_1926 — dropped the `? 1 : 0` normalization loop.
    // C++ ABI guarantees `bool` is stored as 0x00 or 0x01 in a 1-byte slot;
    // the loop only mattered if someone reinterpret_cast'd a garbage byte
    // buffer to bool*, which doesn't happen in our call sites. PyTorch makes
    // the same call: `bool → bool` copies use raw cudaMemcpyAsync, only
    // cross-dtype casts go through a static_cast<bool> functor.
    template<>
    inline void Tensor::set_data<bool>(const bool* source_data, size_t count) {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }
        device::copy_memory(data(), device(),
                            reinterpret_cast<const uint8_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint8_t));
    }

    // std::vector<bool> is the one place we still need a host loop —
    // the standard packs it as a bitset (1 bit per element, NOT 1 byte),
    // so we can't reinterpret_cast it. We unpack into a contiguous uint8_t
    // buffer and then memcpy. This is a std::vector<bool>-specific quirk;
    // arrays of plain `bool` (above) skip this entirely.
    template<>
    inline void Tensor::set_data<bool>(const std::vector<bool>& source_data) {
        if (source_data.size() != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }
        std::vector<uint8_t> temp(source_data.size());
        for (size_t i = 0; i < source_data.size(); ++i) temp[i] = source_data[i];
        device::copy_memory(data(), device(),
                            temp.data(), DeviceIndex(Device::CPU),
                            temp.size() * sizeof(uint8_t));
    }

    template<>
    inline void Tensor::set_grad<bool>(const std::vector<bool>& source_data) {
        if (source_data.size() != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }
        if (!impl_ || !impl_->has_autograd_meta()) throw std::runtime_error("Gradient not allocated");
        // std::vector<bool> bitset → unpack into uint8_t buffer (see above).
        std::vector<uint8_t> temp(source_data.size());
        for (size_t i = 0; i < source_data.size(); ++i) temp[i] = source_data[i];
        device::copy_memory(grad(), device(),
                            temp.data(), DeviceIndex(Device::CPU),
                            temp.size() * sizeof(uint8_t));
    }


    // Specialization for fill with bool
    template<>
    void Tensor::fill<bool>(bool value);

    // Specialization for fill_grad with bool
    // (std::vector<bool> is a bitset — no .data() method, so the generic
    //  template's fallback cudaMemcpyAsync path fails to compile.)
    template<>
    void Tensor::fill_grad<bool>(bool value);

    // edits: Gautam_Reddy_1926 — fp4 structs are `{ uint8_t raw_bits; }`,
    // 1-byte standard-layout, so a `const float4_e2m1_t*` is byte-identical
    // to `const uint8_t*`. Reinterpret_cast + direct memcpy. No host alloc,
    // no per-element loop.
    template<>
    inline void Tensor::set_data<float4_e2m1_t>(const float4_e2m1_t* source_data, size_t count)
    {
        if (count != numel()) throw std::runtime_error("Data size mismatch");
        if (!is_same_type<float4_e2m1_t>(dtype())) throw std::runtime_error("Data type mismatch");
        device::copy_memory(data(), device(),
                            reinterpret_cast<const uint8_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_data<float4_e2m1_t>(const std::vector<float4_e2m1_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    // edits: Gautam_Reddy_1926 — same reinterpret_cast trick (see above).
    template<>
    inline void Tensor::set_data<float4_e2m1_2x_t>(const float4_e2m1_2x_t* source_data, size_t count)
    {
        if (count != numel()) throw std::runtime_error("Data size mismatch");
        if (!is_same_type<float4_e2m1_2x_t>(dtype())) throw std::runtime_error("Data type mismatch");
        device::copy_memory(data(), device(),
                            reinterpret_cast<const uint8_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_data<float4_e2m1_2x_t>(const std::vector<float4_e2m1_2x_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    // edits: Gautam_Reddy_1926 — same reinterpret_cast trick (see above).
    template<>
    inline void Tensor::set_grad<float4_e2m1_2x_t>(const float4_e2m1_2x_t* source_data, size_t count)
    {
        if (count != numel()) throw std::runtime_error("Data size mismatch");
        if (!is_same_type<float4_e2m1_2x_t>(dtype())) throw std::runtime_error("Data type mismatch");
        device::copy_memory(grad(), device(),
                            reinterpret_cast<const uint8_t*>(source_data),
                            DeviceIndex(Device::CPU),
                            count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_grad<float4_e2m1_2x_t>(const std::vector<float4_e2m1_2x_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }
}
#endif // TENSOR_DATAMANIP_H
