// =============================================================================
// fill_cuda_launch<T> — GPU-native constant fill
//
// Design (mirrors PyTorch's vectorized_elementwise_kernel, simplified):
//
//   1. Fast path  : value == 0 for numeric T -> cudaMemsetAsync (DMA engine).
//   2. Main path  : 16-byte vectorized stores via uint4 when ptr is 16-byte
//                   aligned and numel >= VEC. STG.128 on sm_89.
//   3. Tail path  : scalar kernel for unaligned pointers or the remainder
//                   beyond (numel / VEC) * VEC.
//
// PyTorch equivalent: FillFunctor + launch_vectorized_kernel (same vec=16/sizeof(T)
// rule + alignment check + tail handler).
// =============================================================================

#include "ops/helpers/FillKernels.h"
#include "dtype/Types.h"       // float16_t, bfloat16_t, complex*_t
#include <type_traits>
#include <cstring>

namespace OwnTensor { namespace cuda {

// Replicate `value` to fill N bytes (N = 16 or 8), bit-cast to uint4 / uint2.
// memcpy avoids union default-ctor issues for types with user-defined ctors
// (float16_t, bfloat16_t, complex*_t).
template <typename T>
__host__ inline uint4 pack16_value(T value) {
    static_assert(sizeof(T) <= 16 && (16 % sizeof(T) == 0),
                  "fill_cuda_launch: sizeof(T) must divide 16");
    alignas(16) unsigned char bytes[16];
    for (size_t k = 0; k < 16 / sizeof(T); ++k)
        std::memcpy(bytes + k * sizeof(T), &value, sizeof(T));
    uint4 result;
    std::memcpy(&result, bytes, 16);
    return result;
}

template <typename T>
__host__ inline uint2 pack8_value(T value) {
    static_assert(sizeof(T) <= 8 && (8 % sizeof(T) == 0),
                  "fill_cuda_launch (vec8): sizeof(T) must divide 8");
    alignas(8) unsigned char bytes[8];
    for (size_t k = 0; k < 8 / sizeof(T); ++k)
        std::memcpy(bytes + k * sizeof(T), &value, sizeof(T));
    uint2 result;
    std::memcpy(&result, bytes, 8);
    return result;
}

// VEC16 kernel: one STG.128 per thread (16 bytes). Requires 16-byte aligned ptr.
template <typename T>
__global__ void __launch_bounds__(256, 2) fill_kernel_vec16(
    T* __restrict__ out, uint4 packed, int64_t n_vec)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    uint4* out_vec = reinterpret_cast<uint4*>(out);
    for (int64_t i = tid; i < n_vec; i += stride) out_vec[i] = packed;
}

// VEC8 kernel: one STG.64 per thread (8 bytes). Requires 8-byte aligned ptr.
// Fires when ptr is 8-byte aligned but not 16-byte (e.g. view starting at an
// odd multiple of 8 bytes).
template <typename T>
__global__ void __launch_bounds__(256, 2) fill_kernel_vec8(
    T* __restrict__ out, uint2 packed, int64_t n_vec)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    uint2* out_vec = reinterpret_cast<uint2*>(out);
    for (int64_t i = tid; i < n_vec; i += stride) out_vec[i] = packed;
}

// Scalar kernel: handles tail (< VEC elements) or unaligned pointers.
template <typename T>
__global__ void fill_kernel_scalar(T* __restrict__ out, T value,
                                   int64_t start, int64_t end) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = start + tid; i < end; i += stride) {
        out[i] = value;
    }
}

// Universal "bit-pattern all-zero" check — works for ANY trivially-copyable type
// (int, float, double, __half, __nv_bfloat16, our float16_t/bfloat16_t structs,
// complex*_t, etc.). If every byte of `value` is 0, the cudaMemsetAsync shortcut
// (DMA engine, no SMs) is safe — the result is identical to writing `value` via
// a kernel. Avoids per-type overloads.
template <typename T>
__host__ inline bool is_bitwise_zero(const T& value) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) if (b[i] != 0) return false;
    return true;
}

template <typename T>
void fill_cuda_launch(T* ptr, T value, int64_t numel, cudaStream_t stream) {
    if (numel <= 0 || ptr == nullptr) return;

    // ---------- Path 1: bit-pattern zero -> cudaMemsetAsync ----------
    if (is_bitwise_zero(value)) {
        cudaMemsetAsync(ptr, 0, static_cast<size_t>(numel) * sizeof(T), stream);
        return;
    }

    // ---------- Path 2: graded vectorized kernel (VEC16 → VEC8 → scalar) ----------
    // Mirrors PyTorch's can_vectorize_up_to fallback — use the biggest aligned
    // store that fits both the pointer alignment and the element type.
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    constexpr int VEC16_ELEMS = 16 / sizeof(T);           // e.g. 4 for float
    constexpr int VEC8_ELEMS  = (sizeof(T) <= 8) ? (8 / sizeof(T)) : 0;  // e.g. 2 for float

    const bool can_vec16 = (addr % 16 == 0) && (numel >= VEC16_ELEMS);
    // VEC8 only meaningful if sizeof(T) <= 8 AND vec16 didn't fit.
    const bool can_vec8  = (!can_vec16) && (addr % 8 == 0) &&
                           (sizeof(T) <= 8) && (numel >= VEC8_ELEMS);

    int64_t n_vec = 0;
    int64_t tail_start = 0;

    constexpr int THREADS = 256;
    auto launch_cap = [](int64_t n_vec_) {
        int blocks = static_cast<int>((n_vec_ + THREADS - 1) / THREADS);
        return blocks > 1024 ? 1024 : blocks;
    };

    if (can_vec16) {
        n_vec = numel / VEC16_ELEMS;
        tail_start = n_vec * VEC16_ELEMS;
        const uint4 packed = pack16_value<T>(value);
        fill_kernel_vec16<T><<<launch_cap(n_vec), THREADS, 0, stream>>>(
            ptr, packed, n_vec);
    } else if (can_vec8) {
        n_vec = numel / VEC8_ELEMS;
        tail_start = n_vec * VEC8_ELEMS;
        if constexpr (sizeof(T) <= 8) {
            const uint2 packed = pack8_value<T>(value);
            fill_kernel_vec8<T><<<launch_cap(n_vec), THREADS, 0, stream>>>(
                ptr, packed, n_vec);
        }
    }
    // else: everything goes through scalar tail below.

    // ---------- Path 3: scalar tail / fully unaligned ----------
    if (tail_start < numel) {
        int64_t tail_n = numel - tail_start;
        int blocks = static_cast<int>((tail_n + THREADS - 1) / THREADS);
        if (blocks > 1024) blocks = 1024;
        fill_kernel_scalar<T><<<blocks, THREADS, 0, stream>>>(
            ptr, value, tail_start, numel);
    }
}

// =============================================================================
// Explicit instantiations — MUST cover every type `dispatch_by_dtype` emits in
// TensorDispatch.h (lines 60–75), otherwise ones()/full() will fail to link.
// Note:
//   - bool is NOT here because TensorFactory handles it separately via
//     cudaMemsetAsync(ptr, 1 or 0, numel) at the call site.
//   - fp4 types (float4_e2m1*) are currently commented out in the main
//     dispatch switch and aren't emitted, so no instantiation needed yet.
// =============================================================================
template void fill_cuda_launch<float>    (float*,    float,    int64_t, cudaStream_t);
template void fill_cuda_launch<double>   (double*,   double,   int64_t, cudaStream_t);

// Our library's 16-bit float wrappers. sizeof == 2, compatible with uint4 packing.
template void fill_cuda_launch<float16_t>   (float16_t*,    float16_t,    int64_t, cudaStream_t);
template void fill_cuda_launch<bfloat16_t>  (bfloat16_t*,   bfloat16_t,   int64_t, cudaStream_t);

// Integer dtypes emitted by dispatch.
template void fill_cuda_launch<int8_t>   (int8_t*,   int8_t,   int64_t, cudaStream_t);
template void fill_cuda_launch<int16_t>  (int16_t*,  int16_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<int32_t>  (int32_t*,  int32_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<int64_t>  (int64_t*,  int64_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<uint8_t>  (uint8_t*,  uint8_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<uint16_t> (uint16_t*, uint16_t, int64_t, cudaStream_t);
template void fill_cuda_launch<uint32_t> (uint32_t*, uint32_t, int64_t, cudaStream_t);
template void fill_cuda_launch<uint64_t> (uint64_t*, uint64_t, int64_t, cudaStream_t);

// Complex dtypes — all trivially copyable POD structs (real/imag fields), so
// the uint4-packed kernel Just Works. sizeof is 4 / 8 / 16 → VEC = 4 / 2 / 1.
template void fill_cuda_launch<complex32_t>  (complex32_t*,  complex32_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<complex64_t>  (complex64_t*,  complex64_t,  int64_t, cudaStream_t);
template void fill_cuda_launch<complex128_t> (complex128_t*, complex128_t, int64_t, cudaStream_t);

} } // namespace OwnTensor::cuda
