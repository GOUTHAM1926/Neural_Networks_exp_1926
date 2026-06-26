#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/TensorDispatch.h"
#include <stdexcept>
#include "core/Tensor.h"
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"
#include "device/DeviceCore.h"

namespace OwnTensor {

// ============================================================================
// COMPLEX TRAIT
// ============================================================================
template <typename T> struct is_complex_t : std::false_type {};
template <> struct is_complex_t<complex32_t>  : std::true_type {};
template <> struct is_complex_t<complex64_t>  : std::true_type {};
template <> struct is_complex_t<complex128_t> : std::true_type {};

// Convenience alias: true when T is NOT a complex type
template <typename T>
static constexpr bool is_real_type_v = !is_complex_t<T>::value;

// ============================================================================
// HELPERS
// ============================================================================
static constexpr int UNROLL_FACTOR = 4;

inline __host__ __device__ size_t round_up(size_t n, size_t d) {
    return ((n + d - 1) / d) * d;
}

inline dim3 pick_grid(size_t n, dim3 b) {
    // Each thread handles UNROLL_FACTOR elements, so fewer blocks needed
    size_t blocks = (n + b.x * UNROLL_FACTOR - 1) / (b.x * UNROLL_FACTOR);
    if (blocks < 1) blocks = 1;
    if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline Dtype get_division_output_dtype(Dtype input_dtype) {
    if (input_dtype == Dtype::Bool)   return Dtype::Float32;
    if (input_dtype == Dtype::Int16 ||
        input_dtype == Dtype::Int32 ||
        input_dtype == Dtype::Int64)  return Dtype::Float32;
    return input_dtype;
}

// ============================================================================
// KERNELS — In-place (rounded grid-stride + 4× unroll)
// ============================================================================
template<typename T>
__global__ void scalar_add_inplace(T* d, double s, size_t n) {
    const T sv = static_cast<T>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] + sv;
        }
    }
}
template<typename T>
__global__ void scalar_sub_inplace(T* d, double s, size_t n) {
    const T sv = static_cast<T>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] - sv;
        }
    }
}
template<typename T>
__global__ void scalar_mul_inplace(T* d, double s, size_t n) {
    const T sv = static_cast<T>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] * sv;
        }
    }
}
template<typename T>
__global__ void scalar_div_inplace(T* d, double s, size_t n) {
    const T sv = static_cast<T>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) d[i] = d[i] / sv;
        }
    }
}

// ============================================================================
// KERNELS — Copy (SrcT → DstT, scalar passed as double and cast on device)
// Passing the scalar as a simple primitive (double) sidesteps an NVCC ABI
// quirk with small-struct (fp16/bf16) value passing across the launch boundary.
// Template collapses <T, S, U> → <T, U>; scalar becomes a plain double.
// ============================================================================
template<typename T, typename U>
__global__ void scalar_add_copy(const T* a, double s, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) + sv;
        }
    }
}
template<typename T, typename U>
__global__ void scalar_sub_copy(const T* a, double s, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) - sv;
        }
    }
}
template<typename T, typename U>
__global__ void scalar_mul_copy(const T* a, double s, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) * sv;
        }
    }
}
template<typename T, typename U>
__global__ void scalar_div_copy(const T* a, double s, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = static_cast<U>(a[i]) / sv;
        }
    }
}

template<typename T, typename U>
__global__ void scalar_sub_s_t(double s, const T* a, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = sv - static_cast<U>(a[i]);
        }
    }
}

template<typename T, typename U>
__global__ void scalar_div_s_t(double s, const T* a, U* o, size_t n) {
    const U sv = static_cast<U>(s);
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    const size_t n_round = round_up(n, stride * UNROLL_FACTOR);
    for (size_t base = tid; base < n_round; base += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const size_t i = base + u * stride;
            if (i < n) o[i] = sv / static_cast<U>(a[i]);
        }
    }
}
// ============================================================================
// Comparison kernels — scalar passed as double, cast on device
// ============================================================================
template<typename T>
__global__ void scalar_eq_copy(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] == sv);
}

template<typename T>
__global__ void scalar_neq_copy(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] != sv);
}

template<typename T>
__global__ void scalar_gt_t_s(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] > sv);
}

template<typename T>
__global__ void scalar_lt_t_s(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] < sv);
}

template<typename T>
__global__ void scalar_geq_t_s(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] >= sv);
}

template<typename T>
__global__ void scalar_leq_t_s(const T* a, double s, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(a[i] <= sv);
}

template<typename T>
__global__ void scalar_gt_s_t(double s, const T* a, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(sv > a[i]);
}

template<typename T>
__global__ void scalar_lt_s_t(double s, const T* a, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(sv < a[i]);
}

template<typename T>
__global__ void scalar_geq_s_t(double s, const T* a, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(sv >= a[i]);
}

template<typename T>
__global__ void scalar_leq_s_t(double s, const T* a, uint8_t* o, size_t n) {
    const T sv = static_cast<T>(s);
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
        o[i] = static_cast<uint8_t>(sv <= a[i]);
}
// ============================================================================
// PUBLIC CUDA BACKEND — scalar passed as double (single instantiation per op)
// The scalar is cast to the tensor's dtype inside each kernel launch, mirroring
// PyTorch's c10::Scalar approach. Collapses 13 instantiations/op → 1.
// ============================================================================
void cuda_add_inplace(Tensor& t, double s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (is_real_type_v<T>) {
            scalar_add_inplace<<<grid, block, 0, stream>>>(
                t.data<T>(), s, t.numel());
        }
    });
}
void cuda_sub_inplace(Tensor& t, double s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (is_real_type_v<T>) {
            scalar_sub_inplace<<<grid, block, 0, stream>>>(
                t.data<T>(), s, t.numel());
        }
    });
}
void cuda_mul_inplace(Tensor& t, double s, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(t.numel(), block);
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (is_real_type_v<T>) {
            scalar_mul_inplace<<<grid, block, 0, stream>>>(
                t.data<T>(), s, t.numel());
        }
    });
}
void cuda_div_inplace(Tensor& t, double s, cudaStream_t stream) {
    if (s == 0.0) throw std::runtime_error("Division by zero");
    Dtype dt = t.dtype();
    Dtype promoted_dt = get_division_output_dtype(dt);
    if (promoted_dt != dt)
        throw std::runtime_error("In-place /= requires float dtype. Input is " +
            get_dtype_name(dt) + " but needs " + get_dtype_name(promoted_dt));
    dim3 block(256), grid = pick_grid(t.numel(), block);
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (is_real_type_v<T>) {
            scalar_div_inplace<<<grid, block, 0, stream>>>(
                t.data<T>(), s, t.numel());
        }
    });
}

void cuda_add_copy(const Tensor& a, double s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_add_copy<<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

void cuda_sub_copy(const Tensor& a, double s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_sub_copy<<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

void cuda_mul_copy(const Tensor& a, double s, Tensor& b, Dtype promoted_dtype, cudaStream_t stream) {
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(promoted_dtype, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_mul_copy<<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, b.data<DstT>(), a.numel());
                }
            });
        }
    });
}

void cuda_div_copy(const Tensor& a, double s, Tensor& out, Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = get_division_output_dtype(a.dtype());
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_div_copy<<<grid, block, 0, stream>>>(
                        a.data<SrcT>(), s, out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

void cuda_sub_copy_scalar_tensor(double s, const Tensor& a, Tensor& out,
                                 Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = out.dtype();
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_sub_s_t<<<grid, block, 0, stream>>>(
                        s, a.data<SrcT>(), out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

void cuda_div_copy_scalar_tensor(double s, const Tensor& a, Tensor& out, Dtype promoted_dtype, cudaStream_t stream) {
    const Dtype output_dt = out.dtype();
    dim3 block(256), grid = pick_grid(a.numel(), block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            dispatch_by_dtype(output_dt, [&](auto d_dst) {
                using DstT = decltype(d_dst);
                if constexpr (is_real_type_v<DstT>) {
                    scalar_div_s_t<<<grid, block, 0, stream>>>(
                        s, a.data<SrcT>(), out.data<DstT>(), a.numel());
                }
            });
        }
    });
}

// ============================================================================
// Comparison ops — always produce Bool (uint8_t) output
// ============================================================================
void cuda_eq_tensor_scalar(const Tensor& a, double s, Tensor& out,
                           Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_eq_copy<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_neq_tensor_scalar(const Tensor& a, double s, Tensor& out,
                            Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_neq_copy<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_gt_tensor_scalar(const Tensor& a, double s, Tensor& out,
                           Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_gt_t_s<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_lt_tensor_scalar(const Tensor& a, double s, Tensor& out,
                           Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_lt_t_s<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_geq_tensor_scalar(const Tensor& a, double s, Tensor& out,
                            Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_geq_t_s<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_leq_tensor_scalar(const Tensor& a, double s, Tensor& out,
                            Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_leq_t_s<<<grid, block, 0, stream>>>(
                a.data<SrcT>(), s, out.data<uint8_t>(), n);
        }
    });
}

void cuda_gt_scalar_tensor(double s, const Tensor& a, Tensor& out,
                           Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_gt_s_t<<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(), out.data<uint8_t>(), n);
        }
    });
}

void cuda_lt_scalar_tensor(double s, const Tensor& a, Tensor& out,
                           Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_lt_s_t<<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(), out.data<uint8_t>(), n);
        }
    });
}

void cuda_geq_scalar_tensor(double s, const Tensor& a, Tensor& out,
                            Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_geq_s_t<<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(), out.data<uint8_t>(), n);
        }
    });
}

void cuda_leq_scalar_tensor(double s, const Tensor& a, Tensor& out,
                            Dtype /*promoted_dtype*/, cudaStream_t stream) {
    const size_t n = a.numel();
    dim3 block(256), grid = pick_grid(n, block);
    dispatch_by_dtype(a.dtype(), [&](auto d_src) {
        using SrcT = decltype(d_src);
        if constexpr (is_real_type_v<SrcT>) {
            scalar_leq_s_t<<<grid, block, 0, stream>>>(
                s, a.data<SrcT>(), out.data<uint8_t>(), n);
        }
    });
}
// ============================================================================
// NOTE: Explicit instantiations removed — all backend functions are now
// non-templated and take scalar as double (cast to tensor dtype internally).
// This collapses 260 instantiations to ~20 function definitions.
// ============================================================================

} // namespace OwnTensor
#endif // WITH_CUDA

