#if defined(WITH_CUDA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/TensorDispatch.h"
#include "ops/helpers/ConditionalOps.h"
#include "dtype/Types.h"

namespace OwnTensor {

namespace {

// Generic where kernel
template<typename CondT, typename DataT>
__global__ void k_where(const CondT* condition, const DataT* input,
                        const DataT* other, DataT* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        bool cond = (condition[i] != static_cast<CondT>(0.0f));
        out[i] = cond ? input[i] : other[i];
    }
}

// Specialized kernel for fp16/bf16
template<typename CondT, typename __half>
__global__ void k_where_fp16(const CondT* condition, const __half* input,
                             const __half* other, __half* out, 
                             size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        bool cond = (condition[i] != static_cast<CondT>(0.0f));
        out[i] = cond ? input[i] : other[i]; 
    }
}

template<typename CondT, typename __nv_bfloat16>
__global__ void k_where_bf16(const CondT* condition, const __nv_bfloat16* input,
                             const __nv_bfloat16* other, __nv_bfloat16* out, 
                             size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        bool cond = (condition[i] != static_cast<CondT>(0.0f));
        out[i] = cond ? input[i] : other[i]; 
    }
}

inline dim3 pick_grid(size_t n, dim3 b) {
    size_t blocks = (n + b.x - 1) / b.x;
    //if (blocks > 2147483647ULL) blocks = 2147483647ULL;
    return dim3(static_cast<unsigned int>(blocks));
}

inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // anonymous namespace

void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out) {
    const size_t n = condition.numel();
    const dim3 block(256);
    const dim3 grid = pick_grid(n, block);
    
    const Dtype cond_dtype = condition.dtype();
    const Dtype input_dtype = input.dtype();
    
    dispatch_by_dtype(cond_dtype, [&](auto cond_type) {
        using CondT = decltype(cond_type);
        const CondT* cond_ptr = condition.data<CondT>();
        
        if (input_dtype == Dtype::Float16) {
            const __half* input_ptr = input.data<__half>();
            const __half* other_ptr = other.data<__half>();
            __half* out_ptr = out.data<__half>();
            
            k_where_fp16<<<grid, block>>>(cond_ptr, input_ptr, other_ptr, 
                                          out_ptr, n);
        } else if(input_dtype == Dtype::Bfloat16)
        {
            const __nv_bfloat16* input_ptr = input.data<__nv_bfloat16>();
            const __nv_bfloat16* other_ptr = other.data<__nv_bfloat16>();
            __nv_bfloat16* out_ptr = out.data<__nv_bfloat16>();
            
            k_where_fp16<<<grid, block>>>(cond_ptr, input_ptr, other_ptr, 
                                          out_ptr, n);
        }
         else {
            dispatch_by_dtype(input_dtype, [&](auto data_type) {
                using DataT = decltype(data_type);
                const DataT* input_ptr = input.data<DataT>();
                const DataT* other_ptr = other.data<DataT>();
                DataT* out_ptr = out.data<DataT>();
                
                k_where<<<grid, block>>>(cond_ptr, input_ptr, other_ptr, 
                                         out_ptr, n);
            });
        }
    });
    
    check_cuda_error("cuda_where");
}

// ============================================================================
// SCALAR BACKEND VARIANTS - CUDA
// ============================================================================

// Kernel 1: Scalar input, Tensor other
template<typename DataT>
__global__ void k_where_scalar_tensor(const bool* condition, DataT input_val,
                                       const DataT* other, DataT* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        out[i] = condition[i] ? input_val : other[i];
    }
}

// Kernel 2: Tensor input, Scalar other
template<typename DataT>
__global__ void k_where_tensor_scalar(const bool* condition, const DataT* input,
                                       DataT other_val, DataT* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        out[i] = condition[i] ? input[i] : other_val;
    }
}

// Kernel 3: Both scalars
template<typename DataT>
__global__ void k_where_scalar_scalar(const bool* condition, DataT input_val,
                                       DataT other_val, DataT* out, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        out[i] = condition[i] ? input_val : other_val;
    }
}

// Host function 1: Scalar input, Tensor other
template<typename T>
void cuda_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                               const Tensor& other, Tensor& out) {
    const size_t n = out.numel();
    const dim3 block(256);
     dim3 grid = pick_grid(n, block);
    
    const bool* cond_ptr = condition.data<bool>();
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const scalar_t* other_ptr = other.data<scalar_t>();
        scalar_t* out_ptr = out.data<scalar_t>();
        
        // Helper to convert scalar to target type (handles complex types)
        scalar_t input_val;
        if constexpr (std::is_same_v<scalar_t, complex32_t> || std::is_same_v<scalar_t, complex64_t>) {
            input_val = scalar_t(static_cast<float>(input_scalar), 0.0f);
        } else if constexpr (std::is_same_v<scalar_t, complex128_t>) {
            input_val = scalar_t(static_cast<double>(input_scalar), 0.0);
        } else if constexpr (std::is_same_v<scalar_t, float4_e2m1_2x_t> || std::is_same_v<scalar_t, float4_e2m1_t>) {
            input_val = static_cast<scalar_t>(static_cast<float>(input_scalar));
        } else {
            input_val = static_cast<scalar_t>(input_scalar);
        }
        grid = (static_cast<int>(grid.x) <= 56) ? dim3(56) : grid;
        k_where_scalar_tensor<<<grid, block>>>(cond_ptr, input_val, other_ptr, out_ptr, n);
    });
    
    check_cuda_error("cuda_where_scalar_tensor");
}

// Host function 2: Tensor input, Scalar other
template<typename T>
void cuda_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                               T other_scalar, Tensor& out) {
    const size_t n = out.numel();
    const dim3 block(256);
    const dim3 grid = pick_grid(n, block);
    
    const bool* cond_ptr = condition.data<bool>();
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const scalar_t* input_ptr = input.data<scalar_t>();
        scalar_t* out_ptr = out.data<scalar_t>();
        
        // Helper to convert scalar to target type (handles complex types)
        scalar_t other_val;
        if constexpr (std::is_same_v<scalar_t, complex32_t> || std::is_same_v<scalar_t, complex64_t>) {
            other_val = scalar_t(static_cast<float>(other_scalar), 0.0f);
        } else if constexpr (std::is_same_v<scalar_t, complex128_t>) {
            other_val = scalar_t(static_cast<double>(other_scalar), 0.0);
        } else if constexpr (std::is_same_v<scalar_t, float4_e2m1_2x_t> || std::is_same_v<scalar_t, float4_e2m1_t>) {
            other_val = static_cast<scalar_t>(static_cast<float>(other_scalar));
        } else {
            other_val = static_cast<scalar_t>(other_scalar);
        }
        
        k_where_tensor_scalar<<<grid, block>>>(cond_ptr, input_ptr, other_val, out_ptr, n);
    });
    
    check_cuda_error("cuda_where_tensor_scalar");
}

// Host function 3: Both scalars
template<typename T, typename U>
void cuda_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                               U other_scalar, Tensor& out) {
    const size_t n = out.numel();
    const dim3 block(256);
    const dim3 grid = pick_grid(n, block);
    
    const bool* cond_ptr = condition.data<bool>();
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        scalar_t* out_ptr = out.data<scalar_t>();
        
        // Helper to convert scalars to target type (handles complex types)
        scalar_t input_val, other_val;
        if constexpr (std::is_same_v<scalar_t, complex32_t> || std::is_same_v<scalar_t, complex64_t>) {
            input_val = scalar_t(static_cast<float>(input_scalar), 0.0f);
            other_val = scalar_t(static_cast<float>(other_scalar), 0.0f);
        } else if constexpr (std::is_same_v<scalar_t, complex128_t>) {
            input_val = scalar_t(static_cast<double>(input_scalar), 0.0);
            other_val = scalar_t(static_cast<double>(other_scalar), 0.0);
        } else if constexpr (std::is_same_v<scalar_t, float4_e2m1_2x_t> || std::is_same_v<scalar_t, float4_e2m1_t>) {
            input_val = static_cast<scalar_t>(static_cast<float>(input_scalar));
            other_val = static_cast<scalar_t>(static_cast<float>(other_scalar));
        } else {
            input_val = static_cast<scalar_t>(input_scalar);
            other_val = static_cast<scalar_t>(other_scalar);
        }
        
        k_where_scalar_scalar<<<grid, block>>>(cond_ptr, input_val, other_val, out_ptr, n);
    });
    
    check_cuda_error("cuda_where_scalar_scalar");
}

// Explicit instantiations
template void cuda_where_scalar_tensor<int>(const Tensor&, int, const Tensor&, Tensor&);
template void cuda_where_scalar_tensor<float>(const Tensor&, float, const Tensor&, Tensor&);
template void cuda_where_scalar_tensor<double>(const Tensor&, double, const Tensor&, Tensor&);
template void cuda_where_scalar_tensor<long>(const Tensor&, long, const Tensor&, Tensor&);

template void cuda_where_tensor_scalar<int>(const Tensor&, const Tensor&, int, Tensor&);
template void cuda_where_tensor_scalar<float>(const Tensor&, const Tensor&, float, Tensor&);
template void cuda_where_tensor_scalar<double>(const Tensor&, const Tensor&, double, Tensor&);
template void cuda_where_tensor_scalar<long>(const Tensor&, const Tensor&, long, Tensor&);

template void cuda_where_scalar_scalar<int, int>(const Tensor&, int, int, Tensor&);
template void cuda_where_scalar_scalar<float, float>(const Tensor&, float, float, Tensor&);
template void cuda_where_scalar_scalar<double, double>(const Tensor&, double, double, Tensor&);
template void cuda_where_scalar_scalar<long, long>(const Tensor&, long, long, Tensor&);

template void cuda_where_scalar_scalar<int, float>(const Tensor&, int, float, Tensor&);
template void cuda_where_scalar_scalar<int, double>(const Tensor&, int, double, Tensor&);
template void cuda_where_scalar_scalar<int, long>(const Tensor&, int, long, Tensor&);

template void cuda_where_scalar_scalar<float, int>(const Tensor&, float, int, Tensor&);
template void cuda_where_scalar_scalar<float, double>(const Tensor&, float, double, Tensor&);
template void cuda_where_scalar_scalar<float, long>(const Tensor&, float, long, Tensor&);

template void cuda_where_scalar_scalar<double, int>(const Tensor&, double, int, Tensor&);
template void cuda_where_scalar_scalar<double, float>(const Tensor&, double, float, Tensor&);
template void cuda_where_scalar_scalar<double, long>(const Tensor&, double, long, Tensor&);

template void cuda_where_scalar_scalar<long, int>(const Tensor&, long, int, Tensor&);
template void cuda_where_scalar_scalar<long, float>(const Tensor&, long, float, Tensor&);
template void cuda_where_scalar_scalar<long, double>(const Tensor&, long, double, Tensor&);

} // namespace OwnTensor

#endif // WITH_CUDA
