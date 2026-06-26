#pragma once
#ifndef CONDITIONAL_OPS_H
#define CONDITIONAL_OPS_H

#include "core/Tensor.h"
#include "ops/helpers/BroadcastUtils.h"  // For broadcast_rhs_to_lhs
#include "dtype/DtypeTraits.h"            // For type_to_dtype, promote_dtypes_bool
#include "core/TensorDispatch.h"          // For dispatch_by_dtype

namespace OwnTensor {

// Forward declarations for CPU and CUDA backends - Tensor variants
void cpu_where(const Tensor& condition, const Tensor& input, 
               const Tensor& other, Tensor& out);

void cuda_where(const Tensor& condition, const Tensor& input,
                const Tensor& other, Tensor& out);

// Scalar backend variants - CPU
template<typename T>
void cpu_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                              const Tensor& other, Tensor& out);

template<typename T>
void cpu_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                              T other_scalar, Tensor& out);

template<typename T, typename U>
void cpu_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                              U other_scalar, Tensor& out);

// Scalar backend variants - CUDA
template<typename T>
void cuda_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                               const Tensor& other, Tensor& out);

template<typename T>
void cuda_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                               T other_scalar, Tensor& out);

template<typename T, typename U>
void cuda_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                               U other_scalar, Tensor& out);

// Public API - main where function
Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other);

// Scalar overloads
template <typename T>
Tensor where(const Tensor& condition, T input_scalar, const Tensor& other);

template <typename T>
Tensor where(const Tensor& condition, const Tensor& input, T other_scalar);

// Two-template version handles both same-type AND mixed-type scalars
template <typename T, typename U>
Tensor where(const Tensor& condition, T input_scalar, U other_scalar);


// ============================================================================
// TEMPLATE IMPLEMENTATIONS (inline in header to avoid explicit instantiations)
// ============================================================================

// ============================================================================
// CPU BACKEND TEMPLATE IMPLEMENTATIONS
// ============================================================================

// Variant 1: Scalar input, Tensor other
template<typename T>
inline void cpu_where_scalar_tensor(const Tensor& condition, T input_scalar, 
                              const Tensor& other, Tensor& out) {
    const auto& cond_shape = condition.shape().dims;
    const auto& other_shape = other.shape().dims;
    const auto& out_shape = out.shape().dims;
    
    const auto& cond_strides = condition.stride().strides;
    const auto& other_strides = other.stride().strides;
    
    size_t out_ndim = out_shape.size();
    size_t total_elems = out.numel();
    
    bool same_shape = (cond_shape == other_shape) && (other_shape == out_shape);
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const bool* cond_ptr = condition.data<bool>();
        const scalar_t* other_ptr = other.data<scalar_t>();
        scalar_t* out_ptr = out.data<scalar_t>();
        scalar_t input_val = static_cast<scalar_t>(input_scalar);
        
        if (same_shape) {
            for (size_t i = 0; i < total_elems; ++i) {
                out_ptr[i] = cond_ptr[i] ? input_val : other_ptr[i];
            }
        } else {
            // Broadcasting
            std::vector<size_t> cond_bcast_strides(out_ndim, 0);
            std::vector<size_t> other_bcast_strides(out_ndim, 0);
            
            size_t cond_ndim = cond_shape.size();
            size_t other_ndim = other_shape.size();
            
            for (size_t i = 0; i < out_ndim; ++i) {
                size_t cond_dim_idx = cond_ndim - out_ndim + i;
                size_t other_dim_idx = other_ndim - out_ndim + i;
                
                if (cond_dim_idx < cond_ndim && cond_shape[cond_dim_idx] > 1) {
                    cond_bcast_strides[i] = cond_strides[cond_dim_idx];
                }
                if (other_dim_idx < other_ndim && other_shape[other_dim_idx] > 1) {
                    other_bcast_strides[i] = other_strides[other_dim_idx];
                }
            }
            
            std::vector<size_t> coords(out_ndim, 0);
            
            for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
                size_t temp_idx = linear_idx;
                for (int dim = out_ndim - 1; dim >= 0; --dim) {
                    coords[dim] = temp_idx % out_shape[dim];
                    temp_idx /= out_shape[dim];
                }
                
                size_t cond_idx = 0;
                size_t other_idx = 0;
                
                for (size_t dim = 0; dim < out_ndim; ++dim) {
                    cond_idx += coords[dim] * cond_bcast_strides[dim];
                    other_idx += coords[dim] * other_bcast_strides[dim];
                }
                
                out_ptr[linear_idx] = cond_ptr[cond_idx] ? input_val : other_ptr[other_idx];
            }
        }
    });
}

// Variant 2: Tensor input, Scalar other
template<typename T>
inline void cpu_where_tensor_scalar(const Tensor& condition, const Tensor& input, 
                              T other_scalar, Tensor& out) {
    const auto& cond_shape = condition.shape().dims;
    const auto& input_shape = input.shape().dims;
    const auto& out_shape = out.shape().dims;
    
    const auto& cond_strides = condition.stride().strides;
    const auto& input_strides = input.stride().strides;
    
    size_t out_ndim = out_shape.size();
    size_t total_elems = out.numel();
    
    bool same_shape = (cond_shape == input_shape) && (input_shape == out_shape);
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const bool* cond_ptr = condition.data<bool>();
        const scalar_t* input_ptr = input.data<scalar_t>();
        scalar_t* out_ptr = out.data<scalar_t>();
        scalar_t other_val = static_cast<scalar_t>(other_scalar);
        
        if (same_shape) {
            for (size_t i = 0; i < total_elems; ++i) {
                out_ptr[i] = cond_ptr[i] ? input_ptr[i] : other_val;
            }
        } else {
            // Broadcasting
            std::vector<size_t> cond_bcast_strides(out_ndim, 0);
            std::vector<size_t> input_bcast_strides(out_ndim, 0);
            
            size_t cond_ndim = cond_shape.size();
            size_t input_ndim = input_shape.size();
            
            for (size_t i = 0; i < out_ndim; ++i) {
                size_t cond_dim_idx = cond_ndim - out_ndim + i;
                size_t input_dim_idx = input_ndim - out_ndim + i;
                
                if (cond_dim_idx < cond_ndim && cond_shape[cond_dim_idx] > 1) {
                    cond_bcast_strides[i] = cond_strides[cond_dim_idx];
                }
                if (input_dim_idx < input_ndim && input_shape[input_dim_idx] > 1) {
                    input_bcast_strides[i] = input_strides[input_dim_idx];
                }
            }
            
            std::vector<size_t> coords(out_ndim, 0);
            
            for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
                size_t temp_idx = linear_idx;
                for (int dim = out_ndim - 1; dim >= 0; --dim) {
                    coords[dim] = temp_idx % out_shape[dim];
                    temp_idx /= out_shape[dim];
                }
                
                size_t cond_idx = 0;
                size_t input_idx = 0;
                
                for (size_t dim = 0; dim < out_ndim; ++dim) {
                    cond_idx += coords[dim] * cond_bcast_strides[dim];
                    input_idx += coords[dim] * input_bcast_strides[dim];
                }
                
                out_ptr[linear_idx] = cond_ptr[cond_idx] ? input_ptr[input_idx] : other_val;
            }
        }
    });
}

// Variant 3: Both scalars
template<typename T, typename U>
inline void cpu_where_scalar_scalar(const Tensor& condition, T input_scalar, 
                              U other_scalar, Tensor& out) {
    size_t total_elems = out.numel();
    
    dispatch_by_dtype(out.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const bool* cond_ptr = condition.data<bool>();
        scalar_t* out_ptr = out.data<scalar_t>();
        scalar_t input_val = static_cast<scalar_t>(input_scalar);
        scalar_t other_val = static_cast<scalar_t>(other_scalar);
        
        for (size_t i = 0; i < total_elems; ++i) {
            out_ptr[i] = cond_ptr[i] ? input_val : other_val;
        }
    });
}

// ============================================================================
// DISPATCHER TEMPLATE IMPLEMENTATIONS
// ============================================================================

// Scalar input, Tensor other
template <typename T>
inline Tensor where(const Tensor& condition, T input_scalar, const Tensor& other) {
    //  1. Check if other tensor can be broadcasted to condition shape
    Shape output_shape = Shape{broadcast_rhs_to_lhs(condition.shape().dims, other.shape().dims)};
    
    //  2. Get dtype from scalar type T
    Dtype scalar_dtype = type_to_dtype<T>();
    
    //  3. Promote scalar dtype with tensor dtype
    Dtype promoted_dtype = promote_dtypes_bool(scalar_dtype, other.dtype());
    
    //  4. Promote other tensor if needed (no copy, just type conversion)
    Tensor other_promoted = (other.dtype() != promoted_dtype) ? other.as_type(promoted_dtype) : other;
    
    //  5. Create output tensor
    Tensor out(output_shape, promoted_dtype, condition.device(), false);
    
    //  6. Call scalar backend directly - NO tensor creation for scalar!
    if (condition.device().is_cuda()) {
        cuda_where_scalar_tensor(condition, input_scalar, other_promoted, out);
    } else {
        cpu_where_scalar_tensor(condition, input_scalar, other_promoted, out);
    }
    
    return out;
}

// Tensor input, Scalar other
template <typename T>
inline Tensor where(const Tensor& condition, const Tensor& input, T other_scalar) {
    //  1. Check if input tensor can be broadcasted to condition shape
    Shape output_shape = Shape{broadcast_rhs_to_lhs(condition.shape().dims, input.shape().dims)};
    
    //  2. Get dtype from scalar type T
    Dtype scalar_dtype = type_to_dtype<T>();
    
    //  3. Promote scalar dtype with tensor dtype
    Dtype promoted_dtype = promote_dtypes_bool(scalar_dtype, input.dtype());
    
    //  4. Promote input tensor if needed (no copy, just type conversion)  
    Tensor input_promoted = (input.dtype() != promoted_dtype) ? input.as_type(promoted_dtype) : input;
    
    //  5. Create output tensor
    Tensor out(output_shape, promoted_dtype, condition.device(), false);
    
    //  6. Call scalar backend directly - NO tensor creation for scalar!
    if (condition.device().is_cuda()) {
        cuda_where_tensor_scalar(condition, input_promoted, other_scalar, out);
    } else {
        cpu_where_tensor_scalar(condition, input_promoted, other_scalar, out);
    }
    
    return out;
}

// Both scalars (handles same-type and mixed-type)
template <typename T, typename U>
inline Tensor where(const Tensor& condition, T input_scalar, U other_scalar) {
    //  1. Get dtypes from both scalar types
    Dtype scalar_dtype1 = type_to_dtype<T>();
    Dtype scalar_dtype2 = type_to_dtype<U>();
    
    //  2. Promote the two scalar types (handles same-type correctly)
    Dtype output_dtype = promote_dtypes_bool(scalar_dtype1, scalar_dtype2);
    
    //  3. Create output tensor
    Tensor out(condition.shape(), output_dtype, condition.device(), false);
    
    //  4. Call scalar backend directly - NO tensor creation for scalars!
    if (condition.device().is_cuda()) {
        cuda_where_scalar_scalar(condition, input_scalar, other_scalar, out);
    } else {
        cpu_where_scalar_scalar(condition, input_scalar, other_scalar, out);
    }
    
    return out;
}

} // namespace OwnTensor

#endif