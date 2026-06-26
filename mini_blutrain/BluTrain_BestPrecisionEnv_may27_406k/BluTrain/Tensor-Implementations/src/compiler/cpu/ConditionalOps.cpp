// tensor_kernels/cpu/ConditionalOps.cpp
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include <cstddef>
#include <vector>

namespace OwnTensor {

void cpu_where(const Tensor& condition, const Tensor& input,
               const Tensor& other, Tensor& out) {
    const auto& cond_shape = condition.shape().dims;
    const auto& input_shape = input.shape().dims;
    const auto& other_shape = other.shape().dims;
    const auto& out_shape = out.shape().dims;
    
    const auto& cond_strides = condition.stride().strides;
    const auto& input_strides = input.stride().strides;
    const auto& other_strides = other.stride().strides;
    
    size_t out_ndim = out_shape.size();
    size_t total_elems = out.numel();
    
    // Check if all tensors have the same shape (fast path, no broadcasting)
    bool same_shape = (cond_shape == input_shape) && (input_shape == other_shape) && (other_shape == out_shape);
    
    dispatch_by_dtype(input.dtype(), [&](auto dummy) {
        using scalar_t = decltype(dummy);
        const bool* cond_ptr = condition.data<bool>();
        const scalar_t* input_ptr = input.data<scalar_t>();
        const scalar_t* other_ptr = other.data<scalar_t>();
        scalar_t* out_ptr = out.data<scalar_t>();
        
        if (same_shape) {
            // Fast path: all same shape, direct element-wise operation
            for (size_t i = 0; i < total_elems; ++i) {
                out_ptr[i] = cond_ptr[i] ? input_ptr[i] : other_ptr[i];
            }
        } else {
            // Broadcasting path: calculate broadcast strides
            std::vector<size_t> cond_bcast_strides(out_ndim, 0);
            std::vector<size_t> input_bcast_strides(out_ndim, 0);
            std::vector<size_t> other_bcast_strides(out_ndim, 0);
            
            size_t cond_ndim = cond_shape.size();
            size_t input_ndim = input_shape.size();
            size_t other_ndim = other_shape.size();
            
            // Calculate broadcast strides for each tensor
            for (size_t i = 0; i < out_ndim; ++i) {
                size_t cond_dim_idx = cond_ndim - out_ndim + i;
                size_t input_dim_idx = input_ndim - out_ndim + i;
                size_t other_dim_idx = other_ndim - out_ndim + i;
                
                if (cond_dim_idx < cond_ndim && cond_shape[cond_dim_idx] > 1) {
                    cond_bcast_strides[i] = cond_strides[cond_dim_idx];
                }
                if (input_dim_idx < input_ndim && input_shape[input_dim_idx] > 1) {
                    input_bcast_strides[i] = input_strides[input_dim_idx];
                }
                if (other_dim_idx < other_ndim && other_shape[other_dim_idx] > 1) {
                    other_bcast_strides[i] = other_strides[other_dim_idx];
                }
            }
            
            // Iterate over output elements
            std::vector<size_t> coords(out_ndim, 0);
            
            for (size_t linear_idx = 0; linear_idx < total_elems; ++linear_idx) {
                // Calculate coordinates
                size_t temp_idx = linear_idx;
                for (int dim = out_ndim - 1; dim >= 0; --dim) {
                    coords[dim] = temp_idx % out_shape[dim];
                    temp_idx /= out_shape[dim];
                }
                
                // Calculate input indices using broadcast strides
                size_t cond_idx = 0;
                size_t input_idx = 0;
                size_t other_idx = 0;
                
                for (size_t dim = 0; dim < out_ndim; ++dim) {
                    cond_idx += coords[dim] * cond_bcast_strides[dim];
                    input_idx += coords[dim] * input_bcast_strides[dim];
                    other_idx += coords[dim] * other_bcast_strides[dim];
                }
                
                out_ptr[linear_idx] = cond_ptr[cond_idx] ? input_ptr[input_idx] : other_ptr[other_idx];
            }
        }
    });
}

// ============================================================================
// NOTE: Template scalar backend functions are now implemented inline in the
// header (ConditionalOps.h) to avoid explicit instantiations
// ============================================================================

} // namespace OwnTensor
