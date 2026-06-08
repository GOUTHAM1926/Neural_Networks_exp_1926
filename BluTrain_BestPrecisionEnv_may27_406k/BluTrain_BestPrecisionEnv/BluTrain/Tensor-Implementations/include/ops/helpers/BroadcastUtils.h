#pragma once
#include "core/Tensor.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace OwnTensor {

inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& shape_a, 
                                          const std::vector<int64_t>& shape_b) {
    size_t ndim_a = shape_a.size();
    size_t ndim_b = shape_b.size();
    size_t max_ndim = std::max(ndim_a, ndim_b);
    
    std::vector<int64_t> result(max_ndim);
    
    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = (i < ndim_a) ? shape_a[ndim_a - 1 - i] : 1;
        int64_t dim_b = (i < ndim_b) ? shape_b[ndim_b - 1 - i] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        
        result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
    }
    
    return result;
}

inline bool broadcast_compatible(const std::vector<int64_t>& shape_a,
                               const std::vector<int64_t>& shape_b) {
    try {
        broadcast_shape(shape_a, shape_b);
        return true;
    } catch (...) {
        return false;
    }
}

// Check if rhs can be broadcasted to lhs shape and return the target shape
// Throws error if rhs has higher dimensionality than lhs or shapes are incompatible
inline std::vector<int64_t> broadcast_rhs_to_lhs(const std::vector<int64_t>& lhs_shape,
                                                 const std::vector<int64_t>& rhs_shape) {
    size_t lhs_ndim = lhs_shape.size();
    size_t rhs_ndim = rhs_shape.size();
    
    // Error if rhs has more dimensions than lhs
    if (rhs_ndim > lhs_ndim) {
        throw std::runtime_error("Cannot broadcast: rhs tensor has higher dimensionality (" 
                               + std::to_string(rhs_ndim) + ") than lhs tensor (" 
                               + std::to_string(lhs_ndim) + ")");
    }
    
    // Check if rhs can be broadcasted to lhs shape
    // Broadcasting rules: iterate from right to left, dimensions must be either:
    // 1. Equal
    // 2. rhs dimension is 1
    // 3. rhs dimension doesn't exist (implicit 1)
    for (size_t i = 0; i < lhs_ndim; ++i) {
        int64_t lhs_dim = lhs_shape[lhs_ndim - 1 - i];
        int64_t rhs_dim = (i < rhs_ndim) ? rhs_shape[rhs_ndim - 1 - i] : 1;
        
        // Check if broadcasting is valid for this dimension
        if (rhs_dim != lhs_dim && rhs_dim != 1) {
            throw std::runtime_error("Shapes are not broadcastable: rhs dimension " 
                                   + std::to_string(rhs_dim) + " cannot broadcast to lhs dimension " 
                                   + std::to_string(lhs_dim) + " at position " 
                                   + std::to_string(i) + " (from right)");
        }
    }
    
    // If all checks pass, return the lhs shape (target shape after broadcasting)
    return lhs_shape;
}

}

