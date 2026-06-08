#pragma once

#include "core/Tensor.h"  // This is the correct path

namespace OwnTensor {
namespace ViewUtils {

    // Computes row-major strides from a shape
    Stride compute_strides(const Shape& shape);
    
    // Checks if new shape has same number of elements
    bool is_shape_compatible(size_t numel, const Shape& new_shape);
    
    // Replaces -1 dimension with computed value
    void infer_dimension(size_t numel, Shape& shape);

    // Dimension helpers
    int normalize_dim(int dim, int ndim);                 // negative -> positive, bounds check throws
    void swap_dimensions(Shape& shape, Stride& stride, int dim0, int dim1);

    // Flatten helpers
    int64_t compute_flatten_size(const Shape& shape, int start_dim, int end_dim);
    Shape compute_flatten_shape(const Shape& shape, int start_dim, int end_dim);

    // Unflatten helpers
    void validate_unflatten(const Shape& shape, int dim, const Shape& sizes); // throws on mismatch
    Shape compute_unflatten_shape(const Shape& shape, int dim, const Shape& sizes);
    Stride compute_unflatten_strides(const Stride& old_stride, int dim, const Shape& sizes);

    // Smart reshape: returns valid Stride if `new_shape` can be expressed as a
    // pure VIEW of the source (no data movement); returns empty/invalid Stride
    // (strides.empty()) when a copy via .contiguous() is required.
    //
    // Algorithm matches PyTorch's at::detail::computeStride
    // (aten/src/ATen/TensorUtils.cpp:327). Walks old dims back-to-front, tracks
    // memory-contiguous chunks; for each chunk, drains view dims until their
    // product matches the chunk's element count. If all view dims fit cleanly
    // into chunks, a view-reshape is possible.
    //
    // Eliminates the .contiguous() call for cases like:
    //   q [B, T, 768] (non-contig from QKV split, inner stride=1)
    //     reshape → [B, T, 12, 64]   ← view-only OK (inner 768 is dense)
    Stride compute_view_stride(const Shape& old_shape, const Stride& old_stride,
                               const Shape& new_shape);

} // namespace ViewUtils
} // namespace OwnTensor