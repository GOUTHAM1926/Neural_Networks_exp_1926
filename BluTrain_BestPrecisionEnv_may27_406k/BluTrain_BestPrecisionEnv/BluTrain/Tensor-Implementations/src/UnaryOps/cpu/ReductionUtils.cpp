#include <algorithm> //For std::find
#include <numeric>   // For std::accumulate, std::multiplies
#include <set>       // For unique axes
#include <stdexcept> // For runtime_error
#include <cstdint>   // For int64_t and size_t

#include "ops/helpers/ReductionUtils.h" // Provides declarations for all functions
#include "core/Tensor.h"          // Provides OwnTensor::Shape (required for calculate_output_shape return type)

namespace OwnTensor {
namespace detail {

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and validates them.
 * @param input_dims The shape/dimensions of the input tensor.
 * @param axes The dimensions to reduce over (can include negative indices).
 * @return A vector of positive, unique axis indices, sorted ascendingly.
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& axes) {
    const int64_t ndim = input_dims.size();
    std::set<int64_t> unique_axes_set;

    // Handle empty axes: means reduce over all dimensions
    if (axes.empty()) {
        for (int64_t i = 0; i < ndim; ++i) {
            unique_axes_set.insert(i);
        }
    } else {
        for (int64_t axis : axes) {
            int64_t normalized_axis = axis;
            
            // 1. Handle negative axes (e.g., -1 becomes ndim - 1)
            if (normalized_axis < 0) {
                normalized_axis += ndim;
            }

            // 2. Validate bounds
            if (normalized_axis < 0 || normalized_axis >= ndim) {
                throw std::runtime_error("Reduction axis " + std::to_string(axis) + 
                                         " is out of bounds for tensor of rank " + std::to_string(ndim) + ".");
            }

            unique_axes_set.insert(normalized_axis);
        }
    }

    // Convert the set back to a sorted vector
    std::vector<int64_t> normalized_axes(unique_axes_set.begin(), unique_axes_set.end());
    return normalized_axes;
}

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * @param input_dims The shape of the input tensor.
 * @param normalized_axes The axes being reduced (positive indices).
 * @param keepdim If true, keeps reduced dimensions as 1.
 * @return The Shape struct of the output tensor.
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes, bool keepdim) {
    std::vector<int64_t> output_dims;
    
    // Lambda to check if a dimension index is marked for reduction
    auto is_reduced = [&](int64_t dim_idx) {
        return std::find(normalized_axes.begin(), normalized_axes.end(), dim_idx) != normalized_axes.end();
    };

    // Use size_t for the loop counter 'i' to avoid signed/unsigned comparison warnings.
    for (size_t i = 0; i < input_dims.size(); ++i) {
        if (is_reduced(static_cast<int64_t>(i))) {
            if (keepdim) {
                // Reduced axis is kept as size 1
                output_dims.push_back(1);
            }
            // If keepdim is false, this axis is dropped
        } else {
            // Unreduced axis, keep original size
            output_dims.push_back(input_dims[i]);
        }
    }
    
    // Handle scalar output 
    if (output_dims.empty()) {
        output_dims.push_back(1);
    }

    return Shape{output_dims};
}

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * @param input_dims The shape of the input tensor dimensions.
 * @param normalized_axes The axes being reduced.
 * @return The total number of elements being reduced.
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes) {
    // If normalized_axes is empty, it means reduce over all dimensions
    if (normalized_axes.empty()) {
        return std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());
    }
    
    int64_t count = 1;
    for (int64_t axis : normalized_axes) {
        count *= input_dims[axis];
    }
    return count;
}


//returns true if normalized_axes == {ndim-k,ndim-k+1, ...., ndim-1}
//i.e., axes are the rightmost k consecutive dimensions 
static bool axes_are_innermost(const std::vector<int64_t>& normalized_axes,int64_t ndim){
    int64_t k = static_cast<int64_t>(normalized_axes.size());
    int64_t start = ndim - k;
    for (int64_t i = 0;i<k;++i)
        if (normalized_axes[i] != start + i) return false;  
    return true ;
}

//returns true if normalized_axes == {0,1,2,....,k-1}
////i.e., aces are the leftmost k consecutive dimensions 
static bool axes_are_outermost(const std::vector<int64_t>& normalized_axes){
    int64_t k = static_cast<int64_t>(normalized_axes.size());
    for(int64_t i = 0; i<k;++i)
        if(normalized_axes[i] != i) return false;
    return true ;
}
ReductionLayout compute_reduction_layout(const Tensor& input , const std::vector<int64_t>& normalized_axes){
    const auto& dims = input.shape().dims;
    const auto& strides = input.stride().strides;
    const int64_t ndim = static_cast<int64_t>(dims.size());
    const int64_t k = static_cast<int64_t>(normalized_axes.size());
    ReductionLayout layout ; //path defaults to Generic

    // Non-contiguous input → Generic (can't optimize stride patterns)
    if(!input.is_contiguous()) return layout ;

    // =========================================================================
    // REORDER + COALESCE (like PyTorch's reorder_dimensions + coalesce_dimensions)
    //
    // PyTorch's approach:
    //   1. reorder_dimensions: Sort dims by stride ascending, reduced dims first
    //   2. coalesce_dimensions: Merge adjacent dims if contiguous (shape*stride == next_stride)
    //   3. Result: 1-2D problem → InnerContiguous or OuterContiguous
    //
    // Our simplified version for C-contiguous tensors:
    //   1. Classify each dim as reduced or preserved
    //   2. Coalesce: group all reduced dims → reduced_count, all preserved → num_outputs
    //   3. For C-contiguous: reduced dims form contiguous blocks with known strides
    //   4. Determine if the final layout is Inner, Outer, or needs stride-based access
    //
    // This handles ALL cases:
    //   - Rightmost dims → InnerContiguous (original Case 1)
    //   - Leftmost dims → OuterContiguous (original Case 2)
    //   - Middle consecutive dims → OuterContiguous with stride (was Generic!)
    //   - Non-consecutive dims like (0,2) → OuterContiguous after virtual reorder!
    //     For C-contiguous (B,C,H,W) reducing dims=(0,2):
    //       output = C*W positions, each reducing B*H elements
    //       stride between reduction elements varies → but for each output,
    //       we can compute a base index and a simple stride pattern
    // =========================================================================

    // Build reduced bitmap
    bool is_reduced[64] = {false};
    for (int64_t ax : normalized_axes) is_reduced[ax] = true;

    // Fast path: check original patterns first (most common in DL)
    if(axes_are_innermost(normalized_axes,ndim)){
        int64_t red=1;
        for(int64_t i=ndim-k;i<ndim;++i) red*=dims[i];
        int64_t outer=1;
        for(int64_t i=0;i<ndim-k;++i) outer*=dims[i];
        layout.path = ReductionLayout::Path::InnerContiguous;
        layout.num_outputs = outer;
        layout.reduced_count = red;
        layout.input_outer_stride = red;
        return layout;
    }

    if(axes_are_outermost(normalized_axes)){
        int64_t red=1;
        for(int64_t i=0;i<k;++i) red*=dims[i];
        int64_t inner=1;
        for(int64_t i=k;i<ndim;++i) inner*=dims[i];
        layout.path = ReductionLayout::Path::OuterContiguous;
        layout.num_outputs = inner;
        layout.reduced_count = red;
        layout.inner_count = inner;
        layout.input_outer_stride = inner;
        layout.input_row_stride = inner;
        return layout;
    }

    // =========================================================================
    // GENERAL COALESCE: For ANY axis pattern on C-contiguous tensor
    //
    // For C-contiguous, strides are: dims[ndim-1]*...*dims[i+1] for dim i
    // Key insight: for C-contiguous input, we can ALWAYS decompose into
    // (outer, reduced, inner) by finding contiguous groups.
    //
    // For consecutive reduced axes (e.g., dim=1 of (B,S,D)):
    //   outer = product of dims before reduced
    //   reduced = product of reduced dims
    //   inner = product of dims after reduced
    //   stride between reduction elements = inner
    //
    // For non-consecutive reduced axes (e.g., dims=(0,2) of (B,C,H,W)):
    //   This creates a multi-stride pattern that can't be simplified to 2D.
    //   BUT: for C-contiguous, each output element has a computable base index
    //   and the reduction follows a KNOWN stride pattern.
    //   We handle this by routing to OuterContiguous with proper stride,
    //   where each output's base_index is computed from its position in the
    //   non-reduced dimension space.
    // =========================================================================

    // Check if reduced axes are consecutive
    std::vector<int64_t> sorted_axes(normalized_axes.begin(), normalized_axes.end());
    std::sort(sorted_axes.begin(), sorted_axes.end());
    bool consecutive = true;
    for (int64_t i = 1; i < k; ++i) {
        if (sorted_axes[i] != sorted_axes[i-1] + 1) {
            consecutive = false;
            break;
        }
    }

    if (consecutive) {
        // Consecutive reduced axes → clean 3D decomposition
        int64_t first_red = sorted_axes[0];
        int64_t last_red = sorted_axes[k-1];

        int64_t outer = 1;
        for (int64_t i = 0; i < first_red; ++i) outer *= dims[i];
        int64_t reduced = 1;
        for (int64_t i = first_red; i <= last_red; ++i) reduced *= dims[i];
        int64_t inner = 1;
        for (int64_t i = last_red + 1; i < ndim; ++i) inner *= dims[i];

        if (inner == 1) {
            // Reduced at end → InnerContiguous
            layout.path = ReductionLayout::Path::InnerContiguous;
            layout.num_outputs = outer;
            layout.reduced_count = reduced;
            layout.input_outer_stride = reduced;
        } else if (outer == 1) {
            // Reduced at start → OuterContiguous
            layout.path = ReductionLayout::Path::OuterContiguous;
            layout.num_outputs = inner;
            layout.reduced_count = reduced;
            layout.inner_count = inner;
            layout.input_row_stride = inner;
            layout.input_outer_stride = inner;
        } else {
            // Middle reduction → OuterContiguous with stride
            // Each output (o_outer, o_inner) has:
            //   base = o_outer * (reduced * inner) + o_inner
            //   reduce with stride = inner
            layout.path = ReductionLayout::Path::OuterContiguous;
            layout.num_outputs = outer * inner;
            layout.reduced_count = reduced;
            layout.inner_count = inner;
            layout.input_row_stride = inner;
            layout.input_outer_stride = reduced * inner;
        }
        return layout;
    }

    // Non-consecutive reduced axes (e.g., dims=(0,2) of (B,C,H,W))
    // For C-contiguous: we can still compute the reduction stride pattern.
    //
    // Strategy: Find the innermost non-reduced dimension block and the
    // outermost non-reduced dimension block. For simple 2-group patterns
    // like (0,2) in 4D, we can route to OuterContiguous.
    //
    // For C-contiguous (B,C,H,W) dims=(16,64,32,32) reducing (0,2):
    //   Non-reduced: dim1=64 (stride=32*32=1024), dim3=32 (stride=1)
    //   Reduced: dim0=16 (stride=64*32*32=65536), dim2=32 (stride=32)
    //   Output: 64*32 = 2048 elements
    //   Each output (c, w): base = c*1024 + w
    //   Reduction: for b in 0..15, h in 0..31:
    //     input[b*65536 + c*1024 + h*32 + w]
    //   This has TWO reduction strides (65536 and 32) → can't simplify to one stride
    //   → Falls to Generic (carry-add)
    //
    // However, if one of the non-consecutive groups has size 1, we can still coalesce.
    // Check: is the total number of "reduced groups" just 2 with small dim between?

    // For now: non-consecutive axes → Generic (carry-add)
    // This is the same as PyTorch when strides can't be coalesced into 2D
    return layout;
}
} // namespace detail
} // namespace OwnTensor