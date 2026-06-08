#pragma once

#include "core/Tensor.h" // Provides OwnTensor::Tensor and OwnTensor::Shape
#include <vector>
#include <cstdint>
#include <numeric>   // For std::accumulate
#include <set>       // For unique axes check
#include <stdexcept> // For runtime_error

namespace OwnTensor {
namespace detail { // <<< START OF THE INTERNAL DETAIL NAMESPACE

//=========================================================================
// REDUCTION LAYOUT - describes the simplified post-coalesce problem shape (this is a hybrid version designed by gautam_k_1926 similar to functionalities of reorder_dimensions() and coalesce_dimensions() functions in pytorch)
//=========================================================================
//describes which kernel path to take and the 2D parameters needed ,
// computed once before reduce_kernel and replaces per-element div/mod(from unravel_index and ravel_index that we had in earlier naive implementation) ,
// Path 1 ---> InnerContiguous ---> horizantal SIMD (reduce rightmost dims , C - contiguous input );
// Path 2 ---> OuterContiguous ---> vertical SIMD (reduce leftmost dims, C-contiguous input );
// Path 3 ---> Generic  ----> no SIMD , naive reduction loop fallback,generic basic loop
struct ReductionLayout {
    enum class Path : uint8_t {
        InnerContiguous,
        OuterContiguous,
        Generic
    };
    Path path = Path::Generic;
    //InnerContiguous params
    //Outer loop (OpenMP) : for i in 0--->num_outputs
    // in_ptr = input_data + i * input_outer_stride     (element offset)
    // Reduce in_ptr[0.... reduced_count-1] ---> output_data[i]
    // Access is perfectly sequential --> cache-friendly ---> horizantal SIMD
    int64_t num_outputs = 0 ; // independent output elements (outer loop)
    int64_t reduced_count = 0 ; //element per output slice (inner tight loop)
    int64_t input_outer_stride = 0 ; // element stride between slices ( = reduced_count forC-contiguous)

    //OuterContiguous params (also uses num_outputs = inner_count )
    // Reduction loop : for r in (0 ---> reduced_count)
    // in_row = input_data + r * input_row_stride (element offset)
    //for j in (0 ---> inner_count -1) : out[j] = op(out[j],in_row[j])
    // Each SIMD lane accumulates a different output position ---> veritcal SIMD
    int64_t inner_count = 0 ; // preserved innermost dim size (= num_outputs for  outer path)
    int64_t input_row_stride =  0 ; //element stride between reduction rows ( = inner_count for C-contiguous)

};
// ========================================================================
// FUNCTIONS (DECLARATIONS ONLY - IMPLEMENTED IN .CPP)
// ========================================================================

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and validates them.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims,
                                    const std::vector<int64_t>& axes);

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims,
                             const std::vector<int64_t>& normalized_axes,
                             bool keepdim);

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * DECLARED here, IMPLEMENTED in ReductionUtils.cpp
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims,
                                const std::vector<int64_t>& normalized_axes);
/*
* @brief computes the simplified 2D layout of a reduction after coalescing .
*Detects whether input is C-contiguous and axes are innermost or outermost ,
*allowing vectorized_inner_reduction (horizantal SIMD)  or vectorized_outer_reduction (vertical SIMD) to be used ,
* falls back to Path::Generic for non-contiguous or mized-axis reductions .
*Declared here,implemented in reductionUtils.cpp
*/
ReductionLayout compute_reduction_layout(const Tensor& input,const std::vector<int64_t>& normalized_axes);

} // namespace detail
} // namespace OwnTensor
