#pragma once

#include "core/Tensor.h"

namespace OwnTensor {

/**
 * @brief Gathers values along an axis specified by dim.
 * 
 * For a 2D tensor, out[i][j] = input[index[i][j]][j] if dim == 0
 * For a 2D tensor, out[i][j] = input[i][index[i][j]] if dim == 1
 * 
 * Special case: If index has one fewer dimension than input, it is expanded.
 * e.g. input (B, C), dim 1, index (B) -> index treated as (B, 1), output (B, 1)
 */
Tensor gather(const Tensor& input, int64_t dim, const Tensor& index);

} // namespace OwnTensor
