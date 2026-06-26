#pragma once

#include <vector>
#include <cstdint>

namespace OwnTensor {

/**
 * Stride: Represents tensor stride for each dimension
 * Used for memory layout calculations
 */
struct Stride {
    std::vector<int64_t> strides;
};

} // namespace OwnTensor
