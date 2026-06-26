#pragma once

#include <vector>
#include <cstdint>

namespace OwnTensor {

/**
 * Shape: Represents tensor dimensions
 * Simple struct containing a vector of dimension sizes
 */
struct Shape {
    std::vector<int64_t> dims;
    
    // Equality operator
    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }
    
    // Inequality operator
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
};

} // namespace OwnTensor
