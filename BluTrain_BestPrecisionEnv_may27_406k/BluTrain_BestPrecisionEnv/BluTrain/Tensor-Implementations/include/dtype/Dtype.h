#pragma once

#ifndef DTYPE_H
#define DTYPE_H

namespace OwnTensor {
    // Core dtype enumeration used throughout the library
    enum class Dtype {
        Int8,Int16, Int32, Int64,UInt8,UInt16,UInt32,UInt64,
        Bfloat16, Float16, Float32, Float64,Bool,Complex32,Complex64,Complex128,
        Float4_e2m1, Float4_e2m1_2x
    };
} // namespace OwnTensor

#endif // DTYPE_H