// include/dtype/DtypeTraits.h - MERGED VERSION (Compatible with both versions)
#pragma once

#ifndef DTYPE_TRAIT_H
#define DTYPE_TRAIT_H

#include <cstdint>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "core/Tensor.h"
#include <cassert>
//  ALWAYS use custom structs (both CPU and GPU compilation)
#include "dtype/Types.h"
#include "dtype/fp4.h"

// ═══════════════════════════════════════════════════════════
// DTYPE TRAITS
// ═══════════════════════════════════════════════════════════

namespace OwnTensor {
    enum class Dtype;

    template <Dtype dt>
    struct dtype_traits {
        using type = void;
        static constexpr size_t size = 0;
        static constexpr const char* name = "invalid";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    // Integer Types
    template <> struct dtype_traits<Dtype::Int8> {
        using type = int8_t;
        static constexpr size_t size = sizeof(int8_t);
        static constexpr const char* name = "int8";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };
    template <> struct dtype_traits<Dtype::Int16> {
        using type = int16_t;
        static constexpr size_t size = sizeof(int16_t);
        static constexpr const char* name = "int16";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Int32> {
        using type = int32_t;
        static constexpr size_t size = sizeof(int32_t);
        static constexpr const char* name = "int32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Int64> {
        using type = int64_t;
        static constexpr size_t size = sizeof(int64_t);
        static constexpr const char* name = "int64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = false;
    };
//Unsigned int types
template<> struct dtype_traits<Dtype::UInt8> {
    using type = uint8_t;
    static constexpr size_t size = sizeof(uint8_t);
    static constexpr const char* name = "UInt8";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
        
};

template<> struct dtype_traits<Dtype::UInt16> {
    using type = uint16_t;
    static constexpr size_t size = sizeof(uint16_t);
    static constexpr const char* name = "UInt16";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

template<> struct dtype_traits<Dtype::UInt32> {
    using type = uint32_t;
    static constexpr size_t size = sizeof(uint32_t);
    static constexpr const char* name = "UInt32";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

template<> struct dtype_traits<Dtype::UInt64> {
    using type = uint64_t;
    static constexpr size_t size = sizeof(uint64_t);
    static constexpr const char* name = "UInt64";
    static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = true;
        static constexpr bool is_unsigned = true;
};

    // Floating Point Types -  used custom structs for custom floating point types
    template <> struct dtype_traits<Dtype::Float16> {
        using type = float16_t;  // Custom struct
        static constexpr size_t size = sizeof(float16_t);
        static constexpr const char* name = "fp16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Bfloat16> {   
        using type = bfloat16_t;  // Custom struct
        static constexpr size_t size = sizeof(bfloat16_t);
        static constexpr const char* name = "bf16";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float32> {
        using type = float;
        static constexpr size_t size = sizeof(float);
        static constexpr const char* name = "float32";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float64> {
        using type = double;
        static constexpr size_t size = sizeof(double);
        static constexpr const char* name = "float64";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float4_e2m1> {
        using type = float4_e2m1_t;
        static constexpr size_t size = sizeof(float4_e2m1_t); // 1 byte storage
        static constexpr const char* name = "float4_e2m1";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    template <> struct dtype_traits<Dtype::Float4_e2m1_2x> {
        using type = float4_e2m1_2x_t;
        static constexpr size_t size = sizeof(float4_e2m1_2x_t); // 1 byte storage (2 packed)
        static constexpr const char* name = "float4_e2m1_2x";
        static constexpr bool is_floating_point = true;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
    };

    // bool specialization
    template<> struct dtype_traits<Dtype::Bool> {
        static constexpr const char* name = "bool";
        static constexpr Dtype dtype = Dtype::Bool;
        static constexpr size_t size = sizeof(bool);  // Usually 1 byte
        using type = bool;
        // using storage_type = uint8_t;
    };

    // Complex specification - used custom struct for custom complex struct (chalf)
    
    template<> struct dtype_traits<Dtype::Complex32>{
        using type = complex32_t;
        static constexpr Dtype dtype = Dtype::Complex32;
        static constexpr size_t size= sizeof(complex32_t);
        static constexpr const char* name = "complex32";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned =false;
        static constexpr bool is_complex = true;
    };

    template<> struct dtype_traits<Dtype::Complex64>{
        using type = complex64_t;
        static constexpr Dtype dtype = Dtype::Complex64;
        static constexpr size_t size = sizeof(complex64_t);
        static constexpr const char* name = "complex64";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = true;
    };
 
    template<> struct dtype_traits<Dtype::Complex128>{
        using type = complex128_t;
        static constexpr Dtype dtype = Dtype::Complex128;
        static constexpr size_t size = sizeof(complex128_t);
        static constexpr const char* name = "complex128";
        static constexpr bool is_floating_point = false;
        static constexpr bool is_integral = false;
        static constexpr bool is_unsigned = false;
        static constexpr bool is_complex = true;
    };
    // Helper function
    template<typename T>
    bool is_same_type(Dtype dtype) {
        if constexpr (std::is_same_v<T, int32_t>) {
            return dtype == Dtype::Int32;
        }else if constexpr(std::is_same_v<T,int8_t>){
            return dtype == Dtype::Int8;
        }
         else if constexpr (std::is_same_v<T, float>) {
            return dtype == Dtype::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return dtype == Dtype::Float64;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return dtype == Dtype::Int16;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return dtype == Dtype::Int64;
        } 
        else if constexpr (std::is_same_v<T, uint8_t>) {
            return dtype == Dtype::UInt8;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return dtype == Dtype::UInt16;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return dtype == Dtype::UInt32;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return dtype == Dtype::UInt64;
        }
        else if constexpr (std::is_same_v<T, float16_t>) {
            return dtype == Dtype::Float16; 
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return dtype == Dtype::Bfloat16;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return dtype == Dtype::Bool;
        }else if constexpr (std::is_same_v<T, complex32_t>){
            return dtype == Dtype::Complex32;
        }else if constexpr (std::is_same_v<T,complex64_t>){
            return dtype == Dtype::Complex64;
        }else if constexpr (std::is_same_v<T, complex128_t>){
            return dtype == Dtype::Complex128;
        } else if constexpr (std::is_same_v<T, float4_e2m1_t>) {
            return dtype == Dtype::Float4_e2m1;
        } else if constexpr (std::is_same_v<T, float4_e2m1_2x_t>) {
            return dtype == Dtype::Float4_e2m1_2x;
        }
        return false;
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE TO DTYPE CONVERSION
    // ═══════════════════════════════════════════════════════════

    template<typename T>
    constexpr Dtype type_to_dtype() {
        // Integer types
        if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, short>) {
            return Dtype::Int16;
        }
        else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>){
            return Dtype::Int8;
        }
        else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
            return Dtype::Int32;
        }
        else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, long> || 
                           std::is_same_v<T, long long>) {
            return Dtype::Int64;
        }
        // Unsigned integer types
    else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, unsigned char>) {
        return Dtype::UInt8;
    } else if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, unsigned short>) {
        return Dtype::UInt16;
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, unsigned int>) {
        return Dtype::UInt32;
    } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned long long>) {
        return Dtype::UInt64;
    }
        // Floating point types - custom structs
        else if constexpr (std::is_same_v<T, float16_t>) {
            return Dtype::Float16;
        }
        else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return Dtype::Bfloat16;
        }
        else if constexpr (std::is_same_v<T, float>) {
            return Dtype::Float32;
        }
        else if constexpr (std::is_same_v<T, double>) {
            return Dtype::Float64;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return Dtype::Bool;
        }
        else if constexpr (std::is_same_v<T, complex32_t>){
            return Dtype::Complex32;
       }
        else if constexpr (std::is_same_v<T, complex64_t> ){
            return Dtype::Complex64;
        }else if constexpr (std::is_same_v<T,complex128_t> ){
            return Dtype::Complex128;
        }
        else if constexpr (std::is_same_v<T, float4_e2m1_t>) {
            return Dtype::Float4_e2m1;
        }
        else if constexpr (std::is_same_v<T, float4_e2m1_2x_t>) {
            return Dtype::Float4_e2m1_2x;
        }
        else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // TYPE PREDICATES 
    // ═══════════════════════════════════════════════════════════

    constexpr bool is_float(Dtype dt) {
        switch (dt) {
        case Dtype::Float16:
        case Dtype::Bfloat16:
        case Dtype::Float32:
        case Dtype::Float64:
        case Dtype::Float4_e2m1:
        case Dtype::Float4_e2m1_2x:
            return true;
        default:
            return false;
        }
    }

    constexpr bool is_int(Dtype dt) {
        switch (dt) {
        case Dtype::Int16:
        case Dtype::Int32:
        case Dtype::Int64:
        case Dtype::UInt8:
        case Dtype::UInt16:
        case Dtype::UInt32:
        case Dtype::UInt64:
            return true;
        default:
            return false;
        }
    }
    constexpr bool is_unsigned(Dtype dt){
        switch (dt){
            case Dtype::UInt8:
            case Dtype::UInt16:
            case Dtype::UInt32:
            case Dtype::UInt64:
                return true;
            default :
                return false;
        }
    }
    constexpr bool is_bool(Dtype dt) {
        return dt == Dtype::Bool;
    }
    constexpr bool is_complex(Dtype dt) {
        switch (dt){
            case Dtype::Complex32:
            case Dtype::Complex64:
            case Dtype::Complex128:
                return true;
            default :
                return false;
        }
    }
    // ═══════════════════════════════════════════════════════════
    // DTYPE NAME HELPER ( Returns std::string )
    // ═══════════════════════════════════════════════════════════
//Before, returned const char*
    // Now,using std::string for consistency with teammate's code
    inline std::string get_dtype_name(Dtype dtype) {
        switch(dtype) {
            case Dtype::Int8:     return "int8";
            case Dtype::Int16:    return "int16";
            case Dtype::Int32:    return "int32";
            case Dtype::Int64:    return "int64";
            case Dtype::UInt8:    return "uint8";
            case Dtype::UInt16:   return "uint16";
            case Dtype::UInt32:   return "uint32";
            case Dtype::UInt64:   return "uint64";
            case Dtype::Float16:  return "float16";  
            case Dtype::Bfloat16: return "bfloat16";
            case Dtype::Float32:  return "float32";
            case Dtype::Float64:  return "float64";
            case Dtype::Bool:     return "bool";
            case Dtype::Complex32: return "complex32";
            case Dtype::Complex64: return "complex64";
            case Dtype::Complex128: return "complex128";
            case Dtype::Float4_e2m1: return "float4_e2m1";
            case Dtype::Float4_e2m1_2x: return "float4_e2m1_2x";
            default: return "Unknown";  
        }
    }

// ═══════════════════════════════════════════════════════════════════════════
// DTYPE PROMOTION SYSTEM
// ═══════════════════════════════════════════════════════════════════════════
// Three promotion functions:
//   1. promote_tensor_ops(a, b)     → Tensor + Tensor operations
//   2. promote_scalar_ops(t, s)     → Tensor + C++ scalar operations
//   3. promote_dtypes_division(a,b) → Division (always promotes to float)
//
// All three use O(1) lookup tables for fast dispatch.
//
// BACK-COMPAT SHIM:
//   - `promote_dtypes_bool(a, b)` is an alias for `promote_tensor_ops` so that
//     existing callers keep working. New code should call the explicit name
//     for the situation (tensor+tensor vs tensor+scalar).
// ═══════════════════════════════════════════════════════════════════════════

constexpr int DTYPE_COUNT = 18;

// Error markers (negative enum values not used by Dtype)
constexpr int ERR_MIXED_FP4       = -1;  // Float4_e2m1 + Float4_e2m1_2x
constexpr int ERR_FP4_HIGHER      = -2;  // FP4 + Float16/32/64/BFloat16/Complex
constexpr int ERR_S_UINT_INT      = -4;  // UInt tensor + Int scalar (sign mismatch)
constexpr int ERR_S_UINT_LARGER   = -5;  // UInt tensor + larger UInt scalar
constexpr int ERR_S_INT_LARGER    = -6;  // Int tensor + larger Int/UInt scalar

// Index shorthand (matches Dtype enum order, see Dtype.h)
constexpr int I8 = 0, I16 = 1, I32 = 2, I64 = 3;
constexpr int U8 = 4, U16 = 5, U32 = 6, U64 = 7;
constexpr int BF16 = 8, F16 = 9, F32 = 10, F64 = 11;
constexpr int BOOL = 12;
constexpr int C32 = 13, C64 = 14, C128 = 15;
constexpr int FP4_E2M1 = 16, FP4_E2M1_2X = 17;

constexpr int dtype_to_idx(Dtype d) { return static_cast<int>(d); }

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR + TENSOR PROMOTION TABLE
// ═══════════════════════════════════════════════════════════════════════════
// promotion_table[a][b] = resulting dtype index
// Symmetric: promotion_table[a][b] == promotion_table[b][a]
//
// RULES:
// 1. Same type → return that type (diagonal)
// 2. Complex > Float > Int > UInt > Bool
// 3. Int promotion: Int64 > Int32 > Int16 > Int8
// 4. UInt + UInt: larger wins
// 5. UInt8 + Int* → safe widening to Int16/32/64
// 6. UInt16 + Int* → Int32/64 (NumPy-style)
// 7. UInt32 + Int* → Int64 (NumPy-style)
// 8. UInt64 + Int* → Float32 (no integer can hold both ranges)
// 9. FP4 rows/cols COMMENTED OUT — added later when FP4 support is ready
// ═══════════════════════════════════════════════════════════════════════════

constexpr int promotion_table[DTYPE_COUNT][DTYPE_COUNT] = {
//        I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 FP4  FP4_2X
/*I8   */ {I8,  I16, I32, I64, I16, I32, I64, F32, BF16, F16, F32, F64, I8,  C32, C64, C128, 0, 0},
/*I16  */ {I16, I16, I32, I64, I16, I32, I64, F32, BF16, F16, F32, F64, I16, C32, C64, C128, 0, 0},
/*I32  */ {I32, I32, I32, I64, I32, I32, I64, F32, BF16, F16, F32, F64, I32, C64, C64, C128, 0, 0},
/*I64  */ {I64, I64, I64, I64, I64, I64, I64, F32, BF16, F16, F32, F64, I64, C64, C64, C128, 0, 0},
/*U8   */ {I16, I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F64, U8,  C32, C64, C128, 0, 0},
/*U16  */ {I32, I32, I32, I64, U16, U16, U32, U64, BF16, F16, F32, F64, U16, C64, C64, C128, 0, 0},
/*U32  */ {I64, I64, I64, I64, U32, U32, U32, U64, BF16, F16, F32, F64, U32, C64, C64, C128, 0, 0},
/*U64  */ {F32, F32, F32, F32, U64, U64, U64, U64, BF16, F16, F32, F64, U64, C64, C64, C128, 0, 0},
/*BF16 */ {BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, F32, F32, F64, BF16, C64, C64, C128, 0, 0},
/*F16  */ {F16, F16, F16, F16, F16, F16, F16, F16, F32, F16, F32, F64, F16, C32, C64, C128, 0, 0},
/*F32  */ {F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F64, F32, C64, C64, C128, 0, 0},
/*F64  */ {F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128, C128, 0, 0},
/*BOOL */ {I8,  I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F64, BOOL, C32, C64, C128, 0, 0},
/*C32  */ {C32, C32, C64, C64, C32, C64, C64, C64, C64, C32, C64, C128, C32, C32, C64, C128, 0, 0},
/*C64  */ {C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C128, C64, C64, C64, C128, 0, 0},
/*C128 */ {C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, 0, 0},
// FP4 ROWS COMMENTED OUT — not yet supported in scalar-op path.
// Uncomment and fill in when enabling FP4 scalar-op support.
// /*FP4  */ {FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, FP4, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, FP4, ERR_MIXED_FP4},
// /*FP4x */ {FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, FP4x, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_FP4_HIGHER, ERR_MIXED_FP4, FP4x},
// Placeholder rows for enum indices 16, 17 so the 2D table stays 18x18:
/*FP4  */ {0}, /*FP4x */ {0},
};

// ═══════════════════════════════════════════════════════════════════════════
// Tensor + Tensor promotion
// ═══════════════════════════════════════════════════════════════════════════
inline Dtype promote_tensor_ops(Dtype a, Dtype b) {
    int idx_a = dtype_to_idx(a);
    int idx_b = dtype_to_idx(b);
    // Reject FP4 for now — rows/cols are placeholders (all zeros)
    if (idx_a >= FP4_E2M1 || idx_b >= FP4_E2M1) {
        throw std::runtime_error("FP4 tensor-tensor promotion is not yet supported.");
    }
    int result = promotion_table[idx_a][idx_b];
    if (result == ERR_MIXED_FP4)
        throw std::runtime_error("Mixed FP4 variants have no implicit promotion path.");
    if (result == ERR_FP4_HIGHER)
        throw std::runtime_error("FP4 does not support implicit promotion with higher-precision types.");
    return static_cast<Dtype>(result);
}

// Back-compat alias — existing callers (e.g. ScalarOpsDispatcher) use this name
inline Dtype promote_dtypes_bool(Dtype a, Dtype b) {
     // Rule 1: If both are same type, return that type
    if (a == b) return a;
    // Rule 2: Complex types have highest priority
if (a == Dtype::Complex128 || b == Dtype::Complex128) return Dtype::Complex128;
if (a == Dtype::Complex64 || b == Dtype::Complex64) return Dtype::Complex64;
if (a == Dtype::Complex32 || b == Dtype::Complex32) return Dtype::Complex32;
    // Rule 3: Floating point always wins (highest precision first)
    if (a == Dtype::Float64 || b == Dtype::Float64) return Dtype::Float64;
    if (a == Dtype::Float32 || b == Dtype::Float32) return Dtype::Float32;
    if (a == Dtype::Float16 || b == Dtype::Float16) return Dtype::Float16;
    if (a == Dtype::Bfloat16 || b == Dtype::Bfloat16) return Dtype::Bfloat16;
    if (a == Dtype::Float4_e2m1 || b == Dtype::Float4_e2m1) return Dtype::Float4_e2m1;
    if (a == Dtype::Float4_e2m1_2x || b == Dtype::Float4_e2m1_2x) return Dtype::Float4_e2m1_2x;
    
    // Rule 4: Integer promotion (largest size wins)
    if (a == Dtype::Int64 || b == Dtype::Int64) return Dtype::Int64;
    if (a == Dtype::Int32 || b == Dtype::Int32) return Dtype::Int32;
    if (a == Dtype::Int16 || b == Dtype::Int16) return Dtype::Int16;
    if (a == Dtype::Int8 || b== Dtype::Int8) return Dtype::Int8;
     // Only allow uint8 promotion; others fallback to error/unknown
     if (a==Dtype::UInt8 || b==Dtype::UInt8) return Dtype::UInt8;
       // For other unsigned types, handle as error, assert, or define explicit cast logic
       if (a==Dtype::UInt16 || b==Dtype::UInt16 || a==Dtype::UInt32 || b==Dtype::UInt32 || a==Dtype::UInt64 || b==Dtype::UInt64) {
        static_assert("Promotion for uint16, uint32, uint64 is not supported; cast required.");

       }
    // Rule 5: Bool + Bool = Bool
    return Dtype::Bool;
    // return promote_tensor_ops(a, b);
}

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR + SCALAR PROMOTION TABLE (PyTorch-style, scalars are weak)
// ═══════════════════════════════════════════════════════════════════════════
// scalar_tensor_table[tensor_idx][scalar_idx] = result dtype index
//
// KEY DIFFERENCES from tensor-tensor promotion:
//   • Float tensor + any non-complex scalar → tensor dtype wins (scalar is weak)
//   • Float tensor + complex scalar → promotes to preserve complex
//   • Int tensor + larger Int/UInt scalar → ERROR (explicit cast required)
//   • UInt tensor + Int scalar → ERROR (sign mismatch)
//   • UInt tensor + larger UInt scalar → ERROR
//   • Bool tensor → scalar dtype wins (bool is the weakest)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int scalar_tensor_table[DTYPE_COUNT][DTYPE_COUNT] = {
//              I8   I16  I32  I64  U8   U16  U32  U64  BF16 F16  F32  F64  BOOL C32  C64  C128 FP4  FP4_2X
/*T:I8  */ {I8,  ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F16, F32, F32, I8,  C32, C64, C64, 0, 0},
/*T:I16 */ {I16, I16, ERR_S_INT_LARGER, ERR_S_INT_LARGER, I16, ERR_S_INT_LARGER, ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F16, F32, F32, I16, C32, C64, C64, 0, 0},
/*T:I32 */ {I32, I32, I32, ERR_S_INT_LARGER, I32, I32, ERR_S_INT_LARGER, ERR_S_INT_LARGER, BF16, F32, F32, F32, I32, C64, C64, C64, 0, 0},
/*T:I64 */ {I64, I64, I64, I64, I64, I64, I64, ERR_S_INT_LARGER, F32, F32, F32, F32, I64, C64, C64, C64, 0, 0},
/*T:U8  */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U8,  ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, BF16, F16, F32, F32, U8,  C32, C64, C64, 0, 0},
/*T:U16 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U16, U16, ERR_S_UINT_LARGER, ERR_S_UINT_LARGER, BF16, F16, F32, F32, U16, C32, C64, C64, 0, 0},
/*T:U32 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U32, U32, U32, ERR_S_UINT_LARGER, BF16, F32, F32, F32, U32, C64, C64, C64, 0, 0},
/*T:U64 */ {ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, ERR_S_UINT_INT, U64, U64, U64, U64, F32, F32, F32, F32, U64, C64, C64, C64, 0, 0},
/*T:BF16*/ {BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, C64, C64, C64, 0, 0},
/*T:F16 */ {F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, F16, C32, C64, C64, 0, 0},
/*T:F32 */ {F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, F32, C64, C64, C64, 0, 0},
/*T:F64 */ {F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128, C128, 0, 0},
/*T:BOOL*/ {I8,  I16, I32, I64, U8,  U16, U32, U64, BF16, F16, F32, F32, BOOL, C32, C64, C64, 0, 0},
/*T:C32 */ {C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, C32, 0, 0},
/*T:C64 */ {C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, C64, 0, 0},
/*T:C128*/ {C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, C128, 0, 0},
// FP4 TENSOR/SCALAR ROWS COMMENTED OUT — not yet supported.
// Uncomment and fill in when enabling FP4 scalar-op support.
// /*T:FP4  */ {FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, FP4, C32, C64, C64, FP4, FP4},
// /*T:FP4x */ {FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, FP4x, C32, C64, C64, FP4x, FP4x},
/*T:FP4  */ {0}, /*T:FP4x */ {0},
};

inline Dtype promote_scalar_ops(Dtype tensor_dt, Dtype scalar_dt) {
    int t_idx = dtype_to_idx(tensor_dt);
    int s_idx = dtype_to_idx(scalar_dt);
    if (t_idx >= FP4_E2M1 || s_idx >= FP4_E2M1) {
        throw std::runtime_error("FP4 scalar-op promotion is not yet supported.");
    }
    int result = scalar_tensor_table[t_idx][s_idx];
    if (result == ERR_S_UINT_INT)
        throw std::runtime_error("UInt tensor + int scalar: not supported (signed/unsigned mismatch). Cast tensor first.");
    if (result == ERR_S_UINT_LARGER)
        throw std::runtime_error("UInt tensor + larger uint scalar: not supported. Cast tensor first.");
    if (result == ERR_S_INT_LARGER)
        throw std::runtime_error("Int tensor + larger int/uint scalar: not supported. Cast tensor first.");
    return static_cast<Dtype>(result);
}

// ═══════════════════════════════════════════════════════════════════════════
// DIVISION PROMOTION (always goes to float)
// ═══════════════════════════════════════════════════════════════════════════
inline Dtype promote_dtypes_division(Dtype a, Dtype b) {
    // Complex: use normal tensor-tensor promotion (Complex64 / Int32 → Complex64)
    if (a == Dtype::Complex32 || a == Dtype::Complex64 || a == Dtype::Complex128 ||
        b == Dtype::Complex32 || b == Dtype::Complex64 || b == Dtype::Complex128) {
        return promote_tensor_ops(a, b);
    }
    // If either operand is already float, use the highest-precision float
    if (a == Dtype::Float64 || b == Dtype::Float64) return Dtype::Float64;
    if (a == Dtype::Float32 || b == Dtype::Float32) return Dtype::Float32;
    if (a == Dtype::Float16 || b == Dtype::Float16) return Dtype::Float16;
    if (a == Dtype::Bfloat16 || b == Dtype::Bfloat16) return Dtype::Bfloat16;
    if (a == Dtype::Float4_e2m1 || b == Dtype::Float4_e2m1) return Dtype::Float4_e2m1;
    if (a == Dtype::Float4_e2m1_2x || b == Dtype::Float4_e2m1_2x) return Dtype::Float4_e2m1_2x;
    
    //  Otherwise, promote integers and bool to Float32
    // This matches PyTorch's behavior: Int16 / Bool → Float32
    return Dtype::Float32;
}
} // namespace OwnTensor

#endif // DTYPE_TRAIT_H