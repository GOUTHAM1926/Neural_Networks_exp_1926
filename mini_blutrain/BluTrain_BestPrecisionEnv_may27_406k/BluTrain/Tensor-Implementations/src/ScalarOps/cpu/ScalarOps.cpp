// ScalarOps.cpp - FIXED DIVISION OPERATORS
#include <cstdint>
#include <stdexcept>
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "dtype/DtypeTraits.h"  //  For promote_dtypes_division

namespace OwnTensor {

// Helper trait for complex types (compile-time guard)
template <typename T> struct is_complex_t : std::false_type {};
template <> struct is_complex_t<complex32_t> : std::true_type {};
template <> struct is_complex_t<complex64_t> : std::true_type {};
template <> struct is_complex_t<complex128_t> : std::true_type {};

namespace { // file-local helpers

inline bool is_integer_dtype(Dtype dt) {
    return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
}

inline float load_u16_as_f32(uint16_t bits, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float16_to_float(bits);
    if (dt == Dtype::Bfloat16) return detail::bfloat16_to_float(bits);
    return static_cast<float>(bits);
}

inline bool is_complex_dtype(Dtype dt) {
    return dt == Dtype::Complex32 || dt == Dtype::Complex64 || dt == Dtype::Complex128;
}

inline uint16_t store_f32_to_u16(float v, Dtype dt) {
    if (dt == Dtype::Float16)  return detail::float_to_float16(v);
    if (dt == Dtype::Bfloat16) return detail::float_to_bfloat16(v);
    return static_cast<uint16_t>(v);
}

// template <typename T>
// inline double ld(const T* p, size_t i, Dtype) { return static_cast<double>(p[i]); }

// // Specializations for complex types (can't convert complex to double)
// template <>
// inline double ld<complex32_t>(const complex32_t*, size_t, Dtype) {
//    throw std::runtime_error("Cannot perform scalar operations on complex32 types");
// }

// template <>
// inline double ld<complex64_t>(const complex64_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on complex64 types");
// }

// template <>
// inline double ld<complex128_t>(const complex128_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on complex128 types");
// }

// template <>
// inline double ld<uint16_t>(const uint16_t* p, size_t i, Dtype dt) {
//     return static_cast<double>(load_u16_as_f32(p[i], dt));
// }

// template <>
// inline double ld<float4_e2m1_2x_t>(const float4_e2m1_2x_t*, size_t, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on packed FP4 types");
// }

// template <typename T>
// inline void st(T* p, size_t i, double v, Dtype) { p[i] = static_cast<T>(v); }

// template <>
// inline void st<float4_e2m1_2x_t>(float4_e2m1_2x_t*, size_t, double, Dtype) {
//     throw std::runtime_error("Cannot perform scalar operations on packed FP4 types");
// }


// template <>
// inline void st<uint16_t>(uint16_t* p, size_t i, double v, Dtype dt) {
//     p[i] = store_f32_to_u16(static_cast<float>(v), dt);
// }

// template <typename T, typename F>
// inline void apply_inplace(T* data, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) st<T>(data, i, f(ld<T>(data, i, dt)), dt);
// }


// template <typename T, typename F>
// inline void apply_copy(const T* src, T* dst, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) st<T>(dst, i, f(ld<T>(src, i, dt)), dt);
// }

// // Special version for comparison ops that write bool
// template <typename T, typename F>
// inline void apply_copy_to_bool(const T* src, uint8_t* dst, size_t n, Dtype dt, F&& f) {
//     for (size_t i = 0; i < n; ++i) {
//         dst[i] = f(ld<T>(src, i, dt)) ? 1 : 0;
//     }
// }

//  NEW: Helper to determine promoted dtype for division
inline Dtype get_division_output_dtype(Dtype input_dtype) {
    // Bool → Float32
    if (input_dtype == Dtype::Bool) return Dtype::Float32;
    
    // Integer types → Float32
    if (is_integer_dtype(input_dtype)) return Dtype::Float32;
    
    // Float types → Keep same (Float16/BFloat16/Float32/Float64)
    return input_dtype;
}

} // anon


// --------- Non-templated CPU backends (scalar as double) ---------
// Scalar is cast to tensor's dtype inside each dispatch. Mirrors PyTorch's
// c10::Scalar approach. Collapses 260 instantiations → 20 function defs.

void cpu_add_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* p = t.data<T>();
            for (size_t i = 0, n = t.numel(); i < n; ++i) p[i] = p[i] + sv;
        }
    });
}

void cpu_sub_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* p = t.data<T>();
            for (size_t i = 0, n = t.numel(); i < n; ++i) p[i] = p[i] - sv;
        }
    });
}

void cpu_mul_inplace(Tensor& t, double s) {
    dispatch_by_dtype(t.dtype(), [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* p = t.data<T>();
            for (size_t i = 0, n = t.numel(); i < n; ++i) p[i] = p[i] * sv;
        }
    });
}

void cpu_div_inplace(Tensor& t, double s) {
    const Dtype dt = t.dtype();
    if (s == 0.0) throw std::runtime_error("Division by zero");
    Dtype promoted_dt = get_division_output_dtype(dt);
    if (promoted_dt != dt) {
        throw std::runtime_error(
            "In-place division /= requires float dtype. "
            "Input is " + get_dtype_name(dt) + " but division needs " +
            get_dtype_name(promoted_dt) + ". Use regular division (/) instead."
        );
    }
    dispatch_by_dtype(dt, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* p = t.data<T>();
            for (size_t i = 0, n = t.numel(); i < n; ++i) p[i] = p[i] / sv;
        }
    });
}

void cpu_add_copy(const Tensor& a, double s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if (is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar ops are not supported for complex dtypes");
    }
    Tensor b = (dt != promote_dtype) ? a.as_type(promote_dtype) : a;
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* src = b.data<T>();
            T* dst = output.data<T>();
            for (size_t i = 0, n = b.numel(); i < n; ++i) dst[i] = src[i] + sv;
        }
    });
}

void cpu_sub_copy(const Tensor& a, double s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if (is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar ops are not supported for complex dtypes");
    }
    Tensor b = (dt != promote_dtype) ? a.as_type(promote_dtype) : a;
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* src = b.data<T>();
            T* dst = output.data<T>();
            for (size_t i = 0, n = b.numel(); i < n; ++i) dst[i] = src[i] - sv;
        }
    });
}

void cpu_mul_copy(const Tensor& a, double s, Tensor& output, Dtype promote_dtype) {
    Dtype dt = a.dtype();
    if (is_complex_dtype(dt) || is_complex_dtype(output.dtype())) {
        throw std::runtime_error("Scalar ops are not supported for complex dtypes");
    }
    Tensor b = (dt != promote_dtype) ? a.as_type(promote_dtype) : a;
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* src = b.data<T>();
            T* dst = output.data<T>();
            for (size_t i = 0, n = b.numel(); i < n; ++i) dst[i] = src[i] * sv;
        }
    });
}

void cpu_div_copy(const Tensor& a, double s, Tensor& output, Dtype promote_dtype) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = get_division_output_dtype(input_dt);
    if (is_complex_dtype(input_dt) || is_complex_dtype(output_dt)) {
        throw std::runtime_error("Scalar division is not supported for complex dtypes");
    }
    if (s == 0.0) throw std::runtime_error("Division by zero");

    if (input_dt == output_dt) {
        dispatch_by_dtype(input_dt, [&](auto d){
            using T = decltype(d);
            if constexpr (!is_complex_t<T>::value) {
                T sv = static_cast<T>(s);
                const T* src = a.data<T>();
                T* dst = output.data<T>();
                for (size_t i = 0, n = a.numel(); i < n; ++i) dst[i] = src[i] / sv;
            }
        });
    } else {
        dispatch_by_dtype(input_dt, [&](auto d_in){
            using SrcT = decltype(d_in);
            if constexpr (!is_complex_t<SrcT>::value) {
                dispatch_by_dtype(output_dt, [&](auto d_out){
                    using DstT = decltype(d_out);
                    if constexpr (!is_complex_t<DstT>::value) {
                        DstT sv = static_cast<DstT>(s);
                        const SrcT* src = a.data<SrcT>();
                        DstT* dst = output.data<DstT>();
                        for (size_t i = 0, n = a.numel(); i < n; ++i)
                            dst[i] = static_cast<DstT>(src[i]) / sv;
                    }
                });
            }
        });
    }
}

void cpu_sub_copy_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype promote_dtype) {
    const Dtype dt = a.dtype();
    Tensor b = (dt != promote_dtype) ? a.as_type(promote_dtype) : a;
    dispatch_by_dtype(promote_dtype, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            T* src = b.data<T>();
            T* dst = output.data<T>();
            for (size_t i = 0, n = b.numel(); i < n; ++i) dst[i] = sv - src[i];
        }
    });
}

void cpu_div_copy_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype promote_dtype) {
    const Dtype input_dt = a.dtype();
    const Dtype output_dt = output.dtype();
    Tensor b = (input_dt == output_dt) ? a : a.as_type(promote_dtype);
    dispatch_by_dtype(input_dt, [&](auto d){
        using T = decltype(d);
        if constexpr (!is_complex_t<T>::value) {
            T sv = static_cast<T>(s);
            const T* src = b.data<T>();
            T* dst = output.data<T>();
            for (size_t i = 0, n = a.numel(); i < n; ++i) dst[i] = sv / src[i];
        }
    });
}

// --------- Comparison ops ---------
void cpu_eq_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        const T* src = a.data<T>();
        T sv = static_cast<T>(s);
        for (size_t i = 0, n = a.numel(); i < n; ++i)
            out_ptr[i] = static_cast<uint8_t>(src[i] == sv);
    });
}

void cpu_neq_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        const T* src = a.data<T>();
        T sv = static_cast<T>(s);
        for (size_t i = 0, n = a.numel(); i < n; ++i)
            out_ptr[i] = static_cast<uint8_t>(src[i] != sv);
    });
}

void cpu_lt_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(src[i] < sv);
        }
    });
}

void cpu_gt_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(src[i] > sv);
        }
    });
}

void cpu_leq_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than-or-equal comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(src[i] <= sv);
        }
    });
}

void cpu_geq_tensor_scalar(const Tensor& a, double s, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than-or-equal comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(src[i] >= sv);
        }
    });
}

void cpu_lt_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(sv < src[i]);
        }
    });
}

void cpu_gt_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(sv > src[i]);
        }
    });
}

void cpu_leq_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Less-than-or-equal comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(sv <= src[i]);
        }
    });
}

void cpu_geq_scalar_tensor(double s, const Tensor& a, Tensor& output, Dtype /*promote_dtype*/) {
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(output.data());
    dispatch_by_dtype(a.dtype(), [&](auto d) {
        using T = decltype(d);
        if constexpr (is_complex_t<T>::value) {
            throw std::runtime_error("Greater-than-or-equal comparison is not supported for complex types");
        } else {
            const T* src = a.data<T>();
            T sv = static_cast<T>(s);
            for (size_t i = 0, n = a.numel(); i < n; ++i)
                out_ptr[i] = static_cast<uint8_t>(sv >= src[i]);
        }
    });
}

// ============================================================================
// Explicit instantiations removed — backends are non-templated (see comment).
// ============================================================================

} // namespace OwnTensor