// ScalarOpsDispatch.cpp
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "ops/ScalarOps.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include <driver_types.h>
#include "device/DeviceCore.h" //✨✨✨
#include "dtype/DtypeTraits.h"
namespace OwnTensor
{

    // ---- backend declarations (non-templated, scalar as double) ----
    // Collapsed from N×M template instantiations to 1 per op. The scalar is
    // passed as double and cast to the tensor's dtype inside each kernel.
    void cpu_add_inplace(Tensor&, double);
    void cpu_sub_inplace(Tensor&, double);
    void cpu_mul_inplace(Tensor&, double);
    void cpu_div_inplace(Tensor&, double);

    void cpu_add_copy(const Tensor&, double, Tensor&, Dtype);
    void cpu_sub_copy(const Tensor&, double, Tensor&, Dtype);
    void cpu_mul_copy(const Tensor&, double, Tensor&, Dtype);
    void cpu_div_copy(const Tensor&, double, Tensor&, Dtype);

    void cpu_sub_copy_scalar_tensor(double, const Tensor&, Tensor&, Dtype);
    void cpu_div_copy_scalar_tensor(double, const Tensor&, Tensor&, Dtype);

    void cpu_eq_tensor_scalar(const Tensor&, double, Tensor&, Dtype);
    void cpu_neq_tensor_scalar(const Tensor&, double, Tensor&, Dtype);

    void cpu_leq_tensor_scalar(const Tensor&, double, Tensor&, Dtype);
    void cpu_geq_tensor_scalar(const Tensor&, double, Tensor&, Dtype);
    void cpu_lt_tensor_scalar(const Tensor&, double, Tensor&, Dtype);
    void cpu_gt_tensor_scalar(const Tensor&, double, Tensor&, Dtype);

    void cpu_leq_scalar_tensor(double, const Tensor&, Tensor&, Dtype);
    void cpu_geq_scalar_tensor(double, const Tensor&, Tensor&, Dtype);
    void cpu_lt_scalar_tensor(double, const Tensor&, Tensor&, Dtype);
    void cpu_gt_scalar_tensor(double, const Tensor&, Tensor&, Dtype);

    // CUDA backends
    void cuda_add_inplace(Tensor&, double, cudaStream_t);
    void cuda_sub_inplace(Tensor&, double, cudaStream_t);
    void cuda_mul_inplace(Tensor&, double, cudaStream_t);
    void cuda_div_inplace(Tensor&, double, cudaStream_t);

    void cuda_add_copy(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_sub_copy(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_mul_copy(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_div_copy(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_sub_copy_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
    void cuda_div_copy_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);

    void cuda_eq_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_neq_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);

    void cuda_leq_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_geq_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_lt_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);
    void cuda_gt_tensor_scalar(const Tensor&, double, Tensor&, Dtype, cudaStream_t);

    void cuda_leq_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
    void cuda_geq_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
    void cuda_lt_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);
    void cuda_gt_scalar_tensor(double, const Tensor&, Tensor&, Dtype, cudaStream_t);

    // ---- helpers ----
    static inline bool is_integer_dtype(Dtype dt)
    {
        return dt == Dtype::Int16 || dt == Dtype::Int32 || dt == Dtype::Int64;
    }
    template <typename S>
    static inline double to_f64(S s) { return static_cast<double>(s); }

    // ======================= Public API =======================
    template <typename S>
    Tensor &operator+=(Tensor &t, S s)
    {
        if (s == static_cast<S>(0))
            return t;
        Dtype s_dtype = type_to_dtype<S>();

        if (t.dtype() != s_dtype)
        {
            Dtype promoted_dtype = promote_scalar_ops(t.dtype(), s_dtype);
            if (promoted_dtype != t.dtype())
            {
                throw std::runtime_error("Type mismatch: Tensor dtype " + get_dtype_name(t.dtype()) +
                                         " cannot be combined with scalar of type " + get_dtype_name(s_dtype) +
                                         ". Consider upcasting the tensor to " + get_dtype_name(promoted_dtype) + " first.");
            }
        }
        dispatch_by_dtype(t.dtype(), [&](auto dummy)
                          {
        using TensorType = decltype(dummy);
        if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
        } else {
            auto promoted_s = static_cast<TensorType>(s);
            if (t.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_add_inplace(t, promoted_s,   OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_add_inplace(t, promoted_s);
            }
        } });
        return t;
    }

    template <typename S>
    Tensor &operator-=(Tensor &t, S s)
    {
        if (s == static_cast<S>(0))
            return t;
        Dtype s_dtype = type_to_dtype<S>();

        if (t.dtype() != s_dtype)
        {
            Dtype promoted_dtype = promote_scalar_ops(t.dtype(), s_dtype);
            if (promoted_dtype != t.dtype())
            {
                throw std::runtime_error("Type mismatch: Tensor dtype " + get_dtype_name(t.dtype()) +
                                         " cannot be combined with scalar of type " + get_dtype_name(s_dtype) +
                                         ". Consider upcasting the tensor to " + get_dtype_name(promoted_dtype) + " first.");
            }
        }

        dispatch_by_dtype(t.dtype(), [&](auto dummy)
                          {
        using TensorType = decltype(dummy);
        if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
        } else {
            auto promoted_s = static_cast<TensorType>(s);
            if (t.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_sub_inplace(t, promoted_s,  OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_sub_inplace(t, promoted_s);
            }
        } });
        return t;
    }

    template <typename S>
    Tensor &operator*=(Tensor &t, S s)
    {
        if (s == static_cast<S>(1))
            return t;
        Dtype s_dtype = type_to_dtype<S>();

        if (t.dtype() != s_dtype)
        {
            Dtype promoted_dtype = promote_scalar_ops(t.dtype(), s_dtype);
            if (promoted_dtype != t.dtype())
            {
                throw std::runtime_error("Type mismatch: Tensor dtype " + get_dtype_name(t.dtype()) +
                                         " cannot be combined with scalar of type " + get_dtype_name(s_dtype) +
                                         ". Consider upcasting the tensor to " + get_dtype_name(promoted_dtype) + " first.");
            }
        }

        dispatch_by_dtype(t.dtype(), [&](auto dummy)
                          {
        using TensorType = decltype(dummy);
        if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
        } else {
            auto promoted_s = static_cast<TensorType>(s);
            if (t.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_mul_inplace(t, promoted_s,  OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_mul_inplace(t, promoted_s);
            }
        } });
        return t;
    }

    template <typename S>
    Tensor &operator/=(Tensor &t, S s)
    {
        if (s == static_cast<S>(0))
            throw std::runtime_error("Division by zero");
        if (s == static_cast<S>(1))
            return t;
        Dtype s_dtype = type_to_dtype<S>();

        if (t.dtype() != s_dtype)
        {
            Dtype promoted_dtype = promote_dtypes_division(t.dtype(), s_dtype);
            if (promoted_dtype != t.dtype())
            {
                throw std::runtime_error("Type mismatch: Tensor dtype " + get_dtype_name(t.dtype()) +
                                         " cannot be combined with scalar of type " + get_dtype_name(s_dtype) +
                                         ". Consider upcasting the tensor to " + get_dtype_name(promoted_dtype) + " first.");
            }
        }

        dispatch_by_dtype(t.dtype(), [&](auto dummy)
                          {
        using TensorType = decltype(dummy);
        if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
        } else {
            auto promoted_s = static_cast<TensorType>(s);
            if (t.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_div_inplace(t, promoted_s,  OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_div_inplace(t, promoted_s);
            }
        } });
        return t;
    }

    template <typename S>
    Tensor operator+(const Tensor &a, S s)
    {
        if (s == static_cast<S>(0))
            return a.clone();
        Dtype s_dtype = type_to_dtype<S>();
        Dtype promoted_dtype;
        if (a.dtype() != s_dtype)
        {
            promoted_dtype = promote_scalar_ops(a.dtype(), s_dtype);
        }
        else
        {
            promoted_dtype = a.dtype();
        }
        Tensor output(a.shape(), promoted_dtype, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_add_copy(a,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_add_copy(a, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator-(const Tensor &a, S s)
    {
        if (s == static_cast<S>(0))
            return a.clone();
        Dtype s_dtype = type_to_dtype<S>();
        Dtype promoted_dtype;
        if (a.dtype() != s_dtype)
        {
            promoted_dtype = promote_scalar_ops(a.dtype(), s_dtype);
        }
        else
        {
            promoted_dtype = a.dtype();
        }
        Tensor output(a.shape(), promoted_dtype, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_sub_copy(a,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_sub_copy(a, s,output, promoted_dtype);
                }
            } });
        return output;
    }
    template <typename S>
    Tensor operator*(const Tensor &a, S s)
    {
        if (s == static_cast<S>(0))
            return Tensor::zeros(a.shape(), a.opts());
        if (s == static_cast<S>(1))
            return a.clone();
        Dtype s_dtype = type_to_dtype<S>();
        Dtype promoted_dtype;
        if (a.dtype() != s_dtype)
        {
            promoted_dtype = promote_scalar_ops(a.dtype(), s_dtype);
        }
        else
        {
            promoted_dtype = a.dtype();
        }
        Tensor output(a.shape(), promoted_dtype, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_mul_copy(a,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_mul_copy(a, s,output, promoted_dtype);
                }
            } });
        return output;
    }
    template <typename S>
    Tensor operator/(const Tensor &a, S s)
    {
        // if (!a.device().is_cuda() && is_integer_dtype(a.dtype()) && s == static_cast<S>(0))
        if (s == static_cast<S>(0))
        {
            throw std::runtime_error("Division by zero");
        }
        Dtype s_dtype = type_to_dtype<S>();
        Dtype promoted_dtype= promote_dtypes_division(a.dtype(), s_dtype);
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), promoted_dtype, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_div_copy(a_promoted,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_div_copy(a_promoted, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator+(S s, const Tensor &a) { return a + s; }

    template <typename S>
    Tensor operator-(S s, const Tensor &a)
    {
        if (s == static_cast<S>(0))
            return a * -1; // look for edge cases
        Dtype s_dtype = type_to_dtype<S>();
        Dtype promoted_dtype;
        if (a.dtype() != s_dtype)
        {
            promoted_dtype = promote_scalar_ops(a.dtype(), s_dtype);
        }
        else
        {
            promoted_dtype = a.dtype();
        }
        Tensor output(a.shape(), promoted_dtype, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_sub_copy_scalar_tensor(s,a,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_sub_copy_scalar_tensor(s, a,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator*(S s, const Tensor &a) { return a * s; }

    template <typename S>
    Tensor operator/(S s, const Tensor &a)
    {
        Dtype op_type = promote_dtypes_division(type_to_dtype<S>(), a.dtype());

        Tensor output(a.shape(), op_type, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t>) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_div_copy_scalar_tensor(s,a,output,op_type, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_div_copy_scalar_tensor(s, a,output, op_type);
                }
            } });
        return output;
    }

    // Comparison operators:
    template <typename S>
Tensor operator==(const Tensor &a, S s)
{
    Dtype p_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
    Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    Tensor a_promoted = (a.dtype() != p_dtype) ? a.as_type(p_dtype) : a;

    dispatch_by_dtype(p_dtype, [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes.");
        } else {
            
            if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_eq_tensor_scalar(a_promoted, s, output, p_dtype, OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_eq_tensor_scalar(a_promoted, s, output, p_dtype);
            }
        }
    });
    return output;
}

    template <typename S>
    Tensor operator!=(const Tensor &a, S s)
    {
        Dtype p_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
    Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
    Tensor a_promoted = (a.dtype() != p_dtype) ? a.as_type(p_dtype) : a;

    dispatch_by_dtype(p_dtype, [&](auto dummy) {
        using T = decltype(dummy);
        if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t>) {
            throw std::runtime_error("Scalar ops not supported for complex dtypes.");
        } else {
            
            if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                cuda_neq_tensor_scalar(a_promoted, s, output, p_dtype, OwnTensor::cuda::getCurrentStream());
#endif
            } else {
                cpu_neq_tensor_scalar(a_promoted, s, output, p_dtype);
            }
        }
    });
    return output;
    }

    template <typename S>
    Tensor operator<=(const Tensor &a, S s)
    {
         Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_leq_tensor_scalar(a_promoted,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_leq_tensor_scalar(a_promoted, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator>=(const Tensor &a, S s)
    {
       Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_geq_tensor_scalar(a_promoted,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_geq_tensor_scalar(a_promoted, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator>(const Tensor &a, S s)
    {
        Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_gt_tensor_scalar(a_promoted,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_gt_tensor_scalar(a_promoted, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator<(const Tensor &a, S s)
    {
        Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_lt_tensor_scalar(a_promoted,s,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_lt_tensor_scalar(a_promoted, s,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator==(S s, const Tensor &a)
    {
        return (a == s);
    }
    template <typename S>
    Tensor operator!=(S s, const Tensor &a)
    {
        return (a != s);
    }

    template <typename S>
    Tensor operator<=(S s, const Tensor &a)
    {
        Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_leq_scalar_tensor(s,a_promoted,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_leq_scalar_tensor(s, a_promoted,output, promoted_dtype);
                }
            } });
        return output;
    }
    // return a.device().is_cuda() ? cuda_s_leq_copy(to_f64(s), a, OwnTensor::cuda::getCurrentStream()) : cpu_s_leq_copy(to_f64(s), a); // ✨✨✨
    
    template <typename S>
    Tensor operator>=(S s, const Tensor &a)
    {
         Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_geq_scalar_tensor(s,a_promoted,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_geq_scalar_tensor(s, a_promoted,output, promoted_dtype);
                }
            } });
        return output;
    }
    template <typename S>
    Tensor operator>(S s, const Tensor &a)
    {
         Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_gt_scalar_tensor(s,a_promoted,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_gt_scalar_tensor(s, a_promoted,output, promoted_dtype);
                }
            } });
        return output;
    }

    template <typename S>
    Tensor operator<(S s, const Tensor &a)
    {
     Dtype promoted_dtype = promote_scalar_ops(a.dtype(), type_to_dtype<S>());
        Tensor a_promoted = (a.dtype() != promoted_dtype) ? a.as_type(promoted_dtype) : a;

        Tensor output(a.shape(), Dtype::Bool, a.device(), a.requires_grad());
        dispatch_by_dtype(a.dtype(), [&](auto dummy)
                          {
            using TensorType = decltype(dummy);
            if constexpr (std::is_same_v<TensorType, complex32_t> || std::is_same_v<TensorType, complex64_t> || std::is_same_v<TensorType, complex128_t> || is_complex(type_to_dtype<S>())) {
                throw std::runtime_error("Scalar ops not supported for complex dtypes. Both operands must be complex.");
            } else {
                if (a.device().is_cuda()) {
#ifdef WITH_CUDA
                    cuda_lt_scalar_tensor(s,a_promoted,output,promoted_dtype, OwnTensor::cuda::getCurrentStream());
#endif
                } else {
                    cpu_lt_scalar_tensor(s, a_promoted,output, promoted_dtype);
                }
            } });
        return output;   
    }

    template <typename S>
    Tensor logical_AND([[maybe_unused]] const Tensor &a, [[maybe_unused]] S b)
    {
        throw std::runtime_error("logical_AND with scalar is not supported. Try Tensor logical_AND( Tensor, Tensor)");
    }

    template <typename S>
    Tensor logical_OR([[maybe_unused]] const Tensor &a, [[maybe_unused]] S b)
    {
        throw std::runtime_error("logical_OR with scalar is not supported. Try Tensor logical_OR( Tensor, Tensor)");
    }

    template <typename S>
    Tensor logical_XOR([[maybe_unused]] const Tensor &a, [[maybe_unused]] S b)
    {
        throw std::runtime_error("logical_XOR with scalar is not supported. Try Tensor logical_XOR( Tensor, Tensor)");
    }

    template <typename S>
    Tensor logical_NOT([[maybe_unused]] S a)
    {
        throw std::runtime_error("logical_NOT with scalar is not supported. Try Tensor logical_NOT( Tensor )");
    }

    template <typename S>
    Tensor logical_AND([[maybe_unused]] S a, [[maybe_unused]] const Tensor &b)
    {
        throw std::runtime_error("logical_AND with scalar is not supported. Try Tensor logical_AND( Tensor, Tensor)");
    }

    template <typename S>
    Tensor logical_OR([[maybe_unused]] S a, [[maybe_unused]] const Tensor &b)
    {
        throw std::runtime_error("logical_OR with scalar is not supported. Try Tensor logical_OR( Tensor, Tensor)");
    }

    template <typename S>
    Tensor logical_XOR([[maybe_unused]] S a, [[maybe_unused]] const Tensor &b)
    {
        throw std::runtime_error("logical_XOR with scalar is not supported. Try Tensor logical_XOR( Tensor, Tensor)");
    }

    //======================= Explicit instantiations =======================
    // using OwnTensor::float16_t;
    // using OwnTensor::bfloat16_t;

    template Tensor &operator+= <int8_t>(Tensor &, int8_t);
    template Tensor &operator+= <int16_t>(Tensor &, int16_t);
    template Tensor &operator+= <int32_t>(Tensor &, int32_t);
    template Tensor &operator+= <int64_t>(Tensor &, int64_t);
    template Tensor &operator+= <float>(Tensor &, float);
    template Tensor &operator+= <double>(Tensor &, double);
    template Tensor &operator+= <float16_t>(Tensor &, float16_t);
    template Tensor &operator+= <bfloat16_t>(Tensor &, bfloat16_t);
    template Tensor &operator+= <bool>(Tensor &, bool);
    template Tensor &operator+= <uint8_t>(Tensor &, uint8_t);
    template Tensor &operator+= <uint16_t>(Tensor &, uint16_t);
    template Tensor &operator+= <uint32_t>(Tensor &, uint32_t);
    template Tensor &operator+= <uint64_t>(Tensor &, uint64_t);

    template Tensor &operator-= <int8_t>(Tensor &, int8_t);
    template Tensor &operator-= <int16_t>(Tensor &, int16_t);
    template Tensor &operator-= <int32_t>(Tensor &, int32_t);
    template Tensor &operator-= <int64_t>(Tensor &, int64_t);
    template Tensor &operator-= <float>(Tensor &, float);
    template Tensor &operator-= <double>(Tensor &, double);
    template Tensor &operator-= <float16_t>(Tensor &, float16_t);
    template Tensor &operator-= <bfloat16_t>(Tensor &, bfloat16_t);
    template Tensor &operator-= <bool>(Tensor &, bool);
    template Tensor &operator-= <uint8_t>(Tensor &, uint8_t);
    template Tensor &operator-= <uint16_t>(Tensor &, uint16_t);
    template Tensor &operator-= <uint32_t>(Tensor &, uint32_t);
    template Tensor &operator-= <uint64_t>(Tensor &, uint64_t);

    template Tensor &operator*= <int8_t>(Tensor &, int8_t);
    template Tensor &operator*= <int16_t>(Tensor &, int16_t);
    template Tensor &operator*= <int32_t>(Tensor &, int32_t);
    template Tensor &operator*= <int64_t>(Tensor &, int64_t);
    template Tensor &operator*= <float>(Tensor &, float);
    template Tensor &operator*= <double>(Tensor &, double);
    template Tensor &operator*= <float16_t>(Tensor &, float16_t);
    template Tensor &operator*= <bfloat16_t>(Tensor &, bfloat16_t);
    template Tensor &operator*= <bool>(Tensor &, bool);
    template Tensor &operator*= <uint8_t>(Tensor &, uint8_t);
    template Tensor &operator*= <uint16_t>(Tensor &, uint16_t);
    template Tensor &operator*= <uint32_t>(Tensor &, uint32_t);
    template Tensor &operator*= <uint64_t>(Tensor &, uint64_t);

    template Tensor &operator/= <int8_t>(Tensor &, int8_t);
    template Tensor &operator/= <int16_t>(Tensor &, int16_t);
    template Tensor &operator/= <int32_t>(Tensor &, int32_t);
    template Tensor &operator/= <int64_t>(Tensor &, int64_t);
    template Tensor &operator/= <float>(Tensor &, float);
    template Tensor &operator/= <double>(Tensor &, double);
    template Tensor &operator/= <float16_t>(Tensor &, float16_t);
    template Tensor &operator/= <bfloat16_t>(Tensor &, bfloat16_t);
    template Tensor &operator/= <bool>(Tensor &, bool);
    template Tensor &operator/= <uint8_t>(Tensor &, uint8_t);
    template Tensor &operator/= <uint16_t>(Tensor &, uint16_t);
    template Tensor &operator/= <uint32_t>(Tensor &, uint32_t);
    template Tensor &operator/= <uint64_t>(Tensor &, uint64_t);

    // template Tensor operator+=<int16_t>(Tensor&&, int16_t);
    // template Tensor operator+=<int32_t>(Tensor&&, int32_t);
    // template Tensor operator+=<int64_t>(Tensor&&, int64_t);
    // template Tensor operator+=<float>(Tensor&&, float);
    // template Tensor operator+=<double>(Tensor&&, double);
    // template Tensor operator+=<float16_t>(Tensor&&, float16_t);
    // template Tensor operator+=<bfloat16_t>(Tensor&&, bfloat16_t);
    // template Tensor operator+=<bool>(Tensor&&, bool);
    // template Tensor operator+=<uint8_t>(Tensor&&, uint8_t);
    // template Tensor operator+=<uint16_t>(Tensor&&, uint16_t);
    // template Tensor operator+=<uint32_t>(Tensor&&, uint32_t);
    // template Tensor operator+=<uint64_t>(Tensor&&, uint64_t);

    // template Tensor operator-=<int16_t>(Tensor&&, int16_t);
    // template Tensor operator-=<int32_t>(Tensor&&, int32_t);
    // template Tensor operator-=<int64_t>(Tensor&&, int64_t);
    // template Tensor operator-=<float>(Tensor&&, float);
    // template Tensor operator-=<double>(Tensor&&, double);
    // template Tensor operator-=<float16_t>(Tensor&&, float16_t);
    // template Tensor operator-=<bfloat16_t>(Tensor&&, bfloat16_t);
    // template Tensor operator-=<bool>(Tensor&&, bool);
    // template Tensor operator-=<uint8_t>(Tensor&&, uint8_t);
    // template Tensor operator-=<uint16_t>(Tensor&&, uint16_t);
    // template Tensor operator-=<uint32_t>(Tensor&&, uint32_t);
    // template Tensor operator-=<uint64_t>(Tensor&&, uint64_t);

    // template Tensor operator*=<int16_t>(Tensor&&, int16_t);
    // template Tensor operator*=<int32_t>(Tensor&&, int32_t);
    // template Tensor operator*=<int64_t>(Tensor&&, int64_t);
    // template Tensor operator*=<float>(Tensor&&, float);
    // template Tensor operator*=<double>(Tensor&&, double);
    // template Tensor operator*=<float16_t>(Tensor&&, float16_t);
    // template Tensor operator*=<bfloat16_t>(Tensor&&, bfloat16_t);
    // template Tensor operator*=<bool>(Tensor&&, bool);
    // template Tensor operator*=<uint8_t>(Tensor&&, uint8_t);
    // template Tensor operator*=<uint16_t>(Tensor&&, uint16_t);
    // template Tensor operator*=<uint32_t>(Tensor&&, uint32_t);
    // template Tensor operator*=<uint64_t>(Tensor&&, uint64_t);

    // template Tensor operator/=<int16_t>(Tensor&&, int16_t);
    // template Tensor operator/=<int32_t>(Tensor&&, int32_t);
    // template Tensor operator/=<int64_t>(Tensor&&, int64_t);
    // template Tensor operator/=<float>(Tensor&&, float);
    // template Tensor operator/=<double>(Tensor&&, double);
    // template Tensor operator/=<float16_t>(Tensor&&, float16_t);
    // template Tensor operator/=<bfloat16_t>(Tensor&&, bfloat16_t);
    // template Tensor operator/=<bool>(Tensor&&, bool);
    // template Tensor operator/=<uint8_t>(Tensor&&, uint8_t);
    // template Tensor operator/=<uint16_t>(Tensor&&, uint16_t);
    // template Tensor operator/=<uint32_t>(Tensor&&, uint32_t);
    // template Tensor operator/=<uint64_t>(Tensor&&, uint64_t);

    template Tensor operator+ <int8_t>(const Tensor &, int8_t);
    template Tensor operator+ <int16_t>(const Tensor &, int16_t);
    template Tensor operator+ <int32_t>(const Tensor &, int32_t);
    template Tensor operator+ <int64_t>(const Tensor &, int64_t);
    template Tensor operator+ <float>(const Tensor &, float);
    template Tensor operator+ <double>(const Tensor &, double);
    template Tensor operator+ <float16_t>(const Tensor &, float16_t);
    template Tensor operator+ <bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor operator+ <bool>(const Tensor &, bool);
    template Tensor operator+ <uint8_t>(const Tensor &, uint8_t);
    template Tensor operator+ <uint16_t>(const Tensor &, uint16_t);
    template Tensor operator+ <uint32_t>(const Tensor &, uint32_t);
    template Tensor operator+ <uint64_t>(const Tensor &, uint64_t);

    template Tensor operator-<int8_t>(const Tensor&, int8_t);
    template Tensor operator-<int16_t>(const Tensor&, int16_t);
    template Tensor operator-<int32_t>(const Tensor&, int32_t);
    template Tensor operator-<int64_t>(const Tensor&, int64_t);
    template Tensor operator-<float>(const Tensor&, float);
    template Tensor operator-<double>(const Tensor&, double);
    template Tensor operator-<float16_t>(const Tensor&, float16_t);
    template Tensor operator-<bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator-<bool>(const Tensor&, bool);
    template Tensor operator-<uint8_t>(const Tensor&, uint8_t);
    template Tensor operator-<uint16_t>(const Tensor&, uint16_t);
    template Tensor operator-<uint32_t>(const Tensor&, uint32_t);
    template Tensor operator-<uint64_t>(const Tensor&, uint64_t);

    template Tensor operator* <int8_t>(const Tensor &, int8_t);
    template Tensor operator* <int16_t>(const Tensor &, int16_t);
    template Tensor operator* <int32_t>(const Tensor &, int32_t);
    template Tensor operator* <int64_t>(const Tensor &, int64_t);
    template Tensor operator* <float>(const Tensor &, float);
    template Tensor operator* <double>(const Tensor &, double);
    template Tensor operator* <float16_t>(const Tensor &, float16_t);
    template Tensor operator* <bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor operator* <bool>(const Tensor &, bool);
    template Tensor operator* <uint8_t>(const Tensor &, uint8_t);
    template Tensor operator* <uint16_t>(const Tensor &, uint16_t);
    template Tensor operator* <uint32_t>(const Tensor &, uint32_t);
    template Tensor operator* <uint64_t>(const Tensor &, uint64_t);

    template Tensor operator/ <int8_t>(const Tensor &, int8_t);
    template Tensor operator/ <int16_t>(const Tensor &, int16_t);
    template Tensor operator/ <int32_t>(const Tensor &, int32_t);
    template Tensor operator/ <int64_t>(const Tensor &, int64_t);
    template Tensor operator/ <float>(const Tensor &, float);
    template Tensor operator/ <double>(const Tensor &, double);
    template Tensor operator/ <float16_t>(const Tensor &, float16_t);
    template Tensor operator/ <bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor operator/ <bool>(const Tensor &, bool);
    template Tensor operator/ <uint8_t>(const Tensor &, uint8_t);
    template Tensor operator/ <uint16_t>(const Tensor &, uint16_t);
    template Tensor operator/ <uint32_t>(const Tensor &, uint32_t);
    template Tensor operator/ <uint64_t>(const Tensor &, uint64_t);

    template Tensor operator+ <int8_t>(int8_t, const Tensor &);
    template Tensor operator+ <int16_t>(int16_t, const Tensor &);
    template Tensor operator+ <int32_t>(int32_t, const Tensor &);
    template Tensor operator+ <int64_t>(int64_t, const Tensor &);
    template Tensor operator+ <float>(float, const Tensor &);
    template Tensor operator+ <double>(double, const Tensor &);
    template Tensor operator+ <float16_t>(float16_t, const Tensor &);
    template Tensor operator+ <bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor operator+ <bool>(bool, const Tensor &);
    template Tensor operator+ <uint8_t>(uint8_t, const Tensor &);
    template Tensor operator+ <uint16_t>(uint16_t, const Tensor &);
    template Tensor operator+ <uint32_t>(uint32_t, const Tensor &);
    template Tensor operator+ <uint64_t>(uint64_t, const Tensor &);

    template Tensor operator- <int8_t>(int8_t, const Tensor &);
    template Tensor operator- <int16_t>(int16_t, const Tensor &);
    template Tensor operator- <int32_t>(int32_t, const Tensor &);
    template Tensor operator- <int64_t>(int64_t, const Tensor &);
    template Tensor operator- <float>(float, const Tensor &);
    template Tensor operator- <double>(double, const Tensor &);
    template Tensor operator- <float16_t>(float16_t, const Tensor &);
    template Tensor operator- <bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor operator- <bool>(bool, const Tensor &);
    template Tensor operator- <uint8_t>(uint8_t, const Tensor &);
    template Tensor operator- <uint16_t>(uint16_t, const Tensor &);
    template Tensor operator- <uint32_t>(uint32_t, const Tensor &);
    template Tensor operator- <uint64_t>(uint64_t, const Tensor &);

    template Tensor operator* <int8_t>(int8_t, const Tensor &);
    template Tensor operator* <int16_t>(int16_t, const Tensor &);
    template Tensor operator* <int32_t>(int32_t, const Tensor &);
    template Tensor operator* <int64_t>(int64_t, const Tensor &);
    template Tensor operator* <float>(float, const Tensor &);
    template Tensor operator* <double>(double, const Tensor &);
    template Tensor operator* <float16_t>(float16_t, const Tensor &);
    template Tensor operator* <bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor operator* <bool>(bool, const Tensor &);
    template Tensor operator* <uint8_t>(uint8_t, const Tensor &);
    template Tensor operator* <uint16_t>(uint16_t, const Tensor &);
    template Tensor operator* <uint32_t>(uint32_t, const Tensor &);
    template Tensor operator* <uint64_t>(uint64_t, const Tensor &);

    template Tensor operator/ <int8_t>(int8_t, const Tensor &);
    template Tensor operator/<int16_t>(int16_t, const Tensor&);
    template Tensor operator/<int32_t>(int32_t, const Tensor&);
    template Tensor operator/<int64_t>(int64_t, const Tensor&);
    template Tensor operator/<float>(float, const Tensor&);
    template Tensor operator/<double>(double, const Tensor&);
    template Tensor operator/<float16_t>(float16_t, const Tensor&);
    template Tensor operator/<bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator/<bool>(bool, const Tensor&);
    template Tensor operator/<uint8_t>(uint8_t, const Tensor&);
    template Tensor operator/<uint16_t>(uint16_t, const Tensor&);
    template Tensor operator/<uint32_t>(uint32_t, const Tensor&);
    template Tensor operator/<uint64_t>(uint64_t, const Tensor&);

    template Tensor operator==<int8_t>(int8_t, const Tensor&);
    template Tensor operator==<int16_t>(int16_t, const Tensor&);
    template Tensor operator==<int32_t>(int32_t, const Tensor&);
    template Tensor operator==<int64_t>(int64_t, const Tensor&);
    template Tensor operator==<float>(float, const Tensor&);
    template Tensor operator==<double>(double, const Tensor&);
    template Tensor operator==<bool>(bool, const Tensor&);
    template Tensor operator==<float16_t>(float16_t, const Tensor&);
    template Tensor operator==<bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator==<uint8_t>(uint8_t, const Tensor&);
    template Tensor operator==<uint16_t>(uint16_t, const Tensor&);
    template Tensor operator==<uint32_t>(uint32_t, const Tensor&);
    template Tensor operator==<uint64_t>(uint64_t, const Tensor&);

    template Tensor operator!=<int8_t>(int8_t, const Tensor&);
    template Tensor operator!=<int16_t>(int16_t, const Tensor&);
    template Tensor operator!=<int32_t>(int32_t, const Tensor&);
    template Tensor operator!=<int64_t>(int64_t, const Tensor&);
    template Tensor operator!=<float>(float, const Tensor&);
    template Tensor operator!=<double>(double, const Tensor&);
    template Tensor operator!=<bool>(bool, const Tensor&);
    template Tensor operator!=<float16_t>(float16_t, const Tensor&);
    template Tensor operator!=<bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator!=<uint8_t>(uint8_t, const Tensor&);
    template Tensor operator!=<uint16_t>(uint16_t, const Tensor&);
    template Tensor operator!=<uint32_t>(uint32_t, const Tensor&);
    template Tensor operator!=<uint64_t>(uint64_t, const Tensor&);

    template Tensor operator>=<int8_t>(int8_t, const Tensor&);
    template Tensor operator>=<int16_t>(int16_t, const Tensor&);
    template Tensor operator>=<int32_t>(int32_t, const Tensor&);
    template Tensor operator>=<int64_t>(int64_t, const Tensor&);
    template Tensor operator>=<float>(float, const Tensor&);
    template Tensor operator>=<double>(double, const Tensor&);
    template Tensor operator>=<bool>(bool, const Tensor&);
    template Tensor operator>=<float16_t>(float16_t, const Tensor&);
    template Tensor operator>=<bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator>=<uint8_t>(uint8_t, const Tensor&);
    template Tensor operator>=<uint16_t>(uint16_t, const Tensor&);
    template Tensor operator>=<uint32_t>(uint32_t, const Tensor&);
    template Tensor operator>=<uint64_t>(uint64_t, const Tensor&);

    template Tensor operator<=<int8_t>(int8_t, const Tensor&);
    template Tensor operator<=<int16_t>(int16_t, const Tensor&);
    template Tensor operator<=<int32_t>(int32_t, const Tensor&);
    template Tensor operator<=<int64_t>(int64_t, const Tensor&);
    template Tensor operator<=<float>(float, const Tensor&);
    template Tensor operator<=<double>(double, const Tensor&);
    template Tensor operator<=<bool>(bool, const Tensor&);
    template Tensor operator<=<float16_t>(float16_t, const Tensor&);
    template Tensor operator<=<bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator<=<uint8_t>(uint8_t, const Tensor&);
    template Tensor operator<=<uint16_t>(uint16_t, const Tensor&);
    template Tensor operator<=<uint32_t>(uint32_t, const Tensor&);
    template Tensor operator<=<uint64_t>(uint64_t, const Tensor&);

    template Tensor operator><int8_t>(int8_t, const Tensor&);
    template Tensor operator><int16_t>(int16_t, const Tensor&);
    template Tensor operator><int32_t>(int32_t, const Tensor&);
    template Tensor operator><int64_t>(int64_t, const Tensor&);
    template Tensor operator><float>(float, const Tensor&);
    template Tensor operator><double>(double, const Tensor&);
    template Tensor operator><bool>(bool, const Tensor&);
    template Tensor operator><float16_t>(float16_t, const Tensor&);
    template Tensor operator><bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator><uint8_t>(uint8_t, const Tensor&);
    template Tensor operator><uint16_t>(uint16_t, const Tensor&);
    template Tensor operator><uint32_t>(uint32_t, const Tensor&);
    template Tensor operator><uint64_t>(uint64_t, const Tensor&);

    template Tensor operator< <int8_t>(int8_t, const Tensor&);
    template Tensor operator< <int16_t>(int16_t, const Tensor&);
    template Tensor operator< <int32_t>(int32_t, const Tensor&);
    template Tensor operator< <int64_t>(int64_t, const Tensor&);
    template Tensor operator< <float>(float, const Tensor&);
    template Tensor operator< <double>(double, const Tensor&);
    template Tensor operator< <bool>(bool, const Tensor&);
    template Tensor operator< <float16_t>(float16_t, const Tensor&);
    template Tensor operator< <bfloat16_t>(bfloat16_t, const Tensor&);
    template Tensor operator< <uint8_t>(uint8_t, const Tensor&);
    template Tensor operator< <uint16_t>(uint16_t, const Tensor&);
    template Tensor operator< <uint32_t>(uint32_t, const Tensor&);
    template Tensor operator< <uint64_t>(uint64_t, const Tensor&);

    template Tensor operator==<int8_t>(const Tensor&, int8_t);
    template Tensor operator==<int16_t>(const Tensor&, int16_t);
    template Tensor operator==<int32_t>(const Tensor&, int32_t);
    template Tensor operator==<int64_t>(const Tensor&, int64_t);
    template Tensor operator==<float>(const Tensor&, float);
    template Tensor operator==<double>(const Tensor&, double);
    template Tensor operator==<bool>(const Tensor&, bool);
    template Tensor operator==<float16_t>(const Tensor&, float16_t);
    template Tensor operator==<bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator==<uint8_t>(const Tensor&, uint8_t);
    template Tensor operator==<uint16_t>(const Tensor&, uint16_t);
    template Tensor operator==<uint32_t>(const Tensor&, uint32_t);
    template Tensor operator==<uint64_t>(const Tensor&, uint64_t);

    template Tensor operator!=<int8_t>(const Tensor&, int8_t);
    template Tensor operator!=<int16_t>(const Tensor&, int16_t);
    template Tensor operator!=<int32_t>(const Tensor&, int32_t);
    template Tensor operator!=<int64_t>(const Tensor&, int64_t);
    template Tensor operator!=<float>(const Tensor&, float);
    template Tensor operator!=<double>(const Tensor&, double);
    template Tensor operator!=<bool>(const Tensor&, bool);
    template Tensor operator!=<float16_t>(const Tensor&, float16_t);
    template Tensor operator!=<bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator!=<uint8_t>(const Tensor&, uint8_t);
    template Tensor operator!=<uint16_t>(const Tensor&, uint16_t);
    template Tensor operator!=<uint32_t>(const Tensor&, uint32_t);
    template Tensor operator!=<uint64_t>(const Tensor&, uint64_t);

    template Tensor operator>=<int8_t>(const Tensor&, int8_t);
    template Tensor operator>=<int16_t>(const Tensor&, int16_t);
    template Tensor operator>=<int32_t>(const Tensor&, int32_t);
    template Tensor operator>=<int64_t>(const Tensor&, int64_t);
    template Tensor operator>=<float>(const Tensor&, float);
    template Tensor operator>=<double>(const Tensor&, double);
    template Tensor operator>=<bool>(const Tensor&, bool);
    template Tensor operator>=<float16_t>(const Tensor&, float16_t);
    template Tensor operator>=<bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator>=<uint8_t>(const Tensor&, uint8_t);
    template Tensor operator>=<uint16_t>(const Tensor&, uint16_t);
    template Tensor operator>=<uint32_t>(const Tensor&, uint32_t);
    template Tensor operator>=<uint64_t>(const Tensor&, uint64_t);

    template Tensor operator<=<int8_t>(const Tensor&, int8_t);
    template Tensor operator<=<int16_t>(const Tensor&, int16_t);
    template Tensor operator<=<int32_t>(const Tensor&, int32_t);
    template Tensor operator<=<int64_t>(const Tensor&, int64_t);
    template Tensor operator<=<float>(const Tensor&, float);
    template Tensor operator<=<double>(const Tensor&, double);
    template Tensor operator<=<bool>(const Tensor&, bool);
    template Tensor operator<=<float16_t>(const Tensor&, float16_t);
    template Tensor operator<=<bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator<=<uint8_t>(const Tensor&, uint8_t);
    template Tensor operator<=<uint16_t>(const Tensor&, uint16_t);
    template Tensor operator<=<uint32_t>(const Tensor&, uint32_t);
    template Tensor operator<=<uint64_t>(const Tensor&, uint64_t);

    template Tensor operator><int8_t>(const Tensor &, int8_t);
    template Tensor operator><int16_t>(const Tensor &, int16_t);
    template Tensor operator><int32_t>(const Tensor &, int32_t);
    template Tensor operator><int64_t>(const Tensor &, int64_t);
    template Tensor operator><float>(const Tensor &, float);
    template Tensor operator><double>(const Tensor &, double);
    template Tensor operator><bool>(const Tensor &, bool);
    template Tensor operator><float16_t>(const Tensor &, float16_t);
    template Tensor operator><bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor operator><uint8_t>(const Tensor &, uint8_t);
    template Tensor operator><uint16_t>(const Tensor &, uint16_t);
    template Tensor operator><uint32_t>(const Tensor &, uint32_t);
    template Tensor operator><uint64_t>(const Tensor &, uint64_t);

    template Tensor operator< <int8_t>(const Tensor&, int8_t);
    template Tensor operator< <int16_t>(const Tensor&, int16_t);
    template Tensor operator< <int32_t>(const Tensor&, int32_t);
    template Tensor operator< <int64_t>(const Tensor&, int64_t);
    template Tensor operator< <float>(const Tensor&, float);
    template Tensor operator< <double>(const Tensor&, double);
    template Tensor operator< <bool>(const Tensor&, bool);
    template Tensor operator< <float16_t>(const Tensor&, float16_t);
    template Tensor operator< <bfloat16_t>(const Tensor&, bfloat16_t);
    template Tensor operator< <uint8_t>(const Tensor&, uint8_t);
    template Tensor operator< <uint16_t>(const Tensor&, uint16_t);
    template Tensor operator< <uint32_t>(const Tensor&, uint32_t);
    template Tensor operator< <uint64_t>(const Tensor&, uint64_t);

    template Tensor logical_AND<int8_t>(const Tensor &, int8_t);
    template Tensor logical_AND<int16_t>(const Tensor &, int16_t);
    template Tensor logical_AND<int32_t>(const Tensor &, int32_t);
    template Tensor logical_AND<int64_t>(const Tensor &, int64_t);
    template Tensor logical_AND<float>(const Tensor &, float);
    template Tensor logical_AND<double>(const Tensor &, double);
    template Tensor logical_AND<bool>(const Tensor &, bool);
    template Tensor logical_AND<float16_t>(const Tensor &, float16_t);
    template Tensor logical_AND<bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor logical_AND<uint8_t>(const Tensor &, uint8_t);
    template Tensor logical_AND<uint16_t>(const Tensor &, uint16_t);
    template Tensor logical_AND<uint32_t>(const Tensor &, uint32_t);
    template Tensor logical_AND<uint64_t>(const Tensor &, uint64_t);

    template Tensor logical_OR<int8_t>(const Tensor &, int8_t);
    template Tensor logical_OR<int16_t>(const Tensor &, int16_t);
    template Tensor logical_OR<int32_t>(const Tensor &, int32_t);
    template Tensor logical_OR<int64_t>(const Tensor &, int64_t);
    template Tensor logical_OR<float>(const Tensor &, float);
    template Tensor logical_OR<double>(const Tensor &, double);
    template Tensor logical_OR<bool>(const Tensor &, bool);
    template Tensor logical_OR<float16_t>(const Tensor &, float16_t);
    template Tensor logical_OR<bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor logical_OR<uint8_t>(const Tensor &, uint8_t);
    template Tensor logical_OR<uint16_t>(const Tensor &, uint16_t);
    template Tensor logical_OR<uint32_t>(const Tensor &, uint32_t);
    template Tensor logical_OR<uint64_t>(const Tensor &, uint64_t);

    template Tensor logical_XOR<int8_t>(const Tensor &, int8_t);
    template Tensor logical_XOR<int16_t>(const Tensor &, int16_t);
    template Tensor logical_XOR<int32_t>(const Tensor &, int32_t);
    template Tensor logical_XOR<int64_t>(const Tensor &, int64_t);
    template Tensor logical_XOR<float>(const Tensor &, float);
    template Tensor logical_XOR<double>(const Tensor &, double);
    template Tensor logical_XOR<bool>(const Tensor &, bool);
    template Tensor logical_XOR<float16_t>(const Tensor &, float16_t);
    template Tensor logical_XOR<bfloat16_t>(const Tensor &, bfloat16_t);
    template Tensor logical_XOR<uint8_t>(const Tensor &, uint8_t);
    template Tensor logical_XOR<uint16_t>(const Tensor &, uint16_t);
    template Tensor logical_XOR<uint32_t>(const Tensor &, uint32_t);
    template Tensor logical_XOR<uint64_t>(const Tensor &, uint64_t);

    template Tensor logical_NOT<int8_t>(int8_t);
    template Tensor logical_NOT<int16_t>(int16_t);
    template Tensor logical_NOT<int32_t>(int32_t);
    template Tensor logical_NOT<int64_t>(int64_t);
    template Tensor logical_NOT<float>(float);
    template Tensor logical_NOT<double>(double);
    template Tensor logical_NOT<bool>(bool);
    template Tensor logical_NOT<float16_t>(float16_t);
    template Tensor logical_NOT<bfloat16_t>(bfloat16_t);
    template Tensor logical_NOT<uint8_t>(uint8_t);
    template Tensor logical_NOT<uint16_t>(uint16_t);
    template Tensor logical_NOT<uint32_t>(uint32_t);
    template Tensor logical_NOT<uint64_t>(uint64_t);

    template Tensor logical_AND<int8_t>(int8_t, const Tensor &);
    template Tensor logical_AND<int16_t>(int16_t, const Tensor &);
    template Tensor logical_AND<int32_t>(int32_t, const Tensor &);
    template Tensor logical_AND<int64_t>(int64_t, const Tensor &);
    template Tensor logical_AND<float>(float, const Tensor &);
    template Tensor logical_AND<double>(double, const Tensor &);
    template Tensor logical_AND<bool>(bool, const Tensor &);
    template Tensor logical_AND<float16_t>(float16_t, const Tensor &);
    template Tensor logical_AND<bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor logical_AND<uint8_t>(uint8_t, const Tensor &);
    template Tensor logical_AND<uint16_t>(uint16_t, const Tensor &);
    template Tensor logical_AND<uint32_t>(uint32_t, const Tensor &);
    template Tensor logical_AND<uint64_t>(uint64_t, const Tensor &);

    template Tensor logical_OR<int8_t>(int8_t, const Tensor &);
    template Tensor logical_OR<int16_t>(int16_t, const Tensor &);
    template Tensor logical_OR<int32_t>(int32_t, const Tensor &);
    template Tensor logical_OR<int64_t>(int64_t, const Tensor &);
    template Tensor logical_OR<float>(float, const Tensor &);
    template Tensor logical_OR<double>(double, const Tensor &);
    template Tensor logical_OR<bool>(bool, const Tensor &);
    template Tensor logical_OR<float16_t>(float16_t, const Tensor &);
    template Tensor logical_OR<bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor logical_OR<uint8_t>(uint8_t, const Tensor &);
    template Tensor logical_OR<uint16_t>(uint16_t, const Tensor &);
    template Tensor logical_OR<uint32_t>(uint32_t, const Tensor &);
    template Tensor logical_OR<uint64_t>(uint64_t, const Tensor &);

    template Tensor logical_XOR<int8_t>(int8_t, const Tensor &);
    template Tensor logical_XOR<int16_t>(int16_t, const Tensor &);
    template Tensor logical_XOR<int32_t>(int32_t, const Tensor &);
    template Tensor logical_XOR<int64_t>(int64_t, const Tensor &);
    template Tensor logical_XOR<float>(float, const Tensor &);
    template Tensor logical_XOR<double>(double, const Tensor &);
    template Tensor logical_XOR<bool>(bool, const Tensor &);
    template Tensor logical_XOR<float16_t>(float16_t, const Tensor &);
    template Tensor logical_XOR<bfloat16_t>(bfloat16_t, const Tensor &);
    template Tensor logical_XOR<uint8_t>(uint8_t, const Tensor &);
    template Tensor logical_XOR<uint16_t>(uint16_t, const Tensor &);
    template Tensor logical_XOR<uint32_t>(uint32_t, const Tensor &);
    template Tensor logical_XOR<uint64_t>(uint64_t, const Tensor &);
} // namespace OwnTensor