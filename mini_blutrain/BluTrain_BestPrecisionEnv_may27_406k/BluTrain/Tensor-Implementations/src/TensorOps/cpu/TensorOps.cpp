#include "core/Tensor.h"
#include "ops/helpers/TensorOpUtils.h"
#include "ops/helpers/BroadcastUtils.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "device/DeviceCore.h"
#include "dtype/fp4.h"
#include "dtype/DtypeTraits.h"  //  ADD THIS for promote_dtypes_bool
#include <driver_types.h>
#include <stdexcept>
#include <functional>
#include <random>

namespace OwnTensor {
// ============================================================================
// HELPER: Convert tensor to promoted dtype if needed
// ============================================================================
    static Tensor promote_if_needed(const Tensor& input, Dtype target_dtype) {
    if (input.dtype() == target_dtype) {
        return input;  // No conversion needed
    }
    
    // Convert to target dtype
    return input.as_type(target_dtype);
    }


// ============================================================================
// ADDITION OPERATOR
// ============================================================================

    Tensor operator+(const Tensor& lhs, const Tensor& rhs) 
    {
    //  1. Determine promoted dtype
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    
    //  2. Convert operands if needed
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    //  3. Compute output shape (broadcasting)
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    //  4. Create output tensor with promoted dtype
    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    //  5. Perform operation
    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_add_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a + b;
        });
    }
    return output;
    }


// ============================================================================
// SUBTRACTION OPERATOR
// ============================================================================
    Tensor operator-(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_sub_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a - b;
        });
    }
    return output;
    }


// ============================================================================
// MULTIPLICATION OPERATOR
// ============================================================================

    Tensor operator*(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = promote_if_needed(lhs, promoted_dtype);
    Tensor rhs_promoted = promote_if_needed(rhs, promoted_dtype);
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_mul_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a * b;
        });
    }
    return output;
    }
    

// ============================================================================
// DIVISION OPERATOR
// ============================================================================

    Tensor operator/(const Tensor& lhs, const Tensor& rhs) 
    {
    //  USE DIVISION-SPECIFIC PROMOTION
    Dtype promoted_dtype = promote_dtypes_division(lhs.dtype(), rhs.dtype());
    
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, promoted_dtype, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_div_tensor(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else
    {
        apply_binary_operation(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a / b;
        });
    }
    return output;
    }
// ============================================================================
// IN-PLACE OPERATORS (NO TYPE PROMOTION ALLOWED)
// ============================================================================

    Tensor operator+=(Tensor& lhs, const Tensor& rhs)
    {
    //  In-place ops require compatible types (rhs must be promotable to lhs)
    if (lhs.dtype() != rhs.dtype()) {
        // Check if rhs can be safely promoted to lhs type
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator +=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    // Convert rhs to lhs dtype if needed
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_add_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a + b;
        });
    }
    return lhs;
    }


    Tensor operator+=(Tensor&& lhs, const Tensor& rhs) { return lhs += rhs; }

    Tensor operator-=(Tensor& lhs, const Tensor& rhs)
    {
    if (lhs.dtype() != rhs.dtype()) {
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator -=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_sub_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a - b;
        });
    }
    return lhs;
    }

    Tensor operator-=(Tensor&& lhs, const Tensor& rhs) { return lhs -= rhs; }


    Tensor operator*=(Tensor& lhs, const Tensor& rhs)
    {
    if (lhs.dtype() != rhs.dtype()) {
        Dtype promoted = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
        if (promoted != lhs.dtype()) {
            throw std::runtime_error(
                "In-place operator *=: Cannot promote result type beyond lhs dtype. "
                "LHS: " + get_dtype_name(lhs.dtype()) + ", RHS: " + get_dtype_name(rhs.dtype())
            );
        }
    }
    
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_mul_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a * b;
        });
    }
    return lhs;
    }


    Tensor operator*=(Tensor&& lhs, const Tensor& rhs) { return lhs *= rhs; }

    Tensor operator/=(Tensor& lhs, const Tensor& rhs)
    {
    //  Check if division would require float promotion
    Dtype div_promoted = promote_dtypes_division(lhs.dtype(), rhs.dtype());
    
    if (div_promoted != lhs.dtype()) {
        throw std::runtime_error(
            "In-place operator /=: Division requires float promotion. "
            "LHS: " + get_dtype_name(lhs.dtype()) + 
            " cannot store result of type " + get_dtype_name(div_promoted) + ". "
            "Use regular division (/) instead."
        );
    }
    
    // Rest of the code remains the same...
    Tensor rhs_converted = (rhs.dtype() != lhs.dtype()) ? rhs.as_type(lhs.dtype()) : rhs;
    
    if (lhs.shape().dims != rhs_converted.shape().dims) {
        Shape broadcasted = Shape{broadcast_shape(lhs.shape().dims, rhs_converted.shape().dims)};
        if (lhs.shape().dims != broadcasted.dims) {
            throw std::runtime_error("In-place operator: output shape must match lhs shape. Cannot broadcast");
        }
    }

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_div_tensor_inplace(lhs, rhs_converted, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_operation(lhs, rhs_converted, lhs, [](auto a, auto b) {
            return a / b;
        });
    }
    return lhs;
    }


    Tensor operator/=(Tensor&& lhs, const Tensor& rhs) { return lhs /= rhs; }

   // ============================================================================
// COMPARISON OPERATORS WITH TYPE PROMOTION (Fixed Version)
// ============================================================================

    Tensor operator==(const Tensor& lhs, const Tensor& rhs) 
    {
    //  1. Promote types to common dtype (use arithmetic promotion rules)
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    
    //  2. Convert operands if needed
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    //  3. Compute output shape (broadcasting)
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    //  4. Create output tensor (always Bool dtype)
    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    //  5. Perform comparison using promoted types
    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_eq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a == b;
        });
    }
    return output;
    }
    
    Tensor operator!=(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_neq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            return a != b;
        });
    }
    return output;
    }

    Tensor operator<=(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_leq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            using T = decltype(a);
            constexpr bool is_complex = 
                std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t> ||
                std::is_same_v<T, float4_e2m1_2x_t>;
            if constexpr (is_complex) {
                return false; // Complex numbers are not ordered
            } else {
                return a <= b;
            }
        });
    }
    return output;
    }

    Tensor operator>=(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_geq_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            using T = decltype(a);
            constexpr bool is_complex = 
                std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t> ||
                std::is_same_v<T, float4_e2m1_2x_t>;
            if constexpr (is_complex) {
                return false; // Complex numbers are not ordered
            } else {
                return a >= b;
            }
        });
    }
    return output;
    }

    Tensor operator>(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_gt_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            using T = decltype(a);
            constexpr bool is_complex = 
                std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t> ||
                std::is_same_v<T, float4_e2m1_2x_t>;
            if constexpr (is_complex) {
                return false; // Complex numbers are not ordered
            } else {
                return a > b;
            }
        });
    }
    return output;
    }

    Tensor operator<(const Tensor& lhs, const Tensor& rhs) 
    {
    Dtype promoted_dtype = promote_dtypes_bool(lhs.dtype(), rhs.dtype());
    Tensor lhs_promoted = (lhs.dtype() != promoted_dtype) ? lhs.as_type(promoted_dtype) : lhs;
    Tensor rhs_promoted = (rhs.dtype() != promoted_dtype) ? rhs.as_type(promoted_dtype) : rhs;
    
    Shape output_shape = lhs_promoted.shape();
    if (lhs_promoted.shape().dims != rhs_promoted.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs_promoted.shape().dims, rhs_promoted.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cuda_bool_lt_outplace(lhs_promoted, rhs_promoted, output, stream);
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        apply_binary_op_bool(lhs_promoted, rhs_promoted, output, [](auto a, auto b) {
            using T = decltype(a);
            constexpr bool is_complex = 
                std::is_same_v<T, complex32_t> ||
                std::is_same_v<T, complex64_t> ||
                std::is_same_v<T, complex128_t> ||
                std::is_same_v<T, float4_e2m1_2x_t>;
            if constexpr (is_complex) {
                return false; // Complex numbers are not ordered
            } else {
                return a < b;
            }
        });
    }
    return output;
    }

    Tensor logical_AND(const Tensor& lhs, const Tensor& rhs)
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_and_outplace(lhs, rhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            using T1 = decltype(a);
            using T2 = decltype(b);
            bool a_bool = (a != T1(0.0f));
            bool b_bool = (b != T2(0.0f));
            return a_bool && b_bool;
        });
        }
        return output;

    }

    Tensor logical_OR(const Tensor& lhs, const Tensor& rhs)
    {
        Shape output_shape = lhs.shape();
        if (lhs.shape().dims != rhs.shape().dims) {
            output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
        }
    
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda() && rhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_or_outplace(lhs, rhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            using T1 = decltype(a);
            using T2 = decltype(b);
            bool a_bool = (a != T1(0.0f));
            bool b_bool = (b != T2(0.0f));
            return a_bool || b_bool;
        });
        }
        return output;
        
    }

    Tensor logical_XOR(const Tensor& lhs, const Tensor& rhs)
    {
    Shape output_shape = lhs.shape();
    if (lhs.shape().dims != rhs.shape().dims) {
        output_shape = Shape{broadcast_shape(lhs.shape().dims, rhs.shape().dims)};
    }

    Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

    if (lhs.device().is_cuda() && rhs.device().is_cuda())
    {
        #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            cuda_logical_xor_outplace(lhs, rhs, output, stream);//✨✨✨
        #else
            throw std::runtime_error("Tensor Ops: CUDA support not compiled");
        #endif
    }
    else 
    {
        //  FIXED: Convert to bool first, then XOR
        apply_binary_op_bool(lhs, rhs, output, [](auto a, auto b) {
            // Convert to boolean (non-zero = true), then XOR
            using T1 = decltype(a);
            using T2 = decltype(b);
            bool a_bool = (a != T1(0.0f));
            bool b_bool = (b != T2(0.0f));
            return a_bool != b_bool;  // XOR is "not equal" of boolean values
        });
    }
    return output;

    }

    Tensor logical_NOT(const Tensor& lhs)
    {
        Shape output_shape = lhs.shape();
        
        Tensor output(output_shape, Dtype::Bool, lhs.device(), lhs.requires_grad());

        if (lhs.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //✨✨✨
                cuda_logical_not_outplace(lhs,output, stream); //✨✨✨
           #else
                throw std::runtime_error("Tensor Ops: CUDA support not compiled");
            #endif
        }
        else 
        {
        apply_not_bool(lhs, output, [](auto a) {
            using T = decltype(a);
            return !(a != T(0.0f));
        });
        }
        return output;
        
    }

    // Tensor tril(const Tensor& input, int64_t diagonal)
    // {
    //     if (input.ndim() < 2) {
    //         throw std::runtime_error("tril: input tensor must have at least 2 dimensions");
    //     }

    //     Tensor output(input.shape(), input.dtype(), input.device(), input.requires_grad());

    //     if (input.device().is_cuda())
    //     {
    //         #ifdef WITH_CUDA
    //             cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
    //             cuda_tril_tensor(input, output, diagonal, stream);
    //         #else
    //             throw std::runtime_error("tril: CUDA support not compiled");
    //         #endif
    //     }
    //     else
    //     {
    //         //* for fast processing
    //         Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
            
    //         size_t total_elems = output.numel();
    //         const auto& shape_dims = output.shape().dims;
    //         size_t ndim = shape_dims.size();
    //         size_t H = shape_dims[ndim - 2];
    //         size_t W = shape_dims[ndim - 1];

    //         dispatch_by_dtype(output.dtype(), [&](auto dummy) {
    //             using T = decltype(dummy);
    //             const T* in_ptr = input_contig.data<T>();
    //             T* out_ptr = output.data<T>();

    //             for (size_t i = 0; i < total_elems; ++i) {
    //                 size_t row_idx = (i / W) % H;
    //                 size_t col_idx = i % W;

    //                 if (static_cast<int64_t>(col_idx) > static_cast<int64_t>(row_idx) + diagonal) {
    //                     out_ptr[i] = T{};
    //                 } else {
    //                     out_ptr[i] = in_ptr[i];
    //                 }
    //             }
    //         });
    //     }
    //     return output;
    // }

    Tensor tril(const Tensor& input, int64_t diagonal, double value)
    {
        if (input.ndim() < 2) {
            throw std::runtime_error("tril: input tensor must have at least 2 dimensions");
        }

        Tensor output(input.shape(), input.dtype(), input.device(), input.requires_grad());

        if (input.device().is_cuda())
        {
            #ifdef WITH_CUDA
                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                cuda_tril_tensor(input, output, diagonal, value, stream);
            #else
                throw std::runtime_error("tril: CUDA support not compiled");
            #endif
        }
        else
        {
            //* for fast processing
            Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
            
            size_t total_elems = output.numel();
            const auto& shape_dims = output.shape().dims;
            size_t ndim = shape_dims.size();
            size_t H = shape_dims[ndim - 2];
            size_t W = shape_dims[ndim - 1];

            dispatch_by_dtype(output.dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                const T* in_ptr = input_contig.data<T>();
                T* out_ptr = output.data<T>();

                // Cast value to T once
                T fill_val = static_cast<T>(value);

                for (size_t i = 0; i < total_elems; ++i) {
                    size_t row_idx = (i / W) % H;
                    size_t col_idx = i % W;

                    if (static_cast<int64_t>(col_idx) > static_cast<int64_t>(row_idx) + diagonal) {
                        out_ptr[i] = fill_val;
                    } else {
                        out_ptr[i] = in_ptr[i];
                    }
                }
            });
        }
        return output;
    }

    DropoutResult dropout(const Tensor& input, float prob)
    {
        if (prob == 0.0f)
        {
            Tensor ones(input.shape(), input.dtype(), input.device(), false);
            float* mask_ptr = ones.data<float>();
            for (int i = 0; i < ones.numel(); ++i) mask_ptr[i] = 1.0f;
            return {input, ones};
        }

        if (input.device().is_cuda())
        {
        #ifdef WITH_CUDA
            auto result = dropOut_cuda(input, prob);
            return {result.output, result.mask};
        #else
            throw std::runtime_error("CUDA support not compiled");
        #endif
        }

        int N = input.numel();
        const float *src_ptr = input.data<float>();
        float scale = 1.0f / (1.0f - prob);

        Tensor output(input.shape(), input.dtype(), input.device(), false);
        Tensor mask(input.shape(), input.dtype(), input.device(), false);
        float* out_ptr = output.data<float>();
        float* mask_ptr = mask.data<float>();

        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < N; ++i)
        {
            if (dist(engine) < prob)
            {
                out_ptr[i] = 0.0f;
                mask_ptr[i] = 0.0f;
            }
            else
            {
                out_ptr[i] = src_ptr[i] * scale;
                mask_ptr[i] = 1.0f;
            }
        }

        return {output, mask};
    }
}
