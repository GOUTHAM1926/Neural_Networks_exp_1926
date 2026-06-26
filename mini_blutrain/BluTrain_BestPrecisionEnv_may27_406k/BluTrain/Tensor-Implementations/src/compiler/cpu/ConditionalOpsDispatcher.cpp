#include <stdexcept>
#include "core/Tensor.h"
#include "ops/helpers/ConditionalOps.h"
#include "ops/helpers/BroadcastUtils.h"  //  For broadcast_rhs_to_lhs
#include "core/TensorDispatch.h"
#include "dtype/Types.h"
#include "dtype/fp4.h"
#include "dtype/DtypeTraits.h"  //  For type_to_dtype and promote_dtypes_bool
using namespace std;
namespace OwnTensor {

// // Backend declarations
// void cpu_where(const Tensor& condition, const Tensor& input, 
//                const Tensor& other, Tensor& out);
// void cuda_where(const Tensor& condition, const Tensor& input,
//                 const Tensor& other, Tensor& out);

// Main where implementation
    Tensor where(const Tensor& condition, const Tensor& input, const Tensor& other) {
    // Validate condition dtype
    if (condition.dtype() != Dtype::Bool) {
        throw std::runtime_error("Condition must be Bool dtype");
    }
    
    // Validate all tensors on same device
    if (condition.device().index != input.device().index || input.device().index != other.device().index) {
        throw std::runtime_error("All tensors must be on the same device");
    }
    
    //  Determine output dtype (promote input and other)
    Dtype output_dtype = promote_dtypes_bool(input.dtype(), other.dtype());
    
    //  Compute output shape (broadcasting)
    Shape output_shape = condition.shape();
    if (condition.shape().dims != input.shape().dims || condition.shape().dims != other.shape().dims) {
        // Need to broadcast - compute the broadcasted shape
        std::vector<int64_t> temp_shape = broadcast_shape(condition.shape().dims, input.shape().dims);
        temp_shape = broadcast_shape(temp_shape, other.shape().dims);
        output_shape = Shape{temp_shape};
    }
    
    // Create output tensor
    Tensor out(output_shape, output_dtype, input.device(), false);
    
    // Dispatch to backend
    if (condition.device().is_cuda()) {
        cuda_where(condition, input, other, out);
    } else {
        cpu_where(condition, input, other, out);
    }
    
    return out;
}

// ============================================================================
// NOTE: Template scalar overloads are now implemented inline in the header
// (ConditionalOps.h) to avoid explicit instantiations
// ============================================================================





} // namespace OwnTensor
