#include "core/Tensor.h"
#include "dtype/Types.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/arith.hpp"

namespace OwnTensor {

// ---------------------------------------------------------------------------
// Helper to compute conjugate for any supported type
// ---------------------------------------------------------------------------
template <typename T>
inline T conj_op(const T& val) {
    if constexpr (std::is_same_v<T, complex32_t>) {
        return conj(val);
    } else if constexpr (std::is_same_v<T, complex64_t>) {
        return conj(val);
    } else if constexpr (std::is_same_v<T, complex128_t>) {
        return conj(val);
    } else {
        // Real types: identity
        return val;
    }
}

// ---------------------------------------------------------------------------
// Out-of-place conjugate (returns a new tensor)
// ---------------------------------------------------------------------------
Tensor conj(const Tensor& input, [[maybe_unused]] cudaStream_t stream = 0) {
    // For now we ignore stream on CPU path
    Dtype dt = input.dtype();
    Tensor output(input.shape(), dt, input.device(), input.requires_grad());

    dispatch_by_dtype(dt, [&](auto type_instance) {
        using T = decltype(type_instance);
        const T* in_ptr = input.data<T>();
        T* out_ptr = output.data<T>();
        const size_t N = input.numel();
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            out_ptr[i] = conj_op<T>(in_ptr[i]);
        }
    });
    return output;
}

// ---------------------------------------------------------------------------
// In-place conjugate (modifies the input tensor)
// ---------------------------------------------------------------------------
void conj_(Tensor& input, [[maybe_unused]] cudaStream_t stream = 0) {
    // For now we ignore stream on CPU path
    Dtype dt = input.dtype();
    dispatch_by_dtype(dt, [&](auto type_instance) {
        using T = decltype(type_instance);
        T* ptr = input.data<T>();
        const size_t N = input.numel();
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            ptr[i] = conj_op<T>(ptr[i]);
        }
    });
}

} // namespace OwnTensor
