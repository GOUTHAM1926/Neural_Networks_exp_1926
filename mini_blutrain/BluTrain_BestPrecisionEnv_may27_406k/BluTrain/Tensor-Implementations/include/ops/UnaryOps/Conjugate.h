#pragma once

#include "TensorLib.h"

namespace OwnTensor {

// Conjugate operation for tensors (returns a new tensor)
// Supports complex32_t, complex64_t, complex128_t. For real types, returns a copy.
Tensor conj(const Tensor& input, cudaStream_t stream = 0);

// In-place conjugate (modifies the input tensor)
void conj_(Tensor& input, cudaStream_t stream = 0);

} // namespace OwnTensor
