#pragma once
#include <cstdint>
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// Float32
void fused_tril_softmax_backward_cuda(
    const float* grad_output, const float* output, float* grad_input,
    int64_t rows, int64_t cols);

// Float16
void fused_tril_softmax_backward_cuda(
    const float16_t* grad_output, const float16_t* output, float16_t* grad_input,
    int64_t rows, int64_t cols);

// Bfloat16
void fused_tril_softmax_backward_cuda(
    const bfloat16_t* grad_output, const bfloat16_t* output, bfloat16_t* grad_input,
    int64_t rows, int64_t cols);

} // namespace cuda
} // namespace OwnTensor
