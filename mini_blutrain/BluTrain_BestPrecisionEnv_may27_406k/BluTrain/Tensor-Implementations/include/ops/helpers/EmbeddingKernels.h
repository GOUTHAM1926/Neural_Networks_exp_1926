#pragma once
#include "core/Tensor.h"
#include "dtype/Types.h"

namespace OwnTensor {
namespace cuda {

// Float32
void embedding_forward_cuda(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
);

// Float16
void embedding_forward_cuda(
    const uint16_t* indices,
    const float16_t* weight,
    float16_t* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
);

// Bfloat16
void embedding_forward_cuda(
    const uint16_t* indices,
    const bfloat16_t* weight,
    bfloat16_t* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
);

// Float32
void embedding_backward_cuda(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
);

// Float16
void embedding_backward_cuda(
    const uint16_t* indices,
    const float16_t* grad_output,
    float16_t* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
);

// Bfloat16
void embedding_backward_cuda(
    const uint16_t* indices,
    const bfloat16_t* grad_output,
    bfloat16_t* grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
);

} // namespace cuda
} // namespace OwnTensor