#pragma once
#include <cstdint>
#include "dtype/Dtype.h"
#include "dtype/Types.h"


namespace OwnTensor {
namespace cuda {

/**
 * @brief Fused GELU activation kernel
 *
 * Computes GELU in a single kernel launch instead of 6+ separate operations:
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
template <typename T>
void fused_gelu_cuda(const T *input, T *output, int64_t numel);

/**
 * @brief Fused GELU backward kernel
 *
 * Computes gradient in a single kernel launch.
 */
void fused_gelu_backward_cuda(const float *grad_output, const float *input,
                              float *grad_input, int64_t numel);

/**
 * @brief Fused bias + GELU kernel
 *
 * Computes output = gelu(input + bias) in a single kernel.
 * bias is broadcast along the last dimension.
 */
void fused_bias_gelu_cuda(const float *input, const float *bias, float *output,
                          int64_t batch_size, // Total elements / hidden_dim
                          int64_t hidden_dim  // Size of bias vector
);

// ReLU
void relu_forward_cuda(const float *input, float *output, int64_t numel);


// Sigmoid
template <typename T>
void sigmoid_forward_cuda(const T *input, T *output, int64_t numel);
void sigmoid_backward_cuda(const float *grad_output, const float *output,
                           float *grad_input, int64_t numel);

// Softmax (along last dimension)
void softmax_forward_cuda(const float *input, float *output, int64_t rows,
                          int64_t cols);
template <typename T>
void softmax_forward_cuda_typed(const T *input, T *output, int64_t rows,
                                int64_t cols);
void softmax_backward_cuda(const float *grad_output, const float *output,
                           float *grad_input, int64_t rows, int64_t cols);
void softmax_forward_cuda_online(const float* input, float* output, int64_t rows, int64_t cols);
void softmaxonline_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols);
template <typename T>
void swiglu_forward_cuda(const T* input, T* output, int64_t rows, int64_t hidden);

// GELU forward
template <typename T>
void fused_gelu_cuda(const T *input, T *output, int64_t numel);

// GELU backward
void fused_gelu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel);
void fused_gelu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel);
void fused_gelu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel);

// Fused bias + GELU (float32 only)
void fused_bias_gelu_cuda(const float *input, const float *bias, float *output, int64_t batch_size, int64_t hidden_dim);
void fused_bias_gelu_backward_cuda(const float *grad_output, const float *input, const float *bias, float *grad_input, float *grad_bias, int64_t batch_size, int64_t hidden_dim);


// ReLU backward
void relu_backward_cuda(const float *grad_output, const float *input, float *grad_input, int64_t numel);
void relu_backward_cuda(const float16_t *grad_output, const float16_t *input, float16_t *grad_input, int64_t numel);
void relu_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *input, bfloat16_t *grad_input, int64_t numel);


// Sigmoid backward
void sigmoid_backward_cuda(const float *grad_output, const float *output, float *grad_input, int64_t numel);
void sigmoid_backward_cuda(const float16_t *grad_output, const float16_t *output, float16_t *grad_input, int64_t numel);
void sigmoid_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *output, bfloat16_t *grad_input, int64_t numel);


// Softmax backward
void softmax_backward_cuda(const float *grad_output, const float *output, float *grad_input, int64_t rows, int64_t cols);
void softmax_backward_cuda(const float16_t *grad_output, const float16_t *output, float16_t *grad_input, int64_t rows, int64_t cols);
void softmax_backward_cuda(const bfloat16_t *grad_output, const bfloat16_t *output, bfloat16_t *grad_input, int64_t rows, int64_t cols);


// SwiGLU backward
void swiglu_backward_cuda(const float* grad_out, const float* input, float* grad_input, int64_t rows, int64_t hidden);
void swiglu_backward_cuda(const float16_t* grad_out, const float16_t* input, float16_t* grad_input, int64_t rows, int64_t hidden);
void swiglu_backward_cuda(const bfloat16_t* grad_out, const bfloat16_t* input, bfloat16_t* grad_input, int64_t rows, int64_t hidden);


/**
 * @brief Fused Linear + GELU using cuBLASLt epilogue
 * Handles Float32, Float16, and Bfloat16 with architecture detection.
 */
void fused_linear_gelu_forward(
    const void*  input,
    const void*  weight,
    const void*  bias,
    void*        output,
    int64_t      M,
    int64_t      N,
    int64_t      K,
    Dtype        dtype,
    int          device_idx = -1);

void fused_linear_gelu_forward_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int64_t      M,
    int64_t      N,
    int64_t      K);

} // namespace cuda
} // namespace OwnTensor
