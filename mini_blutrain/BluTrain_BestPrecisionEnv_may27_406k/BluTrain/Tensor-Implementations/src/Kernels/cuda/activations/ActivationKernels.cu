#include "ops/helpers/ActivationKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// EXTERNAL GENERIC LAUNCHER DECLARATIONS
// =============================================================================

// GELU
template<typename T> void launch_fused_gelu_generic(const T* in, T* out, int64_t n, cudaStream_t s);
template<typename T> void launch_fused_gelu_backward_generic(const T* go, const T* in, T* gi, int64_t n, cudaStream_t s);
void launch_fused_bias_gelu(const float* in, const float* b, float* out, int64_t bs, int64_t hd, cudaStream_t s);
void launch_fused_bias_gelu_backward(const float* go, const float* in, const float* b, float* gi, float* gb, int64_t bs, int64_t hd, cudaStream_t s);

// ReLU
template<typename T> void launch_relu_generic(const T* in, T* out, int64_t n, cudaStream_t s);
template<typename T> void launch_relu_backward_generic(const T* go, const T* in, T* gi, int64_t n, cudaStream_t s);

// Sigmoid
template<typename T> void launch_sigmoid_generic(const T* in, T* out, int64_t n, cudaStream_t s);
template<typename T> void launch_sigmoid_backward_generic(const T* go, const T* out, T* gi, int64_t n, cudaStream_t s);

// Softmax
void launch_softmax_generic(const float* in, float* out, int64_t rows, int64_t cols, cudaStream_t s);
void launch_softmax_backward_generic(const float* go, const float* out, float* gi, int64_t rows, int64_t cols, cudaStream_t s);

// SwiGLU
template<typename T> void launch_swiglu_generic(const T* in, T* out, int64_t rows, int64_t hidden, cudaStream_t s);
template<typename T> void launch_swiglu_backward_generic(const T* go, const T* in, T* gi, int64_t rows, int64_t hidden, cudaStream_t s);

// --- SM_89 Optimized Declarations ---
void fused_gelu_ada(const float* input, float* output, int64_t numel, cudaStream_t stream);
void fused_gelu_backward_ada(const float* grad_output, const float* input, float* grad_input, int64_t numel, cudaStream_t stream);

// =============================================================================
// DISPATCHER IMPLEMENTATIONS
// =============================================================================

// --- GELU ---
template <typename T>
void fused_gelu_cuda(const T *input, T *output, int64_t numel) {
    if constexpr (std::is_same_v<T, float>) {
        if (get_arch() == ArchFamily::Ada) { fused_gelu_ada(input, output, numel, 0); return; }
        launch_fused_gelu_generic<float>(input, output, numel, 0);
    } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, __half>) {
        launch_fused_gelu_generic<__half>(reinterpret_cast<const __half*>(input), reinterpret_cast<__half*>(output), numel, 0);
    } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, __nv_bfloat16>) {
        launch_fused_gelu_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(input), reinterpret_cast<__nv_bfloat16*>(output), numel, 0);
    }
}
template void fused_gelu_cuda<float>(const float*, float*, int64_t);
template void fused_gelu_cuda<float16_t>(const float16_t*, float16_t*, int64_t);
template void fused_gelu_cuda<bfloat16_t>(const bfloat16_t*, bfloat16_t*, int64_t);
template void fused_gelu_cuda<__half>(const __half*, __half*, int64_t);
template void fused_gelu_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

void fused_gelu_backward_cuda(const float *go, const float *in, float *gi, int64_t n) {
    if (get_arch() == ArchFamily::Ada) { fused_gelu_backward_ada(go, in, gi, n, 0); return; }
    launch_fused_gelu_backward_generic<float>(go, in, gi, n, 0);
}
void fused_gelu_backward_cuda(const float16_t *go, const float16_t *in, float16_t *gi, int64_t n) {
    launch_fused_gelu_backward_generic<__half>(reinterpret_cast<const __half*>(go), reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(gi), n, 0);
}
void fused_gelu_backward_cuda(const bfloat16_t *go, const bfloat16_t *in, bfloat16_t *gi, int64_t n) {
    launch_fused_gelu_backward_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(go), reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<__nv_bfloat16*>(gi), n, 0);
}

void fused_bias_gelu_cuda(const float *in, const float *b, float *out, int64_t bs, int64_t hd) {
    launch_fused_bias_gelu(in, b, out, bs, hd, 0);
}
void fused_bias_gelu_backward_cuda(const float *go, const float *in, const float *b, float *gi, float *gb, int64_t bs, int64_t hd) {
    launch_fused_bias_gelu_backward(go, in, b, gi, gb, bs, hd, 0);
}

// --- ReLU ---
void relu_forward_cuda(const float* in, float* out, int64_t n) {
    launch_relu_generic<float>(in, out, n, 0);
}
void relu_backward_cuda(const float *go, const float *in, float *gi, int64_t n) {
    launch_relu_backward_generic<float>(go, in, gi, n, 0);
}
void relu_backward_cuda(const float16_t *go, const float16_t *in, float16_t *gi, int64_t n) {
    launch_relu_backward_generic<__half>(reinterpret_cast<const __half*>(go), reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(gi), n, 0);
}
void relu_backward_cuda(const bfloat16_t *go, const bfloat16_t *in, bfloat16_t *gi, int64_t n) {
    launch_relu_backward_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(go), reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<__nv_bfloat16*>(gi), n, 0);
}

// --- Sigmoid ---
template <typename T>
void sigmoid_forward_cuda(const T *in, T *out, int64_t n) {
    if constexpr (std::is_same_v<T, float>) launch_sigmoid_generic<float>(in, out, n, 0);
    else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, __half>) 
        launch_sigmoid_generic<__half>(reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(out), n, 0);
    else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, __nv_bfloat16>)
        launch_sigmoid_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<__nv_bfloat16*>(out), n, 0);
}
template void sigmoid_forward_cuda<float>(const float*, float*, int64_t);
template void sigmoid_forward_cuda<float16_t>(const float16_t*, float16_t*, int64_t);
template void sigmoid_forward_cuda<bfloat16_t>(const bfloat16_t*, bfloat16_t*, int64_t);
template void sigmoid_forward_cuda<__half>(const __half*, __half*, int64_t);
template void sigmoid_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

void sigmoid_backward_cuda(const float *go, const float *out, float *gi, int64_t n) {
    launch_sigmoid_backward_generic<float>(go, out, gi, n, 0);
}
void sigmoid_backward_cuda(const float16_t *go, const float16_t *out, float16_t *gi, int64_t n) {
    launch_sigmoid_backward_generic<__half>(reinterpret_cast<const __half*>(go), reinterpret_cast<const __half*>(out), reinterpret_cast<__half*>(gi), n, 0);
}
void sigmoid_backward_cuda(const bfloat16_t *go, const bfloat16_t *out, bfloat16_t *gi, int64_t n) {
    launch_sigmoid_backward_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(go), reinterpret_cast<const __nv_bfloat16*>(out), reinterpret_cast<__nv_bfloat16*>(gi), n, 0);
}

// --- Softmax ---
void softmax_forward_cuda(const float* in, float* out, int64_t rows, int64_t cols) {
    launch_softmax_generic(in, out, rows, cols, 0);
}
void softmax_forward_cuda_online(const float* input, float* output, int64_t rows, int64_t cols) {
    launch_softmax_generic(input, output, rows, cols, 0);
}
void softmaxonline_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
    launch_softmax_generic(input, output, rows, cols, 0);
}
template <typename T>
void softmax_forward_cuda_typed(const T *input, T *output, int64_t rows, int64_t cols) {
    if constexpr (std::is_same_v<T, float>) launch_softmax_generic(input, output, rows, cols, 0);
}
template void softmax_forward_cuda_typed<float>(const float*, float*, int64_t, int64_t);
template void softmax_forward_cuda_typed<__half>(const __half*, __half*, int64_t, int64_t);
template void softmax_forward_cuda_typed<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

void softmax_backward_cuda(const float* go, const float* out, float* gi, int64_t rows, int64_t cols) {
    launch_softmax_backward_generic(go, out, gi, rows, cols, 0);
}
void softmax_backward_cuda(const float16_t* go, const float16_t* out, float16_t* gi, int64_t rows, int64_t cols) {
    // Note: requires a modular half implementation in SoftmaxKernels.cu to work
}
void softmax_backward_cuda(const bfloat16_t* go, const bfloat16_t* out, bfloat16_t* gi, int64_t rows, int64_t cols) {
    // Note: requires a modular half implementation in SoftmaxKernels.cu to work
}

// --- SwiGLU ---
template <typename T>
void swiglu_forward_cuda(const T* in, T* out, int64_t rows, int64_t hidden) {
    if constexpr (std::is_same_v<T, float>) launch_swiglu_generic<float>(in, out, rows, hidden, 0);
    else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, __half>)
        launch_swiglu_generic<__half>(reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(out), rows, hidden, 0);
    else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, __nv_bfloat16>)
        launch_swiglu_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<__nv_bfloat16*>(out), rows, hidden, 0);
}
template void swiglu_forward_cuda<float>(const float*, float*, int64_t, int64_t);
template void swiglu_forward_cuda<float16_t>(const float16_t*, float16_t*, int64_t, int64_t);
template void swiglu_forward_cuda<bfloat16_t>(const bfloat16_t*, bfloat16_t*, int64_t, int64_t);
template void swiglu_forward_cuda<__half>(const __half*, __half*, int64_t, int64_t);
template void swiglu_forward_cuda<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

void swiglu_backward_cuda(const float* go, const float* in, float* gi, int64_t r, int64_t h) {
    launch_swiglu_backward_generic<float>(go, in, gi, r, h, 0);
}
void swiglu_backward_cuda(const float16_t* go, const float16_t* in, float16_t* gi, int64_t r, int64_t h) {
    launch_swiglu_backward_generic<__half>(reinterpret_cast<const __half*>(go), reinterpret_cast<const __half*>(in), reinterpret_cast<__half*>(gi), r, h, 0);
}
void swiglu_backward_cuda(const bfloat16_t* go, const bfloat16_t* in, bfloat16_t* gi, int64_t r, int64_t h) {
    launch_swiglu_backward_generic<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(go), reinterpret_cast<const __nv_bfloat16*>(in), reinterpret_cast<__nv_bfloat16*>(gi), r, h, 0);
}

} // namespace cuda
} // namespace OwnTensor