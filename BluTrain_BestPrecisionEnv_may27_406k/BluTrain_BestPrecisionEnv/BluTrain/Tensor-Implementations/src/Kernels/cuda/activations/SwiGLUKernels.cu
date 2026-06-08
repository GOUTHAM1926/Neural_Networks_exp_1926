#include "ops/cuda/activations/ActivationCommon.cuh"
#include <type_traits>

namespace OwnTensor {
namespace cuda {

template <typename T>
__global__ void swiglu_forward_kernel(
    const T *__restrict__ input, 
    T *__restrict__ output, 
    int64_t rows, 
    int64_t hidden) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = rows * hidden;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / hidden, col = i % hidden, base = row * hidden * 2;
        float a = to_float(input[base + col]);
        float b = to_float(input[base + hidden + col]);
        output[i] = from_float<T>(a * fast_sigmoid(a) * b);
    }
}

template<typename T>
__global__ void swiglu_backward_kernel(
    const T *__restrict__ grad_out, 
    const T *__restrict__ input, 
    T *__restrict__ grad_input, 
    int64_t rows, 
    int64_t hidden) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = rows * hidden;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / hidden, col = i % hidden, base = row * hidden * 2;
        float a = to_float(input[base + col]), b = to_float(input[base + hidden + col]), g = to_float(grad_out[i]);
        float sig = fast_sigmoid(a), swish = a * sig;
        float dA = b * (sig + a * sig * (1.0f - sig)), dB = swish;
        grad_input[base + col] = from_float<T>(g * dA);
        grad_input[base + hidden + col] = from_float<T>(g * dB);
    }
}

template<typename T>
void launch_swiglu_generic(const T* in, T* out, int64_t rows, int64_t hidden, cudaStream_t s) {
    int threads = 256;
    int64_t total = rows * hidden;
    swiglu_forward_kernel<T><<<std::min((total+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, rows, hidden);
}
template void launch_swiglu_generic<float>(const float*, float*, int64_t, int64_t, cudaStream_t);
template void launch_swiglu_generic<__half>(const __half*, __half*, int64_t, int64_t, cudaStream_t);
template void launch_swiglu_generic<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, cudaStream_t);

template<typename T>
void launch_swiglu_backward_generic(const T* go, const T* in, T* gi, int64_t rows, int64_t hidden, cudaStream_t s) {
    int threads = 256;
    int64_t total = rows * hidden;
    swiglu_backward_kernel<T><<<std::min((total+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(go, in, gi, rows, hidden);
}
template void launch_swiglu_backward_generic<float>(const float*, const float*, float*, int64_t, int64_t, cudaStream_t);
template void launch_swiglu_backward_generic<__half>(const __half*, const __half*, __half*, int64_t, int64_t, cudaStream_t);
template void launch_swiglu_backward_generic<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, cudaStream_t);
// Explicit Kernel Instantiations
template __global__ void swiglu_forward_kernel<float>(const float*, float*, int64_t, int64_t);
template __global__ void swiglu_forward_kernel<__half>(const __half*, __half*, int64_t, int64_t);
template __global__ void swiglu_forward_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);
template __global__ void swiglu_backward_kernel<float>(const float*, const float*, float*, int64_t, int64_t);
template __global__ void swiglu_backward_kernel<__half>(const __half*, const __half*, __half*, int64_t, int64_t);
template __global__ void swiglu_backward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

} // namespace cuda
} // namespace OwnTensor
