#include "ops/cuda/activations/ActivationCommon.cuh"
#include <type_traits>

namespace OwnTensor {
namespace cuda {

template<typename T>
__global__ void relu_forward_kernel(
    const T *__restrict__ input,
    T *__restrict__ output, 
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = to_float(input[i]);
        output[i] = from_float<T>(val > 0.0f ? val : 0.0f);
    }
}

template<typename T>
__global__ void relu_backward_kernel(
    const T *__restrict__ grad_output,
    const T *__restrict__ input,
    T *__restrict__ grad_input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = to_float(input[i]);
        float grad = to_float(grad_output[i]);
        grad_input[i] = from_float<T>(val > 0.0f ? grad : 0.0f);
    }
}

void launch_relu_generic(const float* in, float* out, int64_t n, cudaStream_t s) {
    int threads = 256;
    relu_forward_kernel<float><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
}

void launch_relu_backward_generic(const float* go, const float* in, float* gi, int64_t n, cudaStream_t s) {
    int threads = 256;
    relu_backward_kernel<float><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(go, in, gi, n);
}

template<typename T>
void launch_relu_generic(const T* in, T* out, int64_t n, cudaStream_t s) {
    int threads = 256;
    relu_forward_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
}
template void launch_relu_generic<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_relu_generic<__half>(const __half*, __half*, int64_t, cudaStream_t);
template void launch_relu_generic<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);

template<typename T>
void launch_relu_backward_generic(const T* go, const T* in, T* gi, int64_t n, cudaStream_t s) {
    int threads = 256;
    relu_backward_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(go, in, gi, n);
}
template void launch_relu_backward_generic<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_relu_backward_generic<__half>(const __half*, const __half*, __half*, int64_t, cudaStream_t);
template void launch_relu_backward_generic<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);


} // namespace cuda
} // namespace OwnTensor
