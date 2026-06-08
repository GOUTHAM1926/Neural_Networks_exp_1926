#include "ops/cuda/activations/ActivationCommon.cuh"
#include <type_traits>

namespace OwnTensor {
namespace cuda {

template <typename T>
__global__ void sigmoid_forward_kernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = to_float(input[i]);
        output[i] = from_float<T>(fast_sigmoid(val));
    }
}

template<typename T>
__global__ void sigmoid_backward_kernel(
    const T *__restrict__ grad_output,
    const T *__restrict__ output,
    T *__restrict__ grad_input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float s = to_float(output[i]);
        float grad = to_float(grad_output[i]);
        grad_input[i] = from_float<T>(grad * s * (1.0f - s));
    }
}

template<typename T>
void launch_sigmoid_generic(const T* in, T* out, int64_t n, cudaStream_t s) {
    int threads = 256;
    sigmoid_forward_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
}
template void launch_sigmoid_generic<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_sigmoid_generic<__half>(const __half*, __half*, int64_t, cudaStream_t);
template void launch_sigmoid_generic<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);
template<typename T>
void launch_sigmoid_backward_generic(const T* go, const T* out, T* gi, int64_t n, cudaStream_t s) {
    int threads = 256;
    sigmoid_backward_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(go, out, gi, n);
}
template void launch_sigmoid_backward_generic<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_sigmoid_backward_generic<__half>(const __half*, const __half*, __half*, int64_t, cudaStream_t);
template void launch_sigmoid_backward_generic<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);
// Explicit Kernel Instantiations
template __global__ void sigmoid_forward_kernel<float>(const float*, float*, int64_t);
template __global__ void sigmoid_forward_kernel<__half>(const __half*, __half*, int64_t);
template __global__ void sigmoid_forward_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);
template __global__ void sigmoid_backward_kernel<float>(const float*, const float*, float*, int64_t);
template __global__ void sigmoid_backward_kernel<__half>(const __half*, const __half*, __half*, int64_t);
template __global__ void sigmoid_backward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t);

} // namespace cuda
} // namespace OwnTensor
