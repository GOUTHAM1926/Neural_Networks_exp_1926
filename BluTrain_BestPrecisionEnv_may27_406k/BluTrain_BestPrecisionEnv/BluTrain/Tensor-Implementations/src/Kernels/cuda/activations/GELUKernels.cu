#include "ops/cuda/activations/ActivationCommon.cuh"
#include <type_traits>

namespace OwnTensor {
namespace cuda {

template <typename T>
__global__ void fused_gelu_kernel(const T *__restrict__ input,
                                  T *__restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float x = to_float(input[i]);
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x);
        output[i] = from_float<T>(0.5f * x * (1.0f + fast_tanh(inner)));
    }
}

// Vectorized for float
__global__ void fused_gelu_kernel_vectorized(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t numel4 = numel / 4;

    for (int64_t i = idx; i < numel4; i += stride) {
        float4 x_vec = reinterpret_cast<const float4 *>(input)[i];
        float4 out_vec;
        out_vec.x = 0.5f * x_vec.x * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x_vec.x + GELU_COEF * x_vec.x * x_vec.x * x_vec.x)));
        out_vec.y = 0.5f * x_vec.y * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x_vec.y + GELU_COEF * x_vec.y * x_vec.y * x_vec.y)));
        out_vec.z = 0.5f * x_vec.z * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x_vec.z + GELU_COEF * x_vec.z * x_vec.z * x_vec.z)));
        out_vec.w = 0.5f * x_vec.w * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x_vec.w + GELU_COEF * x_vec.w * x_vec.w * x_vec.w)));
        reinterpret_cast<float4 *>(output)[i] = out_vec;
    }
    // Tail
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x)));
    }
}

template<typename T>
__global__ void fused_gelu_backward_kernel(const T *__restrict__ grad_output,
                                           const T *__restrict__ input,
                                           T *__restrict__ grad_input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float x = to_float(input[i]);
        float x2 = x * x;
        float u = SQRT_2_OVER_PI * (x + GELU_COEF * x2 * x);
        float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
        float tanh_u = fast_tanh(u);
        float sech2_u = 1.0f - tanh_u * tanh_u;
        float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
        grad_input[i] = from_float<T>(to_float(grad_output[i]) * gelu_grad);
    }
}

// --- Fused Bias GELU (Float32 only for now as per original) ---
__global__ void fused_bias_gelu_kernel(const float *__restrict__ input, 
    const float *__restrict__ bias, 
    float *__restrict__ output, 
    int64_t batch_size, 
    int64_t hidden_dim) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * hidden_dim;
    if (i < total) {
        float x = input[i] + bias[i % hidden_dim];
        output[i] = 0.5f * x * (1.0f + fast_tanh(SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x)));
    }
}

// Low-level launchers (Generic)
template<typename T>
void launch_fused_gelu_generic(const T* in, T* out, int64_t n, cudaStream_t s) {
    int threads = 256;
    if constexpr (std::is_same_v<T, float>) {
        if (n >= 1024 && n % 4 == 0) {
            fused_gelu_kernel_vectorized<<<std::min((n/4+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
        } else {
            fused_gelu_kernel<float><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
        }
    } else {
        // Fallback for half/bf16
        fused_gelu_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, out, n);
    }
}

template<typename T>
void launch_fused_gelu_backward_generic(const T* go, const T* in, T* gi, int64_t n, cudaStream_t s){
    int threads = 256;
    fused_gelu_backward_kernel<T><<<std::min((n+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(go, in, gi, n);
}



void launch_fused_bias_gelu(const float* in, const float* b, float* out, int64_t bs, int64_t hd, cudaStream_t s) {
    int threads = 256;
    int64_t total = bs * hd;
    fused_bias_gelu_kernel<<<std::min((total+threads-1)/threads, (int64_t)65535), threads, 0, s>>>(in, b, out, bs, hd);
}

// Explicit Instantiations
template void launch_fused_gelu_generic<float>(const float*, float*, int64_t, cudaStream_t);
template void launch_fused_gelu_generic<__half>(const __half*, __half*, int64_t, cudaStream_t);
template void launch_fused_gelu_generic<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);

template void launch_fused_gelu_backward_generic<float>(const float*, const float*, float*, int64_t, cudaStream_t);
template void launch_fused_gelu_backward_generic<__half>(const __half*, const __half*, __half*, int64_t, cudaStream_t);
template void launch_fused_gelu_backward_generic<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t, cudaStream_t);

// Explicitly instantiate for all types
template __global__ void fused_gelu_kernel<float>(const float*, float*, int64_t);
template __global__ void fused_gelu_kernel<__half>(const __half*, __half*, int64_t);
template __global__ void fused_gelu_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t);

template __global__ void fused_gelu_backward_kernel<float>(const float*, const float*, float*, int64_t);
template __global__ void fused_gelu_backward_kernel<__half>(const __half*, const __half*, __half*, int64_t);
template __global__ void fused_gelu_backward_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int64_t);

// Pass 1: compute grad_input (Grid-stride loop to handle any size)
__global__ void fused_bias_gelu_backward_grad_input_kernel(
    const float *__restrict__ grad_output,
    const float *__restrict__ input, 
    const float *__restrict__ bias, 
    float *__restrict__ grad_input, 
    int64_t batch_size, 
    int64_t hidden_dim) {
    int64_t total = batch_size * hidden_dim;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        float x = input[i] + bias[i % hidden_dim];
        float x2 = x * x;
        float u = SQRT_2_OVER_PI * (x + GELU_COEF * x2 * x);
        float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);
        float tanh_u = fast_tanh(u);
        float sech2_u = 1.0f - tanh_u * tanh_u;
        float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
        grad_input[i] = grad_output[i] * gelu_grad;
    }
}

// Pass 2: reduce grad_input columns -> grad_bias (shared-memory reduction)
__global__ void bias_grad_reduce_kernel(const float *__restrict__ grad_input, 
    float *__restrict__ grad_bias, 
    int64_t batch_size, 
    int64_t hidden_dim) {
    int64_t j = blockIdx.x; // one block per bias column
    if (j >= hidden_dim) return;
    extern __shared__ float smem[];
    float partial = 0.0f;
    for (int64_t row = threadIdx.x; row < batch_size; row += blockDim.x)
        partial += grad_input[row * hidden_dim + j];
    smem[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&grad_bias[j], smem[0]);
}


void launch_fused_bias_gelu_backward(const float* go, const float* in, const float* b, float* gi, float* gb, int64_t bs, int64_t hd, cudaStream_t s) {
    int threads = 256;
    int64_t total = bs * hd;
    int grid = std::min((total + threads - 1) / threads, (int64_t)4096); // Capped grid but uses loop
    fused_bias_gelu_backward_grad_input_kernel<<<grid, threads, 0, s>>>(go, in, b, gi, bs, hd);
    
    // Grid size for reduce: one block per hidden_dim column
    bias_grad_reduce_kernel<<<hd, threads, threads * sizeof(float), s>>>(gi, gb, bs, hd);
}

} // namespace cuda
} // namespace OwnTensor
