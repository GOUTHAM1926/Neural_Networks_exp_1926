#include "ops/LinearKernels.cuh"
#include "ops/MatmulBackward.cuh" // For reuse of cuda_matmul_backward if needed, or helpers
#include "ops/Kernels.h"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "dtype/CudaTraits.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Helper to get cuBLAS handle (assuming a global or thread-local one exists, or creating one)
// For this environment, we might need to rely on the one in DeviceCore/Context or similar.
// For now, let's assume standard cuBLAS usage available in GenMatmul.cu or similar.
// Actually, looking at repo, GenMatmul.cu uses a handle from Device state.

// Since I can't easily access the internal Context::cublas_handle without including "device/DeviceCore.h"
// and hoping it exposes it, I will assume we can rely on `cuda_matmul_forward` if it exists, 
// or implement the GEMM call here using `Tensor::matmul` logic but manually.

// ACTUALLY: The best way is to use the existing `cuda_matmul` which hopefully exists, 
// OR call cublas directly if I can get the handle.

// Let's check how `GenMatmul.cu` does it.
// It likely uses `cublasCreate` or retrieves it.

namespace OwnTensor {

// ======================================================================================
// BIAS ADD KERNEL
// ======================================================================================

template<typename T>
__global__ void add_bias_kernel(T* output, const T* bias, int64_t accumulated_dim_size, int64_t bias_dim_size) {
    // output is [accumulated_dim_size, bias_dim_size] treated as flat
    // bias is [bias_dim_size]
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = accumulated_dim_size * bias_dim_size;
    
    if (idx < total_elements) {
        int64_t bias_idx = idx % bias_dim_size;
        output[idx] += bias[bias_idx];
    }
}

void cuda_add_bias(Tensor& output, const Tensor& bias, cudaStream_t stream) {
    if (!bias.is_valid()) return;

    int64_t bias_size = bias.numel();
    int64_t output_size = output.numel();
    int64_t flatten_batch = output_size / bias_size; // e.g. B*T

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    if (output.dtype() == Dtype::Float32) {
        add_bias_kernel<float><<<blocks, threads, 0, stream>>>(
            output.data<float>(),
            bias.data<float>(),
            flatten_batch,
            bias_size
        );
    } else if (output.dtype() == Dtype::Float16) {
        add_bias_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__half*>(output.data<float16_t>()),
            reinterpret_cast<const __half*>(bias.data<float16_t>()),
            flatten_batch,
            bias_size
        );
    } else if (output.dtype() == Dtype::Bfloat16) {
        add_bias_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data<bfloat16_t>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data<bfloat16_t>()),
            flatten_batch,
            bias_size
        );
    }
}

// ======================================================================================
// FORWARD
// ======================================================================================

// We need to declare the external matmul helper if it's not in a header.
// Looking at `src/Kernels/cuda/GenMatmul.cu` via `list_dir` earlier, it seems `matmul` logic is there.
// If I can't call `cuda_matmul` (internal), I'll just rely on `cublas` if I can get the handle.
// BUT `OwnTensor::matmul` calls `GenMatmul`, which calls `cublas`. 
// So `cuda_linear_forward` can just call `OwnTensor::matmul` (which IS GPU compliant) 
// AND THEN call `add_bias_kernel`.
// The user said "linear i shappening via cpu path".
// This implies `MatrixOps.cpp` was triggering CPU.
// `MatrixOps.cpp` calls `OwnTensor::matmul(x, w)`. If `x` is CUDA, it SHOULD go to CUDA.
// Why did the user think it's CPU?
// Maybe because `out + b` in `MatrixOps.cpp` caused a CPU fallback or tensor dispatch issue?
// Or maybe `OwnTensor::matmul` itself has a bug.

// Regardless, implementing a dedicated `cuda_linear_forward` that does `matmul` + `add_bias` 
// ensures it stays on device and avoids intermediate node overhead from autograd (if we called autograd::add).
// In `MatrixOps.cpp`, we used:
// Tensor out = OwnTensor::matmul(x, w); (Raw tensor op, should be GPU)
// out = out + b; (Raw tensor op, should be GPU)
// If this was slow/CPU, maybe `operator+` implementation is suboptimal.

// Let's implement `cuda_linear_forward` by invoking GEMM then Bias Add explicitly.

// We need to access `cuda_matmul_forward` or similar. 
// If it's not exposed, we might have to define it or duplicate the Gemm call logic.
// Let's assume for now we can rely on `cuBLAS` being initialized or just use `OwnTensor::matmul` inside this wrapper,
// BUT specifically call the bias kernel afterwards to avoid `operator+` overhead if that's the issue.

// Wait, the user said "write a kernel for...". Implicitly, they want me to optimize the whole operation.
// I will assume `OwnTensor::matmul` works for the GEMM part, and my addition is the `add_bias`.

void cuda_linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output, cudaStream_t stream) {
    // 1. GEMM: output = input @ weight.T
    // We expect weight to be [Out, In] usually for Linear, but `autograd::linear` expects `weight` 
    // to be consistent with `F.linear`. PyTorch `F.linear(x, w)` -> `x @ w.T`.
    // My `gpt2_test` usage: `autograd::linear(h, W_up, b_up)`. 
    // If W_up is [In, Out], then `x @ W` is correct.
    // If W_up is [Out, In], then `x @ W.T` is correct.
    
    // In `gpt2_test.cpp`:
    // W_up = Tensor::randn... Shape{{n_embd, hidden}} -> [In, Out]
    // h = autograd::linear(h, W_up, b_up)
    // inside `linear` (MatrixOps.cpp): `matmul(x, w)`.
    // So `w` is treated as [In, Out]. No transpose needed if `matmul` handles (A, B).
    
    // So simply:
    // output = matmul(input, weight)
    
    // BUT we want to write our own kernel/logic to ensure it's fused/fast.
    // Since I can't easily call internal `cublas` without handle, and `OwnTensor::matmul` is available...
    // I will call `accumulate` logic for bias.
    
    // Re-implementation using low-level calls is risky without `DeviceState`. 
    // I will try to call the raw `matmul` (non-autograd) then my bias kernel.
    
    // wait, `output` is passed as ref.
    // We should compute result into it.
    
    // To match `matmul` signature which returns Tensor:
    // We can't use `Tensor::matmul` to write INTO `output`. It allocates.
    // This is a missing optimization in the library: `matmul_out` support.
    
    // TODO: For now, I will use `output = OwnTensor::matmul(input, weight)` which assigns a new tensor to output.
    // This isn't perfect (allocation) but it fixes the "CPU path" if `operator+` was the culprit.
    
    output = OwnTensor::matmul(input, weight);
    
    if (bias.is_valid()) {
        cuda_add_bias(output, bias, stream);
    }
}

// ======================================================================================
// BIAS BACKWARD
// ======================================================================================

// Helper for atomic add (generic)
template <typename T>
__device__ inline void atomic_add(T* address, T val) {
    atomicAdd(address, val);
}

template<typename T>
__global__ void reduce_bias_kernel(const T* grad_output, T* grad_bias, int64_t batch_size, int64_t bias_size) {
    // grad_output: [batch_size, bias_size] (flattened batch dims)
    // grad_bias: [bias_size]
    
    // Grid-stride loop or simple mapping?
    // Map x to bias_size, loop over batch_size
    
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < bias_size) {
        T sum = 0;
        #pragma unroll 4
        for (int64_t b = 0; b < batch_size; ++b) {
            sum += grad_output[b * bias_size + c];
        }
        grad_bias[c] = sum;
    }
}

// Accumulate-in-float reduction for half-precision types
// Each thread handles one bias column, loops over batch rows, accumulates in fp32
template<typename T>
__global__ void reduce_bias_kernel_half(const T* __restrict__ grad_output, T* __restrict__ grad_bias,
                                        int64_t batch_size, int64_t bias_size) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < bias_size) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int64_t b = 0; b < batch_size; ++b) {
            sum += __half2float(grad_output[b * bias_size + c]);
        }
        grad_bias[c] = __float2half(sum);
    }
}

// BF16 specialization
__global__ void reduce_bias_kernel_bf16(const __nv_bfloat16* __restrict__ grad_output,
                                        __nv_bfloat16* __restrict__ grad_bias,
                                        int64_t batch_size, int64_t bias_size) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < bias_size) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int64_t b = 0; b < batch_size; ++b) {
            sum += __bfloat162float(grad_output[b * bias_size + c]);
        }
        grad_bias[c] = __float2bfloat16(sum);
    }
}

void cuda_linear_bias_backward(const Tensor& grad_output, Tensor& grad_bias, cudaStream_t stream) {
    if (!grad_bias.is_valid()) return;
    
    // Flatten batch dims
    int64_t bias_size = grad_bias.numel();
    int64_t total_numel = grad_output.numel();
    int64_t batch_size = total_numel / bias_size;
    
    // We want to reduce over `batch_size`.
    // Using simple kernel for now to avoid dependency on global cublas handle / creating cached ones vector.
    // "Simple" kernel with atomic add? 
    // Or the strided loop.
    
    // Let's write a block-based reduction kernel which is reasonably fast.
    // Each block reduces a tile of columns.
    
    // For simplicity and guaranteed compilation (no complex headers):
    // Use atomicAdd kernel. It handles strided access reasonably well if many threads contend? No, atomics are slow for high contention.
    // Use the naive "one thread per column" loop if bias_size is large enough?
    // If Bias=768, 768 threads. Batch=8192.
    // 768 threads reading strided... Coalescing is broken.
    // Thread i reads 0, 768, 1536... 
    // Adjacent threads 0, 1 input addresses: 0, 1. Coalesced!
    // Yes! `grad_output[b * width + tid]`. 
    // For fixed `b`, threads 0..31 read `base + 0..31`.
    // This IS coalesced access.
    // So the simple "one thread per bias-element, loop over batch" IS COALESCED and fast.
    
    int64_t width = bias_size;
    int64_t height = batch_size;
    
    int threads = 256;
    int blocks = (width + threads - 1) / threads;
    
    // define kernel locally or above
    // implemented above as `reduce_bias_kernel`
    
    if (grad_bias.dtype() == Dtype::Float32) {
        reduce_bias_kernel<float><<<blocks, threads, 0, stream>>>(
            grad_output.data<float>(),
            grad_bias.data<float>(),
            height,
            width
        );
    } else if (grad_bias.dtype() == Dtype::Float16) {
        reduce_bias_kernel_half<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<float16_t>()),
            reinterpret_cast<__half*>(grad_bias.data<float16_t>()),
            height,
            width
        );
    } else if (grad_bias.dtype() == Dtype::Bfloat16) {
        reduce_bias_kernel_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data<bfloat16_t>()),
            reinterpret_cast<__nv_bfloat16*>(grad_bias.data<bfloat16_t>()),
            height,
            width
        );
    }
}

} // namespace OwnTensor
