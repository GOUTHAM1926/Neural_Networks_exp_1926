#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "TensorLib.h"
#include <curand.h>
#include <curand_kernel.h>
#include "ops/TensorOps.cuh"
#include "dtype/CudaTraits.h"
#include "checkpointing/RNG.h"

namespace OwnTensor {

template<typename T>
__global__ void dropOut_kernel(const T* input, T* output, T* mask, int N, float p, float scale,
                               unsigned long long seed, unsigned long long offset) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) {
        curandStatePhilox4_32_10_t state;
        // seed  = captured from RNG::gpu_seed_ at launch time (set via RNG::set_seed())
        // idx   = per-thread sequence number — gives each thread an independent Philox stream
        // offset = captured from RNG::gpu_offset_ and advanced after each launch, ensuring
        //          consecutive dropout calls produce different masks even with the same seed,
        //          and that checkpoint recomputation reproduces the exact same mask.
        curand_init(seed, idx, offset, &state);

        float rand_value = curand_uniform(&state);

        if(p >= 1.0f || rand_value < p) {
            mask[idx] = static_cast<T>(0);
            output[idx] = static_cast<T>(0);
        } else {
            mask[idx] = static_cast<T>(1);
            float val = static_cast<float>(input[idx]) * scale;
            output[idx] = static_cast<T>(val);
        }
    }
}

    DropoutCudaResult dropOut_cuda(const Tensor& data, float probability) {
        int N = data.numel();

        Tensor output(data.shape(), TensorOptions().with_dtype(data.dtype()).with_device(data.device()));
        Tensor mask(data.shape(), TensorOptions().with_dtype(data.dtype()).with_device(data.device()));

        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        // Read seed and offset from the authoritative RNG state.
        // get_gpu_offset_and_advance(1) returns the current offset then increments by 1
        // (one Philox 128-bit block per call), so consecutive dropout calls on the same
        // set of positions produce different masks, and checkpoint recomputation using
        // the same (seed, offset) pair reproduces the exact same mask.
        unsigned long long seed   = RNG::get_gpu_seed();
        unsigned long long offset = RNG::get_gpu_offset_and_advance(1);
        float scale = 1.0f / (1.0f - probability);

        switch (data.dtype()) {
          case Dtype::Float32:
            OwnTensor::dropOut_kernel<<<gridSize, blockSize>>>(
                data.data<float>(), output.data<float>(), mask.data<float>(),
                N, probability, scale, seed, offset);
            break;
          case Dtype::Float16: {
            using CudaF16 = detail::CudaNativeType<float16_t>;
            OwnTensor::dropOut_kernel<<<gridSize, blockSize>>>(
                reinterpret_cast<const CudaF16*>(data.data<float16_t>()),
                reinterpret_cast<CudaF16*>(output.data<float16_t>()),
                reinterpret_cast<CudaF16*>(mask.data<float16_t>()),
                N, probability, scale, seed, offset);
            break;
          }
          case Dtype::Bfloat16: {
            using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
            OwnTensor::dropOut_kernel<<<gridSize, blockSize>>>(
                reinterpret_cast<const CudaBF16*>(data.data<bfloat16_t>()),
                reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
                reinterpret_cast<CudaBF16*>(mask.data<bfloat16_t>()),
                N, probability, scale, seed, offset);
            break;
          }
          default:
            throw std::runtime_error("Dropout CUDA: unsupported dtype");
        }

        return {output, mask};
    }
}
