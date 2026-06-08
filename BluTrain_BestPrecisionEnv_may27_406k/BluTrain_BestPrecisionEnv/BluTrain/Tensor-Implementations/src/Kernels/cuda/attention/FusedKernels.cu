#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include "TensorLib.h"
#include "ops/FusedKernels.cuh"
#include "ops/helpers/FusedKernels.h"
#include "autograd/backward/FusedTrilSoftmaxBackward.h"
#include "autograd/ops_template.h"
#include "dtype/Types.h"
#include "dtype/CudaTraits.h"

namespace OwnTensor{

  // ---- Type conversion helpers for templated kernels ----
  template<typename T> __device__ __forceinline__ float to_float(T val);
  template<> __device__ __forceinline__ float to_float(float val) { return val; }
  template<> __device__ __forceinline__ float to_float(__half val) { return __half2float(val); }
  template<> __device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }

  template<typename T> __device__ __forceinline__ T from_float(float val);
  template<> __device__ __forceinline__ float from_float(float val) { return val; }
  template<> __device__ __forceinline__ __half from_float(float val) { return __float2half(val); }
  template<> __device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }

  __inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
  }

  __inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
  }

  __device__ __forceinline__ float fast_rcp(float x) {
    float res;
    asm("rcp.approx.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
  }

  // Forward kernel ──────────────────────────────────────────────────────
  // Fused tril mask + softmax in a single kernel.
  //
  // All intermediate work is done in shared memory:
  //   Pass 1: read input ---> apply tril mask ---> store in smem ---> find row max
  //   Pass 2: exp(smem[i] - max) ---> accumulate sum (still in smem)
  //   Pass 3: normalize and write final result to global memory
  //
  // Total global memory traffic: 1 read (input) + 1 write (output).
template <typename T>
  __global__ void fused_tril_softmax_kernel(
      const T* __restrict__ d_input,
      T* __restrict__ d_output,
      int64_t trilDiag, float mask_value,
      int64_t H, int64_t W,
      int64_t rows, int64_t cols)
  {
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;
    const int row  = blockIdx.x;
    if (row >= rows) return;

    const T* row_input  = d_input  + row * cols;
    T*       row_output = d_output + row * cols;
    const int64_t row_idx   = row % H;

    extern __shared__ float smem[];   // cols floats (always fp32 for compute)

    // ── Pass 1: tril mask → smem + find row max ─────────────────
    float max_val = -INFINITY;
    #pragma unroll 4
    for (int i = tid; i < cols; i += bdim) {
        float v = (static_cast<int64_t>(i) > row_idx + trilDiag)
                  ? mask_value
                  : static_cast<float>(row_input[i]);
        smem[i] = v;
        max_val = fmaxf(max_val, v);
    }

    // Block-wide max reduction
    max_val = warpReduceMax(max_val);
    __shared__ float s_max;
    __shared__ float warp_vals[32];
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;

    if (laneId == 0) warp_vals[warpId] = max_val;
    __syncthreads();
    if (tid == 0) {
        float bmax = -INFINITY;
        int nw = (bdim + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bmax = fmaxf(bmax, warp_vals[w]);
        s_max = bmax;
    }
    __syncthreads();
    max_val = s_max;

    // ── Pass 2: exp(x - max) + accumulate sum (from smem) ───────
    float sum_exp = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < cols; i += bdim) {
        float e = expf(smem[i] - max_val);
        smem[i] = e;
        sum_exp += e;
    }

    // Block-wide sum reduction
    sum_exp = warpReduceSum(sum_exp);
    __shared__ float s_sum;
    if (laneId == 0) warp_vals[warpId] = sum_exp;
    __syncthreads();
    if (tid == 0) {
        float bsum = 0.0f;
        int nw = (bdim + warpSize - 1) / warpSize;
        for (int w = 0; w < nw; ++w) bsum += warp_vals[w];
        s_sum = bsum;
    }
    __syncthreads();

    // ── Pass 3: normalize (from smem) → write to global ─────────
    float inv_sum = fast_rcp(s_sum);
    #pragma unroll 4
    for (int i = tid; i < cols; i += bdim) {
        row_output[i] = static_cast<T>(smem[i] * inv_sum);
    }
  }
  // ── Backward kernel ──────────────────────────────────────────────────────
  // Softmax Jacobian: grad_input[i] = out[i] * (grad_output[i] - dot)
  // where dot = sum_j(grad_output[j] * out[j]).
  // Masked positions have out[i] = 0, so they get zero gradient automatically.
  //
  // Optimization: both output and grad_output are cached in shared memory
  // during the dot-product pass so the second pass reads entirely from smem
  // (eliminates redundant global memory traffic).
  template<typename T>
  __global__ void fused_tril_softmax_backward_kernel(
      const T* __restrict__ grad_output,
      const T* __restrict__ output,
      T* __restrict__ grad_input,
      int64_t rows,
      int64_t cols
  ) {
      const int row = blockIdx.x;
      if (row >= rows) return;

      const int tid  = threadIdx.x;
      const int bdim = blockDim.x;

      const T* row_grad = grad_output + row * cols;
      const T* row_out  = output      + row * cols;
      T*       row_gin  = grad_input  + row * cols;

      // Shared memory always in float for reduction accuracy
      extern __shared__ float smem[];
      float* s_out  = smem;
      float* s_grad = smem + cols;

      // Pass 1: load as T, convert to float for shared mem + dot
      float dot = 0.0f;
      for (int i = tid; i < cols; i += bdim) {
          float o = to_float(row_out[i]);
          float g = to_float(row_grad[i]);
          s_out[i]  = o;
          s_grad[i] = g;
          dot = fmaf(o, g, dot);
      }

      // Block-wide reduction of dot
      dot = warpReduceSum(dot);

      __shared__ float s_dot;
      __shared__ float warp_sums[32];
      const int warpId = tid / warpSize;
      const int laneId = tid % warpSize;

      if (laneId == 0) warp_sums[warpId] = dot;
      __syncthreads();

      if (tid == 0) {
          float acc = 0.0f;
          const int nw = (bdim + warpSize - 1) / warpSize;
          for (int w = 0; w < nw; ++w) acc += warp_sums[w];
          s_dot = acc;
      }
      __syncthreads();
      dot = s_dot;

      // Pass 2: compute in float, store as T
      for (int i = tid; i < cols; i += bdim)
          row_gin[i] = from_float<T>(s_out[i] * (s_grad[i] - dot));
  }

  namespace cuda {

  template<typename T>
  static void launch_fused_tril_softmax_backward(const T* grad_output, const T* output, T* grad_input, int64_t rows, int64_t cols) {
      int threads = (cols <= 1024) ? 256 : 1024;
      if (threads < 32) threads = 32;
      // Shared memory: 2 arrays of `cols` floats (always float for reduction)
      size_t smem = 2 * static_cast<size_t>(cols) * sizeof(float);
      fused_tril_softmax_backward_kernel<T><<<static_cast<int>(rows), threads, smem>>>(
          grad_output, output, grad_input, rows, cols);
  }

  void fused_tril_softmax_backward_cuda(const float* grad_output, const float* output, float* grad_input, int64_t rows, int64_t cols) {
      launch_fused_tril_softmax_backward<float>(grad_output, output, grad_input, rows, cols);
  }
  void fused_tril_softmax_backward_cuda(const float16_t* grad_output, const float16_t* output, float16_t* grad_input, int64_t rows, int64_t cols) {
      launch_fused_tril_softmax_backward<__half>(reinterpret_cast<const __half*>(grad_output), reinterpret_cast<const __half*>(output),
                                                  reinterpret_cast<__half*>(grad_input), rows, cols);
  }
  void fused_tril_softmax_backward_cuda(const bfloat16_t* grad_output, const bfloat16_t* output, bfloat16_t* grad_input, int64_t rows, int64_t cols) {
      launch_fused_tril_softmax_backward<__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(grad_output), reinterpret_cast<const __nv_bfloat16*>(output),
                                                         reinterpret_cast<__nv_bfloat16*>(grad_input), rows, cols);
  }

  } // namespace cuda

  Tensor fused_tril_softmax(Tensor& input, int64_t trilDiag, double value) {
    const auto& shape = input.shape();
    const int ndim = static_cast<int>(shape.dims.size());

    const int64_t W    = shape.dims[ndim - 1];
    const int64_t H    = shape.dims[ndim - 2];
    const int64_t cols = W;
    const int64_t rows = input.numel() / cols;

    Tensor output = Tensor::empty(shape, input.opts());

    const int block   = (cols <= 1024) ? 256 : 1024;
    // Shared memory: cols floats for intermediate fp32 computation
    const size_t shmem = static_cast<size_t>(cols) * sizeof(float);

    switch (input.dtype()) {
      case Dtype::Float32:
        fused_tril_softmax_kernel<float><<<static_cast<int>(rows), block, shmem>>>(
            input.data<float>(), output.data<float>(),
            trilDiag, static_cast<float>(value),
            H, W, rows, cols);
        break;
      case Dtype::Float16: {
        using CudaF16 = detail::CudaNativeType<float16_t>;
        fused_tril_softmax_kernel<CudaF16><<<static_cast<int>(rows), block, shmem>>>(
            reinterpret_cast<const CudaF16*>(input.data<float16_t>()),
            reinterpret_cast<CudaF16*>(output.data<float16_t>()),
            trilDiag, static_cast<float>(value),
            H, W, rows, cols);
        break;
      }
      case Dtype::Bfloat16: {
        using CudaBF16 = detail::CudaNativeType<bfloat16_t>;
        fused_tril_softmax_kernel<CudaBF16><<<static_cast<int>(rows), block, shmem>>>(
            reinterpret_cast<const CudaBF16*>(input.data<bfloat16_t>()),
            reinterpret_cast<CudaBF16*>(output.data<bfloat16_t>()),
            trilDiag, static_cast<float>(value),
            H, W, rows, cols);
        break;
      }
      default:
        throw std::runtime_error("fused_tril_softmax: unsupported dtype");
    }

    if (autograd::GradMode::is_enabled() && input.requires_grad()) {
        auto grad_fn = std::make_shared<autograd::FusedTrilSoftmaxBackward>(
            output.detach(), trilDiag, value);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(input));
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
  }
} //End of OwnTensor