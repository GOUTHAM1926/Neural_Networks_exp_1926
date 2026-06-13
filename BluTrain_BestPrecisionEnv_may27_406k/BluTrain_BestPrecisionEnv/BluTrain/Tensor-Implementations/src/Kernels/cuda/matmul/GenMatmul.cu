// #ifdef WITH_MYBLAS

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <stdio.h>
// #include <stdexcept>
// #include <memory>
// #include <mutex>

// #include "mycublas.h"
// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "utils/Profiler.h"
// namespace OwnTensor {

// //
// ============================================================================
// // DISPATCH LAYER - PURE MYBLAS
// //
// ============================================================================

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output,
// cudaStream_t stream)
// {
//     AUTO_PROFILE_CUDA("Forward::Matmul_CUDA_MyBlas");

//     // Ensure contiguous memory layout so stride computation (M*K, K*N) is
//     correct.
//     // Non-contiguous tensors (e.g. from transpose) have different actual
//     batch strides. const Tensor A_c = A.is_contiguous() ? A : A.contiguous();
//     const Tensor B_c = B.is_contiguous() ? B : B.contiguous();

//     const auto& a_shape = A_c.shape().dims;
//     const auto& b_shape = B_c.shape().dims;
//     const auto& out_shape = output.shape().dims;

//     int M = static_cast<int>(a_shape[a_shape.size() - 2]);
//     int K = static_cast<int>(a_shape[a_shape.size() - 1]);
//     int N = static_cast<int>(b_shape[b_shape.size() - 1]);

//     int batch_count = 1;
//     for (size_t i = 0; i < out_shape.size() - 2; ++i) batch_count *=
//     static_cast<int>(out_shape[i]);

//     long long int strideA = (long long int)M * K;
//     long long int strideB = (long long int)K * N;
//     long long int strideC = (long long int)M * N;

//     // Handle broadcasting (Stride 0 if numel fits exactly one matrix)
//     if (A_c.numel() == (size_t)M * K) strideA = 0;
//     if (B_c.numel() == (size_t)K * N) strideB = 0;

//     dispatch_by_dtype(A_c.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
//         const T* a_ptr = A_c.data<T>();
//         const T* b_ptr = B_c.data<T>();
//         T* out_ptr = output.data<T>();

//         mycublasHandle_t handle;
//         mycublasCreate(&handle);
//         mycublasSetStream(handle, stream);

//         if constexpr (std::is_same<T, float>::value) {
//             mycublasSgemmStridedBatched(handle, MYCUBLAS_OP_N, MYCUBLAS_OP_N,
//             M, N, K, 1.0f, a_ptr, K, strideA, b_ptr, N, strideB, 0.0f,
//             out_ptr, N, strideC, batch_count);
//         }
//         else if constexpr (std::is_same<T, __half>::value || std::is_same<T,
//         OwnTensor::float16_t>::value) {
//             __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
//             mycublasHgemmStridedBatched(handle, MYCUBLAS_OP_N, MYCUBLAS_OP_N,
//             M, N, K, alpha, (const __half*)a_ptr, K, strideA, (const
//             __half*)b_ptr, N, strideB, beta, (__half*)out_ptr, N, strideC,
//             batch_count);
//         }
//         else if constexpr (std::is_same<T, __nv_bfloat16>::value ||
//         std::is_same<T, OwnTensor::bfloat16_t>::value) {
//             __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta =
//             __float2bfloat16(0.0f); mycublasBgemmStridedBatched(handle,
//             MYCUBLAS_OP_N, MYCUBLAS_OP_N, M, N, K, alpha, (const
//             __nv_bfloat16*)a_ptr, K, strideA, (const __nv_bfloat16*)b_ptr, N,
//             strideB, beta, (__nv_bfloat16*)out_ptr, N, strideC, batch_count);
//         }
//         else if constexpr (std::is_same<T, double>::value) {
//             mycublasDgemmStridedBatched(handle, M, N, K, 1.0, (const
//             double*)a_ptr, K, strideA, (const double*)b_ptr, N, strideB, 0.0,
//             (double*)out_ptr, N, strideC, batch_count);
//         }
//         else {
//              throw std::runtime_error("cuda_matmul: Unsupported dtype for
//              MyBlas GEMM.");
//         }
//         mycublasDestroy(handle);
//     });

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA Error after MyBlas matmul launch: " +
//         std::string(cudaGetErrorString(err)));
//     }
// }

// //
// ============================================================================
// // ADDMM IMPLEMENTATION - USING MYBLAS
// //
// ============================================================================

// // Helper kernel for broadcasting and scaling
// template<typename T>
// __global__ void broadcast_add_scaled_kernel(T* out, const T* in, const T*
// bias, float alpha, float beta, int M, int N, int batch_count, int64_t
// stride_c, int64_t bias_numel) {
//     int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     int64_t total = (int64_t)batch_count * M * N;
//     if (idx < total) {
//         T b_val;
//         if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T,
//         complex64_t> || std::is_same_v<T, complex128_t>) {
//             b_val = T(0.0f, 0.0f);
//         } else {
//             b_val = (T)0.0f;
//         }

//         if (bias_numel == 1) b_val = bias[0];
//         else if (bias_numel == N) b_val = bias[idx % N];
//         else if (bias_numel == total) b_val = bias[idx];

//         if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T,
//         complex64_t> || std::is_same_v<T, complex128_t>) {
//             // Complex scale: (a + bi) * beta = (a*beta) + (b*beta)i
//             // This assumes we have a way to scale complex types.
//             // If they don't support * float, we might need to access
//             real/imag.
//             // For now, let's assume T * float works or we just don't scale
//             if not. out[idx] = out[idx] + b_val * (T)beta;
//         } else {
//             out[idx] = out[idx] + (T)(beta * (float)b_val);
//         }
//     }
// }

// void cuda_addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
// float alpha, float beta, Tensor& output, cudaStream_t stream) {
//     AUTO_PROFILE_CUDA("Forward::Addmm_CUDA_MyBlas");

//     // Ensure contiguous so shape-derived strides (M*K, K*N) match actual
//     memory layout. const Tensor mat1_c = mat1.is_contiguous() ? mat1 :
//     mat1.contiguous(); const Tensor mat2_c = mat2.is_contiguous() ? mat2 :
//     mat2.contiguous();

//     if (output.dtype() == Dtype::Float32) {
//         int64_t M = output.shape().dims[output.shape().dims.size() - 2];
//         int64_t N = output.shape().dims[output.shape().dims.size() - 1];
//         int64_t K = mat1_c.shape().dims[mat1_c.shape().dims.size() - 1];
//         int batch_count = 1;
//         for (size_t i = 0; i < output.shape().dims.size() - 2; ++i)
//         batch_count *= (int)output.shape().dims[i];

//         mycublasHandle_t handle;
//         mycublasCreate(&handle);
//         mycublasSetStream(handle, stream);

//         int lda = (int)K;
//         int ldb = (int)N;
//         int ldc = (int)N;
//         long long strideA = (long long)M * K;
//         long long strideB = (long long)K * N;
//         long long strideC = (long long)M * N;

//         if (mat1_c.numel() == (size_t)M * K) strideA = 0;
//         if (mat2_c.numel() == (size_t)K * N) strideB = 0;

//         mycublasSgemmAddmm(handle, (int)M, (int)N, (int)K, alpha,
//                               mat1_c.data<float>(), lda, strideA,
//                               mat2_c.data<float>(), ldb, strideB,
//                               beta, input.data<float>(),
//                               (int64_t)input.numel(), output.data<float>(),
//                               ldc, strideC, batch_count);

//         mycublasDestroy(handle);
//         return;
//     }

//     // Fallback for other dtypes
//     cuda_matmul(mat1_c, mat2_c, output, stream);

//     const auto& out_shape = output.shape().dims;
//     int M = out_shape[out_shape.size() - 2];
//     int N = out_shape[out_shape.size() - 1];
//     int batch_count = 1;
//     for (size_t i = 0; i < out_shape.size() - 2; ++i) batch_count *=
//     static_cast<int>(out_shape[i]);

//     int64_t total = output.numel();
//     int threads = 256;
//     int blocks = (total + threads - 1) / threads;

//     dispatch_by_dtype(output.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
//         broadcast_add_scaled_kernel<T><<<blocks, threads, 0, stream>>>(
//             output.data<T>(), nullptr, input.data<T>(), alpha, beta, M, N,
//             batch_count, (int64_t)M*N, input.numel()
//         );
//     });
// }

// } // namespace OwnTensor
// // #endif
// #else

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include "utils/Profiler.h"
// #include <mma.h>
// #include <cublas_v2.h>
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
// #include <string>
// #include <mutex>

// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "ops/Kernels.h"

// namespace OwnTensor {

// // cuBLAS handle management - thread-safe singleton per device
// static std::mutex cublas_mutex;
// static cublasHandle_t g_cublas_handles[8] = {nullptr};

// static cublasHandle_t get_cublas_handle(int device = 0) {
//    if (g_cublas_handles[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex);
//       if (g_cublas_handles[device] == nullptr) {
//          cudaSetDevice(device);
//          cublasCreate(&g_cublas_handles[device]);
//          // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant
//          speedup cublasSetMathMode(g_cublas_handles[device],
//          CUBLAS_TF32_TENSOR_OP_MATH);
//       }
//    }
//    return g_cublas_handles[device];
// }

// using namespace nvcuda;

// //
// ============================================================================
// // METADATA & CONSTANTS
// //
// ============================================================================

// struct MatmulMetadata {
//    int a_shape[8], b_shape[8], out_shape[8];
//    int a_strides[8], b_strides[8], out_strides[8];
//    int a_ndim, b_ndim, out_ndim;
// };

// __device__ void compute_batch_offset(int batch_idx, const int* shape, const
// int* strides, int ndim, const int* out_shape, int out_ndim, int& offset) {
//    offset = 0; if (out_ndim <= 2) return;
//    int temp_batch = batch_idx;
//    for (int dim = out_ndim - 3; dim >= 0; --dim) {
//       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
//       temp_batch /= b_dim_sz;
//       int c_dim = dim - (out_ndim - ndim);
//       if (c_dim >= 0 && c_dim < ndim - 2) offset += (int64_t)((shape[c_dim] >
//       1) ? b_coord : 0) * strides[c_dim];
//    }
// }

// constexpr int PAD = 8;

// //
// ============================================================================
// // ADDMM HELPERS
// //
// ============================================================================

// template<typename T>
// __global__ void broadcast_scale_kernel(T* __restrict__ output, const T*
// __restrict__ input, float scale,
//     int64_t total_elements, int64_t input_size, int64_t rows, int64_t cols) {
//     const int64_t tid     = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     const int64_t gstride = gridDim.x  * (int64_t)blockDim.x;

//     if constexpr (std::is_same_v<T, float>) {
//         constexpr int VEC = 4;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             float4* __restrict__       out4 =
//             reinterpret_cast<float4*>(output); const float4* __restrict__ in4
//             = reinterpret_cast<const float4*>(input); const int64_t n4 =
//             total_elements / VEC; const float sv = (input_size == 1) ?
//             (float)input[0] * scale : 0.0f; for (int64_t i = tid; i < n4; i
//             += gstride) {
//                 float4 v;
//                 if (input_size == 1) {
//                     v = make_float4(sv, sv, sv, sv);
//                 } else if (input_size == cols) {
//                     const float4 iv = in4[((i * VEC) % cols) / VEC];
//                     v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
//                     iv.w * scale);
//                 } else {
//                     const float4 iv = in4[i];
//                     v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
//                     iv.w * scale);
//                 }
//                 out4[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, double>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             double2* __restrict__       out2 =
//             reinterpret_cast<double2*>(output); const double2* __restrict__
//             in2  = reinterpret_cast<const double2*>(input); const int64_t n2
//             = total_elements / VEC; const double sv = (input_size == 1) ?
//             (double)input[0] * (double)scale : 0.0; for (int64_t i = tid; i <
//             n2; i += gstride) {
//                 double2 v;
//                 if (input_size == 1) {
//                     v = make_double2(sv, sv);
//                 } else if (input_size == cols) {
//                     const double2 iv = in2[((i * VEC) % cols) / VEC];
//                     v = make_double2(iv.x * scale, iv.y * scale);
//                 } else {
//                     const double2 iv = in2[i];
//                     v = make_double2(iv.x * scale, iv.y * scale);
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((double)input[0] * (double)scale); else if (input_size ==
//                 cols)           output[i] = (T)((double)input[i % cols] *
//                 (double)scale); else if (input_size == total_elements)
//                 output[i] = (T)((double)input[i] * (double)scale); else
//                 output[i] = (T)0.0;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, __half>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             half2* __restrict__       out2 =
//             reinterpret_cast<half2*>(output); const half2* __restrict__ in2
//             = reinterpret_cast<const half2*>(input); const int64_t n2     =
//             total_elements / VEC; const half2   scale2 =
//             __float2half2_rn(scale); const half2   sv2    = (input_size == 1)
//             ? __float2half2_rn((float)input[0] * scale)
//                                                      :
//                                                      __float2half2_rn(0.0f);
//             for (int64_t i = tid; i < n2; i += gstride) {
//                 half2 v;
//                 if (input_size == 1) {
//                     v = sv2;
//                 } else if (input_size == cols) {
//                     v = __hmul2(in2[((i * VEC) % cols) / VEC], scale2);
//                 } else {
//                     v = __hmul2(in2[i], scale2);
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             __nv_bfloat162* __restrict__       out2 =
//             reinterpret_cast<__nv_bfloat162*>(output); const __nv_bfloat162*
//             __restrict__ in2  = reinterpret_cast<const
//             __nv_bfloat162*>(input); const int64_t        n2  =
//             total_elements / VEC; const __nv_bfloat162 sv2 = (input_size ==
//             1)
//                                        ? __float2bfloat162_rn((float)input[0]
//                                        * scale) : __float2bfloat162_rn(0.0f);
//             for (int64_t i = tid; i < n2; i += gstride) {
//                 __nv_bfloat162 v;
//                 if (input_size == 1) {
//                     v = sv2;
//                 } else if (input_size == cols) {
//                     const float2 f = __bfloat1622float2(in2[((i * VEC) %
//                     cols) / VEC]); v = __float22bfloat162_rn(make_float2(f.x
//                     * scale, f.y * scale));
//                 } else {
//                     const float2 f = __bfloat1622float2(in2[i]);
//                     v = __float22bfloat162_rn(make_float2(f.x * scale, f.y *
//                     scale));
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_integral_v<T>) {
//         for (int64_t i = tid; i < total_elements; i += gstride) {
//             if      (input_size == 1)             output[i] =
//             (T)((float)input[0] * scale); else if (input_size == cols)
//             output[i] = (T)((float)input[i % cols] * scale); else if
//             (input_size == total_elements) output[i] = (T)((float)input[i] *
//             scale); else                                   output[i] = (T)0;
//         }
//     } else {
//         for (int64_t i = tid; i < total_elements; i += gstride) {
//             if (input_size == total_elements) output[i] = input[i];
//             else                              output[i] = (T)0.0f;
//         }
//     }
// }

// template<typename T>
// void launch_broadcast_scale(T* output, const T* input, float scale, int64_t
// rows, int64_t cols, int64_t input_size, cudaStream_t stream) {
//     int64_t total = rows * cols;
//     // Mirror the vectorization decision the kernel will make so blocks are
//     right-sized constexpr int VEC = std::is_same_v<T, float> ? 4 :
//                        (std::is_same_v<T, double> || std::is_same_v<T,
//                        __half> ||
//                         std::is_same_v<T, __nv_bfloat16>) ? 2 : 1;
//     const bool will_vec = (VEC > 1) && (total % VEC == 0) &&
//                           (input_size == 1 || input_size == total ||
//                            (input_size == cols && cols % VEC == 0));
//     int64_t eff    = will_vec ? total / VEC : total;
//     int     threads = 256;
//     int     blocks  = (int)((eff + threads - 1) / threads);
//     broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(output, input,
//     scale, total, input_size, rows, cols);
// }

// template<typename T>
// __global__ void add_scaled_kernel_typed(T* out, const T* m, float a, int64_t
// n) {
//     int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     if (i < n) {
//         if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>
//         || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
//             out[i] += (T)(a * (float)m[i]);
//         } else {
//             // Fallback for types that might not support float multiplication
//             directly or need different logic
//             // For complex types, we'd need a proper implementation, but for
//             now we just skip or do basic add out[i] += (T)m[i];
//         }
//     }
// }

// //
// ============================================================================
// // FP16 WMMA KERNEL (17 TFLOPS Version)
// // BM=128, BN=128, BK=32, Threads=512
// //
// ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_fp16_optimized(const __half* __restrict__ A, const
// __half* __restrict__ B, __half* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
//    warp_col = warp_id % 4; int ao, bo, co; compute_batch_offset(batch_idx,
//    meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim,
//    ao); compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides,
//    meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
//    compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const __half *Ap = A +
//    ao, *Bp = B + bo; __half *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1]; int
//    s_bk = meta.b_strides[meta.b_ndim-2], s_bn =
//    meta.b_strides[meta.b_ndim-1]; int s_cm =
//    meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ __half As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
//    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
//    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
//    #pragma unroll
//    for(int i=0; i<2; i++) for(int j=0; j<2; j++)
//    wmma::fill_fragment(acc[i][j], __float2half(0.0f));

//    auto load_tiles = [&](int ko, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ?
//          Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2half(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk +
//          (blockIdx.x*BN+c)*s_bn] : __float2half(0.0f);
//       }
//    };
//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int k = 0; k < K; k += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (k + BK < K) load_tiles(k + BK, wi);
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks],
//             BK+PAD); #pragma unroll for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16],
//                BN+PAD); wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
//    __half* sm = reinterpret_cast<__half*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
//       int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN +
//       warp_col*WN + j*16; if (cr < M && cc < N) {
//          __half* wsm = sm + warp_id * 256;
//          wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//          for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
//             if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] =
//             wsm[r*16+c];
//       }
//    }
// }

// //
// ============================================================================
// // BF16 WMMA KERNEL
// //
// ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_bf16_optimized(const __nv_bfloat16* __restrict__ A,
// const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, int M,
// int N, int K, int total_batches, MatmulMetadata meta) {
//    const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
//    warp_col = warp_id % 4; int ao, bo, co; compute_batch_offset(batch_idx,
//    meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim,
//    ao); compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides,
//    meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
//    compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const __nv_bfloat16 *Ap
//    = A + ao, *Bp = B + bo; __nv_bfloat16 *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1]; int
//    s_bk = meta.b_strides[meta.b_ndim-2], s_bn =
//    meta.b_strides[meta.b_ndim-1]; int s_cm =
//    meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ __nv_bfloat16 As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
//    af; wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
//    wmma::row_major> bf; wmma::fragment<wmma::accumulator, 16, 16, 16, float>
//    acc[2][2]; #pragma unroll for(int i=0; i<2; i++) for(int j=0; j<2; j++)
//    wmma::fill_fragment(acc[i][j], 0.0f);

//    auto load_tiles = [&](int ko, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ?
//          Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2bfloat16(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk +
//          (blockIdx.x*BN+c)*s_bn] : __float2bfloat16(0.0f);
//       }
//    };
//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int k = 0; k < K; k += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (k + BK < K) load_tiles(k + BK, wi);
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks],
//             BK+PAD); #pragma unroll for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16],
//                BN+PAD); wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
//    float* sm = reinterpret_cast<float*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
//       int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN +
//       warp_col*WN + j*16; if (cr < M && cc < N) {
//          float* wsm = sm + warp_id * 256;
//          wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//          for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
//             if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] =
//             __float2bfloat16(wsm[r*16+c]);
//       }
//    }
// }

// //
// ============================================================================
// // FP32 KERNEL (5 TFLOPS Version)
// // BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// //
// ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_fp32_optimized(const float* __restrict__ A, const
// float* __restrict__ B, float* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
//    int ao, bo, co;
//    compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
//    meta.out_shape, meta.out_ndim, ao); compute_batch_offset(b_idx,
//    meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim,
//    bo); compute_batch_offset(b_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const float *Ap = A +
//    ao, *Bp = B + bo; float *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk
//    = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1],
//    s_cm = meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ float As[2][BK][BM + PAD];
//    __shared__ float Bs[2][BK][BN + PAD];
//    float results[16] = {0.0f}, regM[4], regN[4];

//    auto load_tiles = [&](int ko, int idx) {
//       if (s_ak == 1 && s_bn == 1) { // Row-Major Contiguous
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / 16, c = li % 16;
//             As[idx][c][r] = (by*128+r < M && ko+c < K) ? Ap[(by*128+r)*s_am +
//             (ko+c)] : 0.0f;
//          }
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / 128, c = li % 128;
//             Bs[idx][r][c] = (ko+r < K && bx*128+c < N) ? Bp[(ko+r)*s_bk +
//             (bx*128+c)] : 0.0f;
//          }
//       } else { // Generic
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / BK, c = li % BK;
//             As[idx][c][r] = (by*BM+r < M && ko+c < K) ? Ap[(by*BM+r)*s_am +
//             (ko+c)*s_ak] : 0.0f;
//          }
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / BN, c = li % BN;
//             Bs[idx][r][c] = (ko+r < K && bx*BN+c < N) ? Bp[(ko+r)*s_bk +
//             (bx*BN+c)*s_bn] : 0.0f;
//          }
//       }
//    };

//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int bk = 0; bk < K; bk += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (bk + BK < K) load_tiles(bk + BK, wi);
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
//          results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
//    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
//       int r = by*BM + tRow*4 + i, c = bx*BN + tCol*4 + j;
//       if (r < M && c < N) Cp[r*s_cm + c*s_cn] = results[i*4+j];
//    }
// }

// //
// ============================================================================
// // FP64 KERNEL (64x64x8 Tile, 256 Threads, 4x4 Tiling)
// //
// ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_fp64_optimized(const double* __restrict__ A, const
// double* __restrict__ B, double* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tC = tid % (BN/TN), tR = tid / (BN/TN);
//    int ao, bo, co;
//    compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
//    meta.out_shape, meta.out_ndim, ao); compute_batch_offset(b_idx,
//    meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim,
//    bo); compute_batch_offset(b_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const double *Ap = A +
//    ao, *Bp = B + bo; double *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk
//    = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1],
//    s_cm = meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];
//    __shared__ double As[BM*BK], Bs[BK*BN];
//    double res[16] = {0.0}, rM[4], rN[4];
//    for (int bk = 0; bk < K; bk += BK) {
//       for (int i = tid; i < BM*BK; i += 256) { int r = i/BK, c = i%BK; As[i]
//       = (by*BM+r < M && bk+c < K) ? Ap[(by*BM+r)*s_am + (bk+c)*s_ak] : 0.0; }
//       for (int i = tid; i < BK*BN; i += 256) { int r = i/BN, c = i%BN; Bs[i]
//       = (bk+r < K && bx*BN+c < N) ? Bp[(bk+r)*s_bk + (bk+c)*s_bn] : 0.0; }
//       __syncthreads();
//       for (int d = 0; d < BK; d++) {
//          for (int i = 0; i < 4; i++) rM[i] = As[(tR*4+i)*BK+d];
//          for (int i = 0; i < 4; i++) rN[i] = Bs[d*BN+tC*4+i];
//          for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res[i*4+j]
//          += rM[i] * rN[j];
//       }
//       __syncthreads();
//    }
//    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { int r =
//    by*BM+tR*4+i, c = bx*BN+tC*4+j; if (r < M && c < N) Cp[r*s_cm + c*s_cn] =
//    res[i*4+j]; }
// }

// //
// ============================================================================
// // DISPATCH LAYER
// //
// ============================================================================

// template<typename T>
// void launch_optimized_matmul(const Tensor& A, const Tensor& B, Tensor&
// output, cudaStream_t stream) {
//    AUTO_PROFILE_CUDA("Forward::Matmul_CUDA");
//    const auto& ash = A.shape().dims, &bsh = B.shape().dims, &osh =
//    output.shape().dims; int an = ash.size(), bn = bsh.size(), on =
//    osh.size(), M = ash[an-2], K = ash[an-1], N = bsh[bn-1], tb = 1; for (int
//    i = 0; i < on - 2; i++) tb *= osh[i]; MatmulMetadata meta; meta.a_ndim =
//    an; meta.b_ndim = bn; meta.out_ndim = on; for (int i = 0; i < an; i++) {
//    meta.a_shape[i] = ash[i]; meta.a_strides[i] = A.stride().strides[i]; } for
//    (int i = 0; i < bn; i++) { meta.b_shape[i] = bsh[i]; meta.b_strides[i] =
//    B.stride().strides[i]; } for (int i = 0; i < on; i++) { meta.out_shape[i]
//    = osh[i]; meta.out_strides[i] = output.stride().strides[i]; } const T* ap
//    = A.data<T>(), *bp = B.data<T>(); T* op = output.data<T>();

//    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T,
//    __half>::value) {
//       // Try cuBLAS first for FP16: compute in FP16, accumulate in FP32
//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool out_contiguous = (meta.out_strides[on-1] == 1);
//       bool is_supported = (a_contiguous || (an >= 2 && meta.a_strides[an-2]
//       == 1)) &&
//                           (b_contiguous || (bn >= 2 && meta.b_strides[bn-2]
//                           == 1)) && out_contiguous;
//       bool cublas_ok = false;
//       if (is_supported) {
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);
//          float alpha_f = 1.0f, beta_f = 0.0f;
//          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//          int lda = a_contiguous ? meta.a_strides[an-2] :
//          meta.a_strides[an-1]; cublasOperation_t opB = b_contiguous ?
//          CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//          meta.b_strides[bn-2] : meta.b_strides[bn-1]; int ldc =
//          meta.out_strides[on-2]; const __half* ap_h = reinterpret_cast<const
//          __half*>(ap); const __half* bp_h = reinterpret_cast<const
//          __half*>(bp);
//          __half* op_h = reinterpret_cast<__half*>(op);
//          cublasStatus_t status;
//          if (tb == 1) {
//             status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_h, CUDA_R_16F, ldb, ap_h, CUDA_R_16F, lda,
//                &beta_f, op_h, CUDA_R_16F, ldc,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          } else {
//             long long stride_a_ll = (an >= 3) ? (long
//             long)meta.a_strides[an-3] : 0LL; long long stride_b_ll = (bn >=
//             3) ? (long long)meta.b_strides[bn-3] : 0LL; long long stride_c_ll
//             = (long long)meta.out_strides[on-3]; status =
//             cublasGemmStridedBatchedEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_h, CUDA_R_16F, ldb, stride_b_ll, ap_h, CUDA_R_16F, lda,
//                stride_a_ll, &beta_f, op_h, CUDA_R_16F, ldc, stride_c_ll, tb,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          }
//          cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
//       }
//       if (!cublas_ok) {
//          matmul_fp16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128,
//          (M+127)/128, tb), 512, 0, stream>>>(
//             reinterpret_cast<const __half*>(ap), reinterpret_cast<const
//             __half*>(bp), reinterpret_cast<__half*>(op), M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T,
//    __nv_bfloat16>::value) {
//       // Try cuBLAS first for BF16: compute in BF16, accumulate in FP32
//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool out_contiguous = (meta.out_strides[on-1] == 1);
//       bool is_supported = (a_contiguous || (an >= 2 && meta.a_strides[an-2]
//       == 1)) &&
//                           (b_contiguous || (bn >= 2 && meta.b_strides[bn-2]
//                           == 1)) && out_contiguous;
//       bool cublas_ok = false;
//       if (is_supported) {
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);
//          float alpha_f = 1.0f, beta_f = 0.0f;
//          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//          int lda = a_contiguous ? meta.a_strides[an-2] :
//          meta.a_strides[an-1]; cublasOperation_t opB = b_contiguous ?
//          CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//          meta.b_strides[bn-2] : meta.b_strides[bn-1]; int ldc =
//          meta.out_strides[on-2]; const __nv_bfloat16* ap_b =
//          reinterpret_cast<const __nv_bfloat16*>(ap); const __nv_bfloat16*
//          bp_b = reinterpret_cast<const __nv_bfloat16*>(bp);
//          __nv_bfloat16* op_b = reinterpret_cast<__nv_bfloat16*>(op);
//          cublasStatus_t status;
//          if (tb == 1) {
//             status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_b, CUDA_R_16BF, ldb, ap_b, CUDA_R_16BF, lda,
//                &beta_f, op_b, CUDA_R_16BF, ldc,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          } else {
//             long long stride_a_ll = (an >= 3) ? (long
//             long)meta.a_strides[an-3] : 0LL; long long stride_b_ll = (bn >=
//             3) ? (long long)meta.b_strides[bn-3] : 0LL; long long stride_c_ll
//             = (long long)meta.out_strides[on-3]; status =
//             cublasGemmStridedBatchedEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_b, CUDA_R_16BF, ldb, stride_b_ll, ap_b, CUDA_R_16BF, lda,
//                stride_a_ll, &beta_f, op_b, CUDA_R_16BF, ldc, stride_c_ll, tb,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          }
//          cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
//       }
//       if (!cublas_ok) {
//          matmul_bf16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128,
//          (M+127)/128, tb), 512, 0, stream>>>(
//             reinterpret_cast<const __nv_bfloat16*>(ap),
//             reinterpret_cast<const __nv_bfloat16*>(bp),
//             reinterpret_cast<__nv_bfloat16*>(op), M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, float>::value) {
//       // Use cuBLAS for FP32 matmuls - enables TF32 Tensor Core acceleration
//       // Check if tensors are contiguous (row-major) OR transposed (col-major
//       equivalent) for cuBLAS fast path

//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool a_transposed = (an >= 2 && meta.a_strides[an-2] == 1);

//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool b_transposed = (bn >= 2 && meta.b_strides[bn-2] == 1);

//       bool out_contiguous = (meta.out_strides[on-1] == 1);

//       // Use cuBLAS for:
//       // 1. 2D matrices (tb=1)
//       // 2. Strided Batched matrices (tb>1)
//       bool is_supported_layout = (a_contiguous || a_transposed) &&
//       (b_contiguous || b_transposed) && out_contiguous; bool is_strided_batch
//       = false; long long stride_a = 0, stride_b = 0, stride_c = 0;

//       if (tb > 1 && is_supported_layout) {
//           // Calculate Strides for Batches
//           // For broadcasted dims (e.g. B is 2D but A is 3D), stride is 0.

//           // Helper to get batch stride check
//           // Assumption: dimensions 0..on-3 are the batch dimensions.
//           // Standard layout: Flattened batch.

//           if (on == 3) {
//              stride_c = meta.out_strides[0];
//              stride_a = (an == 3) ? meta.a_strides[0] : 0; // Broadcast if an
//              < 3 stride_b = (bn == 3) ? meta.b_strides[0] : 0; // Broadcast
//              if bn < 3 is_strided_batch = true;
//           } else if (on == 4) {
//             // 4D [d0, d1, M, N]: use inner batch stride (stride[1]) for
//             cuBLAS stride_c = meta.out_strides[on - 3]; stride_a = (an == 4)
//             ? meta.a_strides[an - 3] : 0; stride_b = (bn == 4) ?
//             meta.b_strides[bn - 3] : 0;

//             // cuBLAS strided batched requires uniform stride across all tb
//             batches.
//             // Uniform iff stride[0] == stride[1] * shape[1] for each tensor.
//             // Non-contiguous transposes (e.g. reshape+transpose without
//             contiguous)
//             // break this — handled by per-outer-batch loop below.
//             bool batch_uniform = (meta.out_strides[0] == meta.out_strides[1]
//             * osh[1]); if (an == 4) batch_uniform = batch_uniform &&
//             (meta.a_strides[0] == meta.a_strides[1] * ash[1]); if (bn == 4)
//             batch_uniform = batch_uniform && (meta.b_strides[0] ==
//             meta.b_strides[1] * bsh[1]);

//             if (batch_uniform) {
//                 is_strided_batch = true;
//             }
//             // else: non-uniform, handled by loop path below
//           }
//       }

//       if ((tb == 1 || is_strided_batch) && is_supported_layout) {
//          // USE EXPLICIT DEVICE INDEXING from output tensor
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);

//          float alpha = 1.0f, beta = 0.0f;

//          // Setup A (Second in CuBLAS, since CuBLAS is ColMajor)
//          cublasOperation_t opA;
//          int lda;
//          if (a_contiguous) { opA = CUBLAS_OP_N; lda = meta.a_strides[an-2]; }
//          else { opA = CUBLAS_OP_T; lda = meta.a_strides[an-1]; }

//          // Setup B (First in CuBLAS)
//          cublasOperation_t opB;
//          int ldb;
//          if (b_contiguous) { opB = CUBLAS_OP_N; ldb = meta.b_strides[bn-2]; }
//          else { opB = CUBLAS_OP_T; ldb = meta.b_strides[bn-1]; }

//          int ldc = meta.out_strides[on-2];

//          // Logic correction:
//          // C = A @ B.
//          // CuBLAS (ColMajor): C^T = B^T @ A^T.
//          // Pass B as "A", A as "B".
//          //
//          // If A is contiguous RM [M, K]: Represents A^T [K, M] CM.
//          // We need A^T in the equation. So Op = N.
//          // If A is transposed RM [M, K] (stride 1, M): Represents A [K, M]
//          CM.
//          // We need A^T. So Op = T.
//          //
//          // If B is contiguous RM [K, N]: Represents B^T [N, K] CM.
//          // We need B^T in the equation. So Op = N.
//          // If B is transposed RM [K, N]: Represents B [N, K] CM.
//          // We need B^T. So Op = T.

//          cublasStatus_t status;

//          if (tb == 1) {
//              status = cublasGemmEx(
//                 handle,
//                 opB, opA,
//                 N, M, K,
//                 &alpha,
//                 bp, CUDA_R_32F, ldb,
//                 ap, CUDA_R_32F, lda,
//                 &beta,
//                 op, CUDA_R_32F, ldc,
//                 CUBLAS_COMPUTE_32F_FAST_TF32,
//                 CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//          } else {
//              status = cublasGemmStridedBatchedEx(
//                 handle,
//                 opB, opA,
//                 N, M, K,
//                 &alpha,
//                 bp, CUDA_R_32F, ldb, stride_b,
//                 ap, CUDA_R_32F, lda, stride_a,
//                 &beta,
//                 op, CUDA_R_32F, ldc, stride_c,
//                 tb,
//                 CUBLAS_COMPUTE_32F_FAST_TF32,
//                 CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//          }

//          if (status != CUBLAS_STATUS_SUCCESS) {
//             // Fallback (e.g. if alignment issues, though Ex handles most)
//              matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//              (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb,
//              meta);
//          }
//       } else if (on == 4 && !is_strided_batch && is_supported_layout) {
//          // Non-uniform 4D batch strides (e.g. after transpose without
//          contiguous):
//          // Loop over outer batch dim d0, each call handles d1 inner batches
//          with uniform stride int current_device;
//          cudaGetDevice(&current_device); cublasHandle_t handle =
//          get_cublas_handle(current_device); cublasSetStream(handle, stream);
//          float alpha_v = 1.0f, beta_v = 0.0f;

//          cublasOperation_t opA_l, opB_l;
//          int lda_l, ldb_l, ldc_l;
//          if (a_contiguous) { opA_l = CUBLAS_OP_N; lda_l =
//          meta.a_strides[an-2]; } else { opA_l = CUBLAS_OP_T; lda_l =
//          meta.a_strides[an-1]; } if (b_contiguous) { opB_l = CUBLAS_OP_N;
//          ldb_l = meta.b_strides[bn-2]; } else { opB_l = CUBLAS_OP_T; ldb_l =
//          meta.b_strides[bn-1]; } ldc_l = meta.out_strides[on-2];

//          int d0 = osh[0], d1 = osh[1];
//          bool all_ok = true;
//          for (int bi = 0; bi < d0 && all_ok; bi++) {
//              const float* a_b = ap + (an == 4 ? bi * meta.a_strides[0] : 0);
//              const float* b_b = bp + (bn == 4 ? bi * meta.b_strides[0] : 0);
//              float* o_b = op + bi * meta.out_strides[0];

//              cublasStatus_t st = cublasGemmStridedBatchedEx(
//                  handle, opB_l, opA_l, N, M, K, &alpha_v,
//                  b_b, CUDA_R_32F, ldb_l, stride_b,
//                  a_b, CUDA_R_32F, lda_l, stride_a,
//                  &beta_v,
//                  o_b, CUDA_R_32F, ldc_l, stride_c,
//                  d1,
//                  CUBLAS_COMPUTE_32F_FAST_TF32,
//                  CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//              if (st != CUBLAS_STATUS_SUCCESS) all_ok = false;
//          }
//          if (!all_ok) {
//              matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//              (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb,
//              meta);
//          }
//       } else {
//          // Fallback to custom kernel
//          matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//          (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, double>::value) {
//       matmul_fp64_optimized<64, 64, 8, 4, 4><<<dim3((N+63)/64, (M+63)/64,
//       tb), 256, 0, stream>>>(reinterpret_cast<const double*>(ap),
//       reinterpret_cast<const double*>(bp), reinterpret_cast<double*>(op), M,
//       N, K, tb, meta);
//    }

//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) throw std::runtime_error("Kernel failed: " +
//    std::string(cudaGetErrorString(err)));
// }

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output,
// cudaStream_t stream) {
//    dispatch_by_dtype(A.dtype(), [&](auto d) {
//       using T = decltype(d);
//       if constexpr (std::is_same<T, float16_t>::value || std::is_same<T,
//       bfloat16_t>::value || std::is_same<T, float>::value || std::is_same<T,
//       double>::value) launch_optimized_matmul<T>(A, B, output, stream); else
//       throw std::runtime_error("Unsupported type");
//    });
// }

// void cuda_addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
// float alpha, float beta, Tensor& output, cudaStream_t stream) {
//     AUTO_PROFILE_CUDA("Forward::Addmm_CUDA");
//     const auto& m1sh = mat1.shape().dims, &m2sh = mat2.shape().dims, &osh =
//     output.shape().dims; int m1n = m1sh.size(), m2n = m2sh.size(), on =
//     osh.size(); int M = m1sh[m1n-2], K = m1sh[m1n-1], N = m2sh[m2n-1];

//     // Compute total batch count from output shape (all dims except last 2)
//     int tb = 1;
//     for (int i = 0; i < on - 2; i++) tb *= osh[i];
//     int64_t total_output_elements = output.numel(); // tb * M * N

//     // 1. Prepare output buffer: output = beta * input (with broadcasting)
//     //    Scale ALL elements of the output, not just M*N
//     if (beta != 0.0f) {
//         dispatch_by_dtype(output.dtype(), [&](auto d) {
//             using T = decltype(d);
//             if constexpr (std::is_same_v<T, float> || std::is_same_v<T,
//             double> || std::is_same_v<T, __half> || std::is_same_v<T,
//             __nv_bfloat16> || std::is_integral_v<T>) {
//                 // Use total_output_elements as both rows*cols and total
//                 // M_total and N stay the same for broadcasting logic (bias
//                 is broadcast per [M,N] slice) int64_t rows_total =
//                 (int64_t)tb * M; launch_broadcast_scale<T>(output.data<T>(),
//                 input.data<T>(), beta, rows_total, N, input.numel(), stream);
//             }
//         });
//     } else {
//         // If beta is 0, input is ignored, zero the entire output
//         dispatch_by_dtype(output.dtype(), [&](auto d) {
//             using T = decltype(d);
//             if constexpr (std::is_same_v<T, float> || std::is_same_v<T,
//             double> || std::is_same_v<T, __half> || std::is_same_v<T,
//             __nv_bfloat16> || std::is_integral_v<T>) {
//                 int threads = 256;
//                 int blocks = (total_output_elements + threads - 1) / threads;
//                 // Simple zero kernel
//                 broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(
//                     output.data<T>(), output.data<T>(), 0.0f,
//                     total_output_elements, total_output_elements, (int64_t)tb
//                     * M, N);
//             }
//         });
//     }

//     // 2. Add alpha * (mat1 @ mat2) using cuBLAS
//     if (output.dtype() == Dtype::Float32) {
//         MatmulMetadata meta; meta.a_ndim = m1n; meta.b_ndim = m2n;
//         meta.out_ndim = on; for (int i = 0; i < m1n; i++) { meta.a_shape[i] =
//         m1sh[i]; meta.a_strides[i] = mat1.stride().strides[i]; } for (int i =
//         0; i < m2n; i++) { meta.b_shape[i] = m2sh[i]; meta.b_strides[i] =
//         mat2.stride().strides[i]; } for (int i = 0; i < on; i++) {
//         meta.out_shape[i] = osh[i]; meta.out_strides[i] =
//         output.stride().strides[i]; }

//         bool a_contiguous = (meta.a_strides[m1n-1] == 1);
//         bool a_transposed = (m1n >= 2 && meta.a_strides[m1n-2] == 1);
//         bool b_contiguous = (meta.b_strides[m2n-1] == 1);
//         bool b_transposed = (m2n >= 2 && meta.b_strides[m2n-2] == 1);
//         bool out_contiguous = (meta.out_strides[on-1] == 1);

//         bool is_supported_layout = (a_contiguous || a_transposed) &&
//         (b_contiguous || b_transposed) && out_contiguous;

//         // Compute batch strides (mirroring launch_optimized_matmul)
//         bool is_strided_batch = false;
//         long long stride_a = 0, stride_b = 0, stride_c = 0;

//         if (tb > 1 && is_supported_layout) {
//             if (on == 3) {
//                 stride_c = meta.out_strides[0];
//                 stride_a = (m1n == 3) ? meta.a_strides[0] : 0;
//                 stride_b = (m2n == 3) ? meta.b_strides[0] : 0;
//                 is_strided_batch = true;
//             } else if (on == 4) {
//                 stride_c = static_cast<long long>(M) * N;
//                 stride_a = (m1n == 4) ? static_cast<long long>(M) * K : 0;
//                 stride_b = (m2n == 4) ? static_cast<long long>(K) * N : 0;
//                 is_strided_batch = true;
//             }
//         }

//         if ((tb == 1 || is_strided_batch) && is_supported_layout) {
//             // USE EXPLICIT DEVICE INDEXING from output tensor
//             int dev_idx = output.device().index;
//             cublasHandle_t handle = get_cublas_handle(dev_idx);
//             cublasSetStream(handle, stream);

//             cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//             int lda = a_contiguous ? meta.a_strides[m1n-2] :
//             meta.a_strides[m1n-1]; cublasOperation_t opB = b_contiguous ?
//             CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//             meta.b_strides[m2n-2] : meta.b_strides[m2n-1]; int ldc =
//             meta.out_strides[on-2];

//             // output already has beta * input, so we pass cublas_beta = 1.0
//             to accumulate float cublas_beta = 1.0f;

//             cublasStatus_t status;

//             if (tb == 1) {
//                 status = cublasGemmEx(
//                     handle, opB, opA, N, M, K, &alpha,
//                     mat2.data<float>(), CUDA_R_32F, ldb,
//                     mat1.data<float>(), CUDA_R_32F, lda,
//                     &cublas_beta, output.data<float>(), CUDA_R_32F, ldc,
//                     CUBLAS_COMPUTE_32F_FAST_TF32,
//                     CUBLAS_GEMM_DEFAULT_TENSOR_OP
//                 );
//             } else {
//                 status = cublasGemmStridedBatchedEx(
//                     handle, opB, opA, N, M, K, &alpha,
//                     mat2.data<float>(), CUDA_R_32F, ldb, stride_b,
//                     mat1.data<float>(), CUDA_R_32F, lda, stride_a,
//                     &cublas_beta, output.data<float>(), CUDA_R_32F, ldc,
//                     stride_c, tb, CUBLAS_COMPUTE_32F_FAST_TF32,
//                     CUBLAS_GEMM_DEFAULT_TENSOR_OP
//                 );
//             }

//             if (status == CUBLAS_STATUS_SUCCESS) return;
//             // Fall through to fallback on failure
//         }
//     } else if (output.dtype() == Dtype::Float16 || output.dtype() ==
//     Dtype::Bfloat16) {
//         // cuBLAS path for FP16/BF16 addmm: compute in fp16/bf16, accumulate
//         in fp32 MatmulMetadata meta16; meta16.a_ndim = m1n; meta16.b_ndim =
//         m2n; meta16.out_ndim = on; for (int i = 0; i < m1n; i++) {
//         meta16.a_shape[i] = m1sh[i]; meta16.a_strides[i] =
//         mat1.stride().strides[i]; } for (int i = 0; i < m2n; i++) {
//         meta16.b_shape[i] = m2sh[i]; meta16.b_strides[i] =
//         mat2.stride().strides[i]; } for (int i = 0; i < on; i++) {
//         meta16.out_shape[i] = osh[i]; meta16.out_strides[i] =
//         output.stride().strides[i]; }

//         bool a_contiguous = (meta16.a_strides[m1n-1] == 1);
//         bool b_contiguous = (meta16.b_strides[m2n-1] == 1);
//         bool out_contiguous = (meta16.out_strides[on-1] == 1);
//         bool is_supported = (a_contiguous || (m1n >= 2 &&
//         meta16.a_strides[m1n-2] == 1)) &&
//                             (b_contiguous || (m2n >= 2 &&
//                             meta16.b_strides[m2n-2] == 1)) && out_contiguous;

//         if (is_supported) {
//             int dev_idx = output.device().index;
//             cublasHandle_t handle = get_cublas_handle(dev_idx);
//             cublasSetStream(handle, stream);

//             cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//             int lda = a_contiguous ? meta16.a_strides[m1n-2] :
//             meta16.a_strides[m1n-1]; cublasOperation_t opB = b_contiguous ?
//             CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//             meta16.b_strides[m2n-2] : meta16.b_strides[m2n-1]; int ldc =
//             meta16.out_strides[on-2];
//             // output already holds beta*input; pass cublas_beta=1.0 to
//             accumulate alpha*(mat1@mat2) float cublas_beta = 1.0f;

//             bool is_strided_batch = (tb > 1) && (on == 3 || on == 4);
//             long long stride_a_ll = 0, stride_b_ll = 0, stride_c_ll = 0;
//             if (is_strided_batch) {
//                 stride_c_ll = meta16.out_strides[on-3];
//                 stride_a_ll = (m1n >= on) ? meta16.a_strides[m1n-3] : 0LL;
//                 stride_b_ll = (m2n >= on) ? meta16.b_strides[m2n-3] : 0LL;
//             }

//             cudaDataType_t cuda_type = (output.dtype() == Dtype::Float16) ?
//             CUDA_R_16F : CUDA_R_16BF; cublasStatus_t status;

//             if (output.dtype() == Dtype::Float16) {
//                 const __half* m1h = reinterpret_cast<const
//                 __half*>(mat1.data<float16_t>()); const __half* m2h =
//                 reinterpret_cast<const __half*>(mat2.data<float16_t>());
//                 __half* outh =
//                 reinterpret_cast<__half*>(output.data<float16_t>()); if (tb
//                 == 1) {
//                     status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha,
//                         m2h, CUDA_R_16F, ldb, m1h, CUDA_R_16F, lda,
//                         &cublas_beta, outh, CUDA_R_16F, ldc,
//                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 } else {
//                     status = cublasGemmStridedBatchedEx(handle, opB, opA, N,
//                     M, K, &alpha,
//                         m2h, CUDA_R_16F, ldb, stride_b_ll, m1h, CUDA_R_16F,
//                         lda, stride_a_ll, &cublas_beta, outh, CUDA_R_16F,
//                         ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
//                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 }
//             } else {
//                 const __nv_bfloat16* m1b = reinterpret_cast<const
//                 __nv_bfloat16*>(mat1.data<bfloat16_t>()); const
//                 __nv_bfloat16* m2b = reinterpret_cast<const
//                 __nv_bfloat16*>(mat2.data<bfloat16_t>());
//                 __nv_bfloat16* outb =
//                 reinterpret_cast<__nv_bfloat16*>(output.data<bfloat16_t>());
//                 if (tb == 1) {
//                     status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha,
//                         m2b, CUDA_R_16BF, ldb, m1b, CUDA_R_16BF, lda,
//                         &cublas_beta, outb, CUDA_R_16BF, ldc,
//                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 } else {
//                     status = cublasGemmStridedBatchedEx(handle, opB, opA, N,
//                     M, K, &alpha,
//                         m2b, CUDA_R_16BF, ldb, stride_b_ll, m1b, CUDA_R_16BF,
//                         lda, stride_a_ll, &cublas_beta, outb, CUDA_R_16BF,
//                         ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
//                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 }
//             }
//             if (status == CUBLAS_STATUS_SUCCESS) return;
//             // Fall through to fallback on failure
//         }
//     }

//     // Fallback: Use existing batched matmul + scaled addition kernel
//     Tensor temp_matmul = OwnTensor::matmul(mat1, mat2, stream);

//     dispatch_by_dtype(output.dtype(), [&](auto d) {
//         using T = decltype(d);
//         if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>
//         || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> ||
//         std::is_integral_v<T>) {
//             int64_t total = output.numel();
//             int threads = 256;
//             int blocks = (total + threads - 1) / threads;
//             add_scaled_kernel_typed<T><<<blocks, threads, 0,
//             stream>>>(output.data<T>(), temp_matmul.data<T>(), alpha, total);
//         }
//     });
// }

// } // namespace OwnTensor
// // #endif
// #endif // WITH_MYBLAS

// #ifdef WITH_MYBLAS

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <stdio.h>
// #include <stdexcept>
// #include <memory>
// #include <mutex>

// #include "mycublas.h"
// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "utils/Profiler.h"

// namespace OwnTensor {

// // Thread-safe cached handle — created once per device, never destroyed.
// // Same pattern as MatmulBackward.cu. Eliminates per-call Create/Destroy
// overhead. static std::mutex g_fwd_mutex; static mycublasHandle_t
// g_mycublas_handles_fwd[8] = {nullptr};

// static mycublasHandle_t get_mycublas_handle_fwd(int device = 0) {
//     if (g_mycublas_handles_fwd[device] == nullptr) {
//         std::lock_guard<std::mutex> lock(g_fwd_mutex);
//         if (g_mycublas_handles_fwd[device] == nullptr) {
//             cudaSetDevice(device);
//             mycublasCreate(&g_mycublas_handles_fwd[device]);
//         }
//     }
//     return g_mycublas_handles_fwd[device];
// }

// //
// ============================================================================
// // DISPATCH LAYER - PURE MYBLAS
// //
// ============================================================================

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output,
// cudaStream_t stream)
// {
//     AUTO_PROFILE_CUDA("Forward::Matmul_CUDA_MyBlas");

//     const auto& a_sh = A.shape().dims;
//     const auto& b_sh = B.shape().dims;
//     const auto& o_sh = output.shape().dims;
//     int a_ndim = (int)a_sh.size();
//     int b_ndim = (int)b_sh.size();

//     // Accept row-major (strides[-1]==1) or col-major/transposed
//     (strides[-2]==1).
//     // Anything else (e.g. arbitrary non-contiguous) falls back to an
//     explicit copy. bool a_row = (A.stride().strides[a_ndim-1] == 1); bool
//     a_col = (a_ndim >= 2 && A.stride().strides[a_ndim-2] == 1); bool b_row =
//     (B.stride().strides[b_ndim-1] == 1); bool b_col = (b_ndim >= 2 &&
//     B.stride().strides[b_ndim-2] == 1);

//     if (!(a_row || a_col) || !(b_row || b_col)) {
//         const Tensor A_c = A.is_contiguous() ? A : A.contiguous();
//         const Tensor B_c = B.is_contiguous() ? B : B.contiguous();
//         cuda_matmul(A_c, B_c, output, stream);
//         return;
//     }

//     int M = (int)a_sh[a_ndim - 2];
//     int K = (int)a_sh[a_ndim - 1];
//     int N = (int)b_sh[b_ndim - 1];

//     // row-major: OP_N, ld = strides[-2]  |  col-major: OP_T, ld =
//     strides[-1] mycublasOperation_t opA = a_row ? MYCUBLAS_OP_N :
//     MYCUBLAS_OP_T; int lda = a_row ? A.stride().strides[a_ndim-2] :
//     A.stride().strides[a_ndim-1]; mycublasOperation_t opB = b_row ?
//     MYCUBLAS_OP_N : MYCUBLAS_OP_T; int ldb = b_row ?
//     B.stride().strides[b_ndim-2] : B.stride().strides[b_ndim-1];

//     int batch_count = 1;
//     for (int i = 0; i < (int)o_sh.size() - 2; ++i) batch_count *=
//     (int)o_sh[i];

//     // Use actual tensor batch stride (handles non-default layouts correctly)
//     long long strideA = (A.numel() == (size_t)M * K) ? 0LL :
//                         (a_ndim >= 3 ? (long
//                         long)A.stride().strides[a_ndim-3] : (long long)M *
//                         K);
//     long long strideB = (B.numel() == (size_t)K * N) ? 0LL :
//                         (b_ndim >= 3 ? (long
//                         long)B.stride().strides[b_ndim-3] : (long long)K *
//                         N);
//     long long strideC = (long long)M * N;

//     int dev = 0; cudaGetDevice(&dev);
//     mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
//     mycublasSetStream(handle, stream);

//     dispatch_by_dtype(A.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
//         const T* a_ptr = A.data<T>();
//         const T* b_ptr = B.data<T>();
//         T* out_ptr = output.data<T>();

//         if constexpr (std::is_same<T, float>::value) {
//             mycublasSgemmStridedBatched(handle, opA, opB, M, N, K, 1.0f,
//                 a_ptr, lda, strideA, b_ptr, ldb, strideB, 0.0f, out_ptr, N,
//                 strideC, batch_count);
//         }
//         else if constexpr (std::is_same<T, __half>::value || std::is_same<T,
//         OwnTensor::float16_t>::value) {
//             __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
//             mycublasHgemmStridedBatched(handle, opA, opB, M, N, K, alpha,
//                 (const __half*)a_ptr, lda, strideA, (const __half*)b_ptr,
//                 ldb, strideB, beta,
//                 (__half*)out_ptr, N, strideC, batch_count);
//         }
//         else if constexpr (std::is_same<T, __nv_bfloat16>::value ||
//         std::is_same<T, OwnTensor::bfloat16_t>::value) {
//             __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta =
//             __float2bfloat16(0.0f); mycublasBgemmStridedBatched(handle, opA,
//             opB, M, N, K, alpha,
//                 (const __nv_bfloat16*)a_ptr, lda, strideA, (const
//                 __nv_bfloat16*)b_ptr, ldb, strideB, beta,
//                 (__nv_bfloat16*)out_ptr, N, strideC, batch_count);
//         }
//         else if constexpr (std::is_same<T, double>::value) {
//             // double: no op-flag support in mycublasDgemm, fall back to
//             contiguous copy const Tensor A_c = A.is_contiguous() ? A :
//             A.contiguous(); const Tensor B_c = B.is_contiguous() ? B :
//             B.contiguous(); long long sa = (A_c.numel() == (size_t)M*K) ? 0LL
//             : (long long)M*K; long long sb = (B_c.numel() == (size_t)K*N) ?
//             0LL : (long long)K*N; mycublasDgemmStridedBatched(handle, M, N,
//             K, 1.0,
//                 (const double*)A_c.data<T>(), K, sa,
//                 (const double*)B_c.data<T>(), N, sb,
//                 0.0, (double*)out_ptr, N, strideC, batch_count);
//         }
//         else {
//             throw std::runtime_error("cuda_matmul: Unsupported dtype for
//             MyBlas GEMM.");
//         }
//     });

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA Error after MyBlas matmul launch: " +
//         std::string(cudaGetErrorString(err)));
//     }
// }

// //
// ============================================================================
// // ADDMM IMPLEMENTATION - USING MYBLAS
// //
// ============================================================================

// // Helper kernel for broadcasting and scaling
// template<typename T>
// __global__ void broadcast_add_scaled_kernel(T* out, const T* in, const T*
// bias, float alpha, float beta, int M, int N, int batch_count, int64_t
// stride_c, int64_t bias_numel) {
//     int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     int64_t total = (int64_t)batch_count * M * N;
//     if (idx < total) {
//         T b_val;
//         if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T,
//         complex64_t> || std::is_same_v<T, complex128_t>) {
//             b_val = T(0.0f, 0.0f);
//         } else {
//             b_val = (T)0.0f;
//         }

//         if (bias_numel == 1) b_val = bias[0];
//         else if (bias_numel == N) b_val = bias[idx % N];
//         else if (bias_numel == total) b_val = bias[idx];

//         if constexpr (std::is_same_v<T, complex32_t> || std::is_same_v<T,
//         complex64_t> || std::is_same_v<T, complex128_t>) {
//             out[idx] = out[idx] + b_val * (T)beta;
//         } else {
//             out[idx] = out[idx] + (T)(beta * (float)b_val);
//         }
//     }
// }

// void cuda_addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
// float alpha, float beta, Tensor& output, cudaStream_t stream) {
//     AUTO_PROFILE_CUDA("Forward::Addmm_CUDA_MyBlas");

//     if (output.dtype() == Dtype::Float32) {
//         const auto& o_sh  = output.shape().dims;
//         const auto& m1_sh = mat1.shape().dims;
//         const auto& m2_sh = mat2.shape().dims;
//         int m1_ndim = (int)m1_sh.size();
//         int m2_ndim = (int)m2_sh.size();

//         int M = (int)o_sh[o_sh.size() - 2];
//         int N = (int)o_sh[o_sh.size() - 1];
//         int K = (int)m1_sh[m1_ndim - 1];
//         int batch_count = 1;
//         for (size_t i = 0; i < o_sh.size() - 2; ++i) batch_count *=
//         (int)o_sh[i];

//         bool m1_row = (mat1.stride().strides[m1_ndim-1] == 1);
//         bool m1_col = (m1_ndim >= 2 && mat1.stride().strides[m1_ndim-2] ==
//         1); bool m2_row = (mat2.stride().strides[m2_ndim-1] == 1); bool
//         m2_col = (m2_ndim >= 2 && mat2.stride().strides[m2_ndim-2] == 1);

//         int dev = 0; cudaGetDevice(&dev);
//         mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
//         mycublasSetStream(handle, stream);

//         // Fast path: both row-major → use the specialized fused SgemmAddmm
//         kernel if (m1_row && m2_row) {
//             int lda = (int)K;
//             int ldb = (int)N;
//             long long strideA = (mat1.numel() == (size_t)M * K) ? 0LL : (long
//             long)M * K; long long strideB = (mat2.numel() == (size_t)K * N) ?
//             0LL : (long long)K * N; mycublasSgemmAddmm(handle, M, N, K,
//             alpha,
//                                mat1.data<float>(), lda, strideA,
//                                mat2.data<float>(), ldb, strideB,
//                                beta, input.data<float>(),
//                                (int64_t)input.numel(), output.data<float>(),
//                                N, (long long)M * N, batch_count);
//             return;
//         }

//         // General path: mat1 or mat2 is transposed (e.g. x @ W.T).
//         // Use layout-aware GEMM + separate bias-add. Avoids the explicit
//         contiguous copy. if ((m1_row || m1_col) && (m2_row || m2_col)) {
//             mycublasOperation_t opA = m1_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
//             int lda = m1_row ? mat1.stride().strides[m1_ndim-2] :
//             mat1.stride().strides[m1_ndim-1]; mycublasOperation_t opB =
//             m2_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T; int ldb = m2_row ?
//             mat2.stride().strides[m2_ndim-2] :
//             mat2.stride().strides[m2_ndim-1];

//             long long strideA = (mat1.numel() == (size_t)M * K) ? 0LL :
//                                 (m1_ndim >= 3 ? (long
//                                 long)mat1.stride().strides[m1_ndim-3] : (long
//                                 long)M * K);
//             long long strideB = (mat2.numel() == (size_t)K * N) ? 0LL :
//                                 (m2_ndim >= 3 ? (long
//                                 long)mat2.stride().strides[m2_ndim-3] : (long
//                                 long)K * N);

//             mycublasSgemmStridedBatched(handle, opA, opB, M, N, K, alpha,
//                 mat1.data<float>(), lda, strideA,
//                 mat2.data<float>(), ldb, strideB,
//                 0.0f, output.data<float>(), N, (long long)M * N,
//                 batch_count);

//             // Add bias: output += beta * input (broadcast)
//             int64_t total = output.numel();
//             int threads = 256, blocks = (total + threads - 1) / threads;
//             broadcast_add_scaled_kernel<float><<<blocks, threads, 0,
//             stream>>>(
//                 output.data<float>(), nullptr, input.data<float>(),
//                 alpha, beta, M, N, batch_count, (int64_t)M*N, input.numel());
//             return;
//         }

//         // True non-contiguous fallback (rare)
//         const Tensor m1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
//         const Tensor m2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
//         mycublasSgemmAddmm(handle, M, N, K, alpha,
//                            m1_c.data<float>(), K, (mat1.numel() ==
//                            (size_t)M*K) ? 0LL : (long long)M*K,
//                            m2_c.data<float>(), N, (mat2.numel() ==
//                            (size_t)K*N) ? 0LL : (long long)K*N, beta,
//                            input.data<float>(), (int64_t)input.numel(),
//                            output.data<float>(), N, (long long)M*N,
//                            batch_count);
//         return;
//     }

//     // Non-float32 path: route through cuda_matmul (which handles transposed
//     inputs),
//     // then add the bias. No forced contiguous copy needed.
//     cuda_matmul(mat1, mat2, output, stream);

//     const auto& out_shape = output.shape().dims;
//     int M = out_shape[out_shape.size() - 2];
//     int N = out_shape[out_shape.size() - 1];
//     int batch_count = 1;
//     for (size_t i = 0; i < out_shape.size() - 2; ++i) batch_count *=
//     static_cast<int>(out_shape[i]);

//     int64_t total = output.numel();
//     int threads = 256;
//     int blocks = (total + threads - 1) / threads;

//     dispatch_by_dtype(output.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
//         broadcast_add_scaled_kernel<T><<<blocks, threads, 0, stream>>>(
//             output.data<T>(), nullptr, input.data<T>(), alpha, beta, M, N,
//             batch_count, (int64_t)M*N, input.numel()
//         );
//     });
// }

// } // namespace OwnTensor
// // #endif
// #else

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include "utils/Profiler.h"
// #include <mma.h>
// #include <cublas_v2.h>
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
// #include <string>
// #include <mutex>

// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "ops/Kernels.h"

// namespace OwnTensor {

// // cuBLAS handle management - thread-safe singleton per device
// static std::mutex cublas_mutex;
// static cublasHandle_t g_cublas_handles[8] = {nullptr};

// static cublasHandle_t get_cublas_handle(int device = 0) {
//    if (g_cublas_handles[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex);
//       if (g_cublas_handles[device] == nullptr) {
//          cudaSetDevice(device);
//          cublasCreate(&g_cublas_handles[device]);
//          // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant
//          speedup cublasSetMathMode(g_cublas_handles[device],
//          CUBLAS_TF32_TENSOR_OP_MATH);
//       }
//    }
//    return g_cublas_handles[device];
// }

// using namespace nvcuda;

// //
// ============================================================================
// // METADATA & CONSTANTS
// //
// ============================================================================

// struct MatmulMetadata {
//    int a_shape[8], b_shape[8], out_shape[8];
//    int a_strides[8], b_strides[8], out_strides[8];
//    int a_ndim, b_ndim, out_ndim;
// };

// __device__ void compute_batch_offset(int batch_idx, const int* shape, const
// int* strides, int ndim, const int* out_shape, int out_ndim, int& offset) {
//    offset = 0; if (out_ndim <= 2) return;
//    int temp_batch = batch_idx;
//    for (int dim = out_ndim - 3; dim >= 0; --dim) {
//       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
//       temp_batch /= b_dim_sz;
//       int c_dim = dim - (out_ndim - ndim);
//       if (c_dim >= 0 && c_dim < ndim - 2) offset += (int64_t)((shape[c_dim] >
//       1) ? b_coord : 0) * strides[c_dim];
//    }
// }

// constexpr int PAD = 8;

// //
// ============================================================================
// // ADDMM HELPERS
// //
// ============================================================================

// template<typename T>
// __global__ void broadcast_scale_kernel(T* __restrict__ output, const T*
// __restrict__ input, float scale,
//     int64_t total_elements, int64_t input_size, int64_t rows, int64_t cols) {
//     const int64_t tid     = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     const int64_t gstride = gridDim.x  * (int64_t)blockDim.x;

//     if constexpr (std::is_same_v<T, float>) {
//         constexpr int VEC = 4;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             float4* __restrict__       out4 =
//             reinterpret_cast<float4*>(output); const float4* __restrict__ in4
//             = reinterpret_cast<const float4*>(input); const int64_t n4 =
//             total_elements / VEC; const float sv = (input_size == 1) ?
//             (float)input[0] * scale : 0.0f; for (int64_t i = tid; i < n4; i
//             += gstride) {
//                 float4 v;
//                 if (input_size == 1) {
//                     v = make_float4(sv, sv, sv, sv);
//                 } else if (input_size == cols) {
//                     const float4 iv = in4[((i * VEC) % cols) / VEC];
//                     v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
//                     iv.w * scale);
//                 } else {
//                     const float4 iv = in4[i];
//                     v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
//                     iv.w * scale);
//                 }
//                 out4[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, double>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             double2* __restrict__       out2 =
//             reinterpret_cast<double2*>(output); const double2* __restrict__
//             in2  = reinterpret_cast<const double2*>(input); const int64_t n2
//             = total_elements / VEC; const double sv = (input_size == 1) ?
//             (double)input[0] * (double)scale : 0.0; for (int64_t i = tid; i <
//             n2; i += gstride) {
//                 double2 v;
//                 if (input_size == 1) {
//                     v = make_double2(sv, sv);
//                 } else if (input_size == cols) {
//                     const double2 iv = in2[((i * VEC) % cols) / VEC];
//                     v = make_double2(iv.x * scale, iv.y * scale);
//                 } else {
//                     const double2 iv = in2[i];
//                     v = make_double2(iv.x * scale, iv.y * scale);
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((double)input[0] * (double)scale); else if (input_size ==
//                 cols)           output[i] = (T)((double)input[i % cols] *
//                 (double)scale); else if (input_size == total_elements)
//                 output[i] = (T)((double)input[i] * (double)scale); else
//                 output[i] = (T)0.0;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, __half>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             half2* __restrict__       out2 =
//             reinterpret_cast<half2*>(output); const half2* __restrict__ in2
//             = reinterpret_cast<const half2*>(input); const int64_t n2     =
//             total_elements / VEC; const half2   scale2 =
//             __float2half2_rn(scale); const half2   sv2    = (input_size == 1)
//             ? __float2half2_rn((float)input[0] * scale)
//                                                      :
//                                                      __float2half2_rn(0.0f);
//             for (int64_t i = tid; i < n2; i += gstride) {
//                 half2 v;
//                 if (input_size == 1) {
//                     v = sv2;
//                 } else if (input_size == cols) {
//                     v = __hmul2(in2[((i * VEC) % cols) / VEC], scale2);
//                 } else {
//                     v = __hmul2(in2[i], scale2);
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
//         constexpr int VEC = 2;
//         const bool can_vec = (total_elements % VEC == 0) &&
//                              (input_size == 1 || input_size == total_elements
//                              ||
//                               (input_size == cols && cols % VEC == 0));
//         if (can_vec) {
//             __nv_bfloat162* __restrict__       out2 =
//             reinterpret_cast<__nv_bfloat162*>(output); const __nv_bfloat162*
//             __restrict__ in2  = reinterpret_cast<const
//             __nv_bfloat162*>(input); const int64_t        n2  =
//             total_elements / VEC; const __nv_bfloat162 sv2 = (input_size ==
//             1)
//                                        ? __float2bfloat162_rn((float)input[0]
//                                        * scale) : __float2bfloat162_rn(0.0f);
//             for (int64_t i = tid; i < n2; i += gstride) {
//                 __nv_bfloat162 v;
//                 if (input_size == 1) {
//                     v = sv2;
//                 } else if (input_size == cols) {
//                     const float2 f = __bfloat1622float2(in2[((i * VEC) %
//                     cols) / VEC]); v = __float22bfloat162_rn(make_float2(f.x
//                     * scale, f.y * scale));
//                 } else {
//                     const float2 f = __bfloat1622float2(in2[i]);
//                     v = __float22bfloat162_rn(make_float2(f.x * scale, f.y *
//                     scale));
//                 }
//                 out2[i] = v;
//             }
//         } else {
//             for (int64_t i = tid; i < total_elements; i += gstride) {
//                 if      (input_size == 1)             output[i] =
//                 (T)((float)input[0] * scale); else if (input_size == cols)
//                 output[i] = (T)((float)input[i % cols] * scale); else if
//                 (input_size == total_elements) output[i] =
//                 (T)((float)input[i] * scale); else output[i] = (T)0.0f;
//             }
//         }
//     } else if constexpr (std::is_integral_v<T>) {
//         for (int64_t i = tid; i < total_elements; i += gstride) {
//             if      (input_size == 1)             output[i] =
//             (T)((float)input[0] * scale); else if (input_size == cols)
//             output[i] = (T)((float)input[i % cols] * scale); else if
//             (input_size == total_elements) output[i] = (T)((float)input[i] *
//             scale); else                                   output[i] = (T)0;
//         }
//     } else {
//         for (int64_t i = tid; i < total_elements; i += gstride) {
//             if (input_size == total_elements) output[i] = input[i];
//             else                              output[i] = (T)0.0f;
//         }
//     }
// }

// template<typename T>
// void launch_broadcast_scale(T* output, const T* input, float scale, int64_t
// rows, int64_t cols, int64_t input_size, cudaStream_t stream) {
//     int64_t total = rows * cols;
//     // Mirror the vectorization decision the kernel will make so blocks are
//     right-sized constexpr int VEC = std::is_same_v<T, float> ? 4 :
//                        (std::is_same_v<T, double> || std::is_same_v<T,
//                        __half> ||
//                         std::is_same_v<T, __nv_bfloat16>) ? 2 : 1;
//     const bool will_vec = (VEC > 1) && (total % VEC == 0) &&
//                           (input_size == 1 || input_size == total ||
//                            (input_size == cols && cols % VEC == 0));
//     int64_t eff    = will_vec ? total / VEC : total;
//     int     threads = 256;
//     int     blocks  = (int)((eff + threads - 1) / threads);
//     broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(output, input,
//     scale, total, input_size, rows, cols);
// }

// template<typename T>
// __global__ void add_scaled_kernel_typed(T* out, const T* m, float a, int64_t
// n) {
//     int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
//     if (i < n) {
//         if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>
//         || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
//             out[i] += (T)(a * (float)m[i]);
//         } else {
//             // Fallback for types that might not support float multiplication
//             directly or need different logic
//             // For complex types, we'd need a proper implementation, but for
//             now we just skip or do basic add out[i] += (T)m[i];
//         }
//     }
// }

// //
// ============================================================================
// // FP16 WMMA KERNEL (17 TFLOPS Version)
// // BM=128, BN=128, BK=32, Threads=512
// //
// ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_fp16_optimized(const __half* __restrict__ A, const
// __half* __restrict__ B, __half* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
//    warp_col = warp_id % 4; int ao, bo, co; compute_batch_offset(batch_idx,
//    meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim,
//    ao); compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides,
//    meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
//    compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const __half *Ap = A +
//    ao, *Bp = B + bo; __half *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1]; int
//    s_bk = meta.b_strides[meta.b_ndim-2], s_bn =
//    meta.b_strides[meta.b_ndim-1]; int s_cm =
//    meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ __half As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
//    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
//    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
//    #pragma unroll
//    for(int i=0; i<2; i++) for(int j=0; j<2; j++)
//    wmma::fill_fragment(acc[i][j], __float2half(0.0f));

//    auto load_tiles = [&](int ko, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ?
//          Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2half(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk +
//          (blockIdx.x*BN+c)*s_bn] : __float2half(0.0f);
//       }
//    };
//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int k = 0; k < K; k += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (k + BK < K) load_tiles(k + BK, wi);
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks],
//             BK+PAD); #pragma unroll for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16],
//                BN+PAD); wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
//    __half* sm = reinterpret_cast<__half*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
//       int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN +
//       warp_col*WN + j*16; if (cr < M && cc < N) {
//          __half* wsm = sm + warp_id * 256;
//          wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//          for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
//             if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] =
//             wsm[r*16+c];
//       }
//    }
// }

// //
// ============================================================================
// // BF16 WMMA KERNEL
// //
// ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_bf16_optimized(const __nv_bfloat16* __restrict__ A,
// const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, int M,
// int N, int K, int total_batches, MatmulMetadata meta) {
//    const int batch_idx = blockIdx.z; if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
//    warp_col = warp_id % 4; int ao, bo, co; compute_batch_offset(batch_idx,
//    meta.a_shape, meta.a_strides, meta.a_ndim, meta.out_shape, meta.out_ndim,
//    ao); compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides,
//    meta.b_ndim, meta.out_shape, meta.out_ndim, bo);
//    compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const __nv_bfloat16 *Ap
//    = A + ao, *Bp = B + bo; __nv_bfloat16 *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1]; int
//    s_bk = meta.b_strides[meta.b_ndim-2], s_bn =
//    meta.b_strides[meta.b_ndim-1]; int s_cm =
//    meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ __nv_bfloat16 As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
//    af; wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
//    wmma::row_major> bf; wmma::fragment<wmma::accumulator, 16, 16, 16, float>
//    acc[2][2]; #pragma unroll for(int i=0; i<2; i++) for(int j=0; j<2; j++)
//    wmma::fill_fragment(acc[i][j], 0.0f);

//    auto load_tiles = [&](int ko, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          As[idx][r][c] = (blockIdx.y*BM+r < M && ko+c < K) ?
//          Ap[(blockIdx.y*BM+r)*s_am + (ko+c)*s_ak] : __float2bfloat16(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          Bs[idx][r][c] = (ko+r < K && blockIdx.x*BN+c < N) ? Bp[(ko+r)*s_bk +
//          (blockIdx.x*BN+c)*s_bn] : __float2bfloat16(0.0f);
//       }
//    };
//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int k = 0; k < K; k += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (k + BK < K) load_tiles(k + BK, wi);
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks],
//             BK+PAD); #pragma unroll for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16],
//                BN+PAD); wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
//    float* sm = reinterpret_cast<float*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
//       int cr = blockIdx.y*BM + warp_row*WM + i*16, cc = blockIdx.x*BN +
//       warp_col*WN + j*16; if (cr < M && cc < N) {
//          float* wsm = sm + warp_id * 256;
//          wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//          for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++)
//             if (cr+r < M && cc+c < N) Cp[(cr+r)*s_cm + (cc+c)*s_cn] =
//             __float2bfloat16(wsm[r*16+c]);
//       }
//    }
// }

// //
// ============================================================================
// // FP32 KERNEL (5 TFLOPS Version)
// // BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// //
// ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_fp32_optimized(const float* __restrict__ A, const
// float* __restrict__ B, float* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
//    int ao, bo, co;
//    compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
//    meta.out_shape, meta.out_ndim, ao); compute_batch_offset(b_idx,
//    meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim,
//    bo); compute_batch_offset(b_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const float *Ap = A +
//    ao, *Bp = B + bo; float *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk
//    = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1],
//    s_cm = meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];

//    __shared__ float As[2][BK][BM + PAD];
//    __shared__ float Bs[2][BK][BN + PAD];
//    float results[16] = {0.0f}, regM[4], regN[4];

//    auto load_tiles = [&](int ko, int idx) {
//       if (s_ak == 1 && s_bn == 1) { // Row-Major Contiguous
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / 16, c = li % 16;
//             As[idx][c][r] = (by*128+r < M && ko+c < K) ? Ap[(by*128+r)*s_am +
//             (ko+c)] : 0.0f;
//          }
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / 128, c = li % 128;
//             Bs[idx][r][c] = (ko+r < K && bx*128+c < N) ? Bp[(ko+r)*s_bk +
//             (bx*128+c)] : 0.0f;
//          }
//       } else { // Generic
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / BK, c = li % BK;
//             As[idx][c][r] = (by*BM+r < M && ko+c < K) ? Ap[(by*BM+r)*s_am +
//             (ko+c)*s_ak] : 0.0f;
//          }
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             int li = tid + i * 1024, r = li / BN, c = li % BN;
//             Bs[idx][r][c] = (ko+r < K && bx*BN+c < N) ? Bp[(ko+r)*s_bk +
//             (bx*BN+c)*s_bn] : 0.0f;
//          }
//       }
//    };

//    int wi = 0; load_tiles(0, wi); __syncthreads();
//    for (int bk = 0; bk < K; bk += BK) {
//       int ri = wi; wi = 1 - wi;
//       if (bk + BK < K) load_tiles(bk + BK, wi);
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
//          results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
//    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
//       int r = by*BM + tRow*4 + i, c = bx*BN + tCol*4 + j;
//       if (r < M && c < N) Cp[r*s_cm + c*s_cn] = results[i*4+j];
//    }
// }

// //
// ============================================================================
// // FP64 KERNEL (64x64x8 Tile, 256 Threads, 4x4 Tiling)
// //
// ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_fp64_optimized(const double* __restrict__ A, const
// double* __restrict__ B, double* __restrict__ C, int M, int N, int K, int
// total_batches, MatmulMetadata meta) {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tC = tid % (BN/TN), tR = tid / (BN/TN);
//    int ao, bo, co;
//    compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
//    meta.out_shape, meta.out_ndim, ao); compute_batch_offset(b_idx,
//    meta.b_shape, meta.b_strides, meta.b_ndim, meta.out_shape, meta.out_ndim,
//    bo); compute_batch_offset(b_idx, meta.out_shape, meta.out_strides,
//    meta.out_ndim, meta.out_shape, meta.out_ndim, co); const double *Ap = A +
//    ao, *Bp = B + bo; double *Cp = C + co; int s_am =
//    meta.a_strides[meta.a_ndim-2], s_ak = meta.a_strides[meta.a_ndim-1], s_bk
//    = meta.b_strides[meta.b_ndim-2], s_bn = meta.b_strides[meta.b_ndim-1],
//    s_cm = meta.out_strides[meta.out_ndim-2], s_cn =
//    meta.out_strides[meta.out_ndim-1];
//    __shared__ double As[BM*BK], Bs[BK*BN];
//    double res[16] = {0.0}, rM[4], rN[4];
//    for (int bk = 0; bk < K; bk += BK) {
//       for (int i = tid; i < BM*BK; i += 256) { int r = i/BK, c = i%BK; As[i]
//       = (by*BM+r < M && bk+c < K) ? Ap[(by*BM+r)*s_am + (bk+c)*s_ak] : 0.0; }
//       for (int i = tid; i < BK*BN; i += 256) { int r = i/BN, c = i%BN; Bs[i]
//       = (bk+r < K && bx*BN+c < N) ? Bp[(bk+r)*s_bk + (bk+c)*s_bn] : 0.0; }
//       __syncthreads();
//       for (int d = 0; d < BK; d++) {
//          for (int i = 0; i < 4; i++) rM[i] = As[(tR*4+i)*BK+d];
//          for (int i = 0; i < 4; i++) rN[i] = Bs[d*BN+tC*4+i];
//          for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res[i*4+j]
//          += rM[i] * rN[j];
//       }
//       __syncthreads();
//    }
//    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { int r =
//    by*BM+tR*4+i, c = bx*BN+tC*4+j; if (r < M && c < N) Cp[r*s_cm + c*s_cn] =
//    res[i*4+j]; }
// }

// //
// ============================================================================
// // DISPATCH LAYER
// //
// ============================================================================

// template<typename T>
// void launch_optimized_matmul(const Tensor& A, const Tensor& B, Tensor&
// output, cudaStream_t stream) {
//    AUTO_PROFILE_CUDA("Forward::Matmul_CUDA");
//    const auto& ash = A.shape().dims, &bsh = B.shape().dims, &osh =
//    output.shape().dims; int an = ash.size(), bn = bsh.size(), on =
//    osh.size(), M = ash[an-2], K = ash[an-1], N = bsh[bn-1], tb = 1; for (int
//    i = 0; i < on - 2; i++) tb *= osh[i]; MatmulMetadata meta; meta.a_ndim =
//    an; meta.b_ndim = bn; meta.out_ndim = on; for (int i = 0; i < an; i++) {
//    meta.a_shape[i] = ash[i]; meta.a_strides[i] = A.stride().strides[i]; } for
//    (int i = 0; i < bn; i++) { meta.b_shape[i] = bsh[i]; meta.b_strides[i] =
//    B.stride().strides[i]; } for (int i = 0; i < on; i++) { meta.out_shape[i]
//    = osh[i]; meta.out_strides[i] = output.stride().strides[i]; } const T* ap
//    = A.data<T>(), *bp = B.data<T>(); T* op = output.data<T>();

//    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T,
//    __half>::value) {
//       // Try cuBLAS first for FP16: compute in FP16, accumulate in FP32
//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool out_contiguous = (meta.out_strides[on-1] == 1);
//       bool is_supported = (a_contiguous || (an >= 2 && meta.a_strides[an-2]
//       == 1)) &&
//                           (b_contiguous || (bn >= 2 && meta.b_strides[bn-2]
//                           == 1)) && out_contiguous;
//       bool cublas_ok = false;
//       if (is_supported) {
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);
//          float alpha_f = 1.0f, beta_f = 0.0f;
//          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//          int lda = a_contiguous ? meta.a_strides[an-2] :
//          meta.a_strides[an-1]; cublasOperation_t opB = b_contiguous ?
//          CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//          meta.b_strides[bn-2] : meta.b_strides[bn-1]; int ldc =
//          meta.out_strides[on-2]; const __half* ap_h = reinterpret_cast<const
//          __half*>(ap); const __half* bp_h = reinterpret_cast<const
//          __half*>(bp);
//          __half* op_h = reinterpret_cast<__half*>(op);
//          cublasStatus_t status;
//          if (tb == 1) {
//             status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_h, CUDA_R_16F, ldb, ap_h, CUDA_R_16F, lda,
//                &beta_f, op_h, CUDA_R_16F, ldc,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          } else {
//             long long stride_a_ll = (an >= 3) ? (long
//             long)meta.a_strides[an-3] : 0LL; long long stride_b_ll = (bn >=
//             3) ? (long long)meta.b_strides[bn-3] : 0LL; long long stride_c_ll
//             = (long long)meta.out_strides[on-3]; status =
//             cublasGemmStridedBatchedEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_h, CUDA_R_16F, ldb, stride_b_ll, ap_h, CUDA_R_16F, lda,
//                stride_a_ll, &beta_f, op_h, CUDA_R_16F, ldc, stride_c_ll, tb,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          }
//          cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
//       }
//       if (!cublas_ok) {
//          matmul_fp16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128,
//          (M+127)/128, tb), 512, 0, stream>>>(
//             reinterpret_cast<const __half*>(ap), reinterpret_cast<const
//             __half*>(bp), reinterpret_cast<__half*>(op), M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T,
//    __nv_bfloat16>::value) {
//       // Try cuBLAS first for BF16: compute in BF16, accumulate in FP32
//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool out_contiguous = (meta.out_strides[on-1] == 1);
//       bool is_supported = (a_contiguous || (an >= 2 && meta.a_strides[an-2]
//       == 1)) &&
//                           (b_contiguous || (bn >= 2 && meta.b_strides[bn-2]
//                           == 1)) && out_contiguous;
//       bool cublas_ok = false;
//       if (is_supported) {
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);
//          float alpha_f = 1.0f, beta_f = 0.0f;
//          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//          int lda = a_contiguous ? meta.a_strides[an-2] :
//          meta.a_strides[an-1]; cublasOperation_t opB = b_contiguous ?
//          CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//          meta.b_strides[bn-2] : meta.b_strides[bn-1]; int ldc =
//          meta.out_strides[on-2]; const __nv_bfloat16* ap_b =
//          reinterpret_cast<const __nv_bfloat16*>(ap); const __nv_bfloat16*
//          bp_b = reinterpret_cast<const __nv_bfloat16*>(bp);
//          __nv_bfloat16* op_b = reinterpret_cast<__nv_bfloat16*>(op);
//          cublasStatus_t status;
//          if (tb == 1) {
//             status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_b, CUDA_R_16BF, ldb, ap_b, CUDA_R_16BF, lda,
//                &beta_f, op_b, CUDA_R_16BF, ldc,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          } else {
//             long long stride_a_ll = (an >= 3) ? (long
//             long)meta.a_strides[an-3] : 0LL; long long stride_b_ll = (bn >=
//             3) ? (long long)meta.b_strides[bn-3] : 0LL; long long stride_c_ll
//             = (long long)meta.out_strides[on-3]; status =
//             cublasGemmStridedBatchedEx(handle, opB, opA, N, M, K, &alpha_f,
//                bp_b, CUDA_R_16BF, ldb, stride_b_ll, ap_b, CUDA_R_16BF, lda,
//                stride_a_ll, &beta_f, op_b, CUDA_R_16BF, ldc, stride_c_ll, tb,
//                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//          }
//          cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
//       }
//       if (!cublas_ok) {
//          matmul_bf16_optimized<128, 128, 32, 32, 32><<<dim3((N+127)/128,
//          (M+127)/128, tb), 512, 0, stream>>>(
//             reinterpret_cast<const __nv_bfloat16*>(ap),
//             reinterpret_cast<const __nv_bfloat16*>(bp),
//             reinterpret_cast<__nv_bfloat16*>(op), M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, float>::value) {
//       // Use cuBLAS for FP32 matmuls - enables TF32 Tensor Core acceleration
//       // Check if tensors are contiguous (row-major) OR transposed (col-major
//       equivalent) for cuBLAS fast path

//       bool a_contiguous = (meta.a_strides[an-1] == 1);
//       bool a_transposed = (an >= 2 && meta.a_strides[an-2] == 1);

//       bool b_contiguous = (meta.b_strides[bn-1] == 1);
//       bool b_transposed = (bn >= 2 && meta.b_strides[bn-2] == 1);

//       bool out_contiguous = (meta.out_strides[on-1] == 1);

//       // Use cuBLAS for:
//       // 1. 2D matrices (tb=1)
//       // 2. Strided Batched matrices (tb>1)
//       bool is_supported_layout = (a_contiguous || a_transposed) &&
//       (b_contiguous || b_transposed) && out_contiguous; bool is_strided_batch
//       = false; long long stride_a = 0, stride_b = 0, stride_c = 0;

//       if (tb > 1 && is_supported_layout) {
//           // Calculate Strides for Batches
//           // For broadcasted dims (e.g. B is 2D but A is 3D), stride is 0.

//           // Helper to get batch stride check
//           // Assumption: dimensions 0..on-3 are the batch dimensions.
//           // Standard layout: Flattened batch.

//           if (on == 3) {
//              stride_c = meta.out_strides[0];
//              stride_a = (an == 3) ? meta.a_strides[0] : 0; // Broadcast if an
//              < 3 stride_b = (bn == 3) ? meta.b_strides[0] : 0; // Broadcast
//              if bn < 3 is_strided_batch = true;
//           } else if (on == 4) {
//             // 4D [d0, d1, M, N]: use inner batch stride (stride[1]) for
//             cuBLAS stride_c = meta.out_strides[on - 3]; stride_a = (an == 4)
//             ? meta.a_strides[an - 3] : 0; stride_b = (bn == 4) ?
//             meta.b_strides[bn - 3] : 0;

//             // cuBLAS strided batched requires uniform stride across all tb
//             batches.
//             // Uniform iff stride[0] == stride[1] * shape[1] for each tensor.
//             // Non-contiguous transposes (e.g. reshape+transpose without
//             contiguous)
//             // break this — handled by per-outer-batch loop below.
//             bool batch_uniform = (meta.out_strides[0] == meta.out_strides[1]
//             * osh[1]); if (an == 4) batch_uniform = batch_uniform &&
//             (meta.a_strides[0] == meta.a_strides[1] * ash[1]); if (bn == 4)
//             batch_uniform = batch_uniform && (meta.b_strides[0] ==
//             meta.b_strides[1] * bsh[1]);

//             if (batch_uniform) {
//                 is_strided_batch = true;
//             }
//             // else: non-uniform, handled by loop path below
//           }
//       }

//       if ((tb == 1 || is_strided_batch) && is_supported_layout) {
//          // USE EXPLICIT DEVICE INDEXING from output tensor
//          int dev_idx = output.device().index;
//          cublasHandle_t handle = get_cublas_handle(dev_idx);
//          cublasSetStream(handle, stream);

//          float alpha = 1.0f, beta = 0.0f;

//          // Setup A (Second in CuBLAS, since CuBLAS is ColMajor)
//          cublasOperation_t opA;
//          int lda;
//          if (a_contiguous) { opA = CUBLAS_OP_N; lda = meta.a_strides[an-2]; }
//          else { opA = CUBLAS_OP_T; lda = meta.a_strides[an-1]; }

//          // Setup B (First in CuBLAS)
//          cublasOperation_t opB;
//          int ldb;
//          if (b_contiguous) { opB = CUBLAS_OP_N; ldb = meta.b_strides[bn-2]; }
//          else { opB = CUBLAS_OP_T; ldb = meta.b_strides[bn-1]; }

//          int ldc = meta.out_strides[on-2];

//          // Logic correction:
//          // C = A @ B.
//          // CuBLAS (ColMajor): C^T = B^T @ A^T.
//          // Pass B as "A", A as "B".
//          //
//          // If A is contiguous RM [M, K]: Represents A^T [K, M] CM.
//          // We need A^T in the equation. So Op = N.
//          // If A is transposed RM [M, K] (stride 1, M): Represents A [K, M]
//          CM.
//          // We need A^T. So Op = T.
//          //
//          // If B is contiguous RM [K, N]: Represents B^T [N, K] CM.
//          // We need B^T in the equation. So Op = N.
//          // If B is transposed RM [K, N]: Represents B [N, K] CM.
//          // We need B^T. So Op = T.

//          cublasStatus_t status;

//          if (tb == 1) {
//              status = cublasGemmEx(
//                 handle,
//                 opB, opA,
//                 N, M, K,
//                 &alpha,
//                 bp, CUDA_R_32F, ldb,
//                 ap, CUDA_R_32F, lda,
//                 &beta,
//                 op, CUDA_R_32F, ldc,
//                 CUBLAS_COMPUTE_32F_FAST_TF32,
//                 CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//          } else {
//              status = cublasGemmStridedBatchedEx(
//                 handle,
//                 opB, opA,
//                 N, M, K,
//                 &alpha,
//                 bp, CUDA_R_32F, ldb, stride_b,
//                 ap, CUDA_R_32F, lda, stride_a,
//                 &beta,
//                 op, CUDA_R_32F, ldc, stride_c,
//                 tb,
//                 CUBLAS_COMPUTE_32F_FAST_TF32,
//                 CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//          }

//          if (status != CUBLAS_STATUS_SUCCESS) {
//             // Fallback (e.g. if alignment issues, though Ex handles most)
//              matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//              (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb,
//              meta);
//          }
//       } else if (on == 4 && !is_strided_batch && is_supported_layout) {
//          // Non-uniform 4D batch strides (e.g. after transpose without
//          contiguous):
//          // Loop over outer batch dim d0, each call handles d1 inner batches
//          with uniform stride int current_device;
//          cudaGetDevice(&current_device); cublasHandle_t handle =
//          get_cublas_handle(current_device); cublasSetStream(handle, stream);
//          float alpha_v = 1.0f, beta_v = 0.0f;

//          cublasOperation_t opA_l, opB_l;
//          int lda_l, ldb_l, ldc_l;
//          if (a_contiguous) { opA_l = CUBLAS_OP_N; lda_l =
//          meta.a_strides[an-2]; } else { opA_l = CUBLAS_OP_T; lda_l =
//          meta.a_strides[an-1]; } if (b_contiguous) { opB_l = CUBLAS_OP_N;
//          ldb_l = meta.b_strides[bn-2]; } else { opB_l = CUBLAS_OP_T; ldb_l =
//          meta.b_strides[bn-1]; } ldc_l = meta.out_strides[on-2];

//          int d0 = osh[0], d1 = osh[1];
//          bool all_ok = true;
//          for (int bi = 0; bi < d0 && all_ok; bi++) {
//              const float* a_b = ap + (an == 4 ? bi * meta.a_strides[0] : 0);
//              const float* b_b = bp + (bn == 4 ? bi * meta.b_strides[0] : 0);
//              float* o_b = op + bi * meta.out_strides[0];

//              cublasStatus_t st = cublasGemmStridedBatchedEx(
//                  handle, opB_l, opA_l, N, M, K, &alpha_v,
//                  b_b, CUDA_R_32F, ldb_l, stride_b,
//                  a_b, CUDA_R_32F, lda_l, stride_a,
//                  &beta_v,
//                  o_b, CUDA_R_32F, ldc_l, stride_c,
//                  d1,
//                  CUBLAS_COMPUTE_32F_FAST_TF32,
//                  CUBLAS_GEMM_DEFAULT_TENSOR_OP
//              );
//              if (st != CUBLAS_STATUS_SUCCESS) all_ok = false;
//          }
//          if (!all_ok) {
//              matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//              (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb,
//              meta);
//          }
//       } else {
//          // Fallback to custom kernel
//          matmul_fp32_optimized<128, 128, 16, 4, 4><<<dim3((N+127)/128,
//          (M+127)/128, tb), 1024, 0, stream>>>(ap, bp, op, M, N, K, tb, meta);
//       }
//    } else if constexpr (std::is_same<T, double>::value) {
//       matmul_fp64_optimized<64, 64, 8, 4, 4><<<dim3((N+63)/64, (M+63)/64,
//       tb), 256, 0, stream>>>(reinterpret_cast<const double*>(ap),
//       reinterpret_cast<const double*>(bp), reinterpret_cast<double*>(op), M,
//       N, K, tb, meta);
//    }

//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) throw std::runtime_error("Kernel failed: " +
//    std::string(cudaGetErrorString(err)));
// }

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output,
// cudaStream_t stream) {
//    dispatch_by_dtype(A.dtype(), [&](auto d) {
//       using T = decltype(d);
//       if constexpr (std::is_same<T, float16_t>::value || std::is_same<T,
//       bfloat16_t>::value || std::is_same<T, float>::value || std::is_same<T,
//       double>::value) launch_optimized_matmul<T>(A, B, output, stream); else
//       throw std::runtime_error("Unsupported type");
//    });
// }

// void cuda_addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
// float alpha, float beta, Tensor& output, cudaStream_t stream) {
//     AUTO_PROFILE_CUDA("Forward::Addmm_CUDA");
//     const auto& m1sh = mat1.shape().dims, &m2sh = mat2.shape().dims, &osh =
//     output.shape().dims; int m1n = m1sh.size(), m2n = m2sh.size(), on =
//     osh.size(); int M = m1sh[m1n-2], K = m1sh[m1n-1], N = m2sh[m2n-1];

//     // Compute total batch count from output shape (all dims except last 2)
//     int tb = 1;
//     for (int i = 0; i < on - 2; i++) tb *= osh[i];
//     int64_t total_output_elements = output.numel(); // tb * M * N

//     // 1. Prepare output buffer: output = beta * input (with broadcasting)
//     //    Scale ALL elements of the output, not just M*N
//     if (beta != 0.0f) {
//         dispatch_by_dtype(output.dtype(), [&](auto d) {
//             using T = decltype(d);
//             if constexpr (std::is_same_v<T, float> || std::is_same_v<T,
//             double> || std::is_same_v<T, __half> || std::is_same_v<T,
//             __nv_bfloat16> || std::is_integral_v<T>) {
//                 // Use total_output_elements as both rows*cols and total
//                 // M_total and N stay the same for broadcasting logic (bias
//                 is broadcast per [M,N] slice) int64_t rows_total =
//                 (int64_t)tb * M; launch_broadcast_scale<T>(output.data<T>(),
//                 input.data<T>(), beta, rows_total, N, input.numel(), stream);
//             }
//         });
//     } else {
//         // If beta is 0, input is ignored, zero the entire output
//         dispatch_by_dtype(output.dtype(), [&](auto d) {
//             using T = decltype(d);
//             if constexpr (std::is_same_v<T, float> || std::is_same_v<T,
//             double> || std::is_same_v<T, __half> || std::is_same_v<T,
//             __nv_bfloat16> || std::is_integral_v<T>) {
//                 int threads = 256;
//                 int blocks = (total_output_elements + threads - 1) / threads;
//                 // Simple zero kernel
//                 broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(
//                     output.data<T>(), output.data<T>(), 0.0f,
//                     total_output_elements, total_output_elements, (int64_t)tb
//                     * M, N);
//             }
//         });
//     }

//     // 2. Add alpha * (mat1 @ mat2) using cuBLAS
//     if (output.dtype() == Dtype::Float32) {
//         MatmulMetadata meta; meta.a_ndim = m1n; meta.b_ndim = m2n;
//         meta.out_ndim = on; for (int i = 0; i < m1n; i++) { meta.a_shape[i] =
//         m1sh[i]; meta.a_strides[i] = mat1.stride().strides[i]; } for (int i =
//         0; i < m2n; i++) { meta.b_shape[i] = m2sh[i]; meta.b_strides[i] =
//         mat2.stride().strides[i]; } for (int i = 0; i < on; i++) {
//         meta.out_shape[i] = osh[i]; meta.out_strides[i] =
//         output.stride().strides[i]; }

//         bool a_contiguous = (meta.a_strides[m1n-1] == 1);
//         bool a_transposed = (m1n >= 2 && meta.a_strides[m1n-2] == 1);
//         bool b_contiguous = (meta.b_strides[m2n-1] == 1);
//         bool b_transposed = (m2n >= 2 && meta.b_strides[m2n-2] == 1);
//         bool out_contiguous = (meta.out_strides[on-1] == 1);

//         bool is_supported_layout = (a_contiguous || a_transposed) &&
//         (b_contiguous || b_transposed) && out_contiguous;

//         // Compute batch strides (mirroring launch_optimized_matmul)
//         bool is_strided_batch = false;
//         long long stride_a = 0, stride_b = 0, stride_c = 0;

//         if (tb > 1 && is_supported_layout) {
//             if (on == 3) {
//                 stride_c = meta.out_strides[0];
//                 stride_a = (m1n == 3) ? meta.a_strides[0] : 0;
//                 stride_b = (m2n == 3) ? meta.b_strides[0] : 0;
//                 is_strided_batch = true;
//             } else if (on == 4) {
//                 stride_c = static_cast<long long>(M) * N;
//                 stride_a = (m1n == 4) ? static_cast<long long>(M) * K : 0;
//                 stride_b = (m2n == 4) ? static_cast<long long>(K) * N : 0;
//                 is_strided_batch = true;
//             }
//         }

//         if ((tb == 1 || is_strided_batch) && is_supported_layout) {
//             // USE EXPLICIT DEVICE INDEXING from output tensor
//             int dev_idx = output.device().index;
//             cublasHandle_t handle = get_cublas_handle(dev_idx);
//             cublasSetStream(handle, stream);

//             cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//             int lda = a_contiguous ? meta.a_strides[m1n-2] :
//             meta.a_strides[m1n-1]; cublasOperation_t opB = b_contiguous ?
//             CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//             meta.b_strides[m2n-2] : meta.b_strides[m2n-1]; int ldc =
//             meta.out_strides[on-2];

//             // output already has beta * input, so we pass cublas_beta = 1.0
//             to accumulate float cublas_beta = 1.0f;

//             cublasStatus_t status;

//             if (tb == 1) {
//                 status = cublasGemmEx(
//                     handle, opB, opA, N, M, K, &alpha,
//                     mat2.data<float>(), CUDA_R_32F, ldb,
//                     mat1.data<float>(), CUDA_R_32F, lda,
//                     &cublas_beta, output.data<float>(), CUDA_R_32F, ldc,
//                     CUBLAS_COMPUTE_32F_FAST_TF32,
//                     CUBLAS_GEMM_DEFAULT_TENSOR_OP
//                 );
//             } else {
//                 status = cublasGemmStridedBatchedEx(
//                     handle, opB, opA, N, M, K, &alpha,
//                     mat2.data<float>(), CUDA_R_32F, ldb, stride_b,
//                     mat1.data<float>(), CUDA_R_32F, lda, stride_a,
//                     &cublas_beta, output.data<float>(), CUDA_R_32F, ldc,
//                     stride_c, tb, CUBLAS_COMPUTE_32F_FAST_TF32,
//                     CUBLAS_GEMM_DEFAULT_TENSOR_OP
//                 );
//             }

//             if (status == CUBLAS_STATUS_SUCCESS) return;
//             // Fall through to fallback on failure
//         }
//     } else if (output.dtype() == Dtype::Float16 || output.dtype() ==
//     Dtype::Bfloat16) {
//         // cuBLAS path for FP16/BF16 addmm: compute in fp16/bf16, accumulate
//         in fp32 MatmulMetadata meta16; meta16.a_ndim = m1n; meta16.b_ndim =
//         m2n; meta16.out_ndim = on; for (int i = 0; i < m1n; i++) {
//         meta16.a_shape[i] = m1sh[i]; meta16.a_strides[i] =
//         mat1.stride().strides[i]; } for (int i = 0; i < m2n; i++) {
//         meta16.b_shape[i] = m2sh[i]; meta16.b_strides[i] =
//         mat2.stride().strides[i]; } for (int i = 0; i < on; i++) {
//         meta16.out_shape[i] = osh[i]; meta16.out_strides[i] =
//         output.stride().strides[i]; }

//         bool a_contiguous = (meta16.a_strides[m1n-1] == 1);
//         bool b_contiguous = (meta16.b_strides[m2n-1] == 1);
//         bool out_contiguous = (meta16.out_strides[on-1] == 1);
//         bool is_supported = (a_contiguous || (m1n >= 2 &&
//         meta16.a_strides[m1n-2] == 1)) &&
//                             (b_contiguous || (m2n >= 2 &&
//                             meta16.b_strides[m2n-2] == 1)) && out_contiguous;

//         if (is_supported) {
//             int dev_idx = output.device().index;
//             cublasHandle_t handle = get_cublas_handle(dev_idx);
//             cublasSetStream(handle, stream);

//             cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
//             int lda = a_contiguous ? meta16.a_strides[m1n-2] :
//             meta16.a_strides[m1n-1]; cublasOperation_t opB = b_contiguous ?
//             CUBLAS_OP_N : CUBLAS_OP_T; int ldb = b_contiguous ?
//             meta16.b_strides[m2n-2] : meta16.b_strides[m2n-1]; int ldc =
//             meta16.out_strides[on-2];
//             // output already holds beta*input; pass cublas_beta=1.0 to
//             accumulate alpha*(mat1@mat2) float cublas_beta = 1.0f;

//             bool is_strided_batch = (tb > 1) && (on == 3 || on == 4);
//             long long stride_a_ll = 0, stride_b_ll = 0, stride_c_ll = 0;
//             if (is_strided_batch) {
//                 stride_c_ll = meta16.out_strides[on-3];
//                 stride_a_ll = (m1n >= on) ? meta16.a_strides[m1n-3] : 0LL;
//                 stride_b_ll = (m2n >= on) ? meta16.b_strides[m2n-3] : 0LL;
//             }

//             cudaDataType_t cuda_type = (output.dtype() == Dtype::Float16) ?
//             CUDA_R_16F : CUDA_R_16BF; cublasStatus_t status;

//             if (output.dtype() == Dtype::Float16) {
//                 const __half* m1h = reinterpret_cast<const
//                 __half*>(mat1.data<float16_t>()); const __half* m2h =
//                 reinterpret_cast<const __half*>(mat2.data<float16_t>());
//                 __half* outh =
//                 reinterpret_cast<__half*>(output.data<float16_t>()); if (tb
//                 == 1) {
//                     status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha,
//                         m2h, CUDA_R_16F, ldb, m1h, CUDA_R_16F, lda,
//                         &cublas_beta, outh, CUDA_R_16F, ldc,
//                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 } else {
//                     status = cublasGemmStridedBatchedEx(handle, opB, opA, N,
//                     M, K, &alpha,
//                         m2h, CUDA_R_16F, ldb, stride_b_ll, m1h, CUDA_R_16F,
//                         lda, stride_a_ll, &cublas_beta, outh, CUDA_R_16F,
//                         ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
//                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 }
//             } else {
//                 const __nv_bfloat16* m1b = reinterpret_cast<const
//                 __nv_bfloat16*>(mat1.data<bfloat16_t>()); const
//                 __nv_bfloat16* m2b = reinterpret_cast<const
//                 __nv_bfloat16*>(mat2.data<bfloat16_t>());
//                 __nv_bfloat16* outb =
//                 reinterpret_cast<__nv_bfloat16*>(output.data<bfloat16_t>());
//                 if (tb == 1) {
//                     status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha,
//                         m2b, CUDA_R_16BF, ldb, m1b, CUDA_R_16BF, lda,
//                         &cublas_beta, outb, CUDA_R_16BF, ldc,
//                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 } else {
//                     status = cublasGemmStridedBatchedEx(handle, opB, opA, N,
//                     M, K, &alpha,
//                         m2b, CUDA_R_16BF, ldb, stride_b_ll, m1b, CUDA_R_16BF,
//                         lda, stride_a_ll, &cublas_beta, outb, CUDA_R_16BF,
//                         ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
//                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                 }
//             }
//             if (status == CUBLAS_STATUS_SUCCESS) return;
//             // Fall through to fallback on failure
//         }
//     }

//     // Fallback: Use existing batched matmul + scaled addition kernel
//     Tensor temp_matmul = OwnTensor::matmul(mat1, mat2, stream);

//     dispatch_by_dtype(output.dtype(), [&](auto d) {
//         using T = decltype(d);
//         if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>
//         || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16> ||
//         std::is_integral_v<T>) {
//             int64_t total = output.numel();
//             int threads = 256;
//             int blocks = (total + threads - 1) / threads;
//             add_scaled_kernel_typed<T><<<blocks, threads, 0,
//             stream>>>(output.data<T>(), temp_matmul.data<T>(), alpha, total);
//         }
//     });
// }

// } // namespace OwnTensor
// // #endif
// #endif // WITH_MYBLAS

#ifdef WITH_MYBLAS

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stdio.h>

#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "mycublas.h"
#include "ops/Matmul.cuh"
#include "utils/Profiler.h"

namespace OwnTensor {

// Thread-safe cached handle — created once per device, never destroyed.
// Same pattern as MatmulBackward.cu. Eliminates per-call Create/Destroy
// overhead.
static std::mutex g_fwd_mutex;
static mycublasHandle_t g_mycublas_handles_fwd[8] = {nullptr};

static mycublasHandle_t get_mycublas_handle_fwd(int device = 0) {
  if (g_mycublas_handles_fwd[device] == nullptr) {
    std::lock_guard<std::mutex> lock(g_fwd_mutex);
    if (g_mycublas_handles_fwd[device] == nullptr) {
      cudaSetDevice(device);
      mycublasCreate(&g_mycublas_handles_fwd[device]);
    }
  }
  return g_mycublas_handles_fwd[device];
}

// ============================================================================
// DISPATCH LAYER - PURE MYBLAS
// ============================================================================

void cuda_matmul(const Tensor &A, const Tensor &B, Tensor &output,
                 cudaStream_t stream) {
  AUTO_PROFILE_CUDA("Forward::Matmul_CUDA_MyBlas");

  const auto &a_sh = A.shape().dims;
  const auto &b_sh = B.shape().dims;
  const auto &o_sh = output.shape().dims;
  int a_ndim = (int)a_sh.size();
  int b_ndim = (int)b_sh.size();

  // Accept row-major (strides[-1]==1) or col-major/transposed (strides[-2]==1).
  // Anything else (e.g. arbitrary non-contiguous) falls back to an explicit
  // copy.
  bool a_row = (A.stride().strides[a_ndim - 1] == 1);
  bool a_col = (a_ndim >= 2 && A.stride().strides[a_ndim - 2] == 1);
  bool b_row = (B.stride().strides[b_ndim - 1] == 1);
  bool b_col = (b_ndim >= 2 && B.stride().strides[b_ndim - 2] == 1);

  if (!(a_row || a_col) || !(b_row || b_col)) {
    const Tensor A_c = A.is_contiguous() ? A : A.contiguous();
    const Tensor B_c = B.is_contiguous() ? B : B.contiguous();
    cuda_matmul(A_c, B_c, output, stream);
    return;
  }

  int M = (int)a_sh[a_ndim - 2];
  int K = (int)a_sh[a_ndim - 1];
  int N = (int)b_sh[b_ndim - 1];

  // row-major: OP_N, ld = strides[-2]  |  col-major: OP_T, ld = strides[-1]
  mycublasOperation_t opA = a_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
  int lda =
      a_row ? A.stride().strides[a_ndim - 2] : A.stride().strides[a_ndim - 1];
  mycublasOperation_t opB = b_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
  int ldb =
      b_row ? B.stride().strides[b_ndim - 2] : B.stride().strides[b_ndim - 1];

  int batch_count = 1;
  for (int i = 0; i < (int)o_sh.size() - 2; ++i)
    batch_count *= (int)o_sh[i];

  // Use actual tensor batch stride (handles non-default layouts correctly)
  long long strideA =
      (A.numel() == (size_t)M * K)
          ? 0LL
          : (a_ndim >= 3 ? (long long)A.stride().strides[a_ndim - 3]
                         : (long long)M * K);
  long long strideB =
      (B.numel() == (size_t)K * N)
          ? 0LL
          : (b_ndim >= 3 ? (long long)B.stride().strides[b_ndim - 3]
                         : (long long)K * N);
  long long strideC = (long long)M * N;

  int dev = 0;
  cudaGetDevice(&dev);
  mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
  mycublasSetStream(handle, stream);

  dispatch_by_dtype(A.dtype(), [&](auto dummy) {
    using T = decltype(dummy);
    const T *a_ptr = A.data<T>();
    const T *b_ptr = B.data<T>();
    T *out_ptr = output.data<T>();

    if constexpr (std::is_same<T, float>::value) {
      mycublasSgemmStridedBatched(handle, opA, opB, M, N, K, 1.0f, a_ptr, lda,
                                  strideA, b_ptr, ldb, strideB, 0.0f, out_ptr,
                                  N, strideC, batch_count);
    } else if constexpr (std::is_same<T, __half>::value ||
                         std::is_same<T, OwnTensor::float16_t>::value) {
      __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
      mycublasHgemmStridedBatched(handle, opA, opB, M, N, K, alpha,
                                  (const __half *)a_ptr, lda, strideA,
                                  (const __half *)b_ptr, ldb, strideB, beta,
                                  (__half *)out_ptr, N, strideC, batch_count);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value ||
                         std::is_same<T, OwnTensor::bfloat16_t>::value) {
      __nv_bfloat16 alpha = __float2bfloat16(1.0f),
                    beta = __float2bfloat16(0.0f);
      mycublasBgemmStridedBatched(
          handle, opA, opB, M, N, K, alpha, (const __nv_bfloat16 *)a_ptr, lda,
          strideA, (const __nv_bfloat16 *)b_ptr, ldb, strideB, beta,
          (__nv_bfloat16 *)out_ptr, N, strideC, batch_count);
    } else if constexpr (std::is_same<T, double>::value) {
      // double: no op-flag support in mycublasDgemm, fall back to contiguous
      // copy
      const Tensor A_c = A.is_contiguous() ? A : A.contiguous();
      const Tensor B_c = B.is_contiguous() ? B : B.contiguous();
      long long sa = (A_c.numel() == (size_t)M * K) ? 0LL : (long long)M * K;
      long long sb = (B_c.numel() == (size_t)K * N) ? 0LL : (long long)K * N;
      mycublasDgemmStridedBatched(handle, M, N, K, 1.0,
                                  (const double *)A_c.data<T>(), K, sa,
                                  (const double *)B_c.data<T>(), N, sb, 0.0,
                                  (double *)out_ptr, N, strideC, batch_count);
    } else {
      throw std::runtime_error(
          "cuda_matmul: Unsupported dtype for MyBlas GEMM.");
    }
  });

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA Error after MyBlas matmul launch: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// ============================================================================
// ADDMM IMPLEMENTATION - USING MYBLAS
// ============================================================================

// Helper kernel for broadcasting and scaling
template <typename T>
__global__ void
broadcast_add_scaled_kernel(T *out, const T *in, const T *bias, float alpha,
                            float beta, int M, int N, int batch_count,
                            int64_t stride_c, int64_t bias_numel) {
  int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  int64_t total = (int64_t)batch_count * M * N;
  if (idx < total) {
    T b_val;
    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      b_val = T(0.0f, 0.0f);
    } else {
      b_val = (T)0.0f;
    }

    if (bias_numel == 1)
      b_val = bias[0];
    else if (bias_numel == N)
      b_val = bias[idx % N];
    else if (bias_numel == total)
      b_val = bias[idx];

    if constexpr (std::is_same_v<T, complex32_t> ||
                  std::is_same_v<T, complex64_t> ||
                  std::is_same_v<T, complex128_t>) {
      out[idx] = out[idx] + b_val * (T)beta;
    } else {
      out[idx] = out[idx] + (T)(beta * (float)b_val);
    }
  }
}

void cuda_addmm(const Tensor &input, const Tensor &mat1, const Tensor &mat2,
                float alpha, float beta, Tensor &output, cudaStream_t stream) {
  AUTO_PROFILE_CUDA("Forward::Addmm_CUDA_MyBlas");

  if (output.dtype() == Dtype::Float32) {
    const auto &o_sh = output.shape().dims;
    const auto &m1_sh = mat1.shape().dims;
    const auto &m2_sh = mat2.shape().dims;
    int m1_ndim = (int)m1_sh.size();
    int m2_ndim = (int)m2_sh.size();

    int M = (int)o_sh[o_sh.size() - 2];
    int N = (int)o_sh[o_sh.size() - 1];
    int K = (int)m1_sh[m1_ndim - 1];
    int batch_count = 1;
    for (size_t i = 0; i < o_sh.size() - 2; ++i)
      batch_count *= (int)o_sh[i];

    bool m1_row = (mat1.stride().strides[m1_ndim - 1] == 1);
    bool m1_col = (m1_ndim >= 2 && mat1.stride().strides[m1_ndim - 2] == 1);
    bool m2_row = (mat2.stride().strides[m2_ndim - 1] == 1);
    bool m2_col = (m2_ndim >= 2 && mat2.stride().strides[m2_ndim - 2] == 1);

    int dev = 0;
    cudaGetDevice(&dev);
    mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
    mycublasSetStream(handle, stream);

    // Fast path: both row-major → use the specialized fused SgemmAddmm kernel
    if (m1_row && m2_row) {
      int lda = (int)K;
      int ldb = (int)N;
      long long strideA =
          (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K;
      long long strideB =
          (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N;
      mycublasSgemmAddmm(handle, M, N, K, alpha, mat1.data<float>(), lda,
                         strideA, mat2.data<float>(), ldb, strideB, beta,
                         input.data<float>(), (int64_t)input.numel(),
                         output.data<float>(), N, (long long)M * N,
                         batch_count);
      return;
    }

    // General path: mat1 or mat2 is transposed (e.g. x @ W.T).
    // Use layout-aware GEMM + separate bias-add. Avoids the explicit contiguous
    // copy.
    if ((m1_row || m1_col) && (m2_row || m2_col)) {
      mycublasOperation_t opA = m1_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int lda = m1_row ? mat1.stride().strides[m1_ndim - 2]
                       : mat1.stride().strides[m1_ndim - 1];
      mycublasOperation_t opB = m2_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int ldb = m2_row ? mat2.stride().strides[m2_ndim - 2]
                       : mat2.stride().strides[m2_ndim - 1];

      long long strideA =
          (mat1.numel() == (size_t)M * K)
              ? 0LL
              : (m1_ndim >= 3 ? (long long)mat1.stride().strides[m1_ndim - 3]
                              : (long long)M * K);
      long long strideB =
          (mat2.numel() == (size_t)K * N)
              ? 0LL
              : (m2_ndim >= 3 ? (long long)mat2.stride().strides[m2_ndim - 3]
                              : (long long)K * N);

      mycublasSgemmStridedBatched(
          handle, opA, opB, M, N, K, alpha, mat1.data<float>(), lda, strideA,
          mat2.data<float>(), ldb, strideB, 0.0f, output.data<float>(), N,
          (long long)M * N, batch_count);

      // Add bias: output += beta * input (broadcast)
      int64_t total = output.numel();
      int threads = 256, blocks = (total + threads - 1) / threads;
      broadcast_add_scaled_kernel<float><<<blocks, threads, 0, stream>>>(
          output.data<float>(), nullptr, input.data<float>(), alpha, beta, M, N,
          batch_count, (int64_t)M * N, input.numel());
      return;
    }

    // True non-contiguous fallback (rare)
    const Tensor m1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    const Tensor m2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    mycublasSgemmAddmm(handle, M, N, K, alpha, m1_c.data<float>(), K,
                       (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K,
                       m2_c.data<float>(), N,
                       (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N,
                       beta, input.data<float>(), (int64_t)input.numel(),
                       output.data<float>(), N, (long long)M * N, batch_count);
    return;
  }

  if (output.dtype() == Dtype::Float16) {
    const auto &o_sh = output.shape().dims;
    const auto &m1_sh = mat1.shape().dims;
    const auto &m2_sh = mat2.shape().dims;
    int m1_ndim = (int)m1_sh.size();
    int m2_ndim = (int)m2_sh.size();

    int M = (int)o_sh[o_sh.size() - 2];
    int N = (int)o_sh[o_sh.size() - 1];
    int K = (int)m1_sh[m1_ndim - 1];
    int batch_count = 1;
    for (size_t i = 0; i < o_sh.size() - 2; ++i)
      batch_count *= (int)o_sh[i];

    bool m1_row = (mat1.stride().strides[m1_ndim - 1] == 1);
    bool m1_col = (m1_ndim >= 2 && mat1.stride().strides[m1_ndim - 2] == 1);
    bool m2_row = (mat2.stride().strides[m2_ndim - 1] == 1);
    bool m2_col = (m2_ndim >= 2 && mat2.stride().strides[m2_ndim - 2] == 1);

    int dev = 0;
    cudaGetDevice(&dev);
    mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
    mycublasSetStream(handle, stream);

    const __half *m1h =
        reinterpret_cast<const __half *>(mat1.data<float16_t>());
    const __half *m2h =
        reinterpret_cast<const __half *>(mat2.data<float16_t>());
    __half *outh = reinterpret_cast<__half *>(output.data<float16_t>());
    const __half *bias_h =
        reinterpret_cast<const __half *>(input.data<float16_t>());

    __half alpha_h = __float2half(alpha);
    __half beta_h = __float2half(beta);

    // Fast path: both row-major → use the specialized fused HgemmAddmm kernel
    if (m1_row && m2_row) {
      int lda = (int)K;
      int ldb = (int)N;
      long long strideA =
          (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K;
      long long strideB =
          (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N;
      mycublasHgemmAddmm_sm89(handle, M, N, K, alpha_h, m1h, lda, strideA, m2h,
                              ldb, strideB, beta_h, outh, N, (long long)M * N,
                              bias_h, (int64_t)input.numel(), batch_count);
      return;
    }

    // General path: mat1 or mat2 is transposed (e.g. x @ W.T).
    // Use layout-aware GEMM + separate bias-add. Avoids the explicit contiguous
    // copy.
    if ((m1_row || m1_col) && (m2_row || m2_col)) {
      mycublasOperation_t opA = m1_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int lda = m1_row ? mat1.stride().strides[m1_ndim - 2]
                       : mat1.stride().strides[m1_ndim - 1];
      mycublasOperation_t opB = m2_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int ldb = m2_row ? mat2.stride().strides[m2_ndim - 2]
                       : mat2.stride().strides[m2_ndim - 1];

      long long strideA =
          (mat1.numel() == (size_t)M * K)
              ? 0LL
              : (m1_ndim >= 3 ? (long long)mat1.stride().strides[m1_ndim - 3]
                              : (long long)M * K);
      long long strideB =
          (mat2.numel() == (size_t)K * N)
              ? 0LL
              : (m2_ndim >= 3 ? (long long)mat2.stride().strides[m2_ndim - 3]
                              : (long long)K * N);

      mycublasHgemmStridedBatched_dispatcher(
          handle, opA, opB, M, N, K, alpha_h, m1h, lda, strideA, m2h, ldb,
          strideB, __float2half(0.0f), outh, N, (long long)M * N, batch_count);

      // Add bias: output += beta * input (broadcast)
      int64_t total = output.numel();
      int threads = 256, blocks = (total + threads - 1) / threads;
      broadcast_add_scaled_kernel<float16_t><<<blocks, threads, 0, stream>>>(
          output.data<float16_t>(), nullptr, input.data<float16_t>(), alpha,
          beta, M, N, batch_count, (int64_t)M * N, input.numel());
      return;
    }

    // True non-contiguous fallback (rare)
    const Tensor m1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    const Tensor m2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    const __half *m1c_h =
        reinterpret_cast<const __half *>(m1_c.data<float16_t>());
    const __half *m2c_h =
        reinterpret_cast<const __half *>(m2_c.data<float16_t>());
    mycublasHgemmAddmm_sm89(
        handle, M, N, K, alpha_h, m1c_h, K,
        (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K, m2c_h, N,
        (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N, beta_h, outh,
        N, (long long)M * N, bias_h, (int64_t)input.numel(), batch_count);
    return;
  }

  if (output.dtype() == Dtype::Bfloat16) {
    const auto &o_sh = output.shape().dims;
    const auto &m1_sh = mat1.shape().dims;
    const auto &m2_sh = mat2.shape().dims;
    int m1_ndim = (int)m1_sh.size();
    int m2_ndim = (int)m2_sh.size();

    int M = (int)o_sh[o_sh.size() - 2];
    int N = (int)o_sh[o_sh.size() - 1];
    int K = (int)m1_sh[m1_ndim - 1];
    int batch_count = 1;
    for (size_t i = 0; i < o_sh.size() - 2; ++i)
      batch_count *= (int)o_sh[i];

    bool m1_row = (mat1.stride().strides[m1_ndim - 1] == 1);
    bool m1_col = (m1_ndim >= 2 && mat1.stride().strides[m1_ndim - 2] == 1);
    bool m2_row = (mat2.stride().strides[m2_ndim - 1] == 1);
    bool m2_col = (m2_ndim >= 2 && mat2.stride().strides[m2_ndim - 2] == 1);

    int dev = 0;
    cudaGetDevice(&dev);
    mycublasHandle_t handle = get_mycublas_handle_fwd(dev);
    mycublasSetStream(handle, stream);

    const __nv_bfloat16 *m1b =
        reinterpret_cast<const __nv_bfloat16 *>(mat1.data<bfloat16_t>());
    const __nv_bfloat16 *m2b =
        reinterpret_cast<const __nv_bfloat16 *>(mat2.data<bfloat16_t>());
    __nv_bfloat16 *outb =
        reinterpret_cast<__nv_bfloat16 *>(output.data<bfloat16_t>());
    const __nv_bfloat16 *bias_b =
        reinterpret_cast<const __nv_bfloat16 *>(input.data<bfloat16_t>());

    __nv_bfloat16 alpha_b = __float2bfloat16(alpha);
    __nv_bfloat16 beta_b = __float2bfloat16(beta);

    // Fast path: both row-major → use the specialized fused BgemmAddmm kernel
    if (m1_row && m2_row) {
      int lda = (int)K;
      int ldb = (int)N;
      long long strideA =
          (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K;
      long long strideB =
          (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N;
      mycublasBgemmAddmm_sm89(handle, M, N, K, alpha_b, m1b, lda, strideA, m2b,
                              ldb, strideB, beta_b, outb, N, (long long)M * N,
                              bias_b, (int64_t)input.numel(), batch_count);
      return;
    }

    // General path: mat1 or mat2 is transposed (e.g. x @ W.T).
    // Use layout-aware GEMM + separate bias-add. Avoids the explicit contiguous
    // copy.
    if ((m1_row || m1_col) && (m2_row || m2_col)) {
      mycublasOperation_t opA = m1_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int lda = m1_row ? mat1.stride().strides[m1_ndim - 2]
                       : mat1.stride().strides[m1_ndim - 1];
      mycublasOperation_t opB = m2_row ? MYCUBLAS_OP_N : MYCUBLAS_OP_T;
      int ldb = m2_row ? mat2.stride().strides[m2_ndim - 2]
                       : mat2.stride().strides[m2_ndim - 1];

      long long strideA =
          (mat1.numel() == (size_t)M * K)
              ? 0LL
              : (m1_ndim >= 3 ? (long long)mat1.stride().strides[m1_ndim - 3]
                              : (long long)M * K);
      long long strideB =
          (mat2.numel() == (size_t)K * N)
              ? 0LL
              : (m2_ndim >= 3 ? (long long)mat2.stride().strides[m2_ndim - 3]
                              : (long long)K * N);

      mycublasBgemmStridedBatched_dispatcher(
          handle, opA, opB, M, N, K, alpha_b, m1b, lda, strideA, m2b, ldb,
          strideB, __float2bfloat16(0.0f), outb, N, (long long)M * N,
          batch_count);

      // Add bias: output += beta * input (broadcast)
      int64_t total = output.numel();
      int threads = 256, blocks = (total + threads - 1) / threads;
      broadcast_add_scaled_kernel<bfloat16_t><<<blocks, threads, 0, stream>>>(
          output.data<bfloat16_t>(), nullptr, input.data<bfloat16_t>(), alpha,
          beta, M, N, batch_count, (int64_t)M * N, input.numel());
      return;
    }

    // True non-contiguous fallback (rare)
    const Tensor m1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    const Tensor m2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    const __nv_bfloat16 *m1c_b =
        reinterpret_cast<const __nv_bfloat16 *>(m1_c.data<bfloat16_t>());
    const __nv_bfloat16 *m2c_b =
        reinterpret_cast<const __nv_bfloat16 *>(m2_c.data<bfloat16_t>());
    mycublasBgemmAddmm_sm89(
        handle, M, N, K, alpha_b, m1c_b, K,
        (mat1.numel() == (size_t)M * K) ? 0LL : (long long)M * K, m2c_b, N,
        (mat2.numel() == (size_t)K * N) ? 0LL : (long long)K * N, beta_b, outb,
        N, (long long)M * N, bias_b, (int64_t)input.numel(), batch_count);
    return;
  }

  // Fallback path: route through cuda_matmul (which handles transposed inputs),
  // then add the bias. No forced contiguous copy needed.
  cuda_matmul(mat1, mat2, output, stream);

  const auto &out_shape = output.shape().dims;
  int M = out_shape[out_shape.size() - 2];
  int N = out_shape[out_shape.size() - 1];
  int batch_count = 1;
  for (size_t i = 0; i < out_shape.size() - 2; ++i)
    batch_count *= static_cast<int>(out_shape[i]);

  int64_t total = output.numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  dispatch_by_dtype(output.dtype(), [&](auto dummy) {
    using T = decltype(dummy);
    broadcast_add_scaled_kernel<T><<<blocks, threads, 0, stream>>>(
        output.data<T>(), nullptr, input.data<T>(), alpha, beta, M, N,
        batch_count, (int64_t)M * N, input.numel());
  });
}

} // namespace OwnTensor
// #endif
#else

#include "utils/Profiler.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/AllocatorRegistry.h"
#include "ops/Kernels.h"
#include "ops/Matmul.cuh"

namespace OwnTensor {

// cuBLAS handle management - thread-safe singleton per device
static std::mutex cublas_mutex;
static cublasHandle_t g_cublas_handles[8] = {nullptr};

static cublasHandle_t get_cublas_handle(int device = 0) {
  if (g_cublas_handles[device] == nullptr) {
    std::lock_guard<std::mutex> lock(cublas_mutex);
    if (g_cublas_handles[device] == nullptr) {
      cudaSetDevice(device);
      cublasCreate(&g_cublas_handles[device]);
      // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
      cublasSetMathMode(g_cublas_handles[device], CUBLAS_TF32_TENSOR_OP_MATH);
    }
  }
  return g_cublas_handles[device];
}

// ============================================================================
// cuBLASLt fused addmm fast path (CUBLASLT_EPILOGUE_BIAS)
//   Routes addmm(bias, A, B) with beta==1 + bias.numel()==N through a single
//   fused matmul-with-bias kernel. Skips the legacy memset + broadcast_scale +
//   cublasGemmEx 3-pass.
//
//   Falls through (returns false) for:
//     - unsupported dtype combos (heuristic returns 0 algos)
//     - non-row/non-col layouts, non-contiguous output
//     - any cublasLt error
//   In that case the caller runs the legacy path below.
// ============================================================================

static std::mutex cublaslt_mutex;
static cublasLtHandle_t g_cublaslt_handles[8] = {nullptr};
// 32 MiB workspace per device, allocated via project's caching allocator
static constexpr size_t CUBLASLT_WORKSPACE_BYTES = 32ull * 1024ull * 1024ull;
static void *g_cublaslt_workspace[8] = {nullptr};

// Diagnostic counters: track WHY the fast path falls through.
// On program exit we print these so we can see what's happening without
// touching the hot path with prints. Set CUBLASLT_DEBUG=1 env to enable
// the atexit dump (off by default to avoid noise in production runs).
struct CublasLtAddmmStats {
  std::atomic<uint64_t> entered{0};
  std::atomic<uint64_t> fail_out_noncontig{0};
  std::atomic<uint64_t> fail_a_layout{0};
  std::atomic<uint64_t> fail_b_layout{0};
  std::atomic<uint64_t> fail_batch_dims{0};
  std::atomic<uint64_t> fail_unsupported_config{0};
  std::atomic<uint64_t> heuristic_empty{0};
  std::atomic<uint64_t> matmul_failed{0};
  std::atomic<cublasStatus_t> last_status{CUBLAS_STATUS_SUCCESS};
  std::atomic<uint64_t> succeeded{0};
};
static CublasLtAddmmStats g_lt_stats;

static void dump_lt_stats() {
  const char *dbg = std::getenv("CUBLASLT_DEBUG");
  if (!dbg || dbg[0] == '0')
    return;
  fprintf(stderr,
          "[CUBLASLT_DEBUG] cuda_addmm fast-path stats:\n"
          "  entered                  = %lu\n"
          "  fail_out_noncontig       = %lu\n"
          "  fail_a_layout            = %lu\n"
          "  fail_b_layout            = %lu\n"
          "  fail_batch_dims          = %lu\n"
          "  fail_unsupported_config  = %lu (bias shape/dtype/alpha/beta)\n"
          "  heuristic_empty          = %lu\n"
          "  matmul_failed            = %lu  (last cublasStatus=%d)\n"
          "  succeeded                = %lu\n",
          (unsigned long)g_lt_stats.entered.load(),
          (unsigned long)g_lt_stats.fail_out_noncontig.load(),
          (unsigned long)g_lt_stats.fail_a_layout.load(),
          (unsigned long)g_lt_stats.fail_b_layout.load(),
          (unsigned long)g_lt_stats.fail_batch_dims.load(),
          (unsigned long)g_lt_stats.fail_unsupported_config.load(),
          (unsigned long)g_lt_stats.heuristic_empty.load(),
          (unsigned long)g_lt_stats.matmul_failed.load(),
          (int)g_lt_stats.last_status.load(),
          (unsigned long)g_lt_stats.succeeded.load());
}
struct LtStatsAtExit {
  LtStatsAtExit() { std::atexit(dump_lt_stats); }
};
static LtStatsAtExit g_lt_stats_at_exit;

static cublasLtHandle_t get_cublaslt_handle(int device = 0) {
  if (g_cublaslt_handles[device] == nullptr) {
    std::lock_guard<std::mutex> lock(cublaslt_mutex);
    if (g_cublaslt_handles[device] == nullptr) {
      cudaSetDevice(device);
      cublasLtCreate(&g_cublaslt_handles[device]);
    }
  }
  return g_cublaslt_handles[device];
}

static void *get_cublaslt_workspace(int device = 0) {
  if (g_cublaslt_workspace[device] == nullptr) {
    std::lock_guard<std::mutex> lock(cublaslt_mutex);
    if (g_cublaslt_workspace[device] == nullptr) {
      cudaSetDevice(device);
      Allocator *alloc = AllocatorRegistry::get_caching_allocator();
      g_cublaslt_workspace[device] = alloc->allocate(CUBLASLT_WORKSPACE_BYTES);
    }
  }
  return g_cublaslt_workspace[device];
}

template <typename T> static cudaDataType_t cublaslt_dtype() {
  if constexpr (std::is_same_v<T, float>)
    return CUDA_R_32F;
  else if constexpr (std::is_same_v<T, double>)
    return CUDA_R_64F;
  else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, float16_t>)
    return CUDA_R_16F;
  else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                     std::is_same_v<T, bfloat16_t>)
    return CUDA_R_16BF;
  else
    return CUDA_R_32F;
}

// Returns true if the fused path ran. Returns false if the caller must
// fall back to the legacy memset + broadcast_scale + cublasGemmEx pipeline.
//
// Layout convention: row-major A [M,K], row-major B [K,N], row-major C [M,N],
// bias is a length-N vector broadcast across rows.
// cuBLASLt operates column-major natively; we swap A/B and treat them as
// column-major B^T [N,K], A^T [K,M] to produce column-major C^T [N,M], which
// is row-major C [M,N].
template <typename T>
static bool launch_cublaslt_addmm(const Tensor &bias, const Tensor &mat1,
                                  const Tensor &mat2, float alpha, float beta,
                                  Tensor &output, cudaStream_t stream) {
  g_lt_stats.entered.fetch_add(1, std::memory_order_relaxed);

  const auto &osh = output.shape().dims;
  const auto &m1sh = mat1.shape().dims;
  const auto &m2sh = mat2.shape().dims;
  int on = osh.size(), m1n = m1sh.size(), m2n = m2sh.size();
  if (on < 2 || m1n < 2 || m2n < 2) {
    g_lt_stats.fail_batch_dims.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  int M = m1sh[m1n - 2];
  int K = m1sh[m1n - 1];
  int N = m2sh[m2n - 1];
  int64_t M_total = 1;
  for (int i = 0; i < on - 1; i++)
    M_total *= osh[i];

  if (alpha != 1.0f || beta != 1.0f) {
    g_lt_stats.fail_unsupported_config.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (bias.numel() != N) {
    g_lt_stats.fail_unsupported_config.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!output.is_contiguous()) {
    g_lt_stats.fail_out_noncontig.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!mat1.is_contiguous()) {
    g_lt_stats.fail_a_layout.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (!mat2.is_contiguous()) {
    g_lt_stats.fail_b_layout.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  cublasLtHandle_t lt = get_cublaslt_handle();
  void *workspace = get_cublaslt_workspace();
  cudaDataType_t dt = cublaslt_dtype<T>();

  cublasComputeType_t compute_type;
  cudaDataType_t scale_type;
  if constexpr (std::is_same_v<T, float>) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    scale_type = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, double>) {
    compute_type = CUBLAS_COMPUTE_64F;
    scale_type = CUDA_R_64F;
  } else {
    compute_type = CUBLAS_COMPUTE_32F; // fp16/bf16 accumulate in fp32
    scale_type = CUDA_R_32F;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr,
                         c_layout = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  bool ok = false;

  do {
    if (cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type) !=
        CUBLAS_STATUS_SUCCESS)
      break;

    cublasOperation_t op_n = CUBLAS_OP_N;
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &op_n,
                                       sizeof(op_n)) != CUBLAS_STATUS_SUCCESS)
      break;
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &op_n,
                                       sizeof(op_n)) != CUBLAS_STATUS_SUCCESS)
      break;

    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epi,
                                       sizeof(epi)) != CUBLAS_STATUS_SUCCESS)
      break;
    const void *bias_ptr = bias.data();
    if (cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
            sizeof(bias_ptr)) != CUBLAS_STATUS_SUCCESS)
      break;

    // Swap A/B + treat row-major as column-major-transposed (see note above).
    // B as cuBLASLt-A: [N x K], leading dim N
    if (cublasLtMatrixLayoutCreate(&a_layout, dt, N, K, N) !=
        CUBLAS_STATUS_SUCCESS)
      break;
    // A as cuBLASLt-B: [K x M_total], leading dim K
    if (cublasLtMatrixLayoutCreate(&b_layout, dt, K, M_total, K) !=
        CUBLAS_STATUS_SUCCESS)
      break;
    // Output: [N x M_total], leading dim N
    if (cublasLtMatrixLayoutCreate(&c_layout, dt, N, M_total, N) !=
        CUBLAS_STATUS_SUCCESS)
      break;

    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS)
      break;
    size_t ws_size = CUBLASLT_WORKSPACE_BYTES;
    if (cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size,
            sizeof(ws_size)) != CUBLAS_STATUS_SUCCESS)
      break;

    cublasLtMatmulHeuristicResult_t heur = {};
    int returned = 0;
    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        lt, op_desc, a_layout, b_layout, c_layout, c_layout, pref, 1, &heur,
        &returned);
    if (heur_status != CUBLAS_STATUS_SUCCESS || returned == 0) {
      g_lt_stats.heuristic_empty.fetch_add(1, std::memory_order_relaxed);
      g_lt_stats.last_status.store(heur_status, std::memory_order_relaxed);
      break;
    }

    float h_alpha = alpha, h_beta = 0.0f; // C is not read (bias is the add)
    double d_alpha = alpha, d_beta = 0.0;
    const void *p_alpha = std::is_same_v<T, double> ? (const void *)&d_alpha
                                                    : (const void *)&h_alpha;
    const void *p_beta = std::is_same_v<T, double> ? (const void *)&d_beta
                                                   : (const void *)&h_beta;

    cublasStatus_t status = cublasLtMatmul(
        lt, op_desc, p_alpha, mat2.data(), a_layout, mat1.data(), b_layout,
        p_beta, output.data(), c_layout, output.data(), c_layout, &heur.algo,
        workspace, CUBLASLT_WORKSPACE_BYTES, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      g_lt_stats.matmul_failed.fetch_add(1, std::memory_order_relaxed);
      g_lt_stats.last_status.store(status, std::memory_order_relaxed);
      break;
    }

    ok = true;
  } while (false);

  if (pref)
    cublasLtMatmulPreferenceDestroy(pref);
  if (c_layout)
    cublasLtMatrixLayoutDestroy(c_layout);
  if (b_layout)
    cublasLtMatrixLayoutDestroy(b_layout);
  if (a_layout)
    cublasLtMatrixLayoutDestroy(a_layout);
  if (op_desc)
    cublasLtMatmulDescDestroy(op_desc);

  if (ok)
    g_lt_stats.succeeded.fetch_add(1, std::memory_order_relaxed);
  return ok;
}

using namespace nvcuda;

// ============================================================================
// METADATA & CONSTANTS
// ============================================================================

struct MatmulMetadata {
  int a_shape[8], b_shape[8], out_shape[8];
  int a_strides[8], b_strides[8], out_strides[8];
  int a_ndim, b_ndim, out_ndim;
};

__device__ void compute_batch_offset(int batch_idx, const int *shape,
                                     const int *strides, int ndim,
                                     const int *out_shape, int out_ndim,
                                     int &offset) {
  offset = 0;
  if (out_ndim <= 2)
    return;
  int temp_batch = batch_idx;
  for (int dim = out_ndim - 3; dim >= 0; --dim) {
    int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
    temp_batch /= b_dim_sz;
    int c_dim = dim - (out_ndim - ndim);
    if (c_dim >= 0 && c_dim < ndim - 2)
      offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
  }
}

constexpr int PAD = 8;

// ============================================================================
// ADDMM HELPERS
// ============================================================================

template <typename T>
__global__ void
broadcast_scale_kernel(T *__restrict__ output, const T *__restrict__ input,
                       float scale, int64_t total_elements, int64_t input_size,
                       int64_t rows, int64_t cols) {
  const int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t gstride = gridDim.x * (int64_t)blockDim.x;

  if constexpr (std::is_same_v<T, float>) {
    constexpr int VEC = 4;
    const bool can_vec = (total_elements % VEC == 0) &&
                         (input_size == 1 || input_size == total_elements ||
                          (input_size == cols && cols % VEC == 0));
    if (can_vec) {
      float4 *__restrict__ out4 = reinterpret_cast<float4 *>(output);
      const float4 *__restrict__ in4 = reinterpret_cast<const float4 *>(input);
      const int64_t n4 = total_elements / VEC;
      const float sv = (input_size == 1) ? (float)input[0] * scale : 0.0f;
      for (int64_t i = tid; i < n4; i += gstride) {
        float4 v;
        if (input_size == 1) {
          v = make_float4(sv, sv, sv, sv);
        } else if (input_size == cols) {
          const float4 iv = in4[((i * VEC) % cols) / VEC];
          v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
                          iv.w * scale);
        } else {
          const float4 iv = in4[i];
          v = make_float4(iv.x * scale, iv.y * scale, iv.z * scale,
                          iv.w * scale);
        }
        out4[i] = v;
      }
    } else {
      for (int64_t i = tid; i < total_elements; i += gstride) {
        if (input_size == 1)
          output[i] = (T)((float)input[0] * scale);
        else if (input_size == cols)
          output[i] = (T)((float)input[i % cols] * scale);
        else if (input_size == total_elements)
          output[i] = (T)((float)input[i] * scale);
        else
          output[i] = (T)0.0f;
      }
    }
  } else if constexpr (std::is_same_v<T, double>) {
    constexpr int VEC = 2;
    const bool can_vec = (total_elements % VEC == 0) &&
                         (input_size == 1 || input_size == total_elements ||
                          (input_size == cols && cols % VEC == 0));
    if (can_vec) {
      double2 *__restrict__ out2 = reinterpret_cast<double2 *>(output);
      const double2 *__restrict__ in2 =
          reinterpret_cast<const double2 *>(input);
      const int64_t n2 = total_elements / VEC;
      const double sv =
          (input_size == 1) ? (double)input[0] * (double)scale : 0.0;
      for (int64_t i = tid; i < n2; i += gstride) {
        double2 v;
        if (input_size == 1) {
          v = make_double2(sv, sv);
        } else if (input_size == cols) {
          const double2 iv = in2[((i * VEC) % cols) / VEC];
          v = make_double2(iv.x * scale, iv.y * scale);
        } else {
          const double2 iv = in2[i];
          v = make_double2(iv.x * scale, iv.y * scale);
        }
        out2[i] = v;
      }
    } else {
      for (int64_t i = tid; i < total_elements; i += gstride) {
        if (input_size == 1)
          output[i] = (T)((double)input[0] * (double)scale);
        else if (input_size == cols)
          output[i] = (T)((double)input[i % cols] * (double)scale);
        else if (input_size == total_elements)
          output[i] = (T)((double)input[i] * (double)scale);
        else
          output[i] = (T)0.0;
      }
    }
  } else if constexpr (std::is_same_v<T, __half> ||
                       std::is_same_v<T, float16_t>) {
    // VEC=8: 8 fp16 elements per thread = 16 bytes = one 128-bit load/store
    // (max hardware single-instruction width). Reinterpret fp16 buffer as
    // float4, unpack as 4×half2 for the half-pair math.
    constexpr int VEC = 8;
    const bool can_vec = (total_elements % VEC == 0) &&
                         (input_size == 1 || input_size == total_elements ||
                          (input_size == cols && cols % VEC == 0));
    if (can_vec) {
      float4 *__restrict__ out4 = reinterpret_cast<float4 *>(output);
      const float4 *__restrict__ in4 = reinterpret_cast<const float4 *>(input);
      const int64_t n4 = total_elements / VEC;
      const half2 scale2 = __float2half2_rn(scale);
      // Pre-build 128-bit value holding 4 copies of (bias_pair) for the
      // input_size==1 broadcast case.
      float4 sv4;
      if (input_size == 1) {
        half2 sv_pair = __float2half2_rn((float)input[0] * scale);
        half2 *hsv = reinterpret_cast<half2 *>(&sv4);
#pragma unroll
        for (int k = 0; k < 4; k++)
          hsv[k] = sv_pair;
      }
      for (int64_t i = tid; i < n4; i += gstride) {
        float4 raw;
        if (input_size == 1) {
          raw = sv4;
        } else if (input_size == cols) {
          raw = in4[((i * VEC) % cols) / VEC];
          half2 *h = reinterpret_cast<half2 *>(&raw);
#pragma unroll
          for (int k = 0; k < 4; k++)
            h[k] = __hmul2(h[k], scale2);
        } else {
          raw = in4[i];
          half2 *h = reinterpret_cast<half2 *>(&raw);
#pragma unroll
          for (int k = 0; k < 4; k++)
            h[k] = __hmul2(h[k], scale2);
        }
        out4[i] = raw;
      }
    } else {
      for (int64_t i = tid; i < total_elements; i += gstride) {
        if (input_size == 1)
          output[i] = (T)((float)input[0] * scale);
        else if (input_size == cols)
          output[i] = (T)((float)input[i % cols] * scale);
        else if (input_size == total_elements)
          output[i] = (T)((float)input[i] * scale);
        else
          output[i] = (T)0.0f;
      }
    }
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> ||
                       std::is_same_v<T, bfloat16_t>) {
    // VEC=8: 8 bf16 elements per thread = 16 bytes = one 128-bit load/store
    // (max hardware single-instruction width). Reinterpret bf16 buffer as
    // float4, unpack as 4×__nv_bfloat162 for the half-pair math.
    constexpr int VEC = 8;
    const bool can_vec = (total_elements % VEC == 0) &&
                         (input_size == 1 || input_size == total_elements ||
                          (input_size == cols && cols % VEC == 0));
    if (can_vec) {
      float4 *__restrict__ out4 = reinterpret_cast<float4 *>(output);
      const float4 *__restrict__ in4 = reinterpret_cast<const float4 *>(input);
      const int64_t n4 = total_elements / VEC;
      // Pre-build 128-bit value holding 4 copies of (bias_pair) for the
      // input_size==1 broadcast case.
      float4 sv4;
      if (input_size == 1) {
        __nv_bfloat162 sv_pair = __float2bfloat162_rn((float)input[0] * scale);
        __nv_bfloat162 *hsv = reinterpret_cast<__nv_bfloat162 *>(&sv4);
#pragma unroll
        for (int k = 0; k < 4; k++)
          hsv[k] = sv_pair;
      }
      for (int64_t i = tid; i < n4; i += gstride) {
        float4 raw;
        if (input_size == 1) {
          raw = sv4;
        } else if (input_size == cols) {
          raw = in4[((i * VEC) % cols) / VEC];
          __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&raw);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const float2 f = __bfloat1622float2(h[k]);
            h[k] = __float22bfloat162_rn(make_float2(f.x * scale, f.y * scale));
          }
        } else {
          raw = in4[i];
          __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&raw);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const float2 f = __bfloat1622float2(h[k]);
            h[k] = __float22bfloat162_rn(make_float2(f.x * scale, f.y * scale));
          }
        }
        out4[i] = raw;
      }
    } else {
      for (int64_t i = tid; i < total_elements; i += gstride) {
        if (input_size == 1)
          output[i] = (T)((float)input[0] * scale);
        else if (input_size == cols)
          output[i] = (T)((float)input[i % cols] * scale);
        else if (input_size == total_elements)
          output[i] = (T)((float)input[i] * scale);
        else
          output[i] = (T)0.0f;
      }
    }
  } else if constexpr (std::is_integral_v<T>) {
    for (int64_t i = tid; i < total_elements; i += gstride) {
      if (input_size == 1)
        output[i] = (T)((float)input[0] * scale);
      else if (input_size == cols)
        output[i] = (T)((float)input[i % cols] * scale);
      else if (input_size == total_elements)
        output[i] = (T)((float)input[i] * scale);
      else
        output[i] = (T)0;
    }
  } else {
    for (int64_t i = tid; i < total_elements; i += gstride) {
      if (input_size == total_elements)
        output[i] = input[i];
      else
        output[i] = (T)0.0f;
    }
  }
}

template <typename T>
void launch_broadcast_scale(T *output, const T *input, float scale,
                            int64_t rows, int64_t cols, int64_t input_size,
                            cudaStream_t stream) {
  int64_t total = rows * cols;
  // Mirror the vectorization decision the kernel will make so blocks are
  // right-sized. fp16/bf16 push to VEC=8 (128-bit per thread, max hardware
  // single-instruction width). float stays at VEC=4 (also 128-bit). double
  // stays at VEC=2 (also 128-bit).
  constexpr int VEC =
      std::is_same_v<T, float> ? 4
      : (std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
         std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, bfloat16_t>)
          ? 8
      : std::is_same_v<T, double> ? 2
                                  : 1;
  const bool will_vec = (VEC > 1) && (total % VEC == 0) &&
                        (input_size == 1 || input_size == total ||
                         (input_size == cols && cols % VEC == 0));
  int64_t eff = will_vec ? total / VEC : total;
  int threads = 256;
  int blocks = (int)((eff + threads - 1) / threads);
  broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(
      output, input, scale, total, input_size, rows, cols);
}

template <typename T>
__global__ void add_scaled_kernel_typed(T *out, const T *m, float a,
                                        int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
                  std::is_same_v<T, __nv_bfloat16> ||
                  std::is_same_v<T, bfloat16_t>) {
      out[i] += (T)(a * (float)m[i]);
    } else {
      // Fallback for types that might not support float multiplication directly
      // or need different logic For complex types, we'd need a proper
      // implementation, but for now we just skip or do basic add
      out[i] += (T)m[i];
    }
  }
}

// ============================================================================
// FP16 WMMA KERNEL (17 TFLOPS Version)
// BM=128, BN=128, BK=32, Threads=512
// ============================================================================

template <int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_fp16_optimized(const __half *__restrict__ A,
                                      const __half *__restrict__ B,
                                      __half *__restrict__ C, int M, int N,
                                      int K, int total_batches,
                                      MatmulMetadata meta) {
  const int batch_idx = blockIdx.z;
  if (batch_idx >= total_batches)
    return;
  const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
            warp_col = warp_id % 4;
  int ao, bo, co;
  compute_batch_offset(batch_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
                       meta.out_shape, meta.out_ndim, ao);
  compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                       meta.out_shape, meta.out_ndim, bo);
  compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
                       meta.out_ndim, meta.out_shape, meta.out_ndim, co);
  const __half *Ap = A + ao, *Bp = B + bo;
  __half *Cp = C + co;
  int s_am = meta.a_strides[meta.a_ndim - 2],
      s_ak = meta.a_strides[meta.a_ndim - 1];
  int s_bk = meta.b_strides[meta.b_ndim - 2],
      s_bn = meta.b_strides[meta.b_ndim - 1];
  int s_cm = meta.out_strides[meta.out_ndim - 2],
      s_cn = meta.out_strides[meta.out_ndim - 1];

  __shared__ __half As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      wmma::fill_fragment(acc[i][j], __float2half(0.0f));

  auto load_tiles = [&](int ko, int idx) {
    for (int i = tid; i < BM * BK; i += 512) {
      int r = i / BK, c = i % BK;
      As[idx][r][c] = (blockIdx.y * BM + r < M && ko + c < K)
                          ? Ap[(blockIdx.y * BM + r) * s_am + (ko + c) * s_ak]
                          : __float2half(0.0f);
    }
    for (int i = tid; i < BK * BN; i += 512) {
      int r = i / BN, c = i % BN;
      Bs[idx][r][c] = (ko + r < K && blockIdx.x * BN + c < N)
                          ? Bp[(ko + r) * s_bk + (blockIdx.x * BN + c) * s_bn]
                          : __float2half(0.0f);
    }
  };
  int wi = 0;
  load_tiles(0, wi);
  __syncthreads();
  for (int k = 0; k < K; k += BK) {
    int ri = wi;
    wi = 1 - wi;
    if (k + BK < K)
      load_tiles(k + BK, wi);
#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wmma::load_matrix_sync(af, &As[ri][warp_row * WM + i * 16][ks],
                               BK + PAD);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col * WN + j * 16],
                                 BN + PAD);
          wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }
  __half *sm = reinterpret_cast<__half *>(As);
#pragma unroll
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      int cr = blockIdx.y * BM + warp_row * WM + i * 16,
          cc = blockIdx.x * BN + warp_col * WN + j * 16;
      if (cr < M && cc < N) {
        __half *wsm = sm + warp_id * 256;
        wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
        for (int r = 0; r < 16; r++)
          for (int c = 0; c < 16; c++)
            if (cr + r < M && cc + c < N)
              Cp[(cr + r) * s_cm + (cc + c) * s_cn] = wsm[r * 16 + c];
      }
    }
}

// ============================================================================
// BF16 WMMA KERNEL
// ============================================================================

template <int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_bf16_optimized(const __nv_bfloat16 *__restrict__ A,
                                      const __nv_bfloat16 *__restrict__ B,
                                      __nv_bfloat16 *__restrict__ C, int M,
                                      int N, int K, int total_batches,
                                      MatmulMetadata meta) {
  const int batch_idx = blockIdx.z;
  if (batch_idx >= total_batches)
    return;
  const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4,
            warp_col = warp_id % 4;
  int ao, bo, co;
  compute_batch_offset(batch_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
                       meta.out_shape, meta.out_ndim, ao);
  compute_batch_offset(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                       meta.out_shape, meta.out_ndim, bo);
  compute_batch_offset(batch_idx, meta.out_shape, meta.out_strides,
                       meta.out_ndim, meta.out_shape, meta.out_ndim, co);
  const __nv_bfloat16 *Ap = A + ao, *Bp = B + bo;
  __nv_bfloat16 *Cp = C + co;
  int s_am = meta.a_strides[meta.a_ndim - 2],
      s_ak = meta.a_strides[meta.a_ndim - 1];
  int s_bk = meta.b_strides[meta.b_ndim - 2],
      s_bn = meta.b_strides[meta.b_ndim - 1];
  int s_cm = meta.out_strides[meta.out_ndim - 2],
      s_cn = meta.out_strides[meta.out_ndim - 1];

  __shared__ __nv_bfloat16 As[2][BM][BK + PAD], Bs[2][BK][BN + PAD];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      wmma::fill_fragment(acc[i][j], 0.0f);

  auto load_tiles = [&](int ko, int idx) {
    for (int i = tid; i < BM * BK; i += 512) {
      int r = i / BK, c = i % BK;
      As[idx][r][c] = (blockIdx.y * BM + r < M && ko + c < K)
                          ? Ap[(blockIdx.y * BM + r) * s_am + (ko + c) * s_ak]
                          : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BK * BN; i += 512) {
      int r = i / BN, c = i % BN;
      Bs[idx][r][c] = (ko + r < K && blockIdx.x * BN + c < N)
                          ? Bp[(ko + r) * s_bk + (blockIdx.x * BN + c) * s_bn]
                          : __float2bfloat16(0.0f);
    }
  };
  int wi = 0;
  load_tiles(0, wi);
  __syncthreads();
  for (int k = 0; k < K; k += BK) {
    int ri = wi;
    wi = 1 - wi;
    if (k + BK < K)
      load_tiles(k + BK, wi);
#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wmma::load_matrix_sync(af, &As[ri][warp_row * WM + i * 16][ks],
                               BK + PAD);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col * WN + j * 16],
                                 BN + PAD);
          wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }
  float *sm = reinterpret_cast<float *>(As);
#pragma unroll
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      int cr = blockIdx.y * BM + warp_row * WM + i * 16,
          cc = blockIdx.x * BN + warp_col * WN + j * 16;
      if (cr < M && cc < N) {
        float *wsm = sm + warp_id * 256;
        wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
        for (int r = 0; r < 16; r++)
          for (int c = 0; c < 16; c++)
            if (cr + r < M && cc + c < N)
              Cp[(cr + r) * s_cm + (cc + c) * s_cn] =
                  __float2bfloat16(wsm[r * 16 + c]);
      }
    }
}

// ============================================================================
// FP32 KERNEL (5 TFLOPS Version)
// BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// ============================================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ void
matmul_fp32_optimized(const float *__restrict__ A, const float *__restrict__ B,
                      float *__restrict__ C, int M, int N, int K,
                      int total_batches, MatmulMetadata meta) {
  const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
  if (b_idx >= total_batches)
    return;
  const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
  int ao, bo, co;
  compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
                       meta.out_shape, meta.out_ndim, ao);
  compute_batch_offset(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                       meta.out_shape, meta.out_ndim, bo);
  compute_batch_offset(b_idx, meta.out_shape, meta.out_strides, meta.out_ndim,
                       meta.out_shape, meta.out_ndim, co);
  const float *Ap = A + ao, *Bp = B + bo;
  float *Cp = C + co;
  int s_am = meta.a_strides[meta.a_ndim - 2],
      s_ak = meta.a_strides[meta.a_ndim - 1],
      s_bk = meta.b_strides[meta.b_ndim - 2],
      s_bn = meta.b_strides[meta.b_ndim - 1],
      s_cm = meta.out_strides[meta.out_ndim - 2],
      s_cn = meta.out_strides[meta.out_ndim - 1];

  __shared__ float As[2][BK][BM + PAD];
  __shared__ float Bs[2][BK][BN + PAD];
  float results[16] = {0.0f}, regM[4], regN[4];

  auto load_tiles = [&](int ko, int idx) {
    if (s_ak == 1 && s_bn == 1) { // Row-Major Contiguous
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int li = tid + i * 1024, r = li / 16, c = li % 16;
        As[idx][c][r] = (by * 128 + r < M && ko + c < K)
                            ? Ap[(by * 128 + r) * s_am + (ko + c)]
                            : 0.0f;
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int li = tid + i * 1024, r = li / 128, c = li % 128;
        Bs[idx][r][c] = (ko + r < K && bx * 128 + c < N)
                            ? Bp[(ko + r) * s_bk + (bx * 128 + c)]
                            : 0.0f;
      }
    } else { // Generic
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int li = tid + i * 1024, r = li / BK, c = li % BK;
        As[idx][c][r] = (by * BM + r < M && ko + c < K)
                            ? Ap[(by * BM + r) * s_am + (ko + c) * s_ak]
                            : 0.0f;
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int li = tid + i * 1024, r = li / BN, c = li % BN;
        Bs[idx][r][c] = (ko + r < K && bx * BN + c < N)
                            ? Bp[(ko + r) * s_bk + (bx * BN + c) * s_bn]
                            : 0.0f;
      }
    }
  };

  int wi = 0;
  load_tiles(0, wi);
  __syncthreads();
  for (int bk = 0; bk < K; bk += BK) {
    int ri = wi;
    wi = 1 - wi;
    if (bk + BK < K)
      load_tiles(bk + BK, wi);
#pragma unroll
    for (int d = 0; d < BK; d++) {
#pragma unroll
      for (int i = 0; i < 4; i++)
        regM[i] = As[ri][d][tRow * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; j++)
        regN[j] = Bs[ri][d][tCol * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          results[i * 4 + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      int r = by * BM + tRow * 4 + i, c = bx * BN + tCol * 4 + j;
      if (r < M && c < N)
        Cp[r * s_cm + c * s_cn] = results[i * 4 + j];
    }
}

// ============================================================================
// FP64 KERNEL (64x64x8 Tile, 256 Threads, 4x4 Tiling)
// ============================================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_fp64_optimized(const double *__restrict__ A,
                                      const double *__restrict__ B,
                                      double *__restrict__ C, int M, int N,
                                      int K, int total_batches,
                                      MatmulMetadata meta) {
  const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
  if (b_idx >= total_batches)
    return;
  const int tid = threadIdx.x, tC = tid % (BN / TN), tR = tid / (BN / TN);
  int ao, bo, co;
  compute_batch_offset(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim,
                       meta.out_shape, meta.out_ndim, ao);
  compute_batch_offset(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                       meta.out_shape, meta.out_ndim, bo);
  compute_batch_offset(b_idx, meta.out_shape, meta.out_strides, meta.out_ndim,
                       meta.out_shape, meta.out_ndim, co);
  const double *Ap = A + ao, *Bp = B + bo;
  double *Cp = C + co;
  int s_am = meta.a_strides[meta.a_ndim - 2],
      s_ak = meta.a_strides[meta.a_ndim - 1],
      s_bk = meta.b_strides[meta.b_ndim - 2],
      s_bn = meta.b_strides[meta.b_ndim - 1],
      s_cm = meta.out_strides[meta.out_ndim - 2],
      s_cn = meta.out_strides[meta.out_ndim - 1];
  __shared__ double As[BM * BK], Bs[BK * BN];
  double res[16] = {0.0}, rM[4], rN[4];
  for (int bk = 0; bk < K; bk += BK) {
    for (int i = tid; i < BM * BK; i += 256) {
      int r = i / BK, c = i % BK;
      As[i] = (by * BM + r < M && bk + c < K)
                  ? Ap[(by * BM + r) * s_am + (bk + c) * s_ak]
                  : 0.0;
    }
    for (int i = tid; i < BK * BN; i += 256) {
      int r = i / BN, c = i % BN;
      Bs[i] = (bk + r < K && bx * BN + c < N)
                  ? Bp[(bk + r) * s_bk + (bk + c) * s_bn]
                  : 0.0;
    }
    __syncthreads();
    for (int d = 0; d < BK; d++) {
      for (int i = 0; i < 4; i++)
        rM[i] = As[(tR * 4 + i) * BK + d];
      for (int i = 0; i < 4; i++)
        rN[i] = Bs[d * BN + tC * 4 + i];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
          res[i * 4 + j] += rM[i] * rN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      int r = by * BM + tR * 4 + i, c = bx * BN + tC * 4 + j;
      if (r < M && c < N)
        Cp[r * s_cm + c * s_cn] = res[i * 4 + j];
    }
}

// ============================================================================
// DISPATCH LAYER
// ============================================================================

template <typename T>
void launch_optimized_matmul(const Tensor &A, const Tensor &B, Tensor &output,
                             cudaStream_t stream) {
  AUTO_PROFILE_CUDA("Forward::Matmul_CUDA");
  const auto &ash = A.shape().dims, &bsh = B.shape().dims,
             &osh = output.shape().dims;
  int an = ash.size(), bn = bsh.size(), on = osh.size(), M = ash[an - 2],
      K = ash[an - 1], N = bsh[bn - 1], tb = 1;
  for (int i = 0; i < on - 2; i++)
    tb *= osh[i];
  MatmulMetadata meta;
  meta.a_ndim = an;
  meta.b_ndim = bn;
  meta.out_ndim = on;
  for (int i = 0; i < an; i++) {
    meta.a_shape[i] = ash[i];
    meta.a_strides[i] = A.stride().strides[i];
  }
  for (int i = 0; i < bn; i++) {
    meta.b_shape[i] = bsh[i];
    meta.b_strides[i] = B.stride().strides[i];
  }
  for (int i = 0; i < on; i++) {
    meta.out_shape[i] = osh[i];
    meta.out_strides[i] = output.stride().strides[i];
  }
  const T *ap = A.data<T>(), *bp = B.data<T>();
  T *op = output.data<T>();

  if constexpr (std::is_same<T, float16_t>::value ||
                std::is_same<T, __half>::value) {
    // Try cuBLAS first for FP16: compute in FP16, accumulate in FP32
    bool a_contiguous = (meta.a_strides[an - 1] == 1);
    bool b_contiguous = (meta.b_strides[bn - 1] == 1);
    bool out_contiguous = (meta.out_strides[on - 1] == 1);
    bool is_supported =
        (a_contiguous || (an >= 2 && meta.a_strides[an - 2] == 1)) &&
        (b_contiguous || (bn >= 2 && meta.b_strides[bn - 2] == 1)) &&
        out_contiguous;
    bool cublas_ok = false;
    if (is_supported) {
      int dev_idx = output.device().index;
      cublasHandle_t handle = get_cublas_handle(dev_idx);
      cublasSetStream(handle, stream);
      float alpha_f = 1.0f, beta_f = 0.0f;
      cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int lda = a_contiguous ? meta.a_strides[an - 2] : meta.a_strides[an - 1];
      cublasOperation_t opB = b_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int ldb = b_contiguous ? meta.b_strides[bn - 2] : meta.b_strides[bn - 1];
      int ldc = meta.out_strides[on - 2];
      const __half *ap_h = reinterpret_cast<const __half *>(ap);
      const __half *bp_h = reinterpret_cast<const __half *>(bp);
      __half *op_h = reinterpret_cast<__half *>(op);
      cublasStatus_t status;
      if (tb == 1) {
        status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f, bp_h,
                              CUDA_R_16F, ldb, ap_h, CUDA_R_16F, lda, &beta_f,
                              op_h, CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
        long long stride_a_ll =
            (an >= 3) ? (long long)meta.a_strides[an - 3] : 0LL;
        long long stride_b_ll =
            (bn >= 3) ? (long long)meta.b_strides[bn - 3] : 0LL;
        long long stride_c_ll = (long long)meta.out_strides[on - 3];
        status = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha_f, bp_h, CUDA_R_16F, ldb,
            stride_b_ll, ap_h, CUDA_R_16F, lda, stride_a_ll, &beta_f, op_h,
            CUDA_R_16F, ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
    }
    if (!cublas_ok) {
      matmul_fp16_optimized<128, 128, 32, 32, 32>
          <<<dim3((N + 127) / 128, (M + 127) / 128, tb), 512, 0, stream>>>(
              reinterpret_cast<const __half *>(ap),
              reinterpret_cast<const __half *>(bp),
              reinterpret_cast<__half *>(op), M, N, K, tb, meta);
    }
  } else if constexpr (std::is_same<T, bfloat16_t>::value ||
                       std::is_same<T, __nv_bfloat16>::value) {
    // Try cuBLAS first for BF16: compute in BF16, accumulate in FP32
    bool a_contiguous = (meta.a_strides[an - 1] == 1);
    bool b_contiguous = (meta.b_strides[bn - 1] == 1);
    bool out_contiguous = (meta.out_strides[on - 1] == 1);
    bool is_supported =
        (a_contiguous || (an >= 2 && meta.a_strides[an - 2] == 1)) &&
        (b_contiguous || (bn >= 2 && meta.b_strides[bn - 2] == 1)) &&
        out_contiguous;
    bool cublas_ok = false;
    if (is_supported) {
      int dev_idx = output.device().index;
      cublasHandle_t handle = get_cublas_handle(dev_idx);
      cublasSetStream(handle, stream);
      float alpha_f = 1.0f, beta_f = 0.0f;
      cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int lda = a_contiguous ? meta.a_strides[an - 2] : meta.a_strides[an - 1];
      cublasOperation_t opB = b_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int ldb = b_contiguous ? meta.b_strides[bn - 2] : meta.b_strides[bn - 1];
      int ldc = meta.out_strides[on - 2];
      const __nv_bfloat16 *ap_b = reinterpret_cast<const __nv_bfloat16 *>(ap);
      const __nv_bfloat16 *bp_b = reinterpret_cast<const __nv_bfloat16 *>(bp);
      __nv_bfloat16 *op_b = reinterpret_cast<__nv_bfloat16 *>(op);
      cublasStatus_t status;
      if (tb == 1) {
        status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha_f, bp_b,
                              CUDA_R_16BF, ldb, ap_b, CUDA_R_16BF, lda, &beta_f,
                              op_b, CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
        long long stride_a_ll =
            (an >= 3) ? (long long)meta.a_strides[an - 3] : 0LL;
        long long stride_b_ll =
            (bn >= 3) ? (long long)meta.b_strides[bn - 3] : 0LL;
        long long stride_c_ll = (long long)meta.out_strides[on - 3];
        status = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha_f, bp_b, CUDA_R_16BF, ldb,
            stride_b_ll, ap_b, CUDA_R_16BF, lda, stride_a_ll, &beta_f, op_b,
            CUDA_R_16BF, ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      cublas_ok = (status == CUBLAS_STATUS_SUCCESS);
    }
    if (!cublas_ok) {
      matmul_bf16_optimized<128, 128, 32, 32, 32>
          <<<dim3((N + 127) / 128, (M + 127) / 128, tb), 512, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16 *>(ap),
              reinterpret_cast<const __nv_bfloat16 *>(bp),
              reinterpret_cast<__nv_bfloat16 *>(op), M, N, K, tb, meta);
    }
  } else if constexpr (std::is_same<T, float>::value) {
    // Use cuBLAS for FP32 matmuls - enables TF32 Tensor Core acceleration
    // Check if tensors are contiguous (row-major) OR transposed (col-major
    // equivalent) for cuBLAS fast path

    bool a_contiguous = (meta.a_strides[an - 1] == 1);
    bool a_transposed = (an >= 2 && meta.a_strides[an - 2] == 1);

    bool b_contiguous = (meta.b_strides[bn - 1] == 1);
    bool b_transposed = (bn >= 2 && meta.b_strides[bn - 2] == 1);

    bool out_contiguous = (meta.out_strides[on - 1] == 1);

    // Use cuBLAS for:
    // 1. 2D matrices (tb=1)
    // 2. Strided Batched matrices (tb>1)
    bool is_supported_layout = (a_contiguous || a_transposed) &&
                               (b_contiguous || b_transposed) && out_contiguous;
    bool is_strided_batch = false;
    long long stride_a = 0, stride_b = 0, stride_c = 0;

    if (tb > 1 && is_supported_layout) {
      // Calculate Strides for Batches
      // For broadcasted dims (e.g. B is 2D but A is 3D), stride is 0.

      // Helper to get batch stride check
      // Assumption: dimensions 0..on-3 are the batch dimensions.
      // Standard layout: Flattened batch.

      if (on == 3) {
        stride_c = meta.out_strides[0];
        stride_a = (an == 3) ? meta.a_strides[0] : 0; // Broadcast if an < 3
        stride_b = (bn == 3) ? meta.b_strides[0] : 0; // Broadcast if bn < 3
        is_strided_batch = true;
      } else if (on == 4) {
        // 4D [d0, d1, M, N]: use inner batch stride (stride[1]) for cuBLAS
        stride_c = meta.out_strides[on - 3];
        stride_a = (an == 4) ? meta.a_strides[an - 3] : 0;
        stride_b = (bn == 4) ? meta.b_strides[bn - 3] : 0;

        // cuBLAS strided batched requires uniform stride across all tb batches.
        // Uniform iff stride[0] == stride[1] * shape[1] for each tensor.
        // Non-contiguous transposes (e.g. reshape+transpose without contiguous)
        // break this — handled by per-outer-batch loop below.
        bool batch_uniform =
            (meta.out_strides[0] == meta.out_strides[1] * osh[1]);
        if (an == 4)
          batch_uniform = batch_uniform &&
                          (meta.a_strides[0] == meta.a_strides[1] * ash[1]);
        if (bn == 4)
          batch_uniform = batch_uniform &&
                          (meta.b_strides[0] == meta.b_strides[1] * bsh[1]);

        if (batch_uniform) {
          is_strided_batch = true;
        }
        // else: non-uniform, handled by loop path below
      }
    }

    if ((tb == 1 || is_strided_batch) && is_supported_layout) {
      // USE EXPLICIT DEVICE INDEXING from output tensor
      int dev_idx = output.device().index;
      cublasHandle_t handle = get_cublas_handle(dev_idx);
      cublasSetStream(handle, stream);

      float alpha = 1.0f, beta = 0.0f;

      // Setup A (Second in CuBLAS, since CuBLAS is ColMajor)
      cublasOperation_t opA;
      int lda;
      if (a_contiguous) {
        opA = CUBLAS_OP_N;
        lda = meta.a_strides[an - 2];
      } else {
        opA = CUBLAS_OP_T;
        lda = meta.a_strides[an - 1];
      }

      // Setup B (First in CuBLAS)
      cublasOperation_t opB;
      int ldb;
      if (b_contiguous) {
        opB = CUBLAS_OP_N;
        ldb = meta.b_strides[bn - 2];
      } else {
        opB = CUBLAS_OP_T;
        ldb = meta.b_strides[bn - 1];
      }

      int ldc = meta.out_strides[on - 2];

      // Logic correction:
      // C = A @ B.
      // CuBLAS (ColMajor): C^T = B^T @ A^T.
      // Pass B as "A", A as "B".
      //
      // If A is contiguous RM [M, K]: Represents A^T [K, M] CM.
      // We need A^T in the equation. So Op = N.
      // If A is transposed RM [M, K] (stride 1, M): Represents A [K, M] CM.
      // We need A^T. So Op = T.
      //
      // If B is contiguous RM [K, N]: Represents B^T [N, K] CM.
      // We need B^T in the equation. So Op = N.
      // If B is transposed RM [K, N]: Represents B [N, K] CM.
      // We need B^T. So Op = T.

      cublasStatus_t status;

      if (tb == 1) {
        status = cublasGemmEx(handle, opB, opA, N, M, K, &alpha, bp, CUDA_R_32F,
                              ldb, ap, CUDA_R_32F, lda, &beta, op, CUDA_R_32F,
                              ldc, CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
        status = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha, bp, CUDA_R_32F, ldb, stride_b,
            ap, CUDA_R_32F, lda, stride_a, &beta, op, CUDA_R_32F, ldc, stride_c,
            tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }

      if (status != CUBLAS_STATUS_SUCCESS) {
        // Fallback (e.g. if alignment issues, though Ex handles most)
        matmul_fp32_optimized<128, 128, 16, 4, 4>
            <<<dim3((N + 127) / 128, (M + 127) / 128, tb), 1024, 0, stream>>>(
                ap, bp, op, M, N, K, tb, meta);
      }
    } else if (on == 4 && !is_strided_batch && is_supported_layout) {
      // Non-uniform 4D batch strides (e.g. after transpose without contiguous):
      // Loop over outer batch dim d0, each call handles d1 inner batches with
      // uniform stride
      int current_device;
      cudaGetDevice(&current_device);
      cublasHandle_t handle = get_cublas_handle(current_device);
      cublasSetStream(handle, stream);
      float alpha_v = 1.0f, beta_v = 0.0f;

      cublasOperation_t opA_l, opB_l;
      int lda_l, ldb_l, ldc_l;
      if (a_contiguous) {
        opA_l = CUBLAS_OP_N;
        lda_l = meta.a_strides[an - 2];
      } else {
        opA_l = CUBLAS_OP_T;
        lda_l = meta.a_strides[an - 1];
      }
      if (b_contiguous) {
        opB_l = CUBLAS_OP_N;
        ldb_l = meta.b_strides[bn - 2];
      } else {
        opB_l = CUBLAS_OP_T;
        ldb_l = meta.b_strides[bn - 1];
      }
      ldc_l = meta.out_strides[on - 2];

      int d0 = osh[0], d1 = osh[1];
      bool all_ok = true;
      for (int bi = 0; bi < d0 && all_ok; bi++) {
        const float *a_b = ap + (an == 4 ? bi * meta.a_strides[0] : 0);
        const float *b_b = bp + (bn == 4 ? bi * meta.b_strides[0] : 0);
        float *o_b = op + bi * meta.out_strides[0];

        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle, opB_l, opA_l, N, M, K, &alpha_v, b_b, CUDA_R_32F, ldb_l,
            stride_b, a_b, CUDA_R_32F, lda_l, stride_a, &beta_v, o_b,
            CUDA_R_32F, ldc_l, stride_c, d1, CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS)
          all_ok = false;
      }
      if (!all_ok) {
        matmul_fp32_optimized<128, 128, 16, 4, 4>
            <<<dim3((N + 127) / 128, (M + 127) / 128, tb), 1024, 0, stream>>>(
                ap, bp, op, M, N, K, tb, meta);
      }
    } else {
      // Fallback to custom kernel
      matmul_fp32_optimized<128, 128, 16, 4, 4>
          <<<dim3((N + 127) / 128, (M + 127) / 128, tb), 1024, 0, stream>>>(
              ap, bp, op, M, N, K, tb, meta);
    }
  } else if constexpr (std::is_same<T, double>::value) {
    matmul_fp64_optimized<64, 64, 8, 4, 4>
        <<<dim3((N + 63) / 64, (M + 63) / 64, tb), 256, 0, stream>>>(
            reinterpret_cast<const double *>(ap),
            reinterpret_cast<const double *>(bp),
            reinterpret_cast<double *>(op), M, N, K, tb, meta);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error("Kernel failed: " +
                             std::string(cudaGetErrorString(err)));
}

void cuda_matmul(const Tensor &A, const Tensor &B, Tensor &output,
                 cudaStream_t stream) {
  dispatch_by_dtype(A.dtype(), [&](auto d) {
    using T = decltype(d);
    if constexpr (std::is_same<T, float16_t>::value ||
                  std::is_same<T, bfloat16_t>::value ||
                  std::is_same<T, float>::value ||
                  std::is_same<T, double>::value)
      launch_optimized_matmul<T>(A, B, output, stream);
    else
      throw std::runtime_error("Unsupported type");
  });
}

void cuda_addmm(const Tensor &input, const Tensor &mat1, const Tensor &mat2,
                float alpha, float beta, Tensor &output, cudaStream_t stream) {
  AUTO_PROFILE_CUDA("Forward::Addmm_CUDA");
  const auto &m1sh = mat1.shape().dims, &m2sh = mat2.shape().dims,
             &osh = output.shape().dims;
  int m1n = m1sh.size(), m2n = m2sh.size(), on = osh.size();
  int M = m1sh[m1n - 2], K = m1sh[m1n - 1], N = m2sh[m2n - 1];

  // Compute total batch count from output shape (all dims except last 2)
  int tb = 1;
  for (int i = 0; i < on - 2; i++)
    tb *= osh[i];
  int64_t total_output_elements = output.numel(); // tb * M * N

  // Fast path: fused matmul+bias via cuBLASLt CUBLASLT_EPILOGUE_BIAS.
  // Eligible when bias is a length-N vector and beta==1 (the nn::Linear case).
  // On success, returns before touching the legacy 3-pass code below.
  if (beta == 1.0f && input.numel() == N) {
    bool success = false;
    dispatch_by_dtype(output.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, __nv_bfloat16> ||
                    std::is_same_v<T, bfloat16_t>) {
        success = launch_cublaslt_addmm<T>(input, mat1, mat2, alpha, beta,
                                           output, stream);
      }
    });
    if (success)
      return;
  }

  // cuBLAS's beta!=0 GEMM accumulates C = alpha*A@B + beta*C and reads C with
  // vectorized (8/16-byte) loads that can touch bytes past the logical tb*M*N
  // (tile/vector padding at the buffer tail). The broadcast-scale below seeds
  // only the logical elements, leaving that padding uninitialized — recycled
  // caching-allocator memory that may hold NaN/Inf, which beta*C then injects.
  // Zero the whole allocation first so any padding C the GEMM reads is 0.
  cudaMemsetAsync(output.data(), 0, output.nbytes(), stream);

  // 1. Prepare output buffer: output = beta * input (with broadcasting)
  //    Scale ALL elements of the output, not just M*N
  if (beta != 0.0f) {
    dispatch_by_dtype(output.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, __nv_bfloat16> ||
                    std::is_same_v<T, bfloat16_t> || std::is_integral_v<T>) {
        // Use total_output_elements as both rows*cols and total
        // M_total and N stay the same for broadcasting logic (bias is broadcast
        // per [M,N] slice)
        int64_t rows_total = (int64_t)tb * M;
        launch_broadcast_scale<T>(output.data<T>(), input.data<T>(), beta,
                                  rows_total, N, input.numel(), stream);
      }
    });
  } else {
    // If beta is 0, input is ignored, zero the entire output
    dispatch_by_dtype(output.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
                    std::is_same_v<T, __nv_bfloat16> ||
                    std::is_same_v<T, bfloat16_t> || std::is_integral_v<T>) {
        int threads = 256;
        int blocks = (total_output_elements + threads - 1) / threads;
        // Simple zero kernel
        broadcast_scale_kernel<T><<<blocks, threads, 0, stream>>>(
            output.data<T>(), output.data<T>(), 0.0f, total_output_elements,
            total_output_elements, (int64_t)tb * M, N);
      }
    });
  }

  // 2. Add alpha * (mat1 @ mat2) using cuBLAS
  if (output.dtype() == Dtype::Float32) {
    MatmulMetadata meta;
    meta.a_ndim = m1n;
    meta.b_ndim = m2n;
    meta.out_ndim = on;
    for (int i = 0; i < m1n; i++) {
      meta.a_shape[i] = m1sh[i];
      meta.a_strides[i] = mat1.stride().strides[i];
    }
    for (int i = 0; i < m2n; i++) {
      meta.b_shape[i] = m2sh[i];
      meta.b_strides[i] = mat2.stride().strides[i];
    }
    for (int i = 0; i < on; i++) {
      meta.out_shape[i] = osh[i];
      meta.out_strides[i] = output.stride().strides[i];
    }

    bool a_contiguous = (meta.a_strides[m1n - 1] == 1);
    bool a_transposed = (m1n >= 2 && meta.a_strides[m1n - 2] == 1);
    bool b_contiguous = (meta.b_strides[m2n - 1] == 1);
    bool b_transposed = (m2n >= 2 && meta.b_strides[m2n - 2] == 1);
    bool out_contiguous = (meta.out_strides[on - 1] == 1);

    bool is_supported_layout = (a_contiguous || a_transposed) &&
                               (b_contiguous || b_transposed) && out_contiguous;

    // Compute batch strides (mirroring launch_optimized_matmul)
    bool is_strided_batch = false;
    long long stride_a = 0, stride_b = 0, stride_c = 0;

    if (tb > 1 && is_supported_layout) {
      if (on == 3) {
        stride_c = meta.out_strides[0];
        stride_a = (m1n == 3) ? meta.a_strides[0] : 0;
        stride_b = (m2n == 3) ? meta.b_strides[0] : 0;
        is_strided_batch = true;
      } else if (on == 4) {
        stride_c = static_cast<long long>(M) * N;
        stride_a = (m1n == 4) ? static_cast<long long>(M) * K : 0;
        stride_b = (m2n == 4) ? static_cast<long long>(K) * N : 0;
        is_strided_batch = true;
      }
    }

    if ((tb == 1 || is_strided_batch) && is_supported_layout) {
      // USE EXPLICIT DEVICE INDEXING from output tensor
      int dev_idx = output.device().index;
      cublasHandle_t handle = get_cublas_handle(dev_idx);
      cublasSetStream(handle, stream);

      cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int lda =
          a_contiguous ? meta.a_strides[m1n - 2] : meta.a_strides[m1n - 1];
      cublasOperation_t opB = b_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int ldb =
          b_contiguous ? meta.b_strides[m2n - 2] : meta.b_strides[m2n - 1];
      int ldc = meta.out_strides[on - 2];

      // output already has beta * input, so we pass cublas_beta = 1.0 to
      // accumulate
      float cublas_beta = 1.0f;

      cublasStatus_t status;

      if (tb == 1) {
        status = cublasGemmEx(
            handle, opB, opA, N, M, K, &alpha, mat2.data<float>(), CUDA_R_32F,
            ldb, mat1.data<float>(), CUDA_R_32F, lda, &cublas_beta,
            output.data<float>(), CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      } else {
        status = cublasGemmStridedBatchedEx(
            handle, opB, opA, N, M, K, &alpha, mat2.data<float>(), CUDA_R_32F,
            ldb, stride_b, mat1.data<float>(), CUDA_R_32F, lda, stride_a,
            &cublas_beta, output.data<float>(), CUDA_R_32F, ldc, stride_c, tb,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }

      if (status == CUBLAS_STATUS_SUCCESS)
        return;
      // Fall through to fallback on failure
    }
  } else if (output.dtype() == Dtype::Float16 ||
             output.dtype() == Dtype::Bfloat16) {
    // cuBLAS path for FP16/BF16 addmm: compute in fp16/bf16, accumulate in fp32
    MatmulMetadata meta16;
    meta16.a_ndim = m1n;
    meta16.b_ndim = m2n;
    meta16.out_ndim = on;
    for (int i = 0; i < m1n; i++) {
      meta16.a_shape[i] = m1sh[i];
      meta16.a_strides[i] = mat1.stride().strides[i];
    }
    for (int i = 0; i < m2n; i++) {
      meta16.b_shape[i] = m2sh[i];
      meta16.b_strides[i] = mat2.stride().strides[i];
    }
    for (int i = 0; i < on; i++) {
      meta16.out_shape[i] = osh[i];
      meta16.out_strides[i] = output.stride().strides[i];
    }

    bool a_contiguous = (meta16.a_strides[m1n - 1] == 1);
    bool b_contiguous = (meta16.b_strides[m2n - 1] == 1);
    bool out_contiguous = (meta16.out_strides[on - 1] == 1);
    bool is_supported =
        (a_contiguous || (m1n >= 2 && meta16.a_strides[m1n - 2] == 1)) &&
        (b_contiguous || (m2n >= 2 && meta16.b_strides[m2n - 2] == 1)) &&
        out_contiguous;

    if (is_supported) {
      int dev_idx = output.device().index;
      cublasHandle_t handle = get_cublas_handle(dev_idx);
      cublasSetStream(handle, stream);

      cublasOperation_t opA = a_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int lda =
          a_contiguous ? meta16.a_strides[m1n - 2] : meta16.a_strides[m1n - 1];
      cublasOperation_t opB = b_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
      int ldb =
          b_contiguous ? meta16.b_strides[m2n - 2] : meta16.b_strides[m2n - 1];
      int ldc = meta16.out_strides[on - 2];
      // output already holds beta*input; pass cublas_beta=1.0 to accumulate
      // alpha*(mat1@mat2)
      float cublas_beta = 1.0f;

      bool is_strided_batch = (tb > 1) && (on == 3 || on == 4);
      long long stride_a_ll = 0, stride_b_ll = 0, stride_c_ll = 0;
      if (is_strided_batch) {
        stride_c_ll = meta16.out_strides[on - 3];
        stride_a_ll = (m1n >= on) ? meta16.a_strides[m1n - 3] : 0LL;
        stride_b_ll = (m2n >= on) ? meta16.b_strides[m2n - 3] : 0LL;
      }

      cudaDataType_t cuda_type =
          (output.dtype() == Dtype::Float16) ? CUDA_R_16F : CUDA_R_16BF;
      cublasStatus_t status;

      if (output.dtype() == Dtype::Float16) {
        const __half *m1h =
            reinterpret_cast<const __half *>(mat1.data<float16_t>());
        const __half *m2h =
            reinterpret_cast<const __half *>(mat2.data<float16_t>());
        __half *outh = reinterpret_cast<__half *>(output.data<float16_t>());
        if (tb == 1) {
          status = cublasGemmEx(
              handle, opB, opA, N, M, K, &alpha, m2h, CUDA_R_16F, ldb, m1h,
              CUDA_R_16F, lda, &cublas_beta, outh, CUDA_R_16F, ldc,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
          status = cublasGemmStridedBatchedEx(
              handle, opB, opA, N, M, K, &alpha, m2h, CUDA_R_16F, ldb,
              stride_b_ll, m1h, CUDA_R_16F, lda, stride_a_ll, &cublas_beta,
              outh, CUDA_R_16F, ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      } else {
        const __nv_bfloat16 *m1b =
            reinterpret_cast<const __nv_bfloat16 *>(mat1.data<bfloat16_t>());
        const __nv_bfloat16 *m2b =
            reinterpret_cast<const __nv_bfloat16 *>(mat2.data<bfloat16_t>());
        __nv_bfloat16 *outb =
            reinterpret_cast<__nv_bfloat16 *>(output.data<bfloat16_t>());
        if (tb == 1) {
          status = cublasGemmEx(
              handle, opB, opA, N, M, K, &alpha, m2b, CUDA_R_16BF, ldb, m1b,
              CUDA_R_16BF, lda, &cublas_beta, outb, CUDA_R_16BF, ldc,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
          status = cublasGemmStridedBatchedEx(
              handle, opB, opA, N, M, K, &alpha, m2b, CUDA_R_16BF, ldb,
              stride_b_ll, m1b, CUDA_R_16BF, lda, stride_a_ll, &cublas_beta,
              outb, CUDA_R_16BF, ldc, stride_c_ll, tb, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
      }
      if (status == CUBLAS_STATUS_SUCCESS)
        return;
      // Fall through to fallback on failure
    }
  }

  // Fallback: Use existing batched matmul + scaled addition kernel
  Tensor temp_matmul = OwnTensor::matmul(mat1, mat2, stream);

  dispatch_by_dtype(output.dtype(), [&](auto d) {
    using T = decltype(d);
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
                  std::is_same_v<T, __nv_bfloat16> ||
                  std::is_same_v<T, bfloat16_t> || std::is_integral_v<T>) {
      int64_t total = output.numel();
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      add_scaled_kernel_typed<T><<<blocks, threads, 0, stream>>>(
          output.data<T>(), temp_matmul.data<T>(), alpha, total);
    }
  });
}

} // namespace OwnTensor
// #endif
#endif // WITH_MYBLAS