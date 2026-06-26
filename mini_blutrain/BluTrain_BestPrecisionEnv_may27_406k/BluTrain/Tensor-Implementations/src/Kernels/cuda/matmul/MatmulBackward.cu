// // #ifdef WITH_CUDA

// // #ifdef WITH_MYBLAS


// // #include <cuda_runtime.h>
// // #include <cuda_fp16.h>
// // #include <cuda_bf16.h>
// // #include <mma.h>
// // #include "mycublas.h"
// // #include <algorithm>
// // #include <stdexcept>
// // #include <vector>
// // #include <string>
// // #include <mutex>

// // #include "ops/MatmulBackward.cuh"
// // #include "core/Tensor.h"
// // #include "core/TensorDispatch.h"
// // #include "utils/Profiler.h"

// // namespace OwnTensor {

// // // mycublas handle management - thread-safe singleton per device
// // static std::mutex mycublas_mutex_bwd;
// // static mycublasHandle_t g_mycublas_handles_bwd[8]  = {nullptr};
// // static mycublasHandle_t g_mycublas_handles_bwd2[8] = {nullptr};  // second handle for parallel grad_B stream

// // // Per-device secondary stream and cached no-timing events for grad_A/grad_B overlap.
// // static cudaStream_t g_secondary_streams_bwd[8]  = {nullptr};
// // static cudaEvent_t  g_bwd_event_ready[8]        = {nullptr};   // primary → secondary (inputs ready)
// // static cudaEvent_t  g_bwd_event_done[8]         = {nullptr};   // secondary → primary (grad_B done)

// // static mycublasHandle_t get_mycublas_handle_bwd(int device = 0) {
// //    if (g_mycublas_handles_bwd[device] == nullptr) {
// //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// //       if (g_mycublas_handles_bwd[device] == nullptr) {
// //          cudaSetDevice(device);
// //          mycublasCreate(&g_mycublas_handles_bwd[device]);
// //       }
// //    }
// //    return g_mycublas_handles_bwd[device];
// // }

// // static mycublasHandle_t get_mycublas_handle_bwd2(int device = 0) {
// //    if (g_mycublas_handles_bwd2[device] == nullptr) {
// //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// //       if (g_mycublas_handles_bwd2[device] == nullptr) {
// //          cudaSetDevice(device);
// //          mycublasCreate(&g_mycublas_handles_bwd2[device]);
// //       }
// //    }
// //    return g_mycublas_handles_bwd2[device];
// // }

// // static cudaStream_t get_secondary_stream_bwd(int device = 0) {
// //    if (g_secondary_streams_bwd[device] == nullptr) {
// //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// //       if (g_secondary_streams_bwd[device] == nullptr) {
// //          cudaSetDevice(device);
// //          // Non-blocking so it doesn't serialize with the default stream
// //          cudaStreamCreateWithFlags(&g_secondary_streams_bwd[device], cudaStreamNonBlocking);
// //       }
// //    }
// //    return g_secondary_streams_bwd[device];
// // }

// // static void init_bwd_events(int device) {
// //    if (g_bwd_event_ready[device] == nullptr) {
// //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// //       if (g_bwd_event_ready[device] == nullptr) {
// //          cudaSetDevice(device);
// //          // DisableTiming avoids hardware timestamp recording, making record/wait ~10× cheaper
// //          cudaEventCreateWithFlags(&g_bwd_event_ready[device], cudaEventDisableTiming);
// //          cudaEventCreateWithFlags(&g_bwd_event_done[device],  cudaEventDisableTiming);
// //       }
// //    }
// // }

// // using namespace nvcuda;

// // // ============================================================================
// // // METADATA & CONSTANTS
// // // ============================================================================

// // struct BackwardMetadata {
// //    int a_shape[8], b_shape[8], grad_out_shape[8];
// //    int a_strides[8], b_strides[8], grad_out_strides[8];
// //    int grad_a_strides[8], grad_b_strides[8];
// //    int a_ndim, b_ndim, grad_out_ndim;
// // };

// // __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
// //                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
// //    offset = 0; 
// //    if (out_ndim <= 2) return;
// //    int temp_batch = batch_idx;
// //    for (int dim = out_ndim - 3; dim >= 0; --dim) {
// //       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
// //       temp_batch /= b_dim_sz;
// //       int c_dim = dim - (out_ndim - ndim);
// //       if (c_dim >= 0 && c_dim < ndim - 2) 
// //          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
// //    }
// // }

// // constexpr int BACKWARD_PAD = 8;

// // // ============================================================================
// // // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // // ============================================================================

// // template<int BM, int BN, int BK, int TM, int TN>
// // __global__ void matmul_backward_dA_fp32(
// //     const float* __restrict__ grad_out,  // [M, N]
// //     const float* __restrict__ B,         // [K, N]
// //     float* __restrict__ grad_A,          // [M, K]
// //     float beta,
// //     int M, int N, int K,
// //     int total_batches,
// //     BackwardMetadata meta) 
// // {
// //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// //    if (b_idx >= total_batches) return;
// //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// //    const float *GO = grad_out + go_offset;
// //    const float *Bp = B + b_offset;
// //    float *GA = grad_A + ga_offset;
   
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// //    __shared__ float As[2][BK][BM + BACKWARD_PAD];
// //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
// //    float results[16] = {0.0f}, regM[4], regN[4];

// //    auto load_tiles = [&](int no, int idx) {
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// //          int gm = by * BM + r, gn = no + c;
// //          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// //       }
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// //          int gn = no + r, gk = bx * BN + c;
// //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
// //       }
// //    };

// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int bk = 0; bk < N; bk += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (bk + BK < N) load_tiles(bk + BK, wi);
      
// //       #pragma unroll
// //       for (int d = 0; d < BK; d++) {
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// //          #pragma unroll
// //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) 
// //             for (int j = 0; j < 4; j++) 
// //                results[i*4+j] += regM[i] * regN[j];
// //       }
// //       __syncthreads();
// //    }
   
// //    for (int i = 0; i < 4; i++) {
// //       for (int j = 0; j < 4; j++) {
// //          int r = by * BM + tRow * 4 + i;
// //          int c = bx * BN + tCol * 4 + j;
// //          if (r < M && c < K) {
// //             int idx = r * s_ga_m + c * s_ga_k;
// //             GA[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GA[idx]);
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_B = A^T @ grad_output
// // // ============================================================================

// // template<int BM, int BN, int BK, int TM, int TN>
// // __global__ void matmul_backward_dB_fp32(
// //     const float* __restrict__ A,         // [M, K]
// //     const float* __restrict__ grad_out,  // [M, N]
// //     float* __restrict__ grad_B,          // [K, N]
// //     float beta,
// //     int M, int K, int N,
// //     int total_batches,
// //     BackwardMetadata meta)
// // {
// //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// //    if (b_idx >= total_batches) return;
// //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// //    int a_offset = 0, go_offset = 0, gb_offset = 0;
// //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
// //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
// //    const float *Ap = A + a_offset;
// //    const float *GO = grad_out + go_offset;
// //    float *GB = grad_B + gb_offset;
   
// //    int s_a_m = meta.a_strides[meta.a_ndim-2];
// //    int s_a_k = meta.a_strides[meta.a_ndim-1];
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// //    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
// //    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

// //    __shared__ float As[2][BK][BM + BACKWARD_PAD];
// //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
// //    float results[16] = {0.0f}, regM[4], regN[4];

// //    auto load_tiles = [&](int mo, int idx) {
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// //          int gk = by * BM + r, gm = mo + c;
// //          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
// //       }
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// //          int gm = mo + r, gn = bx * BN + c;
// //          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// //       }
// //    };

// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int bk = 0; bk < M; bk += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (bk + BK < M) load_tiles(bk + BK, wi);
      
// //       #pragma unroll
// //       for (int d = 0; d < BK; d++) {
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// //          #pragma unroll
// //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) 
// //             for (int j = 0; j < 4; j++) 
// //                results[i*4+j] += regM[i] * regN[j];
// //       }
// //       __syncthreads();
// //    }
   
// //    for (int i = 0; i < 4; i++) {
// //       for (int j = 0; j < 4; j++) {
// //          int r = by * BM + tRow * 4 + i;
// //          int c = bx * BN + tCol * 4 + j;
// //          if (r < K && c < N) {
// //             int idx = r * s_gb_k + c * s_gb_n;
// //             GB[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GB[idx]);
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // FP16 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // // ============================================================================

// // template<int BM, int BN, int BK, int WM, int WN>
// // __global__ void matmul_backward_dA_fp16(
// //     const __half* __restrict__ grad_out,
// //     const __half* __restrict__ B,
// //     __half* __restrict__ grad_A,
// //     float beta,
// //     int M, int N, int K,
// //     int total_batches,
// //     BackwardMetadata meta)
// // {
// //    const int batch_idx = blockIdx.z; 
// //    if (batch_idx >= total_batches) return;
// //    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
// //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// //    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// //    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// //    const __half *GO = grad_out + go_offset;
// //    const __half *Bp = B + b_offset;
// //    __half *GA = grad_A + ga_offset;
   
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// //    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
// //    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
// //    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
// //    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
// //    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
// //    #pragma unroll
// //    for(int i = 0; i < 2; i++) 
// //       for(int j = 0; j < 2; j++) 
// //          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

// //    auto load_tiles = [&](int no, int idx) {
// //       for (int i = tid; i < BM * BK; i += 512) {
// //          int r = i / BK, c = i % BK;
// //          int gm = blockIdx.y * BM + r, gn = no + c;
// //          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
// //       }
// //       for (int i = tid; i < BK * BN; i += 512) {
// //          int r = i / BN, c = i % BN;
// //          int gn = no + r, gk = blockIdx.x * BN + c;
// //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
// //       }
// //    };
   
// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int k = 0; k < N; k += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (k + BK < N) load_tiles(k + BK, wi);
      
// //       #pragma unroll
// //       for (int ks = 0; ks < BK; ks += 16) {
// //          #pragma unroll
// //          for (int i = 0; i < 2; i++) {
// //             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
// //             #pragma unroll
// //             for (int j = 0; j < 2; j++) {
// //                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
// //                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
// //             }
// //          }
// //       }
// //       __syncthreads();
// //    }
   
// //    __half* sm = reinterpret_cast<__half*>(As);
// //    #pragma unroll
// //    for (int i = 0; i < 2; i++) {
// //       for (int j = 0; j < 2; j++) {
// //          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
// //          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
// //          if (cr < M && cc < K) {
// //             __half* wsm = sm + warp_id * 256;
// //             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
// //             for (int r = 0; r < 16; r++) 
// //                for (int c = 0; c < 16; c++)
// //                   if (cr + r < M && cc + c < K) {
// //                      int idx = (cr + r) * s_ga_m + (cc + c) * s_ga_k;
// //                      GA[idx] = wsm[r * 16 + c] + (__half)beta * GA[idx];
// //                   }
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // DISPATCH LAYER
// // // ============================================================================

// // template<typename T>
// // void launch_backward_matmul(
// //     const Tensor& grad_output,
// //     const Tensor& A,
// //     const Tensor& B,
// //     Tensor& grad_A,
// //     Tensor& grad_B,
// //     float beta_a,
// //     float beta_b,
// //     cudaStream_t stream)
// // {
// //    const auto& go_sh = grad_output.shape().dims;
// //    const auto& a_sh = A.shape().dims;
// //    const auto& b_sh = B.shape().dims;
   
// //    int go_ndim = go_sh.size();
// //    int a_ndim = a_sh.size();
// //    int b_ndim = b_sh.size();
   
// //    int M = go_sh[go_ndim - 2];
// //    int N = go_sh[go_ndim - 1];
// //    int K_a = a_sh[a_ndim - 1];
   
// //    int tb = 1;
// //    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
// //    BackwardMetadata meta;
// //    meta.a_ndim = a_ndim;
// //    meta.b_ndim = b_ndim;
// //    meta.grad_out_ndim = go_ndim;
   
// //    for (int i = 0; i < a_ndim; i++) {
// //       meta.a_shape[i] = a_sh[i];
// //       meta.a_strides[i] = A.stride().strides[i];
// //       meta.grad_a_strides[i] = grad_A.stride().strides[i];
// //    }
// //    for (int i = 0; i < b_ndim; i++) {
// //       meta.b_shape[i] = b_sh[i];
// //       meta.b_strides[i] = B.stride().strides[i];
// //       meta.grad_b_strides[i] = grad_B.stride().strides[i];
// //    }
// //    for (int i = 0; i < go_ndim; i++) {
// //       meta.grad_out_shape[i] = go_sh[i];
// //       meta.grad_out_strides[i] = grad_output.stride().strides[i];
// //    }
   
// //    const T* go_ptr = grad_output.data<T>();
// //    const T* a_ptr = A.data<T>();
// //    const T* b_ptr = B.data<T>();
// //    T* ga_ptr = grad_A.data<T>();
// //    T* gb_ptr = grad_B.data<T>();
   
// //    bool supported_for_fast_path = (meta.grad_out_strides[go_ndim-1] == 1) && 
// //                                   (meta.a_strides[a_ndim-1] == 1) && 
// //                                   (meta.b_strides[b_ndim-1] == 1) && 
// //                                   (meta.grad_a_strides[a_ndim-1] == 1) && 
// //                                   (meta.grad_b_strides[b_ndim-1] == 1);

// //     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");

// //    // ---- Acquire device, handles, and secondary stream once ----
// //    // grad_A = grad_output @ B^T  and  grad_B = A^T @ grad_output are fully independent.
// //    // Dispatch them on separate cuBLAS handles bound to separate streams so the GPU
// //    // can overlap them.  Events are cached per-device and use DisableTiming for
// //    // minimal overhead.
// //    int dev = 0;
// //    cudaGetDevice(&dev);
// //    mycublasHandle_t handle_a = get_mycublas_handle_bwd(dev);
// //    mycublasHandle_t handle_b = get_mycublas_handle_bwd2(dev);
// //    cudaStream_t stream_b     = get_secondary_stream_bwd(dev);
// //    init_bwd_events(dev);

// //    mycublasSetStream(handle_a, stream);
// //    mycublasSetStream(handle_b, stream_b);

// //    // stream_b must wait until all upstream data (grad_output, A, B) produced on
// //    // the primary stream is visible before it starts reading them.
// //    cudaEventRecord(g_bwd_event_ready[dev], stream);
// //    cudaStreamWaitEvent(stream_b, g_bwd_event_ready[dev], 0);

// //    // ---- grad_A = grad_output @ B^T  (primary stream) ----
// //    bool grad_a_done = false;
// //    if constexpr (std::is_same<T, float>::value) {
// //        if (supported_for_fast_path) {
// //           int ldb  = meta.b_strides[b_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldga = meta.grad_a_strides[a_ndim-2];
// //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
          
// //           mycublasSgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, 1.0f, go_ptr, ldgo, s_go, b_ptr, ldb, s_b, beta_a, ga_ptr, ldga, s_ga, tb);
// //           grad_a_done = true;
// //        }
// //    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
// //        if (supported_for_fast_path) {
// //           int ldb  = meta.b_strides[b_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldga = meta.grad_a_strides[a_ndim-2];
// //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
// //           const __half alpha = __float2half(1.0f), h_beta_a = __float2half(beta_a);
// //           mycublasHgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, alpha, (const __half*)go_ptr, ldgo, s_go, (const __half*)b_ptr, ldb, s_b, h_beta_a, (__half*)ga_ptr, ldga, s_ga, tb);
// //           grad_a_done = true;
// //        }
// //    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
// //        if (supported_for_fast_path) {
// //           int ldb  = meta.b_strides[b_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldga = meta.grad_a_strides[a_ndim-2];
// //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
// //           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), b_beta_a = __float2bfloat16(beta_a);
// //           mycublasBgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, alpha, (const __nv_bfloat16*)go_ptr, ldgo, s_go, (const __nv_bfloat16*)b_ptr, ldb, s_b, b_beta_a, (__nv_bfloat16*)ga_ptr, ldga, s_ga, tb);
// //           grad_a_done = true;
// //        }
// //    }

// //    if (!grad_a_done) {
// //        if constexpr (std::is_same<T, float>::value)
// //           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, beta_a, M, N, K_a, tb, meta);
// //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
// //           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), beta_a, M, N, K_a, tb, meta);
// //    }

// //    // ---- grad_B = A^T @ grad_output  (secondary stream — runs in parallel with grad_A above) ----
// //    bool grad_b_done = false;
// //    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[b_ndim-3] == 1 && meta.grad_out_shape[go_ndim-3] > 1);

// //    if constexpr (std::is_same<T, float>::value) {
// //        if (supported_for_fast_path) {
// //           int lda  = meta.a_strides[a_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldgb = meta.grad_b_strides[b_ndim-2];
// //           if (need_reduction) {
// //               int M_total = M * tb;
// //               // TN layout but M_total as K
// //               mycublasSgemm(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, 1.0f, a_ptr, lda, go_ptr, ldgo, beta_b, gb_ptr, ldgb);
// //           } else {
// //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]            : 0;
// //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3]    : 0;
// //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]       : 0;
             
// //              mycublasSgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, 1.0f, a_ptr, lda, s_a, go_ptr, ldgo, s_go, beta_b, gb_ptr, ldgb, s_gb, tb);
// //           }
// //           grad_b_done = true;
// //        }
// //    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
// //        if (supported_for_fast_path) {
// //           int lda  = meta.a_strides[a_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldgb = meta.grad_b_strides[b_ndim-2];
// //           const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
// //           if (need_reduction) {
// //               int M_total = M * tb;
// //               const __half h_beta_b = __float2half(beta_b);
// //               mycublasHgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __half*)a_ptr, lda, 0, (const __half*)go_ptr, ldgo, 0, h_beta_b, (__half*)gb_ptr, ldgb, 0, 1);
// //           } else {
// //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
// //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
// //              mycublasHgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __half*)a_ptr, lda, s_a, (const __half*)go_ptr, ldgo, s_go, beta, (__half*)gb_ptr, ldgb, s_gb, tb);
// //           }
// //           grad_b_done = true;
// //        }
// //    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
// //        if (supported_for_fast_path) {
// //           int lda  = meta.a_strides[a_ndim-2];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldgb = meta.grad_b_strides[b_ndim-2];
// //           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(0.0f);
// //           if (need_reduction) {
// //               int M_total = M * tb;
// //               const __nv_bfloat16 b_beta_b = __float2bfloat16(beta_b);
// //               mycublasBgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __nv_bfloat16*)a_ptr, lda, 0, (const __nv_bfloat16*)go_ptr, ldgo, 0, b_beta_b, (__nv_bfloat16*)gb_ptr, ldgb, 0, 1);
// //           } else {
// //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
// //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
// //              mycublasBgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __nv_bfloat16*)a_ptr, lda, s_a, (const __nv_bfloat16*)go_ptr, ldgo, s_go, beta, (__nv_bfloat16*)gb_ptr, ldgb, s_gb, tb);
// //           }
// //           grad_b_done = true;
// //        }
// //    }

// //    if (!grad_b_done) {
// //        if constexpr (std::is_same<T, float>::value)
// //           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream_b>>>(a_ptr, go_ptr, gb_ptr, beta_b, M, K_a, N, tb, meta);
// //    }

// //    // Rejoin stream_b into the primary stream so downstream ops on stream see grad_B.
// //    cudaEventRecord(g_bwd_event_done[dev], stream_b);
// //    cudaStreamWaitEvent(stream, g_bwd_event_done[dev], 0);

// //    cudaError_t err = cudaGetLastError();
// //    if (err != cudaSuccess) {
// //       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
// //    }
// // }

// // void cuda_matmul_backward(
// //     const Tensor& grad_output,
// //     const Tensor& A,
// //     const Tensor& B,
// //     Tensor& grad_A,
// //     Tensor& grad_B,
// //     cudaStream_t stream)
// // {
// //    float beta_a = 0.0f;
// //    float beta_b = 0.0f;
   
// //    dispatch_by_dtype(A.dtype(), [&](auto d) {
// //       using T = decltype(d);
// //       launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, beta_a, beta_b, stream);
// //    });
// // }

// // } // namespace OwnTensor

// // #else // WITH_MYBLAS


// // #include <cuda_runtime.h>
// // #include <cuda_fp16.h>
// // #include <cuda_bf16.h>
// // #include <mma.h>
// // #include <cublas_v2.h>
// // #include <algorithm>
// // #include <stdexcept>
// // #include <vector>
// // #include <string>
// // #include <mutex>

// // #include "ops/MatmulBackward.cuh"
// // #include "core/Tensor.h"
// // #include "core/TensorDispatch.h"
// // #include "utils/Profiler.h"

// // namespace OwnTensor {

// // // cuBLAS handle management - thread-safe singleton per device
// // static std::mutex cublas_mutex_bwd;
// // static cublasHandle_t g_cublas_handles_bwd[8] = {nullptr};

// // static cublasHandle_t get_cublas_handle_bwd(int device = 0) {
// //    if (g_cublas_handles_bwd[device] == nullptr) {
// //       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
// //       if (g_cublas_handles_bwd[device] == nullptr) {
// //          cudaSetDevice(device);
// //          cublasCreate(&g_cublas_handles_bwd[device]);
// //          // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
// //          cublasSetMathMode(g_cublas_handles_bwd[device], CUBLAS_TF32_TENSOR_OP_MATH);
// //       }
// //    }
// //    return g_cublas_handles_bwd[device];
// // }

// // using namespace nvcuda;

// // // ============================================================================
// // // METADATA & CONSTANTS (Reuse from forward)
// // // ============================================================================

// // struct BackwardMetadata {
// //    int a_shape[8], b_shape[8], grad_out_shape[8];
// //    int a_strides[8], b_strides[8], grad_out_strides[8];
// //    int grad_a_strides[8], grad_b_strides[8];
// //    int a_ndim, b_ndim, grad_out_ndim;
// // };

// // __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
// //                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
// //    offset = 0; 
// //    if (out_ndim <= 2) return;
// //    int temp_batch = batch_idx;
// //    for (int dim = out_ndim - 3; dim >= 0; --dim) {
// //       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
// //       temp_batch /= b_dim_sz;
// //       int c_dim = dim - (out_ndim - ndim);
// //       if (c_dim >= 0 && c_dim < ndim - 2) 
// //          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
// //    }
// // }

// // constexpr int BACKWARD_PAD = 8;

// // // ============================================================================
// // // FP32 BACKWARD KERNEL: grad_A = grad_output @ B^T
// // // Computes: C[M,K] = A[M,N] @ B[K,N]^T = A[M,N] @ B^T[N,K]
// // // Note: We read B in transposed order (columns become rows)
// // // BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// // // ============================================================================

// // template<int BM, int BN, int BK, int TM, int TN>
// // __global__ void matmul_backward_dA_fp32(
// //     const float* __restrict__ grad_out,  // [M, N] - grad of output
// //     const float* __restrict__ B,         // [K, N] - second input of forward, read transposed
// //     float* __restrict__ grad_A,          // [M, K] - gradient for A
// //     int M, int N, int K,                 // M=rows of grad_out, N=cols of grad_out (=cols of B), K=rows of B
// //     int total_batches,
// //     BackwardMetadata meta) 
// // {
// //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// //    if (b_idx >= total_batches) return;
// //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// //    const float *GO = grad_out + go_offset;
// //    const float *Bp = B + b_offset;
// //    float *GA = grad_A + ga_offset;
   
// //    // Strides for grad_output [M, N]
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
// //    // Strides for B [K, N] - we read it transposed as B^T [N, K]
// //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// //    int s_b_n = meta.b_strides[meta.b_ndim-1];
   
// //    // Strides for grad_A [M, K]
// //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// //    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // grad_out tile: [BK, BM] transposed storage
// //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // B^T tile: [BK, BN]
// //    float results[16] = {0.0f}, regM[4], regN[4];

// //    // Load tiles with double buffering
// //    // For grad_A = grad_out @ B^T:
// //    //   We iterate over N (the shared dimension between grad_out and B^T)
// //    //   grad_out is [M, N], B^T is [N, K]
// //    //   So we load BK columns of grad_out into shared As
// //    //   And BK rows of B^T (= BK columns of B) into shared Bs
   
// //    auto load_tiles = [&](int no, int idx) {
// //       // Load grad_out[by*BM : by*BM+BM, no : no+BK] into As (transposed for coalescing)
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// //          int gm = by * BM + r, gn = no + c;
// //          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// //       }
      
// //       // Load B^T[no : no+BK, bx*BN : bx*BN+BN] = B[bx*BN : bx*BN+BN, no : no+BK]^T
// //       // B is [K, N], B^T is [N, K]. We want B^T[n, k] = B[k, n]
// //       // So for k in [bx*BN, bx*BN+BN) and n in [no, no+BK)
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// //          int gn = no + r, gk = bx * BN + c;
// //          // B^T[gn, gk] = B[gk, gn]
// //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
// //       }
// //    };

// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int bk = 0; bk < N; bk += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (bk + BK < N) load_tiles(bk + BK, wi);
      
// //       #pragma unroll
// //       for (int d = 0; d < BK; d++) {
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// //          #pragma unroll
// //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) 
// //             for (int j = 0; j < 4; j++) 
// //                results[i*4+j] += regM[i] * regN[j];
// //       }
// //       __syncthreads();
// //    }
   
// //    // Store results to grad_A
// //    for (int i = 0; i < 4; i++) {
// //       for (int j = 0; j < 4; j++) {
// //          int r = by * BM + tRow * 4 + i;
// //          int c = bx * BN + tCol * 4 + j;
// //          if (r < M && c < K) {
// //             GA[r * s_ga_m + c * s_ga_k] = results[i*4+j];
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // FP32 BACKWARD KERNEL: grad_B = A^T @ grad_output
// // // Computes: C[K,N] = A[M,K]^T @ B[M,N] = A^T[K,M] @ B[M,N]
// // // Note: We read A in transposed order
// // // ============================================================================

// // template<int BM, int BN, int BK, int TM, int TN>
// // __global__ void matmul_backward_dB_fp32(
// //     const float* __restrict__ A,         // [M, K] - first input of forward, read transposed
// //     const float* __restrict__ grad_out,  // [M, N] - grad of output
// //     float* __restrict__ grad_B,          // [K, N] - gradient for B
// //     int M, int K, int N,                 // M=rows of A, K=cols of A, N=cols of grad_out
// //     int total_batches,
// //     BackwardMetadata meta)
// // {
// //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// //    if (b_idx >= total_batches) return;
// //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// //    int a_offset = 0, go_offset = 0, gb_offset = 0;
// //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
// //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
// //                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
// //    const float *Ap = A + a_offset;
// //    const float *GO = grad_out + go_offset;
// //    float *GB = grad_B + gb_offset;
   
// //    // Strides for A [M, K] - we read it transposed as A^T [K, M]
// //    int s_a_m = meta.a_strides[meta.a_ndim-2];
// //    int s_a_k = meta.a_strides[meta.a_ndim-1];
   
// //    // Strides for grad_output [M, N]
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
// //    // Strides for grad_B [K, N]
// //    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
// //    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

// //    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // A^T tile: [BK, BM]
// //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // grad_out tile: [BK, BN]
// //    float results[16] = {0.0f}, regM[4], regN[4];

// //    // For grad_B = A^T @ grad_out:
// //    //   We iterate over M (the shared dimension)
// //    //   A^T is [K, M], grad_out is [M, N]
// //    //   So we load BK rows of A^T (= BK columns of A) into shared As
// //    //   And BK rows of grad_out into shared Bs
   
// //    auto load_tiles = [&](int mo, int idx) {
// //       // Load A^T[by*BM : by*BM+BM, mo : mo+BK] = A[mo : mo+BK, by*BM : by*BM+BM]^T
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// //          int gk = by * BM + r, gm = mo + c;
// //          // A^T[gk, gm] = A[gm, gk]
// //          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
// //       }
      
// //       // Load grad_out[mo : mo+BK, bx*BN : bx*BN+BN] into Bs
// //       #pragma unroll
// //       for (int i = 0; i < 2; i++) {
// //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// //          int gm = mo + r, gn = bx * BN + c;
// //          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// //       }
// //    };

// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int bk = 0; bk < M; bk += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (bk + BK < M) load_tiles(bk + BK, wi);
      
// //       #pragma unroll
// //       for (int d = 0; d < BK; d++) {
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// //          #pragma unroll
// //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// //          #pragma unroll
// //          for (int i = 0; i < 4; i++) 
// //             for (int j = 0; j < 4; j++) 
// //                results[i*4+j] += regM[i] * regN[j];
// //       }
// //       __syncthreads();
// //    }
   
// //    // Store results to grad_B
// //    for (int i = 0; i < 4; i++) {
// //       for (int j = 0; j < 4; j++) {
// //          int r = by * BM + tRow * 4 + i;
// //          int c = bx * BN + tCol * 4 + j;
// //          if (r < K && c < N) {
// //             GB[r * s_gb_k + c * s_gb_n] = results[i*4+j];
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // FP16 BACKWARD KERNEL: grad_A = grad_output @ B^T (WMMA)
// // // ============================================================================

// // template<int BM, int BN, int BK, int WM, int WN>
// // __global__ void matmul_backward_dA_fp16(
// //     const __half* __restrict__ grad_out,
// //     const __half* __restrict__ B,
// //     __half* __restrict__ grad_A,
// //     int M, int N, int K,
// //     int total_batches,
// //     BackwardMetadata meta)
// // {
// //    const int batch_idx = blockIdx.z; 
// //    if (batch_idx >= total_batches) return;
// //    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
// //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// //    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// //    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// //    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
// //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// //    const __half *GO = grad_out + go_offset;
// //    const __half *Bp = B + b_offset;
// //    __half *GA = grad_A + ga_offset;
   
// //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// //    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
// //    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
// //    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
// //    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
// //    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
// //    #pragma unroll
// //    for(int i = 0; i < 2; i++) 
// //       for(int j = 0; j < 2; j++) 
// //          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

// //    auto load_tiles = [&](int no, int idx) {
// //       for (int i = tid; i < BM * BK; i += 512) {
// //          int r = i / BK, c = i % BK;
// //          int gm = blockIdx.y * BM + r, gn = no + c;
// //          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
// //       }
// //       for (int i = tid; i < BK * BN; i += 512) {
// //          int r = i / BN, c = i % BN;
// //          int gn = no + r, gk = blockIdx.x * BN + c;
// //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
// //       }
// //    };
   
// //    int wi = 0; 
// //    load_tiles(0, wi); 
// //    __syncthreads();
   
// //    for (int k = 0; k < N; k += BK) {
// //       int ri = wi; 
// //       wi = 1 - wi;
// //       if (k + BK < N) load_tiles(k + BK, wi);
      
// //       #pragma unroll
// //       for (int ks = 0; ks < BK; ks += 16) {
// //          #pragma unroll
// //          for (int i = 0; i < 2; i++) {
// //             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
// //             #pragma unroll
// //             for (int j = 0; j < 2; j++) {
// //                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
// //                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
// //             }
// //          }
// //       }
// //       __syncthreads();
// //    }
   
// //    __half* sm = reinterpret_cast<__half*>(As);
// //    #pragma unroll
// //    for (int i = 0; i < 2; i++) {
// //       for (int j = 0; j < 2; j++) {
// //          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
// //          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
// //          if (cr < M && cc < K) {
// //             __half* wsm = sm + warp_id * 256;
// //             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
// //             for (int r = 0; r < 16; r++) 
// //                for (int c = 0; c < 16; c++)
// //                   if (cr + r < M && cc + c < K) 
// //                      GA[(cr + r) * s_ga_m + (cc + c) * s_ga_k] = wsm[r * 16 + c];
// //          }
// //       }
// //    }
// // }

// // // ============================================================================
// // // DISPATCH LAYER
// // // ============================================================================

// // template<typename T>
// // void launch_backward_matmul(
// //     const Tensor& grad_output,
// //     const Tensor& A,
// //     const Tensor& B,
// //     Tensor& grad_A,
// //     Tensor& grad_B,
// //     cudaStream_t stream)
// // {
// //    const auto& go_sh = grad_output.shape().dims;
// //    const auto& a_sh = A.shape().dims;
// //    const auto& b_sh = B.shape().dims;
   
// //    int go_ndim = go_sh.size();
// //    int a_ndim = a_sh.size();
// //    int b_ndim = b_sh.size();
   
// //    int M = go_sh[go_ndim - 2];  // rows of grad_output
// //    int N = go_sh[go_ndim - 1];  // cols of grad_output (= cols of B)
// //    int K_a = a_sh[a_ndim - 1];  // cols of A (= rows of grad_A's cols dim)
// //    int K_b = b_sh[b_ndim - 2];  // rows of B
   
// //    int tb = 1;
// //    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
// //    BackwardMetadata meta;
// //    meta.a_ndim = a_ndim;
// //    meta.b_ndim = b_ndim;
// //    meta.grad_out_ndim = go_ndim;
   
// //    for (int i = 0; i < a_ndim; i++) {
// //       meta.a_shape[i] = a_sh[i];
// //       meta.a_strides[i] = A.stride().strides[i];
// //       meta.grad_a_strides[i] = grad_A.stride().strides[i];
// //    }
// //    for (int i = 0; i < b_ndim; i++) {
// //       meta.b_shape[i] = b_sh[i];
// //       meta.b_strides[i] = B.stride().strides[i];
// //       meta.grad_b_strides[i] = grad_B.stride().strides[i];
// //    }
// //    for (int i = 0; i < go_ndim; i++) {
// //       meta.grad_out_shape[i] = go_sh[i];
// //       meta.grad_out_strides[i] = grad_output.stride().strides[i];
// //    }
   
// //    const T* go_ptr = grad_output.data<T>();
// //    const T* a_ptr = A.data<T>();
// //    const T* b_ptr = B.data<T>();
// //    T* ga_ptr = grad_A.data<T>();
// //    T* gb_ptr = grad_B.data<T>();
   
// //    // Check stride attributes
// //    bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
// //    bool a_contiguous = (meta.a_strides[a_ndim-1] == 1);
// //    bool a_transposed = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
// //    bool b_contiguous = (meta.b_strides[b_ndim-1] == 1);
// //    bool b_transposed = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
// //    bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
// //    bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

// //    bool supported_for_fast_path = go_contiguous && (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && ga_contiguous && gb_contiguous;

// //     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");
// //     bool grad_a_done = false;
// //     if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// //                   std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// //         // cuBLAS path for FP16/BF16: grad_A = grad_output @ B^T
// //         // compute in fp16/bf16, accumulate in fp32
// //         if (supported_for_fast_path) {
// //             int dev_idx = grad_output.device().index;
// //             cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// //             cublasSetStream(handle, stream);
// //             float alpha_f = 1.0f, beta_f = 0.0f;

// //             cublasOperation_t opB_bwd = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// //             int ldb_bwd = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
// //             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
// //             int ldga_bwd = meta.grad_a_strides[a_ndim-2];

// //             long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3] : 0LL;
// //             long long stride_b_ll  = (b_ndim  >= 3) ? (long long)meta.b_strides[b_ndim-3]         : 0LL;
// //             long long stride_ga_ll = (a_ndim  >= 3) ? (long long)meta.grad_a_strides[a_ndim-3]    : 0LL;
// //             if (b_ndim < go_ndim) stride_b_ll = 0LL;

// //             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
// //                                        ? CUDA_R_16F : CUDA_R_16BF;
// //             cublasStatus_t status;
// //             if (tb == 1) {
// //                 status = cublasGemmEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
// //                     b_ptr,  cuda_type, ldb_bwd,
// //                     go_ptr, cuda_type, ldgo_bwd,
// //                     &beta_f, ga_ptr, cuda_type, ldga_bwd,
// //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //             } else {
// //                 status = cublasGemmStridedBatchedEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
// //                     b_ptr,  cuda_type, ldb_bwd,  stride_b_ll,
// //                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
// //                     &beta_f, ga_ptr, cuda_type, ldga_bwd, stride_ga_ll, tb,
// //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //             }
// //             if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
// //         }
// //     } else if constexpr (std::is_same<T, float>::value) {
// //         if (supported_for_fast_path) {
// //           // USE EXPLICIT DEVICE INDEXING from grad_output
// //           int dev_idx = grad_output.device().index;
// //           cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// //           cublasSetStream(handle, stream);
// //           float alpha = 1.0f, beta = 0.0f;
          
// //           cublasOperation_t opB = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// //           int ldb = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldga = meta.grad_a_strides[a_ndim-2];
          
// //           // Compute batch strides from actual tensor metadata
// //           long long stride_go, stride_b_val, stride_ga;
// //           bool batch_uniform_dA = true;

// //           if (go_ndim <= 3) {
// //               stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
// //               stride_b_val = (b_ndim == 3) ? meta.b_strides[0] : 0;
// //               stride_ga = (go_ndim == 3) ? meta.grad_a_strides[0] : 0;
// //               if (b_ndim < go_ndim) stride_b_val = 0;
// //           } else {
// //               // 4D: use inner batch stride (strides[ndim-3])
// //               stride_go = meta.grad_out_strides[go_ndim - 3];
// //               stride_b_val = (b_ndim >= 4) ? meta.b_strides[b_ndim - 3] : 0;
// //               stride_ga = meta.grad_a_strides[a_ndim - 3];
// //               if (b_ndim < go_ndim) stride_b_val = 0;

// //               // Check uniform batch strides: stride[0] == stride[1] * shape[1]
// //               batch_uniform_dA = (meta.grad_out_strides[0] == stride_go * go_sh[1]);
// //               if (b_ndim >= 4) batch_uniform_dA = batch_uniform_dA && (meta.b_strides[0] == stride_b_val * b_sh[1]);
// //               batch_uniform_dA = batch_uniform_dA && (meta.grad_a_strides[0] == stride_ga * a_sh[1]);
// //           }

// //           cublasStatus_t status;
// //           if (tb == 1) {
// //               status = cublasGemmEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, go_ptr, CUDA_R_32F, ldgo, &beta, ga_ptr, CUDA_R_32F, ldga, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //           } else if (batch_uniform_dA) {
// //               status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, stride_b_val, go_ptr, CUDA_R_32F, ldgo, stride_go, &beta, ga_ptr, CUDA_R_32F, ldga, stride_ga, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //           } else {
// //               // Non-uniform 4D: loop over outer batch dim
// //               int d0 = go_sh[0], d1 = go_sh[1];
// //               status = CUBLAS_STATUS_SUCCESS;
// //               for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
// //                   const T* b_batch = b_ptr + (b_ndim >= 4 ? bi * meta.b_strides[0] : 0);
// //                   const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
// //                   T* ga_batch = ga_ptr + bi * meta.grad_a_strides[0];
// //                   status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_batch, CUDA_R_32F, ldb, stride_b_val, go_batch, CUDA_R_32F, ldgo, stride_go, &beta, ga_batch, CUDA_R_32F, ldga, stride_ga, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //               }
// //           }

// //           if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
// //        }
// //    }
   
// //    if (!grad_a_done) {
// //         #ifdef AUTOGRAD_PROFILER_ENABLED
// //         if (M*N*K_a > 1000000) {
// //             printf("dA Fallback: M=%d N=%d K=%d, Strides: go[%d,%d], b[%d,%d], ga[%d,%d]\n", 
// //                    M, N, K_a, meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
// //                    meta.b_strides[b_ndim-2], meta.b_strides[b_ndim-1],
// //                    meta.grad_a_strides[a_ndim-2], meta.grad_a_strides[a_ndim-1]);
// //         }
// //         #endif
// //        // Fallback
// //        if constexpr (std::is_same<T, float>::value)
// //           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, M, N, K_a, tb, meta);
// //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
// //           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), M, N, K_a, tb, meta);
// //        // bf16 fallback: cuBLAS path above handles all standard layouts; non-standard bf16 not supported
// //    }

// //    // Launch grad_B kernel: grad_B = A^T @ grad_output
// //    // Handle Reduction if B dim < Output dim (Broadcast in forward)
   
// //    bool grad_b_actions_needed = true; // Could optimize if not req grad
// //    bool grad_b_done = false;
   
// //    // Check reduction need
// //    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[0] == 1);
// //    // Standard broadcast check: if tb > 1 but grad_B is not batched.
// //    // Assuming grad_B matches B shape.
// //    if (tb > 1 && b_ndim < go_ndim) need_reduction = true;

// //    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// //                  std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// //         // cuBLAS path for FP16/BF16: grad_B = A^T @ grad_output
// //         // compute in fp16/bf16, accumulate in fp32
// //         if (supported_for_fast_path) {
// //             int dev_idx = grad_output.device().index;
// //             cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// //             cublasSetStream(handle, stream);
// //             float alpha_f = 1.0f, beta_f = 0.0f;

// //             cublasOperation_t opA_bwd = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// //             int lda_bwd  = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];
// //             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
// //             int ldgb_bwd = meta.grad_b_strides[b_ndim-2];

// //             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
// //                                        ? CUDA_R_16F : CUDA_R_16BF;
// //             cublasStatus_t status;
// //             if (need_reduction) {
// //                 // Flatten batch: treat A as [M*tb, K], grad_out as [M*tb, N]
// //                 int M_total = M * tb;
// //                 status = cublasGemmEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M_total, &alpha_f,
// //                     go_ptr, cuda_type, ldgo_bwd,
// //                     a_ptr,  cuda_type, lda_bwd,
// //                     &beta_f, gb_ptr, cuda_type, ldgb_bwd,
// //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //             } else {
// //                 long long stride_a_ll  = (a_ndim  >= 3) ? (long long)meta.a_strides[a_ndim-3]       : 0LL;
// //                 long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3]: 0LL;
// //                 long long stride_gb_ll = (b_ndim  >= 3) ? (long long)meta.grad_b_strides[b_ndim-3]  : 0LL;
// //                 status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M, &alpha_f,
// //                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
// //                     a_ptr,  cuda_type, lda_bwd,  stride_a_ll,
// //                     &beta_f, gb_ptr, cuda_type, ldgb_bwd, stride_gb_ll, tb,
// //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //             }
// //             if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
// //         }
// //    } else if constexpr (std::is_same<T, float>::value) {
// //         if (supported_for_fast_path) {
// //           // USE EXPLICIT DEVICE INDEXING from grad_output
// //           int dev_idx = grad_output.device().index;
// //           cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// //           cublasSetStream(handle, stream);
// //           float alpha = 1.0f, beta = 0.0f;

// //           cublasOperation_t opA = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N; // A^T
// //           int lda = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1]; // Correct stride 
          
// //           int ldgo = meta.grad_out_strides[go_ndim-2];
// //           int ldgb = meta.grad_b_strides[b_ndim-2];
          
// //           cublasStatus_t status;
          
// //           if (need_reduction) {
// //               // Flatten Batch strategy: Treat A as [TotalRows, K], GO as [TotalRows, N]
// //               // grad_B = A^T @ GO.
// //               // A^T [K, TotalRows]. GO [TotalRows, N]. Result [K, N].
// //               int M_total = M * tb;
              
// //               status = cublasGemmEx(handle, CUBLAS_OP_N, opA, N, K_a, M_total, &alpha, go_ptr, CUDA_R_32F, ldgo, a_ptr, CUDA_R_32F, lda, &beta, gb_ptr, CUDA_R_32F, ldgb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              
// //           } else {
// //              // Batched Strategy — compute strides from actual metadata
// //              long long stride_a, stride_go, stride_gb;
// //              bool batch_uniform_dB = true;

// //              if (a_ndim <= 3) {
// //                  stride_a = (a_ndim == 3) ? meta.a_strides[0] : 0;
// //                  stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
// //                  stride_gb = (b_ndim == 3) ? meta.grad_b_strides[0] : 0;
// //              } else {
// //                  // 4D: use dim[-3] strides (inner batch stride)
// //                  stride_a = meta.a_strides[a_ndim - 3];
// //                  stride_go = meta.grad_out_strides[go_ndim - 3];
// //                  stride_gb = meta.grad_b_strides[b_ndim - 3];

// //                  // Check if outer batch stride == inner_stride * inner_dim
// //                  int* a_sh = meta.a_shape;
// //                  int* go_sh = meta.grad_out_shape;
// //                  int* b_sh = meta.b_shape;
// //                  batch_uniform_dB = (meta.a_strides[0] == stride_a * a_sh[1]);
// //                  batch_uniform_dB = batch_uniform_dB && (meta.grad_out_strides[0] == stride_go * go_sh[1]);
// //                  batch_uniform_dB = batch_uniform_dB && (meta.grad_b_strides[0] == stride_gb * b_sh[1]);
// //              }

// //              if (batch_uniform_dB) {
// //                  status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_ptr, CUDA_R_32F, ldgo, stride_go, a_ptr, CUDA_R_32F, lda, stride_a, &beta, gb_ptr, CUDA_R_32F, ldgb, stride_gb, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //              } else {
// //                  // Non-uniform 4D: loop over outer batch dim d0
// //                  int d0 = meta.grad_out_shape[0];
// //                  int d1 = meta.grad_out_shape[1];
// //                  status = CUBLAS_STATUS_SUCCESS;
// //                  for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
// //                      const T* a_batch = a_ptr + bi * meta.a_strides[0];
// //                      const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
// //                      T* gb_batch = gb_ptr + bi * meta.grad_b_strides[0];
// //                      status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_batch, CUDA_R_32F, ldgo, stride_go, a_batch, CUDA_R_32F, lda, stride_a, &beta, gb_batch, CUDA_R_32F, ldgb, stride_gb, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// //                  }
// //              }
// //           }
// //           if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
// //        }
// //    }
   
// //    if (!grad_b_done) {
// //         #ifdef AUTOGRAD_PROFILER_ENABLED
// //         if (M*N*K_a > 1000000) {
// //             printf("dB Fallback: M=%d K=%d N=%d, Strides: a[%d,%d], go[%d,%d], gb[%d,%d]\n", 
// //                    M, K_a, N, meta.a_strides[a_ndim-2], meta.a_strides[a_ndim-1],
// //                    meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
// //                    meta.grad_b_strides[b_ndim-2], meta.grad_b_strides[b_ndim-1]);
// //         }
// //         #endif
// //        // Fallback
// //        if constexpr (std::is_same<T, float>::value)
// //           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream>>>(a_ptr, go_ptr, gb_ptr, M, K_a, N, tb, meta);
// //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// //                           std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value)
// //           throw std::runtime_error("matmul_backward dB: non-contiguous fp16/bf16 fallback kernel not available, ensure contiguous tensors");
// //    }
   
// //    cudaError_t err = cudaGetLastError();
// //    if (err != cudaSuccess) {
// //       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
// //    }
// // }

// // void cuda_matmul_backward(
// //     const Tensor& grad_output,
// //     const Tensor& A,
// //     const Tensor& B,
// //     Tensor& grad_A,
// //     Tensor& grad_B,
// //     cudaStream_t stream)
// // {
// //    dispatch_by_dtype(A.dtype(), [&](auto d) {
// //       using T = decltype(d);
// //       if constexpr (std::is_same<T, float>::value ||
// //                     std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// //                     std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// //          launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, stream);
// //       } else {
// //          throw std::runtime_error("cuda_matmul_backward: Unsupported type (only float, float16, bfloat16 supported)");
// //       }
// //    });
// // }

// // } // namespace OwnTensor


// // #endif // WITH_MYBLAS

// // #endif // WITH_CUDA














// // // #ifdef WITH_CUDA

// // // #ifdef WITH_MYBLAS


// // // #include <cuda_runtime.h>
// // // #include <cuda_fp16.h>
// // // #include <cuda_bf16.h>
// // // #include <mma.h>
// // // #include "mycublas.h"
// // // #include <algorithm>
// // // #include <stdexcept>
// // // #include <vector>
// // // #include <string>
// // // #include <mutex>

// // // #include "ops/MatmulBackward.cuh"
// // // #include "core/Tensor.h"
// // // #include "core/TensorDispatch.h"
// // // #include "utils/Profiler.h"

// // // namespace OwnTensor {
// // // namespace matmul_backward_myblas {

// // // // mycublas handle management - thread-safe singleton per device
// // // static std::mutex mycublas_mutex_bwd;
// // // static mycublasHandle_t g_mycublas_handles_bwd[8]  = {nullptr};
// // // static mycublasHandle_t g_mycublas_handles_bwd2[8] = {nullptr};  // second handle for parallel grad_B stream

// // // // Per-device secondary stream and cached no-timing events for grad_A/grad_B overlap.
// // // static cudaStream_t g_secondary_streams_bwd[8]  = {nullptr};
// // // static cudaEvent_t  g_bwd_event_ready[8]        = {nullptr};   // primary → secondary (inputs ready)
// // // static cudaEvent_t  g_bwd_event_done[8]         = {nullptr};   // secondary → primary (grad_B done)

// // // static mycublasHandle_t get_mycublas_handle_bwd(int device = 0) {
// // //    if (g_mycublas_handles_bwd[device] == nullptr) {
// // //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// // //       if (g_mycublas_handles_bwd[device] == nullptr) {
// // //          cudaSetDevice(device);
// // //          mycublasCreate(&g_mycublas_handles_bwd[device]);
// // //       }
// // //    }
// // //    return g_mycublas_handles_bwd[device];
// // // }

// // // static mycublasHandle_t get_mycublas_handle_bwd2(int device = 0) {
// // //    if (g_mycublas_handles_bwd2[device] == nullptr) {
// // //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// // //       if (g_mycublas_handles_bwd2[device] == nullptr) {
// // //          cudaSetDevice(device);
// // //          mycublasCreate(&g_mycublas_handles_bwd2[device]);
// // //       }
// // //    }
// // //    return g_mycublas_handles_bwd2[device];
// // // }

// // // static cudaStream_t get_secondary_stream_bwd(int device = 0) {
// // //    if (g_secondary_streams_bwd[device] == nullptr) {
// // //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// // //       if (g_secondary_streams_bwd[device] == nullptr) {
// // //          cudaSetDevice(device);
// // //          // Non-blocking so it doesn't serialize with the default stream
// // //          cudaStreamCreateWithFlags(&g_secondary_streams_bwd[device], cudaStreamNonBlocking);
// // //       }
// // //    }
// // //    return g_secondary_streams_bwd[device];
// // // }

// // // static void init_bwd_events(int device) {
// // //    if (g_bwd_event_ready[device] == nullptr) {
// // //       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
// // //       if (g_bwd_event_ready[device] == nullptr) {
// // //          cudaSetDevice(device);
// // //          // DisableTiming avoids hardware timestamp recording, making record/wait ~10× cheaper
// // //          cudaEventCreateWithFlags(&g_bwd_event_ready[device], cudaEventDisableTiming);
// // //          cudaEventCreateWithFlags(&g_bwd_event_done[device],  cudaEventDisableTiming);
// // //       }
// // //    }
// // // }

// // // using namespace nvcuda;

// // // // ============================================================================
// // // // METADATA & CONSTANTS
// // // // ============================================================================

// // // struct BackwardMetadata {
// // //    int a_shape[8], b_shape[8], grad_out_shape[8];
// // //    int a_strides[8], b_strides[8], grad_out_strides[8];
// // //    int grad_a_strides[8], grad_b_strides[8];
// // //    int a_ndim, b_ndim, grad_out_ndim;
// // // };

// // // __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
// // //                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
// // //    offset = 0; 
// // //    if (out_ndim <= 2) return;
// // //    int temp_batch = batch_idx;
// // //    for (int dim = out_ndim - 3; dim >= 0; --dim) {
// // //       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
// // //       temp_batch /= b_dim_sz;
// // //       int c_dim = dim - (out_ndim - ndim);
// // //       if (c_dim >= 0 && c_dim < ndim - 2) 
// // //          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
// // //    }
// // // }

// // // constexpr int BACKWARD_PAD = 8;

// // // // ============================================================================
// // // // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int TM, int TN>
// // // __global__ void matmul_backward_dA_fp32(
// // //     const float* __restrict__ grad_out,  // [M, N]
// // //     const float* __restrict__ B,         // [K, N]
// // //     float* __restrict__ grad_A,          // [M, K]
// // //     float beta,
// // //     int M, int N, int K,
// // //     int total_batches,
// // //     BackwardMetadata meta) 
// // // {
// // //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// // //    if (b_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// // //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// // //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// // //    const float *GO = grad_out + go_offset;
// // //    const float *Bp = B + b_offset;
// // //    float *GA = grad_A + ga_offset;
   
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// // //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// // //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// // //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// // //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// // //    __shared__ float As[2][BK][BM + BACKWARD_PAD];
// // //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
// // //    float results[16] = {0.0f}, regM[4], regN[4];

// // //    auto load_tiles = [&](int no, int idx) {
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// // //          int gm = by * BM + r, gn = no + c;
// // //          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// // //       }
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// // //          int gn = no + r, gk = bx * BN + c;
// // //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
// // //       }
// // //    };

// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int bk = 0; bk < N; bk += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (bk + BK < N) load_tiles(bk + BK, wi);
      
// // //       #pragma unroll
// // //       for (int d = 0; d < BK; d++) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// // //          #pragma unroll
// // //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) 
// // //             for (int j = 0; j < 4; j++) 
// // //                results[i*4+j] += regM[i] * regN[j];
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    for (int i = 0; i < 4; i++) {
// // //       for (int j = 0; j < 4; j++) {
// // //          int r = by * BM + tRow * 4 + i;
// // //          int c = bx * BN + tCol * 4 + j;
// // //          if (r < M && c < K) {
// // //             int idx = r * s_ga_m + c * s_ga_k;
// // //             GA[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GA[idx]);
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_B = A^T @ grad_output
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int TM, int TN>
// // // __global__ void matmul_backward_dB_fp32(
// // //     const float* __restrict__ A,         // [M, K]
// // //     const float* __restrict__ grad_out,  // [M, N]
// // //     float* __restrict__ grad_B,          // [K, N]
// // //     float beta,
// // //     int M, int K, int N,
// // //     int total_batches,
// // //     BackwardMetadata meta)
// // // {
// // //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// // //    if (b_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// // //    int a_offset = 0, go_offset = 0, gb_offset = 0;
// // //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
// // //    const float *Ap = A + a_offset;
// // //    const float *GO = grad_out + go_offset;
// // //    float *GB = grad_B + gb_offset;
   
// // //    int s_a_m = meta.a_strides[meta.a_ndim-2];
// // //    int s_a_k = meta.a_strides[meta.a_ndim-1];
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// // //    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
// // //    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

// // //    __shared__ float As[2][BK][BM + BACKWARD_PAD];
// // //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
// // //    float results[16] = {0.0f}, regM[4], regN[4];

// // //    auto load_tiles = [&](int mo, int idx) {
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// // //          int gk = by * BM + r, gm = mo + c;
// // //          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
// // //       }
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// // //          int gm = mo + r, gn = bx * BN + c;
// // //          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// // //       }
// // //    };

// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int bk = 0; bk < M; bk += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (bk + BK < M) load_tiles(bk + BK, wi);
      
// // //       #pragma unroll
// // //       for (int d = 0; d < BK; d++) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// // //          #pragma unroll
// // //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) 
// // //             for (int j = 0; j < 4; j++) 
// // //                results[i*4+j] += regM[i] * regN[j];
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    for (int i = 0; i < 4; i++) {
// // //       for (int j = 0; j < 4; j++) {
// // //          int r = by * BM + tRow * 4 + i;
// // //          int c = bx * BN + tCol * 4 + j;
// // //          if (r < K && c < N) {
// // //             int idx = r * s_gb_k + c * s_gb_n;
// // //             GB[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GB[idx]);
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // FP16 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int WM, int WN>
// // // __global__ void matmul_backward_dA_fp16(
// // //     const __half* __restrict__ grad_out,
// // //     const __half* __restrict__ B,
// // //     __half* __restrict__ grad_A,
// // //     float beta,
// // //     int M, int N, int K,
// // //     int total_batches,
// // //     BackwardMetadata meta)
// // // {
// // //    const int batch_idx = blockIdx.z; 
// // //    if (batch_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
// // //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// // //    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// // //    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// // //    const __half *GO = grad_out + go_offset;
// // //    const __half *Bp = B + b_offset;
// // //    __half *GA = grad_A + ga_offset;
   
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// // //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// // //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// // //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// // //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// // //    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
// // //    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
// // //    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
// // //    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
// // //    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
// // //    #pragma unroll
// // //    for(int i = 0; i < 2; i++) 
// // //       for(int j = 0; j < 2; j++) 
// // //          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

// // //    auto load_tiles = [&](int no, int idx) {
// // //       for (int i = tid; i < BM * BK; i += 512) {
// // //          int r = i / BK, c = i % BK;
// // //          int gm = blockIdx.y * BM + r, gn = no + c;
// // //          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
// // //       }
// // //       for (int i = tid; i < BK * BN; i += 512) {
// // //          int r = i / BN, c = i % BN;
// // //          int gn = no + r, gk = blockIdx.x * BN + c;
// // //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
// // //       }
// // //    };
   
// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int k = 0; k < N; k += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (k + BK < N) load_tiles(k + BK, wi);
      
// // //       #pragma unroll
// // //       for (int ks = 0; ks < BK; ks += 16) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 2; i++) {
// // //             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
// // //             #pragma unroll
// // //             for (int j = 0; j < 2; j++) {
// // //                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
// // //                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
// // //             }
// // //          }
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    __half* sm = reinterpret_cast<__half*>(As);
// // //    #pragma unroll
// // //    for (int i = 0; i < 2; i++) {
// // //       for (int j = 0; j < 2; j++) {
// // //          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
// // //          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
// // //          if (cr < M && cc < K) {
// // //             __half* wsm = sm + warp_id * 256;
// // //             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
// // //             for (int r = 0; r < 16; r++) 
// // //                for (int c = 0; c < 16; c++)
// // //                   if (cr + r < M && cc + c < K) {
// // //                      int idx = (cr + r) * s_ga_m + (cc + c) * s_ga_k;
// // //                      GA[idx] = wsm[r * 16 + c] + (__half)beta * GA[idx];
// // //                   }
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // DISPATCH LAYER
// // // // ============================================================================

// // // template<typename T>
// // // void launch_backward_matmul(
// // //     const Tensor& grad_output,
// // //     const Tensor& A,
// // //     const Tensor& B,
// // //     Tensor& grad_A,
// // //     Tensor& grad_B,
// // //     float beta_a,
// // //     float beta_b,
// // //     cudaStream_t stream)
// // // {
// // //    const auto& go_sh = grad_output.shape().dims;
// // //    const auto& a_sh = A.shape().dims;
// // //    const auto& b_sh = B.shape().dims;
   
// // //    int go_ndim = go_sh.size();
// // //    int a_ndim = a_sh.size();
// // //    int b_ndim = b_sh.size();
   
// // //    int M = go_sh[go_ndim - 2];
// // //    int N = go_sh[go_ndim - 1];
// // //    int K_a = a_sh[a_ndim - 1];
   
// // //    int tb = 1;
// // //    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
// // //    BackwardMetadata meta;
// // //    meta.a_ndim = a_ndim;
// // //    meta.b_ndim = b_ndim;
// // //    meta.grad_out_ndim = go_ndim;
   
// // //    for (int i = 0; i < a_ndim; i++) {
// // //       meta.a_shape[i] = a_sh[i];
// // //       meta.a_strides[i] = A.stride().strides[i];
// // //       meta.grad_a_strides[i] = grad_A.stride().strides[i];
// // //    }
// // //    for (int i = 0; i < b_ndim; i++) {
// // //       meta.b_shape[i] = b_sh[i];
// // //       meta.b_strides[i] = B.stride().strides[i];
// // //       meta.grad_b_strides[i] = grad_B.stride().strides[i];
// // //    }
// // //    for (int i = 0; i < go_ndim; i++) {
// // //       meta.grad_out_shape[i] = go_sh[i];
// // //       meta.grad_out_strides[i] = grad_output.stride().strides[i];
// // //    }
   
// // //    const T* go_ptr = grad_output.data<T>();
// // //    const T* a_ptr = A.data<T>();
// // //    const T* b_ptr = B.data<T>();
// // //    T* ga_ptr = grad_A.data<T>();
// // //    T* gb_ptr = grad_B.data<T>();
   
// // //    bool supported_for_fast_path = (meta.grad_out_strides[go_ndim-1] == 1) && 
// // //                                   (meta.a_strides[a_ndim-1] == 1) && 
// // //                                   (meta.b_strides[b_ndim-1] == 1) && 
// // //                                   (meta.grad_a_strides[a_ndim-1] == 1) && 
// // //                                   (meta.grad_b_strides[b_ndim-1] == 1);

// // //     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");

// // //    // ---- Acquire device, handles, and secondary stream once ----
// // //    // grad_A = grad_output @ B^T  and  grad_B = A^T @ grad_output are fully independent.
// // //    // Dispatch them on separate cuBLAS handles bound to separate streams so the GPU
// // //    // can overlap them.  Events are cached per-device and use DisableTiming for
// // //    // minimal overhead.
// // //    int dev = 0;
// // //    cudaGetDevice(&dev);
// // //    mycublasHandle_t handle_a = get_mycublas_handle_bwd(dev);
// // //    mycublasHandle_t handle_b = get_mycublas_handle_bwd2(dev);
// // //    cudaStream_t stream_b     = get_secondary_stream_bwd(dev);
// // //    init_bwd_events(dev);

// // //    mycublasSetStream(handle_a, stream);
// // //    mycublasSetStream(handle_b, stream_b);

// // //    // stream_b must wait until all upstream data (grad_output, A, B) produced on
// // //    // the primary stream is visible before it starts reading them.
// // //    cudaEventRecord(g_bwd_event_ready[dev], stream);
// // //    cudaStreamWaitEvent(stream_b, g_bwd_event_ready[dev], 0);

// // //    // ---- grad_A = grad_output @ B^T  (primary stream) ----
// // //    bool grad_a_done = false;
// // //    if constexpr (std::is_same<T, float>::value) {
// // //        if (supported_for_fast_path) {
// // //           int ldb  = meta.b_strides[b_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldga = meta.grad_a_strides[a_ndim-2];
// // //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// // //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// // //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
          
// // //           mycublasSgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, 1.0f, go_ptr, ldgo, s_go, b_ptr, ldb, s_b, beta_a, ga_ptr, ldga, s_ga, tb);
// // //           grad_a_done = true;
// // //        }
// // //    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
// // //        if (supported_for_fast_path) {
// // //           int ldb  = meta.b_strides[b_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldga = meta.grad_a_strides[a_ndim-2];
// // //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// // //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// // //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
// // //           const __half alpha = __float2half(1.0f), h_beta_a = __float2half(beta_a);
// // //           mycublasHgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, alpha, (const __half*)go_ptr, ldgo, s_go, (const __half*)b_ptr, ldb, s_b, h_beta_a, (__half*)ga_ptr, ldga, s_ga, tb);
// // //           grad_a_done = true;
// // //        }
// // //    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
// // //        if (supported_for_fast_path) {
// // //           int ldb  = meta.b_strides[b_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldga = meta.grad_a_strides[a_ndim-2];
// // //           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// // //           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
// // //           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
// // //           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), b_beta_a = __float2bfloat16(beta_a);
// // //           mycublasBgemmStridedBatched(handle_a, MYCUBLAS_OP_N, MYCUBLAS_OP_T, M, K_a, N, alpha, (const __nv_bfloat16*)go_ptr, ldgo, s_go, (const __nv_bfloat16*)b_ptr, ldb, s_b, b_beta_a, (__nv_bfloat16*)ga_ptr, ldga, s_ga, tb);
// // //           grad_a_done = true;
// // //        }
// // //    }

// // //    if (!grad_a_done) {
// // //        if constexpr (std::is_same<T, float>::value)
// // //           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, beta_a, M, N, K_a, tb, meta);
// // //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
// // //           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), beta_a, M, N, K_a, tb, meta);
// // //    }

// // //    // ---- grad_B = A^T @ grad_output  (secondary stream — runs in parallel with grad_A above) ----
// // //    bool grad_b_done = false;
// // //    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[b_ndim-3] == 1 && meta.grad_out_shape[go_ndim-3] > 1);

// // //    if constexpr (std::is_same<T, float>::value) {
// // //        if (supported_for_fast_path) {
// // //           int lda  = meta.a_strides[a_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldgb = meta.grad_b_strides[b_ndim-2];
// // //           if (need_reduction) {
// // //               int M_total = M * tb;
// // //               // TN layout but M_total as K
// // //               mycublasSgemm(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, 1.0f, a_ptr, lda, go_ptr, ldgo, beta_b, gb_ptr, ldgb);
// // //           } else {
// // //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]            : 0;
// // //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3]    : 0;
// // //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]       : 0;
             
// // //              mycublasSgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, 1.0f, a_ptr, lda, s_a, go_ptr, ldgo, s_go, beta_b, gb_ptr, ldgb, s_gb, tb);
// // //           }
// // //           grad_b_done = true;
// // //        }
// // //    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
// // //        if (supported_for_fast_path) {
// // //           int lda  = meta.a_strides[a_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldgb = meta.grad_b_strides[b_ndim-2];
// // //           const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
// // //           if (need_reduction) {
// // //               int M_total = M * tb;
// // //               const __half h_beta_b = __float2half(beta_b);
// // //               mycublasHgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __half*)a_ptr, lda, 0, (const __half*)go_ptr, ldgo, 0, h_beta_b, (__half*)gb_ptr, ldgb, 0, 1);
// // //           } else {
// // //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
// // //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// // //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
// // //              mycublasHgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __half*)a_ptr, lda, s_a, (const __half*)go_ptr, ldgo, s_go, beta, (__half*)gb_ptr, ldgb, s_gb, tb);
// // //           }
// // //           grad_b_done = true;
// // //        }
// // //    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
// // //        if (supported_for_fast_path) {
// // //           int lda  = meta.a_strides[a_ndim-2];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldgb = meta.grad_b_strides[b_ndim-2];
// // //           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(0.0f);
// // //           if (need_reduction) {
// // //               int M_total = M * tb;
// // //               const __nv_bfloat16 b_beta_b = __float2bfloat16(beta_b);
// // //               mycublasBgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __nv_bfloat16*)a_ptr, lda, 0, (const __nv_bfloat16*)go_ptr, ldgo, 0, b_beta_b, (__nv_bfloat16*)gb_ptr, ldgb, 0, 1);
// // //           } else {
// // //              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
// // //              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
// // //              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
// // //              mycublasBgemmStridedBatched(handle_b, MYCUBLAS_OP_T, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __nv_bfloat16*)a_ptr, lda, s_a, (const __nv_bfloat16*)go_ptr, ldgo, s_go, beta, (__nv_bfloat16*)gb_ptr, ldgb, s_gb, tb);
// // //           }
// // //           grad_b_done = true;
// // //        }
// // //    }

// // //    if (!grad_b_done) {
// // //        if constexpr (std::is_same<T, float>::value)
// // //           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream_b>>>(a_ptr, go_ptr, gb_ptr, beta_b, M, K_a, N, tb, meta);
// // //    }

// // //    // Rejoin stream_b into the primary stream so downstream ops on stream see grad_B.
// // //    cudaEventRecord(g_bwd_event_done[dev], stream_b);
// // //    cudaStreamWaitEvent(stream, g_bwd_event_done[dev], 0);

// // //    cudaError_t err = cudaGetLastError();
// // //    if (err != cudaSuccess) {
// // //       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
// // //    }
// // // }

// // // void cuda_matmul_backward(
// // //     const Tensor& grad_output,
// // //     const Tensor& A,
// // //     const Tensor& B,
// // //     Tensor& grad_A,
// // //     Tensor& grad_B,
// // //     cudaStream_t stream)
// // // {
// // //    float beta_a = 0.0f;
// // //    float beta_b = 0.0f;
   
// // //    dispatch_by_dtype(A.dtype(), [&](auto d) {
// // //       using T = decltype(d);
// // //       launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, beta_a, beta_b, stream);
// // //    });
// // // }

// // // } // namespace matmul_backward_myblas
// // // } // namespace OwnTensor

// // // #endif // WITH_MYBLAS

// // // // cuBLAS backend — always compiled.
// // // #include <cuda_runtime.h>
// // // #include <cuda_fp16.h>
// // // #include <cuda_bf16.h>
// // // #include <mma.h>
// // // #include <cublas_v2.h>
// // // #include <algorithm>
// // // #include <stdexcept>
// // // #include <vector>
// // // #include <string>
// // // #include <mutex>
// // // #include <cstdlib>

// // // #include "ops/MatmulBackward.cuh"
// // // #include "core/Tensor.h"
// // // #include "core/TensorDispatch.h"
// // // #include "utils/Profiler.h"

// // // namespace OwnTensor {
// // // namespace matmul_backward_cublas {

// // // // cuBLAS handle management - thread-safe singleton per device
// // // static std::mutex cublas_mutex_bwd;
// // // static cublasHandle_t g_cublas_handles_bwd[8] = {nullptr};

// // // static cublasHandle_t get_cublas_handle_bwd(int device = 0) {
// // //    if (g_cublas_handles_bwd[device] == nullptr) {
// // //       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
// // //       if (g_cublas_handles_bwd[device] == nullptr) {
// // //          cudaSetDevice(device);
// // //          cublasCreate(&g_cublas_handles_bwd[device]);
// // //          // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
// // //          cublasSetMathMode(g_cublas_handles_bwd[device], CUBLAS_TF32_TENSOR_OP_MATH);
// // //       }
// // //    }
// // //    return g_cublas_handles_bwd[device];
// // // }

// // // using namespace nvcuda;

// // // // ============================================================================
// // // // METADATA & CONSTANTS (Reuse from forward)
// // // // ============================================================================

// // // struct BackwardMetadata {
// // //    int a_shape[8], b_shape[8], grad_out_shape[8];
// // //    int a_strides[8], b_strides[8], grad_out_strides[8];
// // //    int grad_a_strides[8], grad_b_strides[8];
// // //    int a_ndim, b_ndim, grad_out_ndim;
// // // };

// // // __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
// // //                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
// // //    offset = 0; 
// // //    if (out_ndim <= 2) return;
// // //    int temp_batch = batch_idx;
// // //    for (int dim = out_ndim - 3; dim >= 0; --dim) {
// // //       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
// // //       temp_batch /= b_dim_sz;
// // //       int c_dim = dim - (out_ndim - ndim);
// // //       if (c_dim >= 0 && c_dim < ndim - 2) 
// // //          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
// // //    }
// // // }

// // // constexpr int BACKWARD_PAD = 8;

// // // // ============================================================================
// // // // FP32 BACKWARD KERNEL: grad_A = grad_output @ B^T
// // // // Computes: C[M,K] = A[M,N] @ B[K,N]^T = A[M,N] @ B^T[N,K]
// // // // Note: We read B in transposed order (columns become rows)
// // // // BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int TM, int TN>
// // // __global__ void matmul_backward_dA_fp32(
// // //     const float* __restrict__ grad_out,  // [M, N] - grad of output
// // //     const float* __restrict__ B,         // [K, N] - second input of forward, read transposed
// // //     float* __restrict__ grad_A,          // [M, K] - gradient for A
// // //     int M, int N, int K,                 // M=rows of grad_out, N=cols of grad_out (=cols of B), K=rows of B
// // //     int total_batches,
// // //     BackwardMetadata meta) 
// // // {
// // //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// // //    if (b_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// // //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// // //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// // //    const float *GO = grad_out + go_offset;
// // //    const float *Bp = B + b_offset;
// // //    float *GA = grad_A + ga_offset;
   
// // //    // Strides for grad_output [M, N]
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
// // //    // Strides for B [K, N] - we read it transposed as B^T [N, K]
// // //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// // //    int s_b_n = meta.b_strides[meta.b_ndim-1];
   
// // //    // Strides for grad_A [M, K]
// // //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// // //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// // //    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // grad_out tile: [BK, BM] transposed storage
// // //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // B^T tile: [BK, BN]
// // //    float results[16] = {0.0f}, regM[4], regN[4];

// // //    // Load tiles with double buffering
// // //    // For grad_A = grad_out @ B^T:
// // //    //   We iterate over N (the shared dimension between grad_out and B^T)
// // //    //   grad_out is [M, N], B^T is [N, K]
// // //    //   So we load BK columns of grad_out into shared As
// // //    //   And BK rows of B^T (= BK columns of B) into shared Bs
   
// // //    auto load_tiles = [&](int no, int idx) {
// // //       // Load grad_out[by*BM : by*BM+BM, no : no+BK] into As (transposed for coalescing)
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// // //          int gm = by * BM + r, gn = no + c;
// // //          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// // //       }
      
// // //       // Load B^T[no : no+BK, bx*BN : bx*BN+BN] = B[bx*BN : bx*BN+BN, no : no+BK]^T
// // //       // B is [K, N], B^T is [N, K]. We want B^T[n, k] = B[k, n]
// // //       // So for k in [bx*BN, bx*BN+BN) and n in [no, no+BK)
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// // //          int gn = no + r, gk = bx * BN + c;
// // //          // B^T[gn, gk] = B[gk, gn]
// // //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
// // //       }
// // //    };

// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int bk = 0; bk < N; bk += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (bk + BK < N) load_tiles(bk + BK, wi);
      
// // //       #pragma unroll
// // //       for (int d = 0; d < BK; d++) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// // //          #pragma unroll
// // //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) 
// // //             for (int j = 0; j < 4; j++) 
// // //                results[i*4+j] += regM[i] * regN[j];
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    // Store results to grad_A
// // //    for (int i = 0; i < 4; i++) {
// // //       for (int j = 0; j < 4; j++) {
// // //          int r = by * BM + tRow * 4 + i;
// // //          int c = bx * BN + tCol * 4 + j;
// // //          if (r < M && c < K) {
// // //             GA[r * s_ga_m + c * s_ga_k] = results[i*4+j];
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // FP32 BACKWARD KERNEL: grad_B = A^T @ grad_output
// // // // Computes: C[K,N] = A[M,K]^T @ B[M,N] = A^T[K,M] @ B[M,N]
// // // // Note: We read A in transposed order
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int TM, int TN>
// // // __global__ void matmul_backward_dB_fp32(
// // //     const float* __restrict__ A,         // [M, K] - first input of forward, read transposed
// // //     const float* __restrict__ grad_out,  // [M, N] - grad of output
// // //     float* __restrict__ grad_B,          // [K, N] - gradient for B
// // //     int M, int K, int N,                 // M=rows of A, K=cols of A, N=cols of grad_out
// // //     int total_batches,
// // //     BackwardMetadata meta)
// // // {
// // //    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
// // //    if (b_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
// // //    int a_offset = 0, go_offset = 0, gb_offset = 0;
// // //    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
// // //                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
// // //    const float *Ap = A + a_offset;
// // //    const float *GO = grad_out + go_offset;
// // //    float *GB = grad_B + gb_offset;
   
// // //    // Strides for A [M, K] - we read it transposed as A^T [K, M]
// // //    int s_a_m = meta.a_strides[meta.a_ndim-2];
// // //    int s_a_k = meta.a_strides[meta.a_ndim-1];
   
// // //    // Strides for grad_output [M, N]
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
// // //    // Strides for grad_B [K, N]
// // //    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
// // //    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

// // //    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // A^T tile: [BK, BM]
// // //    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // grad_out tile: [BK, BN]
// // //    float results[16] = {0.0f}, regM[4], regN[4];

// // //    // For grad_B = A^T @ grad_out:
// // //    //   We iterate over M (the shared dimension)
// // //    //   A^T is [K, M], grad_out is [M, N]
// // //    //   So we load BK rows of A^T (= BK columns of A) into shared As
// // //    //   And BK rows of grad_out into shared Bs
   
// // //    auto load_tiles = [&](int mo, int idx) {
// // //       // Load A^T[by*BM : by*BM+BM, mo : mo+BK] = A[mo : mo+BK, by*BM : by*BM+BM]^T
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BK, c = li % BK;
// // //          int gk = by * BM + r, gm = mo + c;
// // //          // A^T[gk, gm] = A[gm, gk]
// // //          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
// // //       }
      
// // //       // Load grad_out[mo : mo+BK, bx*BN : bx*BN+BN] into Bs
// // //       #pragma unroll
// // //       for (int i = 0; i < 2; i++) {
// // //          int li = tid + i * 1024, r = li / BN, c = li % BN;
// // //          int gm = mo + r, gn = bx * BN + c;
// // //          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
// // //       }
// // //    };

// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int bk = 0; bk < M; bk += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (bk + BK < M) load_tiles(bk + BK, wi);
      
// // //       #pragma unroll
// // //       for (int d = 0; d < BK; d++) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
// // //          #pragma unroll
// // //          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
// // //          #pragma unroll
// // //          for (int i = 0; i < 4; i++) 
// // //             for (int j = 0; j < 4; j++) 
// // //                results[i*4+j] += regM[i] * regN[j];
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    // Store results to grad_B
// // //    for (int i = 0; i < 4; i++) {
// // //       for (int j = 0; j < 4; j++) {
// // //          int r = by * BM + tRow * 4 + i;
// // //          int c = bx * BN + tCol * 4 + j;
// // //          if (r < K && c < N) {
// // //             GB[r * s_gb_k + c * s_gb_n] = results[i*4+j];
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // FP16 BACKWARD KERNEL: grad_A = grad_output @ B^T (WMMA)
// // // // ============================================================================

// // // template<int BM, int BN, int BK, int WM, int WN>
// // // __global__ void matmul_backward_dA_fp16(
// // //     const __half* __restrict__ grad_out,
// // //     const __half* __restrict__ B,
// // //     __half* __restrict__ grad_A,
// // //     int M, int N, int K,
// // //     int total_batches,
// // //     BackwardMetadata meta)
// // // {
// // //    const int batch_idx = blockIdx.z; 
// // //    if (batch_idx >= total_batches) return;
// // //    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
// // //    int go_offset = 0, b_offset = 0, ga_offset = 0;
// // //    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
// // //    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
// // //    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
// // //                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
// // //    const __half *GO = grad_out + go_offset;
// // //    const __half *Bp = B + b_offset;
// // //    __half *GA = grad_A + ga_offset;
   
// // //    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
// // //    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
// // //    int s_b_k = meta.b_strides[meta.b_ndim-2];
// // //    int s_b_n = meta.b_strides[meta.b_ndim-1];
// // //    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
// // //    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

// // //    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
// // //    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
// // //    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
// // //    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
// // //    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
// // //    #pragma unroll
// // //    for(int i = 0; i < 2; i++) 
// // //       for(int j = 0; j < 2; j++) 
// // //          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

// // //    auto load_tiles = [&](int no, int idx) {
// // //       for (int i = tid; i < BM * BK; i += 512) {
// // //          int r = i / BK, c = i % BK;
// // //          int gm = blockIdx.y * BM + r, gn = no + c;
// // //          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
// // //       }
// // //       for (int i = tid; i < BK * BN; i += 512) {
// // //          int r = i / BN, c = i % BN;
// // //          int gn = no + r, gk = blockIdx.x * BN + c;
// // //          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
// // //       }
// // //    };
   
// // //    int wi = 0; 
// // //    load_tiles(0, wi); 
// // //    __syncthreads();
   
// // //    for (int k = 0; k < N; k += BK) {
// // //       int ri = wi; 
// // //       wi = 1 - wi;
// // //       if (k + BK < N) load_tiles(k + BK, wi);
      
// // //       #pragma unroll
// // //       for (int ks = 0; ks < BK; ks += 16) {
// // //          #pragma unroll
// // //          for (int i = 0; i < 2; i++) {
// // //             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
// // //             #pragma unroll
// // //             for (int j = 0; j < 2; j++) {
// // //                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
// // //                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
// // //             }
// // //          }
// // //       }
// // //       __syncthreads();
// // //    }
   
// // //    __half* sm = reinterpret_cast<__half*>(As);
// // //    #pragma unroll
// // //    for (int i = 0; i < 2; i++) {
// // //       for (int j = 0; j < 2; j++) {
// // //          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
// // //          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
// // //          if (cr < M && cc < K) {
// // //             __half* wsm = sm + warp_id * 256;
// // //             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
// // //             for (int r = 0; r < 16; r++) 
// // //                for (int c = 0; c < 16; c++)
// // //                   if (cr + r < M && cc + c < K) 
// // //                      GA[(cr + r) * s_ga_m + (cc + c) * s_ga_k] = wsm[r * 16 + c];
// // //          }
// // //       }
// // //    }
// // // }

// // // // ============================================================================
// // // // DISPATCH LAYER
// // // // ============================================================================

// // // template<typename T>
// // // void launch_backward_matmul(
// // //     const Tensor& grad_output,
// // //     const Tensor& A,
// // //     const Tensor& B,
// // //     Tensor& grad_A,
// // //     Tensor& grad_B,
// // //     cudaStream_t stream)
// // // {
// // //    const auto& go_sh = grad_output.shape().dims;
// // //    const auto& a_sh = A.shape().dims;
// // //    const auto& b_sh = B.shape().dims;
   
// // //    int go_ndim = go_sh.size();
// // //    int a_ndim = a_sh.size();
// // //    int b_ndim = b_sh.size();
   
// // //    int M = go_sh[go_ndim - 2];  // rows of grad_output
// // //    int N = go_sh[go_ndim - 1];  // cols of grad_output (= cols of B)
// // //    int K_a = a_sh[a_ndim - 1];  // cols of A (= rows of grad_A's cols dim)
// // //    int K_b = b_sh[b_ndim - 2];  // rows of B
   
// // //    int tb = 1;
// // //    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
// // //    BackwardMetadata meta;
// // //    meta.a_ndim = a_ndim;
// // //    meta.b_ndim = b_ndim;
// // //    meta.grad_out_ndim = go_ndim;
   
// // //    for (int i = 0; i < a_ndim; i++) {
// // //       meta.a_shape[i] = a_sh[i];
// // //       meta.a_strides[i] = A.stride().strides[i];
// // //       meta.grad_a_strides[i] = grad_A.stride().strides[i];
// // //    }
// // //    for (int i = 0; i < b_ndim; i++) {
// // //       meta.b_shape[i] = b_sh[i];
// // //       meta.b_strides[i] = B.stride().strides[i];
// // //       meta.grad_b_strides[i] = grad_B.stride().strides[i];
// // //    }
// // //    for (int i = 0; i < go_ndim; i++) {
// // //       meta.grad_out_shape[i] = go_sh[i];
// // //       meta.grad_out_strides[i] = grad_output.stride().strides[i];
// // //    }
   
// // //    const T* go_ptr = grad_output.data<T>();
// // //    const T* a_ptr = A.data<T>();
// // //    const T* b_ptr = B.data<T>();
// // //    T* ga_ptr = grad_A.data<T>();
// // //    T* gb_ptr = grad_B.data<T>();
   
// // //    // Check stride attributes
// // //    bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
// // //    bool a_contiguous = (meta.a_strides[a_ndim-1] == 1);
// // //    bool a_transposed = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
// // //    bool b_contiguous = (meta.b_strides[b_ndim-1] == 1);
// // //    bool b_transposed = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
// // //    bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
// // //    bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

// // //    bool supported_for_fast_path = go_contiguous && (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && ga_contiguous && gb_contiguous;

// // //     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");
// // //     bool grad_a_done = false;
// // //     if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// // //                   std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// // //         // cuBLAS path for FP16/BF16: grad_A = grad_output @ B^T
// // //         // compute in fp16/bf16, accumulate in fp32
// // //         if (supported_for_fast_path) {
// // //             int dev_idx = grad_output.device().index;
// // //             cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// // //             cublasSetStream(handle, stream);
// // //             float alpha_f = 1.0f, beta_f = 0.0f;

// // //             cublasOperation_t opB_bwd = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// // //             int ldb_bwd = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
// // //             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
// // //             int ldga_bwd = meta.grad_a_strides[a_ndim-2];

// // //             long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3] : 0LL;
// // //             long long stride_b_ll  = (b_ndim  >= 3) ? (long long)meta.b_strides[b_ndim-3]         : 0LL;
// // //             long long stride_ga_ll = (a_ndim  >= 3) ? (long long)meta.grad_a_strides[a_ndim-3]    : 0LL;
// // //             if (b_ndim < go_ndim) stride_b_ll = 0LL;

// // //             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
// // //                                        ? CUDA_R_16F : CUDA_R_16BF;
// // //             cublasStatus_t status;
// // //             if (tb == 1) {
// // //                 status = cublasGemmEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
// // //                     b_ptr,  cuda_type, ldb_bwd,
// // //                     go_ptr, cuda_type, ldgo_bwd,
// // //                     &beta_f, ga_ptr, cuda_type, ldga_bwd,
// // //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //             } else {
// // //                 status = cublasGemmStridedBatchedEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
// // //                     b_ptr,  cuda_type, ldb_bwd,  stride_b_ll,
// // //                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
// // //                     &beta_f, ga_ptr, cuda_type, ldga_bwd, stride_ga_ll, tb,
// // //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //             }
// // //             if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
// // //         }
// // //     } else if constexpr (std::is_same<T, float>::value) {
// // //         if (supported_for_fast_path) {
// // //           // USE EXPLICIT DEVICE INDEXING from grad_output
// // //           int dev_idx = grad_output.device().index;
// // //           cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// // //           cublasSetStream(handle, stream);
// // //           float alpha = 1.0f, beta = 0.0f;
          
// // //           cublasOperation_t opB = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// // //           int ldb = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldga = meta.grad_a_strides[a_ndim-2];
          
// // //           // Compute batch strides from actual tensor metadata
// // //           long long stride_go, stride_b_val, stride_ga;
// // //           bool batch_uniform_dA = true;

// // //           if (go_ndim <= 3) {
// // //               stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
// // //               stride_b_val = (b_ndim == 3) ? meta.b_strides[0] : 0;
// // //               stride_ga = (go_ndim == 3) ? meta.grad_a_strides[0] : 0;
// // //               if (b_ndim < go_ndim) stride_b_val = 0;
// // //           } else {
// // //               // 4D: use inner batch stride (strides[ndim-3])
// // //               stride_go = meta.grad_out_strides[go_ndim - 3];
// // //               stride_b_val = (b_ndim >= 4) ? meta.b_strides[b_ndim - 3] : 0;
// // //               stride_ga = meta.grad_a_strides[a_ndim - 3];
// // //               if (b_ndim < go_ndim) stride_b_val = 0;

// // //               // Check uniform batch strides: stride[0] == stride[1] * shape[1]
// // //               batch_uniform_dA = (meta.grad_out_strides[0] == stride_go * go_sh[1]);
// // //               if (b_ndim >= 4) batch_uniform_dA = batch_uniform_dA && (meta.b_strides[0] == stride_b_val * b_sh[1]);
// // //               batch_uniform_dA = batch_uniform_dA && (meta.grad_a_strides[0] == stride_ga * a_sh[1]);
// // //           }

// // //           cublasStatus_t status;
// // //           if (tb == 1) {
// // //               status = cublasGemmEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, go_ptr, CUDA_R_32F, ldgo, &beta, ga_ptr, CUDA_R_32F, ldga, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //           } else if (batch_uniform_dA) {
// // //               status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, stride_b_val, go_ptr, CUDA_R_32F, ldgo, stride_go, &beta, ga_ptr, CUDA_R_32F, ldga, stride_ga, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //           } else {
// // //               // Non-uniform 4D: loop over outer batch dim
// // //               int d0 = go_sh[0], d1 = go_sh[1];
// // //               status = CUBLAS_STATUS_SUCCESS;
// // //               for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
// // //                   const T* b_batch = b_ptr + (b_ndim >= 4 ? bi * meta.b_strides[0] : 0);
// // //                   const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
// // //                   T* ga_batch = ga_ptr + bi * meta.grad_a_strides[0];
// // //                   status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_batch, CUDA_R_32F, ldb, stride_b_val, go_batch, CUDA_R_32F, ldgo, stride_go, &beta, ga_batch, CUDA_R_32F, ldga, stride_ga, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //               }
// // //           }

// // //           if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
// // //        }
// // //    }
   
// // //    if (!grad_a_done) {
// // //         #ifdef AUTOGRAD_PROFILER_ENABLED
// // //         if (M*N*K_a > 1000000) {
// // //             printf("dA Fallback: M=%d N=%d K=%d, Strides: go[%d,%d], b[%d,%d], ga[%d,%d]\n", 
// // //                    M, N, K_a, meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
// // //                    meta.b_strides[b_ndim-2], meta.b_strides[b_ndim-1],
// // //                    meta.grad_a_strides[a_ndim-2], meta.grad_a_strides[a_ndim-1]);
// // //         }
// // //         #endif
// // //        // Fallback
// // //        if constexpr (std::is_same<T, float>::value)
// // //           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, M, N, K_a, tb, meta);
// // //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
// // //           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), M, N, K_a, tb, meta);
// // //        // bf16 fallback: cuBLAS path above handles all standard layouts; non-standard bf16 not supported
// // //    }

// // //    // Launch grad_B kernel: grad_B = A^T @ grad_output
// // //    // Handle Reduction if B dim < Output dim (Broadcast in forward)
   
// // //    bool grad_b_actions_needed = true; // Could optimize if not req grad
// // //    bool grad_b_done = false;
   
// // //    // Check reduction need
// // //    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[0] == 1);
// // //    // Standard broadcast check: if tb > 1 but grad_B is not batched.
// // //    // Assuming grad_B matches B shape.
// // //    if (tb > 1 && b_ndim < go_ndim) need_reduction = true;

// // //    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// // //                  std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// // //         // cuBLAS path for FP16/BF16: grad_B = A^T @ grad_output
// // //         // compute in fp16/bf16, accumulate in fp32
// // //         if (supported_for_fast_path) {
// // //             int dev_idx = grad_output.device().index;
// // //             cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// // //             cublasSetStream(handle, stream);
// // //             float alpha_f = 1.0f, beta_f = 0.0f;

// // //             cublasOperation_t opA_bwd = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
// // //             int lda_bwd  = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];
// // //             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
// // //             int ldgb_bwd = meta.grad_b_strides[b_ndim-2];

// // //             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
// // //                                        ? CUDA_R_16F : CUDA_R_16BF;
// // //             cublasStatus_t status;
// // //             if (need_reduction) {
// // //                 // Flatten batch: treat A as [M*tb, K], grad_out as [M*tb, N]
// // //                 int M_total = M * tb;
// // //                 status = cublasGemmEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M_total, &alpha_f,
// // //                     go_ptr, cuda_type, ldgo_bwd,
// // //                     a_ptr,  cuda_type, lda_bwd,
// // //                     &beta_f, gb_ptr, cuda_type, ldgb_bwd,
// // //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //             } else {
// // //                 long long stride_a_ll  = (a_ndim  >= 3) ? (long long)meta.a_strides[a_ndim-3]       : 0LL;
// // //                 long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3]: 0LL;
// // //                 long long stride_gb_ll = (b_ndim  >= 3) ? (long long)meta.grad_b_strides[b_ndim-3]  : 0LL;
// // //                 status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M, &alpha_f,
// // //                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
// // //                     a_ptr,  cuda_type, lda_bwd,  stride_a_ll,
// // //                     &beta_f, gb_ptr, cuda_type, ldgb_bwd, stride_gb_ll, tb,
// // //                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //             }
// // //             if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
// // //         }
// // //    } else if constexpr (std::is_same<T, float>::value) {
// // //         if (supported_for_fast_path) {
// // //           // USE EXPLICIT DEVICE INDEXING from grad_output
// // //           int dev_idx = grad_output.device().index;
// // //           cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
// // //           cublasSetStream(handle, stream);
// // //           float alpha = 1.0f, beta = 0.0f;

// // //           cublasOperation_t opA = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N; // A^T
// // //           int lda = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1]; // Correct stride 
          
// // //           int ldgo = meta.grad_out_strides[go_ndim-2];
// // //           int ldgb = meta.grad_b_strides[b_ndim-2];
          
// // //           cublasStatus_t status;
          
// // //           if (need_reduction) {
// // //               // Flatten Batch strategy: Treat A as [TotalRows, K], GO as [TotalRows, N]
// // //               // grad_B = A^T @ GO.
// // //               // A^T [K, TotalRows]. GO [TotalRows, N]. Result [K, N].
// // //               int M_total = M * tb;
              
// // //               status = cublasGemmEx(handle, CUBLAS_OP_N, opA, N, K_a, M_total, &alpha, go_ptr, CUDA_R_32F, ldgo, a_ptr, CUDA_R_32F, lda, &beta, gb_ptr, CUDA_R_32F, ldgb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              
// // //           } else {
// // //              // Batched Strategy — compute strides from actual metadata
// // //              long long stride_a, stride_go, stride_gb;
// // //              bool batch_uniform_dB = true;

// // //              if (a_ndim <= 3) {
// // //                  stride_a = (a_ndim == 3) ? meta.a_strides[0] : 0;
// // //                  stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
// // //                  stride_gb = (b_ndim == 3) ? meta.grad_b_strides[0] : 0;
// // //              } else {
// // //                  // 4D: use dim[-3] strides (inner batch stride)
// // //                  stride_a = meta.a_strides[a_ndim - 3];
// // //                  stride_go = meta.grad_out_strides[go_ndim - 3];
// // //                  stride_gb = meta.grad_b_strides[b_ndim - 3];

// // //                  // Check if outer batch stride == inner_stride * inner_dim
// // //                  int* a_sh = meta.a_shape;
// // //                  int* go_sh = meta.grad_out_shape;
// // //                  int* b_sh = meta.b_shape;
// // //                  batch_uniform_dB = (meta.a_strides[0] == stride_a * a_sh[1]);
// // //                  batch_uniform_dB = batch_uniform_dB && (meta.grad_out_strides[0] == stride_go * go_sh[1]);
// // //                  batch_uniform_dB = batch_uniform_dB && (meta.grad_b_strides[0] == stride_gb * b_sh[1]);
// // //              }

// // //              if (batch_uniform_dB) {
// // //                  status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_ptr, CUDA_R_32F, ldgo, stride_go, a_ptr, CUDA_R_32F, lda, stride_a, &beta, gb_ptr, CUDA_R_32F, ldgb, stride_gb, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //              } else {
// // //                  // Non-uniform 4D: loop over outer batch dim d0
// // //                  int d0 = meta.grad_out_shape[0];
// // //                  int d1 = meta.grad_out_shape[1];
// // //                  status = CUBLAS_STATUS_SUCCESS;
// // //                  for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
// // //                      const T* a_batch = a_ptr + bi * meta.a_strides[0];
// // //                      const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
// // //                      T* gb_batch = gb_ptr + bi * meta.grad_b_strides[0];
// // //                      status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_batch, CUDA_R_32F, ldgo, stride_go, a_batch, CUDA_R_32F, lda, stride_a, &beta, gb_batch, CUDA_R_32F, ldgb, stride_gb, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// // //                  }
// // //              }
// // //           }
// // //           if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
// // //        }
// // //    }
   
// // //    if (!grad_b_done) {
// // //         #ifdef AUTOGRAD_PROFILER_ENABLED
// // //         if (M*N*K_a > 1000000) {
// // //             printf("dB Fallback: M=%d K=%d N=%d, Strides: a[%d,%d], go[%d,%d], gb[%d,%d]\n", 
// // //                    M, K_a, N, meta.a_strides[a_ndim-2], meta.a_strides[a_ndim-1],
// // //                    meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
// // //                    meta.grad_b_strides[b_ndim-2], meta.grad_b_strides[b_ndim-1]);
// // //         }
// // //         #endif
// // //        // Fallback
// // //        if constexpr (std::is_same<T, float>::value)
// // //           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream>>>(a_ptr, go_ptr, gb_ptr, M, K_a, N, tb, meta);
// // //        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// // //                           std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value)
// // //           throw std::runtime_error("matmul_backward dB: non-contiguous fp16/bf16 fallback kernel not available, ensure contiguous tensors");
// // //    }
   
// // //    cudaError_t err = cudaGetLastError();
// // //    if (err != cudaSuccess) {
// // //       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
// // //    }
// // // }

// // // void cuda_matmul_backward(
// // //     const Tensor& grad_output,
// // //     const Tensor& A,
// // //     const Tensor& B,
// // //     Tensor& grad_A,
// // //     Tensor& grad_B,
// // //     cudaStream_t stream)
// // // {
// // //    dispatch_by_dtype(A.dtype(), [&](auto d) {
// // //       using T = decltype(d);
// // //       if constexpr (std::is_same<T, float>::value ||
// // //                     std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
// // //                     std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
// // //          launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, stream);
// // //       } else {
// // //          throw std::runtime_error("cuda_matmul_backward: Unsupported type (only float, float16, bfloat16 supported)");
// // //       }
// // //    });
// // // }

// // // } // namespace matmul_backward_cublas
// // // } // namespace OwnTensor

// // // // ============================================================================
// // // // Public dispatcher: routes to MyBLAS or cuBLAS based on USE_BLU_BLAS env var.
// // // // ============================================================================
// // // namespace OwnTensor {

// // // #ifdef WITH_MYBLAS
// // // static inline bool blu_blas_enabled_bwd() {
// // //     const char* v = std::getenv("USE_BLU_BLAS");
// // //     return v && v[0] == '1' && v[1] == '\0';
// // // }
// // // #endif

// // // void cuda_matmul_backward(
// // //     const Tensor& grad_output,
// // //     const Tensor& A,
// // //     const Tensor& B,
// // //     Tensor& grad_A,
// // //     Tensor& grad_B,
// // //     cudaStream_t stream)
// // // {
// // // #ifdef WITH_MYBLAS
// // //     if (blu_blas_enabled_bwd()) {
// // //         matmul_backward_myblas::cuda_matmul_backward(grad_output, A, B, grad_A, grad_B, stream);
// // //         return;
// // //     }
// // // #endif
// // //     matmul_backward_cublas::cuda_matmul_backward(grad_output, A, B, grad_A, grad_B, stream);
// // // }

// // // } // namespace OwnTensor

// // // #endif // WITH_CUDA
































#ifdef WITH_CUDA

#ifdef WITH_MYBLAS


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "mycublas.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>

#include "ops/MatmulBackward.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "utils/Profiler.h"

namespace OwnTensor {

// mycublas handle management - thread-safe singleton per device
static std::mutex mycublas_mutex_bwd;
static mycublasHandle_t g_mycublas_handles_bwd[8]  = {nullptr};
static mycublasHandle_t g_mycublas_handles_bwd2[8] = {nullptr};  // second handle for parallel grad_B stream

// Per-device secondary stream and cached no-timing events for grad_A/grad_B overlap.
static cudaStream_t g_secondary_streams_bwd[8]  = {nullptr};
static cudaEvent_t  g_bwd_event_ready[8]        = {nullptr};   // primary → secondary (inputs ready)
static cudaEvent_t  g_bwd_event_done[8]         = {nullptr};   // secondary → primary (grad_B done)

static mycublasHandle_t get_mycublas_handle_bwd(int device = 0) {
   if (g_mycublas_handles_bwd[device] == nullptr) {
      std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
      if (g_mycublas_handles_bwd[device] == nullptr) {
         cudaSetDevice(device);
         mycublasCreate(&g_mycublas_handles_bwd[device]);
      }
   }
   return g_mycublas_handles_bwd[device];
}

static mycublasHandle_t get_mycublas_handle_bwd2(int device = 0) {
   if (g_mycublas_handles_bwd2[device] == nullptr) {
      std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
      if (g_mycublas_handles_bwd2[device] == nullptr) {
         cudaSetDevice(device);
         mycublasCreate(&g_mycublas_handles_bwd2[device]);
      }
   }
   return g_mycublas_handles_bwd2[device];
}

static cudaStream_t get_secondary_stream_bwd(int device = 0) {
   if (g_secondary_streams_bwd[device] == nullptr) {
      std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
      if (g_secondary_streams_bwd[device] == nullptr) {
         cudaSetDevice(device);
         // Non-blocking so it doesn't serialize with the default stream
         cudaStreamCreateWithFlags(&g_secondary_streams_bwd[device], cudaStreamNonBlocking);
      }
   }
   return g_secondary_streams_bwd[device];
}

static void init_bwd_events(int device) {
   if (g_bwd_event_ready[device] == nullptr) {
      std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
      if (g_bwd_event_ready[device] == nullptr) {
         cudaSetDevice(device);
         // DisableTiming avoids hardware timestamp recording, making record/wait ~10× cheaper
         cudaEventCreateWithFlags(&g_bwd_event_ready[device], cudaEventDisableTiming);
         cudaEventCreateWithFlags(&g_bwd_event_done[device],  cudaEventDisableTiming);
      }
   }
}

using namespace nvcuda;

// ============================================================================
// METADATA & CONSTANTS
// ============================================================================

struct BackwardMetadata {
   int a_shape[8], b_shape[8], grad_out_shape[8];
   int a_strides[8], b_strides[8], grad_out_strides[8];
   int grad_a_strides[8], grad_b_strides[8];
   int a_ndim, b_ndim, grad_out_ndim;
};

__device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
                                          int ndim, const int* out_shape, int out_ndim, int& offset) {
   offset = 0; 
   if (out_ndim <= 2) return;
   int temp_batch = batch_idx;
   for (int dim = out_ndim - 3; dim >= 0; --dim) {
      int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
      temp_batch /= b_dim_sz;
      int c_dim = dim - (out_ndim - ndim);
      if (c_dim >= 0 && c_dim < ndim - 2) 
         offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
   }
}

constexpr int BACKWARD_PAD = 8;

// ============================================================================
// FP32 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dA_fp32(
    const float* __restrict__ grad_out,  // [M, N]
    const float* __restrict__ B,         // [K, N]
    float* __restrict__ grad_A,          // [M, K]
    float beta,
    int M, int N, int K,
    int total_batches,
    BackwardMetadata meta) 
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const float *GO = grad_out + go_offset;
   const float *Bp = B + b_offset;
   float *GA = grad_A + ga_offset;
   
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
   float results[16] = {0.0f}, regM[4], regN[4];

   auto load_tiles = [&](int no, int idx) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gm = by * BM + r, gn = no + c;
         As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gn = no + r, gk = bx * BN + c;
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < N; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < N) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < M && c < K) {
            int idx = r * s_ga_m + c * s_ga_k;
            GA[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GA[idx]);
         }
      }
   }
}

// ============================================================================
// FP32 MANUAL BACKWARD KERNEL (Fallback): grad_B = A^T @ grad_output
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dB_fp32(
    const float* __restrict__ A,         // [M, K]
    const float* __restrict__ grad_out,  // [M, N]
    float* __restrict__ grad_B,          // [K, N]
    float beta,
    int M, int K, int N,
    int total_batches,
    BackwardMetadata meta)
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int a_offset = 0, go_offset = 0, gb_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, a_offset);
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
   const float *Ap = A + a_offset;
   const float *GO = grad_out + go_offset;
   float *GB = grad_B + gb_offset;
   
   int s_a_m = meta.a_strides[meta.a_ndim-2];
   int s_a_k = meta.a_strides[meta.a_ndim-1];
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
   int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
   float results[16] = {0.0f}, regM[4], regN[4];

   auto load_tiles = [&](int mo, int idx) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gk = by * BM + r, gm = mo + c;
         As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
      }
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gm = mo + r, gn = bx * BN + c;
         Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < M; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < M) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < K && c < N) {
            int idx = r * s_gb_k + c * s_gb_n;
            GB[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GB[idx]);
         }
      }
   }
}

// ============================================================================
// FP16 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_backward_dA_fp16(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ B,
    __half* __restrict__ grad_A,
    float beta,
    int M, int N, int K,
    int total_batches,
    BackwardMetadata meta)
{
   const int batch_idx = blockIdx.z; 
   if (batch_idx >= total_batches) return;
   const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const __half *GO = grad_out + go_offset;
   const __half *Bp = B + b_offset;
   __half *GA = grad_A + ga_offset;
   
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ __half As[2][BM][BK + BACKWARD_PAD];
   __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
   wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
   #pragma unroll
   for(int i = 0; i < 2; i++) 
      for(int j = 0; j < 2; j++) 
         wmma::fill_fragment(acc[i][j], __float2half(0.0f));

   auto load_tiles = [&](int no, int idx) {
      for (int i = tid; i < BM * BK; i += 512) {
         int r = i / BK, c = i % BK;
         int gm = blockIdx.y * BM + r, gn = no + c;
         As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
      }
      for (int i = tid; i < BK * BN; i += 512) {
         int r = i / BN, c = i % BN;
         int gn = no + r, gk = blockIdx.x * BN + c;
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
      }
   };
   
   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int k = 0; k < N; k += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (k + BK < N) load_tiles(k + BK, wi);
      
      #pragma unroll
      for (int ks = 0; ks < BK; ks += 16) {
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
               wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
               wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
         }
      }
      __syncthreads();
   }
   
   __half* sm = reinterpret_cast<__half*>(As);
   #pragma unroll
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         int cr = blockIdx.y * BM + warp_row * WM + i * 16;
         int cc = blockIdx.x * BN + warp_col * WN + j * 16;
         if (cr < M && cc < K) {
            __half* wsm = sm + warp_id * 256;
            wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
            for (int r = 0; r < 16; r++) 
               for (int c = 0; c < 16; c++)
                  if (cr + r < M && cc + c < K) {
                     int idx = (cr + r) * s_ga_m + (cc + c) * s_ga_k;
                     GA[idx] = wsm[r * 16 + c] + (__half)beta * GA[idx];
                  }
         }
      }
   }
}

// ============================================================================
// DISPATCH LAYER
// ============================================================================

template<typename T>
void launch_backward_matmul(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    float beta_a,
    float beta_b,
    cudaStream_t stream)
{
   const auto& go_sh = grad_output.shape().dims;
   const auto& a_sh = A.shape().dims;
   const auto& b_sh = B.shape().dims;
   
   int go_ndim = go_sh.size();
   int a_ndim = a_sh.size();
   int b_ndim = b_sh.size();
   
   int M = go_sh[go_ndim - 2];
   int N = go_sh[go_ndim - 1];
   int K_a = a_sh[a_ndim - 1];
   
   int tb = 1;
   for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
   BackwardMetadata meta;
   meta.a_ndim = a_ndim;
   meta.b_ndim = b_ndim;
   meta.grad_out_ndim = go_ndim;
   
   for (int i = 0; i < a_ndim; i++) {
      meta.a_shape[i] = a_sh[i];
      meta.a_strides[i] = A.stride().strides[i];
      meta.grad_a_strides[i] = grad_A.stride().strides[i];
   }
   for (int i = 0; i < b_ndim; i++) {
      meta.b_shape[i] = b_sh[i];
      meta.b_strides[i] = B.stride().strides[i];
      meta.grad_b_strides[i] = grad_B.stride().strides[i];
   }
   for (int i = 0; i < go_ndim; i++) {
      meta.grad_out_shape[i] = go_sh[i];
      meta.grad_out_strides[i] = grad_output.stride().strides[i];
   }
   
   const T* go_ptr = grad_output.data<T>();
   const T* a_ptr = A.data<T>();
   const T* b_ptr = B.data<T>();
   T* ga_ptr = grad_A.data<T>();
   T* gb_ptr = grad_B.data<T>();
   
   // Accept both row-major (strides[-1]==1) and col-major/transposed (strides[-2]==1) for A and B.
   // grad_out and grad outputs must always be row-major (strides[-1]==1) because we write to them.
   bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
   bool a_row_major   = (meta.a_strides[a_ndim-1] == 1);
   bool a_col_major   = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
   bool b_row_major   = (meta.b_strides[b_ndim-1] == 1);
   bool b_col_major   = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
   bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
   bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

   bool supported_for_fast_path = go_contiguous && (a_row_major || a_col_major) &&
                                  (b_row_major || b_col_major) && ga_contiguous && gb_contiguous;

   // For grad_A = grad_out @ B^T:
   //   B row-major (strides[-1]==1): stored normally, need MYCUBLAS_OP_T to transpose, ld = strides[-2]
   //   B col-major (strides[-2]==1): already stored as B^T, use MYCUBLAS_OP_N, ld = strides[-1]
   mycublasOperation_t b_op_ga = b_row_major ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;
   int ldb_ga = b_row_major ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];

   // For grad_B = A^T @ grad_out:
   //   A row-major (strides[-1]==1): stored normally, need MYCUBLAS_OP_T to transpose, ld = strides[-2]
   //   A col-major (strides[-2]==1): already stored as A^T, use MYCUBLAS_OP_N, ld = strides[-1]
   mycublasOperation_t a_op_gb = a_row_major ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;
   int lda_gb = a_row_major ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];

    AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");

   // ---- Acquire device, handles, and secondary stream once ----
   // grad_A = grad_output @ B^T  and  grad_B = A^T @ grad_output are fully independent.
   // Dispatch them on separate cuBLAS handles bound to separate streams so the GPU
   // can overlap them.  Events are cached per-device and use DisableTiming for
   // minimal overhead.
   int dev = 0;
   cudaGetDevice(&dev);
   mycublasHandle_t handle_a = get_mycublas_handle_bwd(dev);
   mycublasHandle_t handle_b = get_mycublas_handle_bwd2(dev);
   cudaStream_t stream_b     = get_secondary_stream_bwd(dev);
   init_bwd_events(dev);

   mycublasSetStream(handle_a, stream);
   mycublasSetStream(handle_b, stream_b);

   // stream_b must wait until all upstream data (grad_output, A, B) produced on
   // the primary stream is visible before it starts reading them.
   cudaEventRecord(g_bwd_event_ready[dev], stream);
   cudaStreamWaitEvent(stream_b, g_bwd_event_ready[dev], 0);

   // ---- grad_A = grad_output @ B^T  (primary stream) ----
   bool grad_a_done = false;
   if constexpr (std::is_same<T, float>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldga = meta.grad_a_strides[a_ndim-2];
          long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
          long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
          long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;

          mycublasSgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, 1.0f, go_ptr, ldgo, s_go, b_ptr, ldb_ga, s_b, beta_a, ga_ptr, ldga, s_ga, tb);
          grad_a_done = true;
       }
   } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldga = meta.grad_a_strides[a_ndim-2];
          long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
          long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
          long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
          const __half alpha = __float2half(1.0f), h_beta_a = __float2half(beta_a);
          mycublasHgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, alpha, (const __half*)go_ptr, ldgo, s_go, (const __half*)b_ptr, ldb_ga, s_b, h_beta_a, (__half*)ga_ptr, ldga, s_ga, tb);
          grad_a_done = true;
       }
   } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldga = meta.grad_a_strides[a_ndim-2];
          long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
          long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
          long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
          const __nv_bfloat16 alpha = __float2bfloat16(1.0f), b_beta_a = __float2bfloat16(beta_a);
          mycublasBgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, alpha, (const __nv_bfloat16*)go_ptr, ldgo, s_go, (const __nv_bfloat16*)b_ptr, ldb_ga, s_b, b_beta_a, (__nv_bfloat16*)ga_ptr, ldga, s_ga, tb);
          grad_a_done = true;
       }
   }

   if (!grad_a_done) {
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, beta_a, M, N, K_a, tb, meta);
       else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
          matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), beta_a, M, N, K_a, tb, meta);
   }

   // ---- grad_B = A^T @ grad_output  (secondary stream — runs in parallel with grad_A above) ----
   bool grad_b_done = false;
   bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[b_ndim-3] == 1 && meta.grad_out_shape[go_ndim-3] > 1);

   if constexpr (std::is_same<T, float>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldgb = meta.grad_b_strides[b_ndim-2];
          if (need_reduction) {
              int M_total = M * tb;
              mycublasSgemm(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, 1.0f, a_ptr, lda_gb, go_ptr, ldgo, beta_b, gb_ptr, ldgb);
          } else {
             long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]            : 0;
             long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3]    : 0;
             long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]       : 0;

             mycublasSgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, 1.0f, a_ptr, lda_gb, s_a, go_ptr, ldgo, s_go, beta_b, gb_ptr, ldgb, s_gb, tb);
          }
          grad_b_done = true;
       }
   } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldgb = meta.grad_b_strides[b_ndim-2];
          const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
          if (need_reduction) {
              int M_total = M * tb;
              const __half h_beta_b = __float2half(beta_b);
              mycublasHgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __half*)a_ptr, lda_gb, 0, (const __half*)go_ptr, ldgo, 0, h_beta_b, (__half*)gb_ptr, ldgb, 0, 1);
          } else {
             long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
             long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
             long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
             mycublasHgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __half*)a_ptr, lda_gb, s_a, (const __half*)go_ptr, ldgo, s_go, beta, (__half*)gb_ptr, ldgb, s_gb, tb);
          }
          grad_b_done = true;
       }
   } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
       if (supported_for_fast_path) {
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldgb = meta.grad_b_strides[b_ndim-2];
          const __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(0.0f);
          if (need_reduction) {
              int M_total = M * tb;
              const __nv_bfloat16 b_beta_b = __float2bfloat16(beta_b);
              mycublasBgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __nv_bfloat16*)a_ptr, lda_gb, 0, (const __nv_bfloat16*)go_ptr, ldgo, 0, b_beta_b, (__nv_bfloat16*)gb_ptr, ldgb, 0, 1);
          } else {
             long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
             long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
             long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
             mycublasBgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __nv_bfloat16*)a_ptr, lda_gb, s_a, (const __nv_bfloat16*)go_ptr, ldgo, s_go, beta, (__nv_bfloat16*)gb_ptr, ldgb, s_gb, tb);
          }
          grad_b_done = true;
       }
   }

   if (!grad_b_done) {
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream_b>>>(a_ptr, go_ptr, gb_ptr, beta_b, M, K_a, N, tb, meta);
   }

   // Rejoin stream_b into the primary stream so downstream ops on stream see grad_B.
   cudaEventRecord(g_bwd_event_done[dev], stream_b);
   cudaStreamWaitEvent(stream, g_bwd_event_done[dev], 0);

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
   }
}

void cuda_matmul_backward(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
   float beta_a = 0.0f;
   float beta_b = 0.0f;
   
   dispatch_by_dtype(A.dtype(), [&](auto d) {
      using T = decltype(d);
      launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, beta_a, beta_b, stream);
   });
}

} // namespace OwnTensor

#else // WITH_MYBLAS


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <mutex>

#include "ops/MatmulBackward.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "utils/Profiler.h"

namespace OwnTensor {

// cuBLAS handle management - thread-safe singleton per device
static std::mutex cublas_mutex_bwd;
static cublasHandle_t g_cublas_handles_bwd[8] = {nullptr};

static cublasHandle_t get_cublas_handle_bwd(int device = 0) {
   if (g_cublas_handles_bwd[device] == nullptr) {
      std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
      if (g_cublas_handles_bwd[device] == nullptr) {
         cudaSetDevice(device);
         cublasCreate(&g_cublas_handles_bwd[device]);
         // Enable TF32 for FP32 matmuls on Ampere+ GPUs for significant speedup
         cublasSetMathMode(g_cublas_handles_bwd[device], CUBLAS_TF32_TENSOR_OP_MATH);
      }
   }
   return g_cublas_handles_bwd[device];
}

using namespace nvcuda;

// ============================================================================
// METADATA & CONSTANTS (Reuse from forward)
// ============================================================================

struct BackwardMetadata {
   int a_shape[8], b_shape[8], grad_out_shape[8];
   int a_strides[8], b_strides[8], grad_out_strides[8];
   int grad_a_strides[8], grad_b_strides[8];
   int a_ndim, b_ndim, grad_out_ndim;
};

__device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
                                          int ndim, const int* out_shape, int out_ndim, int& offset) {
   offset = 0; 
   if (out_ndim <= 2) return;
   int temp_batch = batch_idx;
   for (int dim = out_ndim - 3; dim >= 0; --dim) {
      int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
      temp_batch /= b_dim_sz;
      int c_dim = dim - (out_ndim - ndim);
      if (c_dim >= 0 && c_dim < ndim - 2) 
         offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
   }
}

constexpr int BACKWARD_PAD = 8;

// ============================================================================
// FP32 BACKWARD KERNEL: grad_A = grad_output @ B^T
// Computes: C[M,K] = A[M,N] @ B[K,N]^T = A[M,N] @ B^T[N,K]
// Note: We read B in transposed order (columns become rows)
// BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dA_fp32(
    const float* __restrict__ grad_out,  // [M, N] - grad of output
    const float* __restrict__ B,         // [K, N] - second input of forward, read transposed
    float* __restrict__ grad_A,          // [M, K] - gradient for A
    int M, int N, int K,                 // M=rows of grad_out, N=cols of grad_out (=cols of B), K=rows of B
    int total_batches,
    BackwardMetadata meta) 
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const float *GO = grad_out + go_offset;
   const float *Bp = B + b_offset;
   float *GA = grad_A + ga_offset;
   
   // Strides for grad_output [M, N]
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
   // Strides for B [K, N] - we read it transposed as B^T [N, K]
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   
   // Strides for grad_A [M, K]
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];  // grad_out tile: [BK, BM] transposed storage
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // B^T tile: [BK, BN]
   float results[16] = {0.0f}, regM[4], regN[4];

   // Load tiles with double buffering
   // For grad_A = grad_out @ B^T:
   //   We iterate over N (the shared dimension between grad_out and B^T)
   //   grad_out is [M, N], B^T is [N, K]
   //   So we load BK columns of grad_out into shared As
   //   And BK rows of B^T (= BK columns of B) into shared Bs
   
   auto load_tiles = [&](int no, int idx) {
      // Load grad_out[by*BM : by*BM+BM, no : no+BK] into As (transposed for coalescing)
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gm = by * BM + r, gn = no + c;
         As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
      
      // Load B^T[no : no+BK, bx*BN : bx*BN+BN] = B[bx*BN : bx*BN+BN, no : no+BK]^T
      // B is [K, N], B^T is [N, K]. We want B^T[n, k] = B[k, n]
      // So for k in [bx*BN, bx*BN+BN) and n in [no, no+BK)
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gn = no + r, gk = bx * BN + c;
         // B^T[gn, gk] = B[gk, gn]
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < N; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < N) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   // Store results to grad_A
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < M && c < K) {
            GA[r * s_ga_m + c * s_ga_k] = results[i*4+j];
         }
      }
   }
}

// ============================================================================
// FP32 BACKWARD KERNEL: grad_B = A^T @ grad_output
// Computes: C[K,N] = A[M,K]^T @ B[M,N] = A^T[K,M] @ B[M,N]
// Note: We read A in transposed order
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_backward_dB_fp32(
    const float* __restrict__ A,         // [M, K] - first input of forward, read transposed
    const float* __restrict__ grad_out,  // [M, N] - grad of output
    float* __restrict__ grad_B,          // [K, N] - gradient for B
    int M, int K, int N,                 // M=rows of A, K=cols of A, N=cols of grad_out
    int total_batches,
    BackwardMetadata meta)
{
   const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
   if (b_idx >= total_batches) return;
   const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
   int a_offset = 0, go_offset = 0, gb_offset = 0;
   compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, a_offset);
   compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
                            meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
   const float *Ap = A + a_offset;
   const float *GO = grad_out + go_offset;
   float *GB = grad_B + gb_offset;
   
   // Strides for A [M, K] - we read it transposed as A^T [K, M]
   int s_a_m = meta.a_strides[meta.a_ndim-2];
   int s_a_k = meta.a_strides[meta.a_ndim-1];
   
   // Strides for grad_output [M, N]
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
   // Strides for grad_B [K, N]
   int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
   int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

   __shared__ float As[2][BK][BM + BACKWARD_PAD];  // A^T tile: [BK, BM]
   __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // grad_out tile: [BK, BN]
   float results[16] = {0.0f}, regM[4], regN[4];

   // For grad_B = A^T @ grad_out:
   //   We iterate over M (the shared dimension)
   //   A^T is [K, M], grad_out is [M, N]
   //   So we load BK rows of A^T (= BK columns of A) into shared As
   //   And BK rows of grad_out into shared Bs
   
   auto load_tiles = [&](int mo, int idx) {
      // Load A^T[by*BM : by*BM+BM, mo : mo+BK] = A[mo : mo+BK, by*BM : by*BM+BM]^T
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BK, c = li % BK;
         int gk = by * BM + r, gm = mo + c;
         // A^T[gk, gm] = A[gm, gk]
         As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
      }
      
      // Load grad_out[mo : mo+BK, bx*BN : bx*BN+BN] into Bs
      #pragma unroll
      for (int i = 0; i < 2; i++) {
         int li = tid + i * 1024, r = li / BN, c = li % BN;
         int gm = mo + r, gn = bx * BN + c;
         Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
      }
   };

   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int bk = 0; bk < M; bk += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (bk + BK < M) load_tiles(bk + BK, wi);
      
      #pragma unroll
      for (int d = 0; d < BK; d++) {
         #pragma unroll
         for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
         #pragma unroll
         for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
         #pragma unroll
         for (int i = 0; i < 4; i++) 
            for (int j = 0; j < 4; j++) 
               results[i*4+j] += regM[i] * regN[j];
      }
      __syncthreads();
   }
   
   // Store results to grad_B
   for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
         int r = by * BM + tRow * 4 + i;
         int c = bx * BN + tCol * 4 + j;
         if (r < K && c < N) {
            GB[r * s_gb_k + c * s_gb_n] = results[i*4+j];
         }
      }
   }
}

// ============================================================================
// FP16 BACKWARD KERNEL: grad_A = grad_output @ B^T (WMMA)
// ============================================================================

template<int BM, int BN, int BK, int WM, int WN>
__global__ void matmul_backward_dA_fp16(
    const __half* __restrict__ grad_out,
    const __half* __restrict__ B,
    __half* __restrict__ grad_A,
    int M, int N, int K,
    int total_batches,
    BackwardMetadata meta)
{
   const int batch_idx = blockIdx.z; 
   if (batch_idx >= total_batches) return;
   const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
   int go_offset = 0, b_offset = 0, ga_offset = 0;
   compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, go_offset);
   compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, b_offset);
   compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
                            meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
   const __half *GO = grad_out + go_offset;
   const __half *Bp = B + b_offset;
   __half *GA = grad_A + ga_offset;
   
   int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
   int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   int s_b_k = meta.b_strides[meta.b_ndim-2];
   int s_b_n = meta.b_strides[meta.b_ndim-1];
   int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
   int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

   __shared__ __half As[2][BM][BK + BACKWARD_PAD];
   __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
   wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
   wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
   #pragma unroll
   for(int i = 0; i < 2; i++) 
      for(int j = 0; j < 2; j++) 
         wmma::fill_fragment(acc[i][j], __float2half(0.0f));

   auto load_tiles = [&](int no, int idx) {
      for (int i = tid; i < BM * BK; i += 512) {
         int r = i / BK, c = i % BK;
         int gm = blockIdx.y * BM + r, gn = no + c;
         As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
      }
      for (int i = tid; i < BK * BN; i += 512) {
         int r = i / BN, c = i % BN;
         int gn = no + r, gk = blockIdx.x * BN + c;
         Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
      }
   };
   
   int wi = 0; 
   load_tiles(0, wi); 
   __syncthreads();
   
   for (int k = 0; k < N; k += BK) {
      int ri = wi; 
      wi = 1 - wi;
      if (k + BK < N) load_tiles(k + BK, wi);
      
      #pragma unroll
      for (int ks = 0; ks < BK; ks += 16) {
         #pragma unroll
         for (int i = 0; i < 2; i++) {
            wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
            #pragma unroll
            for (int j = 0; j < 2; j++) {
               wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
               wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
         }
      }
      __syncthreads();
   }
   
   __half* sm = reinterpret_cast<__half*>(As);
   #pragma unroll
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         int cr = blockIdx.y * BM + warp_row * WM + i * 16;
         int cc = blockIdx.x * BN + warp_col * WN + j * 16;
         if (cr < M && cc < K) {
            __half* wsm = sm + warp_id * 256;
            wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
            for (int r = 0; r < 16; r++) 
               for (int c = 0; c < 16; c++)
                  if (cr + r < M && cc + c < K) 
                     GA[(cr + r) * s_ga_m + (cc + c) * s_ga_k] = wsm[r * 16 + c];
         }
      }
   }
}

// ============================================================================
// DISPATCH LAYER
// ============================================================================

template<typename T>
void launch_backward_matmul(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
   const auto& go_sh = grad_output.shape().dims;
   const auto& a_sh = A.shape().dims;
   const auto& b_sh = B.shape().dims;
   
   int go_ndim = go_sh.size();
   int a_ndim = a_sh.size();
   int b_ndim = b_sh.size();
   
   int M = go_sh[go_ndim - 2];  // rows of grad_output
   int N = go_sh[go_ndim - 1];  // cols of grad_output (= cols of B)
   int K_a = a_sh[a_ndim - 1];  // cols of A (= rows of grad_A's cols dim)
   int K_b = b_sh[b_ndim - 2];  // rows of B
   
   int tb = 1;
   for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
   BackwardMetadata meta;
   meta.a_ndim = a_ndim;
   meta.b_ndim = b_ndim;
   meta.grad_out_ndim = go_ndim;
   
   for (int i = 0; i < a_ndim; i++) {
      meta.a_shape[i] = a_sh[i];
      meta.a_strides[i] = A.stride().strides[i];
      meta.grad_a_strides[i] = grad_A.stride().strides[i];
   }
   for (int i = 0; i < b_ndim; i++) {
      meta.b_shape[i] = b_sh[i];
      meta.b_strides[i] = B.stride().strides[i];
      meta.grad_b_strides[i] = grad_B.stride().strides[i];
   }
   for (int i = 0; i < go_ndim; i++) {
      meta.grad_out_shape[i] = go_sh[i];
      meta.grad_out_strides[i] = grad_output.stride().strides[i];
   }
   
   const T* go_ptr = grad_output.data<T>();
   const T* a_ptr = A.data<T>();
   const T* b_ptr = B.data<T>();
   T* ga_ptr = grad_A.data<T>();
   T* gb_ptr = grad_B.data<T>();
   
   // Check stride attributes
   bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
   bool a_contiguous = (meta.a_strides[a_ndim-1] == 1);
   bool a_transposed = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
   bool b_contiguous = (meta.b_strides[b_ndim-1] == 1);
   bool b_transposed = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
   bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
   bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

   bool supported_for_fast_path = go_contiguous && (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && ga_contiguous && gb_contiguous;

    AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");
    bool grad_a_done = false;
    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
                  std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
        // cuBLAS path for FP16/BF16: grad_A = grad_output @ B^T
        // compute in fp16/bf16, accumulate in fp32
        if (supported_for_fast_path) {
            int dev_idx = grad_output.device().index;
            cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
            cublasSetStream(handle, stream);
            float alpha_f = 1.0f, beta_f = 0.0f;

            cublasOperation_t opB_bwd = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
            int ldb_bwd = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
            int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
            int ldga_bwd = meta.grad_a_strides[a_ndim-2];

            long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3] : 0LL;
            long long stride_b_ll  = (b_ndim  >= 3) ? (long long)meta.b_strides[b_ndim-3]         : 0LL;
            long long stride_ga_ll = (a_ndim  >= 3) ? (long long)meta.grad_a_strides[a_ndim-3]    : 0LL;
            if (b_ndim < go_ndim) stride_b_ll = 0LL;

            cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
                                       ? CUDA_R_16F : CUDA_R_16BF;
            cublasStatus_t status;
            if (tb == 1) {
                status = cublasGemmEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
                    b_ptr,  cuda_type, ldb_bwd,
                    go_ptr, cuda_type, ldgo_bwd,
                    &beta_f, ga_ptr, cuda_type, ldga_bwd,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else {
                status = cublasGemmStridedBatchedEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
                    b_ptr,  cuda_type, ldb_bwd,  stride_b_ll,
                    go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
                    &beta_f, ga_ptr, cuda_type, ldga_bwd, stride_ga_ll, tb,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
        }
    } else if constexpr (std::is_same<T, float>::value) {
        if (supported_for_fast_path) {
          // USE EXPLICIT DEVICE INDEXING from grad_output
          int dev_idx = grad_output.device().index;
          cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
          cublasSetStream(handle, stream);
          float alpha = 1.0f, beta = 0.0f;
          
          cublasOperation_t opB = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
          int ldb = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldga = meta.grad_a_strides[a_ndim-2];
          
          // Compute batch strides from actual tensor metadata
          long long stride_go, stride_b_val, stride_ga;
          bool batch_uniform_dA = true;

          if (go_ndim <= 3) {
              stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
              stride_b_val = (b_ndim == 3) ? meta.b_strides[0] : 0;
              stride_ga = (go_ndim == 3) ? meta.grad_a_strides[0] : 0;
              if (b_ndim < go_ndim) stride_b_val = 0;
          } else {
              // 4D: use inner batch stride (strides[ndim-3])
              stride_go = meta.grad_out_strides[go_ndim - 3];
              stride_b_val = (b_ndim >= 4) ? meta.b_strides[b_ndim - 3] : 0;
              stride_ga = meta.grad_a_strides[a_ndim - 3];
              if (b_ndim < go_ndim) stride_b_val = 0;

              // Check uniform batch strides: stride[0] == stride[1] * shape[1]
              batch_uniform_dA = (meta.grad_out_strides[0] == stride_go * go_sh[1]);
              if (b_ndim >= 4) batch_uniform_dA = batch_uniform_dA && (meta.b_strides[0] == stride_b_val * b_sh[1]);
              batch_uniform_dA = batch_uniform_dA && (meta.grad_a_strides[0] == stride_ga * a_sh[1]);
          }

          cublasStatus_t status;
          if (tb == 1) {
              status = cublasGemmEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, go_ptr, CUDA_R_32F, ldgo, &beta, ga_ptr, CUDA_R_32F, ldga, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          } else if (batch_uniform_dA) {
              status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, stride_b_val, go_ptr, CUDA_R_32F, ldgo, stride_go, &beta, ga_ptr, CUDA_R_32F, ldga, stride_ga, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          } else {
              // Non-uniform 4D: loop over outer batch dim
              int d0 = go_sh[0], d1 = go_sh[1];
              status = CUBLAS_STATUS_SUCCESS;
              for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
                  const T* b_batch = b_ptr + (b_ndim >= 4 ? bi * meta.b_strides[0] : 0);
                  const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
                  T* ga_batch = ga_ptr + bi * meta.grad_a_strides[0];
                  status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_batch, CUDA_R_32F, ldb, stride_b_val, go_batch, CUDA_R_32F, ldgo, stride_go, &beta, ga_batch, CUDA_R_32F, ldga, stride_ga, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              }
          }

          if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
       }
   }
   
   if (!grad_a_done) {
        #ifdef AUTOGRAD_PROFILER_ENABLED
        if (M*N*K_a > 1000000) {
            printf("dA Fallback: M=%d N=%d K=%d, Strides: go[%d,%d], b[%d,%d], ga[%d,%d]\n", 
                   M, N, K_a, meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
                   meta.b_strides[b_ndim-2], meta.b_strides[b_ndim-1],
                   meta.grad_a_strides[a_ndim-2], meta.grad_a_strides[a_ndim-1]);
        }
        #endif
       // Fallback
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, M, N, K_a, tb, meta);
       else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
          matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), M, N, K_a, tb, meta);
       // bf16 fallback: cuBLAS path above handles all standard layouts; non-standard bf16 not supported
   }

   // Launch grad_B kernel: grad_B = A^T @ grad_output
   // Handle Reduction if B dim < Output dim (Broadcast in forward)
   
   bool grad_b_actions_needed = true; // Could optimize if not req grad
   bool grad_b_done = false;
   
   // Check reduction need
   bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[0] == 1);
   // Standard broadcast check: if tb > 1 but grad_B is not batched.
   // Assuming grad_B matches B shape.
   if (tb > 1 && b_ndim < go_ndim) need_reduction = true;

   if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
                 std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
        // cuBLAS path for FP16/BF16: grad_B = A^T @ grad_output
        // compute in fp16/bf16, accumulate in fp32
        if (supported_for_fast_path) {
            int dev_idx = grad_output.device().index;
            cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
            cublasSetStream(handle, stream);
            float alpha_f = 1.0f, beta_f = 0.0f;

            cublasOperation_t opA_bwd = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
            int lda_bwd  = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];
            int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
            int ldgb_bwd = meta.grad_b_strides[b_ndim-2];

            cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
                                       ? CUDA_R_16F : CUDA_R_16BF;
            cublasStatus_t status;
            if (need_reduction) {
                // Flatten batch: treat A as [M*tb, K], grad_out as [M*tb, N]
                int M_total = M * tb;
                status = cublasGemmEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M_total, &alpha_f,
                    go_ptr, cuda_type, ldgo_bwd,
                    a_ptr,  cuda_type, lda_bwd,
                    &beta_f, gb_ptr, cuda_type, ldgb_bwd,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else {
                long long stride_a_ll  = (a_ndim  >= 3) ? (long long)meta.a_strides[a_ndim-3]       : 0LL;
                long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3]: 0LL;
                long long stride_gb_ll = (b_ndim  >= 3) ? (long long)meta.grad_b_strides[b_ndim-3]  : 0LL;
                status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M, &alpha_f,
                    go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
                    a_ptr,  cuda_type, lda_bwd,  stride_a_ll,
                    &beta_f, gb_ptr, cuda_type, ldgb_bwd, stride_gb_ll, tb,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
        }
   } else if constexpr (std::is_same<T, float>::value) {
        if (supported_for_fast_path) {
          // USE EXPLICIT DEVICE INDEXING from grad_output
          int dev_idx = grad_output.device().index;
          cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
          cublasSetStream(handle, stream);
          float alpha = 1.0f, beta = 0.0f;

          cublasOperation_t opA = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N; // A^T
          int lda = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1]; // Correct stride 
          
          int ldgo = meta.grad_out_strides[go_ndim-2];
          int ldgb = meta.grad_b_strides[b_ndim-2];
          
          cublasStatus_t status;
          
          if (need_reduction) {
              // Flatten Batch strategy: Treat A as [TotalRows, K], GO as [TotalRows, N]
              // grad_B = A^T @ GO.
              // A^T [K, TotalRows]. GO [TotalRows, N]. Result [K, N].
              int M_total = M * tb;
              
              status = cublasGemmEx(handle, CUBLAS_OP_N, opA, N, K_a, M_total, &alpha, go_ptr, CUDA_R_32F, ldgo, a_ptr, CUDA_R_32F, lda, &beta, gb_ptr, CUDA_R_32F, ldgb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              
          } else {
             // Batched Strategy — compute strides from actual metadata
             long long stride_a, stride_go, stride_gb;
             bool batch_uniform_dB = true;

             if (a_ndim <= 3) {
                 stride_a = (a_ndim == 3) ? meta.a_strides[0] : 0;
                 stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
                 stride_gb = (b_ndim == 3) ? meta.grad_b_strides[0] : 0;
             } else {
                 // 4D: use dim[-3] strides (inner batch stride)
                 stride_a = meta.a_strides[a_ndim - 3];
                 stride_go = meta.grad_out_strides[go_ndim - 3];
                 stride_gb = meta.grad_b_strides[b_ndim - 3];

                 // Check if outer batch stride == inner_stride * inner_dim
                 int* a_sh = meta.a_shape;
                 int* go_sh = meta.grad_out_shape;
                 int* b_sh = meta.b_shape;
                 batch_uniform_dB = (meta.a_strides[0] == stride_a * a_sh[1]);
                 batch_uniform_dB = batch_uniform_dB && (meta.grad_out_strides[0] == stride_go * go_sh[1]);
                 batch_uniform_dB = batch_uniform_dB && (meta.grad_b_strides[0] == stride_gb * b_sh[1]);
             }

             if (batch_uniform_dB) {
                 status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_ptr, CUDA_R_32F, ldgo, stride_go, a_ptr, CUDA_R_32F, lda, stride_a, &beta, gb_ptr, CUDA_R_32F, ldgb, stride_gb, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
             } else {
                 // Non-uniform 4D: loop over outer batch dim d0
                 int d0 = meta.grad_out_shape[0];
                 int d1 = meta.grad_out_shape[1];
                 status = CUBLAS_STATUS_SUCCESS;
                 for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
                     const T* a_batch = a_ptr + bi * meta.a_strides[0];
                     const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
                     T* gb_batch = gb_ptr + bi * meta.grad_b_strides[0];
                     status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_batch, CUDA_R_32F, ldgo, stride_go, a_batch, CUDA_R_32F, lda, stride_a, &beta, gb_batch, CUDA_R_32F, ldgb, stride_gb, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                 }
             }
          }
          if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
       }
   }
   
   if (!grad_b_done) {
        #ifdef AUTOGRAD_PROFILER_ENABLED
        if (M*N*K_a > 1000000) {
            printf("dB Fallback: M=%d K=%d N=%d, Strides: a[%d,%d], go[%d,%d], gb[%d,%d]\n", 
                   M, K_a, N, meta.a_strides[a_ndim-2], meta.a_strides[a_ndim-1],
                   meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
                   meta.grad_b_strides[b_ndim-2], meta.grad_b_strides[b_ndim-1]);
        }
        #endif
       // Fallback
       if constexpr (std::is_same<T, float>::value)
          matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream>>>(a_ptr, go_ptr, gb_ptr, M, K_a, N, tb, meta);
       else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
                          std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value)
          throw std::runtime_error("matmul_backward dB: non-contiguous fp16/bf16 fallback kernel not available, ensure contiguous tensors");
   }
   
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
   }
}

void cuda_matmul_backward(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
   dispatch_by_dtype(A.dtype(), [&](auto d) {
      using T = decltype(d);
      if constexpr (std::is_same<T, float>::value ||
                    std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
                    std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
         launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, stream);
      } else {
         throw std::runtime_error("cuda_matmul_backward: Unsupported type (only float, float16, bfloat16 supported)");
      }
   });
}

} // namespace OwnTensor


#endif // WITH_MYBLAS

#endif // WITH_CUDA
























































//-----------------------------------------------------------------------multistream off -=--------------------=--=-==========-=-=--------------==================



































// #ifdef WITH_CUDA

// #ifdef WITH_MYBLAS


// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <mma.h>
// #include "mycublas.h"
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
// #include <string>
// #include <mutex>

// #include "ops/MatmulBackward.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "utils/Profiler.h"

// namespace OwnTensor {

// // mycublas handle management - thread-safe singleton per device
// static std::mutex mycublas_mutex_bwd;
// static mycublasHandle_t g_mycublas_handles_bwd[8]  = {nullptr};
// static mycublasHandle_t g_mycublas_handles_bwd2[8] = {nullptr};  // second handle for parallel grad_B stream

// // Per-device secondary stream and cached no-timing events for grad_A/grad_B overlap.
// static cudaStream_t g_secondary_streams_bwd[8]  = {nullptr};
// static cudaEvent_t  g_bwd_event_ready[8]        = {nullptr};   // primary → secondary (inputs ready)
// static cudaEvent_t  g_bwd_event_done[8]         = {nullptr};   // secondary → primary (grad_B done)

// static mycublasHandle_t get_mycublas_handle_bwd(int device = 0) {
//    if (g_mycublas_handles_bwd[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
//       if (g_mycublas_handles_bwd[device] == nullptr) {
//          cudaSetDevice(device);
//          mycublasCreate(&g_mycublas_handles_bwd[device]);
//       }
//    }
//    return g_mycublas_handles_bwd[device];
// }

// static mycublasHandle_t get_mycublas_handle_bwd2(int device = 0) {
//    if (g_mycublas_handles_bwd2[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
//       if (g_mycublas_handles_bwd2[device] == nullptr) {
//          cudaSetDevice(device);
//          mycublasCreate(&g_mycublas_handles_bwd2[device]);
//       }
//    }
//    return g_mycublas_handles_bwd2[device];
// }

// static cudaStream_t get_secondary_stream_bwd(int device = 0) {
//    if (g_secondary_streams_bwd[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
//       if (g_secondary_streams_bwd[device] == nullptr) {
//          cudaSetDevice(device);
//          // Non-blocking so it doesn't serialize with the default stream
//          cudaStreamCreateWithFlags(&g_secondary_streams_bwd[device], cudaStreamNonBlocking);
//       }
//    }
//    return g_secondary_streams_bwd[device];
// }

// static void init_bwd_events(int device) {
//    if (g_bwd_event_ready[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(mycublas_mutex_bwd);
//       if (g_bwd_event_ready[device] == nullptr) {
//          cudaSetDevice(device);
//          // DisableTiming avoids hardware timestamp recording, making record/wait ~10× cheaper
//          cudaEventCreateWithFlags(&g_bwd_event_ready[device], cudaEventDisableTiming);
//          cudaEventCreateWithFlags(&g_bwd_event_done[device],  cudaEventDisableTiming);
//       }
//    }
// }

// using namespace nvcuda;

// // ============================================================================
// // METADATA & CONSTANTS
// // ============================================================================

// struct BackwardMetadata {
//    int a_shape[8], b_shape[8], grad_out_shape[8];
//    int a_strides[8], b_strides[8], grad_out_strides[8];
//    int grad_a_strides[8], grad_b_strides[8];
//    int a_ndim, b_ndim, grad_out_ndim;
// };

// __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
//                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
//    offset = 0; 
//    if (out_ndim <= 2) return;
//    int temp_batch = batch_idx;
//    for (int dim = out_ndim - 3; dim >= 0; --dim) {
//       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
//       temp_batch /= b_dim_sz;
//       int c_dim = dim - (out_ndim - ndim);
//       if (c_dim >= 0 && c_dim < ndim - 2) 
//          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
//    }
// }

// constexpr int BACKWARD_PAD = 8;

// // ============================================================================
// // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_backward_dA_fp32(
//     const float* __restrict__ grad_out,  // [M, N]
//     const float* __restrict__ B,         // [K, N]
//     float* __restrict__ grad_A,          // [M, K]
//     float beta,
//     int M, int N, int K,
//     int total_batches,
//     BackwardMetadata meta) 
// {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
//    int go_offset = 0, b_offset = 0, ga_offset = 0;
//    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
//    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
//    const float *GO = grad_out + go_offset;
//    const float *Bp = B + b_offset;
//    float *GA = grad_A + ga_offset;
   
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
//    int s_b_k = meta.b_strides[meta.b_ndim-2];
//    int s_b_n = meta.b_strides[meta.b_ndim-1];
//    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
//    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

//    __shared__ float As[2][BK][BM + BACKWARD_PAD];
//    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
//    float results[16] = {0.0f}, regM[4], regN[4];

//    auto load_tiles = [&](int no, int idx) {
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BK, c = li % BK;
//          int gm = by * BM + r, gn = no + c;
//          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
//       }
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BN, c = li % BN;
//          int gn = no + r, gk = bx * BN + c;
//          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
//       }
//    };

//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int bk = 0; bk < N; bk += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (bk + BK < N) load_tiles(bk + BK, wi);
      
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) 
//             for (int j = 0; j < 4; j++) 
//                results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
   
//    for (int i = 0; i < 4; i++) {
//       for (int j = 0; j < 4; j++) {
//          int r = by * BM + tRow * 4 + i;
//          int c = bx * BN + tCol * 4 + j;
//          if (r < M && c < K) {
//             int idx = r * s_ga_m + c * s_ga_k;
//             GA[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GA[idx]);
//          }
//       }
//    }
// }

// // ============================================================================
// // FP32 MANUAL BACKWARD KERNEL (Fallback): grad_B = A^T @ grad_output
// // ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_backward_dB_fp32(
//     const float* __restrict__ A,         // [M, K]
//     const float* __restrict__ grad_out,  // [M, N]
//     float* __restrict__ grad_B,          // [K, N]
//     float beta,
//     int M, int K, int N,
//     int total_batches,
//     BackwardMetadata meta)
// {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
//    int a_offset = 0, go_offset = 0, gb_offset = 0;
//    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
//    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
//    const float *Ap = A + a_offset;
//    const float *GO = grad_out + go_offset;
//    float *GB = grad_B + gb_offset;
   
//    int s_a_m = meta.a_strides[meta.a_ndim-2];
//    int s_a_k = meta.a_strides[meta.a_ndim-1];
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
//    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
//    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

//    __shared__ float As[2][BK][BM + BACKWARD_PAD];
//    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];
//    float results[16] = {0.0f}, regM[4], regN[4];

//    auto load_tiles = [&](int mo, int idx) {
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BK, c = li % BK;
//          int gk = by * BM + r, gm = mo + c;
//          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
//       }
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BN, c = li % BN;
//          int gm = mo + r, gn = bx * BN + c;
//          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
//       }
//    };

//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int bk = 0; bk < M; bk += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (bk + BK < M) load_tiles(bk + BK, wi);
      
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) 
//             for (int j = 0; j < 4; j++) 
//                results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
   
//    for (int i = 0; i < 4; i++) {
//       for (int j = 0; j < 4; j++) {
//          int r = by * BM + tRow * 4 + i;
//          int c = bx * BN + tCol * 4 + j;
//          if (r < K && c < N) {
//             int idx = r * s_gb_k + c * s_gb_n;
//             GB[idx] = results[i*4+j] + (beta == 0.0f ? 0.0f : beta * GB[idx]);
//          }
//       }
//    }
// }

// // ============================================================================
// // FP16 MANUAL BACKWARD KERNEL (Fallback): grad_A = grad_output @ B^T
// // ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_backward_dA_fp16(
//     const __half* __restrict__ grad_out,
//     const __half* __restrict__ B,
//     __half* __restrict__ grad_A,
//     float beta,
//     int M, int N, int K,
//     int total_batches,
//     BackwardMetadata meta)
// {
//    const int batch_idx = blockIdx.z; 
//    if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
//    int go_offset = 0, b_offset = 0, ga_offset = 0;
//    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
//    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
//    const __half *GO = grad_out + go_offset;
//    const __half *Bp = B + b_offset;
//    __half *GA = grad_A + ga_offset;
   
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
//    int s_b_k = meta.b_strides[meta.b_ndim-2];
//    int s_b_n = meta.b_strides[meta.b_ndim-1];
//    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
//    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

//    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
//    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
//    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
//    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
//    #pragma unroll
//    for(int i = 0; i < 2; i++) 
//       for(int j = 0; j < 2; j++) 
//          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

//    auto load_tiles = [&](int no, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          int gm = blockIdx.y * BM + r, gn = no + c;
//          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          int gn = no + r, gk = blockIdx.x * BN + c;
//          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
//       }
//    };
   
//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int k = 0; k < N; k += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (k + BK < N) load_tiles(k + BK, wi);
      
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
//             #pragma unroll
//             for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
//                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
   
//    __half* sm = reinterpret_cast<__half*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) {
//       for (int j = 0; j < 2; j++) {
//          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
//          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
//          if (cr < M && cc < K) {
//             __half* wsm = sm + warp_id * 256;
//             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//             for (int r = 0; r < 16; r++) 
//                for (int c = 0; c < 16; c++)
//                   if (cr + r < M && cc + c < K) {
//                      int idx = (cr + r) * s_ga_m + (cc + c) * s_ga_k;
//                      GA[idx] = wsm[r * 16 + c] + (__half)beta * GA[idx];
//                   }
//          }
//       }
//    }
// }

// // ============================================================================
// // DISPATCH LAYER
// // ============================================================================

// template<typename T>
// void launch_backward_matmul(
//     const Tensor& grad_output,
//     const Tensor& A,
//     const Tensor& B,
//     Tensor& grad_A,
//     Tensor& grad_B,
//     float beta_a,
//     float beta_b,
//     cudaStream_t stream)
// {
//    const auto& go_sh = grad_output.shape().dims;
//    const auto& a_sh = A.shape().dims;
//    const auto& b_sh = B.shape().dims;
   
//    int go_ndim = go_sh.size();
//    int a_ndim = a_sh.size();
//    int b_ndim = b_sh.size();
   
//    int M = go_sh[go_ndim - 2];
//    int N = go_sh[go_ndim - 1];
//    int K_a = a_sh[a_ndim - 1];
   
//    int tb = 1;
//    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
//    BackwardMetadata meta;
//    meta.a_ndim = a_ndim;
//    meta.b_ndim = b_ndim;
//    meta.grad_out_ndim = go_ndim;
   
//    for (int i = 0; i < a_ndim; i++) {
//       meta.a_shape[i] = a_sh[i];
//       meta.a_strides[i] = A.stride().strides[i];
//       meta.grad_a_strides[i] = grad_A.stride().strides[i];
//    }
//    for (int i = 0; i < b_ndim; i++) {
//       meta.b_shape[i] = b_sh[i];
//       meta.b_strides[i] = B.stride().strides[i];
//       meta.grad_b_strides[i] = grad_B.stride().strides[i];
//    }
//    for (int i = 0; i < go_ndim; i++) {
//       meta.grad_out_shape[i] = go_sh[i];
//       meta.grad_out_strides[i] = grad_output.stride().strides[i];
//    }
   
//    const T* go_ptr = grad_output.data<T>();
//    const T* a_ptr = A.data<T>();
//    const T* b_ptr = B.data<T>();
//    T* ga_ptr = grad_A.data<T>();
//    T* gb_ptr = grad_B.data<T>();
   
//    // Accept both row-major (strides[-1]==1) and col-major/transposed (strides[-2]==1) for A and B.
//    // grad_out and grad outputs must always be row-major (strides[-1]==1) because we write to them.
//    bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
//    bool a_row_major   = (meta.a_strides[a_ndim-1] == 1);
//    bool a_col_major   = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
//    bool b_row_major   = (meta.b_strides[b_ndim-1] == 1);
//    bool b_col_major   = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
//    bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
//    bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

//    bool supported_for_fast_path = go_contiguous && (a_row_major || a_col_major) &&
//                                   (b_row_major || b_col_major) && ga_contiguous && gb_contiguous;

//    // For grad_A = grad_out @ B^T:
//    //   B row-major (strides[-1]==1): stored normally, need MYCUBLAS_OP_T to transpose, ld = strides[-2]
//    //   B col-major (strides[-2]==1): already stored as B^T, use MYCUBLAS_OP_N, ld = strides[-1]
//    mycublasOperation_t b_op_ga = b_row_major ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;
//    int ldb_ga = b_row_major ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];

//    // For grad_B = A^T @ grad_out:
//    //   A row-major (strides[-1]==1): stored normally, need MYCUBLAS_OP_T to transpose, ld = strides[-2]
//    //   A col-major (strides[-2]==1): already stored as A^T, use MYCUBLAS_OP_N, ld = strides[-1]
//    mycublasOperation_t a_op_gb = a_row_major ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;
//    int lda_gb = a_row_major ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];

//     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");

//    // ---- Acquire device, handles, and secondary stream once ----
//    // grad_A = grad_output @ B^T  and  grad_B = A^T @ grad_output are fully independent.
//    // Dispatch them on separate cuBLAS handles bound to separate streams so the GPU
//    // can overlap them.  Events are cached per-device and use DisableTiming for
//    // minimal overhead.
//    int dev = 0;
//    cudaGetDevice(&dev);
//    mycublasHandle_t handle_a = get_mycublas_handle_bwd(dev);
//    mycublasHandle_t handle_b = get_mycublas_handle_bwd2(dev);
//    cudaStream_t stream_b     = get_secondary_stream_bwd(dev);
//    init_bwd_events(dev);

//    mycublasSetStream(handle_a, stream);
//    mycublasSetStream(handle_b, stream_b);

//    // stream_b must wait until all upstream data (grad_output, A, B) produced on
//    // the primary stream is visible before it starts reading them.
//    cudaEventRecord(g_bwd_event_ready[dev], stream);
//    cudaStreamWaitEvent(stream_b, g_bwd_event_ready[dev], 0);

//    // ---- grad_A = grad_output @ B^T  (primary stream) ----
//    bool grad_a_done = false;
//    if constexpr (std::is_same<T, float>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldga = meta.grad_a_strides[a_ndim-2];
//           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
//           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
//           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;

//           mycublasSgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, 1.0f, go_ptr, ldgo, s_go, b_ptr, ldb_ga, s_b, beta_a, ga_ptr, ldga, s_ga, tb);
//           grad_a_done = true;
//        }
//    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldga = meta.grad_a_strides[a_ndim-2];
//           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
//           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
//           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
//           const __half alpha = __float2half(1.0f), h_beta_a = __float2half(beta_a);
//           mycublasHgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, alpha, (const __half*)go_ptr, ldgo, s_go, (const __half*)b_ptr, ldb_ga, s_b, h_beta_a, (__half*)ga_ptr, ldga, s_ga, tb);
//           grad_a_done = true;
//        }
//    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldga = meta.grad_a_strides[a_ndim-2];
//           long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
//           long long s_b  = (b_ndim  >= 3) ? meta.b_strides[b_ndim-3]        : 0;
//           long long s_ga = (a_ndim  >= 3) ? meta.grad_a_strides[a_ndim-3]   : 0;
//           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), b_beta_a = __float2bfloat16(beta_a);
//           mycublasBgemmStridedBatched(handle_a, MYCUBLAS_OP_N, b_op_ga, M, K_a, N, alpha, (const __nv_bfloat16*)go_ptr, ldgo, s_go, (const __nv_bfloat16*)b_ptr, ldb_ga, s_b, b_beta_a, (__nv_bfloat16*)ga_ptr, ldga, s_ga, tb);
//           grad_a_done = true;
//        }
//    }

//    if (!grad_a_done) {
//        if constexpr (std::is_same<T, float>::value)
//           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, beta_a, M, N, K_a, tb, meta);
//        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
//           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), beta_a, M, N, K_a, tb, meta);
//    }

//    // ---- grad_B = A^T @ grad_output  (secondary stream — runs in parallel with grad_A above) ----
//    bool grad_b_done = false;
//    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[b_ndim-3] == 1 && meta.grad_out_shape[go_ndim-3] > 1);

//    if constexpr (std::is_same<T, float>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldgb = meta.grad_b_strides[b_ndim-2];
//           if (need_reduction) {
//               int M_total = M * tb;
//               mycublasSgemm(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, 1.0f, a_ptr, lda_gb, go_ptr, ldgo, beta_b, gb_ptr, ldgb);
//           } else {
//              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]            : 0;
//              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3]    : 0;
//              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]       : 0;

//              mycublasSgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, 1.0f, a_ptr, lda_gb, s_a, go_ptr, ldgo, s_go, beta_b, gb_ptr, ldgb, s_gb, tb);
//           }
//           grad_b_done = true;
//        }
//    } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, float16_t>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldgb = meta.grad_b_strides[b_ndim-2];
//           const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
//           if (need_reduction) {
//               int M_total = M * tb;
//               const __half h_beta_b = __float2half(beta_b);
//               mycublasHgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __half*)a_ptr, lda_gb, 0, (const __half*)go_ptr, ldgo, 0, h_beta_b, (__half*)gb_ptr, ldgb, 0, 1);
//           } else {
//              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
//              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
//              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
//              mycublasHgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __half*)a_ptr, lda_gb, s_a, (const __half*)go_ptr, ldgo, s_go, beta, (__half*)gb_ptr, ldgb, s_gb, tb);
//           }
//           grad_b_done = true;
//        }
//    } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, bfloat16_t>::value) {
//        if (supported_for_fast_path) {
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldgb = meta.grad_b_strides[b_ndim-2];
//           const __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(0.0f);
//           if (need_reduction) {
//               int M_total = M * tb;
//               const __nv_bfloat16 b_beta_b = __float2bfloat16(beta_b);
//               mycublasBgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M_total, alpha, (const __nv_bfloat16*)a_ptr, lda_gb, 0, (const __nv_bfloat16*)go_ptr, ldgo, 0, b_beta_b, (__nv_bfloat16*)gb_ptr, ldgb, 0, 1);
//           } else {
//              long long s_a  = (a_ndim  >= 3) ? meta.a_strides[a_ndim-3]         : 0;
//              long long s_go = (go_ndim >= 3) ? meta.grad_out_strides[go_ndim-3] : 0;
//              long long s_gb = (b_ndim  >= 3) ? meta.grad_b_strides[b_ndim-3]    : 0;
//              mycublasBgemmStridedBatched(handle_b, a_op_gb, MYCUBLAS_OP_N, K_a, N, M, alpha, (const __nv_bfloat16*)a_ptr, lda_gb, s_a, (const __nv_bfloat16*)go_ptr, ldgo, s_go, beta, (__nv_bfloat16*)gb_ptr, ldgb, s_gb, tb);
//           }
//           grad_b_done = true;
//        }
//    }

//    if (!grad_b_done) {
//        if constexpr (std::is_same<T, float>::value)
//           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream_b>>>(a_ptr, go_ptr, gb_ptr, beta_b, M, K_a, N, tb, meta);
//    }

//    // Rejoin stream_b into the primary stream so downstream ops on stream see grad_B.
//    cudaEventRecord(g_bwd_event_done[dev], stream_b);
//    cudaStreamWaitEvent(stream, g_bwd_event_done[dev], 0);

//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
//    }
// }

// void cuda_matmul_backward(
//     const Tensor& grad_output,
//     const Tensor& A,
//     const Tensor& B,
//     Tensor& grad_A,
//     Tensor& grad_B,
//     cudaStream_t stream)
// {
//    float beta_a = 0.0f;
//    float beta_b = 0.0f;
   
//    dispatch_by_dtype(A.dtype(), [&](auto d) {
//       using T = decltype(d);
//       launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, beta_a, beta_b, stream);
//    });
// }

// } // namespace OwnTensor

// #else // WITH_MYBLAS


// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <mma.h>
// #include <cublas_v2.h>
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
// #include <string>
// #include <mutex>

// #include "ops/MatmulBackward.cuh"
// #include "core/Tensor.h"
// #include "core/TensorDispatch.h"
// #include "utils/Profiler.h"

// namespace OwnTensor {

// // cuBLAS handle management - thread-safe singleton per device
// static std::mutex cublas_mutex_bwd;
// static cublasHandle_t g_cublas_handles_bwd[8]  = {nullptr};
// static cublasHandle_t g_cublas_handles_bwd2[8] = {nullptr};

// static cublasHandle_t get_cublas_handle_bwd(int device = 0) {
//    if (g_cublas_handles_bwd[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
//       if (g_cublas_handles_bwd[device] == nullptr) {
//          cudaSetDevice(device);
//          cublasCreate(&g_cublas_handles_bwd[device]);
//          cublasSetMathMode(g_cublas_handles_bwd[device], CUBLAS_TF32_TENSOR_OP_MATH);
//       }
//    }
//    return g_cublas_handles_bwd[device];
// }

// // Second handle for grad_B on the secondary stream (mirrors mycublas dual-stream design)
// static cublasHandle_t get_cublas_handle_bwd2(int device = 0) {
//    if (g_cublas_handles_bwd2[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
//       if (g_cublas_handles_bwd2[device] == nullptr) {
//          cudaSetDevice(device);
//          cublasCreate(&g_cublas_handles_bwd2[device]);
//          cublasSetMathMode(g_cublas_handles_bwd2[device], CUBLAS_TF32_TENSOR_OP_MATH);
//       }
//    }
//    return g_cublas_handles_bwd2[device];
// }

// // Secondary stream: NonBlocking so it doesn't serialize with stream 0
// static cudaStream_t g_cublas_secondary_streams[8] = {nullptr};
// static cudaStream_t get_cublas_secondary_stream(int device = 0) {
//    if (g_cublas_secondary_streams[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
//       if (g_cublas_secondary_streams[device] == nullptr) {
//          cudaSetDevice(device);
//          cudaStreamCreateWithFlags(&g_cublas_secondary_streams[device], cudaStreamNonBlocking);
//       }
//    }
//    return g_cublas_secondary_streams[device];
// }

// // DisableTiming events are ~10x cheaper to record/wait than default events
// static cudaEvent_t g_cublas_evt_ready[8] = {nullptr};
// static cudaEvent_t g_cublas_evt_done[8]  = {nullptr};
// static void init_cublas_bwd_events(int device) {
//    if (g_cublas_evt_ready[device] == nullptr) {
//       std::lock_guard<std::mutex> lock(cublas_mutex_bwd);
//       if (g_cublas_evt_ready[device] == nullptr) {
//          cudaSetDevice(device);
//          cudaEventCreateWithFlags(&g_cublas_evt_ready[device], cudaEventDisableTiming);
//          cudaEventCreateWithFlags(&g_cublas_evt_done[device],  cudaEventDisableTiming);
//       }
//    }
// }

// using namespace nvcuda;

// // ============================================================================
// // METADATA & CONSTANTS (Reuse from forward)
// // ============================================================================

// struct BackwardMetadata {
//    int a_shape[8], b_shape[8], grad_out_shape[8];
//    int a_strides[8], b_strides[8], grad_out_strides[8];
//    int grad_a_strides[8], grad_b_strides[8];
//    int a_ndim, b_ndim, grad_out_ndim;
// };

// __device__ void compute_batch_offset_bwd(int batch_idx, const int* shape, const int* strides, 
//                                           int ndim, const int* out_shape, int out_ndim, int& offset) {
//    offset = 0; 
//    if (out_ndim <= 2) return;
//    int temp_batch = batch_idx;
//    for (int dim = out_ndim - 3; dim >= 0; --dim) {
//       int b_dim_sz = out_shape[dim], b_coord = temp_batch % b_dim_sz;
//       temp_batch /= b_dim_sz;
//       int c_dim = dim - (out_ndim - ndim);
//       if (c_dim >= 0 && c_dim < ndim - 2) 
//          offset += (int64_t)((shape[c_dim] > 1) ? b_coord : 0) * strides[c_dim];
//    }
// }

// constexpr int BACKWARD_PAD = 8;

// // ============================================================================
// // FP32 BACKWARD KERNEL: grad_A = grad_output @ B^T
// // Computes: C[M,K] = A[M,N] @ B[K,N]^T = A[M,N] @ B^T[N,K]
// // Note: We read B in transposed order (columns become rows)
// // BM=128, BN=128, BK=16, TM=4, TN=4, Threads=1024
// // ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_backward_dA_fp32(
//     const float* __restrict__ grad_out,  // [M, N] - grad of output
//     const float* __restrict__ B,         // [K, N] - second input of forward, read transposed
//     float* __restrict__ grad_A,          // [M, K] - gradient for A
//     int M, int N, int K,                 // M=rows of grad_out, N=cols of grad_out (=cols of B), K=rows of B
//     int total_batches,
//     BackwardMetadata meta) 
// {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
//    int go_offset = 0, b_offset = 0, ga_offset = 0;
//    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.b_strides, meta.b_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
//    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
//    const float *GO = grad_out + go_offset;
//    const float *Bp = B + b_offset;
//    float *GA = grad_A + ga_offset;
   
//    // Strides for grad_output [M, N]
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
//    // Strides for B [K, N] - we read it transposed as B^T [N, K]
//    int s_b_k = meta.b_strides[meta.b_ndim-2];
//    int s_b_n = meta.b_strides[meta.b_ndim-1];
   
//    // Strides for grad_A [M, K]
//    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
//    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

//    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // grad_out tile: [BK, BM] transposed storage
//    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // B^T tile: [BK, BN]
//    float results[16] = {0.0f}, regM[4], regN[4];

//    // Load tiles with double buffering
//    // For grad_A = grad_out @ B^T:
//    //   We iterate over N (the shared dimension between grad_out and B^T)
//    //   grad_out is [M, N], B^T is [N, K]
//    //   So we load BK columns of grad_out into shared As
//    //   And BK rows of B^T (= BK columns of B) into shared Bs
   
//    auto load_tiles = [&](int no, int idx) {
//       // Load grad_out[by*BM : by*BM+BM, no : no+BK] into As (transposed for coalescing)
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BK, c = li % BK;
//          int gm = by * BM + r, gn = no + c;
//          As[idx][c][r] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
//       }
      
//       // Load B^T[no : no+BK, bx*BN : bx*BN+BN] = B[bx*BN : bx*BN+BN, no : no+BK]^T
//       // B is [K, N], B^T is [N, K]. We want B^T[n, k] = B[k, n]
//       // So for k in [bx*BN, bx*BN+BN) and n in [no, no+BK)
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BN, c = li % BN;
//          int gn = no + r, gk = bx * BN + c;
//          // B^T[gn, gk] = B[gk, gn]
//          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : 0.0f;
//       }
//    };

//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int bk = 0; bk < N; bk += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (bk + BK < N) load_tiles(bk + BK, wi);
      
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) 
//             for (int j = 0; j < 4; j++) 
//                results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
   
//    // Store results to grad_A
//    for (int i = 0; i < 4; i++) {
//       for (int j = 0; j < 4; j++) {
//          int r = by * BM + tRow * 4 + i;
//          int c = bx * BN + tCol * 4 + j;
//          if (r < M && c < K) {
//             GA[r * s_ga_m + c * s_ga_k] = results[i*4+j];
//          }
//       }
//    }
// }

// // ============================================================================
// // FP32 BACKWARD KERNEL: grad_B = A^T @ grad_output
// // Computes: C[K,N] = A[M,K]^T @ B[M,N] = A^T[K,M] @ B[M,N]
// // Note: We read A in transposed order
// // ============================================================================

// template<int BM, int BN, int BK, int TM, int TN>
// __global__ void matmul_backward_dB_fp32(
//     const float* __restrict__ A,         // [M, K] - first input of forward, read transposed
//     const float* __restrict__ grad_out,  // [M, N] - grad of output
//     float* __restrict__ grad_B,          // [K, N] - gradient for B
//     int M, int K, int N,                 // M=rows of A, K=cols of A, N=cols of grad_out
//     int total_batches,
//     BackwardMetadata meta)
// {
//    const int bx = blockIdx.x, by = blockIdx.y, b_idx = blockIdx.z;
//    if (b_idx >= total_batches) return;
//    const int tid = threadIdx.x, tCol = tid % 32, tRow = tid / 32;
   
//    int a_offset = 0, go_offset = 0, gb_offset = 0;
//    compute_batch_offset_bwd(b_idx, meta.a_shape, meta.a_strides, meta.a_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, a_offset);
//    compute_batch_offset_bwd(b_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(b_idx, meta.b_shape, meta.grad_b_strides, meta.b_ndim, 
//                             meta.grad_out_shape, meta.grad_out_ndim, gb_offset);
   
//    const float *Ap = A + a_offset;
//    const float *GO = grad_out + go_offset;
//    float *GB = grad_B + gb_offset;
   
//    // Strides for A [M, K] - we read it transposed as A^T [K, M]
//    int s_a_m = meta.a_strides[meta.a_ndim-2];
//    int s_a_k = meta.a_strides[meta.a_ndim-1];
   
//    // Strides for grad_output [M, N]
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
   
//    // Strides for grad_B [K, N]
//    int s_gb_k = meta.grad_b_strides[meta.b_ndim-2];
//    int s_gb_n = meta.grad_b_strides[meta.b_ndim-1];

//    __shared__ float As[2][BK][BM + BACKWARD_PAD];  // A^T tile: [BK, BM]
//    __shared__ float Bs[2][BK][BN + BACKWARD_PAD];  // grad_out tile: [BK, BN]
//    float results[16] = {0.0f}, regM[4], regN[4];

//    // For grad_B = A^T @ grad_out:
//    //   We iterate over M (the shared dimension)
//    //   A^T is [K, M], grad_out is [M, N]
//    //   So we load BK rows of A^T (= BK columns of A) into shared As
//    //   And BK rows of grad_out into shared Bs
   
//    auto load_tiles = [&](int mo, int idx) {
//       // Load A^T[by*BM : by*BM+BM, mo : mo+BK] = A[mo : mo+BK, by*BM : by*BM+BM]^T
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BK, c = li % BK;
//          int gk = by * BM + r, gm = mo + c;
//          // A^T[gk, gm] = A[gm, gk]
//          As[idx][c][r] = (gk < K && gm < M) ? Ap[gm * s_a_m + gk * s_a_k] : 0.0f;
//       }
      
//       // Load grad_out[mo : mo+BK, bx*BN : bx*BN+BN] into Bs
//       #pragma unroll
//       for (int i = 0; i < 2; i++) {
//          int li = tid + i * 1024, r = li / BN, c = li % BN;
//          int gm = mo + r, gn = bx * BN + c;
//          Bs[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : 0.0f;
//       }
//    };

//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int bk = 0; bk < M; bk += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (bk + BK < M) load_tiles(bk + BK, wi);
      
//       #pragma unroll
//       for (int d = 0; d < BK; d++) {
//          #pragma unroll
//          for (int i = 0; i < 4; i++) regM[i] = As[ri][d][tRow*4 + i];
//          #pragma unroll
//          for (int j = 0; j < 4; j++) regN[j] = Bs[ri][d][tCol*4 + j];
//          #pragma unroll
//          for (int i = 0; i < 4; i++) 
//             for (int j = 0; j < 4; j++) 
//                results[i*4+j] += regM[i] * regN[j];
//       }
//       __syncthreads();
//    }
   
//    // Store results to grad_B
//    for (int i = 0; i < 4; i++) {
//       for (int j = 0; j < 4; j++) {
//          int r = by * BM + tRow * 4 + i;
//          int c = bx * BN + tCol * 4 + j;
//          if (r < K && c < N) {
//             GB[r * s_gb_k + c * s_gb_n] = results[i*4+j];
//          }
//       }
//    }
// }

// // ============================================================================
// // FP16 BACKWARD KERNEL: grad_A = grad_output @ B^T (WMMA)
// // ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_backward_dA_fp16(
//     const __half* __restrict__ grad_out,
//     const __half* __restrict__ B,
//     __half* __restrict__ grad_A,
//     int M, int N, int K,
//     int total_batches,
//     BackwardMetadata meta)
// {
//    const int batch_idx = blockIdx.z; 
//    if (batch_idx >= total_batches) return;
//    const int tid = threadIdx.x, warp_id = tid / 32, warp_row = warp_id / 4, warp_col = warp_id % 4;
   
//    int go_offset = 0, b_offset = 0, ga_offset = 0;
//    compute_batch_offset_bwd(batch_idx, meta.grad_out_shape, meta.grad_out_strides, meta.grad_out_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, go_offset);
//    compute_batch_offset_bwd(batch_idx, meta.b_shape, meta.b_strides, meta.b_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, b_offset);
//    compute_batch_offset_bwd(batch_idx, meta.a_shape, meta.grad_a_strides, meta.a_ndim,
//                             meta.grad_out_shape, meta.grad_out_ndim, ga_offset);
   
//    const __half *GO = grad_out + go_offset;
//    const __half *Bp = B + b_offset;
//    __half *GA = grad_A + ga_offset;
   
//    int s_go_m = meta.grad_out_strides[meta.grad_out_ndim-2];
//    int s_go_n = meta.grad_out_strides[meta.grad_out_ndim-1];
//    int s_b_k = meta.b_strides[meta.b_ndim-2];
//    int s_b_n = meta.b_strides[meta.b_ndim-1];
//    int s_ga_m = meta.grad_a_strides[meta.a_ndim-2];
//    int s_ga_k = meta.grad_a_strides[meta.a_ndim-1];

//    __shared__ __half As[2][BM][BK + BACKWARD_PAD];
//    __shared__ __half Bs[2][BK][BN + BACKWARD_PAD];
   
//    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
//    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
//    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];
   
//    #pragma unroll
//    for(int i = 0; i < 2; i++) 
//       for(int j = 0; j < 2; j++) 
//          wmma::fill_fragment(acc[i][j], __float2half(0.0f));

//    auto load_tiles = [&](int no, int idx) {
//       for (int i = tid; i < BM * BK; i += 512) {
//          int r = i / BK, c = i % BK;
//          int gm = blockIdx.y * BM + r, gn = no + c;
//          As[idx][r][c] = (gm < M && gn < N) ? GO[gm * s_go_m + gn * s_go_n] : __float2half(0.0f);
//       }
//       for (int i = tid; i < BK * BN; i += 512) {
//          int r = i / BN, c = i % BN;
//          int gn = no + r, gk = blockIdx.x * BN + c;
//          Bs[idx][r][c] = (gn < N && gk < K) ? Bp[gk * s_b_k + gn * s_b_n] : __float2half(0.0f);
//       }
//    };
   
//    int wi = 0; 
//    load_tiles(0, wi); 
//    __syncthreads();
   
//    for (int k = 0; k < N; k += BK) {
//       int ri = wi; 
//       wi = 1 - wi;
//       if (k + BK < N) load_tiles(k + BK, wi);
      
//       #pragma unroll
//       for (int ks = 0; ks < BK; ks += 16) {
//          #pragma unroll
//          for (int i = 0; i < 2; i++) {
//             wmma::load_matrix_sync(af, &As[ri][warp_row*WM + i*16][ks], BK + BACKWARD_PAD);
//             #pragma unroll
//             for (int j = 0; j < 2; j++) {
//                wmma::load_matrix_sync(bf, &Bs[ri][ks][warp_col*WN + j*16], BN + BACKWARD_PAD);
//                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
//             }
//          }
//       }
//       __syncthreads();
//    }
   
//    __half* sm = reinterpret_cast<__half*>(As);
//    #pragma unroll
//    for (int i = 0; i < 2; i++) {
//       for (int j = 0; j < 2; j++) {
//          int cr = blockIdx.y * BM + warp_row * WM + i * 16;
//          int cc = blockIdx.x * BN + warp_col * WN + j * 16;
//          if (cr < M && cc < K) {
//             __half* wsm = sm + warp_id * 256;
//             wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);
//             for (int r = 0; r < 16; r++) 
//                for (int c = 0; c < 16; c++)
//                   if (cr + r < M && cc + c < K) 
//                      GA[(cr + r) * s_ga_m + (cc + c) * s_ga_k] = wsm[r * 16 + c];
//          }
//       }
//    }
// }

// // ============================================================================
// // DISPATCH LAYER
// // ============================================================================

// template<typename T>
// void launch_backward_matmul(
//     const Tensor& grad_output,
//     const Tensor& A,
//     const Tensor& B,
//     Tensor& grad_A,
//     Tensor& grad_B,
//     cudaStream_t stream)
// {
//    const auto& go_sh = grad_output.shape().dims;
//    const auto& a_sh = A.shape().dims;
//    const auto& b_sh = B.shape().dims;
   
//    int go_ndim = go_sh.size();
//    int a_ndim = a_sh.size();
//    int b_ndim = b_sh.size();
   
//    int M = go_sh[go_ndim - 2];  // rows of grad_output
//    int N = go_sh[go_ndim - 1];  // cols of grad_output (= cols of B)
//    int K_a = a_sh[a_ndim - 1];  // cols of A (= rows of grad_A's cols dim)
//    int K_b = b_sh[b_ndim - 2];  // rows of B
   
//    int tb = 1;
//    for (int i = 0; i < go_ndim - 2; i++) tb *= go_sh[i];
   
//    BackwardMetadata meta;
//    meta.a_ndim = a_ndim;
//    meta.b_ndim = b_ndim;
//    meta.grad_out_ndim = go_ndim;
   
//    for (int i = 0; i < a_ndim; i++) {
//       meta.a_shape[i] = a_sh[i];
//       meta.a_strides[i] = A.stride().strides[i];
//       meta.grad_a_strides[i] = grad_A.stride().strides[i];
//    }
//    for (int i = 0; i < b_ndim; i++) {
//       meta.b_shape[i] = b_sh[i];
//       meta.b_strides[i] = B.stride().strides[i];
//       meta.grad_b_strides[i] = grad_B.stride().strides[i];
//    }
//    for (int i = 0; i < go_ndim; i++) {
//       meta.grad_out_shape[i] = go_sh[i];
//       meta.grad_out_strides[i] = grad_output.stride().strides[i];
//    }
   
//    const T* go_ptr = grad_output.data<T>();
//    const T* a_ptr = A.data<T>();
//    const T* b_ptr = B.data<T>();
//    T* ga_ptr = grad_A.data<T>();
//    T* gb_ptr = grad_B.data<T>();
   
//    // Check stride attributes
//    bool go_contiguous = (meta.grad_out_strides[go_ndim-1] == 1);
//    bool a_contiguous = (meta.a_strides[a_ndim-1] == 1);
//    bool a_transposed = (a_ndim >= 2 && meta.a_strides[a_ndim-2] == 1);
//    bool b_contiguous = (meta.b_strides[b_ndim-1] == 1);
//    bool b_transposed = (b_ndim >= 2 && meta.b_strides[b_ndim-2] == 1);
//    bool ga_contiguous = (meta.grad_a_strides[a_ndim-1] == 1);
//    bool gb_contiguous = (meta.grad_b_strides[b_ndim-1] == 1);

//    bool supported_for_fast_path = go_contiguous && (a_contiguous || a_transposed) && (b_contiguous || b_transposed) && ga_contiguous && gb_contiguous;

//    // Dual-stream setup: grad_A on primary stream, grad_B on secondary stream (concurrent).
//    // stream_b is NonBlocking so it never implicitly serializes with stream 0.
//    const int dev = grad_output.device().index;
//    cudaStream_t stream_b = get_cublas_secondary_stream(dev);
//    init_cublas_bwd_events(dev);
//    // signal that input tensors on the primary stream are ready for stream_b to read
//    cudaEventRecord(g_cublas_evt_ready[dev], stream);
//    cudaStreamWaitEvent(stream_b, g_cublas_evt_ready[dev], 0);

//     AUTO_PROFILE_CUDA("Backward::Matmul_CUDA");
//     bool grad_a_done = false;
//     if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
//                   std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
//         // cuBLAS path for FP16/BF16: grad_A = grad_output @ B^T
//         // compute in fp16/bf16, accumulate in fp32
//         if (supported_for_fast_path) {
//             int dev_idx = grad_output.device().index;
//             cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
//             cublasSetStream(handle, stream);
//             float alpha_f = 1.0f, beta_f = 0.0f;

//             cublasOperation_t opB_bwd = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
//             int ldb_bwd = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
//             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
//             int ldga_bwd = meta.grad_a_strides[a_ndim-2];

//             long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3] : 0LL;
//             long long stride_b_ll  = (b_ndim  >= 3) ? (long long)meta.b_strides[b_ndim-3]         : 0LL;
//             long long stride_ga_ll = (a_ndim  >= 3) ? (long long)meta.grad_a_strides[a_ndim-3]    : 0LL;
//             if (b_ndim < go_ndim) stride_b_ll = 0LL;

//             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
//                                        ? CUDA_R_16F : CUDA_R_16BF;
//             cublasStatus_t status;
//             if (tb == 1) {
//                 status = cublasGemmEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
//                     b_ptr,  cuda_type, ldb_bwd,
//                     go_ptr, cuda_type, ldgo_bwd,
//                     &beta_f, ga_ptr, cuda_type, ldga_bwd,
//                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             } else {
//                 status = cublasGemmStridedBatchedEx(handle, opB_bwd, CUBLAS_OP_N, K_a, M, N, &alpha_f,
//                     b_ptr,  cuda_type, ldb_bwd,  stride_b_ll,
//                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
//                     &beta_f, ga_ptr, cuda_type, ldga_bwd, stride_ga_ll, tb,
//                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             }
//             if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
//         }
//     } else if constexpr (std::is_same<T, float>::value) {
//         if (supported_for_fast_path) {
//           // USE EXPLICIT DEVICE INDEXING from grad_output
//           int dev_idx = grad_output.device().index;
//           cublasHandle_t handle = get_cublas_handle_bwd(dev_idx);
//           cublasSetStream(handle, stream);
//           float alpha = 1.0f, beta = 0.0f;
          
//           cublasOperation_t opB = b_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
//           int ldb = b_contiguous ? meta.b_strides[b_ndim-2] : meta.b_strides[b_ndim-1];
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldga = meta.grad_a_strides[a_ndim-2];
          
//           // Compute batch strides from actual tensor metadata
//           long long stride_go, stride_b_val, stride_ga;
//           bool batch_uniform_dA = true;

//           if (go_ndim <= 3) {
//               stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
//               stride_b_val = (b_ndim == 3) ? meta.b_strides[0] : 0;
//               stride_ga = (go_ndim == 3) ? meta.grad_a_strides[0] : 0;
//               if (b_ndim < go_ndim) stride_b_val = 0;
//           } else {
//               // 4D: use inner batch stride (strides[ndim-3])
//               stride_go = meta.grad_out_strides[go_ndim - 3];
//               stride_b_val = (b_ndim >= 4) ? meta.b_strides[b_ndim - 3] : 0;
//               stride_ga = meta.grad_a_strides[a_ndim - 3];
//               if (b_ndim < go_ndim) stride_b_val = 0;

//               // Check uniform batch strides: stride[0] == stride[1] * shape[1]
//               batch_uniform_dA = (meta.grad_out_strides[0] == stride_go * go_sh[1]);
//               if (b_ndim >= 4) batch_uniform_dA = batch_uniform_dA && (meta.b_strides[0] == stride_b_val * b_sh[1]);
//               batch_uniform_dA = batch_uniform_dA && (meta.grad_a_strides[0] == stride_ga * a_sh[1]);
//           }

//           cublasStatus_t status;
//           if (tb == 1) {
//               status = cublasGemmEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, go_ptr, CUDA_R_32F, ldgo, &beta, ga_ptr, CUDA_R_32F, ldga, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//           } else if (batch_uniform_dA) {
//               status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_ptr, CUDA_R_32F, ldb, stride_b_val, go_ptr, CUDA_R_32F, ldgo, stride_go, &beta, ga_ptr, CUDA_R_32F, ldga, stride_ga, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//           } else {
//               // Non-uniform 4D: loop over outer batch dim
//               int d0 = go_sh[0], d1 = go_sh[1];
//               status = CUBLAS_STATUS_SUCCESS;
//               for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
//                   const T* b_batch = b_ptr + (b_ndim >= 4 ? bi * meta.b_strides[0] : 0);
//                   const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
//                   T* ga_batch = ga_ptr + bi * meta.grad_a_strides[0];
//                   status = cublasGemmStridedBatchedEx(handle, opB, CUBLAS_OP_N, K_a, M, N, &alpha, b_batch, CUDA_R_32F, ldb, stride_b_val, go_batch, CUDA_R_32F, ldgo, stride_go, &beta, ga_batch, CUDA_R_32F, ldga, stride_ga, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//               }
//           }

//           if (status == CUBLAS_STATUS_SUCCESS) grad_a_done = true;
//        }
//    }
   
//    if (!grad_a_done) {
//         #ifdef AUTOGRAD_PROFILER_ENABLED
//         if (M*N*K_a > 1000000) {
//             printf("dA Fallback: M=%d N=%d K=%d, Strides: go[%d,%d], b[%d,%d], ga[%d,%d]\n", 
//                    M, N, K_a, meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
//                    meta.b_strides[b_ndim-2], meta.b_strides[b_ndim-1],
//                    meta.grad_a_strides[a_ndim-2], meta.grad_a_strides[a_ndim-1]);
//         }
//         #endif
//        // Fallback
//        if constexpr (std::is_same<T, float>::value)
//           matmul_backward_dA_fp32<128, 128, 16, 4, 4><<<dim3((K_a+127)/128, (M+127)/128, tb), 1024, 0, stream>>>(go_ptr, b_ptr, ga_ptr, M, N, K_a, tb, meta);
//        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value)
//           matmul_backward_dA_fp16<128, 128, 32, 32, 32><<<dim3((K_a+127)/128, (M+127)/128, tb), 512, 0, stream>>>(reinterpret_cast<const __half*>(go_ptr), reinterpret_cast<const __half*>(b_ptr), reinterpret_cast<__half*>(ga_ptr), M, N, K_a, tb, meta);
//        // bf16 fallback: cuBLAS path above handles all standard layouts; non-standard bf16 not supported
//    }

//    // Launch grad_B kernel: grad_B = A^T @ grad_output
//    // Handle Reduction if B dim < Output dim (Broadcast in forward)
   
//    bool grad_b_actions_needed = true; // Could optimize if not req grad
//    bool grad_b_done = false;
   
//    // Check reduction need
//    bool need_reduction = (b_ndim < go_ndim) || (tb > 1 && b_ndim == go_ndim && meta.b_shape[0] == 1);
//    // Standard broadcast check: if tb > 1 but grad_B is not batched.
//    // Assuming grad_B matches B shape.
//    if (tb > 1 && b_ndim < go_ndim) need_reduction = true;

//    if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
//                  std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
//         // cuBLAS path for FP16/BF16: grad_B = A^T @ grad_output  (secondary stream)
//         // compute in fp16/bf16, accumulate in fp32
//         if (supported_for_fast_path) {
//             int dev_idx = grad_output.device().index;
//             cublasHandle_t handle = get_cublas_handle_bwd2(dev_idx);
//             cublasSetStream(handle, stream_b);
//             float alpha_f = 1.0f, beta_f = 0.0f;

//             cublasOperation_t opA_bwd = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
//             int lda_bwd  = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1];
//             int ldgo_bwd = meta.grad_out_strides[go_ndim-2];
//             int ldgb_bwd = meta.grad_b_strides[b_ndim-2];

//             cudaDataType_t cuda_type = std::is_same<T, float16_t>::value || std::is_same<T, __half>::value
//                                        ? CUDA_R_16F : CUDA_R_16BF;
//             cublasStatus_t status;
//             if (need_reduction) {
//                 // Flatten batch: treat A as [M*tb, K], grad_out as [M*tb, N]
//                 int M_total = M * tb;
//                 status = cublasGemmEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M_total, &alpha_f,
//                     go_ptr, cuda_type, ldgo_bwd,
//                     a_ptr,  cuda_type, lda_bwd,
//                     &beta_f, gb_ptr, cuda_type, ldgb_bwd,
//                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             } else {
//                 long long stride_a_ll  = (a_ndim  >= 3) ? (long long)meta.a_strides[a_ndim-3]       : 0LL;
//                 long long stride_go_ll = (go_ndim >= 3) ? (long long)meta.grad_out_strides[go_ndim-3]: 0LL;
//                 long long stride_gb_ll = (b_ndim  >= 3) ? (long long)meta.grad_b_strides[b_ndim-3]  : 0LL;
//                 status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA_bwd, N, K_a, M, &alpha_f,
//                     go_ptr, cuda_type, ldgo_bwd, stride_go_ll,
//                     a_ptr,  cuda_type, lda_bwd,  stride_a_ll,
//                     &beta_f, gb_ptr, cuda_type, ldgb_bwd, stride_gb_ll, tb,
//                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//             }
//             if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
//         }
//    } else if constexpr (std::is_same<T, float>::value) {
//         if (supported_for_fast_path) {
//           // FP32 grad_B on secondary stream (concurrent with grad_A on primary)
//           int dev_idx = grad_output.device().index;
//           cublasHandle_t handle = get_cublas_handle_bwd2(dev_idx);
//           cublasSetStream(handle, stream_b);
//           float alpha = 1.0f, beta = 0.0f;

//           cublasOperation_t opA = a_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N; // A^T
//           int lda = a_contiguous ? meta.a_strides[a_ndim-2] : meta.a_strides[a_ndim-1]; // Correct stride 
          
//           int ldgo = meta.grad_out_strides[go_ndim-2];
//           int ldgb = meta.grad_b_strides[b_ndim-2];
          
//           cublasStatus_t status;
          
//           if (need_reduction) {
//               // Flatten Batch strategy: Treat A as [TotalRows, K], GO as [TotalRows, N]
//               // grad_B = A^T @ GO.
//               // A^T [K, TotalRows]. GO [TotalRows, N]. Result [K, N].
//               int M_total = M * tb;
              
//               status = cublasGemmEx(handle, CUBLAS_OP_N, opA, N, K_a, M_total, &alpha, go_ptr, CUDA_R_32F, ldgo, a_ptr, CUDA_R_32F, lda, &beta, gb_ptr, CUDA_R_32F, ldgb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
              
//           } else {
//              // Batched Strategy — compute strides from actual metadata
//              long long stride_a, stride_go, stride_gb;
//              bool batch_uniform_dB = true;

//              if (a_ndim <= 3) {
//                  stride_a = (a_ndim == 3) ? meta.a_strides[0] : 0;
//                  stride_go = (go_ndim == 3) ? meta.grad_out_strides[0] : 0;
//                  stride_gb = (b_ndim == 3) ? meta.grad_b_strides[0] : 0;
//              } else {
//                  // 4D: use dim[-3] strides (inner batch stride)
//                  stride_a = meta.a_strides[a_ndim - 3];
//                  stride_go = meta.grad_out_strides[go_ndim - 3];
//                  stride_gb = meta.grad_b_strides[b_ndim - 3];

//                  // Check if outer batch stride == inner_stride * inner_dim
//                  int* a_sh = meta.a_shape;
//                  int* go_sh = meta.grad_out_shape;
//                  int* b_sh = meta.b_shape;
//                  batch_uniform_dB = (meta.a_strides[0] == stride_a * a_sh[1]);
//                  batch_uniform_dB = batch_uniform_dB && (meta.grad_out_strides[0] == stride_go * go_sh[1]);
//                  batch_uniform_dB = batch_uniform_dB && (meta.grad_b_strides[0] == stride_gb * b_sh[1]);
//              }

//              if (batch_uniform_dB) {
//                  status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_ptr, CUDA_R_32F, ldgo, stride_go, a_ptr, CUDA_R_32F, lda, stride_a, &beta, gb_ptr, CUDA_R_32F, ldgb, stride_gb, tb, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//              } else {
//                  // Non-uniform 4D: loop over outer batch dim d0
//                  int d0 = meta.grad_out_shape[0];
//                  int d1 = meta.grad_out_shape[1];
//                  status = CUBLAS_STATUS_SUCCESS;
//                  for (int bi = 0; bi < d0 && status == CUBLAS_STATUS_SUCCESS; bi++) {
//                      const T* a_batch = a_ptr + bi * meta.a_strides[0];
//                      const T* go_batch = go_ptr + bi * meta.grad_out_strides[0];
//                      T* gb_batch = gb_ptr + bi * meta.grad_b_strides[0];
//                      status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, opA, N, K_a, M, &alpha, go_batch, CUDA_R_32F, ldgo, stride_go, a_batch, CUDA_R_32F, lda, stride_a, &beta, gb_batch, CUDA_R_32F, ldgb, stride_gb, d1, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
//                  }
//              }
//           }
//           if (status == CUBLAS_STATUS_SUCCESS) grad_b_done = true;
//        }
//    }
   
//    if (!grad_b_done) {
//         #ifdef AUTOGRAD_PROFILER_ENABLED
//         if (M*N*K_a > 1000000) {
//             printf("dB Fallback: M=%d K=%d N=%d, Strides: a[%d,%d], go[%d,%d], gb[%d,%d]\n", 
//                    M, K_a, N, meta.a_strides[a_ndim-2], meta.a_strides[a_ndim-1],
//                    meta.grad_out_strides[go_ndim-2], meta.grad_out_strides[go_ndim-1],
//                    meta.grad_b_strides[b_ndim-2], meta.grad_b_strides[b_ndim-1]);
//         }
//         #endif
//        // Fallback — also runs on stream_b to maintain concurrency with grad_A
//        if constexpr (std::is_same<T, float>::value)
//           matmul_backward_dB_fp32<128, 128, 16, 4, 4><<<dim3((N+127)/128, (K_a+127)/128, tb), 1024, 0, stream_b>>>(a_ptr, go_ptr, gb_ptr, M, K_a, N, tb, meta);
//        else if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
//                           std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value)
//           throw std::runtime_error("matmul_backward dB: non-contiguous fp16/bf16 fallback kernel not available, ensure contiguous tensors");
//    }

//    // Rejoin: primary stream waits for grad_B on stream_b before downstream ops read grad_B.
//    cudaEventRecord(g_cublas_evt_done[dev], stream_b);
//    cudaStreamWaitEvent(stream, g_cublas_evt_done[dev], 0);

//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//       throw std::runtime_error("Backward kernel failed: " + std::string(cudaGetErrorString(err)));
//    }
// }

// void cuda_matmul_backward(
//     const Tensor& grad_output,
//     const Tensor& A,
//     const Tensor& B,
//     Tensor& grad_A,
//     Tensor& grad_B,
//     cudaStream_t stream)
// {
//    dispatch_by_dtype(A.dtype(), [&](auto d) {
//       using T = decltype(d);
//       if constexpr (std::is_same<T, float>::value ||
//                     std::is_same<T, float16_t>::value || std::is_same<T, __half>::value ||
//                     std::is_same<T, bfloat16_t>::value || std::is_same<T, __nv_bfloat16>::value) {
//          launch_backward_matmul<T>(grad_output, A, B, grad_A, grad_B, stream);
//       } else {
//          throw std::runtime_error("cuda_matmul_backward: Unsupported type (only float, float16, bfloat16 supported)");
//       }
//    });
// }

// } // namespace OwnTensor


// #endif // WITH_MYBLAS

// #endif // WITH_CUDA