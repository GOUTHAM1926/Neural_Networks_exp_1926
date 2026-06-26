#pragma once

#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace OwnTensor
{

// Optimization: Tiled AVX2 Matmul for float32
template<typename T>
void cpu_matmul_optimized_float(const T* a, const T* b, T* c, size_t M, size_t N, size_t K,
                                size_t lda, size_t ldb, size_t ldc) {
    if constexpr (!std::is_same_v<T, float>) {
        return; // Only for float32
    } else {
        #ifdef __AVX2__
        const int BLOCK_M = 64;
        const int BLOCK_N = 64;
        const int BLOCK_K = 32;

        #pragma omp parallel for collapse(2)
        for (size_t im = 0; im < M; im += BLOCK_M) {
            for (size_t jn = 0; jn < N; jn += BLOCK_N) {
                for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                    size_t m_end = std::min(im + BLOCK_M, M);
                    size_t n_end = std::min(jn + BLOCK_N, N);
                    size_t k_end = std::min(kk + BLOCK_K, K);

                    for (size_t i = im; i < m_end; ++i) {
                        for (size_t k = kk; k < k_end; ++k) {
                            __m256 va = _mm256_set1_ps(a[i * lda + k]);
                            size_t j = jn;
                            for (; j + 7 < n_end; j += 8) {
                                __m256 vb = _mm256_loadu_ps(&b[k * ldb + j]);
                                __m256 vc = _mm256_loadu_ps(&c[i * ldc + j]);
                                // c = a*b + c (fused multiply-add if possible, but load/store is fine)
                                // We use fmadd if available, but for now standard mul+add
                                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                                _mm256_storeu_ps(&c[i * ldc + j], vc);
                            }
                            // Remainder
                            for (; j < n_end; ++j) {
                                c[i * ldc + j] += a[i * lda + k] * b[k * ldb + j];
                            }
                        }
                    }
                }
            }
        }
        #else
        // Fallback to standard tiled if no AVX2
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_val = a[i * lda + k];
                for (size_t j = 0; j < N; ++j) {
                    c[i * ldc + j] += a_val * b[k * ldb + j];
                }
            }
        }
        #endif
    }
}

void cpu_matmul(const Tensor& A, const Tensor& B, Tensor& output)
{
    // Initialize output with zeros if not already (important for the accumulation logic)
    output.fill<float>(0.0f); 

    dispatch_by_dtype(A.dtype(), [&](auto dummy)
    {
        using T = decltype(dummy);
        if constexpr (std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>) {
             throw std::runtime_error("Matrix Multiplication is not supported for FP4 types.");
        } else {
            const T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            T* out_ptr = output.data<T>();
    
            const auto& a_shape = A.shape().dims;
            const auto& b_shape = B.shape().dims;
            const auto& out_shape = output.shape().dims;
    
            const auto& a_strides = A.stride().strides;
            const auto& b_strides = B.stride().strides;
            const auto& out_strides = output.stride().strides;
    
            size_t a_ndim = a_shape.size();
            size_t b_ndim = b_shape.size();
            size_t out_ndim = out_shape.size();
    
            size_t m = a_shape[a_ndim - 2];
            size_t n = a_shape[a_ndim - 1]; // K in matmul terms
            size_t p = b_shape[b_ndim - 1]; // N in matmul terms
    
            size_t batch_dims = out_ndim - 2;

            // Fast Path: Contiguous Float32 2D Matrices
            if constexpr (std::is_same_v<T, float>) {
                if (out_ndim == 2 && a_ndim == 2 && b_ndim == 2 &&
                    a_strides[1] == 1 && b_strides[1] == 1 && out_strides[1] == 1) {
                    cpu_matmul_optimized_float<float>(a_ptr, b_ptr, out_ptr, m, p, n, 
                                                       a_strides[0], b_strides[0], out_strides[0]);
                    return;
                }
            }

            std::vector<size_t> batch_idx(batch_dims, 0);
            while (true)
            {
                size_t a_batch_offset = 0;
                size_t b_batch_offset = 0;
                size_t out_batch_offset = 0;
    
                for (size_t i = 0; i < batch_dims; ++i)
                {
                    size_t out_idx = batch_idx[i];
                    out_batch_offset += out_idx * out_strides[i];
    
                    size_t a_batch_dim = i - (batch_dims - (a_ndim - 2));
                    size_t b_batch_dim = i - (batch_dims - (b_ndim - 2));
    
                    if (i >= batch_dims - (a_ndim - 2)) {
                        size_t a_dim = a_shape[a_batch_dim];
                        size_t a_idx = (a_dim > 1) ? out_idx : 0;
                        a_batch_offset += a_idx * a_strides[a_batch_dim];
                    }
    
                    if (i >= batch_dims - (b_ndim - 2)) {
                        size_t b_dim = b_shape[b_batch_dim];
                        size_t b_idx = (b_dim > 1) ? out_idx : 0;
                        b_batch_offset += b_idx * b_strides[b_batch_dim];
                    }
                }
                
                // For Batched Matmul, check if specific batch is contiguous
                if constexpr (std::is_same_v<T, float>) {
                    if (a_strides[a_ndim-1] == 1 && b_strides[b_ndim-1] == 1 && out_strides[out_ndim-1] == 1) {
                        cpu_matmul_optimized_float<float>(a_ptr + a_batch_offset, b_ptr + b_batch_offset, 
                                                           out_ptr + out_batch_offset, m, p, n,
                                                           a_strides[a_ndim-2], b_strides[b_ndim-2], out_strides[out_ndim-2]);
                    } else {
                        // Regular slow path for non-contiguous patches
                        for (size_t i = 0; i < m; ++i) {
                            for (size_t j = 0; j < p; ++j) {
                                T sum{};
                                for (size_t k = 0; k < n; ++k) {
                                    size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
                                    size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
                                    sum += a_ptr[a_idx] * b_ptr[b_idx];
                                }
                                size_t o_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
                                out_ptr[o_idx] = sum;
                            }
                        }
                    }
                } else {
                    // Non-float path
                    for (size_t i = 0; i < m; ++i)
                    {
                        for (size_t j = 0; j < p; ++j)
                        {
                            T sum{};
                            for (size_t k = 0; k < n; ++k)
                            {
                                size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
                                size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
                                sum += a_ptr[a_idx] * b_ptr[b_idx];
                            }
                            size_t o_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
                            out_ptr[o_idx] = sum;
                        }
                    }
                }
    
                bool all_batches_processed = true;
                for (size_t dim = batch_dims; dim-- > 0;)
                {
                    batch_idx[dim]++;
                    if (batch_idx[dim] < static_cast<size_t>(out_shape[dim])) {
                        all_batches_processed = false;
                        break;
                    }
                    batch_idx[dim] = 0;
                }
    
                if (all_batches_processed) break;
            }
        }
    });
}
}
            
