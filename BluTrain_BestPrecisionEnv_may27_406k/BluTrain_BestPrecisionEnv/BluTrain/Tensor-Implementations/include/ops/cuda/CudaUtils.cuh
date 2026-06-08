#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <type_traits>
#include "core/Tensor.h"
#include "dtype/Types.h"

namespace OwnTensor {

    // Vectorization traits
    template <typename T, int VecSize>
    struct VectorType;

    template <> struct VectorType<float, 4> { using type = float4; };
    template <> struct VectorType<float, 2> { using type = float2; };
    template <> struct VectorType<double, 2> { using type = double2; };
    template <> struct VectorType<__half, 2> { using type = half2; };
    template <> struct VectorType<float16_t, 2> { using type = half2; };

    // BFloat16 vectorization (nv_bfloat162)
    #if __CUDA_ARCH__ >= 800
    template <> struct VectorType<__nv_bfloat16, 2> { using type = __nv_bfloat162; };
    template <> struct VectorType<bfloat16_t, 2> { using type = __nv_bfloat162; };
    #endif

    // Helper to get number of SMs
    inline int get_num_sms() {
        int deviceId;
        cudaGetDevice(&deviceId);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
        return numSMs;
    }

    // Dynamic grid size calculation based on occupancy
    template <typename T_kernel>
    inline size_t get_optimal_grid_size(T_kernel kernel, int block_size, size_t total_elems) {
        int deviceId;
        cudaGetDevice(&deviceId);
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, deviceId);
        //return num_sms*4;
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

        // Aim for ~2-4 waves to hide memory latency
        const int wave_factor = 4;
        size_t optimal_grid = (size_t)num_sms * max_active_blocks * wave_factor;
        
        // Clamp to total_elems/block_size (don't launch more blocks than elements)
        size_t max_useful_blocks = (total_elems + block_size - 1) / block_size;
        return std::min(optimal_grid, max_useful_blocks > 0 ? max_useful_blocks : (size_t)1);
    }

    // Metadata for broadcasting (passed by value to save launch latency)
    struct SimplifiedBroadcastingMetadata {
        size_t a_shape[8];
        size_t b_shape[8];
        size_t out_shape[8];
        size_t a_strides[8];
        size_t b_strides[8];
        int ndim;
    };

    inline SimplifiedBroadcastingMetadata align_and_collapse_dims(
        const Shape& A_shape, const Stride& A_stride,
        const Shape& B_shape, const Stride& B_stride,
        const Shape& Out_shape, const Stride& Out_stride) 
    {
        size_t a_ndim = A_shape.dims.size();
        size_t b_ndim = B_shape.dims.size();
        size_t out_ndim = Out_shape.dims.size();

        size_t a_dims_padded[8], a_strides_padded[8];
        size_t b_dims_padded[8], b_strides_padded[8];
        size_t out_dims_padded[8], out_strides_padded[8];

        for (size_t i = 0; i < out_ndim; ++i) {
            int a_idx = (int)i - ((int)out_ndim - (int)a_ndim);
            if (a_idx >= 0) {
                a_dims_padded[i] = A_shape.dims[a_idx];
                a_strides_padded[i] = A_stride.strides[a_idx];
            } else {
                a_dims_padded[i] = 1;
                a_strides_padded[i] = 0;
            }

            int b_idx = (int)i - ((int)out_ndim - (int)b_ndim);
            if (b_idx >= 0) {
                b_dims_padded[i] = B_shape.dims[b_idx];
                b_strides_padded[i] = B_stride.strides[b_idx];
            } else {
                b_dims_padded[i] = 1;
                b_strides_padded[i] = 0;
            }

            out_dims_padded[i] = Out_shape.dims[i];
            out_strides_padded[i] = Out_stride.strides[i];
        }

        SimplifiedBroadcastingMetadata meta;
        int current_dim = 0;
        meta.out_shape[0] = out_dims_padded[0];
        meta.a_shape[0] = a_dims_padded[0];
        meta.b_shape[0] = b_dims_padded[0];
        meta.a_strides[0] = a_strides_padded[0];
        meta.b_strides[0] = b_strides_padded[0];

        for (size_t i = 1; i < out_ndim; ++i) {
            bool can_collapse = true;
            if (out_strides_padded[i-1] != out_strides_padded[i] * out_dims_padded[i]) can_collapse = false;
            
            auto check_mergeable = [](size_t d_prev, size_t d_curr, size_t s_prev, size_t s_curr) {
                if (d_curr == 1 && d_prev == 1) return true;
                if (d_curr > 1 && d_prev > 1 && s_prev == s_curr * d_curr) return true;
                return false; 
            };

            if (!check_mergeable(a_dims_padded[i-1], a_dims_padded[i], a_strides_padded[i-1], a_strides_padded[i])) can_collapse = false;
            if (!check_mergeable(b_dims_padded[i-1], b_dims_padded[i], b_strides_padded[i-1], b_strides_padded[i])) can_collapse = false;

            if (can_collapse) {
                meta.out_shape[current_dim] *= out_dims_padded[i];
                meta.a_shape[current_dim] *= a_dims_padded[i];
                meta.b_shape[current_dim] *= b_dims_padded[i];
                meta.a_strides[current_dim] = a_strides_padded[i];
                meta.b_strides[current_dim] = b_strides_padded[i];
            } else {
                current_dim++;
                meta.out_shape[current_dim] = out_dims_padded[i];
                meta.a_shape[current_dim] = a_dims_padded[i];
                meta.b_shape[current_dim] = b_dims_padded[i];
                meta.a_strides[current_dim] = a_strides_padded[i];
                meta.b_strides[current_dim] = b_strides_padded[i];
            }
        }
        meta.ndim = current_dim + 1;
        return meta;
    }

} // namespace OwnTensor
