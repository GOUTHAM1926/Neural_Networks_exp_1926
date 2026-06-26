#include "ops/cuda/activations/ActivationCommon.cuh"
#include <type_traits>
#include <memory>

namespace OwnTensor {
namespace cuda {

// --- Warp Reduction Helpers ---
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

//* PTX helper for cp.async (16-byte load - 4 floats)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(glob_ptr));
}

//TODO: Apply vectorization and inline ptx
// if it outperforms the online one go for this kernel.
// __global__ void softmax_forward_kernel(
//     const float* __restrict__ input,
//     float* __restrict__ output,
//     int64_t rows,
//     int64_t cols
// ) {
//     extern __shared__ float s_data[]; //* alignas(16)
//     const int tid = threadIdx.x;
//     const int bdim = blockDim.x;
//     const int bid = blockIdx.x;

//     int row = bid;
//     if (row >= rows) return;

//     // One block per row
//     const float* row_input = input + row * cols;
//     float* row_output = output + row * cols;

//     // 1. Find Max for numerical stability
//     // Each thread accumulates its own partial max — no __syncthreads() needed here.
//     float max_val = -INFINITY;
//     //* Optimization: Vectorized loads for float type
//     const int64_t vec_size = 4;
//     const int64_t vec_count = cols / vec_size;
//     if((reinterpret_cast<uintptr_t>(row_input) & 0xF) == 0){
//         const float4* vec_input = reinterpret_cast<const float4*>(row_input);
//         for(int64_t i = tid * 4; i < cols; i += bdim * 4){
//             cp_async_16(&s_data[tid * 4], &row_input[i]);
//             asm volatile("cp.async.commit_group;\n" ::: "memory");
//             asm volatile("cp.async.wait_group 0;\n" ::: "memory");
//             #pragma unroll
//             for(int k = 0; k < 4; ++k){
//                 max_val = fmaxf(max_val, s_data[tid * 4 + k]);
//             }
//         }
//     } else {
//         //* Fallback to scalar for unaligned rows
//         for (int64_t i = tid; i < vec_count * vec_size; i += bdim) {
//             max_val = fmaxf(max_val, row_input[i]);
//         }
//     }

//     //* Handle tail elements
//     for (int64_t i = vec_count * vec_size + tid; i < cols; i += bdim) {
//         max_val = fmaxf(max_val, row_input[i]);
//     }
    
//     // Block reduce max
//     max_val = warpReduceMax(max_val);
    
//     static __shared__ float s_max;
//     if (threadIdx.x == 0) s_max = -INFINITY;
//     __syncthreads();
    
//     static __shared__ float warp_vals[32]; // Max 1024 threads = 32 warps
//     int warpId = threadIdx.x / warpSize;
//     int laneId = threadIdx.x % warpSize;
    
//     if (laneId == 0) warp_vals[warpId] = -INFINITY;
//     __syncthreads();
    
//     if (laneId == 0) warp_vals[warpId] = max_val;
//     __syncthreads();
    
//     if (threadIdx.x == 0) {
//         float block_max = -INFINITY;
//         int num_warps = (blockDim.x + warpSize - 1) / warpSize;
//         for (int i=0; i<num_warps; ++i) {
//             block_max = fmaxf(block_max, warp_vals[i]);
//         }
//         s_max = block_max;
//     }
//     __syncthreads();
//     max_val = s_max;
    
//     // 2. Compute Exp and Sum
//     float sum_exp = 0.0f;
//     #pragma unroll 4
//     for (int i = threadIdx.x; i < cols; i += blockDim.x) {
//         float val = expf(row_input[i] - max_val);
//         sum_exp += val;
//         row_output[i] = val; // Store exp(x-max)
//     }
//     sum_exp = warpReduceSum(sum_exp);
    
//     // Block reduce sum
//     static __shared__ float s_sum;
//     if (laneId == 0) warp_vals[warpId] = sum_exp;
//     __syncthreads();
    
//     if (threadIdx.x == 0) {
//         float block_sum = 0.0f;
//         int num_warps = (blockDim.x + warpSize - 1) / warpSize;
//         for (int i=0; i<num_warps; ++i) {
//             block_sum += warp_vals[i];
//         }
//         s_sum = block_sum;
//     }
//     __syncthreads();
//     sum_exp = s_sum;
    
//     // 3. Normalize
//     float inv_sum = fast_rcp(sum_exp);
//     #pragma unroll 4
//     for (int i = tid; i < cols; i += bdim) {
//         row_output[i] *= inv_sum;
//     }
// }

// --- Online Softmax Forward ---
__global__ void softmaxOnline_forward_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    int64_t rows,
    int64_t cols
){
    extern __shared__ float s_data[]; //* alignas(16)
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;

    int row = bid;
    if (row >= rows) return;

    const float* row_input = d_input + row * cols;
    float* row_output = d_output + row * cols;
    
    float local_max = -1e38f;
    float local_sum = 0.0f;

    //* Optimization: Vectorized loads for float type
    const int64_t vec_size = 4;
    const int64_t vec_count = cols / vec_size;
    
    //* Check row alignment for float4 (16 bytes)
    if ((reinterpret_cast<uintptr_t>(row_input) & 0xF) == 0) {
        const float4* vec_input = reinterpret_cast<const float4*>(row_input);
        for (int64_t j = threadIdx.x* 4; j < cols; j += blockDim.x * 4) {
            cp_async_16(&s_data[tid * 4], &row_input[j]);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            // No __syncthreads() here: each thread only reads its own s_data[tid*4..tid*4+3] slot.
            // __syncthreads() inside a variable-iteration loop causes a deadlock when
            // cols is not a multiple of bdim*4.
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val = s_data[tid * 4 + k];
                if (val > local_max) {
                    local_sum = local_sum * expf(local_max - val) + 1.0f;
                    local_max = val;
                } else {
                    local_sum += expf(val - local_max);
                }
            }
        }
    } else {
        //* Fallback to scalar for unaligned rows
        for (int64_t j = threadIdx.x; j < vec_count * vec_size; j += blockDim.x) {
            float val = static_cast<float>(row_input[j]);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    
    //* Handle tail elements for float type
    for (int64_t j = vec_count * vec_size + threadIdx.x; j < cols; j += blockDim.x) {
        float val = static_cast<float>(row_input[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    //* Block reduction in shared memory
    extern __shared__ float sdata[];
    //* Use two halves of shared memory: one for max and one for sum
    float *smax = sdata;
    float *ssum = sdata + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads(); //* Barrier synchronization

    //* Reduction loop: down to warp size
    for(unsigned int s = bdim/2; s >= 32; s >>= 1){
        if(tid < s){
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];

            //* Merge logic (Online softmax)
            if(other_max > smax[tid]){
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else{
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }

    //* Warp-level reduction
    if (tid < 32) {
        float cur_max = smax[tid];
        float cur_sum = ssum[tid];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xFFFFFFFF, cur_max, offset);
            float other_sum = __shfl_down_sync(0xFFFFFFFF, cur_sum, offset);
            
            if (other_max > cur_max) {
                cur_sum = cur_sum * expf(cur_max - other_max) + other_sum;
                cur_max = other_max;
            } else {
                cur_sum += other_sum * expf(other_max - cur_max);
            }
        }
        smax[tid] = cur_max;
        ssum[tid] = cur_sum;
    }
    __syncthreads();


    //* Now index 0 in the shared mem contains the block wide max and sum
    float final_max = smax[0];
    float final_sum = ssum[0];
    //* smax and ssum contains the global max and global sum
    for (size_t i = tid; i < cols; i += bdim) { //* problem size_t i = gidx; i < cols; i += gdim * bdim
        row_output[i] = expf(row_input[i] - final_max) / final_sum; //* should work now...
    }
}

// --- Softmax Backward ---
template<typename T>
__global__ void softmax_backward_kernel(
    const T* __restrict__ grad_output, 
    const T* __restrict__ output, 
    T* __restrict__ grad_input, 
    int64_t rows, 
    int64_t cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const T* row_grad = grad_output + row * cols;
    const T* row_out = output + row * cols;
    T* row_gin = grad_input + row * cols;

    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dot += to_float(row_grad[i]) * to_float(row_out[i]);
    }
    dot = warpReduceSum(dot);

    static __shared__ float s_dot;
    static __shared__ float warp_vals[32];
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    if (laneId == 0) warp_vals[warpId] = dot;
    __syncthreads();

    if (threadIdx.x == 0) {
        float b_dot = 0.0f;
        for (int i=0; i<(blockDim.x+31)/32; ++i) b_dot += warp_vals[i];
        s_dot = b_dot;
    }
    __syncthreads();
    dot = s_dot;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float s = to_float(row_out[i]), g = to_float(row_grad[i]);
        row_gin[i] = from_float<T>(s * (g - dot));
    }
}

void launch_softmax_generic(const float* in, float* out, int64_t rows, int64_t cols, cudaStream_t s) {
    int threads = (cols <= 1024) ? 256 : 1024;
    softmaxOnline_forward_kernel<<<rows, threads, 4 * threads * sizeof(float), s>>>(in, out, rows, cols);
}

void launch_softmax_backward_generic(const float* go, const float* out, float* gi, int64_t rows, int64_t cols, cudaStream_t s) {
    int threads = (cols <= 1024) ? 256 : 1024;
    softmax_backward_kernel<float><<<rows, threads, 0, s>>>(go, out, gi, rows, cols);
}

} // namespace cuda
} // namespace OwnTensor



//! Future - for low precision
// =============================================================================
// TEMPLATED SOFTMAX FOR HALF TYPES (read T → compute fp32 → write T)
// =============================================================================
// template <typename T>
// __global__ void softmax_forward_kernel_mixed(
//     const T* __restrict__ input,
//     T* __restrict__ output,
//     int64_t rows,
//     int64_t cols
// ) {
//     extern __shared__ float s_data[];
//     const int tid = threadIdx.x;
//     const int bdim = blockDim.x;

//     int row = blockIdx.x;
//     if (row >= rows) return;

//     const T* row_input = input + row * cols;
//     T* row_output = output + row * cols;

//     // 1. Find max for numerical stability
//     float max_val = -INFINITY;
//     for (int64_t i = tid; i < cols; i += bdim) {
//         max_val = fmaxf(max_val, static_cast<float>(row_input[i]));
//     }

//     // Warp reduce max
//     max_val = warpReduceMax(max_val);

//     static __shared__ float s_max;
//     static __shared__ float warp_vals[32];
//     int warpId = tid / warpSize;
//     int laneId = tid % warpSize;

//     if (laneId == 0) warp_vals[warpId] = max_val;
//     __syncthreads();

//     if (tid == 0) {
//         float block_max = -INFINITY;
//         int num_warps = (bdim + warpSize - 1) / warpSize;
//         for (int i = 0; i < num_warps; ++i)
//             block_max = fmaxf(block_max, warp_vals[i]);
//         s_max = block_max;
//     }
//     __syncthreads();
//     max_val = s_max;

//     // 2. Compute exp and sum
//     float sum_exp = 0.0f;
//     for (int64_t i = tid; i < cols; i += bdim) {
//         float val = expf(static_cast<float>(row_input[i]) - max_val);
//         sum_exp += val;
//         s_data[i] = val; // store in shared memory (fp32)
//     }
//     __syncthreads();

//     sum_exp = warpReduceSum(sum_exp);

//     static __shared__ float s_sum;
//     if (laneId == 0) warp_vals[warpId] = sum_exp;
//     __syncthreads();

//     if (tid == 0) {
//         float block_sum = 0.0f;
//         int num_warps = (bdim + warpSize - 1) / warpSize;
//         for (int i = 0; i < num_warps; ++i)
//             block_sum += warp_vals[i];
//         s_sum = block_sum;
//     }
//     __syncthreads();
//     sum_exp = s_sum;

//     // 3. Normalize and write back as T
//     float inv_sum = fast_rcp(sum_exp);
//     for (int64_t i = tid; i < cols; i += bdim) {
//         row_output[i] = static_cast<T>(s_data[i] * inv_sum);
//     }
// }

// template <typename T>
// void softmax_forward_cuda_typed(const T* input, T* output, int64_t rows, int64_t cols) {
//     int threads = (cols <= 1024) ? 256 : 1024;
//     if (threads < 32) threads = 32;
//     dim3 blocks(rows);
//     // Shared memory: cols floats for intermediate exp values
//     size_t smem = cols * sizeof(float);
//     softmax_forward_kernel_mixed<T><<<blocks, threads, smem>>>(input, output, rows, cols);
// }

// // Explicit instantiations
// template void softmax_forward_cuda_typed<__half>(const __half*, __half*, int64_t, int64_t);
// template void softmax_forward_cuda_typed<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t);

// void softmaxonline_forward_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
//     int threads = (cols <= 1024) ? 256 : 1024;
//     // One block per row
//     dim3 blocks(rows);
//     softmaxOnline_forward_kernel<<<blocks, threads, 4 * threads * sizeof(float)>>>(input, output, rows, cols);
// }