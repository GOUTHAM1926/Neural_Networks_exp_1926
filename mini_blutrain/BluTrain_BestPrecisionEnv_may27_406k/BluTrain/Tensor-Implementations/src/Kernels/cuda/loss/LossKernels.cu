#include "ops/helpers/LossKernels.h"
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <type_traits>
#include "device/CachingCudaAllocator.h"

namespace OwnTensor {
namespace cuda {

// ============================================================================
// Forward Pass CUDA Kernels
// ============================================================================

/*
* ###################################################################################################################################
* ####################################   Implementation of the two kernel approch  [Vectoirzed]   ###################################
* ###################################################################################################################################
*/
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    // Converts generic shared pointer to a 32-bit segment-specific address for PTX [6, 7]
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr); 
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" 
                 : : "r"(smem_addr), "l"(glob_ptr));
}

template<typename T>
__global__ void sum_reduction_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    #pragma unroll 4
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = static_cast<T>(sdata[0]);
    }
}

//! forward pass sparse 
template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_vec(
    const T* logits,
    const T_idx* targets,
    T* losses,
    int64_t batch_size,
    int64_t vocab_size
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;

    extern __shared__ float s_data[];
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;

    //* Online softmax
    float local_max = -1e38f;
    float local_sum = 0.0f;

    // #pragma unroll 4
    // for(int64_t j = threadIdx.x; j < vocab_size; j += blockDim.x){
    //     float val = static_cast<float>(row_logits[j]);
    //     if(val > local_max){
    //         //* Update sum based on new max: current_sum * exp(old_max - new_max) + 1.0f
    //         local_sum = local_sum * expf(local_max - val) + 1.0f;
    //         local_max = val;
    //     } else{
    //         //* Standard sum accumulation
    //         local_sum += expf(val - local_max);
    //     }
    // }
    const int64_t vec_size = 4;
    const int64_t vec_count = vocab_size/vec_size;
    //* Check row alignment for float4 (16 bytes)
    if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
        const float4* vec_input = reinterpret_cast<const float4*>(row_logits);
        for (int64_t j = threadIdx.x * 4; j < vocab_size; j += blockDim.x * 4) {
            cp_async_16(&s_data[tid * 4], &row_logits[j]);
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
            float val = static_cast<float>(row_logits[j]);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    
    //* Handle tail elements for float type
    for (int64_t j = vec_count * vec_size + threadIdx.x; j < vocab_size; j += blockDim.x) {
        float val = static_cast<float>(row_logits[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    //* Block reduction in shared memory
    // smax/ssum are placed AFTER the staging buffer to avoid a race condition:
    // with s_data aliased to the same base, smax[128] = s_data[128] would
    // conflict with thread 32's in-flight cp.async DMA to s_data[128..131].
    float *smax = s_data + bdim * 4;   // starts after staging (bdim*4 floats)
    float *ssum = s_data + bdim * 4 + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads(); //* Barrier synchronization

    //* Standard reduction loop
    for(unsigned int s = bdim/2; s > 0; s >>= 1){
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

    //* Now index 0 in the shared mem contains the block wide max and sum
    float final_max = smax[0];
    float final_sum = ssum[0];

    //* Final loss computation
    if(tid == 0){
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        float target_logit = static_cast<float>(row_logits[target_idx]);

        //* Loss = log(sum(exp(logits - max))) + max - target_logit
        float loss = logf(final_sum) + final_max - target_logit;
        losses[row] = static_cast<T>(loss);
    }
}

// =============================================================================
// Variant of sparse_ce_forward_kernel_vec that ALSO writes per-row (max, sum)
// to scratch tensors. The autograd node holds these and the backward pass uses
// them directly, skipping the redundant Reduce kernel that recomputes them.
// =============================================================================
template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_vec_save_stats(
    const T* logits,
    const T_idx* targets,
    T* losses,
    float* saved_max,
    float* saved_sum,
    int64_t batch_size,
    int64_t vocab_size
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    extern __shared__ float s_data[];
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;
    float local_max = -1e38f;
    float local_sum = 0.0f;

    const int64_t vec_size = 4;
    const int64_t vec_count = vocab_size / vec_size;
    if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
        for (int64_t j = threadIdx.x * 4; j < vocab_size; j += blockDim.x * 4) {
            cp_async_16(&s_data[tid * 4], &row_logits[j]);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
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
        for (int64_t j = threadIdx.x; j < vec_count * vec_size; j += blockDim.x) {
            float val = static_cast<float>(row_logits[j]);
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    for (int64_t j = vec_count * vec_size + threadIdx.x; j < vocab_size; j += blockDim.x) {
        float val = static_cast<float>(row_logits[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    float* smax = s_data + bdim * 4;
    float* ssum = s_data + bdim * 4 + bdim;
    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];
            if (other_max > smax[tid]) {
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }
    float final_max = smax[0];
    float final_sum = ssum[0];

    if (tid == 0) {
        saved_max[row] = final_max;       // ← saved for backward
        saved_sum[row] = final_sum;       // ← saved for backward
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        float target_logit = static_cast<float>(row_logits[target_idx]);
        float loss = logf(final_sum) + final_max - target_logit;
        losses[row] = static_cast<T>(loss);
    }
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (batch_size == 0) {
        // Set loss to 0
        cudaMemsetAsync(loss_output, 0, sizeof(T), stream);
        return;
    }
    
    // Allocate temporary buffer for per-sample losses
    T* d_losses = static_cast<T*>(CachingCUDAAllocator::instance().allocate(batch_size * sizeof(T), stream));
    
    // Kernel 1: Compute loss for each sample
    int threads = 512;//256
    int blocks = batch_size;
    // 6*threads: bdim*4 for staging buffer + bdim for smax + bdim for ssum
    size_t shared_mem = 6 * threads * sizeof(float);
    sparse_ce_forward_kernel_vec<T, T_idx><<<blocks, threads, shared_mem, stream>>>(
        logits, targets, d_losses, batch_size, vocab_size
    );
    
    // Kernel 2: Reduce to sum all losses.
    // sum_reduction_kernel loads one element per thread, so a single block
    // of reduce_threads can only sum reduce_threads elements — not 1024.
    {
        int reduce_threads = 256;
        int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;

        if (reduce_blocks == 1) {
            sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_losses, loss_output, batch_size
            );
        } else {
            T* d_partial = static_cast<T*>(CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(T), stream));

            sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_losses, d_partial, batch_size
            );
            sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_partial, loss_output, reduce_blocks
            );

            CachingCUDAAllocator::instance().deallocate(d_partial);
        }
    }

    CachingCUDAAllocator::instance().deallocate(d_losses);
}


/*
* ###################################################################################################################################
* ###################################################################################################################################
* ###################################################################################################################################
*/

/**
 * @brief Compute sparse cross entropy loss for each sample (forward pass)
 *
 * Uses numerically stable log-softmax:
 * loss[i] = log(sum(exp(logits[i] - max(logits[i])))) + max(logits[i]) - logits[i, target[i]]
 */
template<typename T, typename T_idx>
__global__ void sparse_ce_forward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* losses,
    int64_t batch_size,
    int64_t vocab_size
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x;

    extern __shared__ float s_data[];
    int64_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;

    //* Online softmax
    float local_max = -1e38f;
    float local_sum = 0.0f;

    #pragma unroll 4
    for(int64_t j = threadIdx.x; j < vocab_size; j += blockDim.x){
        float val = static_cast<float>(row_logits[j]);
        if(val > local_max){
            //* Update sum based on new max: current_sum * exp(old_max - new_max) + 1.0f
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else{
            //* Standard sum accumulation
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

    //* Standard reduction loop
    for(unsigned int s = bdim/2; s > 0; s >>= 1){
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

    //* Now index 0 in the shared mem contains the block wide max and sum
    float final_max = smax[0];
    float final_sum = ssum[0];

    //* Final loss computation
    if(tid == 0){
        int64_t target_idx = static_cast<int64_t>(targets[row]);
        float target_logit = static_cast<float>(row_logits[target_idx]);

        //* Loss = log(sum(exp(logits - max))) + max - target_logit
        float loss = logf(final_sum) + final_max - target_logit;
        losses[row] = static_cast<T>(loss);
    }
}

/**
 * @brief Parallel reduction to sum all losses
 * Uses shared memory for efficient reduction
 */

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (batch_size == 0) {
        // Set loss to 0
        cudaMemsetAsync(loss_output, 0, sizeof(T), stream);
        return;
    }
    
    // Allocate temporary buffer for per-sample losses
    T* d_losses = static_cast<T*>(CachingCUDAAllocator::instance().allocate(batch_size * sizeof(T), stream));
    
    // Kernel 1: Compute loss for each sample
    //* updated the launch config
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = 4 * threads * sizeof(float);
    sparse_ce_forward_kernel_typed<T, T_idx><<<blocks, threads, shared_mem, stream>>>(
        logits, targets, d_losses, batch_size, vocab_size
    );
    
    // Kernel 2: Reduce to sum all losses.
    // sum_reduction_kernel loads one element per thread, so a single block
    // of reduce_threads can only sum reduce_threads elements — not 1024.
    {
        int reduce_threads = 256;
        int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;

        if (reduce_blocks == 1) {
            sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_losses, loss_output, batch_size
            );
        } else {
            T* d_partial = static_cast<T*>(CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(T), stream));

            sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_losses, d_partial, batch_size
            );
            sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
                d_partial, loss_output, reduce_blocks
            );

            CachingCUDAAllocator::instance().deallocate(d_partial);
        }
    }

    CachingCUDAAllocator::instance().deallocate(d_losses);
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    sparse_cross_entropy_forward_cuda_impl<T, T_idx>(
        logits, targets, loss_output, batch_size, vocab_size, stream
    );
}

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_vec(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
) {
    sparse_cross_entropy_forward_cuda_impl_vec<T, T_idx>(
        logits, targets, loss_output, batch_size, vocab_size, stream
    );
}

// ============================================================================
// Backward Pass CUDA Kernels (existing)
// ============================================================================

/**
 * @brief CUDA kernel for sparse cross entropy backward pass
 * 
 * Gradient formula (derived from calculus):
 *   For class j:
 *     grad[j] = softmax(logits)[j] * scale           if j != target
 *     grad[j] = (softmax(logits)[j] - 1.0) * scale   if j == target
 * 
 * This is equivalent to: grad = (softmax - target_indicator) * scale
 * where target_indicator equals 1 only at the target class index.
 * 
 * @param logits     Input logits [batch_size, vocab_size]
 * @param targets    Sparse target indices [batch_size] (NOT one-hot!)
 * @param grad       Output gradient [batch_size, vocab_size]
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param scale      Gradient scaling factor (typically 1/batch_size)
 */
// template<typename T, typename T_idx>
// __global__ void sparse_ce_backward_kernel_typed(
//     const T* logits,
//     const T_idx* targets,
//     T* grad,
//     int64_t batch_size,
//     int64_t vocab_size,
//     T scale
// ) {
//     int64_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (sample_idx >= batch_size) return;

//     // Get pointer to this sample's logits and gradients
//     const T* sample_logits = logits + sample_idx * vocab_size;
//     T* sample_grad = grad + sample_idx * vocab_size;
    
//     // Get the target class for this sample (sparse index, not one-hot)
//     int64_t target_class = static_cast<int64_t>(targets[sample_idx]);
    
//     // ========================================================================
//     // Step 1: Find max logit for numerical stability
//     // ========================================================================
//     float max_logit = -1e38f;
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         float logit_val = static_cast<float>(sample_logits[c]);
//         if (logit_val > max_logit) max_logit = logit_val;
//     }
    
//     // ========================================================================
//     // Step 2: Compute sum of exp(logits - max) for softmax denominator
//     // ========================================================================
//     float sum_exp = 0.0f;
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         sum_exp += expf(static_cast<float>(sample_logits[c]) - max_logit);
//     }
    
//     // ========================================================================
//     // Step 3: Compute gradient for each class
//     // 
//     // The gradient derivation:
//     //   Loss = -log(p_target) where p = softmax(logits)
//     //   
//     //   For non-target classes (c != target):
//     //     dL/d(logit_c) = p_c
//     //   
//     //   For target class (c == target):
//     //     dL/d(logit_c) = p_c - 1
//     //
//     // This can be written as: grad_c = p_c - (c == target ? 1 : 0)
//     // ========================================================================
//     float f_scale = static_cast<float>(scale);
    
//     for (int64_t c = 0; c < vocab_size; ++c) {
//         // Compute softmax probability for this class
//         float prob = expf(static_cast<float>(sample_logits[c]) - max_logit) / sum_exp;
        
//         // Compute gradient:
//         // - For non-target classes: grad = prob * scale
//         // - For target class: grad = (prob - 1) * scale
//         float grad_val;
//         if (c == target_class) {
//             grad_val = (prob - 1.0f) * f_scale;  // Target class: subtract 1
//         } else {
//             grad_val = prob * f_scale;           // Other classes: just softmax
//         }
        
//         sample_grad[c] = static_cast<T>(grad_val);
//     }
// }
/*
* ###################################################################################################################################
* ####################################   Implementation of the two kernel approch  [Vectoirzed]   ###################################
* ###################################################################################################################################
*/

template<typename T, typename T_idx>
__global__ void sparseCEReduce_kernel_optimized(
    const T* __restrict__ logits, 
    const T_idx* __restrict__ targets,
    T* grad,
    float *d_partial_max, 
    float *d_partial_sum,
    int64_t batch_size, 
    int64_t vocab_size
){
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int xbid = blockIdx.x;
    const int ybid = blockIdx.y;
    const int xgdim = gridDim.x;
    const int bid = xbid + ybid * xgdim;

    const int row = ybid;
    if(row >= batch_size) return;

    // Identify the segment of the row for this block
    int elementsPerBlock = vocab_size / xgdim;
    int startCol = xbid * elementsPerBlock;
    int endCol = startCol + elementsPerBlock; 

    const T* row_logits = logits + row * vocab_size;

    // --- SHARED MEMORY SETUP ---
    // We need 2 sections: 1 for staging loads and 1 for the reduction stats
    extern __shared__ float sdata[];
    float* s_stage = sdata;              // Staging buffer for cp.async (bdim * 4 floats)
    float* smax = sdata + (bdim * 4);    // Reduction max (bdim floats)
    float* ssum = smax + bdim;           // Reduction sum (bdim floats)

    float local_max = -1e38f;
    float local_sum = 0.0f;

    // Vectorized Loop: Process 4 elements (16 bytes) at a time [2, 8]
    for (int64_t col = startCol + tid * 4; col < endCol; col += bdim * 4) {

        // 1. ASYNC LOAD: Initiate 128-bit transfer from Global to SMEM [2, 3, 9]
        // This bypasses L1 cache (.cg) to avoid pollution [10, 11]
        cp_async_16(&s_stage[tid * 4], &row_logits[col]);

        // 2. COORDINATION: Commit the batch and wait for arrival [12-14]
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        // No __syncthreads() here: each thread reads only its own s_stage[tid*4..tid*4+3] slot.
        // __syncthreads() inside a variable-iteration loop causes deadlock when
        // elementsPerBlock is not a multiple of bdim*4 (e.g. vocab_size=50304, bdim=256).

        // 3. COMPUTE: Perform Online Softmax math from the staging buffer
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float val = s_stage[tid * 4 + k];
            if (val > local_max) {
                local_sum = local_sum * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }

    // --- PHASE 2: BLOCK REDUCTION ---
    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = smax[tid + s];
            float other_sum = ssum[tid + s];
            
            if (other_max > smax[tid]) {
                ssum[tid] = ssum[tid] * expf(smax[tid] - other_max) + other_sum;
                smax[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - smax[tid]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial_max[bid] = smax[0];
        d_partial_sum[bid] = ssum[0];
    }
}

// =============================================================================
// Variant of sparseCENormalize_kernel_optimized that takes saved (max, sum)
// from forward DIRECTLY (one float per row, no consolidation needed). The
// backward pass that uses this kernel skips the Reduce kernel entirely,
// saving ~3.7 ms / call (~1,184 ms / training run = ~1.1% wall time).
// =============================================================================
template<typename T, typename T_idx>
__global__ void sparseCENormalize_from_stats(
    const T* __restrict__ logits,
    const T_idx* __restrict__ targets,
    T* __restrict__ grad,
    const float* __restrict__ saved_max,    // [batch_size]  (one float / row)
    const float* __restrict__ saved_sum,    // [batch_size]  (one float / row)
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale)
{
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    // Direct from saved stats — zero consolidation work
    const float final_max = saved_max[row];
    const float inv_sum   = 1.0f / saved_sum[row];
    const float f_scale   = static_cast<float>(*grad_output) * host_scale;

    extern __shared__ float sdata[];
    float* s_stage = sdata;

    for (int64_t j = tid * 4; j < vocab_size; j += bdim * 4) {
        cp_async_16(&s_stage[tid * 4], &row_logits[j]);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");

        float4 out_grad;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int64_t curr_col = j + k;
            float val = s_stage[tid * 4 + k];
            float prob = expf(val - final_max) * inv_sum;
            float indicator = (curr_col == target_idx) ? 1.0f : 0.0f;
            float grad_val = (prob - indicator) * f_scale;
            if (k == 0) out_grad.x = grad_val;
            if (k == 1) out_grad.y = grad_val;
            if (k == 2) out_grad.z = grad_val;
            if (k == 3) out_grad.w = grad_val;
        }
        if (j + 3 < vocab_size) {
            reinterpret_cast<float4*>(row_grad)[j / 4] = out_grad;
        }
    }
}

template<typename T, typename T_idx>
__global__ void sparseCENormalize_kernel_optimized(
    const T* __restrict__ logits,
    const T_idx* __restrict__ targets,
    T* __restrict__ grad,
    const float* d_partial_max,
    const float* d_partial_sum,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    int num_partials
){
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x; 

    const int row = bid;
    if(row >= batch_size) return;

    // Row-specific pointers and target info
    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    // --- PHASE 1: STATS CONSOLIDATION ---
    __shared__ float final_max_shared;
    __shared__ float final_sum_shared;

    if (tid == 0) {
        float g_max = -1e38f;
        float g_sum = 0.0f;
        for (int i = 0; i < num_partials; ++i) {
            float p_max = d_partial_max[row * num_partials + i];
            float p_sum = d_partial_sum[row * num_partials + i];
            
            if (p_max > g_max) {
                g_sum = g_sum * expf(g_max - p_max) + p_sum;
                g_max = p_max;
            } else {
                g_sum += p_sum * expf(p_max - g_max);
            }
        }
        final_max_shared = g_max;
        final_sum_shared = g_sum;
    }
    __syncthreads();

    const float final_max = final_max_shared;
    const float inv_sum = 1.0f / final_sum_shared;
    const float f_scale = static_cast<float>(*grad_output) * host_scale;

    // --- PHASE 2: VECTORIZED NORMALIZATION & STORE ---
    extern __shared__ float sdata[];
    float* s_stage = sdata; // Staging area for cp.async (blockDim.x * 4 floats)

    // Stride loop: Processes 4 elements (16 bytes) per thread
    for (int64_t j = tid * 4; j < vocab_size; j += bdim * 4) {

        // 1. ASYNC LOAD: Stream logits directly into Shared Memory
        // .cg qualifier avoids L1 pollution for data read only once [1-3].
        cp_async_16(&s_stage[tid * 4], &row_logits[j]);

        // 2. COORDINATION: Commit and wait for the "conveyor belt" [4-7].
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        // No __syncthreads() here: each thread reads only its own s_stage[tid*4..tid*4+3] slot.
        // __syncthreads() inside a variable-iteration loop causes deadlock when
        // vocab_size is not a multiple of bdim*4 (e.g. vocab_size=50304, bdim=256).

        // 3. COMPUTE & VECTORIZED STORE
        float4 out_grad;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int64_t curr_col = j + k;
            float val = s_stage[tid * 4 + k];

            // Softmax Normalization
            float prob = expf(val - final_max) * inv_sum;

            // Gradient: (Prob - Target) * Scale [8, 9]
            float indicator = (curr_col == target_idx) ? 1.0f : 0.0f;
            float grad_val = (prob - indicator) * f_scale;

            // Store in temporary float4 for vectorized write-back
            if (k == 0) out_grad.x = grad_val;
            if (k == 1) out_grad.y = grad_val;
            if (k == 2) out_grad.z = grad_val;
            if (k == 3) out_grad.w = grad_val;
        }

        // 4. Vectorized Global Store (128-bit) [10, 11]
        if (j + 3 < vocab_size) {
            reinterpret_cast<float4*>(row_grad)[j / 4] = out_grad;
        }
    }
}

//! launch function
template<typename T, typename Tidx>
void sparse_ce_backward_cuda_vec(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
){
    if(batch_size == 0) return;
    dim3 block{256}; //! play around with this..
    //* one block per row
    dim3 grid_reduce{2,batch_size}; //* lets split and then do a 2d grid...
    dim3 grid_norm{(unsigned)batch_size};
    //* Buffers for multi-block online softmax
    float *d_partial_max, *d_partial_sum;
    const size_t partial_bytes = (size_t)batch_size * grid_reduce.x * sizeof(float);
    d_partial_max = static_cast<float*>(CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
    d_partial_sum = static_cast<float*>(CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
    //* same stream for serializing and avoiding synchronization
    //* by deafult stream 0
    sparseCEReduce_kernel_optimized<<<grid_reduce, block, 6 * block.x * sizeof(float)>>>(logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size, vocab_size);
    sparseCENormalize_kernel_optimized<<<grid_norm, block, block.x * 4 * sizeof(float)>>>(logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size, vocab_size, grad_output, host_scale, (int)grid_reduce.x);

    CachingCUDAAllocator::instance().deallocate(d_partial_max);
    CachingCUDAAllocator::instance().deallocate(d_partial_sum);
}

// =============================================================================
// Forward launcher that ALSO writes per-row (max, sum) for use by the
// stats-aware backward. Allocator-allocates the two scratch tensors which
// the autograd node holds onto until backward fires.
// =============================================================================
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec_save_stats(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    float* saved_max,    // pre-allocated [batch_size]
    float* saved_sum,    // pre-allocated [batch_size]
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream)
{
    if (batch_size == 0) {
        cudaMemsetAsync(loss_output, 0, sizeof(T), stream);
        return;
    }
    T* d_losses = static_cast<T*>(
        CachingCUDAAllocator::instance().allocate(batch_size * sizeof(T), stream));

    int threads = 512;
    int blocks  = batch_size;
    size_t shared_mem = 6 * threads * sizeof(float);
    sparse_ce_forward_kernel_vec_save_stats<T, T_idx><<<blocks, threads, shared_mem, stream>>>(
        logits, targets, d_losses, saved_max, saved_sum, batch_size, vocab_size);

    int reduce_threads = 256;
    int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;
    if (reduce_blocks == 1) {
        sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, loss_output, batch_size);
    } else {
        T* d_partial = static_cast<T*>(
            CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(T), stream));
        sum_reduction_kernel<T><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_losses, d_partial, batch_size);
        sum_reduction_kernel<T><<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
            d_partial, loss_output, reduce_blocks);
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }
    CachingCUDAAllocator::instance().deallocate(d_losses);
}

// =============================================================================
// Backward launcher that REUSES forward's saved (max, sum). Only the Normalize
// kernel runs — Reduce is skipped entirely. ~3.7 ms / call faster.
// =============================================================================
template<typename T, typename Tidx>
void sparse_ce_backward_cuda_vec_with_stats(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    const float* saved_max,
    const float* saved_sum,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream)
{
    if (batch_size == 0) return;
    dim3 block{256};
    dim3 grid_norm{(unsigned)batch_size};
    sparseCENormalize_from_stats<<<grid_norm, block, block.x * 4 * sizeof(float), stream>>>(
        logits, targets, grad_logits,
        saved_max, saved_sum,
        batch_size, vocab_size,
        grad_output, host_scale);
}


/*
* ###################################################################################################################################
* ###################################################################################################################################
* ###################################################################################################################################
*/

/*
* ###################################################################################################################################
* #########################################   Implementation of the two kernel approch   ############################################
* ###################################################################################################################################
*/

//! kerenl function for reduction
template<typename T, typename T_idx>
__global__ void sparseCEReduce_kernel(
    const T* logits, //! d_input
    const T_idx* targets,
    T* grad,
    float *d_partial_max, 
    float *d_partial_sum,
    int64_t batch_size, //! rows
    int64_t vocab_size //! cols
){
    //* set the cuda variables
    auto tid = threadIdx.x;
    auto xbid = blockIdx.x;
    auto ybid = blockIdx.y;
    auto bdim = blockDim.x;
    auto xgdim = gridDim.x;
    auto bid = xbid + ybid * xgdim;

    //* 2 block per row;
    auto row = ybid;
    if(row >= batch_size) return;

    //* Identify which half of the row this block handles
    int elementsPerBlock = vocab_size / xgdim;
    int startCol = xbid * elementsPerBlock;
    int endCol = startCol + elementsPerBlock; 

    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    float local_max = -1e38f;
    float local_sum = 0.0f;

    #pragma unroll 4
    for (int64_t col = startCol + tid; col < endCol; col += bdim) {
        float val = static_cast<float>(row_logits[col]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Phase 2: Block reduction using shared memory
    extern __shared__ float sdata[];
    float *smax = sdata;
    float *ssum = sdata + bdim;

    smax[tid] = local_max;
    ssum[tid] = local_sum;
    __syncthreads();


    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = sdata[tid + s];
            float other_sum = ssum[tid + s];
            
            if (other_max > sdata[tid]) {
                ssum[tid] = ssum[tid] * expf(sdata[tid] - other_max) + other_sum;
                sdata[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - sdata[tid]);
            }
        }
        __syncthreads();
    }

    // Write this block's result to global memory
    if (tid == 0) {
        d_partial_max[bid] = smax[0];
        d_partial_sum[bid] = ssum[0];
    }
}

//! kernel function to normalize and compute loss
template<typename T, typename T_idx>
__global__ void sparseCENormalize_kernel(
    const T* logits,
    const T_idx* targets,
    T* grad,
    const float* d_partial_max,
    const float* d_partial_sum,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    int num_partials
){
    //* set the cuda variables
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int bid = blockIdx.x; //* not used
    const int gdim = gridDim.x;
    const int gidx = bid * bdim + tid;

    //* try rowin and rowout
    int row = bid;
    if(row >= batch_size) return;

    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    // All blocks need the global max and sum.
    // We can have each block reduce the partial results (which are few: num_blocks).
    __shared__ float global_max_shared;
    __shared__ float global_sum_shared;

    if (tid == 0) {
        float g_max = -1e38f;
        float g_sum = 0.0f;
        for (int i = 0; i < num_partials; ++i) {
            float p_max = d_partial_max[row * num_partials + i];
            float p_sum = d_partial_sum[row * num_partials + i];
            
            if (p_max > g_max) {
                g_sum = g_sum * expf(g_max - p_max) + p_sum;
                g_max = p_max;
            } else {
                g_sum += p_sum * expf(p_max - g_max);
            }
        }
        global_max_shared = g_max;
        global_sum_shared = g_sum;
    }
    __syncthreads();

    float final_max = global_max_shared;
    float final_sum = global_sum_shared;

    //! loss computation
    float f_scale = static_cast<float>(*grad_output) * host_scale;
    float inv_sum = 1.0f / final_sum;

    #pragma unroll 4
    for (int64_t j = tid; j < vocab_size; j += bdim) {
        float val = static_cast<float>(row_logits[j]);
        float prob = expf(val - final_max) * inv_sum;
        
        float grad_val = (j == target_idx) ? (prob - 1.0f) * f_scale : prob * f_scale;
        row_grad[j] = static_cast<T>(grad_val);
    }
}

//! launch function
template<typename T, typename Tidx>
void sparse_ce_backward_cuda(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
){
    if(batch_size == 0) return;
    dim3 block{256}; //! play around with this..
    //* one block per row
    dim3 grid_reduce{2,batch_size}; //* lets split and then do a 2d grid...
    dim3 grid_norm{(unsigned)batch_size};
    //* Buffers for multi-block online softmax
    float *d_partial_max, *d_partial_sum;
    const size_t partial_bytes = (size_t)batch_size * grid_reduce.x * sizeof(float);
    d_partial_max = static_cast<float*>(CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
    d_partial_sum = static_cast<float*>(CachingCUDAAllocator::instance().allocate(partial_bytes, stream));
    //* same stream for serializing and avoiding synchronization
    //* by deafult stream 0
    sparseCEReduce_kernel<<<grid_reduce, block, 2 * block.x * sizeof(float)>>>(logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size, vocab_size);
    sparseCENormalize_kernel<<<grid_norm, block>>>(logits, targets, grad_logits, d_partial_max, d_partial_sum, batch_size, vocab_size, grad_output, host_scale, (int)grid_reduce.x);

    CachingCUDAAllocator::instance().deallocate(d_partial_max);
    CachingCUDAAllocator::instance().deallocate(d_partial_sum);
}



/*
* ###################################################################################################################################
* ###################################################################################################################################
* ###################################################################################################################################
*/

template<typename T, typename T_idx>
__global__ void sparse_ce_backward_kernel_typed(
    const T* logits,
    const T_idx* targets,
    T* grad,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale
) {
    int64_t row = blockIdx.x; // One block per sample
    if (row >= batch_size) return;

    // Shared memory for block reduction: blockDim.x floats
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int bdim = blockDim.x;

    const T* row_logits = logits + row * vocab_size;
    T* row_grad = grad + row * vocab_size;
    int64_t target_idx = static_cast<int64_t>(targets[row]);

    // 1. Online Softmax Pass
    // We compute both Max and Sum-of-Exponents in a single pass over memory
    float local_max = -1e38f;
    float local_sum = 0.0f;

    #pragma unroll 4
    for (int64_t j = tid; j < vocab_size; j += bdim) {
        float val = static_cast<float>(row_logits[j]);
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Phase 1 Reduction: Reduce Max and Sum within the block
    sdata[tid] = local_max;
    float* ssum = sdata + bdim; // Use second half of shared memory
    ssum[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_max = sdata[tid + s];
            float other_sum = ssum[tid + s];
            
            if (other_max > sdata[tid]) {
                ssum[tid] = ssum[tid] * expf(sdata[tid] - other_max) + other_sum;
                sdata[tid] = other_max;
            } else {
                ssum[tid] += other_sum * expf(other_max - sdata[tid]);
            }
        }
        __syncthreads();
    }

    float max_val = sdata[0];
    float sum_exp = ssum[0];
    if (sum_exp < 1e-20f) sum_exp = 1e-20f;

    // 2. Compute Gradients
    float f_scale = static_cast<float>(*grad_output) * host_scale;
    float inv_sum = 1.0f / sum_exp;

    #pragma unroll 4
    for (int64_t j = tid; j < vocab_size; j += bdim) {
        float val = static_cast<float>(row_logits[j]);
        float prob = expf(val - max_val) * inv_sum;
        
        float grad_val = (j == target_idx) ? (prob - 1.0f) * f_scale : prob * f_scale;
        row_grad[j] = static_cast<T>(grad_val);
    }
}


// Specialization/Guard for non-numeric types if needed, but static_cast<float> should work for most primitives.
// Complex types will still fail to static_cast to float.

// We will restrict the instantiations to real numeric types.
template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
) {
    if (batch_size == 0) return;
    int threads = 256;
    int blocks = batch_size; // One block per sample

    sparse_ce_backward_kernel_typed<T, T_idx><<<blocks, threads, 2 * threads * sizeof(float), stream>>>(
        logits, targets, grad_logits, batch_size, vocab_size, grad_output, host_scale
    );
}

template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
) {
    sparse_ce_backward_cuda_vec<T, T_idx>(logits, targets, grad_logits, batch_size, vocab_size, grad_output, host_scale, stream);
}

// Explicit instantiations for two-kernel backward
#define INSTANTIATE_TWO_KERNEL(T) \
    template void sparse_ce_backward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t);

INSTANTIATE_TWO_KERNEL(float)
INSTANTIATE_TWO_KERNEL(double)

// Explicit instantiations for vectorized two-kernel backward
#define INSTANTIATE_TWO_KERNEL_VEC(T) \
    template void sparse_ce_backward_cuda_vec<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda_vec<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda_vec<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda_vec<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t);

INSTANTIATE_TWO_KERNEL_VEC(float)
INSTANTIATE_TWO_KERNEL_VEC(double)

// Stats-aware variants: forward writes (max, sum); backward reads them and skips Reduce.
#define INSTANTIATE_FWD_SAVE_STATS(T) \
    template void sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, uint16_t>(const T*, const uint16_t*, T*, float*, float*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, int32_t>(const T*, const int32_t*, T*, float*, float*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, int64_t>(const T*, const int64_t*, T*, float*, float*, int64_t, int64_t, cudaStream_t);
INSTANTIATE_FWD_SAVE_STATS(float)
INSTANTIATE_FWD_SAVE_STATS(float16_t)
INSTANTIATE_FWD_SAVE_STATS(bfloat16_t)

#define INSTANTIATE_BWD_WITH_STATS(T) \
    template void sparse_ce_backward_cuda_vec_with_stats<T, uint16_t>(const T*, const uint16_t*, T*, const float*, const float*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda_vec_with_stats<T, int32_t>(const T*, const int32_t*, T*, const float*, const float*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_ce_backward_cuda_vec_with_stats<T, int64_t>(const T*, const int64_t*, T*, const float*, const float*, int64_t, int64_t, const T*, float, cudaStream_t);
INSTANTIATE_BWD_WITH_STATS(float)
INSTANTIATE_BWD_WITH_STATS(float16_t)
INSTANTIATE_BWD_WITH_STATS(bfloat16_t)

// Explicit instantiations for supported types
#define INSTANTIATE_GIVEN_T(T) \
    template void sparse_cross_entropy_backward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t); \
    template void sparse_cross_entropy_backward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, const T*, float, cudaStream_t);

INSTANTIATE_GIVEN_T(float)
INSTANTIATE_GIVEN_T(double)
INSTANTIATE_GIVEN_T(float16_t)
INSTANTIATE_GIVEN_T(bfloat16_t)

// Explicit instantiations for forward pass
#define INSTANTIATE_FORWARD_GIVEN_T(T) \
    template void sparse_cross_entropy_forward_cuda<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, cudaStream_t);

INSTANTIATE_FORWARD_GIVEN_T(float)
INSTANTIATE_FORWARD_GIVEN_T(double)
INSTANTIATE_FORWARD_GIVEN_T(float16_t)
INSTANTIATE_FORWARD_GIVEN_T(bfloat16_t)

// Explicit instantiations for vectorized forward pass
#define INSTANTIATE_FORWARD_VEC_GIVEN_T(T) \
    template void sparse_cross_entropy_forward_cuda_vec<T, uint8_t>(const T*, const uint8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, uint16_t>(const T*, const uint16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, uint32_t>(const T*, const uint32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, uint64_t>(const T*, const uint64_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, int8_t>(const T*, const int8_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, int16_t>(const T*, const int16_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, int32_t>(const T*, const int32_t*, T*, int64_t, int64_t, cudaStream_t); \
    template void sparse_cross_entropy_forward_cuda_vec<T, int64_t>(const T*, const int64_t*, T*, int64_t, int64_t, cudaStream_t);

INSTANTIATE_FORWARD_VEC_GIVEN_T(float)
INSTANTIATE_FORWARD_VEC_GIVEN_T(double)
INSTANTIATE_FORWARD_VEC_GIVEN_T(float16_t)
INSTANTIATE_FORWARD_VEC_GIVEN_T(bfloat16_t)


// ============================================================================
// Categorical Cross Entropy Extensions
// ============================================================================

__global__ void cce_forward_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ losses, // [N]
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t i = blockIdx.x; // One block per sample
    if (i >= batch_size) return;

    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - 1e-7f;

    const float* row_pred = predictions + i * num_classes;
    const float* row_targ = targets + i * num_classes;

    float sum = 0.0f;
    #pragma unroll 4
    for (int64_t j = threadIdx.x; j < num_classes; j += blockDim.x) {
        float p = row_pred[j];
        float t = row_targ[j];
        
        // Clip
        if (p < epsilon) p = epsilon;
        else if (p > one_minus_epsilon) p = one_minus_epsilon;
        
        sum += t * logf(p);
    }
    
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        losses[i] = -sdata[0]; // Negate here (-sum is loss)
    }
}

__global__ void scale_loss_kernel(float* val, float s) {
    if (threadIdx.x == 0) *val *= s;
}

void categorical_cross_entropy_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t batch_size,
    int64_t num_classes
) {
    if (batch_size == 0) return;
    
    float* d_losses = static_cast<float*>(
        CachingCUDAAllocator::instance().allocate(batch_size * sizeof(float)));

    int threads = 256;
    int blocks = batch_size;

    // 1. Per-sample loss
    cce_forward_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        predictions, targets, d_losses, batch_size, num_classes);

    // 2. Reduce sum
    int reduce_threads = 256;
    int reduce_blocks = (batch_size + reduce_threads - 1) / reduce_threads;

    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, loss_output, batch_size);
    } else {
        float* d_partial = static_cast<float*>(
            CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(float)));
         sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_losses, d_partial, batch_size);
         sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
            d_partial, loss_output, reduce_blocks);
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }

    // 3. Average (Divide by batch_size)
    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(batch_size));

    CachingCUDAAllocator::instance().deallocate(d_losses);
}

__global__ void cce_backward_kernel(
    const float* __restrict__ grad_output, // scalar
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ grad_input,
    int64_t numel,
    float scale // 1/N
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    
    const float epsilon = 1e-7f;
    const float one_minus_epsilon = 1.0f - 1e-7f;
    
    float p = predictions[i];
    float t = targets[i];
    float g = *grad_output; // Access scalar gradient (assumed on GPU)
    
    float grad = 0.0f;
    // d(log(clip(p))) = 1/clipped_p if not clamped, else 0
    if (p >= epsilon && p <= one_minus_epsilon) {
        grad = g * (-t / p);
    }
    
    grad_input[i] = grad * scale;
}

void categorical_cross_entropy_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t numel = batch_size * num_classes;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    cce_backward_kernel<<<blocks, threads>>>(
        grad_output, predictions, targets, grad_input, numel, 1.0f / static_cast<float>(batch_size));
}


// ============================================================================
// MSE / MAE / BCE Extensions
// ============================================================================

// --- MSE ---
__global__ void mse_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float diff = p[i] - t[i];
    out[i] = diff * diff;
}

__global__ void mse_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = *grad_out; // scalar
    float diff = p[i] - t[i];
    grad_in[i] = 2.0f * diff * g * scale;
}

void mse_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses = static_cast<float*>(
        CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mse_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);

    // Reduce
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial = static_cast<float*>(
            CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(float)));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }

    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    CachingCUDAAllocator::instance().deallocate(d_losses);
}

void mse_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mse_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

// --- MAE ---
__global__ void mae_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float diff = p[i] - t[i];
    out[i] = fabsf(diff);
}

__global__ void mae_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = *grad_out;
    float diff = p[i] - t[i];
    float sign = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    grad_in[i] = sign * g * scale;
}

void mae_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses = static_cast<float*>(
        CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mae_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);

    // Reduce (same logic as MSE, could abstract but copy-paste is safer for now without templating spaghetti)
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial = static_cast<float*>(
            CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(float)));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }

    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    CachingCUDAAllocator::instance().deallocate(d_losses);
}

void mae_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mae_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

// --- BCE ---
__global__ void bce_forward_kernel(
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float pi = p[i];
    float ti = t[i];
    
    // Clip
    const float eps = 1e-7f;
    const float one_minus_eps = 1.0f - 1e-7f;
    if (pi < eps) pi = eps;
    else if (pi > one_minus_eps) pi = one_minus_eps;
    
    // Loss = -(t * log(p) + (1-t) * log(1-p))
    out[i] = -(ti * logf(pi) + (1.0f - ti) * logf(1.0f - pi));
}

__global__ void bce_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ p,
    const float* __restrict__ t,
    float* __restrict__ grad_in,
    int64_t n,
    float scale
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float g = *grad_out;
    float pi = p[i];
    float ti = t[i];
    
    // Clip
    const float eps = 1e-7f;
    const float one_minus_eps = 1.0f - eps;
    float p_clipped = pi;
    if (p_clipped < eps) p_clipped = eps;
    else if (p_clipped > one_minus_eps) p_clipped = one_minus_eps;
    
    // grad = (-t/p + (1-t)/(1-p)) * scale * g
    // if clipped, grad might be 0 theoretically, but standard DL frameworks usually pass gradient through clipped values or use logits.
    // Here we replicate strict derivative of the loss function formula with clipped p.
    
    float term1 = -ti / p_clipped;
    float term2 = (1.0f - ti) / (1.0f - p_clipped);
    grad_in[i] = (term1 + term2) * g * scale;
}

void bce_loss_forward_cuda(const float* predictions, const float* targets, float* loss_output, int64_t numel) {
    if (numel == 0) return;
    float* d_losses = static_cast<float*>(
        CachingCUDAAllocator::instance().allocate(numel * sizeof(float)));

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    bce_forward_kernel<<<blocks, threads>>>(predictions, targets, d_losses, numel);

    // Reduce
    int reduce_threads = 256;
    int reduce_blocks = (numel + reduce_threads - 1) / reduce_threads;

    if (reduce_blocks == 1) {
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, loss_output, numel);
    } else {
        float* d_partial = static_cast<float*>(
            CachingCUDAAllocator::instance().allocate(reduce_blocks * sizeof(float)));
        sum_reduction_kernel<float><<<reduce_blocks, reduce_threads, reduce_threads*sizeof(float)>>>(d_losses, d_partial, numel);
        sum_reduction_kernel<float><<<1, reduce_threads, reduce_threads*sizeof(float)>>>(d_partial, loss_output, reduce_blocks);
        CachingCUDAAllocator::instance().deallocate(d_partial);
    }

    scale_loss_kernel<<<1, 1>>>(loss_output, 1.0f / static_cast<float>(numel));
    CachingCUDAAllocator::instance().deallocate(d_losses);
}

void bce_loss_backward_cuda(const float* grad_output, const float* predictions, const float* targets, float* grad_input, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    bce_backward_kernel<<<blocks, threads>>>(grad_output, predictions, targets, grad_input, numel, 1.0f/numel);
}

} // namespace cuda
} // namespace OwnTensor