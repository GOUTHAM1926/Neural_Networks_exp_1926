#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

namespace OwnTensor {
namespace cuda {

// Input: logits [B, T, V_local], targets [B, T]
// Output: out [B, T, 1]
// Check if target[b,t] is in [start_v, start_v + V_local). If so, gather logit.
__global__ void extract_target_logits_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ out,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = B * T;

    if (idx < numel) {
        // targets should be integer-like, but passed as float
        int64_t target_idx = static_cast<int64_t>(targets[idx]);
        
        // Check if target is in local vocabulary partition
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            int64_t local_col = target_idx - start_v;
            // logits shape is [B*T, V_local] logically
            // logits are contiguous: [row0, row1, ...]
            out[idx] = logits[idx * V_local + local_col];
        } else {
            out[idx] = 0.0f; // Padding or other rank holds the target
        }
    }
}

void launch_extract_target_logits(const float* logits, const float* targets, float* out,
                                  int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream) {
    int64_t numel = B * T;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    extract_target_logits_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, out, B, T, V_local, start_v
    );
    
    // Check for launch errors (optional, but good for debugging)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA Error in extract_target_logits: " << cudaGetErrorString(err) << std::endl;
    // }
}

// Input: grad [B, T, V_local], targets [B, T]
// Operation: grad[b, t, target - start_v] -= g_out
// Only if target is in local range.
__global__ void sparse_subtract_kernel(
    float* __restrict__ grad,
    const float* __restrict__ targets,
    float g_out,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = B * T;

    if (idx < numel) {
        int64_t target_idx = static_cast<int64_t>(targets[idx]);

        // Check availability in local shard
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            int64_t local_col = target_idx - start_v;
            // Subtract g_out from the specific logit gradient
            grad[idx * V_local + local_col] -= g_out;
        }
    }
}

void launch_sparse_subtract(float* grad, const float* targets, float g_out,
                           int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream) {
    int64_t numel = B * T;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    sparse_subtract_kernel<<<blocks, threads, 0, stream>>>(
        grad, targets, g_out, B, T, V_local, start_v
    );
}

// =============================================================================
// Fused Forward: sum_exp + target_logit extraction in ONE pass
// =============================================================================
// One thread block per (b,t) position. Block reduction over V_local elements.
// Replaces: exp(logits-max) → reduce_sum → extract_target (3 passes → 1)

__global__ void fused_sum_exp_target_kernel(
    const float* __restrict__ logits,      // [BT * V_local]
    const float* __restrict__ global_max,  // [BT]
    const float* __restrict__ targets,     // [BT] as float
    float* __restrict__ packed_out,        // [BT * 2]: first BT = sum_exp, next BT = target_logit
    float* __restrict__ exp_logits,        // [BT * V_local]
    int64_t BT, int64_t V_local, int64_t start_v
) {
    extern __shared__ float sdata[];  // blockDim.x floats

    int64_t bt = blockIdx.x;
    if (bt >= BT) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = global_max[bt];
    int64_t target_idx = (int64_t)targets[bt];
    const float* row = logits + bt * V_local;

    // Phase 1: Each thread accumulates over its stripe of V_local
    float partial_sum = 0.0f;

    for (int64_t v = tid; v < V_local; v += blockSize) {
        float logit = row[v];
        float exp_val = expf(logit - max_val);
        partial_sum += exp_val;
        exp_logits[bt * V_local + v] = exp_val;
    }

    // Phase 2: Block reduction for sum_exp
    sdata[tid] = partial_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        packed_out[bt] = sdata[0];  // sum_exp
        
        // Phase 3: Direct O(1) target logit lookup
        float target_val = 0.0f;
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            target_val = row[target_idx - start_v];
        }
        packed_out[BT + bt] = target_val;  // target_logit
    }
    
    // Phase 4 (Optional): Compute and store local raw softmax numerator (exp)
    // We will divide by global sum_exp later in PyTorch/C++ or here if we have it.
    // Wait, the python code caches exp_logits / sum_exp. 
    // In our forward pass, we don't have global sum_exp yet (needs AllReduce).
    // So we can only store the exp(logit - max) part here, OR we do a second pass.
}

// Second pass kernel to compute final softmax probs IN-PLACE: probs = exp_logits / global_sum_exp
__global__ void compute_softmax_probs_kernel(
    float* __restrict__ exp_logits,        // [BT * V_local] (in-place)
    const float* __restrict__ global_sum,  // [BT]
    int64_t BT, int64_t V_local
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * V_local;
    if (idx < total) {
        int64_t bt = idx / V_local;
        exp_logits[idx] = exp_logits[idx] / global_sum[bt];
    }
}

void launch_compute_softmax_probs(
    float* exp_logits, const float* global_sum,
    int64_t B, int64_t T, int64_t V_local, cudaStream_t stream
) {
    int64_t total = B * T * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    compute_softmax_probs_kernel<<<blocks, threads, 0, stream>>>(
        exp_logits, global_sum, B * T, V_local
    );
}

// VECTORIZED (float4) Second pass kernel to compute final softmax probs IN-PLACE
__global__ void compute_softmax_probs_kernel_float4(
    float4* __restrict__ exp_logits_f4,     // [BT * (V_local/4)] (in-place)
    const float*  __restrict__ global_sum,  // [BT]
    int64_t BT, int64_t V_local_f4
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_f4 = BT * V_local_f4;
    
    if (idx < total_f4) {
        int64_t bt = idx / V_local_f4;
        
        float sum_val = global_sum[bt];
        float4 exp4 = exp_logits_f4[idx];
        
        float4 prob4;
        prob4.x = exp4.x / sum_val;
        prob4.y = exp4.y / sum_val;
        prob4.z = exp4.z / sum_val;
        prob4.w = exp4.w / sum_val;
        
        exp_logits_f4[idx] = prob4;
    }
}

void launch_compute_softmax_probs_float4(
    float* exp_logits, const float* global_sum,
    int64_t B, int64_t T, int64_t V_local, cudaStream_t stream
) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(exp_logits) % 16 == 0);

    if (V_local % 4 != 0 || !is_aligned) {
        launch_compute_softmax_probs(exp_logits, global_sum, B, T, V_local, stream);
        return;
    }
    
    int64_t V_local_f4 = V_local / 4;
    int64_t total_f4 = B * T * V_local_f4;
    int threads = 256;
    int blocks = (total_f4 + threads - 1) / threads;
    
    compute_softmax_probs_kernel_float4<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(exp_logits), global_sum, B * T, V_local_f4
    );
}

void launch_fused_sum_exp_target(
    const float* logits, const float* global_max, const float* targets,
    float* packed_out, float* exp_logits, int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, cudaStream_t stream
) {
    int64_t BT = B * T;
    int threads = 256;
    int shared_mem = threads * sizeof(float);

    fused_sum_exp_target_kernel<<<BT, threads, shared_mem, stream>>>(
        logits, global_max, targets, packed_out, exp_logits, BT, V_local, start_v
    );
}

// =============================================================================
// VECTORIZED (float4) Fused Forward: sum_exp + target_logit in ONE pass
// =============================================================================
__global__ void fused_sum_exp_target_kernel_float4(
    const float4* __restrict__ logits_f4,   // [BT * (V_local/4)]
    const float*  __restrict__ global_max,  // [BT]
    const float*  __restrict__ targets,     // [BT] as float
    float*        __restrict__ packed_out,  // [BT * 2]
    float4*       __restrict__ exp_logits_f4, // [BT * (V_local/4)]
    int64_t BT, int64_t V_local, int64_t start_v
) {
    extern __shared__ float sdata[];  // blockDim.x floats

    int64_t bt = blockIdx.x;
    if (bt >= BT) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = global_max[bt];
    int64_t target_idx = (int64_t)targets[bt];
    
    // Each row is V_local floats = (V_local/4) float4s
    int64_t V_local_f4 = V_local / 4;
    const float4* row_f4 = logits_f4 + bt * V_local_f4;

    float partial_sum = 0.0f;

    for (int64_t v = tid; v < V_local_f4; v += blockSize) {
        float4 logit4 = row_f4[v];
        
        float4 exp4;
        exp4.x = expf(logit4.x - max_val);
        exp4.y = expf(logit4.y - max_val);
        exp4.z = expf(logit4.z - max_val);
        exp4.w = expf(logit4.w - max_val);

        partial_sum += exp4.x + exp4.y + exp4.z + exp4.w;
        
        exp_logits_f4[bt * V_local_f4 + v] = exp4;
    }

    sdata[tid] = partial_sum;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        packed_out[bt] = sdata[0];
        
        // Direct O(1) target logit lookup
        float target_val = 0.0f;
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            const float* row_scalar = reinterpret_cast<const float*>(row_f4);
            target_val = row_scalar[target_idx - start_v];
        }
        packed_out[BT + bt] = target_val;
    }
}

void launch_fused_sum_exp_target_float4(
    const float* logits, const float* global_max, const float* targets,
    float* packed_out, float* exp_logits, int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, cudaStream_t stream
) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(logits) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(exp_logits) % 16 == 0);

    if (V_local % 4 != 0 || !is_aligned) {
        // Fallback to non-vectorized if vocab is not perfectly divisible by 4 or pointers unaligned
        launch_fused_sum_exp_target(logits, global_max, targets, packed_out, exp_logits, B, T, V_local, start_v, stream);
        return;
    }
    
    int64_t BT = B * T;
    int threads = 256;
    int shared_mem = threads * sizeof(float);

    fused_sum_exp_target_kernel_float4<<<BT, threads, shared_mem, stream>>>(
        reinterpret_cast<const float4*>(logits), global_max, targets, packed_out, 
        reinterpret_cast<float4*>(exp_logits), BT, V_local, start_v
    );
}

// =============================================================================
// Fused Backward: softmax + grad_scale + sparse_subtract in ONE kernel
// =============================================================================
// Reads grad_output from device pointer (no cudaMemcpy to host needed)
// Now uses precomputed softmax probabilities to avoid expf()

__global__ void vocab_parallel_ce_backward_kernel(
    const float* __restrict__ softmax_probs, // [BT * V_local]
    const float* __restrict__ targets,       // [BT] as float
    const float* __restrict__ grad_output,   // [1] scalar on GPU
    float* __restrict__ grad_logits,         // [BT * V_local] output
    int64_t BT, int64_t V_local,
    int64_t start_v, float scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * V_local;

    if (idx < total) {
        int64_t bt = idx / V_local;
        int64_t v  = idx % V_local;

        float g_out = grad_output[0] * scale;

        // Load precomputed softmax probability
        float prob = softmax_probs[idx];

        // grad = softmax * g_out
        float grad = prob * g_out;

        // Sparse subtract: if this vocab position is the target
        int64_t target_idx = (int64_t)targets[bt];
        if (target_idx >= start_v && target_idx < start_v + V_local) {
            if (v == (target_idx - start_v)) {
                grad -= g_out;
            }
        }

        grad_logits[idx] = grad;
    }
}

void launch_vocab_parallel_ce_backward(
    const float* softmax_probs, const float* targets,
    const float* grad_output, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, float scale, cudaStream_t stream
) {
    int64_t total = B * T * V_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_ce_backward_kernel<<<blocks, threads, 0, stream>>>(
        softmax_probs, targets, grad_output, grad_logits,
        B * T, V_local, start_v, scale
    );
}

// =============================================================================
// VECTORIZED (float4) Fused Backward: softmax + grad_scale + sparse_subtract
// =============================================================================
__global__ void vocab_parallel_ce_backward_kernel_float4(
    const float4* __restrict__ softmax_probs_f4, // [BT * (V_local/4)]
    const float*  __restrict__ targets,          // [BT] as float
    const float*  __restrict__ grad_output,      // [1] scalar on GPU
    float4*       __restrict__ grad_logits_f4,   // [BT * (V_local/4)]
    int64_t BT, int64_t V_local_f4,
    int64_t start_v, float scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_f4 = BT * V_local_f4;

    if (idx < total_f4) {
        int64_t bt = idx / V_local_f4;
        int64_t v_f4 = idx % V_local_f4;
        int64_t base_v = v_f4 * 4;

        float g_out = grad_output[0] * scale;
        float4 prob4 = softmax_probs_f4[idx];

        float4 grad4;
        grad4.x = prob4.x * g_out;
        grad4.y = prob4.y * g_out;
        grad4.z = prob4.z * g_out;
        grad4.w = prob4.w * g_out;

        int64_t target_idx = (int64_t)targets[bt];
        if (target_idx >= start_v && target_idx < start_v + (V_local_f4 * 4)) {
            int64_t target_local = target_idx - start_v;
            if (target_local == base_v + 0) grad4.x -= g_out;
            if (target_local == base_v + 1) grad4.y -= g_out;
            if (target_local == base_v + 2) grad4.z -= g_out;
            if (target_local == base_v + 3) grad4.w -= g_out;
        }

        grad_logits_f4[idx] = grad4;
    }
}

void launch_vocab_parallel_ce_backward_float4(
    const float* softmax_probs, const float* targets,
    const float* grad_output, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, float scale, cudaStream_t stream
) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(softmax_probs) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(grad_logits) % 16 == 0);

    if (V_local % 4 != 0 || !is_aligned) {
        launch_vocab_parallel_ce_backward(softmax_probs, targets, grad_output, grad_logits, B, T, V_local, start_v, scale, stream);
        return;
    }
    
    int64_t V_local_f4 = V_local / 4;
    int64_t total_f4 = B * T * V_local_f4;
    int threads = 256;
    int blocks = (total_f4 + threads - 1) / threads;

    vocab_parallel_ce_backward_kernel_float4<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float4*>(softmax_probs), targets, grad_output, 
        reinterpret_cast<float4*>(grad_logits), B * T, V_local_f4, start_v, scale
    );
}

// =============================================================================
// Fused Vocab-Parallel Embedding Kernels
// =============================================================================
// Forward: replaces 9 separate ops (type cast, compare, mask, subtract, mul, embedding, reshape, mul)
//          with a single kernel: check-in-shard → lookup-or-zero
// Backward: scatter-add upstream gradients into weight.grad at looked-up indices

__global__ void vocab_parallel_embedding_fwd_kernel(
    const int64_t* __restrict__ input,    // [B*T] token indices
    const float*   __restrict__ weight,   // [local_V, D] embedding table
    float*         __restrict__ output,   // [B*T*D] output embeddings
    int64_t BT, int64_t D,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D;

    if (idx < total) {
        int64_t bt = idx / D;
        int64_t d  = idx % D;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            output[idx] = weight[local_idx * D + d];
        } else {
            output[idx] = 0.0f;
        }
    }
}

void launch_vocab_parallel_embedding_fwd(
    const int64_t* input, const float* weight, float* output,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    int64_t total = B * T * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_embedding_fwd_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output, B * T, D, start_v, end_v
    );
}

__global__ void vocab_parallel_embedding_bwd_kernel(
    const int64_t* __restrict__ input,        // [B*T] token indices
    const float*   __restrict__ grad_output,  // [B*T*D] upstream grad
    float*         __restrict__ grad_weight,  // [local_V, D] weight grad (accumulate)
    int64_t BT, int64_t D,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D;

    if (idx < total) {
        int64_t bt = idx / D;
        int64_t d  = idx % D;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            atomicAdd(&grad_weight[local_idx * D + d], grad_output[idx]);
        }
    }
}

void launch_vocab_parallel_embedding_bwd(
    const int64_t* input, const float* grad_output, float* grad_weight,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    int64_t total = B * T * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    vocab_parallel_embedding_bwd_kernel<<<blocks, threads, 0, stream>>>(
        input, grad_output, grad_weight, B * T, D, start_v, end_v
    );
}

// =============================================================================
// Vectorized (float4) Vocab-Parallel Embedding Kernels
// =============================================================================

__global__ void vocab_parallel_embedding_fwd_kernel_float4(
    const int64_t* __restrict__ input,        // [B*T] token indices
    const float4*  __restrict__ weight_f4,    // [local_V, D/4] embedding table
    float4*        __restrict__ output_f4,    // [B*T, D/4] output embeddings
    int64_t BT, int64_t D_f4,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D_f4;

    if (idx < total) {
        int64_t bt = idx / D_f4;
        int64_t d_f4  = idx % D_f4;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            output_f4[idx] = weight_f4[local_idx * D_f4 + d_f4];
        } else {
            float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
            output_f4[idx] = zeros;
        }
    }
}

void launch_vocab_parallel_embedding_fwd_float4(
    const int64_t* input, const float* weight, float* output,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(weight) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(output) % 16 == 0);

    if (D % 4 != 0 || !is_aligned) {
        launch_vocab_parallel_embedding_fwd(input, weight, output, B, T, D, start_v, end_v, stream);
        return;
    }
    
    int64_t D_f4 = D / 4;
    int64_t total_f4 = B * T * D_f4;
    int threads = 256;
    int blocks = (total_f4 + threads - 1) / threads;

    vocab_parallel_embedding_fwd_kernel_float4<<<blocks, threads, 0, stream>>>(
        input, reinterpret_cast<const float4*>(weight), reinterpret_cast<float4*>(output), B * T, D_f4, start_v, end_v
    );
}

__global__ void vocab_parallel_embedding_bwd_kernel_float4(
    const int64_t* __restrict__ input,           // [B*T] token indices
    const float4*  __restrict__ grad_output_f4,  // [B*T, D/4] upstream grad
    float*         __restrict__ grad_weight,     // [local_V, D] weight grad (scalar)
    int64_t BT, int64_t D_f4,
    int64_t start_v, int64_t end_v
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = BT * D_f4;

    if (idx < total) {
        int64_t bt = idx / D_f4;
        int64_t d_f4  = idx % D_f4;

        int64_t token = input[bt];

        if (token >= start_v && token < end_v) {
            int64_t local_idx = token - start_v;
            
            float4 g4 = grad_output_f4[idx];
            int64_t base_d = d_f4 * 4;
            int64_t weight_base_idx = local_idx * (D_f4 * 4) + base_d;
            
            // Four scalar atomic adds since atomicAdd on float4 isn't supported natively
            atomicAdd(&grad_weight[weight_base_idx + 0], g4.x);
            atomicAdd(&grad_weight[weight_base_idx + 1], g4.y);
            atomicAdd(&grad_weight[weight_base_idx + 2], g4.z);
            atomicAdd(&grad_weight[weight_base_idx + 3], g4.w);
        }
    }
}

void launch_vocab_parallel_embedding_bwd_float4(
    const int64_t* input, const float* grad_output, float* grad_weight,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream
) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(grad_output) % 16 == 0);

    if (D % 4 != 0 || !is_aligned) {
        launch_vocab_parallel_embedding_bwd(input, grad_output, grad_weight, B, T, D, start_v, end_v, stream);
        return;
    }
    
    int64_t D_f4 = D / 4;
    int64_t total_f4 = B * T * D_f4;
    int threads = 256;
    int blocks = (total_f4 + threads - 1) / threads;

    vocab_parallel_embedding_bwd_kernel_float4<<<blocks, threads, 0, stream>>>(
        input, reinterpret_cast<const float4*>(grad_output), grad_weight, B * T, D_f4, start_v, end_v
    );
}

} // namespace cuda
} // namespace OwnTensor
