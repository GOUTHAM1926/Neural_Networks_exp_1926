#include "ops/helpers/EmbeddingKernels.h"
#include "dtype/CudaTraits.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>

namespace OwnTensor {
namespace cuda {

// =============================================================================
// OPTIMIZED EMBEDDING FORWARD KERNEL
// Each thread processes one element of the embedding vector
// This allows for coalesced memory access
// Supports strided weight tensors (e.g., transposed views for weight tying)
// =============================================================================

template<typename T>
__global__ void __launch_bounds__(256) embedding_forward_kernel_optimized(
    const uint16_t* __restrict__ indices,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int64_t N,      // Number of indices (batch * seq_len)
    int64_t C,      // Embedding dimension
    int64_t V,      // Vocabulary size
    int padding_idx,
    int64_t weight_stride_row,  // stride between rows (tokens) in weight
    int64_t weight_stride_col   // stride between cols (embed dims) in weight
) {
    // 2D thread indexing: x = embedding element, y = index position
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;  // embedding dimension
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;  // index position

    if (n >= N || c >= C) return;

    uint16_t tok = indices[n];

    if (tok == (uint16_t)padding_idx) {
        output[n * C + c] = T{};
        return;
    }

    if (tok >= V) {
        return;
    }

    output[n * C + c] = weight[(size_t)tok * weight_stride_row + c * weight_stride_col];
}

// -----------------------------------------------------------------------------
// VECTORIZED FORWARD KERNEL (128-bit loads/stores + Broadcast + ILP=4)
// -----------------------------------------------------------------------------
// ILP Factor: 4 (Process 4 float4s per thread -> 64 bytes)
// This aggressively hides memory latency and maximizes bandwidth.
__global__ void __launch_bounds__(256) embedding_forward_kernel_vectorized(
    const uint16_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx
) {
    // Each thread handles ILP=4 float4 elements (64 bytes total)
    // Coalescing: Threads 0-31 handle contiguous memory.
    
    // Stride for the block: blockDim.x * ILP = 32 * 4 = 128 elements per block-row
    int64_t c_vec_start = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    
    int64_t C_vec = C / 4;
    
    if (n >= N) return;

    // WARP BROADCAST OPTIMIZATION
    // Only lane 0 loads the index and broadcasts it.
    // Optimization: Reduces index loads by 32x.
    uint16_t tok;
    if (threadIdx.x == 0) {
        // Use __ldg for read-only cache hint
        tok = __ldg(&indices[n]);
    }
    // Broadcast from lane 0
    tok = __shfl_sync(0xffffffff, tok, 0);
    
    // Reinterpret cast output pointers (float4*)
    float4* out_ptr = reinterpret_cast<float4*>(output);
    const float4* weight_ptr = reinterpret_cast<const float4*>(weight);

    // Padding Handling
    if (tok == (uint16_t)padding_idx) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int64_t c_vec = c_vec_start + i * 32; // Stride by warp size
            if (c_vec < C_vec) {
                out_ptr[n * C_vec + c_vec] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }
    
    if (tok >= V) return;
    
    // Main Load/Store Loop with ILP=4
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t c_vec = c_vec_start + i * 32;
        if (c_vec < C_vec) {
            // Use __ldg for weight read-only cache
            float4 w_val = __ldg(&weight_ptr[(size_t)tok * C_vec + c_vec]);
            out_ptr[n * C_vec + c_vec] = w_val;
        }
    }
}

// =============================================================================
// OPTIMIZED EMBEDDING BACKWARD KERNEL (FALLBACK)
// Uses atomicAdd — kept for strided (non-contiguous) weight tensors only.
// =============================================================================

template<typename T>
__global__ void __launch_bounds__(256) embedding_backward_kernel_optimized(
    const uint16_t* __restrict__ indices,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_weight,
    int64_t N,
    int64_t C,
    int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    // 2D thread indexing
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || c >= C) return;

    uint16_t tok = indices[n];
    if (tok == (uint16_t)padding_idx || tok >= V) return;

    T grad = grad_output[n * C + c];
    atomicAdd(&grad_weight[(size_t)tok * grad_weight_stride_row + c * grad_weight_stride_col], grad);
}

// =============================================================================
// COOPERATIVE EMBEDDING BACKWARD KERNEL (PRIMARY)
// =============================================================================
// Inspired by PyTorch's embedding_backward_feature_kernel.
//
// PROBLEM SOLVED:
//   The previous vectorized backward kernel decomposed each float4 load into
//   4 scalar atomicAdd calls.  When duplicate tokens appear in a batch (very
//   common: "the", padding, etc.), those atomics serialize on the same global
//   memory address, causing WORSE throughput than the plain scalar kernel.
//
// SOLUTION:
//   Eliminate atomicAdd entirely via cooperative shared-memory accumulation:
//     1. The entire block loads a batch of (blockDim.x * blockDim.y) indices
//        into shared memory.
//     2. We process the batch in chunks of blockDim.y (one warp per index row).
//     3. Each warp loads its gradient row into shared-memory accumulators.
//     4. __ballot_sync detects which warps in the chunk target the SAME
//        grad_weight row.
//     5. The lowest-indexed matching warp is elected "leader".  The leader
//        serially accumulates all matching warps' shared-memory data.
//     6. The leader performs a SINGLE non-atomic write to grad_weight.
//
//   This converts O(N) atomic global writes → O(unique_targets) plain writes.
//   For batches with repeated tokens the speedup can be substantial.
//
// THREAD LAYOUT:
//   blockDim = (WARP_SIZE, BLOCKDIMY)  e.g. (32, 32) → 1024 threads/block
//   gridDim.x = ceil(C / (WARP_SIZE * SZ))   — covers the embedding dimension
//   gridDim.y = 1  (the kernel loops over all N internally)
//
// GRAIN SIZE (ILP):
//   SZ = 4  →  each thread handles 4 elements of the embedding dimension,
//   spaced WARP_SIZE apart, to hide memory latency and maximise ILP.
// =============================================================================

static constexpr int BLOCKDIMY = 32;

__global__ void __launch_bounds__(1024) embedding_backward_kernel_cooperative(
    const uint16_t* __restrict__ indices,
    const float*    __restrict__ grad_output,
    float*          __restrict__ grad_weight,
    int64_t N,      // total number of indices
    int64_t C,      // embedding dimension (== stride)
    int64_t V,      // vocabulary size
    int     padding_idx
) {
    // --- Shared memory layout ---
    // [0 .. WARP_SIZE*BLOCKDIMY)  : float accumulators (one row per warp)
    // [WARP_SIZE*BLOCKDIMY .. +WARP_SIZE*BLOCKDIMY) : int indices_batch
    extern __shared__ char buf[];
    float* smem       = reinterpret_cast<float*>(buf);
    float* my_s       = smem + 32 * threadIdx.y;          // this warp's accumulator row
    int*  indices_batch = reinterpret_cast<int*>(
        buf + sizeof(float) * 32 * BLOCKDIMY);

    const int s = static_cast<int>(C);     // stride (embed_dim)
    const int SZ = 4;                       // grain size / ILP factor

    // Feature-dimension offset for this thread
    const int f = threadIdx.x + blockIdx.x * blockDim.x * SZ;

    // --- Outer loop: process ALL N indices in batches of (blockDim.x * blockDim.y) ---
    for (int batch_start = 0; batch_start < static_cast<int>(N);
         batch_start += blockDim.x * BLOCKDIMY)
    {
        // ---- Step 1: Cooperatively load a batch of indices into shared memory ----
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        if (batch_start + tid < static_cast<int>(N)) {
            indices_batch[tid] = static_cast<int>(indices[batch_start + tid]);
        }

        int batch_end = min(batch_start + blockDim.x * BLOCKDIMY, static_cast<int>(N));

        // ---- Step 2: Process the batch in chunks of BLOCKDIMY (32) ----
        for (int chunk_start = batch_start; chunk_start < batch_end;
             chunk_start += BLOCKDIMY)
        {
            // Sync: ensures indices_batch is ready AND that leader warps
            // from the previous chunk are done with their accumulates.
            __syncthreads();

            int n_this_chunk = min(batch_end - chunk_start, BLOCKDIMY);

            int src_row = chunk_start + threadIdx.y;    // global index pos
            int dst_row = -1;
            if (src_row < static_cast<int>(N)) {
                dst_row = indices_batch[src_row - batch_start];
            }

            // ---- Step 3: Each warp loads its gradient row into shared mem ----
            // ILP = SZ: process SZ elements per thread, spaced by warp size
            #pragma unroll
            for (int ii = 0; ii < SZ; ++ii) {
                int feature_dim = f + ii * 32;  // 32 = warp size
                if (src_row < static_cast<int>(N) && feature_dim < s
                    && dst_row != padding_idx && dst_row >= 0 && dst_row < static_cast<int>(V)) {
                    my_s[threadIdx.x] = grad_output[src_row * C + feature_dim];
                } else {
                    my_s[threadIdx.x] = 0.0f;
                }

                __syncthreads();

                // ---- Step 4 & 5: Detect duplicates, elect leader, accumulate ----
                if (dst_row != padding_idx && dst_row >= 0
                    && dst_row < static_cast<int>(V) && src_row < static_cast<int>(N))
                {
                    // Ballot: which warps in this chunk target the same dst_row?
                    int match_found_this_thread = 0;
                    if (threadIdx.x < static_cast<unsigned>(n_this_chunk)) {
                        match_found_this_thread =
                            (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
                    }

                    unsigned int matchmask = __ballot_sync(0xffffffff, match_found_this_thread);
                    int first_remaining_peer = __ffs(matchmask) - 1;

                    // Only the leader warp (lowest matching index) does the work
                    if (static_cast<int>(threadIdx.y) == first_remaining_peer) {
                        // Remove leader from mask
                        matchmask ^= (1u << first_remaining_peer);

                        // Accumulate from all other matching warps
                        while (matchmask) {
                            first_remaining_peer = __ffs(matchmask) - 1;
                            my_s[threadIdx.x] +=
                                smem[threadIdx.x + 32 * first_remaining_peer];
                            matchmask ^= (1u << first_remaining_peer);
                        }

                        // ---- Step 6: Single non-atomic write to global ----
                        if (feature_dim < s) {
                            grad_weight[dst_row * C + feature_dim] +=
                                my_s[threadIdx.x];
                        }
                    }
                }

                __syncthreads();
            }
        }
    }
}


template<typename T>
static void launch_embedding_forward(
    const uint16_t* indices, const T* weight, T* output,
    int64_t N, int64_t C, int64_t V, int padding_idx,
    int64_t weight_stride_row, int64_t weight_stride_col
) {
    dim3 block(32, 8);
    dim3 grid(
        (C + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    embedding_forward_kernel_optimized<T><<<grid, block>>>(
        indices, weight, output, N, C, V, padding_idx,
        weight_stride_row, weight_stride_col
    );
}

// Float32 — uses vectorized kernel when possible
void embedding_forward_cuda(
    const uint16_t* indices,
    const float* weight,
    float* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    bool can_vectorize = (C % 4 == 0) &&
                         (weight_stride_col == 1) &&
                         (weight_stride_row == C) &&
                         ((uintptr_t)weight % 16 == 0) &&
                         ((uintptr_t)output % 16 == 0);

    if (can_vectorize) {
        int64_t C_vec = C / 4;
        const int ILP = 4;
        dim3 block(32, 8);
        dim3 grid(
            (C_vec + (block.x * ILP) - 1) / (block.x * ILP),
            (N + block.y - 1) / block.y
        );
        embedding_forward_kernel_vectorized<<<grid, block>>>(
            indices, weight, output, N, C, V, padding_idx
        );
    } else {
        launch_embedding_forward<float>(indices, weight, output, N, C, V, padding_idx,
                                        weight_stride_row, weight_stride_col);
    }
}

// Float16
void embedding_forward_cuda(
    const uint16_t* indices,
    const float16_t* weight,
    float16_t* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    launch_embedding_forward<__half>(indices,
        reinterpret_cast<const __half*>(weight), reinterpret_cast<__half*>(output),
        N, C, V, padding_idx, weight_stride_row, weight_stride_col);
}

// Bfloat16
void embedding_forward_cuda(
    const uint16_t* indices,
    const bfloat16_t* weight,
    bfloat16_t* output,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    launch_embedding_forward<__nv_bfloat16>(indices,
        reinterpret_cast<const __nv_bfloat16*>(weight), reinterpret_cast<__nv_bfloat16*>(output),
        N, C, V, padding_idx, weight_stride_row, weight_stride_col);
}

// Helper to launch the templated backward kernel
template<typename T>
static void launch_embedding_backward(
    const uint16_t* indices,
    const T* grad_output,
    T* grad_weight,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    dim3 block(32, 8);
    dim3 grid(
        (C + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );
    embedding_backward_kernel_optimized<T><<<grid, block>>>(
        indices, grad_output, grad_weight, N, C, V, padding_idx,
        grad_weight_stride_row, grad_weight_stride_col
    );
}

// Float32
void embedding_backward_cuda(
    const uint16_t* indices,
    const float* grad_output,
    float* grad_weight,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    launch_embedding_backward<float>(indices, grad_output, grad_weight,
                                     N, C, V, padding_idx,
                                     grad_weight_stride_row, grad_weight_stride_col);
}

// Float16
void embedding_backward_cuda(
    const uint16_t* indices,
    const float16_t* grad_output,
    float16_t* grad_weight,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    launch_embedding_backward<__half>(indices,
                                      reinterpret_cast<const __half*>(grad_output),
                                      reinterpret_cast<__half*>(grad_weight),
                                      N, C, V, padding_idx,
                                      grad_weight_stride_row, grad_weight_stride_col);
}

// Bfloat16
void embedding_backward_cuda(
    const uint16_t* indices,
    const bfloat16_t* grad_output,
    bfloat16_t* grad_weight,
    int64_t N, int64_t C, int64_t V,
    int padding_idx,
    int64_t grad_weight_stride_row,
    int64_t grad_weight_stride_col
) {
    launch_embedding_backward<__nv_bfloat16>(indices,
                                              reinterpret_cast<const __nv_bfloat16*>(grad_output),
                                              reinterpret_cast<__nv_bfloat16*>(grad_weight),
                                              N, C, V, padding_idx,
                                              grad_weight_stride_row, grad_weight_stride_col);
}

} // namespace cuda
} // namespace OwnTensor