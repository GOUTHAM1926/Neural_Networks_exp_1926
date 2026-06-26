#include "ops/helpers/MultiTensorKernels.h"
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include <device_launch_parameters.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace OwnTensor { namespace cuda {

// =============================================================================
// PYTORCH-LIKE MULTI-TENSOR METADATA & CONSTANTS
// =============================================================================

// CHUNK_SIZE refers to how many float elements a single block processes.
// 512 threads * 4 floats (float4) * 16 unrolled loop steps = 32768 floats
static const int CHUNK_SIZE = 32768;
static const int MAX_BLOCKS_PER_LAUNCH = 320;
static const int MAX_TENSORS_PER_LAUNCH = 48;

struct AdamLaunchMetadata {
    float* params[MAX_TENSORS_PER_LAUNCH];
    float* grads[MAX_TENSORS_PER_LAUNCH];
    float* ms[MAX_TENSORS_PER_LAUNCH];
    float* vs[MAX_TENSORS_PER_LAUNCH];
    int64_t sizes[MAX_TENSORS_PER_LAUNCH];
    unsigned char block_to_tensor[MAX_BLOCKS_PER_LAUNCH];
    int block_to_chunk[MAX_BLOCKS_PER_LAUNCH];
};

struct ScaleLaunchMetadata {
    float* tensors[MAX_TENSORS_PER_LAUNCH];
    int64_t sizes[MAX_TENSORS_PER_LAUNCH];
    unsigned char block_to_tensor[MAX_BLOCKS_PER_LAUNCH];
    int block_to_chunk[MAX_BLOCKS_PER_LAUNCH];
};

// Byte-based metadata — the zero kernel is dtype-agnostic. `tensors` holds
// raw void pointers; `nbytes` is the total byte count per tensor.
struct ZeroLaunchMetadata {
    void*   tensors[MAX_TENSORS_PER_LAUNCH];
    int64_t nbytes[MAX_TENSORS_PER_LAUNCH];
    unsigned char block_to_tensor[MAX_BLOCKS_PER_LAUNCH];
    int block_to_chunk[MAX_BLOCKS_PER_LAUNCH];
};

}} // namespace OwnTensor::cuda

// =============================================================================
// FORWARD DECLARATIONS — sm86 fallbacks
// =============================================================================
namespace OwnTensor { namespace cuda {
void multi_tensor_grad_norm_cuda(const std::vector<TensorInfo>&, float*);
void multi_tensor_scale_cuda(const std::vector<TensorInfo>&, const float*);
void multi_tensor_adam_cuda(
    const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
    const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
    float, float, float, float, float, float, float, bool);
}}

namespace OwnTensor { namespace cuda {

// =============================================================================
// MULTI-TENSOR L2 NORM / SCALE  (Ada Lovelace — sm89)
// =============================================================================

void multi_tensor_grad_norm_sm89_cuda(
    const std::vector<TensorInfo>& tensors, float* norm_sq_accumulator) {
    multi_tensor_grad_norm_cuda(tensors, norm_sq_accumulator);
}

__global__ void __launch_bounds__(256, 2) multi_tensor_scale_sm89_kernel(
    ScaleLaunchMetadata meta,
    const float* clip_coef
) {
    float scale = *clip_coef;
    if (scale >= 1.0f) return;

    int loc_block_idx = blockIdx.x;
    if (loc_block_idx >= MAX_BLOCKS_PER_LAUNCH) return;

    int tensor_idx = meta.block_to_tensor[loc_block_idx];
    int chunk_idx = meta.block_to_chunk[loc_block_idx];
    int64_t numel = meta.sizes[tensor_idx];

    int64_t start = (int64_t)chunk_idx * CHUNK_SIZE;
    int64_t end = start + CHUNK_SIZE;
    if (end > numel) end = numel;

    float* p = meta.tensors[tensor_idx];

    int64_t vec_start = (start + 3) / 4 * 4;
    int64_t vec_end   = end / 4 * 4;
    if (vec_start > vec_end) vec_start = vec_end;

    // Scalar head
    for (int64_t i = start + threadIdx.x; i < vec_start; i += blockDim.x) {
        p[i] *= scale;
    }

    // Vectorized main loop
    for (int64_t i = vec_start + threadIdx.x * 4; i < vec_end; i += (int64_t)blockDim.x * 4) {
        float4* p4 = (float4*)(&p[i]);
        float4 vec = *p4;
        vec.x *= scale;
        vec.y *= scale;
        vec.z *= scale;
        vec.w *= scale;
        *p4 = vec;
    }

    // Scalar tail
    int64_t tail_start = max(start, vec_end);
    for (int64_t i = tail_start + threadIdx.x; i < end; i += blockDim.x) {
        p[i] *= scale;
    }
}

void multi_tensor_scale_sm89_cuda(
    const std::vector<TensorInfo>& tensors, const float* clip_coef) {
    if (tensors.empty()) return;

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    int total_tensors = tensors.size();
    int tensor_idx = 0;
    int chunk_idx = 0;

    // Process all tensors efficiently looping over launch limits
    while (tensor_idx < total_tensors) {
        int loc_tensor_idx = 0;
        int loc_block_idx  = 0;
        ScaleLaunchMetadata meta;
        
        while (tensor_idx < total_tensors && 
            loc_tensor_idx < MAX_TENSORS_PER_LAUNCH && 
            loc_block_idx < MAX_BLOCKS_PER_LAUNCH) {
            
            int64_t numel = tensors[tensor_idx].numel;
            int64_t chunks_for_tensor = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            int chunks_remaining = chunks_for_tensor - chunk_idx;
            
            meta.tensors[loc_tensor_idx] = tensors[tensor_idx].ptr;
            meta.sizes[loc_tensor_idx]  = numel;
            
            int chunks_to_add = std::min(chunks_remaining, MAX_BLOCKS_PER_LAUNCH - loc_block_idx);
            for (int c = 0; c < chunks_to_add; c++) {
                meta.block_to_tensor[loc_block_idx] = loc_tensor_idx;
                meta.block_to_chunk[loc_block_idx] = chunk_idx + c;
                loc_block_idx++;
            }
            
            chunk_idx += chunks_to_add;
            
            if (chunk_idx >= chunks_for_tensor) {
                tensor_idx++;
                chunk_idx = 0;
            }
            loc_tensor_idx++;
        }
        
        // Single kernel launch to process up to 320 blocks 
        if (loc_block_idx > 0) {
            multi_tensor_scale_sm89_kernel<<<loc_block_idx, 256, 0, stream>>>(
                meta, clip_coef
            );
        }
    }
}

// =============================================================================
// UNIFIED VECTORIZED ADAM KERNEL — DIRECT STRUCT METADATA (O(1) OVERHEAD)
// =============================================================================
// No `cudaMemcpyAsync` API calls for pointer transfers.
// Metadata struct `AdamLaunchMetadata` passed directly by value to kernel.
// Distributes tensor chunks specifically over multiple blocks for high occupancy.
__global__ void __launch_bounds__(256, 2) multi_tensor_adam_sm89_kernel(
    AdamLaunchMetadata meta,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    int loc_block_idx = blockIdx.x;
    if (loc_block_idx >= MAX_BLOCKS_PER_LAUNCH) return;

    int tensor_idx = meta.block_to_tensor[loc_block_idx];
    int chunk_idx = meta.block_to_chunk[loc_block_idx];
    int64_t numel = meta.sizes[tensor_idx];

    int64_t start = (int64_t)chunk_idx * CHUNK_SIZE;
    int64_t end = start + CHUNK_SIZE;
    if (end > numel) end = numel;

    float* p = meta.params[tensor_idx];
    float* g = meta.grads[tensor_idx];
    float* m = meta.ms[tensor_idx];
    float* v = meta.vs[tensor_idx];

    // Compute float4-aligned vector boundaries
    int64_t vec_start = (start + 3) / 4 * 4;
    int64_t vec_end   = end / 4 * 4;
    if (vec_start > vec_end) vec_start = vec_end; // In case chunk is extremely tiny

    // Scalar head (if the overall data start doesn't align to float4, extremely rare since PyTorch pools align to 16-8 bytes usually)
    for (int64_t i = start + threadIdx.x; i < vec_start; i += blockDim.x) {
        float gi = g[i];
        float pi = p[i];
        float mi = m[i];
        float vi = v[i];
        
        float m_new = fmaf(beta1, mi, (1.0f - beta1) * gi);
        float v_new = fmaf(beta2, vi, (1.0f - beta2) * gi * gi);
        m[i] = m_new;
        v[i] = v_new;
        // PyTorch update: p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p).
        // eps MUST be inside the denominator — when v_hat = 0 (a sparse
        // embedding row with no token hits this step), rsqrt(0) = +Inf and
        // 0 * Inf = NaN. With eps in the denominator, v_hat=0 → denom=eps,
        // update=0, no poisoning.
        float denom = sqrtf(v_new / bias_correction2) + eps;
        p[i] = pi - lr * ((m_new / bias_correction1) / denom + weight_decay * pi);
    }

    // Vectorized main loop
    for (int64_t i = vec_start + threadIdx.x * 4; i < vec_end; i += (int64_t)blockDim.x * 4) {
        float4* g4 = (float4*)(&g[i]);
        float4* p4 = (float4*)(&p[i]);
        float4* m4 = (float4*)(&m[i]);
        float4* v4 = (float4*)(&v[i]);
        
        float4 g_vec = *g4;
        float4 p_vec = *p4;
        float4 m_vec = *m4;
        float4 v_vec = *v4;

        float4 m_out, v_out, p_out;

#define ADAM_UPDATE(gj, pj, mj, vj, m_out_j, v_out_j, p_out_j)          \
        m_out_j = fmaf(beta1, mj, (1.0f - beta1) * gj);                 \
        v_out_j = fmaf(beta2, vj, (1.0f - beta2) * gj * gj);            \
        p_out_j = pj - lr * ((m_out_j / bias_correction1) /             \
                  (sqrtf(v_out_j / bias_correction2) + eps)             \
                  + weight_decay * pj);

        ADAM_UPDATE(g_vec.x, p_vec.x, m_vec.x, v_vec.x, m_out.x, v_out.x, p_out.x)
        ADAM_UPDATE(g_vec.y, p_vec.y, m_vec.y, v_vec.y, m_out.y, v_out.y, p_out.y)
        ADAM_UPDATE(g_vec.z, p_vec.z, m_vec.z, v_vec.z, m_out.z, v_out.z, p_out.z)
        ADAM_UPDATE(g_vec.w, p_vec.w, m_vec.w, v_vec.w, m_out.w, v_out.w, p_out.w)
#undef ADAM_UPDATE

        *m4 = m_out;
        *v4 = v_out;
        *p4 = p_out;
    }

    // Scalar tail
    int64_t tail_start = max(start, vec_end);
    for (int64_t i = tail_start + threadIdx.x; i < end; i += blockDim.x) {
        float gi = g[i];
        float pi = p[i];
        float mi = m[i];
        float vi = v[i];
        
        float m_new = fmaf(beta1, mi, (1.0f - beta1) * gi);
        float v_new = fmaf(beta2, vi, (1.0f - beta2) * gi * gi);
        m[i] = m_new;
        v[i] = v_new;
        float denom = sqrtf(v_new / bias_correction2) + eps;
        p[i] = pi - lr * ((m_new / bias_correction1) / denom + weight_decay * pi);
    }
}

// =============================================================================
// HOST FUNCTION
// =============================================================================

void multi_tensor_adam_sm89_cuda(
    const std::vector<TensorInfo>& params,
    const std::vector<TensorInfo>& grads,
    const std::vector<TensorInfo>& ms,
    const std::vector<TensorInfo>& vs,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, bool is_adamw
) {
    if (params.empty()) return;

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    int total_tensors = params.size();
    int tensor_idx = 0;
    int chunk_idx = 0;

    // Process all tensors efficiently looping over launch limits
    while (tensor_idx < total_tensors) {
        int loc_tensor_idx = 0;
        int loc_block_idx  = 0;
        AdamLaunchMetadata meta;
        
        while (tensor_idx < total_tensors && 
               loc_tensor_idx < MAX_TENSORS_PER_LAUNCH && 
               loc_block_idx < MAX_BLOCKS_PER_LAUNCH) {
            
            int64_t numel = params[tensor_idx].numel;
            int64_t chunks_for_tensor = (numel + CHUNK_SIZE - 1) / CHUNK_SIZE;
            int chunks_remaining = chunks_for_tensor - chunk_idx;
            
            // Register pointer/size arrays exactly once per partitioned tensor per kernel launch
            meta.params[loc_tensor_idx] = params[tensor_idx].ptr;
            meta.grads[loc_tensor_idx]  = grads[tensor_idx].ptr;
            meta.ms[loc_tensor_idx]     = ms[tensor_idx].ptr;
            meta.vs[loc_tensor_idx]     = vs[tensor_idx].ptr;
            meta.sizes[loc_tensor_idx]  = numel;
            
            int chunks_to_add = std::min(chunks_remaining, MAX_BLOCKS_PER_LAUNCH - loc_block_idx);
            for (int c = 0; c < chunks_to_add; c++) {
                meta.block_to_tensor[loc_block_idx] = loc_tensor_idx;
                meta.block_to_chunk[loc_block_idx] = chunk_idx + c;
                loc_block_idx++;
            }
            
            chunk_idx += chunks_to_add;
            
            if (chunk_idx >= chunks_for_tensor) {
                // Tensor fully distributed, proceed to next
                tensor_idx++;
                chunk_idx = 0;
            }
            loc_tensor_idx++;
        }
        
        // Single kernel launch to process up to 320 blocks 
        if (loc_block_idx > 0) {
            multi_tensor_adam_sm89_kernel<<<loc_block_idx, 256, 0, stream>>>(
                meta, lr, beta1, beta2, eps, weight_decay, 
                bias_correction1, bias_correction2
            );
        }
    }
}

// =============================================================================
// MULTI-TENSOR ZERO KERNEL (Ada Lovelace — sm89). Dtype-agnostic.
// Operates on raw byte buffers; writes zero via uint4 stores (STG.128 — 16B
// per instruction). Supports ANY numeric dtype because bit-pattern 0 == value 0
// for every floating, integer, and complex type we use.
//
// Chunk size is in BYTES: 131,072 B = 32K × 4 B = same total work as the
// Adam/Scale kernels (which operate on 32K floats per chunk).
// =============================================================================
static const int64_t ZERO_CHUNK_BYTES = 131072;   // 128 KB per chunk

__global__ void __launch_bounds__(256, 2) multi_tensor_zero_sm89_kernel(
    ZeroLaunchMetadata meta
) {
    int loc_block_idx = blockIdx.x;
    if (loc_block_idx >= MAX_BLOCKS_PER_LAUNCH) return;

    int tensor_idx = meta.block_to_tensor[loc_block_idx];
    int chunk_idx  = meta.block_to_chunk[loc_block_idx];
    int64_t nbytes = meta.nbytes[tensor_idx];

    int64_t start = (int64_t)chunk_idx * ZERO_CHUNK_BYTES;
    int64_t end   = start + ZERO_CHUNK_BYTES;
    if (end > nbytes) end = nbytes;

    unsigned char* p = reinterpret_cast<unsigned char*>(meta.tensors[tensor_idx]);

    // 16-byte aligned span inside [start, end)
    int64_t vec_start = (start + 15) / 16 * 16;
    int64_t vec_end   = end / 16 * 16;
    if (vec_start > vec_end) vec_start = vec_end;

    // Scalar head — unaligned leading bytes
    for (int64_t i = start + threadIdx.x; i < vec_start; i += blockDim.x) {
        p[i] = 0;
    }

    // Vectorized main loop — uint4 store (16 zero bytes per thread)
    uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    for (int64_t i = vec_start + (int64_t)threadIdx.x * 16;
         i < vec_end;
         i += (int64_t)blockDim.x * 16) {
        *reinterpret_cast<uint4*>(&p[i]) = z4;
    }

    // Scalar tail — bytes beyond vec_end
    int64_t tail_start = max(start, vec_end);
    for (int64_t i = tail_start + threadIdx.x; i < end; i += blockDim.x) {
        p[i] = 0;
    }
}

void multi_tensor_zero_sm89_cuda(const std::vector<ZeroTensorInfo>& tensors) {
    if (tensors.empty()) return;

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    int total_tensors = tensors.size();
    int tensor_idx = 0;
    int chunk_idx  = 0;

    while (tensor_idx < total_tensors) {
        int loc_tensor_idx = 0;
        int loc_block_idx  = 0;
        ZeroLaunchMetadata meta;

        while (tensor_idx < total_tensors &&
               loc_tensor_idx < MAX_TENSORS_PER_LAUNCH &&
               loc_block_idx  < MAX_BLOCKS_PER_LAUNCH) {

            int64_t nbytes = tensors[tensor_idx].nbytes;
            int64_t chunks_for_tensor = (nbytes + ZERO_CHUNK_BYTES - 1) / ZERO_CHUNK_BYTES;
            int chunks_remaining = (int)(chunks_for_tensor - chunk_idx);

            meta.tensors[loc_tensor_idx] = tensors[tensor_idx].ptr;
            meta.nbytes[loc_tensor_idx]  = nbytes;

            int chunks_to_add = std::min(chunks_remaining, MAX_BLOCKS_PER_LAUNCH - loc_block_idx);
            for (int c = 0; c < chunks_to_add; c++) {
                meta.block_to_tensor[loc_block_idx] = loc_tensor_idx;
                meta.block_to_chunk[loc_block_idx]  = chunk_idx + c;
                loc_block_idx++;
            }

            chunk_idx += chunks_to_add;

            if (chunk_idx >= chunks_for_tensor) {
                tensor_idx++;
                chunk_idx = 0;
            }
            loc_tensor_idx++;
        }

        if (loc_block_idx > 0) {
            multi_tensor_zero_sm89_kernel<<<loc_block_idx, 256, 0, stream>>>(meta);
        }
    }
}

} // namespace cuda
} // namespace OwnTensor