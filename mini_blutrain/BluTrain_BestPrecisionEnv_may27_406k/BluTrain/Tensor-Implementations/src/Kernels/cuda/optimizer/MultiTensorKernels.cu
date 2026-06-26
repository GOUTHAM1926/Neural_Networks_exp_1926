#include "ops/helpers/MultiTensorKernels.h"
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#include <device_launch_parameters.h>
#include <cstring>
#include <algorithm>
#include "ops/helpers/KernelDispatch.h"

namespace OwnTensor {
namespace cuda {

// Persistent GPU buffer for metadata to avoid frequent allocations
// 512 tensors is enough for GPT-2 models
static const size_t MAX_MULTI_TENSORS = 512;
static TensorInfo* d_metadata_A = nullptr;
static TensorInfo* h_metadata_A = nullptr;
static TensorInfo* d_metadata_B = nullptr;
static TensorInfo* h_metadata_B = nullptr;
static TensorInfo* d_metadata_C = nullptr;
static TensorInfo* h_metadata_C = nullptr;
static TensorInfo* d_metadata_D = nullptr;
static TensorInfo* h_metadata_D = nullptr;

// Prefix-sum offset buffers for global work distribution
static int64_t* d_offsets = nullptr;
static int64_t* h_offsets = nullptr;

// Cached SM count for the current device
static int cached_num_sms = 0;

static void ensure_metadata_buffers() {
    if (!d_metadata_A) {
        cudaMalloc(&d_metadata_A, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMallocHost(&h_metadata_A, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMalloc(&d_metadata_B, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMallocHost(&h_metadata_B, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMalloc(&d_metadata_C, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMallocHost(&h_metadata_C, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMalloc(&d_metadata_D, MAX_MULTI_TENSORS * sizeof(TensorInfo));
        cudaMallocHost(&h_metadata_D, MAX_MULTI_TENSORS * sizeof(TensorInfo));

        // Allocate prefix-sum offset buffers (n+1 entries for n tensors)
        cudaMalloc(&d_offsets, (MAX_MULTI_TENSORS + 1) * sizeof(int64_t));
        cudaMallocHost(&h_offsets, (MAX_MULTI_TENSORS + 1) * sizeof(int64_t));
    }
}

// Query and cache the number of SMs on the current device
static int get_num_sms() {
    if (cached_num_sms == 0) {
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&cached_num_sms, cudaDevAttrMultiProcessorCount, deviceId);
    }
    return cached_num_sms;
}

// Compute optimal grid size: at least BLOCKS_PER_SM blocks per SM,
// but also enough to cover total_work with the given block_size
static int compute_grid_size(int64_t total_work, int block_size) {
    int num_sms = get_num_sms();
    int min_blocks = num_sms * 4; // At least 4 blocks per SM for good load balancing
    int work_blocks = (int)std::min((int64_t)INT32_MAX, (total_work + block_size - 1) / block_size);
    return std::max(min_blocks, std::min(work_blocks, 1024));
}

// Build prefix-sum offsets on host and copy to device asynchronously
// h_offsets[0] = 0, h_offsets[i] = sum of numel for tensors 0..i-1
static void build_offsets(const TensorInfo* tensors, int n, cudaStream_t stream) {
    h_offsets[0] = 0;
    for (int i = 0; i < n; i++) {
        h_offsets[i + 1] = h_offsets[i] + tensors[i].numel;
    }
    cudaMemcpyAsync(d_offsets, h_offsets, (n + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
}

// =============================================================================
// Device helper: map global_idx -> (tensor_id, local_idx) using binary search
// on the prefix-sum offsets array stored in shared memory
// =============================================================================
__device__ __forceinline__
void find_tensor_and_local_idx(
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t global_idx,
    int& tensor_id,
    int64_t& local_idx
) {
    // Binary search: find largest t such that offsets[t] <= global_idx
    int lo = 0, hi = num_tensors - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (offsets[mid] <= global_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    tensor_id = lo;
    local_idx = global_idx - offsets[lo];
}

// =============================================================================
// MULTI-TENSOR L2 NORM  (Global Work Distribution)
// =============================================================================

__global__ void multi_tensor_grad_norm_kernel(
    const TensorInfo* __restrict__ tensors,
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t total_work,
    float* accumulator
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float total_sq = 0.0f;

    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    // Grid-stride loop over ALL elements across ALL tensors
    for (int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + tid;
         global_idx < total_work;
         global_idx += grid_stride) {
        // Map global index to tensor + local element
        int t;
        int64_t local_idx;
        find_tensor_and_local_idx(offsets, num_tensors, global_idx, t, local_idx);

        float val = tensors[t].ptr[local_idx];
        total_sq += val * val;
    }

    // Block-level reduction in shared memory
    sdata[tid] = total_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(accumulator, sdata[0]);
}

void multi_tensor_grad_norm_cuda(const std::vector<TensorInfo>& tensors, float* norm_sq_accumulator) {
    if (tensors.empty()) return;
    ensure_metadata_buffers();
    int n = std::min((int)tensors.size(), (int)MAX_MULTI_TENSORS);

    // Use pinned host buffer and async copy
    std::memcpy(h_metadata_A, tensors.data(), n * sizeof(TensorInfo));
    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
    cudaMemcpyAsync(d_metadata_A, h_metadata_A, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);

    // Build prefix-sum offsets for global work distribution
    build_offsets(tensors.data(), n, stream);
    int64_t total_work = h_offsets[n];

    int threads = 256;
    int blocks = compute_grid_size(total_work, threads);
    multi_tensor_grad_norm_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        d_metadata_A, d_offsets, n, total_work, norm_sq_accumulator);
}

// =============================================================================
// MULTI-TENSOR SCALE  (Global Work Distribution)
// =============================================================================

__global__ void multi_tensor_scale_kernel(
    const TensorInfo* __restrict__ tensors,
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t total_work,
    const float* clip_coef
) {
    float scale = *clip_coef;
    if (scale >= 1.0f) return;

    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    // Grid-stride loop over ALL elements across ALL tensors
    for (int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_work;
         global_idx += grid_stride) {
        int t;
        int64_t local_idx;
        find_tensor_and_local_idx(offsets, num_tensors, global_idx, t, local_idx);

        tensors[t].ptr[local_idx] *= scale;
    }
}

void multi_tensor_scale_cuda(const std::vector<TensorInfo>& tensors, const float* clip_coef) {
    if (get_arch() == ArchFamily::Ada) {
        multi_tensor_scale_sm89_cuda(tensors, clip_coef);
        return;
    }
    if (tensors.empty()) return;
    ensure_metadata_buffers();
    int n = std::min((int)tensors.size(), (int)MAX_MULTI_TENSORS);

    // Pinned async copy
    std::memcpy(h_metadata_A, tensors.data(), n * sizeof(TensorInfo));
    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
    cudaMemcpyAsync(d_metadata_A, h_metadata_A, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);

    // Build prefix-sum offsets
    build_offsets(tensors.data(), n, stream);
    int64_t total_work = h_offsets[n];

    int threads = 256;
    int blocks = compute_grid_size(total_work, threads);
    multi_tensor_scale_kernel<<<blocks, threads, 0, stream>>>(
        d_metadata_A, d_offsets, n, total_work, clip_coef);
}

// =============================================================================
// MULTI-TENSOR ADAM  (Global Work Distribution)
// =============================================================================

__global__ void multi_tensor_adam_kernel(
    const TensorInfo* __restrict__ params,
    const TensorInfo* __restrict__ grads,
    const TensorInfo* __restrict__ ms,
    const TensorInfo* __restrict__ vs,
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t total_work,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    // Grid-stride loop over ALL elements across ALL tensors
    for (int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_work;
         global_idx += grid_stride) {
        int t;
        int64_t local_idx;
        find_tensor_and_local_idx(offsets, num_tensors, global_idx, t, local_idx);

        float* p = params[t].ptr;
        const float* g = grads[t].ptr;
        float* m = ms[t].ptr;
        float* v = vs[t].ptr;

        float grad = g[local_idx];
        float param = p[local_idx];

        float m_val = beta1 * m[local_idx] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[local_idx] + (1.0f - beta2) * grad * grad;

        m[local_idx] = m_val;
        v[local_idx] = v_val;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        // AdamW: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        p[local_idx] = param - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

void multi_tensor_adam_cuda(
    const std::vector<TensorInfo>& params,
    const std::vector<TensorInfo>& grads,
    const std::vector<TensorInfo>& ms,
    const std::vector<TensorInfo>& vs,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2, bool is_adamw
) {
    if (get_arch() == ArchFamily::Ada) {
        multi_tensor_adam_sm89_cuda(params, grads, ms, vs, lr, beta1, beta2, eps, 
                                    weight_decay, bias_correction1, bias_correction2, is_adamw);
        return;
    }

    if (params.empty()) return;
    ensure_metadata_buffers();
    int n = std::min((int)params.size(), (int)MAX_MULTI_TENSORS);

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    std::memcpy(h_metadata_A, params.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(d_metadata_A, h_metadata_A, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);
    
    std::memcpy(h_metadata_B, grads.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(d_metadata_B, h_metadata_B, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);

    std::memcpy(h_metadata_C, ms.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(d_metadata_C, h_metadata_C, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);

    std::memcpy(h_metadata_D, vs.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(d_metadata_D, h_metadata_D, n * sizeof(TensorInfo), cudaMemcpyHostToDevice, stream);

    // Build prefix-sum offsets from params (all arrays share same element counts)
    build_offsets(params.data(), n, stream);
    int64_t total_work = h_offsets[n];

    int threads = 256;
    int blocks = compute_grid_size(total_work, threads);
    multi_tensor_adam_kernel<<<blocks, threads, 0, stream>>>(
        d_metadata_A, d_metadata_B, d_metadata_C, d_metadata_D,
        d_offsets, n, total_work,
        lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2
    );
}

// =============================================================================
// Portable multi_tensor_zero dispatcher — forwards to the sm89 fast path.
// Dtype-agnostic: takes (void*, nbytes) pairs so it works for FP32 / FP16 /
// BF16 / int / complex / any type whose zero is all-zero bytes.
// =============================================================================
void multi_tensor_zero_cuda(const std::vector<ZeroTensorInfo>& tensors) {
    multi_tensor_zero_sm89_cuda(tensors);
}

} // namespace cuda
} // namespace OwnTensor