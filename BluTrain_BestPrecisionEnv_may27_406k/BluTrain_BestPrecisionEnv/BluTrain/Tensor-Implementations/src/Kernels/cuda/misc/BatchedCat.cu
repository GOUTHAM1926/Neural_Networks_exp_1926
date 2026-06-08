// Fused batched CAT kernel — PyTorch-style CatArrayBatchedCopy.
// Replaces the per-input-tensor cudaMemcpy2DAsync + .contiguous() loop in
// Tensor::cat Path B with a single kernel launch that handles all input
// tensors at once.
//
// Matches the design of
// pytorch/aten/src/ATen/native/cuda/Shape.cu :: CatArrayBatchedCopy
//
// gridDim.y = number of input tensors
// gridDim.x = workers per input tensor (scaled to max input nelements)
//
// Supports:
//   - Contiguous input sources (fast path)
//   - Strided inputs where the CONCAT axis has stride dim_stride and trailing
//     dims are contiguous (covers our QKV backward use case and more)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>


namespace OwnTensor {

// Max number of input tensors we can concatenate in one kernel launch.
// Typical cat operations have 2-8 inputs; QKV backward has 3.
// The whole CatBatchMeta struct is passed by value into the kernel (kernel
// arg space ~4 KB on CUDA, each CatInputMeta ~56 bytes → 16 fits well).
static constexpr int CAT_MAX_INPUTS = 16;

struct CatInputMeta {
    const void* ptr;
    int64_t nelements;
    int64_t dim_size;       // size along the concat axis
    int64_t d_offset;       // running offset along concat axis in output
    int64_t outer_stride;   // memory step for the aggregated outer axis (only if !is_contig)
    int64_t dim_stride;     // memory step for the concat axis           (only if !is_contig)
    int32_t is_contig;      // 1 = read via linear index; 0 = read via strides
};

struct CatBatchMeta {
    CatInputMeta inputs[CAT_MAX_INPUTS];
};

template<typename T>
__global__ void cat_batched_kernel(
    T* __restrict__ output,
    int64_t total_dim_size,     // sum of dim_size across all inputs
    int64_t inner_size,          // product of trailing dims (after concat axis)
    CatBatchMeta batch)
{
    const int input_idx = blockIdx.y;
    const CatInputMeta& meta = batch.inputs[input_idx];

    const int64_t nelem = meta.nelements;
    const int64_t dim_size = meta.dim_size;
    const int64_t d_offset = meta.d_offset;

    const T* __restrict__ src = reinterpret_cast<const T*>(meta.ptr);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    while (tid < nelem) {
        // Decompose this thread's element index (tid) as if the input were
        // virtually shaped [outer_size, dim_size, inner_size] in row-major:
        //   inner = tid % inner_size
        //   d_local = (tid / inner_size) % dim_size
        //   outer = tid / (inner_size * dim_size)
        int64_t inner   = tid % inner_size;
        int64_t tmp     = tid / inner_size;
        int64_t d_local = tmp % dim_size;
        int64_t outer   = tmp / dim_size;

        // Linear index into output. The output has shape with concat dim
        // extended to total_dim_size; its stride for the concat dim is
        // inner_size, and its "outer" stride is total_dim_size * inner_size.
        int64_t out_idx =
              outer * (total_dim_size * inner_size)
            + (d_offset + d_local) * inner_size
            + inner;

        T val;
        if (meta.is_contig) {
            // Contiguous source: linear index works directly.
            val = src[tid];
        } else {
            // Strided source: honour per-axis strides. Last-dim stride = 1
            // (required — this matches PyTorch's last-contig assumption).
            int64_t src_idx =
                  outer * meta.outer_stride
                + d_local * meta.dim_stride
                + inner;
            val = src[src_idx];
        }

        output[out_idx] = val;
        tid += stride;
    }
}

// =============================================================================
// Host launcher
// =============================================================================
// Arrays (input_ptrs, input_nelements, ...) have `num_inputs` entries each.
// elem_size selects the typed kernel specialization (1/2/4/8 byte elements).

extern "C" void cat_batched_cuda(
    void* output_ptr,
    int32_t elem_size,
    int64_t total_dim_size,
    int64_t inner_size,
    const void* const* input_ptrs,
    const int64_t* input_nelements,
    const int64_t* input_dim_sizes,
    const int64_t* input_d_offsets,
    const int64_t* input_outer_strides,
    const int64_t* input_dim_strides,
    const int32_t* input_is_contig,
    int64_t num_inputs,
    cudaStream_t stream)
{
    if (num_inputs <= 0) return;
    if (num_inputs > CAT_MAX_INPUTS) {
        // Fallback is the caller's responsibility — we don't silently truncate.
        printf("cat_batched_cuda: num_inputs=%lld exceeds CAT_MAX_INPUTS=%d\n",
               (long long)num_inputs, CAT_MAX_INPUTS);
        return;
    }

    // Pack metadata. This struct is passed by value to the kernel
    // (lives in constant-arg memory). No cudaMalloc needed.
    CatBatchMeta batch;
    int64_t max_nelem = 0;
    for (int64_t i = 0; i < num_inputs; ++i) {
        batch.inputs[i].ptr          = input_ptrs[i];
        batch.inputs[i].nelements    = input_nelements[i];
        batch.inputs[i].dim_size     = input_dim_sizes[i];
        batch.inputs[i].d_offset     = input_d_offsets[i];
        batch.inputs[i].outer_stride = input_outer_strides[i];
        batch.inputs[i].dim_stride   = input_dim_strides[i];
        batch.inputs[i].is_contig    = input_is_contig[i];
        if (input_nelements[i] > max_nelem) max_nelem = input_nelements[i];
    }

    if (max_nelem == 0) return;

    constexpr int THREADS = 256;
    int64_t blocks_x = (max_nelem + THREADS - 1) / THREADS;
    // Cap grid.x so we don't over-allocate the grid for small inputs; each
    // block will grid-stride loop over its share.
    if (blocks_x > 4096) blocks_x = 4096;

    dim3 grid((unsigned)blocks_x, (unsigned)num_inputs);
    dim3 block(THREADS);

    // Typed dispatch by element byte width. Unsupported widths raise instead of
    // silently no-opping (the old printf left the output buffer uninitialized).
    switch (elem_size) {
        case 4:
            cat_batched_kernel<uint32_t><<<grid, block, 0, stream>>>(
                (uint32_t*)output_ptr, total_dim_size, inner_size, batch);
            break;
        case 2:
            cat_batched_kernel<uint16_t><<<grid, block, 0, stream>>>(
                (uint16_t*)output_ptr, total_dim_size, inner_size, batch);
            break;
        case 8:
            cat_batched_kernel<uint64_t><<<grid, block, 0, stream>>>(
                (uint64_t*)output_ptr, total_dim_size, inner_size, batch);
            break;
        case 1:
            cat_batched_kernel<uint8_t><<<grid, block, 0, stream>>>(
                (uint8_t*)output_ptr, total_dim_size, inner_size, batch);
            break;
        case 16:
            // 16-byte types: complex128, or any hypothetical 128-bit payload.
            // uint4 reads/writes natively as STG.128 on sm_89.
            cat_batched_kernel<uint4><<<grid, block, 0, stream>>>(
                (uint4*)output_ptr, total_dim_size, inner_size, batch);
            break;
        default:
            throw std::runtime_error(
                "cat_batched_cuda: unsupported elem_size=" +
                std::to_string(elem_size) +
                " (supported: 1, 2, 4, 8, 16 bytes)");
    }
}

} // namespace OwnTensor
