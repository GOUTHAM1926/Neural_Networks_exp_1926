#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>     // printf for loud-fail paths
#include <algorithm>  // std::min

namespace OwnTensor {

// =============================================================================
// FastDivmod — magic-number fast divisor
// Replaces 64-bit div/mod (~40 cycles) with multiply-high + shift (~6 cycles).
// =============================================================================
struct FastDivmod {
    uint32_t d;
    uint32_t magic;
    uint32_t shift;

    static FastDivmod make(uint32_t divisor) {
        FastDivmod f;
        f.d = divisor;
        f.shift = 0;
        if (divisor == 1) { f.magic = 0; return f; }
        f.magic = (uint32_t)(((uint64_t(1) << 32) + (uint64_t)divisor - 1) / divisor);
        return f;
    }

    __device__ __forceinline__
    void divmod(uint32_t n, uint32_t &q, uint32_t &r) const {
        if (d == 1) { q = n; r = 0; return; }
        q = __umulhi(n, magic);
        if (q * d > n) --q;
        r = n - q * d;
    }
};

// Single source of truth for the dim cap. kMaxContigDims=10 fits comfortably
// in the kernel arg space (~4 KB on CUDA): one ContiguousMeta is
// 10*12 + 10*8 + 16 = ~216 bytes.
constexpr int kMaxContigDims = 10;

struct ContiguousMeta {
    FastDivmod divmods[kMaxContigDims];
    int64_t    strides[kMaxContigDims];
    int32_t    ndim;
    int64_t    storage_offset_elems;
    int32_t    elem_size;
};

// =============================================================================
// PATH 1: TILED 2D TRANSPOSE (TensorFlow-style, ~2x speedup)
// When the operation reduces to a 2D transpose (dim0, dim1 swap), use shared
// memory tiling with 32x32 tiles. Both read and write are coalesced.
// Bank-conflict avoidance via +1 padding in shared memory.
// =============================================================================
template<typename T, int TileSize = 32, int BlockRows = 8>
__global__ void transpose_2d_tiled_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int rows, int cols)
{
    __shared__ T tile[TileSize][TileSize + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TileSize + threadIdx.x;
    int y = blockIdx.y * TileSize + threadIdx.y;

    // Phase 1: coalesced read from src into shared memory
    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * cols + x];
        }
    }
    __syncthreads();

    // Phase 2: coalesced write from shared memory (transposed) to dst
    x = blockIdx.y * TileSize + threadIdx.x;
    y = blockIdx.x * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < rows && (y + j) < cols) {
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 3D batched transpose: batched [B, rows, cols] -> [B, cols, rows]
template<typename T, int TileSize = 32, int BlockRows = 8>
__global__ void transpose_3d_tiled_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int batch, int rows, int cols)
{
    __shared__ T tile[TileSize][TileSize + 1];
    int b = blockIdx.z;
    if (b >= batch) return;

    const T* src_b = src + (int64_t)b * rows * cols;
    T* dst_b = dst + (int64_t)b * rows * cols;

    int x = blockIdx.x * TileSize + threadIdx.x;
    int y = blockIdx.y * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < cols && (y + j) < rows)
            tile[threadIdx.y + j][threadIdx.x] = src_b[(y + j) * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TileSize + threadIdx.x;
    y = blockIdx.x * TileSize + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TileSize; j += BlockRows) {
        if (x < rows && (y + j) < cols)
            dst_b[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// =============================================================================
// PATH 2: STRIDED-WITH-CONTIGUOUS-INNER-DIM VECTORIZED COPY
//
// When the strides describe the OUTER dims as scattered but the INNERMOST dim
// has stride==1 and is divisible by VEC, we can: (a) divmod-walk only the
// OUTER ndim-1 dims to get the inner-row base offset, then (b) read VEC
// contiguous elements as a single 8/16-byte vector load. PyTorch's
// elementwise_kernel uses the same pattern (Loops.cuh).
//
// 4D training tensors after a permute (e.g. [B,H,T,D] view of [B,T,H,D]) hit
// this path: outer 3 dims are strided, inner D dim is unit-stride.
//
// Grid layout: outer goes on gridDim.x (supports 2^31-1) — earlier version
// had it on gridDim.y which caps at 65,535 and silently failed launch for
// 4D shapes like [16,12,1024,64] (outer_total=196608).
// =============================================================================
template<typename VecT, typename T, int VecCount>
__global__ void strided_inner_vec_copy_kernel(
    const T* __restrict__ src_typed,
    T* __restrict__ dst_typed,
    int64_t outer_total,        // product of outer (ndim-1) dims
    int64_t inner_size,          // size of innermost dim, multiple of VecCount
    const __grid_constant__ ContiguousMeta meta)
{
    int64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_total) return;

    // Decompose outer_idx across the outer (ndim-1) dims via FastDivmod.
    // strides[]/divmods[] only carry the outer dims (dispatcher set
    // meta.ndim = ndim-1). Strides are in source ELEMENTS (typed pointer).
    int64_t src_row_off = meta.storage_offset_elems;
    {
        uint32_t linear = (uint32_t)outer_idx;
        #pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            uint32_t q, r;
            meta.divmods[d].divmod(linear, q, r);
            src_row_off += (int64_t)r * meta.strides[d];
            linear = q;
        }
    }
    // dst is contiguous output → row offset is just outer_idx * inner_size.
    int64_t dst_row_off = outer_idx * inner_size;

    const VecT* src_v = reinterpret_cast<const VecT*>(src_typed + src_row_off);
    VecT*       dst_v = reinterpret_cast<VecT*>      (dst_typed + dst_row_off);
    int64_t n_vec = inner_size / VecCount;

    int64_t i = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.y * blockDim.x;
    #pragma unroll 4
    for (; i < n_vec; i += stride) {
        dst_v[i] = __ldg(&src_v[i]);
    }
}

// =============================================================================
// PATH 3: GENERIC STRIDED COPY (fallback — current kernel)
// Handles any shape/stride pattern with FastDivmod.
// =============================================================================
template<int MaxDims>
__global__ void generic_strided_copy_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t total_elems,
    const __grid_constant__ ContiguousMeta meta)
{
    int64_t i_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i_base >= total_elems) return;

    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int64_t i = i_base + v;
        if (i >= total_elems) break;

        uint32_t linear = (uint32_t)i;
        int64_t elem_off = meta.storage_offset_elems;

        #pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            uint32_t q, r;
            meta.divmods[d].divmod(linear, q, r);
            elem_off += (int64_t)r * meta.strides[d];
            linear = q;
        }

        int64_t src_byte = elem_off * meta.elem_size;
        int64_t dst_byte = i       * meta.elem_size;

        if (meta.elem_size == 4) {
            *reinterpret_cast<float*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const float*>(src + src_byte));
        } else if (meta.elem_size == 8) {
            *reinterpret_cast<double*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const double*>(src + src_byte));
        } else if (meta.elem_size == 2) {
            *reinterpret_cast<uint16_t*>(dst + dst_byte) =
                __ldg(reinterpret_cast<const uint16_t*>(src + src_byte));
        } else {
            for (int k = 0; k < meta.elem_size; ++k)
                dst[dst_byte + k] = src[src_byte + k];
        }
    }
}

// =============================================================================
// HOST-SIDE DISPATCHER WITH DIMENSION COALESCING
// =============================================================================

// Coalesce adjacent dimensions where strides are aligned.
// E.g. [2,3,4] with strides [12,4,1] → [24] with stride [1]
// Returns new ndim after coalescing.
static int coalesce_dimensions(
    int64_t* dims, int64_t* strides, int ndim)
{
    if (ndim <= 1) return ndim;
    int write = 0;
    for (int read = 1; read < ndim; ++read) {
        // If dims[write] has stride that matches dims[read] * strides[read]
        // (meaning they form a contiguous block), merge them.
        if (strides[write] == dims[read] * strides[read]) {
            dims[write] *= dims[read];
            // stride[write] stays the same (from the inner dim)
            strides[write] = strides[read];
        } else {
            ++write;
            dims[write] = dims[read];
            strides[write] = strides[read];
        }
    }
    return write + 1;
}

// Detect if the coalesced pattern is a 2D transpose: shape [a,b] with strides [1,a]
// Returns true if this is a transpose pattern; fills out rows/cols.
static bool is_2d_transpose(
    const int64_t* dims, const int64_t* strides, int ndim,
    int& rows_out, int& cols_out)
{
    if (ndim != 2) return false;
    // After transpose, outer dim has stride 1 (was inner), inner dim has stride = outer size
    if (strides[0] == 1 && strides[1] == dims[0]) {
        rows_out = (int)dims[1];  // output rows
        cols_out = (int)dims[0];  // output cols = input rows
        return true;
    }
    return false;
}

// Removed:
//   - is_3d_batched_transpose: the condition (strides[2]==1 && strides[1]==
//     dims[2] && strides[0]==dims[1]*dims[2]) describes the IDENTITY
//     contiguous layout, not a transpose. Path 3c never fired correctly.
//   - vectorized_contiguous_copy_kernel + dead Path 3d block: the
//     "fully contiguous" case is already handled by 3a (cudaMemcpyAsync,
//     DMA path). The kernel was declared but never launched.
//   - old can_vectorize helper: replaced by can_vectorize_inner below
//     which targets the actually-useful strided-with-contig-inner-dim case.

// Detect if everything is contiguous after coalescing (single dim with stride 1)
static bool is_fully_contiguous(
    const int64_t* strides, int ndim)
{
    return ndim == 1 && strides[0] == 1;
}

// Vectorize the inner dim when stride==1, the inner size is a multiple of
// vec_count, and src/dst are vec-aligned. Outer dims still walk via
// FastDivmod, but each thread loads vec_count elements at once
// (float4/int4 = 16 B, ~2-3x bandwidth on hot patterns).
static bool can_vectorize_inner(
    const int64_t* dims, const int64_t* strides, int ndim,
    int vec_count, int elem_size,
    const void* src, const void* dst)
{
    if (ndim < 2) return false;                      // need outer dims to walk
    if (strides[ndim - 1] != 1) return false;        // inner must be unit-stride
    if (dims[ndim - 1] % vec_count != 0) return false;
    int align = vec_count * elem_size;               // 16 for vec4 of float
    if ((reinterpret_cast<uintptr_t>(src) % align) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(dst) % align) != 0) return false;
    return true;
}

extern "C" void contiguous_strided_copy_cuda(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims_in, const int64_t* strides_in, int32_t ndim_in,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream)
{
    // Fixed MaxDims mismatch: previously ContiguousMeta sized for 10 dims,
    // local arrays sized 12, template MaxDims=12 — so dims 10..11 were
    // silently dropped from meta (real correctness bug for >10-dim tensors).
    // Now consistently kMaxContigDims everywhere; loud-fail rather than
    // truncate.
    constexpr int MaxDims = kMaxContigDims;  // single source of truth (=10)
    constexpr int Threads = 256;

    if (ndim_in > MaxDims) {
        // Realistic deep-learning tensors are at most 6-D. PyTorch's hard
        // cap is also single-digit. Loud-fail beats silent truncation.
        printf("contiguous_strided_copy_cuda: ndim=%d exceeds MaxDims=%d\n",
               (int)ndim_in, MaxDims);
        return;
    }

    // Step 1: Copy dims/strides locally for coalescing
    int64_t dims[MaxDims], strides[MaxDims];
    int ndim = ndim_in;
    for (int i = 0; i < ndim; ++i) {
        dims[i] = dims_in[i];
        strides[i] = strides_in[i];
    }

    // Step 2: Dimension coalescing (merge contiguous runs)
    // Only safe when storage_offset == 0 (otherwise offset logic gets complicated).
    if (storage_offset == 0) {
        ndim = coalesce_dimensions(dims, strides, ndim);
    }

    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst);

    // Step 3: DISPATCH — pick the fastest applicable path

    // ── 3a. Fully contiguous → cudaMemcpyAsync (hardware DMA, fastest) ──
    if (storage_offset == 0 && is_fully_contiguous(strides, ndim)) {
        cudaMemcpyAsync(dst_bytes, src_bytes,
                        total_elems * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // ── 3b. 2D transpose → TILED SHARED-MEM KERNEL (TensorFlow-style) ──
    // Common case: matrix transpose, attention head permute (after coalescing)
    {
        int rows = 0, cols = 0;
        if (storage_offset == 0 && is_2d_transpose(dims, strides, ndim, rows, cols)
            && rows >= 16 && cols >= 16) {
            dim3 block(32, 8);
            dim3 grid((cols + 31) / 32, (rows + 31) / 32);
            if (elem_size == 4) {
                transpose_2d_tiled_kernel<float><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), rows, cols);
                return;
            } else if (elem_size == 2) {
                transpose_2d_tiled_kernel<uint16_t><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const uint16_t*>(src), reinterpret_cast<uint16_t*>(dst), rows, cols);
                return;
            } else if (elem_size == 8) {
                transpose_2d_tiled_kernel<double><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const double*>(src), reinterpret_cast<double*>(dst), rows, cols);
                return;
            }
            // Otherwise fall through to generic
        }
    }

    // ── 3c. STRIDED-WITH-CONTIGUOUS-INNER-DIM VECTORIZED ──
    //   Wins ONLY when inner is large enough that one block's worth of
    //   threads has useful work. For attn head_dim=64 (n_vec=16), the
    //   generic kernel's 4-way coarsening over total_elems beats this
    //   approach because per-block thread utilization here would be
    //   only n_vec/InnerThreads = 12.5%. Gate dispatch on n_vec ≥
    //   InnerThreads — below that, fall through to generic.
    //
    //   Where this DOES win: BTE/MLP-intermediate copies (inner ≥ 768
    //   → n_vec ≥ 192 ≥ 128 threshold), where the float4 stride saves
    //   ~2× over generic's scalar 4-way coarsening.
    {
        constexpr int InnerThreads = 128;
        constexpr int VEC          = 4;  // 4 elements per vector load
        if (storage_offset == 0
            && (elem_size == 4 || elem_size == 2)
            && can_vectorize_inner(dims, strides, ndim, VEC, elem_size, src, dst)
            && (dims[ndim - 1] / VEC) >= InnerThreads)
        {
            int64_t inner_size  = dims[ndim - 1];
            int64_t outer_total = total_elems / inner_size;
            int64_t n_vec       = inner_size / VEC;

            // Pack only the OUTER (ndim-1) dims into meta. Strides are in
            // ELEMENTS (typed pointer in the kernel), no byte conversion.
            ContiguousMeta meta = {};
            meta.ndim = ndim - 1;
            meta.storage_offset_elems = storage_offset;
            meta.elem_size = elem_size;
            for (int d = 0; d < meta.ndim && d < MaxDims; ++d) {
                meta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
                meta.strides[d] = strides[d];
            }

            int64_t inner_blocks = std::min<int64_t>(
                (n_vec + InnerThreads - 1) / InnerThreads, 256);
            // outer on grid.x (2^31-1 limit, fits 4D training shapes ~200K),
            // inner row chunks on grid.y (≤256 by construction).
            dim3 grid((unsigned)outer_total, (unsigned)inner_blocks);
            dim3 block(InnerThreads);

            if (elem_size == 4) {
                strided_inner_vec_copy_kernel<float4, float, VEC><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const float*>(src),
                    reinterpret_cast<float*>(dst),
                    outer_total, inner_size, meta);
            } else { // elem_size == 2 → pack 4 halves as uint2 (8 B)
                strided_inner_vec_copy_kernel<uint2, uint16_t, VEC><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const uint16_t*>(src),
                    reinterpret_cast<uint16_t*>(dst),
                    outer_total, inner_size, meta);
            }
            // Loud-fail on launch error — silent failure here = garbage
            // copies into compute kernels = NaN/explosion in training.
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("strided_inner_vec_copy_kernel launch failed: %s "
                       "(grid=%lldx%lld, block=%d, outer_total=%lld, n_vec=%lld)\n",
                       cudaGetErrorString(err),
                       (long long)grid.x, (long long)grid.y,
                       (int)block.x, (long long)outer_total, (long long)n_vec);
            }
            return;
        }
    }

    // ── 3d. GENERIC FALLBACK: strided copy with FastDivmod ──
    int64_t threads_needed = (total_elems + 3) / 4;
    int64_t blocks = (threads_needed + Threads - 1) / Threads;

    ContiguousMeta meta = {};
    meta.ndim = ndim;
    meta.storage_offset_elems = storage_offset;
    meta.elem_size = elem_size;
    for (int d = 0; d < ndim && d < MaxDims; ++d) {
        meta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
        meta.strides[d] = strides[d];
    }

    generic_strided_copy_kernel<MaxDims><<<blocks, Threads, 0, stream>>>(
        src_bytes, dst_bytes, total_elems, meta);
}

} // namespace OwnTensor
