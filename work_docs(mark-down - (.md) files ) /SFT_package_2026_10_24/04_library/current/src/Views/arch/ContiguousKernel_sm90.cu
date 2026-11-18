// Strided copy (contiguous materialization), sm_90 path. A REPLACEMENT for the
// dispatch in ../ContiguousKernel.cu, not an addition to it: the two cannot be
// interleaved. Reached only when get_arch() == ArchFamily::Hopper.
//
// FastDivmod and ContiguousMeta are duplicated here, not shared: ContiguousMeta
// is passed by value into the generic file's kernels, so a shared definition
// would change that path's kernel ABI.
//
// Path map, in dispatch order:
//     3a  fully contiguous          -> cudaMemcpyAsync
//     3c  contiguous inner dim      -> vectorized, flat grid-stride
//     3b  2D transpose (family A)   -> XOR-swizzled smem tile
//     3f  inner-transposed (fam B)  -> flat one-element-per-thread
//     3d  anything else             -> generic FastDivmod walk

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>     // printf for loud-fail paths
#include <algorithm>  // std::min

namespace OwnTensor {
namespace sm90 {

// Magic-number divisor: multiply-high plus correction instead of 64-bit div/mod.
struct FastDivmod {
    uint32_t d;
    uint32_t magic;
    uint32_t one;      // 1 when d == 1; see divmod

    static FastDivmod make(uint32_t divisor) {
        FastDivmod f;
        f.d = divisor;
        f.one = (divisor == 1) ? 1u : 0u;
        if (divisor == 1) { f.magic = 0; return f; }
        f.magic = (uint32_t)(((uint64_t(1) << 32) + (uint64_t)divisor - 1) / divisor);
        return f;
    }

    // A 32-bit magic cannot represent d == 1, so it folds in as a select on the
    // host-precomputed `one` flag. Do not replace with an `if (d == 1)` return:
    // a branch on a runtime field pulls in the software divide sequence.
    __device__ __forceinline__
    void divmod(uint32_t n, uint32_t &q, uint32_t &r) const {
        uint32_t qm = __umulhi(n, magic);
        qm -= (qm * d > n) ? 1u : 0u;
        q = one ? n : qm;
        r = one ? 0u : (n - qm * d);
    }
};

// 10 keeps ContiguousMeta inside the kernel argument space.
constexpr int kMaxContigDims = 10;

struct ContiguousMeta {
    FastDivmod divmods[kMaxContigDims];
    int64_t    strides[kMaxContigDims];
    int32_t    ndim;
    int64_t    storage_offset_elems;
    int32_t    elem_size;
    int64_t    dims64[kMaxContigDims];   // the dims, for the 64-bit walk (copies past 2^31 elements)
};

// 2026-10-31: the walk for copies that do not fit 32 bits: 64-bit element number, plain 64-bit division. The int64_t
// instantiations used it to cast the element number to uint32_t (wrong past 2^32 elements). Copies the dispatcher in
// ../ContiguousKernel.cu can cut into 32-bit pieces never get here; the rest (one outer slice past 32 bits) do.
template<int NDim, typename OffT>
__device__ __forceinline__
OffT walk_index64(uint64_t linear, const ContiguousMeta& meta)
{
    OffT off = (OffT)meta.storage_offset_elems;
    if (NDim > 0) {
        #pragma unroll
        for (int d = NDim - 1; d >= 0; --d) {
            const uint64_t dd = (uint64_t)meta.dims64[d];
            const uint64_t q = linear / dd;
            off += (OffT)(linear - q * dd) * (OffT)meta.strides[d];
            linear = q;
        }
    } else {
        for (int d = meta.ndim - 1; d >= 0; --d) {
            const uint64_t dd = (uint64_t)meta.dims64[d];
            const uint64_t q = linear / dd;
            off += (OffT)(linear - q * dd) * (OffT)meta.strides[d];
            linear = q;
        }
    }
    return off;
}

// NDim = 0 means read meta.ndim at runtime; nonzero gives a compile-time trip
// count. Caller contract: OffT may be uint32_t only if every offset fits in
// 2^31, else int64_t.
template<int NDim, typename OffT>
__device__ __forceinline__
OffT walk_index(uint32_t linear, const ContiguousMeta& meta)
{
    OffT off = (OffT)meta.storage_offset_elems;
    if (NDim > 0) {
        #pragma unroll
        for (int d = NDim - 1; d >= 0; --d) {
            uint32_t q, r;
            meta.divmods[d].divmod(linear, q, r);
            off += (OffT)r * (OffT)meta.strides[d];
            linear = q;
        }
    } else {
        for (int d = meta.ndim - 1; d >= 0; --d) {
            uint32_t q, r;
            meta.divmods[d].divmod(linear, q, r);
            off += (OffT)r * (OffT)meta.strides[d];
            linear = q;
        }
    }
    return off;
}

// -- PATH 3c: strided outer dims, contiguous inner dim, vectorized --
// Grid is flat over vectors rather than one block per row, so the path stays
// reachable for small inner sizes.
template<typename VecT, typename T, int VecCount, int OuterDims, typename OffT>
__global__ void strided_inner_vec_copy_kernel(
    const T* __restrict__ src_typed,
    T* __restrict__ dst_typed,
    int64_t outer_total,         // product of outer (ndim-1) dims
    int64_t inner_size,          // innermost dim, multiple of VecCount
    FastDivmod nvec_divmod,      // divisor = inner_size / VecCount
    int64_t total_vecs,          // outer_total * (inner_size / VecCount)
    const __grid_constant__ ContiguousMeta meta)
{
    const int64_t gstride = (int64_t)gridDim.x * blockDim.x;

    if constexpr (sizeof(OffT) == 8) {   // 64-bit row and vector numbers (2026-10-31)
        const int64_t nv = inner_size / VecCount;
        for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; v < total_vecs; v += gstride) {
            const int64_t row = v / nv, vec_in_row = v - row * nv;
            const OffT src_row_off = walk_index64<OuterDims, OffT>((uint64_t)row, meta);
            const VecT* src_v = reinterpret_cast<const VecT*>(src_typed + (int64_t)src_row_off);
            VecT* dst_v = reinterpret_cast<VecT*>(dst_typed + row * inner_size);
            dst_v[vec_in_row] = __ldg(&src_v[vec_in_row]);
        }
        return;
    }
    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         v < total_vecs; v += gstride)
    {
        uint32_t row, vec_in_row;
        nvec_divmod.divmod((uint32_t)v, row, vec_in_row);

        // Strides are in source elements, not bytes.
        OffT src_row_off = walk_index<OuterDims, OffT>(row, meta);

        const VecT* src_v =
            reinterpret_cast<const VecT*>(src_typed + (int64_t)src_row_off);
        VecT* dst_v =
            reinterpret_cast<VecT*>(dst_typed + (int64_t)row * inner_size);

        dst_v[vec_in_row] = __ldg(&src_v[vec_in_row]);
    }
}

// -- PATH 3d: generic strided copy, any shape or stride --
// InnerContig amortizes one walk over a 4-element run, valid only while the run
// stays inside a row; the straddling case must keep walking per element.
template<typename T, int NDim, bool InnerContig, typename OffT>
__global__ void generic_strided_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t total_elems,
    uint32_t inner_size,          // only used if InnerContig
    FastDivmod inner_divmod,      // divisor = inner_size, only if InnerContig
    const __grid_constant__ ContiguousMeta meta)
{
    const int64_t nruns   = (total_elems + 3) >> 2;
    const int64_t gstride = (int64_t)gridDim.x * blockDim.x;

    if constexpr (sizeof(OffT) == 8) {   // 64-bit element numbers (2026-10-31)
        const int64_t isz = meta.dims64[meta.ndim - 1];
        for (int64_t run = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; run < nruns; run += gstride) {
            const int64_t i_base = run << 2;
            const int64_t left = total_elems - i_base;
            const int remaining = (left < 4) ? (int)left : 4;
            if (InnerContig) {
                const int64_t col = i_base % isz;
                if (col + remaining <= isz) {
                    const OffT base = walk_index64<NDim, OffT>((uint64_t)i_base, meta);
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        if (v >= remaining) break;
                        dst[i_base + v] = __ldg(&src[(int64_t)base + v]);
                    }
                    continue;
                }
            }
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                if (v >= remaining) break;
                const OffT off = walk_index64<NDim, OffT>((uint64_t)(i_base + v), meta);
                dst[i_base + v] = __ldg(&src[(int64_t)off]);
            }
        }
        return;
    }
    for (int64_t run = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         run < nruns; run += gstride)
    {
        const int64_t i_base = run << 2;
        const int64_t left   = total_elems - i_base;
        const int  remaining = (left < 4) ? (int)left : 4;

        if (InnerContig) {
            uint32_t row, col;
            inner_divmod.divmod((uint32_t)i_base, row, col);
            if (col + (uint32_t)remaining <= inner_size) {
                OffT base = walk_index<NDim, OffT>((uint32_t)i_base, meta);
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    if (v >= remaining) break;
                    dst[i_base + v] = __ldg(&src[(int64_t)base + v]);
                }
                continue;
            }
        }

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            if (v >= remaining) break;
            OffT off = walk_index<NDim, OffT>((uint32_t)(i_base + v), meta);
            dst[i_base + v] = __ldg(&src[(int64_t)off]);
        }
    }
}

// -- PATH 3b: tiled transpose through shared memory --
// Smem staging makes both global sides contiguous. Bank conflicts are avoided by
// XOR swizzle (logical [r][c] stored at physical column c ^ r), not by padding
// to [32][33]: padding breaks the 16-byte alignment vector smem access needs.
//
// Layout contract, in coalesced dims: dims [M, N], strides [1, sM], dst is
// [M][N] row-major, src element (m,n) at m + n*sM. Family A is S == 1.
template<typename T, int TileM, int TileN>
__global__ void transpose_tiled_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    uint32_t M,            // extent of the source's stride-1 axis
    uint32_t N,            // extent of the axis that is fastest in dst
    int64_t  stride_n,     // source stride along N
    int64_t  slice_src,    // source stride between slices, else 0
    int64_t  slice_dst,    // M*N, destination stride between slices
    uint32_t tiles_m,      // ceil(M / TileM)
    uint32_t tiles_n)      // ceil(N / TileN)
{
    __shared__ T tile[TileM][TileN];

    const uint32_t tiles_per_slice = tiles_m * tiles_n;
    const uint32_t slice  = blockIdx.x / tiles_per_slice;
    const uint32_t within = blockIdx.x % tiles_per_slice;
    const uint32_t tile_m = within / tiles_n;
    const uint32_t tile_n = within % tiles_n;

    const uint32_t m0 = tile_m * TileM;
    const uint32_t n0 = tile_n * TileN;

    const T* src_slice = src + (int64_t)slice * slice_src;
    T*       dst_slice = dst + (int64_t)slice * slice_dst;

    // Read coalesced along M. Loops stride by the full block extent so a blockDim
    // not matching the tile extent still covers the tile exactly once.
    for (uint32_t ml = threadIdx.x; ml < TileM; ml += blockDim.x) {
        for (uint32_t nl = threadIdx.y; nl < TileN; nl += blockDim.y) {
            const uint32_t m = m0 + ml;
            const uint32_t n = n0 + nl;
            if (m < M && n < N) {
                // Mask to TileN-1 keeps the XOR inside the row when TileM > TileN.
                tile[ml][nl ^ (ml & (TileN - 1))] =
                    __ldg(&src_slice[(int64_t)m + (int64_t)n * stride_n]);
            }
        }
    }

    __syncthreads();

    // Write coalesced along N. nl must be bounded by TileN, not blockDim.x:
    // bounding by blockDim.x drives nl to N-1 whenever N > TileN and writes past
    // the end of the tile row.
    for (uint32_t nl = threadIdx.x; nl < TileN; nl += blockDim.x) {
        for (uint32_t ml = threadIdx.y; ml < TileM; ml += blockDim.y) {
            const uint32_t m = m0 + ml;
            const uint32_t n = n0 + nl;
            if (m < M && n < N) {
                dst_slice[(int64_t)m * N + n] =
                    tile[ml][nl ^ (ml & (TileN - 1))];
            }
        }
    }
}

// -- PATH 3f: flat one-element-per-thread copy --
// Design rule: in a gather-scatter one side must be strided, so protect the
// STORE and let the load scatter. A strided load can still hit in L1/L2, while a
// partial-sector store is an unrecoverable read-modify-write at the controller.
// Hence the destination index is the thread index (3d's coarsening scatters it).
template<typename T, int EPT> struct FlatVec;
template<> struct FlatVec<float,    1> { using type = float;    };
template<> struct FlatVec<float,    2> { using type = float2;   };
template<> struct FlatVec<uint16_t, 1> { using type = uint16_t; };
template<> struct FlatVec<uint16_t, 2> { using type = uint32_t; };
template<> struct FlatVec<uint16_t, 4> { using type = uint2;    };

template<typename T, int NDim, typename OffT, int EPT>
__global__ void flat_strided_copy_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t total_elems,
    const __grid_constant__ ContiguousMeta meta)
{
    // The EPT elements a lane owns must be ADJACENT, not strided by blockDim:
    // strided chunks write a partial line for 2-byte types.
    using VecT = typename FlatVec<T, EPT>::type;

    // dst MUST be walked as VecT* and indexed BY VECTOR. Writing
    // `*reinterpret_cast<VecT*>(dst + i0) = v` inside a grid-stride loop silently
    // emits EPT scalar stores, because nvcc cannot prove alignment inside a loop.
    // Indexing a VecT* makes alignment a property of the type; launcher checks dst.
    VecT* dst_v = reinterpret_cast<VecT*>(dst);
    const int64_t n_vec   = total_elems / EPT;   // whole groups only
    const int64_t gstride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         v < n_vec; v += gstride)
    {
        const int64_t i0 = v * EPT;
        union { VecT vec; T elem[EPT]; } u;
        #pragma unroll
        for (int c = 0; c < EPT; ++c) {
            OffT off;
            if constexpr (sizeof(OffT) == 8) off = walk_index64<NDim, OffT>((uint64_t)(i0 + c), meta);   // 64-bit
            else off = walk_index<NDim, OffT>((uint32_t)(i0 + c), meta);
            u.elem[c] = __ldg(&src[(int64_t)off]);
        }
        dst_v[v] = u.vec;
    }

    // Ragged tail: total_elems is not guaranteed divisible by EPT.
    if (blockIdx.x == 0) {
        for (int64_t i = n_vec * EPT + threadIdx.x; i < total_elems;
             i += blockDim.x)
        {
            OffT off;
            if constexpr (sizeof(OffT) == 8) off = walk_index64<NDim, OffT>((uint64_t)i, meta);   // 64-bit
            else off = walk_index<NDim, OffT>((uint32_t)i, meta);
            dst[i] = __ldg(&src[(int64_t)off]);
        }
    }
}

// -- host-side dispatch helpers --

// Merge adjacent dims that form a contiguous block.
static int coalesce_dimensions(int64_t* dims, int64_t* strides, int ndim)
{
    if (ndim <= 1) return ndim;
    int write = 0;
    for (int read = 1; read < ndim; ++read) {
        if (strides[write] == dims[read] * strides[read]) {
            dims[write] *= dims[read];
            strides[write] = strides[read];
        } else {
            ++write;
            dims[write] = dims[read];
            strides[write] = strides[read];
        }
    }
    return write + 1;
}

static bool is_fully_contiguous(const int64_t* strides, int ndim)
{
    return ndim == 1 && strides[0] == 1;
}

// Inner dim unit-stride, divisible by vec_count, and both pointers aligned.
static bool can_vectorize_inner(
    const int64_t* dims, const int64_t* strides, int ndim,
    int vec_count, int elem_size,
    const void* src, const void* dst)
{
    if (ndim < 2) return false;                      // need outer dims to walk
    if (strides[ndim - 1] != 1) return false;
    if (dims[ndim - 1] % vec_count != 0) return false;
    int align = vec_count * elem_size;
    if ((reinterpret_cast<uintptr_t>(src) % align) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(dst) % align) != 0) return false;
    return true;
}

// Shape predicates, valid only after coalescing. The stride >= extent tests
// exclude aliasing views, which must fall through to the generic path rather
// than get a guess.
static bool is_2d_transpose(const int64_t* dims, const int64_t* strides, int ndim)
{
    return ndim == 2
        && strides[0] == 1
        && strides[1] >= dims[0];
}

static bool is_inner_transpose(const int64_t* dims, const int64_t* strides, int ndim)
{
    return ndim == 3
        && strides[1] == 1
        && strides[2] >= dims[1]
        && strides[0] >= dims[1] * dims[2];
}

// Largest source element index the view can touch, for the 32-bit offset test.
static int64_t source_span(const int64_t* dims, const int64_t* strides, int ndim)
{
    int64_t span = 1;
    for (int i = 0; i < ndim; ++i) span += (dims[i] - 1) * strides[i];
    return span;
}

static int grid_for(int64_t work_items, int threads)
{
    int64_t want = (work_items + threads - 1) / threads;
    if (want < 1) want = 1;
    const int64_t kMaxBlocks = 65535 * 8;
    return (int)std::min<int64_t>(want, kMaxBlocks);
}

// Launch helpers. Instantiated only for the walk depths that occur; the default
// case is the runtime-ndim fallback.
#define LAUNCH_3C(VECT, T, VEC, OD, OFFT)                                     \
    strided_inner_vec_copy_kernel<VECT, T, VEC, OD, OFFT>                     \
        <<<grid, Threads, 0, stream>>>(                                       \
            reinterpret_cast<const T*>(src), reinterpret_cast<T*>(dst),       \
            outer_total, inner_size, nvec_dm, total_vecs, meta)

template<typename VecT, typename T, int VecCount>
static void dispatch_3c(
    const void* src, void* dst,
    int64_t outer_total, int64_t inner_size,
    FastDivmod nvec_dm, int64_t total_vecs,
    int outer_dims, bool use32,
    const ContiguousMeta& meta, cudaStream_t stream)
{
    constexpr int Threads = 256;
    const int grid = grid_for(total_vecs, Threads);

    if (use32) {
        switch (outer_dims) {
            case 1: LAUNCH_3C(VecT, T, VecCount, 1, uint32_t); return;
            case 2: LAUNCH_3C(VecT, T, VecCount, 2, uint32_t); return;
            case 3: LAUNCH_3C(VecT, T, VecCount, 3, uint32_t); return;
            case 4: LAUNCH_3C(VecT, T, VecCount, 4, uint32_t); return;
            default: LAUNCH_3C(VecT, T, VecCount, 0, uint32_t); return;
        }
    }
    switch (outer_dims) {
        case 1: LAUNCH_3C(VecT, T, VecCount, 1, int64_t); return;
        case 2: LAUNCH_3C(VecT, T, VecCount, 2, int64_t); return;
        case 3: LAUNCH_3C(VecT, T, VecCount, 3, int64_t); return;
        case 4: LAUNCH_3C(VecT, T, VecCount, 4, int64_t); return;
        default: LAUNCH_3C(VecT, T, VecCount, 0, int64_t); return;
    }
}

// blockDim.x must be the widest tile axis so threadIdx.x runs along the
// coalesced axis in both phases.
template<typename T, int TileM, int TileN>
static void launch_transpose(
    const void* src, void* dst,
    uint32_t M, uint32_t N,
    int64_t stride_n, int64_t slice_src, int64_t slice_dst,
    uint32_t slices, cudaStream_t stream)
{
    const uint32_t tiles_m = (M + TileM - 1) / TileM;
    const uint32_t tiles_n = (N + TileN - 1) / TileN;
    const uint32_t blocks  = slices * tiles_m * tiles_n;

    dim3 block(TileM > TileN ? TileM : TileN, 8);
    if (block.x * block.y > 1024) block.y = 1024 / block.x;

    transpose_tiled_kernel<T, TileM, TileN><<<blocks, block, 0, stream>>>(
        reinterpret_cast<const T*>(src), reinterpret_cast<T*>(dst),
        M, N, stride_n, slice_src, slice_dst, tiles_m, tiles_n);
}

#define LAUNCH_3F(T, ND, OFFT, EPT)                                           \
    flat_strided_copy_kernel<T, ND, OFFT, EPT><<<grid, Threads, 0, stream>>>( \
        reinterpret_cast<const T*>(src), reinterpret_cast<T*>(dst),           \
        total_elems, meta)

#define LAUNCH_3F_ND(T, OFFT, EPT)                                            \
    do {                                                                      \
        switch (ndim) {                                                       \
            case 2:  LAUNCH_3F(T, 2, OFFT, EPT); return;                      \
            case 3:  LAUNCH_3F(T, 3, OFFT, EPT); return;                      \
            case 4:  LAUNCH_3F(T, 4, OFFT, EPT); return;                      \
            default: LAUNCH_3F(T, 0, OFFT, EPT); return;                      \
        }                                                                     \
    } while (0)

template<typename T>
static void dispatch_3f(
    const void* src, void* dst, int64_t total_elems, int ndim, bool use32,
    const ContiguousMeta& meta, cudaStream_t stream)
{
    constexpr int Threads = 256;
    // 8 bytes per lane regardless of dtype, not one 128-byte line per warp:
    // equal bytes per lane keeps index-math amortization the same for fp32 and
    // bf16 while the store stays coalesced.
    constexpr int EPT = (int)(8 / sizeof(T)) > 1 ? (int)(8 / sizeof(T)) : 1;
    const int grid = grid_for((total_elems + EPT - 1) / EPT, Threads);

    // Do not assume a fresh allocation: a misaligned dst would fault on the
    // wide store, so fall back to EPT == 1.
    const bool aligned =
        (reinterpret_cast<uintptr_t>(dst) % (EPT * sizeof(T))) == 0;

    if (EPT == 1 || aligned) {
        if (use32) LAUNCH_3F_ND(T, uint32_t, EPT);
        else       LAUNCH_3F_ND(T, int64_t,  EPT);
    } else {
        if (use32) LAUNCH_3F_ND(T, uint32_t, 1);
        else       LAUNCH_3F_ND(T, int64_t,  1);
    }
}

#define LAUNCH_3D(T, ND, IC, OFFT)                                            \
    generic_strided_copy_kernel<T, ND, IC, OFFT>                              \
        <<<grid, Threads, 0, stream>>>(                                       \
            reinterpret_cast<const T*>(src), reinterpret_cast<T*>(dst),       \
            total_elems, inner_size, inner_dm, meta)

#define LAUNCH_3D_ND(T, IC, OFFT)                                             \
    do {                                                                      \
        switch (ndim) {                                                       \
            case 2: LAUNCH_3D(T, 2, IC, OFFT); return;                        \
            case 3: LAUNCH_3D(T, 3, IC, OFFT); return;                        \
            case 4: LAUNCH_3D(T, 4, IC, OFFT); return;                        \
            case 5: LAUNCH_3D(T, 5, IC, OFFT); return;                        \
            default: LAUNCH_3D(T, 0, IC, OFFT); return;                       \
        }                                                                     \
    } while (0)

template<typename T>
static void dispatch_3d(
    const void* src, void* dst,
    int64_t total_elems, int ndim,
    uint32_t inner_size, FastDivmod inner_dm,
    bool inner_contig, bool use32,
    const ContiguousMeta& meta, cudaStream_t stream)
{
    constexpr int Threads = 256;
    const int grid = grid_for((total_elems + 3) / 4, Threads);

    if (use32) {
        if (inner_contig) LAUNCH_3D_ND(T, true,  uint32_t);
        else              LAUNCH_3D_ND(T, false, uint32_t);
    } else {
        if (inner_contig) LAUNCH_3D_ND(T, true,  int64_t);
        else              LAUNCH_3D_ND(T, false, int64_t);
    }
}

} // namespace sm90

// Entry point, called from ../ContiguousKernel.cu on Hopper. Returns true if it
// handled the copy; false only for element sizes it does not implement (1 and 8
// byte), which must reach the generic dispatcher instead.
bool contiguous_strided_copy_sm90(
    const void* src, void* dst,
    int64_t total_elems,
    const int64_t* dims_in, const int64_t* strides_in, int32_t ndim_in,
    int64_t storage_offset, int32_t elem_size,
    cudaStream_t stream)
{
    using namespace sm90;

    constexpr int MaxDims = kMaxContigDims;

    if (ndim_in > MaxDims) return false;     // generic path loud-fails on this
    if (total_elems <= 0) return true;
    if (elem_size != 4 && elem_size != 2) return false;

    int64_t dims[MaxDims], strides[MaxDims];
    int ndim = ndim_in;
    for (int i = 0; i < ndim; ++i) {
        dims[i] = dims_in[i];
        strides[i] = strides_in[i];
    }

    // Coalescing is only safe at storage_offset == 0.
    if (storage_offset == 0) {
        ndim = coalesce_dimensions(dims, strides, ndim);
    }

    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst);

    // -- 3a. Fully contiguous --
    if (storage_offset == 0 && is_fully_contiguous(strides, ndim)) {
        cudaMemcpyAsync(dst_bytes, src_bytes,
                        total_elems * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return true;
    }

    // 32-bit walk needs every source offset AND the destination linear index to
    // fit, with storage_offset folded in.
    const int64_t span = source_span(dims, strides, ndim) + storage_offset;
    const bool use32 = (span < (int64_t)2147483647)
                    && (total_elems < (int64_t)2147483647);

    // -- 3c. Contiguous inner dim, vectorized. Widths are tried largest-first. --
    if (storage_offset == 0 && ndim >= 2) {
        const int outer_dims = ndim - 1;

        // Pack only the outer dims; strides stay in elements.
        ContiguousMeta meta = {};
        meta.ndim = outer_dims;
        meta.storage_offset_elems = storage_offset;
        meta.elem_size = elem_size;
        for (int d = 0; d < outer_dims && d < MaxDims; ++d) {
            meta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
            meta.strides[d] = strides[d];
            meta.dims64[d] = dims[d];
        }

        const int64_t inner_size  = dims[ndim - 1];
        const int64_t outer_total = total_elems / inner_size;

        bool launched = false;
        if (elem_size == 4) {
            if (can_vectorize_inner(dims, strides, ndim, 4, 4, src, dst)) {
                const int64_t n_vec = inner_size / 4;
                FastDivmod nvec_dm = FastDivmod::make((uint32_t)n_vec);
                dispatch_3c<float4, float, 4>(src, dst, outer_total, inner_size,
                    nvec_dm, outer_total * n_vec, outer_dims, use32, meta, stream);
                launched = true;
            } else if (can_vectorize_inner(dims, strides, ndim, 2, 4, src, dst)) {
                const int64_t n_vec = inner_size / 2;
                FastDivmod nvec_dm = FastDivmod::make((uint32_t)n_vec);
                dispatch_3c<float2, float, 2>(src, dst, outer_total, inner_size,
                    nvec_dm, outer_total * n_vec, outer_dims, use32, meta, stream);
                launched = true;
            }
        } else { // elem_size == 2
            if (can_vectorize_inner(dims, strides, ndim, 8, 2, src, dst)) {
                const int64_t n_vec = inner_size / 8;
                FastDivmod nvec_dm = FastDivmod::make((uint32_t)n_vec);
                dispatch_3c<uint4, uint16_t, 8>(src, dst, outer_total, inner_size,
                    nvec_dm, outer_total * n_vec, outer_dims, use32, meta, stream);
                launched = true;
            } else if (can_vectorize_inner(dims, strides, ndim, 4, 2, src, dst)) {
                const int64_t n_vec = inner_size / 4;
                FastDivmod nvec_dm = FastDivmod::make((uint32_t)n_vec);
                dispatch_3c<uint2, uint16_t, 4>(src, dst, outer_total, inner_size,
                    nvec_dm, outer_total * n_vec, outer_dims, use32, meta, stream);
                launched = true;
            } else if (can_vectorize_inner(dims, strides, ndim, 2, 2, src, dst)) {
                const int64_t n_vec = inner_size / 2;
                FastDivmod nvec_dm = FastDivmod::make((uint32_t)n_vec);
                dispatch_3c<uint32_t, uint16_t, 2>(src, dst, outer_total, inner_size,
                    nvec_dm, outer_total * n_vec, outer_dims, use32, meta, stream);
                launched = true;
            }
        }

        if (launched) {
            // Loud-fail: a silently failed copy feeds garbage into compute.
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("sm90 strided_inner_vec_copy_kernel launch failed: %s "
                       "(outer_total=%lld, inner_size=%lld)\n",
                       cudaGetErrorString(err),
                       (long long)outer_total, (long long)inner_size);
            }
            return true;
        }
    }

    // -- 3b. Tiled transpose, family A only. Family B is deliberately NOT
    //        intercepted here and falls through to 3f: two transpose designs
    //        were tried on it and both lost, fixing the load while leaving the
    //        store scattered. --
    if (storage_offset == 0 && is_2d_transpose(dims, strides, ndim)) {
        const uint32_t M = (uint32_t)dims[0];
        const uint32_t N = (uint32_t)dims[1];
        const int64_t stride_n = strides[1];

        if (elem_size == 4)
            launch_transpose<float, 32, 32>(
                src, dst, M, N, stride_n, 0, (int64_t)M * N, 1, stream);
        else
            launch_transpose<uint16_t, 32, 32>(
                src, dst, M, N, stride_n, 0, (int64_t)M * N, 1, stream);

        cudaError_t terr = cudaGetLastError();
        if (terr != cudaSuccess) {
            printf("sm90 transpose_tiled_kernel launch failed: %s (M=%u, N=%u)\n",
                   cudaGetErrorString(terr), M, N);
        }
        return true;
    }

    // -- 3f. Inner-transposed: protect the store, let the load scatter --
    if (storage_offset == 0 && is_inner_transpose(dims, strides, ndim)) {
        ContiguousMeta fmeta = {};
        fmeta.ndim = ndim;
        fmeta.storage_offset_elems = storage_offset;
        fmeta.elem_size = elem_size;
        for (int d = 0; d < ndim && d < MaxDims; ++d) {
            fmeta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
            fmeta.strides[d] = strides[d];
            fmeta.dims64[d] = dims[d];
        }
        if (elem_size == 4)
            dispatch_3f<float>(src, dst, total_elems, ndim, use32, fmeta, stream);
        else
            dispatch_3f<uint16_t>(src, dst, total_elems, ndim, use32, fmeta, stream);

        cudaError_t ferr = cudaGetLastError();
        if (ferr != cudaSuccess) {
            printf("sm90 flat_strided_copy_kernel launch failed: %s "
                   "(total_elems=%lld, ndim=%d)\n",
                   cudaGetErrorString(ferr), (long long)total_elems, ndim);
        }
        return true;
    }

    // -- 3d. Generic fallback --
    ContiguousMeta meta = {};
    meta.ndim = ndim;
    meta.storage_offset_elems = storage_offset;
    meta.elem_size = elem_size;
    for (int d = 0; d < ndim && d < MaxDims; ++d) {
        meta.divmods[d] = FastDivmod::make((uint32_t)dims[d]);
        meta.strides[d] = strides[d];
        meta.dims64[d] = dims[d];
    }

    // The amortized walk needs a unit-stride innermost dim so consecutive
    // destination indices inside a row are consecutive source indices too.
    const bool inner_contig   = (strides[ndim - 1] == 1);
    const uint32_t inner_size = (uint32_t)dims[ndim - 1];
    FastDivmod inner_dm = FastDivmod::make(inner_size);

    if (elem_size == 4)
        dispatch_3d<float>(src, dst, total_elems, ndim, inner_size, inner_dm,
                           inner_contig, use32, meta, stream);
    else
        dispatch_3d<uint16_t>(src, dst, total_elems, ndim, inner_size, inner_dm,
                              inner_contig, use32, meta, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("sm90 generic_strided_copy_kernel launch failed: %s "
               "(total_elems=%lld, ndim=%d)\n",
               cudaGetErrorString(err), (long long)total_elems, ndim);
    }
    return true;
}

} // namespace OwnTensor
