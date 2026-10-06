#!/usr/bin/env python3
"""V1n fix for muon_frobenius_scale_cuda (Tensor-Implementations src/Kernels/cuda/optimizer/MuonKernels.cu).

The sum of squares stops using atomicAdd across blocks: each block writes its own slot, one block adds the slots in
a fixed order, and the reduction's block count depends on numel only. Same pattern as dist/zero ZeroKernels.cu
(sumsq_partial_kernel + sumsq_finalize_kernel). Guarded by the md5 of the file it was written against (TI a60099b);
every edit must match exactly once.   usage: patch_muon_v1n.py <MuonKernels.cu>
"""
import hashlib
import sys

EXPECTED_MD5 = "f6d821c2c64cb1f39e04625e2a74ee47"   # TI a60099b, dist_side_merge_BLUCCL

EDITS = [
# ---- Pass 1: one partial per block instead of atomicAdd --------------------------------------------------------
("""// Pass 1: sum of squares reduced into a single device accumulator.
__global__ void muon_sumsq_kernel(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int64_t numel
) {""",
"""// Pass 1: sum of squares, one partial per block (partials[blockIdx.x]).
__global__ void muon_sumsq_kernel(
    const float* __restrict__ x,
    float* __restrict__ partials,
    int64_t numel
) {"""),

("""        // Thread 0 writes the final block sum to global memory
        if (tid == 0) {
            atomicAdd(acc, thread_sum);
        }
    }
}
""",
"""        // Thread 0 writes this block's sum to its own slot (no atomicAdd, see Pass 1b)
        if (tid == 0) {
            partials[blockIdx.x] = thread_sum;
        }
    }
}

// Pass 1b: add the block partials in ONE FIXED ORDER (thread-strided, then the same
// warp/shared tree as Pass 1) and write the total. One block, so the order depends
// only on the number of partials -- never on which block of Pass 1 finished first.
//
// Why not atomicAdd into one float: float addition depends on order, and the order
// blocks reach an atomic changes with GPU state (cache contents, other work on the
// SMs). The same input then summed to different totals from call to call (H200,
// 400 calls: 3 different totals for 96x96, ~20 for >= 768x768, up to ~19 ulp
// apart). One ulp of the norm is enough to flip bf16 roundings in Newton-Schulz,
// so the same gradient gave different Muon updates; this is what made dist/zero
// test_muon_rsag (bit-exact REDUCE_BROADCAST vs REDUCE_SCATTER_ALL_GATHER at W=1)
// fail intermittently. Same pattern as dist/zero ZeroKernels.cu sumsq_finalize.
__global__ void muon_sumsq_finalize_kernel(
    const float* __restrict__ partials,
    int nparts,
    float* __restrict__ acc
) {
    __shared__ float sdata[32];

    int tid = threadIdx.x;
    float v = 0.0f;
    for (int i = tid; i < nparts; i += blockDim.x) {
        v += partials[i];
    }
    v = warpReduceSum(v);

    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        sdata[warpId] = v;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        v = (tid < num_warps) ? sdata[lane] : 0.0f;
        v = warpReduceSum(v);
        if (tid == 0) {
            *acc = v;   // assignment, not +=: no memset of acc needed
        }
    }
}
"""),

# ---- scratch buffers + the numel-only block count ---------------------------------------------------------------
("""// Persistent single-float scratch for the norm accumulator (one-time alloc).
static float* s_muon_norm = nullptr;

void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps) {
    if (numel == 0) return;
    if (!s_muon_norm) cudaMalloc(&s_muon_norm, sizeof(float));

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
    cudaMemsetAsync(s_muon_norm, 0, sizeof(float), stream);

    const int threads = 256;

    // Occupancy-based grid: fill every SM instead of the old fixed 1024-block
    // cap. Both kernels are grid-stride, so any block count is correct; sizing
    // to blocksPerSM * SM_count keeps the (now float4) reduction from being
    // launch/latency-bound on the many small GPT-2 weight matrices. Capped at
    // the float4 blocks actually needed so we never over-subscribe.
    //
    // dev/sm_count/nbps_red/nbps_sc are invariant for the lifetime of the""",
"""// Persistent scratch (one-time alloc): the Pass 1 partials and the total. Shared by
// every call, so calls must stay on one stream (as with the single accumulator before).
static float* s_muon_norm = nullptr;
static float* s_muon_partials = nullptr;

// Pass 1 block count is a function of numel ONLY, never of the GPU, so a tensor is
// summed in the same order -- to the same bits -- on any GPU. 768 fits in one wave on
// H200 (6 blocks/SM x 132 SMs = 792 resident). Measured on H200: 1024 forced a second,
// mostly empty wave (+7.7 us on 4096x14336); 256 and 512 under-filled the large tensors.
static constexpr int kMuonSumsqMaxBlocks = 768;

void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps) {
    if (numel == 0) return;
    if (!s_muon_norm) {
        cudaMalloc(&s_muon_norm, sizeof(float));
        cudaMalloc(&s_muon_partials, kMuonSumsqMaxBlocks * sizeof(float));
    }

    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    const int threads = 256;

    // Scale pass: occupancy-based grid -- fill every SM, capped at the float4 blocks
    // actually needed so we never over-subscribe. It is elementwise and grid-stride,
    // so its block count never changes the result (the reduction's would, which is
    // why Pass 1 uses kMuonSumsqMaxBlocks instead).
    //
    // dev/sm_count/nbps_sc are invariant for the lifetime of the"""),

("""    static int sm_count = 0;
    static int nbps_red = 0;
    static int nbps_sc = 0;
    static bool occ_cached = false;
    if (!occ_cached) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbps_red, muon_sumsq_kernel, threads, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbps_sc, muon_scale_by_inv_frob_kernel, threads, 0);
        occ_cached = true;
    }
    const int64_t need4 = (numel / 4 + threads - 1) / threads;

    int red_blocks = (int)std::max<int64_t>(
        1, std::min<int64_t>((int64_t)nbps_red * sm_count, need4));
    muon_sumsq_kernel<<<red_blocks, threads, 0, stream>>>(
        x, s_muon_norm, numel);

    int scale_blocks = (int)std::max<int64_t>(
        1, std::min<int64_t>((int64_t)nbps_sc * sm_count, need4));
    // Same stream => the reduction is guaranteed complete before the scale reads acc.
    muon_scale_by_inv_frob_kernel<<<scale_blocks, threads, 0, stream>>>(
        x, s_muon_norm, numel, eps);
}
""",
"""    static int sm_count = 0;
    static int nbps_sc = 0;
    static bool occ_cached = false;
    if (!occ_cached) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbps_sc, muon_scale_by_inv_frob_kernel, threads, 0);
        occ_cached = true;
    }
    const int64_t need4 = (numel / 4 + threads - 1) / threads;

    const int red_blocks = (int)std::max<int64_t>(
        1, std::min<int64_t>((int64_t)kMuonSumsqMaxBlocks, need4));
    muon_sumsq_kernel<<<red_blocks, threads, 0, stream>>>(
        x, s_muon_partials, numel);
    muon_sumsq_finalize_kernel<<<1, threads, 0, stream>>>(
        s_muon_partials, red_blocks, s_muon_norm);

    int scale_blocks = (int)std::max<int64_t>(
        1, std::min<int64_t>((int64_t)nbps_sc * sm_count, need4));
    // Same stream => the total is written before the scale reads acc.
    muon_scale_by_inv_frob_kernel<<<scale_blocks, threads, 0, stream>>>(
        x, s_muon_norm, numel, eps);
}
"""),
]


def main():
    path = sys.argv[1]
    data = open(path, "rb").read()
    md5 = hashlib.md5(data).hexdigest()
    if md5 != EXPECTED_MD5:
        sys.exit(f"REFUSED: {path} md5 {md5} != {EXPECTED_MD5} (not the file this patch was written against)")
    s = data.decode()
    for i, (old, new) in enumerate(EDITS, 1):
        c = s.count(old)
        if c != 1:
            sys.exit(f"REFUSED: edit {i} matched {c} times (expected 1): {old[:70]!r}")
        s = s.replace(old, new)
    for gone in ("atomicAdd(acc", "cudaMemsetAsync(s_muon_norm", "nbps_red"):
        if gone in s:
            sys.exit(f"REFUSED: {gone!r} still present after patching")
    open(path, "w").write(s)
    print(f"patched {path}: {len(EDITS)} edits, each matched once; new md5 {hashlib.md5(s.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
