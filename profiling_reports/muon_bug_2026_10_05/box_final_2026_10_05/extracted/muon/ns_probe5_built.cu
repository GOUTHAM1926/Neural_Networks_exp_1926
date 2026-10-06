// ns_probe5.cu -- is the REDUCE_BROADCAST-only "scaled" value (309adc60...) just another order of the 9 atomicAdds
// in muon_sumsq_kernel? Input: tensor 15's step-4 Newton-Schulz input dumped by run_debug2.sh.
//  part 1: the library's own kernel code (copied verbatim at build time from MuonKernels.cu, see run_probe5.sh) with the
//          atomicAdd replaced by "store my block's sum", gives the 9 block sums. The host adds them in all 9! orders,
//          the way atomicAdd does (one float add at a time onto 0), and runs the verbatim scale kernel for every distinct
//          total. Which totals give the RS-mode value b3f448f1, and which give the RB-mode value 309adc60?
//  part 2: the REAL library function muon_frobenius_scale_cuda, 200 calls per condition: idle GPU, L2 flushed first,
//          other kernels busy on another stream. Does the RB-mode value appear outside the test?
// Scratch diagnostic, 2026-10-05.   usage: ns_probe5 <nsin_step4.bin>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>
#include <cuda_runtime.h>

namespace OwnTensor { namespace cuda { void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps); } }

namespace copy {
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Pass 1: sum of squares reduced into a single device accumulator.
__global__ void muon_sumsq_kernel(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int64_t numel
) {
    // Shared memory for warp leaders (max 32 warps per block)
    __shared__ float sdata[32];
    
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // float4 (128-bit) vectorized loads + 4 independent accumulators to break
    // the loop-carried FMA dependency (4-way ILP). Requires the base pointer be
    // 16B-aligned; cudaMalloc guarantees >=256B and Muon always passes a whole
    // weight tensor's base pointer, so this holds. The scalar tail covers the
    // numel % 4 remainder.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int64_t numel4 = numel / 4;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    for (int64_t i = idx; i < numel4; i += stride) {
        float4 v = x4[i];
        s0 += v.x * v.x;
        s1 += v.y * v.y;
        s2 += v.z * v.z;
        s3 += v.w * v.w;
    }
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        float v = x[i];
        s0 += v * v;
    }
    float thread_sum = (s0 + s1) + (s2 + s3);

    // 1. Intra-warp reduction (no shared memory needed)
    thread_sum = warpReduceSum(thread_sum);

    // 2. Write warp leaders to shared memory
    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) {
        sdata[warpId] = thread_sum;
    }

    // 3. Wait for all warp leaders to write
    __syncthreads();

    // 4. First warp reduces the shared memory array
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        thread_sum = (tid < num_warps) ? sdata[lane] : 0.0f;
        thread_sum = warpReduceSum(thread_sum);
        
        // Thread 0 writes the final block sum to global memory
        if (tid == 0) {
            atomicAdd(acc, thread_sum);
        }
    }
}

// Pass 2: read the accumulated sum-of-squares, derive scale = 1/max(||X||_F,eps),
// and apply X *= scale in place (vectorized via float4).
__global__ void muon_scale_by_inv_frob_kernel(
    float* __restrict__ x,
    const float* __restrict__ acc,
    int64_t numel,
    float eps
) {
    // scale = 1/max(sqrt(acc), eps) == rsqrt(max(acc, eps*eps)) for eps >= 0
    // (sqrt is monotonic). Replaces a SQRT + division with one MUFU.RSQ.
    const float scale = rsqrtf(fmaxf(*acc, eps * eps));

    const int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int64_t numel4 = numel / 4;
    float4* __restrict__ x4 = reinterpret_cast<float4*>(x);   // hoisted cast

    // 2x ILP widening: two independent float4 loads in flight to overlap DRAM
    // latency. Coverage is exact (pairs + trailing chunk + scalar tail).
    int64_t i = idx;
    for (; i + stride < numel4; i += 2 * stride) {
        float4 a = x4[i];
        float4 b = x4[i + stride];
        a.x *= scale; a.y *= scale; a.z *= scale; a.w *= scale;
        b.x *= scale; b.y *= scale; b.z *= scale; b.w *= scale;
        x4[i]          = a;
        x4[i + stride] = b;
    }
    for (; i < numel4; i += stride) {
        float4 a = x4[i];
        a.x *= scale; a.y *= scale; a.z *= scale; a.w *= scale;
        x4[i] = a;
    }
    for (int64_t j = numel4 * 4 + idx; j < numel; j += stride) {
        x[j] *= scale;
    }
}


// ---- the same Pass 1, storing each block's sum instead of atomicAdd ----
__global__ void muon_sumsq_partials(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int64_t numel
) {
    // Shared memory for warp leaders (max 32 warps per block)
    __shared__ float sdata[32];
    
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // float4 (128-bit) vectorized loads + 4 independent accumulators to break
    // the loop-carried FMA dependency (4-way ILP). Requires the base pointer be
    // 16B-aligned; cudaMalloc guarantees >=256B and Muon always passes a whole
    // weight tensor's base pointer, so this holds. The scalar tail covers the
    // numel % 4 remainder.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int64_t numel4 = numel / 4;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    for (int64_t i = idx; i < numel4; i += stride) {
        float4 v = x4[i];
        s0 += v.x * v.x;
        s1 += v.y * v.y;
        s2 += v.z * v.z;
        s3 += v.w * v.w;
    }
    for (int64_t i = numel4 * 4 + idx; i < numel; i += stride) {
        float v = x[i];
        s0 += v * v;
    }
    float thread_sum = (s0 + s1) + (s2 + s3);

    // 1. Intra-warp reduction (no shared memory needed)
    thread_sum = warpReduceSum(thread_sum);

    // 2. Write warp leaders to shared memory
    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) {
        sdata[warpId] = thread_sum;
    }

    // 3. Wait for all warp leaders to write
    __syncthreads();

    // 4. First warp reduces the shared memory array
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        thread_sum = (tid < num_warps) ? sdata[lane] : 0.0f;
        thread_sum = warpReduceSum(thread_sum);
        
        // Thread 0 writes the final block sum to global memory
        if (tid == 0) {
            acc[blockIdx.x] = thread_sum;
        }
    }
}

// Pass 2: read the accumulated sum-of-squares, derive scale = 1/max(||X||_F,eps),
// and apply X *= scale in place (vectorized via float4).

}

static unsigned long long fp_dev(const float* d, int64_t n) {
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long long x = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (size_t i = 0; i < h.size() * sizeof(float); ++i) { x ^= b[i]; x *= 1099511628211ull; }
    return x;
}
static unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
__global__ void spin(long long cycles) { long long t = clock64(); while (clock64() - t < cycles) {} }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <nsin_step4.bin>\n", argv[0]); return 2; }
    const int64_t n = 96 * 96;
    std::vector<float> in(n);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(in.data(), sizeof(float), n, f) != (size_t)n) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    std::fclose(f);
    float *x0, *x, *acc, *part;
    cudaMalloc(&x0, n * 4); cudaMalloc(&x, n * 4); cudaMalloc(&acc, 4); cudaMalloc(&part, 64 * 4);
    cudaMemcpy(x0, in.data(), n * 4, cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = (int)((n / 4 + threads - 1) / threads);    // need4 in muon_frobenius_scale_cuda = 9 here
    std::printf("=== %s | 96x96 | the library launches muon_sumsq_kernel with %d blocks for this size\n", argv[1], blocks);

    // ---- part 1 ----
    copy::muon_sumsq_partials<<<blocks, threads>>>(x0, part, n);
    std::vector<float> p(blocks);
    cudaMemcpy(p.data(), part, blocks * 4, cudaMemcpyDeviceToHost);
    std::printf("part 1: the %d block sums:", blocks);
    for (float v : p) std::printf(" %.9g", v);
    std::printf("\n");
    std::vector<int> order(blocks);
    for (int i = 0; i < blocks; ++i) order[i] = i;
    std::map<unsigned, long> totals;          // float bits -> number of orders that give it
    unsigned in_block_order = 0;
    bool first = true;
    do {
        volatile float s = 0.0f;              // volatile: one IEEE float add per step, like atomicAdd
        for (int i : order) s = s + p[i];
        totals[bits(s)]++;
        if (first) { in_block_order = bits(s); first = false; }
    } while (std::next_permutation(order.begin(), order.end()));
    std::printf("part 1: %zu different totals over all add orders\n", totals.size());
    for (auto& [b, cnt] : totals) {
        float s; std::memcpy(&s, &b, 4);
        cudaMemcpy(acc, &s, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(x, x0, n * 4, cudaMemcpyDeviceToDevice);
        copy::muon_scale_by_inv_frob_kernel<<<blocks, threads>>>(x, acc, n, 1e-8f);
        const unsigned long long fpv = fp_dev(x, n);
        const char* tag = ((fpv >> 32) == 0xb3f448f1ull) ? "  <- the RS-mode / standalone value b3f448f1"
                        : ((fpv >> 32) == 0x309adc60ull) ? "  <- the RB-mode value 309adc60" : "";
        std::printf("   total %.9g (bits %08x) from %6ld orders%s -> scaled fingerprint %016llx%s\n", s, b, cnt,
                    b == in_block_order ? " [block order 0..8]" : "", fpv, tag);
    }

    // ---- part 2 ----
    cudaStream_t other; cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking);
    char* junk; const size_t junk_bytes = 512ull << 20; cudaMalloc(&junk, junk_bytes);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    auto trial = [&](const char* name, int mode) {
        std::map<unsigned long long, int> seen;
        for (int r = 0; r < 200; ++r) {
            cudaMemcpy(x, x0, n * 4, cudaMemcpyDeviceToDevice);
            if (mode == 1) cudaMemset(junk, r & 0xff, junk_bytes);                       // push x out of L2
            if (mode == 2) spin<<<sms - 4, 1024, 0, other>>>(2000000);                    // most SMs busy
            if (mode == 3) spin<<<(r % sms) + 1, 1024, 0, other>>>(200000 + 5000 * r);   // a varying number of SMs busy
            OwnTensor::cuda::muon_frobenius_scale_cuda(x, n, 1e-8f);
            seen[fp_dev(x, n)]++;
        }
        std::printf("   %-46s", name);
        for (auto& [k, c] : seen) std::printf("  %016llx x%d", k, c);
        std::printf("\n");
    };
    std::printf("part 2: the real library muon_frobenius_scale_cuda, 200 calls each (fingerprint x times seen)\n");
    trial("idle GPU", 0);
    trial("L2 flushed before each call", 1);
    trial("all but 4 SMs busy on another stream", 2);
    trial("1..132 SMs busy on another stream (varies)", 3);
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
