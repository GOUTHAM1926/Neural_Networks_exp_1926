// ns_probe6.cu -- choose the fix for muon_frobenius_scale_cuda by MEASUREMENT: determinism, precision, speed.
// The library's kernels are copied VERBATIM from MuonKernels.cu at build time (run_probe6.sh); the candidate fixes
// are made from those same kernels by small, asserted text edits, so per-thread and per-block arithmetic is identical
// and only the ACROSS-BLOCK step differs.
//   V0  library today: memset acc=0, sumsq kernel (atomicAdd per block), scale kernel             3 GPU ops
//   V1  sumsq kernel stores block sums; 1-block finalize kernel adds them in FIXED order; scale   3 GPU ops
//   V2  sumsq kernel stores block sums; the LAST block to finish adds them in fixed order; scale 2 GPU ops
//   V3  sumsq kernel stores block sums; every scale block adds them in fixed order itself         2 GPU ops
//   V3n V3 with the block count a function of numel only (cap 1024), not of the GPU's SM count  2 GPU ops
//   V1n V1 with the block count a function of numel only (cap 1024) = Navin's ZeroKernels design  3 GPU ops
// For each: (1) how many different totals over 400 calls in 4 GPU conditions (idle / L2 flushed / busy SMs /
// varying busy SMs); (2) error of the total vs an exact fp64 sum; (3) microseconds per call.
// The scaled output is x * rsqrtf(total), elementwise, so the same total means the same output.
// Scratch diagnostic, 2026-10-05.   usage: ns_probe6 [real_input.bin]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

namespace copy {
// Fixed-order sum of n partials by one whole block: thread-strided, then a shuffle tree, then the warp leaders.
// Every thread returns the same value; the order depends only on n and blockDim, never on timing.
__device__ float fixed_order_total(const float* __restrict__ p, int n, bool bypass_l1) {
    __shared__ float wsum[32];
    __shared__ float total;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += bypass_l1 ? __ldcg(p + i) : p[i];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) wsum[w] = v;
    __syncthreads();
    if (w == 0) {
        const int nw = (blockDim.x + 31) / 32;
        v = (lane < nw) ? wsum[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) total = v;
    }
    __syncthreads();
    return total;
}
__global__ void finalize_kernel(const float* __restrict__ p, int n, float* __restrict__ acc) {
    const float t = fixed_order_total(p, n, false);
    if (threadIdx.x == 0) *acc = t;
}
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


// ===== candidate fixes, made from the verbatim kernels above =====
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

__global__ void muon_sumsq_lastblock(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int64_t numel,
    float* __restrict__ partials,
    unsigned int* __restrict__ counter
) {
    __shared__ bool s_last;
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
            partials[blockIdx.x] = thread_sum;
            __threadfence();
            s_last = (atomicInc(counter, gridDim.x - 1) == gridDim.x - 1);   // the last one also resets it to 0
        }
    }
    __syncthreads();
    if (s_last) {
        const float t = fixed_order_total(partials, gridDim.x, true);
        if (threadIdx.x == 0) *acc = t;
    }
}

// Pass 2: read the accumulated sum-of-squares, derive scale = 1/max(||X||_F,eps),
// and apply X *= scale in place (vectorized via float4).

__global__ void muon_scale_fixedsum_kernel(
    float* __restrict__ x,
    const float* __restrict__ partials,
    int nparts,
    int64_t numel,
    float eps,
    float* __restrict__ dbg_total
) {
    // scale = 1/max(sqrt(acc), eps) == rsqrt(max(acc, eps*eps)) for eps >= 0
    // (sqrt is monotonic). Replaces a SQRT + division with one MUFU.RSQ.
    const float total_fixed = fixed_order_total(partials, nparts, false);
    if (dbg_total && blockIdx.x == 0 && threadIdx.x == 0) *dbg_total = total_fixed;
    const float scale = rsqrtf(fmaxf(total_fixed, eps * eps));

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


}

static unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
__global__ void spin(long long cycles) { long long t = clock64(); while (clock64() - t < cycles) {} }
__global__ void fill(float* x, int64_t n, uint64_t seed) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t s = seed * 0x9E3779B97F4A7C15ull + (uint64_t)i;
        s ^= s >> 33; s *= 0xff51afd7ed558ccdull; s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull; s ^= s >> 33;
        x[i] = (static_cast<float>(static_cast<int64_t>(s % 2000001) - 1000000) * 1e-6f) * 0.02f;   // like a momentum
    }
}

struct Ctx {
    cudaStream_t s; float* acc; float* partials; unsigned* counter; float* dbg;
    int sm = 0, nbps_red = 0, nbps_sc = 0;
};
static int red_blocks_lib(const Ctx& c, int64_t n) {          // the library's formula (MuonKernels.cu)
    const int64_t need4 = (n / 4 + 256 - 1) / 256;
    return (int)std::max<int64_t>(1, std::min<int64_t>((int64_t)c.nbps_red * c.sm, need4));
}
static int scale_blocks_lib(const Ctx& c, int64_t n) {
    const int64_t need4 = (n / 4 + 256 - 1) / 256;
    return (int)std::max<int64_t>(1, std::min<int64_t>((int64_t)c.nbps_sc * c.sm, need4));
}
static int g_cap = 1024;                                        // the numel-only cap; argv[2] overrides it
static int red_blocks_n_only(int64_t n) {                       // a function of numel alone
    const int64_t need4 = (n / 4 + 256 - 1) / 256;
    return (int)std::max<int64_t>(1, std::min<int64_t>(g_cap, need4));
}
// One call of each variant. Returns nothing; the total is read separately (outside timing).
static void call(int v, Ctx& c, float* x, int64_t n, float eps) {
    const int rb = (v == 4 || v == 5) ? red_blocks_n_only(n) : red_blocks_lib(c, n), sb = scale_blocks_lib(c, n);
    switch (v) {
    case 0:
        cudaMemsetAsync(c.acc, 0, sizeof(float), c.s);
        copy::muon_sumsq_kernel<<<rb, 256, 0, c.s>>>(x, c.acc, n);
        copy::muon_scale_by_inv_frob_kernel<<<sb, 256, 0, c.s>>>(x, c.acc, n, eps);
        break;
    case 1: case 5:
        copy::muon_sumsq_partials<<<rb, 256, 0, c.s>>>(x, c.partials, n);
        copy::finalize_kernel<<<1, 256, 0, c.s>>>(c.partials, rb, c.acc);
        copy::muon_scale_by_inv_frob_kernel<<<sb, 256, 0, c.s>>>(x, c.acc, n, eps);
        break;
    case 2:
        copy::muon_sumsq_lastblock<<<rb, 256, 0, c.s>>>(x, c.acc, n, c.partials, c.counter);
        copy::muon_scale_by_inv_frob_kernel<<<sb, 256, 0, c.s>>>(x, c.acc, n, eps);
        break;
    case 3: case 4:
        copy::muon_sumsq_partials<<<rb, 256, 0, c.s>>>(x, c.partials, n);
        copy::muon_scale_fixedsum_kernel<<<sb, 256, 0, c.s>>>(x, c.partials, rb, n, eps, c.dbg);
        break;
    }
}
static float total_of(int v, Ctx& c) {                           // the sum of squares this call used
    CK(cudaStreamSynchronize(c.s));
    float t; CK(cudaMemcpy(&t, (v == 3 || v == 4) ? c.dbg : c.acc, 4, cudaMemcpyDeviceToHost));
    return t;
}

int main(int argc, char** argv) {
    if (argc > 2) g_cap = std::atoi(argv[2]);
    const char* names[] = {"V0 library today (atomicAdd)", "V1 fixed order, 1-block finalize", "V2 fixed order, last block",
                           "V3 fixed order, inside scale", "V3n V3 + blocks from numel only",
                           "V1n V1 + blocks from numel only"};
    const int NV = 6;
    Ctx c;
    CK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    CK(cudaMalloc(&c.acc, 4)); CK(cudaMalloc(&c.dbg, 4)); CK(cudaMalloc(&c.partials, 8192 * 4));
    CK(cudaMalloc(&c.counter, 4)); CK(cudaMemset(c.counter, 0, 4));
    int dev = 0; CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&c.sm, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c.nbps_red, copy::muon_sumsq_kernel, 256, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c.nbps_sc, copy::muon_scale_by_inv_frob_kernel, 256, 0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("=== %s | %d SMs | library grid cap = %d blocks/SM x %d SMs = %d | numel-only cap %d\n", prop.name, c.sm, c.nbps_red, c.sm, c.nbps_red * c.sm, g_cap);

    cudaStream_t other; CK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
    char* junk; const size_t junk_bytes = 512ull << 20; CK(cudaMalloc(&junk, junk_bytes));

    struct Shape { int64_t r, c; const char* tag; };
    std::vector<Shape> shapes = {{96, 96, "test tensor 15"}, {768, 768, "GPT-2 attn"}, {768, 3072, "GPT-2 MLP"},
                                 {2048, 2048, ""}, {2048, 8192, ""}, {4096, 4096, ""}, {4096, 14336, "large MLP"}};
    std::vector<float> real;
    if (argc > 1 && std::string(argv[1]) != "-") {
        FILE* f = std::fopen(argv[1], "rb");
        real.resize(96 * 96);
        if (!f || std::fread(real.data(), 4, real.size(), f) != real.size()) { std::printf("cannot read %s\n", argv[1]); return 2; }
        std::fclose(f);
        shapes.insert(shapes.begin(), Shape{96, 96, "REAL tensor-15 step-4 input"});
    }
    for (size_t si = 0; si < shapes.size(); ++si) {
        const int64_t n = shapes[si].r * shapes[si].c;
        float *x0, *x; CK(cudaMalloc(&x0, n * 4)); CK(cudaMalloc(&x, n * 4));
        if (si == 0 && !real.empty()) CK(cudaMemcpy(x0, real.data(), n * 4, cudaMemcpyHostToDevice));
        else { fill<<<1024, 256>>>(x0, n, 1000 + si); CK(cudaDeviceSynchronize()); }
        std::vector<float> h(n); CK(cudaMemcpy(h.data(), x0, n * 4, cudaMemcpyDeviceToHost));
        long double exact = 0; for (float v : h) exact += (long double)v * v;     // exact enough: 64-bit mantissa
        const float exact_f = (float)exact;
        std::printf("\n--- %lldx%lld %s | %lld elements | library blocks %d, numel-only blocks %d | exact sum %.10Lg (nearest float %08x)\n",
                    (long long)shapes[si].r, (long long)shapes[si].c, shapes[si].tag, (long long)n,
                    red_blocks_lib(c, n), red_blocks_n_only(n), exact, bits(exact_f));
        // (1) determinism and (2) precision
        for (int v = 0; v < NV; ++v) {
            std::map<unsigned, int> seen;
            double worst_rel = 0;
            for (int cond = 0; cond < 4; ++cond)
                for (int r = 0; r < 100; ++r) {
                    CK(cudaMemcpyAsync(x, x0, n * 4, cudaMemcpyDeviceToDevice, c.s));
                    CK(cudaStreamSynchronize(c.s));
                    if (cond == 1) CK(cudaMemset(junk, r & 0xff, junk_bytes));
                    if (cond == 2) spin<<<c.sm - 4, 1024, 0, other>>>(2000000);
                    if (cond == 3) spin<<<(r % c.sm) + 1, 1024, 0, other>>>(200000 + 5000 * r);
                    call(v, c, x, n, 1e-8f);
                    const float t = total_of(v, c);
                    seen[bits(t)]++;
                    worst_rel = std::max(worst_rel, (double)std::fabs(((long double)t - exact) / exact));
                    CK(cudaDeviceSynchronize());
                }
            std::printf("   %-33s %d different total(s) in 400 calls |", names[v], (int)seen.size());
            for (auto& [b, k] : seen) { float t; std::memcpy(&t, &b, 4); std::printf(" %08x x%d", b, k); }
            std::printf(" | worst rel error vs exact %.2e (%.2f ulp of the total)\n", worst_rel,
                        worst_rel * (double)exact / (std::ldexp(1.0, std::ilogb(exact_f) - 23)));
        }
        // (3) speed: back-to-back calls on an idle GPU, variants interleaved, 5 rounds, median
        const int calls = n <= (1 << 20) ? 2000 : (n <= (1 << 24) ? 500 : 100);
        std::vector<std::vector<float>> us(NV);
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int round = 0; round < 5; ++round)
            for (int v = 0; v < NV; ++v) {
                for (int w = 0; w < 20; ++w) call(v, c, x, n, 1e-8f);
                CK(cudaEventRecord(e0, c.s));
                for (int k = 0; k < calls; ++k) call(v, c, x, n, 1e-8f);
                CK(cudaEventRecord(e1, c.s)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                us[v].push_back(ms * 1000.0f / calls);
            }
        std::printf("   speed, us per call (median of 5 rounds, %d calls each):", calls);
        for (int v = 0; v < NV; ++v) {
            std::sort(us[v].begin(), us[v].end());
            std::printf("  V%s %.2f", v == 4 ? "3n" : v == 5 ? "1n" : std::to_string(v).c_str(), us[v][2]);
        }
        std::printf("\n");
        CK(cudaFree(x0)); CK(cudaFree(x));
    }
    std::printf("\n=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
