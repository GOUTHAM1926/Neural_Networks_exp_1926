// test_types_conversions_gpu.cu -- GPU check of the patched include/dtype/Types.h (2026-09-26).
//
// The GPU (device) path of every conversion must give the same bits as the CPU (host)
// path of the same header. The CPU path was checked separately against PyTorch, Eigen,
// NumPy, the FP16 library, x86 F16C, GCC and CUDA's host code (all 2^32 fp32 inputs).
//   G1 fp32 -> bf16      all 2^32        bfloat16_t(float)
//   G2 fp32 -> fp16      all 2^32        float16_t(float)
//   G3 bf16/fp16 -> fp32 all 2^16 each   operator float
//   G4 double -> bf16/fp16  every bf16/fp16 step (midpoint, neighbours) + 2^26 random
//   G5 int32/uint32 -> bf16 and fp16   all 2^32 each
//   G6 int64/uint64 -> bf16/fp16   step boundaries at every exponent + 2^26 random
//   G7 fp16 <-> bf16: the library's path (through fp32) vs the hardware's direct
//      cvt.rn.bf16.f16 / cvt.rn.f16.bf16 (sm_90 and newer), all 2^16 each
// For NaN input the GPU output must be a NaN; the NaN bit patterns the GPU produced are
// printed (the CPU path returns 0x7FFF).
//
// Build + run from the Tensor-Implementations root (H200 = sm_90a; same flags as the library):
//   nvcc -std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
//        -Xcompiler -fopenmp -Iinclude -o /tmp/test_types_gpu Tests/kernel_tests/test_types_conversions_gpu.cu
//   CUDA_VISIBLE_DEVICES=<gpu> /tmp/test_types_gpu
#include "dtype/Types.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

using OwnTensor::bfloat16_t;
using OwnTensor::float16_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class To, class From> static inline __host__ __device__ To bc(const From& f) { To t; memcpy(&t, &f, sizeof(To)); return t; }
static inline bool nan_bf(uint16_t h) { return (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu); }
static inline bool nan_h(uint16_t h)  { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }

// ---------------- kernels (device path of Types.h) ----------------
__global__ void k_f2bf(uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = bfloat16_t(bc<float>(base + i)).raw_bits; }
__global__ void k_f2h (uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = float16_t(bc<float>(base + i)).raw_bits; }
__global__ void k_i2bf(uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = bfloat16_t((int32_t)(base + i)).raw_bits; }
__global__ void k_u2bf(uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = bfloat16_t((uint32_t)(base + i)).raw_bits; }
__global__ void k_u2h (uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = float16_t((uint32_t)(base + i)).raw_bits; }
__global__ void k_i2h (uint32_t base, uint32_t n, uint16_t* o) { for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) o[i] = float16_t((int32_t)(base + i)).raw_bits; }
__global__ void k_to_f32(uint32_t* obf, uint32_t* oh) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= 65536) return;
    bfloat16_t b; b.raw_bits = (uint16_t)i; float16_t h; h.raw_bits = (uint16_t)i;
    obf[i] = bc<uint32_t>(static_cast<float>(b)); oh[i] = bc<uint32_t>(static_cast<float>(h));
}
__global__ void k_d(const double* in, size_t n, uint16_t* obf, uint16_t* oh) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { obf[i] = bfloat16_t(in[i]).raw_bits; oh[i] = float16_t(in[i]).raw_bits; }
}
__global__ void k_64(const uint64_t* in, size_t n, uint16_t* os, uint16_t* ou, uint16_t* oh, uint16_t* ouh) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        os[i] = bfloat16_t((int64_t)in[i]).raw_bits; ou[i] = bfloat16_t((uint64_t)in[i]).raw_bits; oh[i] = float16_t((int64_t)in[i]).raw_bits;
        ouh[i] = float16_t((uint64_t)in[i]).raw_bits;
    }
}
__global__ void k_cross(uint16_t* h2b_lib, uint16_t* h2b_hw, uint16_t* b2h_lib, uint16_t* b2h_hw) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= 65536) return;
    float16_t h; h.raw_bits = (uint16_t)i; bfloat16_t b; b.raw_bits = (uint16_t)i;
    h2b_lib[i] = bfloat16_t(static_cast<float>(h)).raw_bits;          // the library's path: through fp32
    b2h_lib[i] = float16_t(static_cast<float>(b)).raw_bits;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint16_t r1, r2;
    asm("cvt.rn.bf16.f16 %0, %1;" : "=h"(r1) : "h"((uint16_t)i));
    asm("cvt.rn.f16.bf16 %0, %1;" : "=h"(r2) : "h"((uint16_t)i));
    h2b_hw[i] = r1; b2h_hw[i] = r2;
#else
    h2b_hw[i] = 0; b2h_hw[i] = 0;                                     // not available before sm_90 (G7 is skipped there)
#endif
}

// ---------------- tally ----------------
struct Tally { const char* name; long long bad = 0, nanbad = 0; uint64_t first = ~0ull; std::map<uint16_t, long long> gpu_nan; };
static void print(const Tally& t, long long n) {
    std::printf("   %-44s inputs %11lld  wrong %lld  NaN-not-NaN %lld", t.name, n, t.bad, t.nanbad);
    if (t.bad) std::printf("  first input 0x%llx", (unsigned long long)t.first);
    if (!t.gpu_nan.empty()) { std::printf("  GPU NaN bits:"); int k = 0; for (auto& p : t.gpu_nan) { if (k++ < 4) std::printf(" 0x%04x x%lld", p.first, p.second); } if (t.gpu_nan.size() > 4) std::printf(" (+%zu more)", t.gpu_nan.size() - 4); }
    std::printf("\n");
}
// compare one 16-bit output chunk: gpu[i] vs host(i); is_nan_in(i) says whether input i is a NaN
template <class Host, class NanIn, class NanOut>
static void cmp16(Tally& t, const uint16_t* gpu, uint64_t n, uint64_t key0, Host host, NanIn nan_in, NanOut nan_out) {
    long long bad = 0, nb = 0; uint64_t first = ~0ull; std::map<uint16_t, long long> nans;
    #pragma omp parallel
    {
        long long b_ = 0, nb_ = 0; uint64_t f_ = ~0ull; std::map<uint16_t, long long> m_;
        #pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; ++i) {
            if (nan_in(i)) { if (!nan_out(gpu[i])) ++nb_; else m_[gpu[i]]++; continue; }   // count each NaN bit pattern the GPU produced
            if (gpu[i] != host(i)) { ++b_; if (key0 + i < f_) f_ = key0 + i; }
        }
        #pragma omp critical
        { bad += b_; nb += nb_; if (f_ < first) first = f_; for (auto& p : m_) nans[p.first] += p.second; }
    }
    t.bad += bad; t.nanbad += nb; if (first < t.first) t.first = first; for (auto& p : nans) t.gpu_nan[p.first] += p.second;
}

int main() {
    int fails = 0;
    auto need0 = [&](const Tally& t) { if (t.bad || t.nanbad) { ++fails; std::printf("   ^^^ FAIL %s\n", t.name); } };
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s  sm_%d%d\n", prop.name, prop.major, prop.minor);
    const uint32_t CH = 1u << 28;                       // 268M outputs per chunk (512 MB)
    uint16_t* d16; CK(cudaMalloc(&d16, (size_t)CH * 2)); std::vector<uint16_t> h16(CH);

    // ---- G1, G2, G5: 2^32 sweeps ----
    struct Sweep { const char* name; void (*k)(uint32_t, uint32_t, uint16_t*); int kind; };
    const Sweep sw[6] = {{"G1 fp32->bf16  bfloat16_t(float)", k_f2bf, 0}, {"G2 fp32->fp16  float16_t(float)", k_f2h, 1},
                         {"G5 int32->bf16  bfloat16_t(int32)", k_i2bf, 2}, {"G5 uint32->bf16 bfloat16_t(uint32)", k_u2bf, 3},
                         {"G5 int32->fp16  float16_t(int32)", k_i2h, 4}, {"G5 uint32->fp16 float16_t(uint32)", k_u2h, 5}};
    for (const Sweep& s : sw) {
        Tally t; t.name = s.name;
        for (uint64_t base = 0; base < (1ull << 32); base += CH) {
            s.k<<<4096, 256>>>((uint32_t)base, CH, d16); CK(cudaGetLastError()); CK(cudaMemcpy(h16.data(), d16, (size_t)CH * 2, cudaMemcpyDeviceToHost));
            const uint32_t b0 = (uint32_t)base;
            switch (s.kind) {
                case 0: cmp16(t, h16.data(), CH, base, [&](long long i) { return bfloat16_t(bc<float>(b0 + (uint32_t)i)).raw_bits; },
                              [&](long long i) { return std::isnan(bc<float>(b0 + (uint32_t)i)); }, nan_bf); break;
                case 1: cmp16(t, h16.data(), CH, base, [&](long long i) { return float16_t(bc<float>(b0 + (uint32_t)i)).raw_bits; },
                              [&](long long i) { return std::isnan(bc<float>(b0 + (uint32_t)i)); }, nan_h); break;
                case 2: cmp16(t, h16.data(), CH, base, [&](long long i) { return bfloat16_t((int32_t)(b0 + (uint32_t)i)).raw_bits; }, [](long long) { return false; }, nan_bf); break;
                case 3: cmp16(t, h16.data(), CH, base, [&](long long i) { return bfloat16_t((uint32_t)(b0 + (uint32_t)i)).raw_bits; }, [](long long) { return false; }, nan_bf); break;
                case 4: cmp16(t, h16.data(), CH, base, [&](long long i) { return float16_t((int32_t)(b0 + (uint32_t)i)).raw_bits; }, [](long long) { return false; }, nan_h); break;
                case 5: cmp16(t, h16.data(), CH, base, [&](long long i) { return float16_t((uint32_t)(b0 + (uint32_t)i)).raw_bits; }, [](long long) { return false; }, nan_h); break;
            }
        }
        print(t, 1ll << 32); need0(t);
    }
    // ---- G3: 16-bit -> fp32, bitwise ----
    {
        uint32_t *dbf, *dh; CK(cudaMalloc(&dbf, 65536 * 4)); CK(cudaMalloc(&dh, 65536 * 4));
        k_to_f32<<<256, 256>>>(dbf, dh); CK(cudaGetLastError());
        std::vector<uint32_t> gbf(65536), gh(65536); CK(cudaMemcpy(gbf.data(), dbf, 65536 * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gh.data(), dh, 65536 * 4, cudaMemcpyDeviceToHost));
        Tally a, b; a.name = "G3 bf16->fp32 (bitwise, NaN payload too)"; b.name = "G3 fp16->fp32 (bitwise; NaN: must be NaN)";
        for (uint32_t i = 0; i < 65536; ++i) {
            bfloat16_t x; x.raw_bits = (uint16_t)i; float16_t y; y.raw_bits = (uint16_t)i;
            if (gbf[i] != bc<uint32_t>(static_cast<float>(x))) { ++a.bad; if (i < a.first) a.first = i; }
            const float hf = static_cast<float>(y);
            if (std::isnan(hf)) { if (!std::isnan(bc<float>(gh[i]))) ++b.nanbad; }
            else if (gh[i] != bc<uint32_t>(hf)) { ++b.bad; if (i < b.first) b.first = i; }
        }
        print(a, 65536); print(b, 65536); need0(a); need0(b);
        CK(cudaFree(dbf)); CK(cudaFree(dh));
    }
    // ---- G4: double -> 16 bit ----
    {
        std::vector<double> in;
        auto add = [&](double x) { in.push_back(x); in.push_back(-x); };
        for (uint32_t c = 0; c < 0x7F80u; ++c) {
            bfloat16_t lo_; lo_.raw_bits = (uint16_t)c; bfloat16_t hi_; hi_.raw_bits = (uint16_t)(c + 1);
            const double lo = (double)static_cast<float>(lo_), hi = (c + 1 == 0x7F80u) ? std::ldexp(1.0, 128) : (double)static_cast<float>(hi_), m = (lo + hi) / 2;
            add(lo); add(m); add(std::nextafter(m, 1e300)); add(std::nextafter(m, -1e300)); add(m * (1 + 0x1p-30)); add(m * (1 - 0x1p-30));
        }
        for (uint32_t c = 0; c < 0x7C00u; ++c) {
            float16_t lo_; lo_.raw_bits = (uint16_t)c; float16_t hi_; hi_.raw_bits = (uint16_t)(c + 1);
            const double lo = (double)static_cast<float>(lo_), hi = (c + 1 == 0x7C00u) ? 65536.0 : (double)static_cast<float>(hi_), m = (lo + hi) / 2;
            add(lo); add(m); add(std::nextafter(m, 1e300)); add(std::nextafter(m, -1e300)); add(m * (1 + 0x1p-40)); add(m * (1 - 0x1p-40));
        }
        uint64_t z = 0x243F6A8885A308D3ull;
        for (int i = 0; i < (1 << 26); ++i) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; in.push_back(bc<double>(i & 1 ? z : ((z & 0x800FFFFFFFFFFFFFull) | ((uint64_t)(1023 - 140 + (z >> 52) % 271) << 52)))); }
        in.push_back(NAN); in.push_back(INFINITY); in.push_back(-INFINITY); in.push_back(0.0); in.push_back(-0.0);
        const size_t n = in.size();
        double* din; uint16_t *dbf, *dh; CK(cudaMalloc(&din, n * 8)); CK(cudaMalloc(&dbf, n * 2)); CK(cudaMalloc(&dh, n * 2));
        CK(cudaMemcpy(din, in.data(), n * 8, cudaMemcpyHostToDevice));
        k_d<<<4096, 256>>>(din, n, dbf, dh); CK(cudaGetLastError());
        std::vector<uint16_t> gbf(n), gh(n); CK(cudaMemcpy(gbf.data(), dbf, n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gh.data(), dh, n * 2, cudaMemcpyDeviceToHost));
        Tally a, b; a.name = "G4 double->bf16  bfloat16_t(double)"; b.name = "G4 double->fp16  float16_t(double)";
        cmp16(a, gbf.data(), n, 0, [&](long long i) { return bfloat16_t(in[i]).raw_bits; }, [&](long long i) { return std::isnan(in[i]); }, nan_bf);
        cmp16(b, gh.data(), n, 0, [&](long long i) { return float16_t(in[i]).raw_bits; }, [&](long long i) { return std::isnan(in[i]); }, nan_h);
        print(a, (long long)n); print(b, (long long)n); need0(a); need0(b);
        CK(cudaFree(din)); CK(cudaFree(dbf)); CK(cudaFree(dh));
    }
    // ---- G6: 64-bit integers ----
    {
        std::vector<uint64_t> in;
        for (int e = 0; e < 64; ++e) for (uint64_t mant = 0; mant < 2048; ++mant) {
            const uint64_t base = (e >= 11) ? ((2048 + mant) << (e - 11)) : mant, half = (e >= 12) ? (1ull << (e - 12)) : 0;
            for (int64_t off = -2; off <= 2; ++off) { in.push_back(base + half + off); in.push_back(base + off); }
        }
        uint64_t z = 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < (1 << 26); ++i) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; in.push_back(i & 1 ? z : (z >> (z & 63))); }
        in.push_back(0x7FFFFFFFFFFFFFFFull); in.push_back(0x8000000000000000ull); in.push_back(~0ull); in.push_back(0);
        const size_t n = in.size();
        uint64_t* din; uint16_t *ds, *du, *dh, *duh; CK(cudaMalloc(&din, n * 8)); CK(cudaMalloc(&ds, n * 2)); CK(cudaMalloc(&du, n * 2)); CK(cudaMalloc(&dh, n * 2)); CK(cudaMalloc(&duh, n * 2));
        CK(cudaMemcpy(din, in.data(), n * 8, cudaMemcpyHostToDevice));
        k_64<<<4096, 256>>>(din, n, ds, du, dh, duh); CK(cudaGetLastError());
        std::vector<uint16_t> gs(n), gu(n), gh(n), guh(n); CK(cudaMemcpy(guh.data(), duh, n * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gs.data(), ds, n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gu.data(), du, n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(gh.data(), dh, n * 2, cudaMemcpyDeviceToHost));
        Tally a, b, c, d; a.name = "G6 int64->bf16   bfloat16_t(int64)"; b.name = "G6 uint64->bf16  bfloat16_t(uint64)"; c.name = "G6 int64->fp16   float16_t(int64)"; d.name = "G6 uint64->fp16  float16_t(uint64)";
        auto no = [](long long) { return false; };
        cmp16(a, gs.data(), n, 0, [&](long long i) { return bfloat16_t((int64_t)in[i]).raw_bits; }, no, nan_bf);
        cmp16(b, gu.data(), n, 0, [&](long long i) { return bfloat16_t((uint64_t)in[i]).raw_bits; }, no, nan_bf);
        cmp16(c, gh.data(), n, 0, [&](long long i) { return float16_t((int64_t)in[i]).raw_bits; }, no, nan_h);
        cmp16(d, guh.data(), n, 0, [&](long long i) { return float16_t((uint64_t)in[i]).raw_bits; }, no, nan_h);
        print(a, (long long)n); print(b, (long long)n); print(c, (long long)n); print(d, (long long)n); need0(a); need0(b); need0(c); need0(d);
        CK(cudaFree(din)); CK(cudaFree(ds)); CK(cudaFree(du)); CK(cudaFree(dh)); CK(cudaFree(duh));
    }
    // ---- G7: fp16 <-> bf16, library path vs hardware direct instruction ----
    if (prop.major >= 9) {
        uint16_t *a1, *a2, *b1, *b2; CK(cudaMalloc(&a1, 131072)); CK(cudaMalloc(&a2, 131072)); CK(cudaMalloc(&b1, 131072)); CK(cudaMalloc(&b2, 131072));
        k_cross<<<256, 256>>>(a1, a2, b1, b2); CK(cudaGetLastError());
        std::vector<uint16_t> h1(65536), h2(65536), g1(65536), g2(65536);
        CK(cudaMemcpy(h1.data(), a1, 131072, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2.data(), a2, 131072, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g1.data(), b1, 131072, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(g2.data(), b2, 131072, cudaMemcpyDeviceToHost));
        Tally x, y; x.name = "G7 fp16->bf16 through fp32 vs cvt.rn.bf16.f16"; y.name = "G7 bf16->fp16 through fp32 vs cvt.rn.f16.bf16";
        for (uint32_t i = 0; i < 65536; ++i) {
            if (nan_h((uint16_t)i)) { if (!nan_bf(h1[i]) || !nan_bf(h2[i])) ++x.nanbad; } else if (h1[i] != h2[i]) { ++x.bad; if (i < x.first) x.first = i; }
            if (nan_bf((uint16_t)i)) { if (!nan_h(g1[i]) || !nan_h(g2[i])) ++y.nanbad; } else if (g1[i] != g2[i]) { ++y.bad; if (i < y.first) y.first = i; }
        }
        print(x, 65536); print(y, 65536); need0(x); need0(y);
    } else {
        std::printf("   G7 skipped: direct fp16<->bf16 cvt needs sm_90 or newer\n");
    }
    std::printf("\nRESULT: %s (%d required checks failed)\n", fails ? "FAIL" : "PASS", fails);
    CK(cudaFree(d16));
    return fails ? 1 : 0;
}
