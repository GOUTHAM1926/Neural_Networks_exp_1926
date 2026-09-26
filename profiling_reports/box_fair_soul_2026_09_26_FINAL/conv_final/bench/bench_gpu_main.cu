// bench_gpu_main.cu -- GPU micro-benchmark: OLD vs NEW conversion code (2026-09-26).
//
// Same program holds both versions (see bench_gpu_body.cu). For every conversion path x
// input kind:
//   * time OLD and NEW back to back, 12 rounds, order alternating, cudaEvent timing,
//     median + paired new/old ratio (min, max);
//   * check both outputs against the CPU code of the patched header (itself checked bit
//     for bit against PyTorch, Eigen, NumPy, F16C, GCC and CUDA's host code).
// Shapes: memory-shaped (1 conversion per element, like real ops), conversion-heavy
// (64 dependent round trips per element in registers), bf16 operator chain.
// Fingerprints (host and device) prove the two copies are really old and new.
#include "dtype/Types.h"            // NEW header: host reference only (namespace OwnTensor)
#include "bench_gpu_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using OwnTensor::bfloat16_t;
using OwnTensor::float16_t;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)
template <class To, class From> static To bc(const From& f) { To t; std::memcpy(&t, &f, sizeof(To)); return t; }
static bool nan_bf(uint16_t h) { return (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu); }
static bool nan_h(uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }

static std::mt19937_64 rng(20260926);
static double logu(double lo, double hi) { std::uniform_real_distribution<double> u(std::log(lo), std::log(hi)); return std::exp(u(rng)); }
static double sgn() { return (rng() & 1) ? -1.0 : 1.0; }

enum Cls { C_NORMAL, C_LARGE, C_TINY16, C_TINYBF, C_SPECIAL, C_RANDOM, C_SORTED };
static const char* cls_name[] = {"normal", "large>=65536", "f16-subnormal", "bf16-subnormal", "specials", "random-bits", "random-sorted"};

static float gen_f32(int c) {
    switch (c) {
        case C_NORMAL: return (float)(sgn() * logu(1e-3, 1e3));
        case C_LARGE:  return (float)(sgn() * logu(65536.0, 1e30));
        case C_TINY16: return (float)(sgn() * logu(0x1p-24, 0x1p-14));
        case C_TINYBF: return bc<float>(((uint32_t)rng() & 0x807FFFFFu) | 1u);
        case C_SPECIAL: { const float s[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, -NAN}; return s[rng() % 6]; }
        default: return bc<float>((uint32_t)rng());
    }
}
static double gen_f64(int c) {
    switch (c) {
        case C_NORMAL: return sgn() * logu(1e-3, 1e3);
        case C_LARGE:  return sgn() * logu(65536.0, 1e300);
        case C_TINY16: return sgn() * logu(1e-12, 1e-5);
        case C_TINYBF: return sgn() * logu(1e-45, 1e-38);
        case C_SPECIAL: { const double s[] = {0.0, -0.0, INFINITY, -INFINITY, NAN}; return s[rng() % 5]; }
        default: return bc<double>(rng());
    }
}
// bucket sort by the exponent field (O(n)): neighbouring threads then see the same kind of value
template <class T> static void sort_by_exp(std::vector<T>& v) {
    constexpr int SH = sizeof(T) == 4 ? 23 : 52; constexpr uint64_t MASK = sizeof(T) == 4 ? 0xFF : 0x7FF;
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
    std::vector<std::vector<T>> bucket(MASK + 1);
    for (T x : v) bucket[(bc<U>(x) >> SH) & MASK].push_back(x);
    size_t k = 0; for (auto& bk : bucket) for (T x : bk) v[k++] = x;
}

struct Case { int fn; const char* fname; int cls; size_t n; };
struct Buffers { std::vector<unsigned char> host_in; size_t in_bytes = 0, out_bytes = 0; };

// build the host input for a case; returns bytes of input / output
static void make(const Case& c, Buffers& b) {
    const size_t n = c.n;
    auto f32 = [&](void) { std::vector<float> v(n); for (auto& x : v) x = gen_f32(c.cls == C_SORTED ? C_RANDOM : c.cls); if (c.cls == C_SORTED) sort_by_exp(v);
                           b.host_in.resize(n * 4); std::memcpy(b.host_in.data(), v.data(), n * 4); b.in_bytes = n * 4; };
    auto f64 = [&](void) { std::vector<double> v(n); for (auto& x : v) x = gen_f64(c.cls == C_SORTED ? C_RANDOM : c.cls); if (c.cls == C_SORTED) sort_by_exp(v);
                           b.host_in.resize(n * 8); std::memcpy(b.host_in.data(), v.data(), n * 8); b.in_bytes = n * 8; };
    auto u16 = [&](size_t per, bool half) {
        std::vector<uint16_t> v(n * per);
        for (auto& x : v) {
            if (c.cls == C_RANDOM || c.cls == C_SORTED) { x = (uint16_t)rng(); continue; }
            const float f = c.cls == C_LARGE ? (float)(sgn() * logu(256.0, half ? 250.0 * 256 : 1e20)) : (float)(sgn() * logu(0.5, 2.0));
            x = half ? float16_t(f).raw_bits : bfloat16_t(f).raw_bits;
        }
        b.host_in.resize(v.size() * 2); std::memcpy(b.host_in.data(), v.data(), v.size() * 2); b.in_bytes = v.size() * 2; };
    switch (c.fn) {
        case G_F32_BF16: case G_F32_F16: case G_RT_BF16: case G_RT_F16: f32(); break;
        case G_F64_BF16: case G_F64_F16: case G_RT_F64_BF16: f64(); break;
        case G_BF16_F32: case G_BF16_TO_F16: u16(1, false); break;
        case G_F16_F32: u16(1, true); break;
        case G_BF16_MUL: u16(2, false); break;
        case G_F16_MUL: u16(2, true); break;
        case G_CHAIN_BF16: {   // triples (x, b, c): b near 1, c small, so 64 steps stay in range
            std::vector<uint16_t> v(3 * n);
            for (size_t i = 0; i < n; ++i) { v[3 * i] = bfloat16_t((float)(sgn() * logu(0.5, 4.0))).raw_bits;
                                             v[3 * i + 1] = bfloat16_t((float)(0.9 + 0.2 * (rng() % 1000) / 1000.0)).raw_bits;
                                             v[3 * i + 2] = bfloat16_t((float)(sgn() * logu(1e-2, 1.0))).raw_bits; }
            b.host_in.resize(6 * n); std::memcpy(b.host_in.data(), v.data(), 6 * n); b.in_bytes = 6 * n; } break;
        case G_I32_BF16: case G_I32_F16: case G_RT_I32_BF16: {
            std::vector<int32_t> v(n);
            for (auto& x : v) x = c.cls == C_NORMAL ? (int32_t)(rng() % 511) - 255 : (int32_t)(rng() % (1u << 31)) - (1 << 30);   // |v| < 2^30
            b.host_in.resize(n * 4); std::memcpy(b.host_in.data(), v.data(), n * 4); b.in_bytes = n * 4; } break;
        case G_I64_BF16: {
            std::vector<int64_t> v(n);
            for (auto& x : v) x = c.cls == C_NORMAL ? (int64_t)(rng() % 511) - 255 : (int64_t)rng();
            b.host_in.resize(n * 8); std::memcpy(b.host_in.data(), v.data(), n * 8); b.in_bytes = n * 8; } break;
    }
    b.out_bytes = (c.fn == G_BF16_F32 || c.fn == G_F16_F32) ? n * 4 : n * 2;
}

// reference = CPU path of the patched header; counts wrong outputs and NaN-in -> non-NaN-out
static void check(const Case& c, const Buffers& b, const std::vector<unsigned char>& out, long long& wrong, long long& nanbad) {
    long long w = 0, nb = 0;
    const size_t n = c.n; const unsigned char* in = b.host_in.data();
    #pragma omp parallel for reduction(+ : w, nb) schedule(static)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        const size_t i = (size_t)ii;
        if (c.fn == G_BF16_F32) { if (((const uint32_t*)out.data())[i] != ((uint32_t)((const uint16_t*)in)[i] << 16)) ++w; continue; }
        if (c.fn == G_F16_F32) { float16_t h; h.raw_bits = ((const uint16_t*)in)[i]; const float r = static_cast<float>(h), g = ((const float*)out.data())[i];
                                 if (std::isnan(r)) { if (!std::isnan(g)) ++nb; } else if (bc<uint32_t>(r) != bc<uint32_t>(g)) ++w; continue; }
        uint16_t ref = 0; bool bf = true;
        switch (c.fn) {
            case G_F32_BF16: case G_RT_BF16: ref = bfloat16_t(((const float*)in)[i]).raw_bits; break;
            case G_F32_F16: case G_RT_F16: ref = float16_t(((const float*)in)[i]).raw_bits; bf = false; break;
            case G_F64_BF16: case G_RT_F64_BF16: ref = bfloat16_t(((const double*)in)[i]).raw_bits; break;
            case G_F64_F16: ref = float16_t(((const double*)in)[i]).raw_bits; bf = false; break;
            case G_I32_BF16: case G_RT_I32_BF16: ref = bfloat16_t(((const int32_t*)in)[i]).raw_bits; break;
            case G_I32_F16: ref = float16_t(((const int32_t*)in)[i]).raw_bits; bf = false; break;
            case G_I64_BF16: ref = bfloat16_t(((const int64_t*)in)[i]).raw_bits; break;
            case G_BF16_MUL: { bfloat16_t x, y; x.raw_bits = ((const uint16_t*)in)[2 * i]; y.raw_bits = ((const uint16_t*)in)[2 * i + 1]; ref = (x * y).raw_bits; } break;
            case G_F16_MUL: { float16_t x, y; x.raw_bits = ((const uint16_t*)in)[2 * i]; y.raw_bits = ((const uint16_t*)in)[2 * i + 1]; ref = (x * y).raw_bits; bf = false; } break;
            case G_BF16_TO_F16: { bfloat16_t x; x.raw_bits = ((const uint16_t*)in)[i]; ref = float16_t(static_cast<float>(x)).raw_bits; bf = false; } break;
            case G_CHAIN_BF16: { const uint16_t* t = (const uint16_t*)in; bfloat16_t x, bb, cc; x.raw_bits = t[3 * i]; bb.raw_bits = t[3 * i + 1]; cc.raw_bits = t[3 * i + 2];
                                 for (int k = 0; k < RT_K; ++k) x = x * bb + cc; ref = x.raw_bits; } break;
        }
        const uint16_t got = ((const uint16_t*)out.data())[i];
        const bool rnan = bf ? nan_bf(ref) : nan_h(ref), gnan = bf ? nan_bf(got) : nan_h(got);
        if (rnan) { if (!gnan) ++nb; } else if (got != ref) ++w;
    }
    wrong = w; nanbad = nb;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU micro-benchmark, old vs new conversion code. GPU: %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    // ---- fingerprints ----
    {
        const uint16_t ho = gbench_old::probe_f2bf(70000.0f), hn = gbench_new::probe_f2bf(70000.0f);
        float* d_in; uint16_t* d_out; float v = 70000.0f; uint16_t go = 0, gn = 0;
        CK(cudaMalloc(&d_in, 4)); CK(cudaMalloc(&d_out, 2)); CK(cudaMemcpy(d_in, &v, 4, cudaMemcpyHostToDevice));
        gbench_old::run(G_F32_BF16, d_in, d_out, 1, 1, s); CK(cudaMemcpy(&go, d_out, 2, cudaMemcpyDeviceToHost));
        gbench_new::run(G_F32_BF16, d_in, d_out, 1, 1, s); CK(cudaMemcpy(&gn, d_out, 2, cudaMemcpyDeviceToHost));
        std::printf("fingerprint bf16(70000): host old=0x%04x new=0x%04x   GPU old=0x%04x new=0x%04x   (expect old 0x7f80 = Inf, new 0x4789)\n", ho, hn, go, gn);
        CK(cudaFree(d_in)); CK(cudaFree(d_out));
        if (ho != 0x7F80 || hn != 0x4789 || go != 0x7F80 || gn != 0x4789) { std::printf("FINGERPRINT MISMATCH: not an old-vs-new build. Stop.\n"); return 2; }
    }
    const size_t NM = 64ull << 20, NC = 16ull << 20;   // memory-shaped 64M elements, compute-shaped 16M
    std::vector<Case> cases;
    for (int c : {C_NORMAL, C_LARGE, C_TINYBF, C_SPECIAL, C_RANDOM, C_SORTED}) cases.push_back({G_F32_BF16, "f32->bf16", c, NM});
    cases.push_back({G_F32_BF16, "f32->bf16", C_NORMAL, 4096}); cases.push_back({G_F32_BF16, "f32->bf16", C_NORMAL, 1u << 20});
    for (int c : {C_NORMAL, C_RANDOM}) cases.push_back({G_BF16_F32, "bf16->f32", c, NM});
    cases.push_back({G_BF16_F32, "bf16->f32", C_NORMAL, 4096});
    for (int c : {C_NORMAL, C_LARGE, C_TINY16, C_SPECIAL, C_RANDOM, C_SORTED}) cases.push_back({G_F32_F16, "f32->f16", c, NM});
    for (int c : {C_NORMAL, C_RANDOM}) cases.push_back({G_F16_F32, "f16->f32", c, NM});
    for (int c : {C_NORMAL, C_LARGE, C_TINYBF, C_RANDOM}) cases.push_back({G_F64_BF16, "f64->bf16", c, NM});
    for (int c : {C_NORMAL, C_LARGE, C_TINY16, C_RANDOM}) cases.push_back({G_F64_F16, "f64->f16", c, NM});
    for (int c : {C_NORMAL, C_RANDOM}) { cases.push_back({G_I32_BF16, "i32->bf16", c, NM}); cases.push_back({G_I32_F16, "i32->f16", c, NM}); cases.push_back({G_I64_BF16, "i64->bf16", c, NM}); }
    for (int c : {C_NORMAL, C_LARGE, C_RANDOM}) cases.push_back({G_BF16_MUL, "bf16*bf16", c, NM});
    for (int c : {C_NORMAL, C_RANDOM}) { cases.push_back({G_F16_MUL, "f16*f16", c, NM}); cases.push_back({G_BF16_TO_F16, "bf16->f16", c, NM}); }
    for (int c : {C_NORMAL, C_LARGE, C_TINYBF, C_RANDOM, C_SORTED}) cases.push_back({G_RT_BF16, "RTx64 bf16", c, NC});
    for (int c : {C_NORMAL, C_LARGE, C_TINY16, C_RANDOM, C_SORTED}) cases.push_back({G_RT_F16, "RTx64 f16", c, NC});
    for (int c : {C_NORMAL, C_LARGE, C_RANDOM}) cases.push_back({G_RT_F64_BF16, "RTx64 f64->bf16", c, NC});
    for (int c : {C_NORMAL, C_RANDOM}) cases.push_back({G_RT_I32_BF16, "RTx64 i32->bf16", c, NC});
    cases.push_back({G_CHAIN_BF16, "bf16 a*b+c x64", C_NORMAL, NC});

    const int ROUNDS = 12;
    struct Res { std::string name; double o, n, rmed, rmin, rmax; long long ow, on, nw, nn; };
    std::vector<Res> all;
    std::printf("\n%-16s %-15s %9s %10s %10s %8s %17s  %-8s %-10s  %s\n", "function", "input", "elements", "old us", "new us", "speedup", "new/old min..max", "new GB/s", "Gconv/s", "precision old wrong/NaN-bad | new wrong/NaN-bad");
    for (const Case& c : cases) {
        Buffers b; make(c, b);
        void *d_in, *d_old, *d_new; CK(cudaMalloc(&d_in, b.in_bytes)); CK(cudaMalloc(&d_old, b.out_bytes)); CK(cudaMalloc(&d_new, b.out_bytes));
        CK(cudaMemcpy(d_in, b.host_in.data(), b.in_bytes, cudaMemcpyHostToDevice));
        // calibrate to ~20 ms per measurement, then warm up both
        float one = gbench_new::run(c.fn, d_in, d_new, c.n, 3, s) / 3.0f;
        const int reps = std::max(3, std::min(20000, (int)(20.0f / std::max(one, 1e-4f))));
        gbench_old::run(c.fn, d_in, d_old, c.n, reps, s); gbench_new::run(c.fn, d_in, d_new, c.n, reps, s);
        std::vector<double> to, tn, ratio;
        for (int r = 0; r < ROUNDS; ++r) {
            double a, bn;
            if ((r & 1) == 0) { a = gbench_old::run(c.fn, d_in, d_old, c.n, reps, s); bn = gbench_new::run(c.fn, d_in, d_new, c.n, reps, s); }
            else { bn = gbench_new::run(c.fn, d_in, d_new, c.n, reps, s); a = gbench_old::run(c.fn, d_in, d_old, c.n, reps, s); }
            to.push_back(1000.0 * a / reps); tn.push_back(1000.0 * bn / reps); ratio.push_back(bn / a);
        }
        CK(cudaGetLastError());
        std::vector<unsigned char> oo(b.out_bytes), on(b.out_bytes);
        CK(cudaMemcpy(oo.data(), d_old, b.out_bytes, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(on.data(), d_new, b.out_bytes, cudaMemcpyDeviceToHost));
        long long ow, onb, nw, nnb; check(c, b, oo, ow, onb); check(c, b, on, nw, nnb);
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]); };
        Res R{std::string(c.fname) + " | " + cls_name[c.cls] + " | n=" + std::to_string(c.n), med(to), med(tn), med(ratio),
              *std::min_element(ratio.begin(), ratio.end()), *std::max_element(ratio.begin(), ratio.end()), ow, onb, nw, nnb};
        const bool heavy = c.fn >= G_RT_BF16;
        const double gbs = (b.in_bytes + b.out_bytes) / (R.n * 1e3), gconv = heavy ? (double)c.n * RT_K * 2 / (R.n * 1e3) : (double)c.n / (R.n * 1e3);
        std::printf("%-16s %-15s %9zu %10.2f %10.2f %7.2fx %7.3f..%-8.3f  %-8.0f %-10.1f  %lld/%lld | %lld/%lld\n", c.fname, cls_name[c.cls], c.n, R.o, R.n, R.o / R.n, R.rmin, R.rmax, gbs, gconv, ow, onb, nw, nnb);
        std::fflush(stdout);
        all.push_back(R);
        CK(cudaFree(d_in)); CK(cudaFree(d_old)); CK(cudaFree(d_new));
    }
    std::printf("\n==================== SUMMARY ====================\n");
    long long nw = 0, nn = 0, ow = 0; for (auto& r : all) { nw += r.nw; nn += r.nn; ow += r.ow; }
    std::printf("precision: old code wrong on %lld outputs, new code wrong on %lld (NaN-bad %lld)\n", ow, nw, nn);
    int slow = 0, fast = 0;
    std::printf("slower (every round new/old > 1.03):\n");
    for (auto& r : all) if (r.rmin > 1.03) { ++slow; std::printf("   %-55s median %.3f (min %.3f)\n", r.name.c_str(), r.rmed, r.rmin); }
    if (!slow) std::printf("   none\n");
    for (auto& r : all) fast += r.rmax < 0.97;
    std::printf("faster (every round < 0.97): %d of %zu;  unclear (rounds straddle 0.97..1.03): %zu\n", fast, all.size(), all.size() - fast - slow);
    std::printf("RESULT: precision %s\n", (nw || nn) ? "FAIL" : "PASS");
    return (nw || nn) ? 1 : 0;
}
