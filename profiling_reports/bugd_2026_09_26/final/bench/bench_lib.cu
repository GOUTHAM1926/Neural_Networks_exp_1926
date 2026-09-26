// bench_lib.cu -- library-level benchmark, one run against whichever libtensor.so the loader picks
// (the driver run_lib_bench.sh alternates the OLD and the NEW library, 12 rounds).
//
// Prints:  LIB|<path of the libtensor.so really mapped>
//          FP|<fingerprint: GPU bf16 70144 * 2 -> old 0x7f80 (Inf, Bug D), new 0x4809>
//          CASE|<name>|<median us>|<wrong>|<nan-bad>|<checksum>|<exact|approx|none>
// Timed work is inside libtensor.so. This program never calls a Types.h conversion itself:
// inputs are raw bits and the references are private helpers below (so an old library can
// not be silently bound to new inline code).
#include <cuda_runtime.h>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>

#include "core/Tensor.h"
#include "ops/Kernels.h"
#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Activations.h"
#include "ops/UnaryOps/Reduction.h"

using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

namespace {   // private helpers, internal linkage
uint32_t u_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float f_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
float fbf(uint16_t b) { return f_of((uint32_t)b << 16); }
uint16_t rbf(float f) {                                   // round to nearest even, NaN -> 0x7FFF
    uint32_t u = u_of(f); if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FFFu;
    u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16);
}
__attribute__((target("f16c"))) float fh(uint16_t h) { return _cvtsh_ss(h); }
__attribute__((target("f16c"))) uint16_t rh(float f) { return (uint16_t)_cvtss_sh(f, 0); }
uint16_t rbf_d(double d) {                                // one rounding from double (by value)
    uint64_t x; std::memcpy(&x, &d, 8); const uint16_t s = (uint16_t)((x >> 48) & 0x8000u); const uint64_t ax = x & 0x7FFFFFFFFFFFFFFFull;
    if (ax > 0x7FF0000000000000ull) return 0x7FFFu;
    if (ax >= 0x47EFF00000000000ull) return s | 0x7F80u;
    if (ax >= 0x3810000000000000ull) { uint64_t r = ax - 0x3800000000000000ull; r += 0xFFFFFFFFFFFull + ((r >> 45) & 1u); return (uint16_t)(s | (uint16_t)(r >> 45)); }
    return s;   // inputs of this benchmark never land in the bf16 subnormal range
}
bool nan_bf(uint16_t h) { return (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu); }
std::string mapped_libtensor() {
    std::ifstream m("/proc/self/maps"); std::string line;
    while (std::getline(m, line)) { auto p = line.find("libtensor.so"); if (p != std::string::npos) return line.substr(line.find('/')); }
    return "NOT-FOUND";
}
double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t mix = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { mix += 0x9E3779B97F4A7C15ull; uint64_t z = mix; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
float u01() { return (rnd() >> 40) * (1.0f / 16777216.0f); }

Tensor make(Dtype dt, Shape s, bool cuda, const void* bits, size_t bytes) {
    Tensor t(s, TensorOptions().with_dtype(dt).with_device(DeviceIndex(Device::CPU)));
    std::memcpy(t.data(), bits, bytes);
    return cuda ? t.to_cuda(0) : t;
}
template <class T> std::vector<T> read(const Tensor& t) {
    std::vector<T> v(t.numel());
    if (t.device().is_cuda()) CK(cudaMemcpy(v.data(), t.data(), v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    else std::memcpy(v.data(), t.data(), v.size() * sizeof(T));
    return v;
}
template <class T> uint64_t checksum(const std::vector<T>& v) { uint64_t h = 1469598103934665603ull; const unsigned char* p = (const unsigned char*)v.data();
    for (size_t i = 0; i < v.size() * sizeof(T); ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }

struct Case { std::string name; bool gpu; std::function<Tensor()> op; std::function<void(const Tensor&, long long&, long long&, uint64_t&)> check; const char* kind; };
}  // namespace

int main() {
    // ---- fingerprint: runs the library's templated GPU scalar kernel on a Bug D value ----
    {
        const uint16_t v = 0x4789;   // 70144
        Tensor r = make(Dtype::Bfloat16, Shape{{1}}, true, &v, 2) * 2.0f; CK(cudaDeviceSynchronize());
        std::printf("LIB|%s\nFP|0x%04x\n", mapped_libtensor().c_str(), read<uint16_t>(r)[0]);
    }
    const size_t NG = 64ull << 20, NS = 1ull << 20, NC = 4ull << 20;
    // ---- inputs (raw bits) ----
    std::vector<uint16_t> bfA(NG), bfB(NG), bfBig(NG), hA(NG), hB(NG), c32(4096 * 64 * 2);
    std::vector<float> f32(NC); std::vector<double> f64(NC);
    for (size_t i = 0; i < NG; ++i) {
        bfA[i] = rbf((u01() * 1.5f + 0.5f) * ((rnd() & 1) ? -1.f : 1.f));          // |x| in [0.5, 2]
        bfB[i] = rbf(u01() * 1.5f + 0.5f);
        bfBig[i] = rbf((65536.0f + u01() * 1e6f) * ((rnd() & 1) ? -1.f : 1.f));    // >= 65,536 (the Bug D range)
        hA[i] = rh(u01() * 1.5f + 0.5f); hB[i] = rh(u01() * 1.5f + 0.5f);
    }
    for (size_t i = 0; i < c32.size(); ++i) c32[i] = rh(1.0f + (u01() - 0.5f) * 0.02f);   // complex32 parts near 1: the product of 64 stays in range
    for (size_t i = 0; i < NC; ++i) { f32[i] = (u01() * 1.5f + 0.5f) * ((rnd() & 1) ? -1.f : 1.f); f64[i] = (double)f32[i] * (1.0 + 1e-9 * u01()); }

    Tensor gA = make(Dtype::Bfloat16, Shape{{(int64_t)NG}}, true, bfA.data(), NG * 2), gB = make(Dtype::Bfloat16, Shape{{(int64_t)NG}}, true, bfB.data(), NG * 2);
    Tensor gBig = make(Dtype::Bfloat16, Shape{{(int64_t)NG}}, true, bfBig.data(), NG * 2);
    Tensor gAs = make(Dtype::Bfloat16, Shape{{(int64_t)NS}}, true, bfA.data(), NS * 2);
    Tensor gHA = make(Dtype::Float16, Shape{{(int64_t)NG}}, true, hA.data(), NG * 2), gHB = make(Dtype::Float16, Shape{{(int64_t)NG}}, true, hB.data(), NG * 2);
    Tensor gA2d = make(Dtype::Bfloat16, Shape{{(int64_t)(NG / 1024), 1024}}, true, bfA.data(), NG * 2);
    Tensor gC32r = make(Dtype::Complex32, Shape{{4096, 64}}, true, c32.data(), c32.size() * 2), gC32c = make(Dtype::Complex32, Shape{{64, 4096}}, true, c32.data(), c32.size() * 2);
    Tensor gF32 = make(Dtype::Float32, Shape{{(int64_t)NC}}, true, f32.data(), NC * 4);
    Tensor cA = make(Dtype::Bfloat16, Shape{{(int64_t)NC}}, false, bfA.data(), NC * 2), cB = make(Dtype::Bfloat16, Shape{{(int64_t)NC}}, false, bfB.data(), NC * 2);
    Tensor cA2 = make(Dtype::Bfloat16, Shape{{2, (int64_t)(NC / 2)}}, false, bfA.data(), NC * 2);
    Tensor cF32 = make(Dtype::Float32, Shape{{(int64_t)NC}}, false, f32.data(), NC * 4), cF32b = make(Dtype::Float32, Shape{{(int64_t)NC}}, false, f32.data(), NC * 4);
    Tensor cF64 = make(Dtype::Float64, Shape{{(int64_t)NC}}, false, f64.data(), NC * 8);

    // elementwise check helpers: reference from the inputs with the library's float formula
    auto ew_bf = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>* b, std::function<float(float, float)> f) {
        return [&, b, f](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint16_t>(out); w = nb = 0; cs = checksum(o);
            for (size_t i = 0; i < o.size(); ++i) { const uint16_t ref = rbf(f(fbf(a[i]), b ? fbf((*b)[i]) : 0.f));
                if (nan_bf(ref)) { if (!nan_bf(o[i])) ++nb; } else if (o[i] != ref) ++w; } };
    };
    auto ew_h = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>* b, std::function<float(float, float)> f) {
        return [&, b, f](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint16_t>(out); w = nb = 0; cs = checksum(o);
            for (size_t i = 0; i < o.size(); ++i) if (o[i] != rh(f(fh(a[i]), b ? fh((*b)[i]) : 0.f))) ++w; };
    };
    auto only_sum = [](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {   // no exact reference: checksum (old vs new) only
        w = nb = 0; const size_t bytes = out.nbytes(); std::vector<unsigned char> v(bytes);
        if (out.device().is_cuda()) CK(cudaMemcpy(v.data(), out.data(), bytes, cudaMemcpyDeviceToHost)); else std::memcpy(v.data(), out.data(), bytes);
        cs = checksum(v); };

    std::vector<Case> cases = {
        // ---- GPU, changed kernels ----
        {"gpu bf16 t*2.0f (64M)",          true, [&] { return gA * 2.0f; },  ew_bf(bfA, nullptr, [](float x, float) { return x * 2.0f; }), "exact"},
        {"gpu bf16 t*2.0f abs>=65536 (64M)", true, [&] { return gBig * 2.0f; }, ew_bf(bfBig, nullptr, [](float x, float) { return x * 2.0f; }), "exact"},
        {"gpu bf16 t*2.0f (1M)",           true, [&] { return gAs * 2.0f; }, ew_bf(bfA, nullptr, [](float x, float) { return x * 2.0f; }), "exact"},
        {"gpu bf16 t+1.5f (64M)",          true, [&] { return gA + 1.5f; },  ew_bf(bfA, nullptr, [](float x, float) { return x + 1.5f; }), "exact"},
        {"gpu bf16 t/3.0f (64M)",          true, [&] { return gA / 3.0f; },  ew_bf(bfA, nullptr, [](float x, float) { return x / 3.0f; }), "approx"},
        {"gpu f16 t*2.0f (64M)",           true, [&] { return gHA * 2.0f; }, ew_h(hA, nullptr, [](float x, float) { return x * 2.0f; }), "exact"},
        {"gpu bf16 a+b (64M)",             true, [&] { return gA + gB; },    ew_bf(bfA, &bfB, [](float x, float y) { return x + y; }), "exact"},
        {"gpu bf16 a*b (64M)",             true, [&] { return gA * gB; },    ew_bf(bfA, &bfB, [](float x, float y) { return x * y; }), "exact"},
        {"gpu bf16 a/b (64M)",             true, [&] { return gA / gB; },    ew_bf(bfA, &bfB, [](float x, float y) { return x / y; }), "approx"},
        {"gpu f16 a-b (64M)",              true, [&] { return gHA - gHB; },  ew_h(hA, &hB, [](float x, float y) { return x - y; }), "exact"},
        {"gpu bf16 a>b (64M)",             true, [&] { return gA > gB; },    only_sum, "none"},
        {"gpu bf16 reduce_sum all (64M)",  true, [&] { return OwnTensor::reduce_sum(gA); }, only_sum, "none"},
        // (removed: reduce_mean over axis 0 of this [65536, 1024] bf16 tensor makes the GPU kernel
        //  fault with an illegal memory access in the OLD and the NEW library alike -- a separate,
        //  pre-existing library bug, noted for later; it is unrelated to the conversion patch)
        {"gpu c32 reduce_product axis1 [4096,64]", true, [&] { return OwnTensor::reduce_product(gC32r, {1}); }, only_sum, "none"},
        {"gpu c32 reduce_product axis0 [64,4096]", true, [&] { return OwnTensor::reduce_product(gC32c, {0}); }, only_sum, "none"},
        // ---- GPU controls (kernels the patch did not change) ----
        {"CONTROL gpu as_type f32->bf16 (4M)", true, [&] { return gF32.as_type(Dtype::Bfloat16); }, only_sum, "none"},
        {"CONTROL gpu gelu_forward bf16 (64M)", true, [&] { return OwnTensor::gelu_forward(gA); }, only_sum, "none"},
        // ---- CPU ----
        {"cpu as_type f32->bf16 (4M)",     false, [&] { return cF32.as_type(Dtype::Bfloat16); }, [&](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint16_t>(out); w = nb = 0; cs = checksum(o); for (size_t i = 0; i < o.size(); ++i) w += o[i] != rbf(f32[i]); }, "exact"},
        {"cpu as_type bf16->f32 (4M)",     false, [&] { return cA.as_type(Dtype::Float32); }, [&](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint32_t>(out); w = nb = 0; cs = checksum(o); for (size_t i = 0; i < o.size(); ++i) w += o[i] != ((uint32_t)bfA[i] << 16); }, "exact"},
        {"cpu as_type f32->f16 (4M)",      false, [&] { return cF32.as_type(Dtype::Float16); }, [&](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint16_t>(out); w = nb = 0; cs = checksum(o); for (size_t i = 0; i < o.size(); ++i) w += o[i] != rh(f32[i]); }, "exact"},
        {"cpu as_type f64->bf16 (4M)",     false, [&] { return cF64.as_type(Dtype::Bfloat16); }, [&](const Tensor& out, long long& w, long long& nb, uint64_t& cs) {
            const auto o = read<uint16_t>(out); w = nb = 0; cs = checksum(o); for (size_t i = 0; i < o.size(); ++i) w += o[i] != rbf_d(f64[i]); }, "exact"},
        {"cpu bf16 reduce_mean axis0 [2,2M]", false, [&] { return OwnTensor::reduce_mean(cA2, {0}); }, only_sum, "none"},
        {"cpu bf16 reduce_sum axis0 [2,2M]", false, [&] { return OwnTensor::reduce_sum(cA2, {0}); }, only_sum, "none"},
        {"cpu bf16 gelu_forward (4M)",     false, [&] { return OwnTensor::gelu_forward(cA); }, only_sum, "none"},
        {"cpu bf16 t*2.0f (4M)",           false, [&] { return cA * 2.0f; }, ew_bf(bfA, nullptr, [](float x, float) { return x * 2.0f; }), "exact"},
        {"cpu bf16 a+b (4M)",              false, [&] { return cA + cB; },  ew_bf(bfA, &bfB, [](float x, float y) { return x + y; }), "exact"},
        // ---- CPU controls ----
        {"CONTROL cpu as_type f32->f64 (4M)", false, [&] { return cF32.as_type(Dtype::Float64); }, only_sum, "none"},
        {"CONTROL cpu f32 a+b (4M)",       false, [&] { return cF32 + cF32b; }, only_sum, "none"},
    };
    for (Case& c : cases) {
        Tensor out = c.op(); if (c.gpu) CK(cudaDeviceSynchronize());   // warm-up
        // repetitions: ~20 ms per measurement
        double t0 = now_us(); out = c.op(); if (c.gpu) CK(cudaDeviceSynchronize()); const double one = now_us() - t0;
        const int reps = std::max(2, std::min(2000, (int)(20000.0 / std::max(one, 1.0))));
        std::vector<double> us;
        for (int m = 0; m < 7; ++m) {
            if (c.gpu) CK(cudaDeviceSynchronize());
            const double s = now_us();
            for (int r = 0; r < reps; ++r) out = c.op();
            if (c.gpu) CK(cudaDeviceSynchronize());
            us.push_back((now_us() - s) / reps);
        }
        std::sort(us.begin(), us.end());
        long long w = 0, nb = 0; uint64_t cs = 0; c.check(out, w, nb, cs);
        std::printf("CASE|%s|%.3f|%lld|%lld|%016llx|%s\n", c.name.c_str(), us[3], w, nb, (unsigned long long)cs, c.kind);
        std::fflush(stdout);
    }
    return 0;
}
