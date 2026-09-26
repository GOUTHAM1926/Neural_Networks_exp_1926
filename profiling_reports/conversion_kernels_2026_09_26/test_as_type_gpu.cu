// test_as_type_gpu.cu -- GPU Tensor::as_type against the CPU as_type, every dtype pair.
//
// The CPU as_type (AsTypeTensor.cpp) converts directly (static_cast / the Types.h one-rounding
// constructors / component-wise complex) and is the reference. For every (src, dst) pair the GPU
// result must be byte-identical, except:
//   - elements whose conversion is undefined behaviour in C++ (a float / double / bf16 / fp16 value
//     that is NaN, Inf or out of range for an INTEGER destination): counted as "skipped (UB)";
//   - NaN outputs: any NaN on both sides counts as equal (NaN payload bits are not specified);
//   - FTZ: the GPU gives +-0 where the CPU value lies in the fp32 subnormal range. The library is
//     built with --use_fast_math (fp32 subnormals flush to zero on the GPU); counted, not a mismatch.
// complex <-> real must be refused on both devices; the direct convert_type_cuda_generic call must
// also refuse it (it silently skipped before the fix).
// Also prints TIME lines (median of 7) for a few pairs, to compare speed before / after.
//
// Output: PAIR|src->dst|ok/MISMATCH/CRASH/...|details, TIME|..., then a SUMMARY block.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "core/Tensor.h"
#include "ops/helpers/ConversionKernels.cuh"

using namespace OwnTensor;

namespace {
enum Kind { SINT, UINT, F32, F64, BF16, F16, BOOL, C32, C64, C128 };
struct DT { Dtype dt; const char* name; int size; Kind kind; };
const std::vector<DT> REAL = {
    {Dtype::Int8, "int8", 1, SINT},   {Dtype::Int16, "int16", 2, SINT},  {Dtype::Int32, "int32", 4, SINT},
    {Dtype::Int64, "int64", 8, SINT}, {Dtype::UInt8, "uint8", 1, UINT},  {Dtype::UInt16, "uint16", 2, UINT},
    {Dtype::UInt32, "uint32", 4, UINT}, {Dtype::UInt64, "uint64", 8, UINT},
    {Dtype::Float32, "fp32", 4, F32}, {Dtype::Float64, "fp64", 8, F64}, {Dtype::Bfloat16, "bf16", 2, BF16},
    {Dtype::Float16, "fp16", 2, F16}, {Dtype::Bool, "bool", 1, BOOL}};
const std::vector<DT> CPLX = {{Dtype::Complex32, "c32", 4, C32}, {Dtype::Complex64, "c64", 8, C64},
                              {Dtype::Complex128, "c128", 16, C128}};

constexpr int64_t N = 65536 * 2;   // bf16/fp16 sources: every bit pattern, twice
std::mt19937_64 rng(12345);

float f_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
double bf_to_d(uint16_t b) { return f_of((uint32_t)b << 16); }
double h_to_d(uint16_t h) {   // exact fp16 -> double
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? std::ldexp((double)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25);
    return s ? -v : v;
}
bool is_float_kind(Kind k) { return k == F32 || k == F64 || k == BF16 || k == F16; }

// a wide-range floating value set (specials + random magnitudes 1e-320 .. 1e308 + raw bit patterns)
double rand_double(int64_t i) {
    static const double sp[] = {0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, 65504.0, 65519.99, 65520.0, 70000.0,
        16777216.0, 16777217.0, 9007199254740993.0, 1e10, 10000000001.0, 1e20, 1e38, 3.4e38, 1e300,
        1e-40, 1e-310, INFINITY, -INFINITY, NAN, 1.00390625, 1.0 + 1.0 / 512, 2147483647.0, 2147483648.0,
        -2147483649.0, 4294967295.0, 4294967296.0, 9.2233720368547758e18, -9.2233720368547758e18,
        1.8446744073709552e19, 127.5, 255.5, -128.5, 32767.5, 65535.5};
    const int ns = sizeof(sp) / sizeof(sp[0]);
    if (i < ns) return sp[i];
    const uint64_t r = rng();
    switch (i % 4) {
        case 0: { double d; std::memcpy(&d, &r, 8); return d; }                    // raw pattern
        case 1: return (r & 1 ? -1.0 : 1.0) * std::pow(10.0, (double)((r >> 8) % 600) - 300.0) * (1.0 + (double)((r >> 20) % 1000) / 1000.0);
        case 2: return (double)(int64_t)(r >> (r % 60));                             // integers of all sizes
        default: return (r & 1 ? -1.0 : 1.0) * (double)((r >> 1) % 70000000) / ((r >> 40) % 8 + 1);
    }
}
int64_t rand_int(int64_t i) {
    static const int64_t sp[] = {0, 1, -1, 127, -128, 255, 256, 32767, -32768, 65535, 65536, 65519, 65520, 70000,
        (1LL << 24) - 1, 1LL << 24, (1LL << 24) + 1, 2147483647LL, -2147483648LL, 4294967295LL, 4294967296LL,
        (1LL << 53) + 1, 10000000001LL, INT64_MAX, INT64_MIN};
    const int ns = sizeof(sp) / sizeof(sp[0]);
    if (i < ns) return sp[i];
    const uint64_t r = rng();
    return (int64_t)(r >> (r % 64));
}

// fill n source elements of type s; also return each element's exact value (for the UB check)
void fill(const DT& s, std::vector<unsigned char>& buf, std::vector<double>& val) {
    buf.assign(N * s.size, 0); val.assign(N, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        unsigned char* p = &buf[i * s.size];
        switch (s.kind) {
            case SINT: case UINT: { int64_t v = rand_int(i); std::memcpy(p, &v, s.size); break; }   // wraps to the size
            case BOOL: p[0] = (unsigned char)(i & 1); break;
            case F32: { float f = (float)rand_double(i); std::memcpy(p, &f, 4); val[i] = f; break; }
            case F64: { double d = rand_double(i); std::memcpy(p, &d, 8); val[i] = d; break; }
            case BF16: { uint16_t b = (uint16_t)i; std::memcpy(p, &b, 2); val[i] = bf_to_d(b); break; }
            case F16: { uint16_t h = (uint16_t)i; std::memcpy(p, &h, 2); val[i] = h_to_d(h); break; }
            case C32: { uint16_t h[2] = {(uint16_t)i, (uint16_t)(i * 7919)}; std::memcpy(p, h, 4); break; }
            case C64: { float f[2] = {(float)rand_double(i), (float)rand_double(i + 7)}; std::memcpy(p, f, 8); break; }
            case C128: { double d[2] = {rand_double(i), rand_double(i + 7)}; std::memcpy(p, d, 16); break; }
        }
    }
}

// is converting a floating value v to integer destination d undefined behaviour?
bool ub_to_int(double v, const DT& d) {
    if (d.kind != SINT && d.kind != UINT) return false;
    if (std::isnan(v) || std::isinf(v)) return true;
    long double lo, hi;
    if (d.kind == SINT) { lo = -std::ldexp(1.0L, 8 * d.size - 1) - 1.0L; hi = std::ldexp(1.0L, 8 * d.size - 1); }
    else { lo = -1.0L; hi = std::ldexp(1.0L, 8 * d.size); }
    return !((long double)v > lo && (long double)v < hi);
}

bool is_nan_elem(const unsigned char* p, Kind k) {
    switch (k) {
        case F32: { float f; std::memcpy(&f, p, 4); return std::isnan(f); }
        case F64: { double d; std::memcpy(&d, p, 8); return std::isnan(d); }
        case BF16: { uint16_t b; std::memcpy(&b, p, 2); return (b & 0x7F80) == 0x7F80 && (b & 0x7F); }
        case F16: { uint16_t h; std::memcpy(&h, p, 2); return (h & 0x7C00) == 0x7C00 && (h & 0x3FF); }
        default: return false;
    }
}
double as_d(const unsigned char* p, Kind k) {
    switch (k) {
        case F32: { float f; std::memcpy(&f, p, 4); return f; }
        case F64: { double d; std::memcpy(&d, p, 8); return d; }
        case BF16: { uint16_t b; std::memcpy(&b, p, 2); return bf_to_d(b); }
        case F16: { uint16_t h; std::memcpy(&h, p, 2); return h_to_d(h); }
        default: return 0.0;
    }
}
// 0 = equal, 1 = FTZ (GPU gave +-0 where the CPU value is in the fp32 subnormal range: the
// library is built with --use_fast_math, which flushes fp32 subnormals on the GPU), 2 = mismatch
int cmp_scalar(const unsigned char* a, const unsigned char* b, Kind k, int size) {
    if (std::memcmp(a, b, size) == 0 || (is_nan_elem(a, k) && is_nan_elem(b, k))) return 0;
    if (k == F32 || k == F64 || k == BF16 || k == F16) {
        const double ca = as_d(a, k), gb = as_d(b, k);
        if (gb == 0.0 && ca != 0.0 && std::fabs(ca) < (double)std::numeric_limits<float>::min()) return 1;
    }
    return 2;
}
int elem_cmp(const unsigned char* a, const unsigned char* b, const DT& d) {
    if (d.kind == C32 || d.kind == C64 || d.kind == C128) {
        const Kind ck = d.kind == C32 ? F16 : d.kind == C64 ? F32 : F64; const int cs = d.size / 2;
        int r = 0;
        for (int c = 0; c < 2; ++c) r = std::max(r, cmp_scalar(a + c * cs, b + c * cs, ck, cs));
        return r;
    }
    return cmp_scalar(a, b, d.kind, d.size);
}
std::string hexb(const unsigned char* p, int n) { std::string s; char b[4]; for (int i = n - 1; i >= 0; --i) { std::snprintf(b, 4, "%02x", p[i]); s += b; } return s; }

int g_ok = 0, g_bad = 0;
void report(const std::string& pair, bool ok, const std::string& d) {
    std::printf("PAIR|%s|%s|%s\n", pair.c_str(), ok ? "ok" : "MISMATCH", d.c_str()); std::fflush(stdout);
    (ok ? g_ok : g_bad)++;
}

void run_pair(const DT& s, const DT& d, const std::vector<unsigned char>& buf, const std::vector<double>& val) {
    const std::string pair = std::string(s.name) + "->" + d.name;
    Shape sh; sh.dims = {N};
    Tensor t(sh, TensorOptions().with_dtype(s.dt).with_device(DeviceIndex(Device::CPU)));
    if (buf.size() != (size_t)N * s.size) { report(pair, false, "TEST BUG: buffer size mismatch"); return; }
    std::memcpy(t.data(), buf.data(), buf.size());
    const bool cross = (s.kind >= C32) != (d.kind >= C32);
    std::string cpu_err, gpu_err;
    Tensor co, oc;
    try { co = t.as_type(d.dt); } catch (const std::exception& e) { cpu_err = e.what(); }
    try {
        Tensor go = t.to_cuda(0).as_type(d.dt);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { report(pair, false, std::string("CRASH ") + cudaGetErrorString(e)); return; }
        oc = go.to_cpu();
    } catch (const std::exception& e) { gpu_err = e.what(); }
    if (cross) {   // must be refused on both devices
        report(pair, !cpu_err.empty() && !gpu_err.empty(), std::string("complex<->real: CPU ") +
               (cpu_err.empty() ? "ACCEPTED" : "refused") + ", GPU " + (gpu_err.empty() ? "ACCEPTED" : "refused"));
        return;
    }
    if (!cpu_err.empty() || !gpu_err.empty()) { report(pair, false, "EXCEPTION cpu[" + cpu_err + "] gpu[" + gpu_err + "]"); return; }
    int64_t cmp = 0, skip = 0, bad = 0, ftz = 0; std::string first;
    const auto* a = static_cast<const unsigned char*>(co.data());
    const auto* b = static_cast<const unsigned char*>(oc.data());
    for (int64_t i = 0; i < N; ++i) {
        if (is_float_kind(s.kind) && ub_to_int(val[i], d)) { ++skip; continue; }
        ++cmp;
        int r = elem_cmp(a + i * d.size, b + i * d.size, d);
        if (r == 2 && d.kind == BOOL && is_float_kind(s.kind) && val[i] != 0.0 &&
            std::fabs(val[i]) < (double)std::numeric_limits<float>::min() && a[i] == 1 && b[i] == 0)
            r = 1;   // FTZ: a subnormal source compared as zero under --use_fast_math
        if (r == 1) { ++ftz; continue; }
        if (r == 2) {
            if (!bad) first = "first i=" + std::to_string(i) + " src=" + hexb(&buf[i * s.size], s.size) +
                              " cpu=" + hexb(a + i * d.size, d.size) + " gpu=" + hexb(b + i * d.size, d.size);
            ++bad;
        }
    }
    report(pair, bad == 0, "compared " + std::to_string(cmp) + ", skipped (UB) " + std::to_string(skip) +
                           ", FTZ " + std::to_string(ftz) + ", mismatches " + std::to_string(bad) + (bad ? "; " + first : ""));
}

void time_pair(const DT& s, const DT& d, int64_t n) {
    Shape sh; sh.dims = {n};
    Tensor g = Tensor::zeros(sh, TensorOptions().with_dtype(s.dt).with_device(DeviceIndex{Device::CUDA, 0}));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r < 9; ++r) {
        cudaDeviceSynchronize(); cudaEventRecord(e0);
        Tensor o = g.as_type(d.dt);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t = 0; cudaEventElapsedTime(&t, e0, e1); if (r >= 2) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("TIME|%s->%s|n=%lld|median %.1f us\n", s.name, d.name, (long long)n, ms[ms.size() / 2] * 1000.0f);
    std::fflush(stdout);
}
}  // namespace

int main() {
    std::vector<unsigned char> buf; std::vector<double> val;
    for (const auto& s : REAL) { fill(s, buf, val); for (const auto& d : REAL) if (d.dt != s.dt) run_pair(s, d, buf, val); }
    for (const auto& s : CPLX) { fill(s, buf, val); for (const auto& d : CPLX) if (d.dt != s.dt) run_pair(s, d, buf, val); }
    for (const auto& s : CPLX) {
        fill(s, buf, val); run_pair(s, REAL[8], buf, val);
        fill(REAL[8], buf, val); run_pair(REAL[8], s, buf, val);   // fp32 buffer for an fp32 source
    }
    // the kernel entry point itself must refuse complex <-> real (it silently skipped before)
    {
        void *x = nullptr, *y = nullptr; cudaMalloc(&x, 1 << 16); cudaMalloc(&y, 1 << 16);
        std::string err;
        try { convert_type_cuda_generic(x, Dtype::Complex64, y, Dtype::Float32, 1024, 0); cudaDeviceSynchronize(); }
        catch (const std::exception& e) { err = e.what(); }
        report("direct convert_type_cuda_generic c64->fp32", !err.empty(), err.empty() ? "ACCEPTED (silent skip)" : "refused: " + err);
        cudaFree(x); cudaFree(y);
    }
    // speed of the hot pairs (fp32<->bf16 are the vectorised fast paths, untouched by the fix)
    const int64_t T = 1LL << 26;
    time_pair(REAL[8], REAL[10], T); time_pair(REAL[10], REAL[8], T); time_pair(REAL[8], REAL[11], T);
    time_pair(REAL[11], REAL[8], T); time_pair(REAL[2], REAL[8], T); time_pair(REAL[9], REAL[8], T);
    time_pair(REAL[8], REAL[9], T); time_pair(REAL[3], REAL[2], T); time_pair(REAL[9], REAL[10], T);
    std::printf("\n==================== SUMMARY ====================\n");
    std::printf("pairs: %d, ok %d, MISMATCH %d\nRESULT: %s\n", g_ok + g_bad, g_ok, g_bad, g_bad ? "FAIL" : "PASS");
    return g_bad ? 1 : 0;
}
