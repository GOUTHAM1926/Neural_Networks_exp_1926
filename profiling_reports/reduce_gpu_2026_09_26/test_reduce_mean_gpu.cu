// test_reduce_mean_gpu.cu -- GPU reduce_mean vs an exact host reference, one case per process.
//   usage: test_reduce_mean_gpu <case index>   (prints one line: CASE|name|ok/WRONG/CRASH|details)
//          test_reduce_mean_gpu list           (prints the number of cases)
// Suspected bug (dispatch_mean_gpu): config and shared memory sized with the INPUT type (CudaT)
// instead of the accumulator (arg_t = float for bf16/fp16), so block_y_reduce / block_x_reduce
// write past the dynamic shared memory for 2-byte inputs.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;

namespace {
uint64_t s = 0x9E3779B97F4A7C15ull;
double u01() { s += 0x9E3779B97F4A7C15ull; uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return ((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0); }
float f_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
uint32_t u_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
uint16_t to_bf(float f) { uint32_t u = u_of(f); u += 0x7FFFu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
float from_bf(uint16_t b) { return f_of((uint32_t)b << 16); }
__attribute__((target("f16c"))) uint16_t to_h(float f) { return (uint16_t)_cvtss_sh(f, 0); }
__attribute__((target("f16c"))) float from_h(uint16_t h) { return _cvtsh_ss(h); }

// tr = true (2-D only): the input is a TRANSPOSED view (non-contiguous) of a [shape[1], shape[0]] tensor
struct Case { const char* dtype; std::vector<int64_t> shape; std::vector<int64_t> axes; const char* op = "mean"; bool tr = false; };
const std::vector<Case> CASES = {
    {"bf16", {65536, 1024}, {0}},   // the crash seen in the library benchmark (multi-CTA path)
    {"bf16", {1024, 1024}, {0}},    {"bf16", {4096, 512}, {0}},  {"bf16", {33, 1000}, {0}},
    {"bf16", {512, 4096}, {1}},     {"bf16", {1024, 65536}, {1}}, {"bf16", {7, 300, 5}, {1}},
    {"bf16", {1 << 20}, {}},        {"bf16", {256, 256}, {0, 1}},
    {"f16",  {65536, 1024}, {0}},   {"f16",  {1024, 1024}, {0}}, {"f16",  {512, 4096}, {1}},
    {"f32",  {65536, 1024}, {0}},   {"f32",  {1024, 1024}, {0}}, {"f32",  {512, 4096}, {1}},
    {"f64",  {65536, 256}, {0}},    {"f64",  {512, 4096}, {1}},
    // same kernel, other ops: is the last-axis error specific to mean?
    {"bf16", {512, 4096}, {1}, "sum"}, {"f32", {512, 4096}, {1}, "sum"}, {"f32", {7, 300, 5}, {1}, "sum"},
    {"bf16", {1024, 1024}, {0}, "sum"}, {"f32", {1024, 1024}, {0}, "sum"}, {"f32", {1 << 20}, {}, "sum"},
    {"bf16", {512, 4096}, {1}, "max"}, {"f32", {512, 4096}, {1}, "max"}, {"f32", {1024, 1024}, {0}, "max"},
    // block_x_reduce shapes: block_width 16 / 8 (several rows in one warp: the shuffle must stay
    // inside a row), block_width > 32 (cross-warp shared-memory stage), multi-CTA last axis
    // (block_y_reduce then block_x_reduce on the same buffer: the trailing barrier)
    {"f32", {4096, 16}, {1}, "sum"}, {"f32", {4096, 8}, {1}, "max"}, {"f32", {2, 65536}, {1}, "sum"},
    {"f32", {8, 1 << 20}, {1}, "sum"}, {"bf16", {4, 100000}, {1}},
    // integer mean (accumulator int64: same config-size mismatch; output is float32)
    {"i32", {512, 4096}, {1}}, {"i32", {1024, 1024}, {0}},
    // middle axis (outer > 1 and inner > 1: OuterContiguous with slabs)
    {"f32", {4, 64, 128}, {1}, "sum"}, {"bf16", {3, 1000, 7}, {1}}, {"f32", {2, 3, 4, 5}, {1, 2}, "max"},
    {"i32", {6, 50, 9}, {1}},
    // non-consecutive axes (Generic path)
    {"f32", {4, 5, 6}, {0, 2}, "sum"}, {"bf16", {8, 16, 32}, {0, 2}}, {"f32", {6, 7, 8, 9}, {1, 3}, "max"},
    // non-contiguous input: transposed views (Generic path)
    {"f32", {48, 64}, {0}, "sum", true}, {"f32", {48, 64}, {1}, "sum", true}, {"bf16", {300, 200}, {1}, "mean", true},
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <case>|list\n", argv[0]); return 2; }
    if (std::string(argv[1]) == "list") { std::printf("%zu\n", CASES.size()); return 0; }
    const Case& c = CASES.at(std::atoi(argv[1]));
    const std::string op = c.op;
    std::string name = op + " " + std::string(c.dtype) + " [";
    int64_t n = 1; for (size_t i = 0; i < c.shape.size(); ++i) { n *= c.shape[i]; name += std::to_string(c.shape[i]) + (i + 1 < c.shape.size() ? "," : ""); }
    name += "] axes {"; for (size_t i = 0; i < c.axes.size(); ++i) name += std::to_string(c.axes[i]) + (i + 1 < c.axes.size() ? "," : ""); name += "}";

    const std::string dt = c.dtype;
    const Dtype dtype = dt == "bf16" ? Dtype::Bfloat16 : dt == "f16" ? Dtype::Float16 : dt == "f32" ? Dtype::Float32
                      : dt == "i32" ? Dtype::Int32 : Dtype::Float64;
    const size_t esz = dt == "f64" ? 8 : (dt == "f32" || dt == "i32") ? 4 : 2;
    const std::string odt = dt == "i32" ? "f64" : dt;   // integer mean -> float64 output (as the CPU)
    std::vector<unsigned char> bytes(n * esz); std::vector<double> val(n);
    for (int64_t i = 0; i < n; ++i) {
        const float v = (float)(0.5 + 1.5 * u01());
        if (dt == "bf16") { uint16_t b = to_bf(v); std::memcpy(&bytes[i * 2], &b, 2); val[i] = from_bf(b); }
        else if (dt == "f16") { uint16_t h = to_h(v); std::memcpy(&bytes[i * 2], &h, 2); val[i] = from_h(h); }
        else if (dt == "f32") { std::memcpy(&bytes[i * 4], &v, 4); val[i] = v; }
        else if (dt == "i32") { int32_t k = (int32_t)(u01() * 10.0); std::memcpy(&bytes[i * 4], &k, 4); val[i] = k; }
        else { double d = v; std::memcpy(&bytes[i * 8], &d, 8); val[i] = d; }
    }
    // Build the CPU input. tr: fill a [B, A] tensor so that its transposed view is the logical
    // [A, B] tensor `val` describes, and reduce that (non-contiguous) view on both devices.
    Tensor t, g;
    if (c.tr) {
        const int64_t A = c.shape[0], B = c.shape[1];
        std::vector<unsigned char> pb(n * esz);
        for (int64_t i = 0; i < A; ++i)
            for (int64_t j = 0; j < B; ++j) std::memcpy(&pb[(j * A + i) * esz], &bytes[(i * B + j) * esz], esz);
        Shape ps; ps.dims = {B, A};
        Tensor tp(ps, TensorOptions().with_dtype(dtype).with_device(DeviceIndex(Device::CPU)));
        std::memcpy(tp.data(), pb.data(), pb.size());
        t = tp.t();
        g = tp.to_cuda(0).t();
        name += " TRANSPOSED" + std::string(g.is_contiguous() ? " (but contiguous?)" : " view");
    } else {
        Shape sh; sh.dims = c.shape;
        t = Tensor(sh, TensorOptions().with_dtype(dtype).with_device(DeviceIndex(Device::CPU)));
        std::memcpy(t.data(), bytes.data(), bytes.size());
        g = t.to_cuda(0);
    }
    auto run = [&](const Tensor& x) {
        return op == "sum" ? OwnTensor::reduce_sum(x, c.axes) : op == "max" ? OwnTensor::reduce_max(x, c.axes)
                                                             : OwnTensor::reduce_mean(x, c.axes);
    };
    Tensor out;
    try { out = run(g); }
    catch (const std::exception& ex) { std::printf("CASE|%s|EXCEPTION|GPU: %s\n", name.c_str(), ex.what()); return 4; }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("CASE|%s|CRASH|%s\n", name.c_str(), cudaGetErrorString(e)); return 3; }
    Tensor oc = out.to_cpu();
    Tensor cc;
    std::string cpu_err;
    try { cc = run(t); } catch (const std::exception& ex) { cpu_err = ex.what(); }

    // exact host reference in double; allow 1 output ulp (float accumulation in another order)
    std::vector<int64_t> dims = c.shape; std::vector<bool> red(dims.size(), c.axes.empty());
    for (int64_t a2 : c.axes) red[a2] = true;
    std::vector<int64_t> ostr(dims.size()); int64_t on = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) { ostr[i] = red[i] ? 0 : on; if (!red[i]) on *= dims[i]; }
    int64_t rc = 1; for (size_t i = 0; i < dims.size(); ++i) if (red[i]) rc *= dims[i];
    std::vector<double> sum(on, 0.0), mx(on, -1e300);
    for (int64_t i = 0; i < n; ++i) { int64_t rem = i, oi = 0; for (int d = (int)dims.size() - 1; d >= 0; --d) { oi += (rem % dims[d]) * ostr[d]; rem /= dims[d]; } sum[oi] += val[i]; if (val[i] > mx[oi]) mx[oi] = val[i]; }
    auto check = [&](const Tensor& res, int64_t& wrong, double& worst) {
        wrong = 0; worst = 0;
        if (res.numel() != on) { wrong = on; worst = -1; return; }
        for (int64_t o = 0; o < on; ++o) {
            const double ref = op == "sum" ? sum[o] : op == "max" ? mx[o] : sum[o] / rc; double got, ulp;
            if (odt == "bf16") { uint16_t b; std::memcpy(&b, (const char*)res.data() + o * 2, 2); got = from_bf(b); ulp = std::ldexp(1.0, std::ilogb(ref) - 7); }
            else if (odt == "f16") { uint16_t h; std::memcpy(&h, (const char*)res.data() + o * 2, 2); got = from_h(h); ulp = std::ldexp(1.0, std::ilogb(ref) - 10); }
            else if (odt == "f32") { float f; std::memcpy(&f, (const char*)res.data() + o * 4, 4); got = f; ulp = std::ldexp(1.0, std::ilogb(ref) - 23) * 64; }
            else { double d; std::memcpy(&d, (const char*)res.data() + o * 8, 8); got = d; ulp = std::ldexp(1.0, std::ilogb(ref) - 52) * 1024; }
            const double err = std::fabs(got - ref);
            if (!(err <= 1.0 * ulp)) ++wrong;
            if (err / ulp > worst) worst = err / ulp;
        }
    };
    int64_t wrong = 0, cwrong = 0; double worst = 0, cworst = 0;
    check(oc, wrong, worst);
    std::string cpu_part;
    if (!cpu_err.empty()) { cpu_part = " | CPU: EXCEPTION " + cpu_err; ++cwrong; }
    else {
        check(cc, cwrong, cworst);
        char buf[160]; std::snprintf(buf, sizeof buf, " | CPU: %lld of %lld wrong (worst %.1f ulp)", (long long)cwrong, (long long)on, cworst);
        cpu_part = buf;
    }
    // integer mean: the GPU must return the CPU's dtype and the CPU's bits (same formula:
    // exact int64 sum, then double(sum) * (1.0 / n))
    if (dt == "i32" && op == "mean" && cpu_err.empty()) {
        const bool same_dtype = cc.dtype() == oc.dtype();
        const bool same_bits = same_dtype && cc.numel() == oc.numel() &&
            std::memcmp(cc.data(), oc.data(), (size_t)oc.numel() * 8) == 0;
        cpu_part += std::string("; GPU vs CPU: dtype ") + (same_dtype ? "same" : "DIFFERENT") +
                    ", bits " + (same_bits ? "identical" : "DIFFERENT");
        if (!same_bits) ++wrong;
    }
    const bool bad = wrong || cwrong;
    std::printf("CASE|%s|%s|GPU: %lld of %lld wrong (worst %.1f ulp)%s\n", name.c_str(), bad ? "WRONG" : "ok",
                (long long)wrong, (long long)on, worst, cpu_part.c_str());
    return bad ? 1 : 0;
}
