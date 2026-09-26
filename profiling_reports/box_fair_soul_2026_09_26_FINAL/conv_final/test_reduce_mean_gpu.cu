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

struct Case { const char* dtype; std::vector<int64_t> shape; std::vector<int64_t> axes; const char* op = "mean"; };
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
    const Dtype dtype = dt == "bf16" ? Dtype::Bfloat16 : dt == "f16" ? Dtype::Float16 : dt == "f32" ? Dtype::Float32 : Dtype::Float64;
    const size_t esz = dt == "f64" ? 8 : dt == "f32" ? 4 : 2;
    std::vector<unsigned char> bytes(n * esz); std::vector<double> val(n);
    for (int64_t i = 0; i < n; ++i) {
        const float v = (float)(0.5 + 1.5 * u01());
        if (dt == "bf16") { uint16_t b = to_bf(v); std::memcpy(&bytes[i * 2], &b, 2); val[i] = from_bf(b); }
        else if (dt == "f16") { uint16_t h = to_h(v); std::memcpy(&bytes[i * 2], &h, 2); val[i] = from_h(h); }
        else if (dt == "f32") { std::memcpy(&bytes[i * 4], &v, 4); val[i] = v; }
        else { double d = v; std::memcpy(&bytes[i * 8], &d, 8); val[i] = d; }
    }
    Shape sh; sh.dims = c.shape;
    Tensor t(sh, TensorOptions().with_dtype(dtype).with_device(DeviceIndex(Device::CPU)));
    std::memcpy(t.data(), bytes.data(), bytes.size());
    Tensor g = t.to_cuda(0);
    Tensor out = op == "sum" ? OwnTensor::reduce_sum(g, c.axes) : op == "max" ? OwnTensor::reduce_max(g, c.axes) : OwnTensor::reduce_mean(g, c.axes);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("CASE|%s|CRASH|%s\n", name.c_str(), cudaGetErrorString(e)); return 3; }
    Tensor oc = out.to_cpu();

    // exact host reference in double, then rounded to the output type; allow 1 output ulp
    // (the GPU accumulates in float, in a different order)
    std::vector<int64_t> dims = c.shape; std::vector<bool> red(dims.size(), c.axes.empty());
    for (int64_t a : c.axes) red[a] = true;
    std::vector<int64_t> ostr(dims.size()); int64_t on = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) { ostr[i] = red[i] ? 0 : on; if (!red[i]) on *= dims[i]; }
    int64_t rc = 1; for (size_t i = 0; i < dims.size(); ++i) if (red[i]) rc *= dims[i];
    std::vector<double> sum(on, 0.0), mx(on, -1e300);
    for (int64_t i = 0; i < n; ++i) { int64_t rem = i, oi = 0; for (int d = (int)dims.size() - 1; d >= 0; --d) { oi += (rem % dims[d]) * ostr[d]; rem /= dims[d]; } sum[oi] += val[i]; if (val[i] > mx[oi]) mx[oi] = val[i]; }
    int64_t wrong = 0; double worst = 0;
    for (int64_t o = 0; o < on; ++o) {
        const double ref = op == "sum" ? sum[o] : op == "max" ? mx[o] : sum[o] / rc; double got, ulp;
        if (dt == "bf16") { uint16_t b; std::memcpy(&b, (char*)oc.data() + o * 2, 2); got = from_bf(b); ulp = std::ldexp(1.0, std::ilogb(ref) - 7); }
        else if (dt == "f16") { uint16_t h; std::memcpy(&h, (char*)oc.data() + o * 2, 2); got = from_h(h); ulp = std::ldexp(1.0, std::ilogb(ref) - 10); }
        else if (dt == "f32") { float f; std::memcpy(&f, (char*)oc.data() + o * 4, 4); got = f; ulp = std::ldexp(1.0, std::ilogb(ref) - 23) * 64; }
        else { double d; std::memcpy(&d, (char*)oc.data() + o * 8, 8); got = d; ulp = std::ldexp(1.0, std::ilogb(ref) - 52) * 1024; }
        const double err = std::fabs(got - ref);
        if (!(err <= 1.0 * ulp)) ++wrong;
        if (err / ulp > worst) worst = err / ulp;
    }
    std::printf("CASE|%s|%s|%lld of %lld outputs off by more than 1 ulp (worst %.1f ulp)\n", name.c_str(), wrong ? "WRONG" : "ok", (long long)wrong, (long long)on, worst);
    return wrong ? 1 : 0;
}
