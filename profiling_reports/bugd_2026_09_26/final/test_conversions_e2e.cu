// test_conversions_e2e.cu -- end-to-end check of the conversion fixes through real library ops (2026-09-26).
//
// Every case is built so that the OLD code and the NEW code give different bits, and the
// expected answer is exact (no dependence on FMA contraction or tanh accuracy):
//   E1 GPU  bf16 tensor * 2.0f and tensor *= 2.0f, values >= 65,536
//                                                        OLD: +-Inf (Bug D clamp)   NEW: exact doubles
//   E2 CPU  reduce_mean over 4 bf16 values whose exact mean is 1 + 2^-8 + 2^-30
//           (double accumulation)                        OLD: 0x3F80 (two roundings) NEW: 0x3F81
//   E3 CPU  reduce_mean over axis 0 of a [2, N] bf16 tensor whose column means are exact ties
//           with an odd low bit (1.01171875)             truncation: 0x3F81          NEW: 0x3F82
//   E4 CPU  gelu_forward on 7 bf16 values (all go through the one-by-one tail); each value is
//           picked so its fp32 result has low 16 bits in [0xA000, 0xE000], far from a tie:
//           truncation gives the lower bf16, rounding the upper one
//   (The 8-wide vector store is covered by E3, whose expected answer is exact; the vector GELU
//    uses an approximate tanh, so it cannot give an exact expected value.)
//
// Build + run from the Tensor-Implementations root, after the patched library is built:
//   make run-cuda FILE=Tests/kernel_tests/test_conversions_e2e.cu WITH_BLUBLAS=1
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/ScalarOps.h"
#include "ops/UnaryOps/Activations.h"
#include "ops/UnaryOps/Reduction.h"

using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static Tensor bf16_cpu(const std::vector<uint16_t>& bits, Shape s) {
    Tensor t(s, TensorOptions().with_dtype(Dtype::Bfloat16).with_device(DeviceIndex(Device::CPU)));
    std::memcpy(t.data(), bits.data(), bits.size() * 2);
    return t;
}
static std::vector<uint16_t> bits_of(const Tensor& t) {
    std::vector<uint16_t> out(t.numel());
    if (t.device().is_cuda()) CK(cudaMemcpy(out.data(), t.data(), out.size() * 2, cudaMemcpyDeviceToHost));
    else std::memcpy(out.data(), t.data(), out.size() * 2);
    return out;
}
static float f_of(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
// the library's scalar GELU formula (Activations.cpp, remainder loop)
static float gelu_ref(float x) {
    const float beta_s = std::sqrt(2.0f / M_PI);
    const float x3 = x * x * x;
    const float inner = beta_s * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
}
// bf16 inputs whose GELU result has low 16 bits in [0xA000, 0xE000] (rounding goes up, truncation down)
static std::vector<uint16_t> gelu_inputs(int n) {
    std::vector<uint16_t> v;
    for (uint32_t b = 0x3F80; b < 0x4100 && (int)v.size() < n; b += 3) {
        const uint32_t lo = u_of(gelu_ref(f_of((uint16_t)b))) & 0xFFFFu;
        if (lo >= 0xA000u && lo <= 0xE000u) v.push_back((uint16_t)b);
    }
    return v;
}

int main() {
    int fails = 0;
    auto check = [&](const char* name, bool ok, const char* detail) {
        std::printf("   %-62s %s   %s\n", name, ok ? "ok  " : "FAIL", detail); fails += !ok;
    };
    char buf[256];

    // ---- E1: GPU bf16 * 2.0f with values >= 65,536 ----
    {
        // raw bf16 bits: 70144 (0x4789), 65536 (0x4780), 1000 (0x447A), -999424 (0xC974)
        const std::vector<uint16_t> in = {0x4789, 0x4780, 0x447A, 0xC974};
        const std::vector<uint16_t> want = {0x4809, 0x4800, 0x44FA, 0xC9F4};   // x2, all exact in bf16
        Tensor t = bf16_cpu(in, Shape{{4}}).to_cuda(0);
        Tensor r = t * 2.0f;
        CK(cudaDeviceSynchronize());
        const std::vector<uint16_t> got = bits_of(r);
        bool ok = got == want;
        std::snprintf(buf, sizeof buf, "got %04x %04x %04x %04x  want %04x %04x %04x %04x", got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
        check("E1 GPU bf16 tensor * 2.0f, |x| >= 65,536 (Bug D)", ok, buf);
        Tensor u = bf16_cpu(in, Shape{{4}}).to_cuda(0);
        u *= 2.0f;                                           // in place: scalar_mul_inplace<bfloat16_t>
        CK(cudaDeviceSynchronize());
        const std::vector<uint16_t> got2 = bits_of(u);
        std::snprintf(buf, sizeof buf, "got %04x %04x %04x %04x", got2[0], got2[1], got2[2], got2[3]);
        check("E1 GPU bf16 tensor *= 2.0f, |x| >= 65,536 (Bug D)", got2 == want, buf);
    }
    // ---- E2: CPU mean, exact double mean just above a bf16 tie ----
    {
        // 2, 2, 2^-6, 2^-28 -> sum 4 + 2^-6 + 2^-28 (exact in double), mean 1 + 2^-8 + 2^-30
        const std::vector<uint16_t> in = {0x4000, 0x4000, 0x3C80, 0x3180};
        Tensor m = OwnTensor::reduce_mean(bf16_cpu(in, Shape{{4}}));
        const uint16_t got = bits_of(m)[0];
        std::snprintf(buf, sizeof buf, "got 0x%04x  want 0x3f81 (0x3f80 = two roundings, or a float-accumulating path)", got);
        check("E2 CPU reduce_mean bf16, mean = 1 + 2^-8 + 2^-30", got == 0x3F81, buf);
    }
    // ---- E3: CPU mean over axis 0, exact ties with an odd low bit ----
    {
        const int N = 24;                                   // 3 x 8 lanes
        std::vector<uint16_t> in(2 * N);
        for (int j = 0; j < N; ++j) { in[j] = 0x3F82; in[N + j] = 0x3F81; }   // (1.015625 + 1.0078125) / 2 = 1.01171875
        Tensor m = OwnTensor::reduce_mean(bf16_cpu(in, Shape{{2, N}}), {0});
        const std::vector<uint16_t> got = bits_of(m);
        int bad = 0, trunc = 0;
        for (uint16_t g : got) { bad += g != 0x3F82; trunc += g == 0x3F81; }
        std::snprintf(buf, sizeof buf, "%d of %d wrong (%d truncated to 0x3f81); want all 0x3f82", bad, N, trunc);
        check("E3 CPU reduce_mean bf16 over axis 0 (vector store path)", bad == 0, buf);
    }
    // ---- E4: CPU GELU, 7 elements, all through the one-by-one tail ----
    {
        const std::vector<uint16_t> in = gelu_inputs(7);
        const std::vector<uint16_t> got = bits_of(OwnTensor::gelu_forward(bf16_cpu(in, Shape{{(int64_t)in.size()}})));
        int bad = 0;
        for (size_t i = 0; i < in.size(); ++i) {
            const uint32_t r = u_of(gelu_ref(f_of(in[i])));
            const uint16_t up = (uint16_t)((r >> 16) + 1);            // low bits in [0xA000, 0xE000] -> rounds up
            bad += got[i] != up;
        }
        std::snprintf(buf, sizeof buf, "%d of %zu not rounded up (truncation gives the lower value)", bad, in.size());
        check("E4 CPU gelu_forward bf16, 7 elements (scalar tail)", bad == 0 && in.size() == 7, buf);
    }
    std::printf("\nRESULT: %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
