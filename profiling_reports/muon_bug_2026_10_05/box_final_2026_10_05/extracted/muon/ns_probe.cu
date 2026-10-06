// ns_probe.cu -- does cuda_matmul(Xb, Xb.t(), A) (the first GEMM of the bf16 Newton-Schulz) write all of A?
// And does newton_schulz_orthogonalize(use_bf16=true) depend on what memory held before?
// Scratch diagnostic for the test_muon_rsag W=1 failure (2026-10-05). Uses only library functions.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "TensorLib.h"
#include "ops/Matmul.cuh"
#include "nn/optimizer/Optim.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;

static float synth(uint64_t s) {
    s ^= s >> 33; s *= 0xff51afd7ed558ccdull; s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull; s ^= s >> 33;
    return static_cast<float>(static_cast<int64_t>(s % 20001) - 10000) * 1e-4f;
}
static float bf2f(uint16_t b) { uint32_t u = static_cast<uint32_t>(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

static Tensor make_f32(int64_t r, int64_t c, uint64_t seed) {
    std::vector<float> h(r * c);
    for (size_t i = 0; i < h.size(); ++i) h[i] = synth(seed * 7919ull + i);
    Shape s; s.dims = {r, c};
    Tensor t = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
    t.set_data(h);
    return t;
}

int main() {
    cudaStream_t st = cuda::getCurrentStream();
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {96, 96}, {64, 64}, {32, 32}, {48, 48}, {56, 56}, {40, 40}, {128, 64}, {64, 128}, {96, 48}, {100, 50}, {160, 32}};
    std::printf("=== part 1: cuda_matmul(Xb, Xb.t(), A) with A pre-filled with a marker (0x4141)\n");
    for (auto [r, c] : shapes) {
        Tensor X = make_f32(r, c, r * 1000 + c);
        if (r > c) X = X.t().contiguous();                       // what newton_schulz_orthogonalize does
        const int64_t k = X.shape().dims[0], n = X.shape().dims[1];
        Tensor Xb = X.as_type(Dtype::Bfloat16);
        Shape sa; sa.dims = {k, k};
        Tensor A = Tensor::empty(sa, TensorOptions().with_dtype(Dtype::Bfloat16).with_device(DeviceIndex(Device::CUDA, 0)));
        cudaMemsetAsync(A.data(), 0x41, A.nbytes(), st);
        cuda_matmul(Xb, Xb.t(), A, st);
        cudaStreamSynchronize(st);
        std::vector<uint16_t> ha(k * k), hx(k * n);
        cudaMemcpy(ha.data(), A.data(), k * k * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hx.data(), Xb.data(), k * n * 2, cudaMemcpyDeviceToHost);
        int64_t marker = 0; double maxerr = 0;
        for (int64_t i = 0; i < k; ++i) for (int64_t j = 0; j < k; ++j) {
            if (ha[i * k + j] == 0x4141) ++marker;
            double ref = 0; for (int64_t t = 0; t < n; ++t) ref += (double)bf2f(hx[i * n + t]) * bf2f(hx[j * n + t]);
            maxerr = std::max(maxerr, std::fabs((double)bf2f(ha[i * k + j]) - ref) / (std::fabs(ref) + 1e-3));
        }
        std::printf("  shape %3lldx%-3lld -> X %3lldx%-3lld A %3lldx%-3lld | A entries still = marker: %6lld of %6lld | max rel err vs CPU %.3g %s\n",
                    (long long)r, (long long)c, (long long)k, (long long)n, (long long)k, (long long)k,
                    (long long)marker, (long long)(k * k), maxerr, (marker || maxerr > 0.05) ? "<-- A NOT WRITTEN CORRECTLY" : "ok");
    }
    std::printf("=== part 2: newton_schulz_orthogonalize(use_bf16) twice on the same input, freed memory poisoned in between\n");
    for (auto [r, c] : shapes) {
        Tensor G1 = make_f32(r, c, 42 + r + c), G2 = make_f32(r, c, 42 + r + c);
        Tensor U1 = nn::newton_schulz_orthogonalize(G1, 5, 1e-8f, true);
        cudaStreamSynchronize(st);
        std::vector<float> h1(U1.numel()); cudaMemcpy(h1.data(), U1.data(), h1.size() * 4, cudaMemcpyDeviceToHost);
        {   // fill a few freed blocks of every size class the NS temporaries can land in with garbage (bf16 ~ 1e4)
            std::vector<Tensor> junk;
            for (int64_t sz : {r * c, r * r, c * c, r * c * 2, 9216LL * 2, 16384LL})
                for (int rep = 0; rep < 3; ++rep) {
                    Shape s; s.dims = {sz};
                    Tensor t = Tensor::empty(s, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
                    cudaMemsetAsync(t.data(), 0x46, t.nbytes(), st);
                    junk.push_back(t);
                }
            cudaStreamSynchronize(st);
        }   // junk freed here -> back to the caching allocator with garbage inside
        Tensor U2 = nn::newton_schulz_orthogonalize(G2, 5, 1e-8f, true);
        cudaStreamSynchronize(st);
        std::vector<float> h2(U2.numel()); cudaMemcpy(h2.data(), U2.data(), h2.size() * 4, cudaMemcpyDeviceToHost);
        int64_t diff = 0; for (size_t i = 0; i < h1.size(); ++i) diff += (std::memcmp(&h1[i], &h2[i], 4) != 0);
        std::printf("  shape %3lldx%-3lld | NS output elements that differ (clean vs poisoned memory): %6lld of %6zu %s\n",
                    (long long)r, (long long)c, (long long)diff, h1.size(), diff ? "<-- DEPENDS ON LEFTOVER MEMORY" : "same");
    }
    cudaError_t e = cudaGetLastError();
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(e));
    return 0;
}
