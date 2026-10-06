// ns_probe3.cu -- does the bf16 Newton-Schulz depend on leftover CONTENTS (a real uninitialized read) or only on
// buffer ADDRESSES (cuBLAS choosing a different kernel by alignment)? Every buffer stays at the SAME address across
// runs; only its starting contents change (zeros vs garbage), one buffer at a time. Same input every run.
// Part B moves the buffers to other addresses with clean contents. Scratch diagnostic, 2026-10-05.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "TensorLib.h"
#include "ops/Matmul.cuh"
#include "device/DeviceCore.h"

using namespace OwnTensor;

static float synth(uint64_t s) {
    s ^= s >> 33; s *= 0xff51afd7ed558ccdull; s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull; s ^= s >> 33;
    return static_cast<float>(static_cast<int64_t>(s % 20001) - 10000) * 1e-6f;
}

struct Bufs { Tensor Xb, A, B, Xn; };

static Bufs alloc(int64_t k, int64_t n, const TensorOptions& bf) {
    return Bufs{ Tensor::empty(Shape{{k, n}}, bf), Tensor::empty(Shape{{k, k}}, bf),
                 Tensor::empty(Shape{{k, k}}, bf), Tensor::empty(Shape{{k, n}}, bf) };
}

// The loop of newton_schulz_orthogonalize's bf16 branch (Optim.cpp:1561-1566), on given buffers.
static std::vector<uint16_t> run(Bufs bufs, const Tensor& Xb0, int fillA, int fillB, int fillXn, cudaStream_t st) {
    cudaMemcpyAsync(bufs.Xb.data(), Xb0.data(), Xb0.nbytes(), cudaMemcpyDeviceToDevice, st);
    cudaMemsetAsync(bufs.A.data(),  fillA,  bufs.A.nbytes(),  st);
    cudaMemsetAsync(bufs.B.data(),  fillB,  bufs.B.nbytes(),  st);
    cudaMemsetAsync(bufs.Xn.data(), fillXn, bufs.Xn.nbytes(), st);
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    Tensor Xb = bufs.Xb, Xn = bufs.Xn;
    for (int i = 0; i < 5; ++i) {
        cuda_matmul(Xb, Xb.t(), bufs.A, st);
        cuda_addmm(bufs.A, bufs.A, bufs.A, c, b, bufs.B, st);
        cuda_addmm(Xb, bufs.B, Xb, 1.0f, a, Xn, st);
        std::swap(Xb, Xn);
    }
    cudaStreamSynchronize(st);
    std::vector<uint16_t> h(Xb.numel());
    cudaMemcpy(h.data(), Xb.data(), Xb.nbytes(), cudaMemcpyDeviceToHost);
    return h;
}

static int64_t ndiff(const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
    int64_t d = 0; for (size_t i = 0; i < x.size(); ++i) d += (x[i] != y[i]); return d;
}

int main() {
    cudaStream_t st = cuda::getCurrentStream();
    for (auto [r, c] : std::vector<std::pair<int64_t, int64_t>>{{64, 64}, {96, 96}, {128, 64}, {48, 48}}) {
        std::vector<float> h(r * c);
        for (size_t i = 0; i < h.size(); ++i) h[i] = synth(777 + i);
        Shape s; s.dims = {r, c};
        Tensor X = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
        X.set_data(h);
        if (r > c) X = X.t().contiguous();
        const int64_t k = X.shape().dims[0], n = X.shape().dims[1];
        TensorOptions bf = X.opts().with_dtype(Dtype::Bfloat16);
        Tensor Xb0 = X.as_type(Dtype::Bfloat16);
        Bufs b1 = alloc(k, n, bf);
        std::vector<uint16_t> base  = run(b1, Xb0, 0x00, 0x00, 0x00, st);
        std::vector<uint16_t> again = run(b1, Xb0, 0x00, 0x00, 0x00, st);
        std::vector<uint16_t> gA    = run(b1, Xb0, 0x46, 0x00, 0x00, st);
        std::vector<uint16_t> gB    = run(b1, Xb0, 0x00, 0x46, 0x00, st);
        std::vector<uint16_t> gXn   = run(b1, Xb0, 0x00, 0x00, 0x46, st);
        std::vector<uint16_t> gAll  = run(b1, Xb0, 0x46, 0x46, 0x46, st);
        // Part B: same contents (zeros), different addresses (allocate a padding block first to shift them).
        Tensor pad = Tensor::empty(Shape{{1, 777}}, bf);
        Bufs b2 = alloc(k, n, bf);
        std::vector<uint16_t> moved = run(b2, Xb0, 0x00, 0x00, 0x00, st);
        std::printf("shape %3lldx%-3lld (k=%lld) | elements that differ from the zero-filled run (of %zu):\n",
                    (long long)r, (long long)c, (long long)k, base.size());
        std::printf("   same again %lld | garbage in A only %lld | in B only %lld | in Xb_new only %lld | in all %lld | "
                    "zeros but OTHER addresses %lld  (A at %p vs %p)\n",
                    (long long)ndiff(base, again), (long long)ndiff(base, gA), (long long)ndiff(base, gB),
                    (long long)ndiff(base, gXn), (long long)ndiff(base, gAll), (long long)ndiff(base, moved),
                    b1.A.data(), b2.A.data());
    }
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
