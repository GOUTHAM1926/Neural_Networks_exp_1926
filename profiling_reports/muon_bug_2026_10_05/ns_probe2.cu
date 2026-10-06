// ns_probe2.cu -- which buffer inside the bf16 Newton-Schulz is read before it is written?
// Repeats newton_schulz_orthogonalize's bf16 loop step by step (Optim.cpp:1554-1566), prints every buffer's
// device address range, and is meant to run under `compute-sanitizer --tool initcheck`: the address in the
// first "Uninitialized __global__ memory read" names the buffer. Scratch diagnostic, 2026-10-05.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "TensorLib.h"
#include "ops/Matmul.cuh"
#include "device/DeviceCore.h"

using namespace OwnTensor;

static float synth(uint64_t s) {
    s ^= s >> 33; s *= 0xff51afd7ed558ccdull; s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ull; s ^= s >> 33;
    return static_cast<float>(static_cast<int64_t>(s % 20001) - 10000) * 1e-6f;   // small: stays stable without the norm scale
}
static void range(const char* name, const Tensor& t) {
    const char* p = static_cast<const char*>(t.data());
    std::printf("  %-8s [%p, %p)  %zu bytes  shape %lldx%lld\n", name, (const void*)p, (const void*)(p + t.nbytes()),
                t.nbytes(), (long long)t.shape().dims[0], (long long)t.shape().dims[1]);
}

int main(int argc, char** argv) {
    const int64_t r = argc > 1 ? atoll(argv[1]) : 64, c = argc > 2 ? atoll(argv[2]) : 64;
    cudaStream_t st = cuda::getCurrentStream();
    std::vector<float> h(r * c);
    for (size_t i = 0; i < h.size(); ++i) h[i] = synth(12345 + i);
    Shape s; s.dims = {r, c};
    Tensor X = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
    X.set_data(h);
    if (r > c) X = X.t().contiguous();
    const int64_t k = X.shape().dims[0];
    const float a = 3.4445f, b = -4.7750f, cc = 2.0315f;
    TensorOptions bf = X.opts().with_dtype(Dtype::Bfloat16);
    Tensor Xb = X.as_type(Dtype::Bfloat16);
    Tensor A = Tensor::empty(Shape{{k, k}}, bf);
    Tensor B = Tensor::empty(Shape{{k, k}}, bf);
    Tensor Xn = Tensor::empty(Xb.shape(), bf);
    std::printf("=== bf16 Newton-Schulz by hand, input %lldx%lld (k=%lld)\n", (long long)r, (long long)c, (long long)k);
    range("X", X); range("Xb", Xb); range("A", A); range("B", B); range("Xb_new", Xn);
    for (int i = 0; i < 5; ++i) {
        std::printf("--- iteration %d: A = Xb @ Xb.T\n", i);           cudaStreamSynchronize(st);
        cuda_matmul(Xb, Xb.t(), A, st);                               cudaStreamSynchronize(st);
        std::printf("--- iteration %d: B = b*A + c*(A@A)\n", i);       cudaStreamSynchronize(st);
        cuda_addmm(A, A, A, cc, b, B, st);                            cudaStreamSynchronize(st);
        std::printf("--- iteration %d: Xb_new = a*Xb + B@Xb\n", i);   cudaStreamSynchronize(st);
        cuda_addmm(Xb, B, Xb, 1.0f, a, Xn, st);                       cudaStreamSynchronize(st);
        std::swap(Xb, Xn);
        std::printf("    after swap: Xb=%p Xb_new=%p\n", Xb.data(), Xn.data());
    }
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
