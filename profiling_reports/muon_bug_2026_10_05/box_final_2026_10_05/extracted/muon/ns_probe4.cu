// ns_probe4.cu -- is the library Newton-Schulz deterministic on the EXACT input the test feeds it?
// Reads tensor 15's Newton-Schulz input (96x96 fp32) dumped by the debug copy (run_debug2.sh) and, on that one input:
//   [1] muon_frobenius_scale_cuda alone, R times          -> how many different scaled outputs?
//   [2] newton_schulz_orthogonalize (library, bf16), R times -> how many different outputs?
//   [3] same NS, but the Frobenius sum done in ONE fixed order (host, double) -> how many different outputs?
// If [1] and [2] give >1 and [3] gives 1, the multi-block atomicAdd in muon_sumsq_kernel is the cause.
// Scratch diagnostic, 2026-10-05.   usage: ns_probe4 <file.bin> [rows] [cols] [R]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <map>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "TensorLib.h"
#include "ops/Matmul.cuh"
#include "nn/optimizer/Optim.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;
namespace OwnTensor { namespace cuda { void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps); } }

static unsigned long long fp(const void* p, size_t bytes) {
    unsigned long long x = 1469598103934665603ull;
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < bytes; ++i) { x ^= b[i]; x *= 1099511628211ull; }
    return x;
}
static Tensor upload(const std::vector<float>& h, int64_t r, int64_t c) {
    Shape s; s.dims = {r, c};
    Tensor t = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
    t.set_data(h);
    return t;
}
static std::vector<float> download(const Tensor& t) {
    cudaStreamSynchronize(cuda::getCurrentStream());
    std::vector<float> h(t.numel());
    cudaMemcpy(h.data(), t.data(), h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
static void report(const char* what, const std::map<unsigned long long, int>& seen, int R) {
    std::printf("  %-62s %d different result(s) in %d calls:", what, (int)seen.size(), R);
    for (auto& [k, n] : seen) std::printf("  %016llx x%d", k, n);
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <file.bin> [rows] [cols] [R]\n", argv[0]); return 2; }
    const int64_t r = argc > 2 ? atoll(argv[2]) : 96, c = argc > 3 ? atoll(argv[3]) : 96;
    const int R = argc > 4 ? atoi(argv[4]) : 200;
    std::vector<float> in(r * c);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(in.data(), sizeof(float), in.size(), f) != in.size()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    std::fclose(f);
    std::printf("=== %s | %lldx%lld | input fingerprint %016llx\n", argv[1], (long long)r, (long long)c, fp(in.data(), in.size() * 4));

    std::map<unsigned long long, int> s1, s2, s3;
    for (int i = 0; i < R; ++i) {                                            // [1] the Frobenius scale alone
        Tensor x = upload(in, r, c);
        cuda::muon_frobenius_scale_cuda(x.data<float>(), x.numel(), 1e-8f);
        std::vector<float> h = download(x); s1[fp(h.data(), h.size() * 4)]++;
    }
    for (int i = 0; i < R; ++i) {                                            // [2] the library NS, as update_one calls it
        Tensor x = upload(in, r, c);
        Tensor u = nn::newton_schulz_orthogonalize(x, 5, 1e-8f, true);
        std::vector<float> h = download(u.contiguous()); s2[fp(h.data(), h.size() * 4)]++;
    }
    double ss = 0; for (float v : in) ss += (double)v * v;                     // [3] fixed-order norm, then the same bf16 loop
    const float scale = (float)(1.0 / std::max(std::sqrt(ss), 1e-8));
    std::vector<float> scaled(in.size()); for (size_t i = 0; i < in.size(); ++i) scaled[i] = in[i] * scale;
    for (int i = 0; i < R; ++i) {
        Tensor X = upload(scaled, r, c);
        bool tr = r > c; if (tr) X = X.t().contiguous();
        const int64_t k = X.shape().dims[0];
        cudaStream_t st = cuda::getCurrentStream();
        TensorOptions bf = X.opts().with_dtype(Dtype::Bfloat16);
        Tensor Xb = X.as_type(Dtype::Bfloat16);
        Tensor A = Tensor::empty(Shape{{k, k}}, bf), B = Tensor::empty(Shape{{k, k}}, bf), Xn = Tensor::empty(Xb.shape(), bf);
        for (int it = 0; it < 5; ++it) {
            cuda_matmul(Xb, Xb.t(), A, st);
            cuda_addmm(A, A, A, 2.0315f, -4.7750f, B, st);
            cuda_addmm(Xb, B, Xb, 1.0f, 3.4445f, Xn, st);
            std::swap(Xb, Xn);
        }
        Tensor out = Xb.as_type(Dtype::Float32);
        if (tr) out = out.t().contiguous();
        std::vector<float> h = download(out); s3[fp(h.data(), h.size() * 4)]++;
    }
    report("[1] muon_frobenius_scale_cuda (multi-block atomicAdd sum)", s1, R);
    report("[2] library newton_schulz_orthogonalize(bf16)", s2, R);
    report("[3] same NS with the sum done in one fixed order", s3, R);
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
