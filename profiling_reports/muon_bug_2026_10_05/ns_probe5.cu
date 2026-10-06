// ns_probe5.cu -- is the REDUCE_BROADCAST-only "scaled" value (309adc60...) just another order of the 9 atomicAdds
// in muon_sumsq_kernel? Input: tensor 15's step-4 Newton-Schulz input dumped by run_debug2.sh.
//  part 1: the library's own kernel code (copied verbatim at build time from MuonKernels.cu, see run_probe5.sh) with the
//          atomicAdd replaced by "store my block's sum", gives the 9 block sums. The host adds them in all 9! orders,
//          the way atomicAdd does (one float add at a time onto 0), and runs the verbatim scale kernel for every distinct
//          total. Which totals give the RS-mode value b3f448f1, and which give the RB-mode value 309adc60?
//  part 2: the REAL library function muon_frobenius_scale_cuda, 200 calls per condition: idle GPU, L2 flushed first,
//          other kernels busy on another stream. Does the RB-mode value appear outside the test?
// Scratch diagnostic, 2026-10-05.   usage: ns_probe5 <nsin_step4.bin>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <map>
#include <vector>
#include <cuda_runtime.h>

namespace OwnTensor { namespace cuda { void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps); } }

namespace copy {
//@@KERNELS@@
}

static unsigned long long fp_dev(const float* d, int64_t n) {
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long long x = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (size_t i = 0; i < h.size() * sizeof(float); ++i) { x ^= b[i]; x *= 1099511628211ull; }
    return x;
}
static unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
__global__ void spin(long long cycles) { long long t = clock64(); while (clock64() - t < cycles) {} }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <nsin_step4.bin>\n", argv[0]); return 2; }
    const int64_t n = 96 * 96;
    std::vector<float> in(n);
    FILE* f = std::fopen(argv[1], "rb");
    if (!f || std::fread(in.data(), sizeof(float), n, f) != (size_t)n) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    std::fclose(f);
    float *x0, *x, *acc, *part;
    cudaMalloc(&x0, n * 4); cudaMalloc(&x, n * 4); cudaMalloc(&acc, 4); cudaMalloc(&part, 64 * 4);
    cudaMemcpy(x0, in.data(), n * 4, cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = (int)((n / 4 + threads - 1) / threads);    // need4 in muon_frobenius_scale_cuda = 9 here
    std::printf("=== %s | 96x96 | the library launches muon_sumsq_kernel with %d blocks for this size\n", argv[1], blocks);

    // ---- part 1 ----
    copy::muon_sumsq_partials<<<blocks, threads>>>(x0, part, n);
    std::vector<float> p(blocks);
    cudaMemcpy(p.data(), part, blocks * 4, cudaMemcpyDeviceToHost);
    std::printf("part 1: the %d block sums:", blocks);
    for (float v : p) std::printf(" %.9g", v);
    std::printf("\n");
    std::vector<int> order(blocks);
    for (int i = 0; i < blocks; ++i) order[i] = i;
    std::map<unsigned, long> totals;          // float bits -> number of orders that give it
    unsigned in_block_order = 0;
    bool first = true;
    do {
        volatile float s = 0.0f;              // volatile: one IEEE float add per step, like atomicAdd
        for (int i : order) s = s + p[i];
        totals[bits(s)]++;
        if (first) { in_block_order = bits(s); first = false; }
    } while (std::next_permutation(order.begin(), order.end()));
    std::printf("part 1: %zu different totals over all add orders\n", totals.size());
    for (auto& [b, cnt] : totals) {
        float s; std::memcpy(&s, &b, 4);
        cudaMemcpy(acc, &s, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(x, x0, n * 4, cudaMemcpyDeviceToDevice);
        copy::muon_scale_by_inv_frob_kernel<<<blocks, threads>>>(x, acc, n, 1e-8f);
        const unsigned long long fpv = fp_dev(x, n);
        const char* tag = ((fpv >> 32) == 0xb3f448f1ull) ? "  <- the RS-mode / standalone value b3f448f1"
                        : ((fpv >> 32) == 0x309adc60ull) ? "  <- the RB-mode value 309adc60" : "";
        std::printf("   total %.9g (bits %08x) from %6ld orders%s -> scaled fingerprint %016llx%s\n", s, b, cnt,
                    b == in_block_order ? " [block order 0..8]" : "", fpv, tag);
    }

    // ---- part 2 ----
    cudaStream_t other; cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking);
    char* junk; const size_t junk_bytes = 512ull << 20; cudaMalloc(&junk, junk_bytes);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    auto trial = [&](const char* name, int mode) {
        std::map<unsigned long long, int> seen;
        for (int r = 0; r < 200; ++r) {
            cudaMemcpy(x, x0, n * 4, cudaMemcpyDeviceToDevice);
            if (mode == 1) cudaMemset(junk, r & 0xff, junk_bytes);                       // push x out of L2
            if (mode == 2) spin<<<sms - 4, 1024, 0, other>>>(2000000);                    // most SMs busy
            if (mode == 3) spin<<<(r % sms) + 1, 1024, 0, other>>>(200000 + 5000 * r);   // a varying number of SMs busy
            OwnTensor::cuda::muon_frobenius_scale_cuda(x, n, 1e-8f);
            seen[fp_dev(x, n)]++;
        }
        std::printf("   %-46s", name);
        for (auto& [k, c] : seen) std::printf("  %016llx x%d", k, c);
        std::printf("\n");
    };
    std::printf("part 2: the real library muon_frobenius_scale_cuda, 200 calls each (fingerprint x times seen)\n");
    trial("idle GPU", 0);
    trial("L2 flushed before each call", 1);
    trial("all but 4 SMs busy on another stream", 2);
    trial("1..132 SMs busy on another stream (varies)", 3);
    std::printf("=== CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
