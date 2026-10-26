// =============================================================================
// bench_rope_sft.cu  (2026-10-21)
//
// Our fused QK-RMSNorm + RoPE (fused_rms_rope_forward / _backward, the SFT trainer's path with USE_FUSED_RMS_ROPE=1) at the
// SFT shape: q [8, 18, 4096, 128], k [8, 6, 4096, 128] fp32, gammas [128], NeoX rotation, theta 10000, eps 1e-6 (the
// production YAML's rope_theta / qk_eps). Three cases, forward and backward:
//   nopos     positions = none                    -> the pretraining path (the original kernel instance)
//   identity  positions[b][t] = t                 -> the SFT path with the same values: must be BIT-IDENTICAL to nopos,
//                                                    and its time minus nopos's is the positions feature's overhead
//   packed    positions restart at every chat     -> real stage-0 packed rows 0-7 (seq_lengths.bin / seq_offsets.bin);
//                                                    a row's pad tail gets positions 0, 1, 2, ... like a short chat
// Timing: --warmup calls, then --iters calls each, CUDA events per call, median / min; GB/s from the bytes moved
// (forward reads q, k and writes q_out, k_out; backward reads grad_q, grad_k, q, k and writes dq, dk).
// usage: bench_rope_sft <packed train dir> [--warmup N] [--iters N]
// =============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "device/Device.h"
#include "ops/UnaryOps/Normalizations.h"
#include "autograd/operations/RoPEOps.h"

using namespace OwnTensor;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int64_t kB = 8, kHq = 18, kHkv = 6, kT = 4096, kHd = 128;
static constexpr float kEps = 1e-6f, kTheta = 10000.0f;

__global__ void fill_normal_like(float *p, int64_t n, uint32_t seed) {   // ~N(0,1): sum of 4 uniforms, scaled
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float s = 0.f;
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ (seed * 4u + j) * 2246822519u ^ static_cast<uint32_t>(i >> 32);
      x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
      s += static_cast<float>(x >> 8) * (1.0f / 16777216.0f) - 0.5f;
    }
    p[i] = s * 1.7320508f;
  }
}
static uint64_t checksum_dev(const void *p, size_t bytes) {
  std::vector<uint32_t> h(bytes / 4);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
  uint64_t acc = 0;
  for (size_t i = 0; i < h.size(); ++i) acc += (uint64_t)h[i] * (uint64_t)(i | 1);
  return acc;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static std::vector<char> read_file(const std::string &p) {
  FILE *f = std::fopen(p.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<char> b(n);
  if (std::fread(b.data(), 1, n, f) != (size_t)n) { std::fprintf(stderr, "short read %s\n", p.c_str()); std::exit(1); }
  std::fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <packed train dir> [--warmup N] [--iters N]\n", argv[0]); return 1; }
  const std::string dir = argv[1];
  int warmup = 20, iters = 200;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--warmup") warmup = std::atoi(argv[i + 1]);
    else if (a == "--iters") iters = std::atoi(argv[i + 1]);
  }
  const DeviceIndex gpu(Device::CUDA, 0);
  auto mk = [&](Shape s, Dtype d) { return Tensor::empty(s, TensorOptions().with_dtype(d).with_device(gpu)); };
  Tensor q = mk(Shape{{kB, kHq, kT, kHd}}, Dtype::Float32), k = mk(Shape{{kB, kHkv, kT, kHd}}, Dtype::Float32);
  Tensor gq = mk(Shape{{kB, kHq, kT, kHd}}, Dtype::Float32), gk = mk(Shape{{kB, kHkv, kT, kHd}}, Dtype::Float32);
  Tensor qg = mk(Shape{{kHd}}, Dtype::Float32), kg = mk(Shape{{kHd}}, Dtype::Float32);
  fill_normal_like<<<4096, 256>>>(q.data<float>(), q.numel(), 1);
  fill_normal_like<<<4096, 256>>>(k.data<float>(), k.numel(), 2);
  fill_normal_like<<<4096, 256>>>(gq.data<float>(), gq.numel(), 3);
  fill_normal_like<<<4096, 256>>>(gk.data<float>(), gk.numel(), 4);
  {
    std::vector<float> g(kHd);
    for (int i = 0; i < kHd; ++i) g[i] = 1.0f + 0.01f * static_cast<float>((i * 37) % 17 - 8);
    CK(cudaMemcpy(qg.data<float>(), g.data(), kHd * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kg.data<float>(), g.data(), kHd * 4, cudaMemcpyHostToDevice));
  }
  CK(cudaGetLastError());
  Tensor cache = autograd::build_rope_cache(kT, kHd, kTheta, gpu);

  // positions: identity, and the real packed rows 0-7
  std::vector<int32_t> ident(kB * kT), packed(kB * kT);
  for (int64_t b = 0; b < kB; ++b) for (int64_t t = 0; t < kT; ++t) ident[b * kT + t] = static_cast<int32_t>(t);
  {
    auto lb = read_file(dir + "/seq_lengths.bin"), ob = read_file(dir + "/seq_offsets.bin");
    const int32_t *lens = reinterpret_cast<const int32_t *>(lb.data());
    const int64_t *offs = reinterpret_cast<const int64_t *>(ob.data());
    int chats = 0;
    for (int64_t b = 0; b < kB; ++b) {
      int64_t t = 0;
      for (int64_t c = offs[b]; c < offs[b + 1]; ++c, ++chats)
        for (int32_t p = 0; p < lens[c] && t < kT; ++p) packed[b * kT + t++] = p;
      for (int32_t p = 0; t < kT; ++p) packed[b * kT + t++] = p;   // pad tail
    }
    std::printf("packed rows 0-7: %d chats (%.2f per row)\n", chats, chats / 8.0);
  }
  Tensor pos_ident = mk(Shape{{kB, kT}}, Dtype::Int32), pos_packed = mk(Shape{{kB, kT}}, Dtype::Int32);
  CK(cudaMemcpy(pos_ident.data<int32_t>(), ident.data(), ident.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(pos_packed.data<int32_t>(), packed.data(), packed.size() * 4, cudaMemcpyHostToDevice));

  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | q [%ld,%ld,%ld,%ld] k [%ld,%ld,%ld,%ld] fp32 | theta %.0f eps %g | warmup %d iters %d\n", prop.name,
              (long)kB, (long)kHq, (long)kT, (long)kHd, (long)kB, (long)kHkv, (long)kT, (long)kHd, kTheta, kEps, warmup, iters);
  const double qk_bytes = static_cast<double>(q.numel() + k.numel()) * 4.0;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  uint64_t cs[3][4];
  const char *names[3] = {"nopos", "identity", "packed"};
  const Tensor *poss[3] = {nullptr, &pos_ident, &pos_packed};
  for (int c = 0; c < 3; ++c) {
    const Tensor none;
    const Tensor &pos = poss[c] ? *poss[c] : none;
    auto fwd = [&] { return fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false, pos); };
    auto f0 = fwd();
    auto bwd = [&] { return fused_rms_rope_backward(gq, gk, q, k, qg, kg, cache, f0.q_rstd, f0.k_rstd, 0, false, pos); };
    for (int i = 0; i < warmup; ++i) { auto f = fwd(); auto g = bwd(); (void)f; (void)g; }
    CK(cudaDeviceSynchronize());
    std::vector<float> tf, tb;
    for (int i = 0; i < iters; ++i) {
      CK(cudaEventRecord(e0)); auto f = fwd(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); tf.push_back(ms);
    }
    for (int i = 0; i < iters; ++i) {
      CK(cudaEventRecord(e0)); auto g = bwd(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); tb.push_back(ms);
    }
    auto f = fwd(); auto g = bwd();
    cs[c][0] = checksum_dev(f.q_out.data(), f.q_out.numel() * 4);
    cs[c][1] = checksum_dev(f.k_out.data(), f.k_out.numel() * 4);
    cs[c][2] = checksum_dev(g.grad_q.data(), g.grad_q.numel() * 4);
    cs[c][3] = checksum_dev(g.grad_k.data(), g.grad_k.numel() * 4);
    const float fm = median(tf), bm = median(tb);
    std::printf("ROPE case=%s fwd_med_ms=%.4f fwd_min_ms=%.4f fwd_GBs=%.0f bwd_med_ms=%.4f bwd_min_ms=%.4f bwd_GBs=%.0f "
                "q_out_cs=%016llx k_out_cs=%016llx dq_cs=%016llx dk_cs=%016llx\n", names[c], fm,
                *std::min_element(tf.begin(), tf.end()), 2.0 * qk_bytes / (fm * 1e6), bm,
                *std::min_element(tb.begin(), tb.end()), 3.0 * qk_bytes / (bm * 1e6), (unsigned long long)cs[c][0],
                (unsigned long long)cs[c][1], (unsigned long long)cs[c][2], (unsigned long long)cs[c][3]);
  }
  const bool same = cs[0][0] == cs[1][0] && cs[0][1] == cs[1][1] && cs[0][2] == cs[1][2] && cs[0][3] == cs[1][3];
  std::printf("ROPE check identity_vs_nopos=%s (q_out, k_out, dq, dk checksums)\n", same ? "BIT-IDENTICAL" : "DIFFERENT");
  CK(cudaGetLastError());
  return same ? 0 : 3;
}
