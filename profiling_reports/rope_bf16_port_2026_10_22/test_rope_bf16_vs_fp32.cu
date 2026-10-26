// =============================================================================
// test_rope_bf16_vs_fp32.cu  (2026-10-22, stage A of the bf16 RoPE port)
//
// Does Kruthik's bf16 fused QK-RMSNorm + RoPE (fused_rms_rope_*_bf16_tm, BLUCCL a60099b) give the same numbers as the
// fp32 path our SFT trainer uses today? Same random inputs through both, every output compared bit for bit:
//   fp32 path (bluscript_sft.cpp 640-652):  q bf16 [B,T,Hq*hd] -> reshape [B,T,Hq,hd] -> transpose(1,2) -> contiguous
//            -> to fp32 -> fused_rms_rope_forward (fp32) -> to bf16;  backward: grad bf16 -> to fp32 ->
//            fused_rms_rope_backward (fp32) -> to bf16 -> transpose back to [B,T,Hq*hd]
//   bf16 path (production trainer default): fused_rms_rope_forward_bf16_tm (bf16 in, bf16 out) and
//            fused_rms_rope_backward_bf16_tm
// Compared: q_out, k_out (bf16), q_rstd, k_rstd (fp32), dq, dk (bf16), dgamma_q, dgamma_k (fp32).
// Then both paths are timed (CUDA events, median), the fp32 path WITH its copies, as the trainer runs it.
// Shape: T 4096, 18 query / 6 KV heads, head dim 128, NeoX, eps 1e-6, theta 10000; B = 4 and 8.
// usage: test_rope_bf16_vs_fp32 [--iters N]
// =============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "device/Device.h"
#include "ops/UnaryOps/Normalizations.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/RoPEOps.h"

using namespace OwnTensor;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int64_t kT = 4096, kHq = 18, kHkv = 6, kHd = 128;
static constexpr float kEps = 1e-6f, kTheta = 10000.0f;

static uint16_t f2bf(float f) {   // round to nearest even, as __float2bfloat16_rn (no NaN in these inputs)
  uint32_t u; std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
template <typename T> static std::vector<T> host(const Tensor& t) {
  std::vector<T> h(t.numel());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), t.data(), h.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
template <typename T> static int64_t mismatches(const std::vector<T>& a, const std::vector<T>& b) {
  int64_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += (std::memcmp(&a[i], &b[i], sizeof(T)) != 0);
  return n;
}
static double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i]) / std::max(1e-30, std::fabs((double)b[i])));
  return m;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  int iters = 200;
  for (int i = 1; i + 1 < argc; i += 2) if (std::string(argv[i]) == "--iters") iters = std::atoi(argv[i + 1]);
  const DeviceIndex gpu(Device::CUDA, 0);
  auto opts = [&](Dtype d) { return TensorOptions().with_dtype(d).with_device(gpu); };
  Tensor cache = autograd::build_rope_cache(kT, kHd, kTheta, gpu);
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | T %ld, %ld / %ld heads, hd %ld, NeoX, eps %g, theta %.0f | %d timed calls\n", prop.name,
              (long)kT, (long)kHq, (long)kHkv, (long)kHd, kEps, kTheta, iters);
  bool all_equal = true;

  for (int64_t B : {4, 8}) {
    // ---- inputs: bf16 q, k token-major (what the projections give), bf16 grads head-major, fp32 gammas ----
    std::vector<uint16_t> hq(B * kT * kHq * kHd), hk(B * kT * kHkv * kHd), hgq(B * kHq * kT * kHd), hgk(B * kHkv * kT * kHd);
    uint64_t s = 0x9E3779B97F4A7C15ull + B;
    auto rnd = [&]() {   // ~N(0,1): sum of 4 uniforms
      float acc = 0.f;
      for (int j = 0; j < 4; ++j) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; acc += (float)((s >> 40) * (1.0 / 16777216.0)) - 0.5f; }
      return acc * 1.7320508f;
    };
    for (auto* v : {&hq, &hk, &hgq, &hgk}) for (auto& x : *v) x = f2bf(rnd());
    std::vector<float> hg(kHd);
    for (int i = 0; i < kHd; ++i) hg[i] = 1.0f + 0.01f * static_cast<float>((i * 37) % 17 - 8);
    Tensor q = Tensor::empty(Shape{{B, kT, kHq * kHd}}, opts(Dtype::Bfloat16));
    Tensor k = Tensor::empty(Shape{{B, kT, kHkv * kHd}}, opts(Dtype::Bfloat16));
    Tensor gq = Tensor::empty(Shape{{B, kHq, kT, kHd}}, opts(Dtype::Bfloat16));
    Tensor gk = Tensor::empty(Shape{{B, kHkv, kT, kHd}}, opts(Dtype::Bfloat16));
    Tensor qg = Tensor::empty(Shape{{kHd}}, opts(Dtype::Float32)), kg = Tensor::empty(Shape{{kHd}}, opts(Dtype::Float32));
    CK(cudaMemcpy(q.data(), hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(k.data(), hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gq.data(), hgq.data(), hgq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gk.data(), hgk.data(), hgk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(qg.data(), hg.data(), kHd * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kg.data(), hg.data(), kHd * 4, cudaMemcpyHostToDevice));

    // ---- the fp32 path, as bluscript_sft.cpp runs it ----
    auto to4d = [&](const Tensor& x, int64_t H) {
      return autograd::to_dtype(autograd::contiguous(autograd::transpose(autograd::reshape(x, Shape{{B, kT, H, kHd}}), 1, 2)),
                                Dtype::Float32);
    };
    Tensor q4, k4; FusedRMSRoPEForwardResult f32;
    auto fwd32 = [&]() {
      q4 = to4d(q, kHq); k4 = to4d(k, kHkv);
      f32 = fused_rms_rope_forward(q4, k4, qg, kg, cache, 0, kEps, false);
      return std::make_pair(autograd::to_dtype(f32.q_out, Dtype::Bfloat16), autograd::to_dtype(f32.k_out, Dtype::Bfloat16));
    };
    auto bwd32 = [&]() {
      auto r = fused_rms_rope_backward(autograd::to_dtype(gq, Dtype::Float32), autograd::to_dtype(gk, Dtype::Float32),
                                       q4, k4, qg, kg, cache, f32.q_rstd, f32.k_rstd, 0, false);
      auto back = [&](const Tensor& d, int64_t H) {   // fp32 [B,H,T,hd] -> bf16 [B,T,H*hd]
        return autograd::reshape(autograd::contiguous(autograd::transpose(autograd::to_dtype(d, Dtype::Bfloat16), 1, 2)),
                                 Shape{{B, kT, H * kHd}});
      };
      return std::make_tuple(back(r.grad_q, kHq), back(r.grad_k, kHkv), r.grad_q_gamma, r.grad_k_gamma);
    };
    // ---- the bf16 path ----
    FusedRMSRoPEForwardResult f16;
    auto fwd16 = [&]() { f16 = fused_rms_rope_forward_bf16_tm(q, k, kHq, kHkv, qg, kg, cache, 0, kEps, false); };
    auto bwd16 = [&]() {
      return fused_rms_rope_backward_bf16_tm(gq, gk, q, k, qg, kg, cache, f16.q_rstd, f16.k_rstd, 0, false);
    };

    // ---- values ----
    auto [q32, k32] = fwd32();
    auto [dq32, dk32, dgq32, dgk32] = bwd32();
    fwd16();
    auto b16 = bwd16();
    struct Row { const char* name; int64_t bad, n; double rel; };
    std::vector<Row> rows;
    auto cmp16 = [&](const char* nm, const Tensor& a, const Tensor& b) {
      auto ha = host<uint16_t>(a), hb = host<uint16_t>(b); rows.push_back({nm, mismatches(ha, hb), (int64_t)ha.size(), 0.0});
    };
    auto cmp32 = [&](const char* nm, const Tensor& a, const Tensor& b) {
      auto ha = host<float>(a), hb = host<float>(b); rows.push_back({nm, mismatches(ha, hb), (int64_t)ha.size(), max_rel(ha, hb)});
    };
    cmp16("q_out (bf16)", q32, f16.q_out);
    cmp16("k_out (bf16)", k32, f16.k_out);
    cmp32("q_rstd (fp32)", f32.q_rstd, f16.q_rstd);
    cmp32("k_rstd (fp32)", f32.k_rstd, f16.k_rstd);
    cmp16("dq (bf16)", dq32, b16.grad_q);
    cmp16("dk (bf16)", dk32, b16.grad_k);
    cmp32("dgamma_q (fp32)", dgq32, b16.grad_q_gamma);
    cmp32("dgamma_k (fp32)", dgk32, b16.grad_k_gamma);
    for (const Row& r : rows) {
      std::printf("PARITY B=%ld %-16s differing %ld of %ld%s", (long)B, r.name, (long)r.bad, (long)r.n,
                  r.bad ? "" : "  (bit-identical)");
      if (r.bad && r.rel > 0) std::printf("  max relative difference %.2e", r.rel);
      std::printf("\n");
      if (r.bad && std::string(r.name).find("dgamma") == std::string::npos) all_equal = false;
    }

    // ---- timing (median of `iters` calls after 20 warm-up calls) ----
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_it = [&](auto&& body) {
      for (int i = 0; i < 20; ++i) body();
      CK(cudaDeviceSynchronize());
      std::vector<float> t;
      for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(e0)); body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
      }
      return median(t);
    };
    const float tf32 = time_it([&] { auto r = fwd32(); (void)r; });
    const float tb32 = time_it([&] { auto r = bwd32(); (void)r; });
    const float tf16 = time_it([&] { fwd16(); });
    const float tb16 = time_it([&] { auto r = bwd16(); (void)r; });
    std::printf("TIME   B=%ld fp32 path (with its copies): fwd %.4f ms, bwd %.4f ms, total %.4f | bf16 kernel: fwd %.4f ms, "
                "bwd %.4f ms, total %.4f | bf16 / fp32 = %.3f\n", (long)B, tf32, tb32, tf32 + tb32, tf16, tb16, tf16 + tb16,
                (tf16 + tb16) / (tf32 + tb32));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  }
  std::printf("%s\n", all_equal ? "RESULT: outputs, rstd and gradients BIT-IDENTICAL (dgamma: see lines above)"
                                 : "RESULT: DIFFERENCES in outputs / rstd / gradients (see lines above)");
  CK(cudaGetLastError());
  return all_equal ? 0 : 3;
}
