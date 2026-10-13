// sanitize_vj2.cu -- one small causal GQA forward + backward through the library API, made to run under
// compute-sanitizer (memcheck / synccheck / racecheck). Added 2026-10-13 for the Vj2 backward
// (GQA_bwd_vj2_sm90.cu): its new synchronisation (dQ writer warp, named barriers, 2-stage mbarrier loads) had
// never been run under a checker, and the full-size harness cases are far too slow under one.
// Shape: B=1, 6 query heads / 2 KV heads (GQA 3:1, as our 18/6 model), T=256, head_dim 128. T=256 keeps the
// forward on v62 (T % 256 == 0) and puts the backward on Vj2 when USE_VJ2=1 (T % 128 == 0); without USE_VJ2 the
// same program runs Vj1d, for comparison.
// Prints whether every output is finite and the sum of |dQ|, |dK|, |dV|. Exit code 0 = ran, 1 = non-finite output.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/helpers/AttentionKernels.h"
#include "dtype/Types.h"

using OwnTensor::bfloat16_t;
using bf16 = __nv_bfloat16;
namespace otc = OwnTensor::cuda;

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t e_ = (call);                                                                   \
    if (e_ != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

static bf16* upload(const std::vector<bf16>& h) {
  bf16* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(bf16)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  return d;
}
static double abs_sum(const bf16* d, size_t n, bool& finite) {
  std::vector<bf16> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(bf16), cudaMemcpyDeviceToHost));
  double s = 0;
  for (const bf16& x : h) {
    const float f = __bfloat162float(x);
    if (!std::isfinite(f)) finite = false;
    s += std::fabs(f);
  }
  return s;
}

int main() {
  const int B = 1, Hq = 6, Hkv = 2, T = 256, hd = 128;
  const size_t nq = (size_t)B * Hq * T * hd, nkv = (size_t)B * Hkv * T * hd;
  std::mt19937 gen(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto rnd = [&](size_t n) { std::vector<bf16> v(n); for (auto& x : v) x = __float2bfloat16(nd(gen)); return v; };
  bf16 *q = upload(rnd(nq)), *k = upload(rnd(nkv)), *v = upload(rnd(nkv)), *dO = upload(rnd(nq));
  bf16 *O = nullptr, *dQ = nullptr, *dK = nullptr, *dV = nullptr;
  float* lse = nullptr;
  CUDA_CHECK(cudaMalloc(&O, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dQ, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dK, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dV, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&lse, (size_t)B * Hq * T * sizeof(float)));
  auto cb = [](const bf16* x) { return reinterpret_cast<const bfloat16_t*>(x); };
  auto mb = [](bf16* x) { return reinterpret_cast<bfloat16_t*>(x); };
  otc::gqa_flash_attn_forward_qkv(cb(q), cb(k), cb(v), mb(O), lse, B, Hq, Hkv, T, hd, /*is_causal=*/true);
  otc::gqa_flash_attn_backward_qkv(cb(q), cb(k), cb(v), cb(O), cb(dO), lse, mb(dQ), mb(dK), mb(dV), B, Hq, Hkv, T,
                                   hd, /*is_causal=*/true);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  bool finite = true;
  const double sq = abs_sum(dQ, nq, finite), sk = abs_sum(dK, nkv, finite), sv = abs_sum(dV, nkv, finite);
  const char* mode = std::getenv("USE_VJ2");
  std::printf("sanitize_vj2: backward %s | B=%d %d/%d T=%d | sum|dQ| %.6f sum|dK| %.6f sum|dV| %.6f | outputs %s\n",
              (mode && mode[0] == '1') ? "Vj2 (USE_VJ2=1)" : "Vj1d (default)", B, Hq, Hkv, T, sq, sk, sv,
              finite ? "all finite" : "NON-FINITE");
  return finite ? 0 : 1;
}
