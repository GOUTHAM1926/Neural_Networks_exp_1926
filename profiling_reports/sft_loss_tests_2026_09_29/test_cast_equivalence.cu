// =============================================================================
// test_cast_equivalence.cu
//
// QUESTION: does the fp32 cast in front of the loss (BLU_BF16_LOGITS=0) buy any
// precision over feeding the bf16 logits straight in (BLU_BF16_LOGITS=1)?
//
//   PATH A (cast ON,  arm 2): bf16 logits -> as_type(fp32) -> CE fwd+bwd at fp32
//                             -> fp32 grad -> as_type(bf16)      (ToDtypeBackward)
//   PATH B (cast OFF, arm 3): bf16 logits -> CE fwd+bwd at bf16 -> bf16 grad
//
// Both paths call the library's own code: convert_type_cuda_generic (exactly what
// Tensor::as_type calls, per the arm-2 stack trace) and the production stats
// kernels (..._impl_vec_save_stats + sparse_ce_backward_cuda_vec_with_stats).
//
// PREDICTIONS -- written BEFORE the first run, so the test cannot be fitted to it:
//   C0  widening bf16->fp32 is exact (fp32 bits == bf16 bits << 16)   : 0 mismatches
//   C5  CONTROL: path B run twice is bit-identical (kernel determinism) : 0 diffs
//   C1  saved_max A vs B   (max is order-independent and exact)         : 0 mismatches
//   C2  saved_sum A vs B   (VEC=4 vs VEC=8 changes fp32 summation order) : tiny, <= ~1e-6 rel
//   C3  loss A vs B                                                     : tiny, <= ~1e-6 rel
//   C4  grad reaching the matmul, A vs B : mostly bit-identical, rest 1 ulp, NONE > 1 ulp
// If C5 fails, every other A-vs-B difference is uninterpretable.
//
// NOTE: on sm_86 convert_type_cuda uses the generic kernel (__float2bfloat16);
// on sm_90 it uses vec8_convert_to_bf16 (__float22bfloat162_rn). Both are RNE,
// but this binary only exercises the one for the GPU it runs on.
//
// Build + run (from Tensor-Implementations/):
//   make run-cuda FILE=Tests/kernel_tests/test_cast_equivalence.cu
// =============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/helpers/ConversionKernels.cuh"
#include "ops/helpers/LossKernels.h"

using OwnTensor::bfloat16_t;
using OwnTensor::Dtype;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                   __FILE__, __LINE__);                                        \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

template <typename T> struct DBuf {
  T *p = nullptr;
  size_t n = 0;
  explicit DBuf(size_t count) : n(count) {
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
  }
  ~DBuf() { if (p) cudaFree(p); }
  DBuf(const DBuf &) = delete;
  DBuf &operator=(const DBuf &) = delete;
  void up(const std::vector<T> &h) {
    CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> down() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  // non-zero poison, so "kernel wrote 0" cannot be confused with "never written"
  void poison(float v) { up(std::vector<T>(n, static_cast<T>(v))); }
};

static uint16_t bits16(bfloat16_t b) { uint16_t u; std::memcpy(&u, &b, 2); return u; }
static uint32_t bits32(float f)      { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float    bf2f(uint16_t b)     { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static double   rel(double a, double b) { double d = std::fabs(a - b), m = std::max(std::fabs(a), std::fabs(b)); return m > 0 ? d / m : 0.0; }

struct Result { std::vector<float> loss, mx, sm; std::vector<bfloat16_t> grad; };

struct Case { const char *name; int64_t B, V; float sigma, peak; };

int main() {
  const Case cases[] = {
      // step-0-like: near-uniform logits, loss ~ ln(V) = 10.83 at V=50304
      {"init-like   (sigma 0.5)",           2048, 50304, 0.5f, 0.0f},
      // later-training-like: wide logits and a confident target
      {"trained-like (sigma 3, target +8)", 2048, 50304, 3.0f, 8.0f},
  };
  int total_bad = 0;

  for (const Case &c : cases) {
    const int64_t B = c.B, V = c.V, N = B * V;
    std::printf("\n==================== %s  B=%lld V=%lld ====================\n",
                c.name, (long long)B, (long long)V);

    // ---- identical inputs for both paths: bf16 logits = the matmul's output dtype
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, c.sigma);
    std::uniform_int_distribution<int> ud(0, (int)V - 1);
    std::vector<uint16_t> h_t(B);
    std::vector<bfloat16_t> h_z(N);
    for (int64_t r = 0; r < B; ++r) h_t[r] = (uint16_t)ud(rng);
    for (int64_t i = 0; i < N; ++i) {
      float z = nd(rng);
      if (c.peak != 0.f && (i % V) == h_t[i / V]) z += c.peak;
      h_z[i] = bfloat16_t(z);          // |z| << 65536, so the Types.h overflow bug is not hit
    }

    DBuf<bfloat16_t> d_z(N);  d_z.up(h_z);
    DBuf<uint16_t>   d_t(B);  d_t.up(h_t);
    DBuf<float> d_go(1);  d_go.up({1.0f / 64.0f});        // = 1/GRAD_ACCUM at B=4
    DBuf<float> d_den(1); d_den.up({(float)B});

    // ---------------- PATH A: cast ON ----------------
    DBuf<float> d_z32(N);
    d_z32.poison(7.f);
    OwnTensor::convert_type_cuda_generic(d_z.p, Dtype::Bfloat16, d_z32.p, Dtype::Float32, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run_A = [&]() {
      Result R;
      DBuf<float> L(1), MX(B), SM(B), G32(N);
      DBuf<bfloat16_t> G16(N);
      G32.poison(7.f); G16.poison(7.f); MX.poison(-999.f); SM.poison(-999.f);
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<float, uint16_t>(
          d_z32.p, d_t.p, L.p, MX.p, SM.p, B, V, 0, -1);
      CUDA_CHECK(cudaDeviceSynchronize());
      OwnTensor::cuda::sparse_ce_backward_cuda_vec_with_stats<float, uint16_t>(
          d_z32.p, d_t.p, G32.p, MX.p, SM.p, B, V, d_go.p, d_den.p, 0, -1);
      CUDA_CHECK(cudaDeviceSynchronize());
      // ToDtypeBackward: grads[0].as_type(bf16)
      OwnTensor::convert_type_cuda_generic(G32.p, Dtype::Float32, G16.p, Dtype::Bfloat16, N, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      R.loss = L.down(); R.mx = MX.down(); R.sm = SM.down(); R.grad = G16.down();
      return R;
    };

    // ---------------- PATH B: cast OFF ----------------
    auto run_B = [&]() {
      Result R;
      DBuf<float> L(1), MX(B), SM(B);
      DBuf<bfloat16_t> G16(N);
      G16.poison(7.f); MX.poison(-999.f); SM.poison(-999.f);
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<bfloat16_t, uint16_t>(
          d_z.p, d_t.p, L.p, MX.p, SM.p, B, V, 0, -1);
      CUDA_CHECK(cudaDeviceSynchronize());
      OwnTensor::cuda::sparse_ce_backward_cuda_vec_with_stats<bfloat16_t, uint16_t>(
          d_z.p, d_t.p, G16.p, MX.p, SM.p, B, V, d_go.p, d_den.p, 0, -1);
      CUDA_CHECK(cudaDeviceSynchronize());
      R.loss = L.down(); R.mx = MX.down(); R.sm = SM.down(); R.grad = G16.down();
      return R;
    };

    Result A = run_A(), Bq = run_B(), B2 = run_B();

    // ---- C0: widening is exact zero-padding
    {
      std::vector<float> z32 = d_z32.down();
      int64_t bad = 0;
      for (int64_t i = 0; i < N; ++i)
        if (bits32(z32[i]) != ((uint32_t)bits16(h_z[i]) << 16)) ++bad;
      std::printf("C0 widen bf16->fp32 exact          : %lld / %lld mismatches   (predict 0)\n",
                  (long long)bad, (long long)N);
      total_bad += (bad != 0);
    }

    // ---- C5: CONTROL, path B vs itself
    {
      int64_t d = 0;
      for (int64_t i = 0; i < N; ++i) d += bits16(Bq.grad[i]) != bits16(B2.grad[i]);
      for (int64_t r = 0; r < B; ++r) d += (bits32(Bq.mx[r]) != bits32(B2.mx[r])) + (bits32(Bq.sm[r]) != bits32(B2.sm[r]));
      d += bits32(Bq.loss[0]) != bits32(B2.loss[0]);
      std::printf("C5 CONTROL B vs B (determinism)    : %lld differing values        (predict 0)\n", (long long)d);
      total_bad += (d != 0);
    }

    // ---- anti-vacuity: every grad written, signs sane, not all zero
    {
      const uint16_t poison = bits16(bfloat16_t(7.f));
      int64_t unwrittenA = 0, unwrittenB = 0, zeroB = 0, badsign = 0;
      for (int64_t i = 0; i < N; ++i) {
        unwrittenA += bits16(A.grad[i]) == poison;
        unwrittenB += bits16(Bq.grad[i]) == poison;
        float g = bf2f(bits16(Bq.grad[i]));
        zeroB += (g == 0.f);
        bool is_t = (i % V) == h_t[i / V];
        if (is_t ? !(g < 0.f) : (g < 0.f)) ++badsign;   // (p - 1) < 0 on target, p >= 0 off it
      }
      std::printf("   anti-vacuity: unwritten A=%lld B=%lld | zero grads B=%lld | wrong-sign B=%lld\n",
                  (long long)unwrittenA, (long long)unwrittenB, (long long)zeroB, (long long)badsign);
      total_bad += (unwrittenA || unwrittenB || badsign);
    }

    // ---- C1: saved_max
    {
      int64_t bad = 0;
      for (int64_t r = 0; r < B; ++r) bad += bits32(A.mx[r]) != bits32(Bq.mx[r]);
      std::printf("C1 saved_max A vs B                : %lld / %lld mismatches     (predict 0)\n",
                  (long long)bad, (long long)B);
      total_bad += (bad != 0);
    }

    // ---- C2: saved_sum
    {
      int64_t same = 0; double worst = 0;
      for (int64_t r = 0; r < B; ++r) {
        same += bits32(A.sm[r]) == bits32(Bq.sm[r]);
        worst = std::max(worst, rel(A.sm[r], Bq.sm[r]));
      }
      std::printf("C2 saved_sum A vs B                : %lld / %lld bit-identical, worst rel %.3e  (predict <= ~1e-6)\n",
                  (long long)same, (long long)B, worst);
      total_bad += (worst > 1e-5);
    }

    // ---- C3: loss
    {
      double r3 = rel(A.loss[0], Bq.loss[0]);
      std::printf("C3 loss  A=%.9f  B=%.9f  rel %.3e  (predict <= ~1e-6)\n", A.loss[0], Bq.loss[0], r3);
      total_bad += (r3 > 1e-5);
    }

    // ---- C4: the gradient that reaches the lm_head matmul backward
    {
      int64_t same = 0, ulp1 = 0, more = 0; double maxabs = 0;
      for (int64_t i = 0; i < N; ++i) {
        uint16_t a = bits16(A.grad[i]), b = bits16(Bq.grad[i]);
        if (a == b) { ++same; continue; }
        maxabs = std::max(maxabs, (double)std::fabs(bf2f(a) - bf2f(b)));
        bool same_sign = ((a ^ b) & 0x8000u) == 0;
        if (same_sign && std::abs((int)a - (int)b) == 1) ++ulp1; else ++more;
      }
      std::printf("C4 grad A vs B (bf16, into matmul) : identical %lld (%.4f%%) | 1-ulp %lld | >1-ulp %lld | max|diff| %.3e\n",
                  (long long)same, 100.0 * same / N, (long long)ulp1, (long long)more, maxabs);
      std::printf("   (predict: mostly identical, rest 1 ulp, NONE > 1 ulp)\n");
      total_bad += (more != 0);
    }
  }

  std::printf("\n==================== SUMMARY ====================\n");
  std::printf("predictions violated: %d\n", total_bad);
  return total_bad ? 1 : 0;
}
