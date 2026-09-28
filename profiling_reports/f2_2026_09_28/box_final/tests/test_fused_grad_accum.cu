// =============================================================================
// test_fused_grad_accum.cu
//
// OLD : acc += upd.as_type(Float32)                 (AutogradMeta.cpp:173 before)
// NEW : cuda_accumulate_lowp_into_fp32(acc, upd)    (one fused kernel)
//
// PREDICTIONS -- written before the first run:
//   P1  NEW == OLD bit for bit after 4 accumulation steps, for bf16 and fp16
//       updates and every size (1 .. 32M+5, incl. sizes that are not a multiple
//       of 8 and one large enough to take the sm90 flat add in OLD), with
//       +-0, subnormals, +-inf, NaN, fp32/bf16/fp16 max and overflow planted.
//   P2  the misaligned path (views offset by one element -> scalar kernel) gives
//       the same bits as the aligned path (vector kernel) on the same data.
//   P3  NEW returns false and leaves acc untouched for: fp32 update, bf16 acc,
//       shape mismatch, non-contiguous update, CPU tensors.
//   P4  CONTROL: NEW with one of the 4 updates skipped differs from OLD (gate for n > 32).
//       If P4 "passes" as equal, P1 proves nothing. For n <= 32 it is printed as INFO: with so few
//       elements a skipped update can be absorbed by fp32 rounding on both paths (first run, bf16 n=1).
//   P5  WIRING: two backward passes of autograd::matmul(x, W) on a bf16 W with
//       no zero_grad in between. Pass 2 goes through AutogradMeta.cpp:173, so
//       the fused call counter must rise -- unless BLUTRAIN_NO_FUSED_GRAD_ACC=1,
//       then it must stay flat. The grad is fp32 and equals exactly 2x the
//       one-pass grad in both cases (x + x is exact in fp32).
//   P6  speed (report, not a gate): NEW faster than OLD on large tensors.
//       Memory-bound estimate: 10 B vs 18 B per element -> up to ~1.8x.
//
// Build + run (from Tensor-Implementations/, after the lib is built):
//   make run-cuda FILE=Tests/kernel_tests/test_fused_grad_accum.cu
//   BLUTRAIN_NO_FUSED_GRAD_ACC=1 make run-cuda FILE=Tests/kernel_tests/test_fused_grad_accum.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "autograd/operations/MatrixOps.h"

using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                   __FILE__, __LINE__);                                        \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static int g_fail = 0;
static void verdict(const char* id, bool ok, const std::string& detail) {
  std::printf("%s %s  %s\n", ok ? "PASS" : "FAIL", id, detail.c_str());
  if (!ok) g_fail++;
}

static const DeviceIndex kCuda(Device::CUDA, 0);
static const DeviceIndex kCpu(Device::CPU, 0);

static Tensor make(const Shape& s, Dtype d, DeviceIndex dev = kCuda) {
  return Tensor::empty(s, TensorOptions().with_dtype(d).with_device(dev));
}
template <typename T> static void upload(Tensor& t, const std::vector<T>& h) {
  CUDA_CHECK(cudaMemcpy(t.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T> static std::vector<T> download(const Tensor& t, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h.data(), t.data(), n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// ---- host bit helpers ------------------------------------------------------
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = f2u(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return 0x7FC0;  // NaN
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
static uint16_t f32_to_fp16_rn(float f) {
  __half h = __float2half_rn(f);
  uint16_t b; std::memcpy(&b, &h, 2); return b;
}

// Random value spread over many exponents (so rounding and cancellation happen).
static float rnd(std::mt19937& g, int emin, int emax) {
  std::uniform_int_distribution<int> e(emin, emax);
  std::uniform_real_distribution<float> m(1.0f, 2.0f);
  std::bernoulli_distribution s(0.5);
  float v = std::ldexp(m(g), e(g));
  return s(g) ? -v : v;
}

// Special (acc, update) pairs planted at the start of every case.
struct Special { float acc; float upd; };
static std::vector<Special> specials(bool bf16) {
  const float inf = INFINITY, nan = NAN;
  const float lowp_max = bf16 ? u2f(0x7F7F0000u) : 65504.0f;
  const float sub = bf16 ? u2f(0x00010000u) : 5.9604645e-08f;   // smallest bf16/fp16 subnormal
  return {
    {0.0f, 0.0f}, {0.0f, -0.0f}, {-0.0f, 0.0f}, {-0.0f, -0.0f},
    {1.0f, sub}, {0.0f, sub}, {u2f(0x00000001u), 0.0f},           // fp32 subnormal acc
    {3.0e38f, bf16 ? 3.0e38f : 65504.0f},                          // overflow (bf16) / big add
    {1.0f, inf}, {1.0f, -inf}, {inf, -inf}, {1.0f, nan},
    {2.5f, -2.5f}, {lowp_max, lowp_max}, {-lowp_max, lowp_max}, {1e-30f, -1e-30f},
  };
}

// ---- P1 / P4 ---------------------------------------------------------------
static void case_equal(bool bf16, size_t n, uint32_t seed) {
  const Dtype ud = bf16 ? Dtype::Bfloat16 : Dtype::Float16;
  std::mt19937 g(seed);
  const int emin = bf16 ? -30 : -14, emax = bf16 ? 30 : 14;

  std::vector<float> acc0(n);
  for (auto& v : acc0) v = rnd(g, emin, emax);
  std::vector<std::vector<uint16_t>> upd(4, std::vector<uint16_t>(n));
  for (int k = 0; k < 4; k++)
    for (size_t i = 0; i < n; i++) {
      const float f = rnd(g, emin, emax);
      upd[k][i] = bf16 ? f32_to_bf16_rne(f) : f32_to_fp16_rn(f);
    }
  const auto sp = specials(bf16);
  for (size_t i = 0; i < sp.size() && i < n; i++) {
    acc0[i] = sp[i].acc;
    upd[0][i] = bf16 ? f32_to_bf16_rne(sp[i].upd) : f32_to_fp16_rn(sp[i].upd);
  }

  Tensor A = make(Shape{{(int64_t)n}}, Dtype::Float32), B = make(Shape{{(int64_t)n}}, Dtype::Float32),
         C = make(Shape{{(int64_t)n}}, Dtype::Float32);
  upload(A, acc0); upload(B, acc0); upload(C, acc0);
  Tensor U = make(Shape{{(int64_t)n}}, ud);
  const cudaStream_t st = OwnTensor::cuda::getCurrentStream();

  bool all_true = true;
  for (int k = 0; k < 4; k++) {
    upload(U, upd[k]);
    A += U.as_type(Dtype::Float32);                                         // OLD
    all_true &= OwnTensor::cuda_accumulate_lowp_into_fp32(B, U, st);        // NEW
    if (k != 3) all_true &= OwnTensor::cuda_accumulate_lowp_into_fp32(C, U, st);  // NEW, skip one
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const auto a = download<uint32_t>(A, n), b = download<uint32_t>(B, n), c = download<uint32_t>(C, n);

  size_t mism = 0, changed = 0, ctrl_diff = 0, first = n;
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) { mism++; if (first == n) first = i; }
    if (b[i] != f2u(acc0[i])) changed++;
    if (a[i] != c[i]) ctrl_diff++;
  }
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s n=%zu mismatches=%zu changed=%zu/%zu", bf16 ? "bf16" : "fp16",
                n, mism, changed, n);
  std::string d = buf;
  if (mism) {
    std::snprintf(buf, sizeof buf, "  first@%zu old=0x%08x new=0x%08x", first, a[first], b[first]);
    d += buf;
  }
  // changed-ness sanity only for sizes past the planted specials (tiny n are mostly specials)
  const bool sane = (n <= 32) || (changed * 10 >= n * 9);
  verdict("P1", all_true && mism == 0 && sane, d + (sane ? "" : "  (too few elements changed)"));
  // P4 is a gate only above 32 elements. With very few elements every skipped update can legitimately
  // change nothing: fp32 keeps ~2^-24 relative, so an update 1e-13 x the running sum is absorbed on both
  // paths (seen at bf16 n=1: 55808 + -4.6e-9 == 55808). With many elements some skipped update is always
  // visible, so a zero there would mean the comparison cannot fail.
  std::snprintf(buf, sizeof buf, "%s n=%zu control_diffs=%zu (must be > 0)", bf16 ? "bf16" : "fp16", n,
                ctrl_diff);
  if (n > 32) verdict("P4", ctrl_diff > 0, buf);
  else        std::printf("INFO P4  %s  (not a gate for n <= 32)\n", buf);
}

// ---- P2 --------------------------------------------------------------------
static void case_misaligned(bool bf16, size_t n) {
  const Dtype ud = bf16 ? Dtype::Bfloat16 : Dtype::Float16;
  std::mt19937 g(1234 + (bf16 ? 1 : 0));
  std::vector<float> acc0(n + 1);
  std::vector<uint16_t> up(n + 1);
  for (auto& v : acc0) v = rnd(g, -20, 20);
  for (auto& v : up) { float f = rnd(g, -14, 14); v = bf16 ? f32_to_bf16_rne(f) : f32_to_fp16_rn(f); }

  // aligned reference: elements 1..n in their own buffers
  Tensor Ar = make(Shape{{(int64_t)n}}, Dtype::Float32), Ur = make(Shape{{(int64_t)n}}, ud);
  upload(Ar, std::vector<float>(acc0.begin() + 1, acc0.end()));
  upload(Ur, std::vector<uint16_t>(up.begin() + 1, up.end()));
  // misaligned: views starting at element 1 of an (n+1) buffer
  Tensor Abig = make(Shape{{(int64_t)n + 1}}, Dtype::Float32), Ubig = make(Shape{{(int64_t)n + 1}}, ud);
  upload(Abig, acc0); upload(Ubig, up);
  Tensor Am = Abig.slice_inplace(1, n), Um = Ubig.slice_inplace(1, n);

  const bool mis = (reinterpret_cast<uintptr_t>(Am.data()) % 16 != 0) &&
                   (reinterpret_cast<uintptr_t>(Um.data()) % 16 != 0);
  const cudaStream_t st = OwnTensor::cuda::getCurrentStream();
  bool ok1 = OwnTensor::cuda_accumulate_lowp_into_fp32(Ar, Ur, st);
  bool ok2 = OwnTensor::cuda_accumulate_lowp_into_fp32(Am, Um, st);
  const auto r = download<uint32_t>(Ar, n);
  const auto m = download<uint32_t>(Abig, n + 1);
  size_t mism = 0;
  for (size_t i = 0; i < n; i++) if (r[i] != m[i + 1]) mism++;
  const bool untouched0 = (m[0] == f2u(acc0[0]));   // element 0 is outside the view
  char buf[200];
  std::snprintf(buf, sizeof buf, "%s n=%zu views_misaligned=%d mismatches=%zu elem0_untouched=%d",
                bf16 ? "bf16" : "fp16", n, (int)mis, mism, (int)untouched0);
  verdict("P2", ok1 && ok2 && mis && mism == 0 && untouched0, buf);
}

// ---- P3 --------------------------------------------------------------------
static void case_fallbacks() {
  const cudaStream_t st = OwnTensor::cuda::getCurrentStream();
  auto check = [&](const char* what, Tensor acc, const Tensor& upd) {
    const size_t n = acc.numel();
    const size_t esz = Tensor::dtype_size(acc.dtype());
    std::vector<uint8_t> before(n * esz), after(n * esz);
    const bool cuda = acc.device().is_cuda();
    if (cuda) { CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(before.data(), acc.data(), n * esz, cudaMemcpyDeviceToHost)); }
    else std::memcpy(before.data(), acc.data(), n * esz);
    const bool r = OwnTensor::cuda_accumulate_lowp_into_fp32(acc, upd, st);
    if (cuda) { CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(after.data(), acc.data(), n * esz, cudaMemcpyDeviceToHost)); }
    else std::memcpy(after.data(), acc.data(), n * esz);
    verdict("P3", !r && before == after, std::string(what) + (r ? "  returned true" : "  returned false, acc untouched"));
  };
  auto filled = [](const Shape& s, Dtype d, DeviceIndex dev) {
    Tensor t = make(s, d, dev);
    const size_t bytes = t.numel() * Tensor::dtype_size(d);
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t)(i * 37 + 11);
    if (dev.is_cuda()) CUDA_CHECK(cudaMemcpy(t.data(), h.data(), bytes, cudaMemcpyHostToDevice));
    else std::memcpy(t.data(), h.data(), bytes);
    return t;
  };
  check("fp32 update", filled(Shape{{64}}, Dtype::Float32, kCuda), filled(Shape{{64}}, Dtype::Float32, kCuda));
  check("bf16 acc", filled(Shape{{64}}, Dtype::Bfloat16, kCuda), filled(Shape{{64}}, Dtype::Bfloat16, kCuda));
  check("shape mismatch [8,4] vs [4,8]", filled(Shape{{8, 4}}, Dtype::Float32, kCuda),
        filled(Shape{{4, 8}}, Dtype::Bfloat16, kCuda));
  {
    Tensor base = filled(Shape{{4, 8}}, Dtype::Bfloat16, kCuda);
    Tensor tr = base.t();   // [8,4] view, not contiguous
    if (tr.is_contiguous()) verdict("P3", false, "setup: t() returned a contiguous tensor");
    else check("non-contiguous update", filled(Shape{{8, 4}}, Dtype::Float32, kCuda), tr);
  }
  check("CPU tensors", filled(Shape{{64}}, Dtype::Float32, kCpu), filled(Shape{{64}}, Dtype::Bfloat16, kCpu));
}

// ---- P5 --------------------------------------------------------------------
static void case_wiring() {
  const char* env = std::getenv("BLUTRAIN_NO_FUSED_GRAD_ACC");
  const bool disabled = env && (std::strcmp(env, "1") == 0);
  const int64_t Bn = 2, T = 8, C = 64, F = 48;
  std::mt19937 g(77);
  auto bf = [&](size_t n, int emin, int emax) {
    std::vector<uint16_t> h(n);
    for (auto& v : h) v = f32_to_bf16_rne(rnd(g, emin, emax));
    return h;
  };
  const auto hw = bf(C * F, -6, 0), hx = bf(Bn * T * C, -6, 0), hg = bf(Bn * T * F, -6, 0);

  auto run = [&](int passes, uint64_t* delta_last) {
    Tensor W(Shape{{C, F}}, Dtype::Bfloat16, kCuda, /*requires_grad=*/true);
    Tensor X = make(Shape{{Bn, T, C}}, Dtype::Bfloat16), G = make(Shape{{Bn, T, F}}, Dtype::Bfloat16);
    upload(W, hw); upload(X, hx); upload(G, hg);
    for (int p = 0; p < passes; p++) {
      Tensor Y = OwnTensor::autograd::matmul(X, W);
      const uint64_t c0 = OwnTensor::fused_grad_accum_call_count();
      Y.backward(&G);
      CUDA_CHECK(cudaDeviceSynchronize());
      *delta_last = OwnTensor::fused_grad_accum_call_count() - c0;
    }
    Tensor gv = W.grad_view();
    return std::make_pair(gv.dtype(), download<uint32_t>(gv, (size_t)(C * F)));
  };
  uint64_t d1 = 0, d2 = 0;
  auto one = run(1, &d1);
  auto two = run(2, &d2);
  size_t bad = 0;
  for (size_t i = 0; i < one.second.size(); i++)
    if (two.second[i] != f2u(2.0f * u2f(one.second[i]))) bad++;
  const bool fp32 = (one.first == Dtype::Float32 && two.first == Dtype::Float32);
  const bool counter_ok = disabled ? (d2 == 0) : (d2 >= 1);
  char buf[240];
  std::snprintf(buf, sizeof buf,
                "fused_switch=%s pass1_delta=%llu pass2_delta=%llu grad_fp32=%d grad(2 passes)!=2*grad(1): %zu",
                disabled ? "OFF(BLUTRAIN_NO_FUSED_GRAD_ACC=1)" : "ON", (unsigned long long)d1,
                (unsigned long long)d2, (int)fp32, bad);
  verdict("P5", fp32 && counter_ok && d1 == 0 && bad == 0, buf);
}

// ---- P6 --------------------------------------------------------------------
static void case_speed(size_t n) {
  Tensor A = make(Shape{{(int64_t)n}}, Dtype::Float32), U = make(Shape{{(int64_t)n}}, Dtype::Bfloat16);
  CUDA_CHECK(cudaMemset(A.data(), 0, n * 4));
  CUDA_CHECK(cudaMemset(U.data(), 0x3C, n * 2));   // small positive bf16 values
  const cudaStream_t st = OwnTensor::cuda::getCurrentStream();
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  const int reps = 20, rounds = 12;
  auto time_old = [&] {
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(e0, st));
    for (int r = 0; r < reps; r++) A += U.as_type(Dtype::Float32);
    CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); return ms / reps;
  };
  auto time_new = [&] {
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(e0, st));
    for (int r = 0; r < reps; r++) OwnTensor::cuda_accumulate_lowp_into_fp32(A, U, st);
    CUDA_CHECK(cudaEventRecord(e1, st)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); return ms / reps;
  };
  time_old(); time_new();   // warm-up (allocator, occupancy caches)
  std::vector<float> o, w;
  for (int r = 0; r < rounds; r++) {          // A-B-B-A order
    if (r % 2 == 0) { o.push_back(time_old()); w.push_back(time_new()); }
    else            { w.push_back(time_new()); o.push_back(time_old()); }
  }
  std::sort(o.begin(), o.end()); std::sort(w.begin(), w.end());
  const float mo = 0.5f * (o[5] + o[6]), mw = 0.5f * (w[5] + w[6]);
  std::printf("P6   n=%-10zu old %.4f ms (%.0f GB/s at 18 B/elem)  new %.4f ms (%.0f GB/s at 10 B/elem)  speedup %.2fx\n",
              n, mo, 18.0 * n / (mo * 1e6), mw, 10.0 * n / (mw * 1e6), mo / mw);
  CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
}

int main() {
  int dev = 0; cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::printf("device %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);

  const size_t sizes[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 1023, 1024, 1025, 65539,
                          2304 * 2304, 2048 * 5632, 2304 * 6144, 33554437};
  for (bool bf16 : {true, false})
    for (size_t n : sizes) case_equal(bf16, n, (uint32_t)(n * 2654435761u + (bf16 ? 1 : 2)));
  for (bool bf16 : {true, false}) { case_misaligned(bf16, 1025); case_misaligned(bf16, 2048 * 5632 + 3); }
  case_fallbacks();
  case_wiring();
  for (size_t n : {(size_t)2304 * 2304, (size_t)2048 * 5632, (size_t)2304 * 6144, (size_t)33554437}) case_speed(n);

  std::printf("\nRESULT %s  (%d failed checks)  fused calls in this process: %llu\n",
              g_fail ? "FAIL" : "PASS", g_fail,
              (unsigned long long)OwnTensor::fused_grad_accum_call_count());
  return g_fail ? 1 : 0;
}
