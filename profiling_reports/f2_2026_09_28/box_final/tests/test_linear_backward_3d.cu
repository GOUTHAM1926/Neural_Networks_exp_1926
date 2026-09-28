// =============================================================================
// test_linear_backward_3d.cu
//
// LinearBackward.cpp "Case 2" (input [B,T,C], weight [C,F]) declared a local
// `Tensor grad_weight` that shadowed the outer one, so the outer stayed empty and
// autograd::linear gave the weight NO gradient. The trainer never calls
// autograd::linear (nn::Linear uses autograd::matmul / addmm), which is why
// training was unaffected.
//
// Both LinearBackward and MatmulBackward call cuda_matmul_backward on the same
// flattened shapes, so after the fix their gradients must be bit-identical.
//
// PREDICTIONS -- written before the first run:
//   L1  3-D input, no bias: weight has a gradient.
//       BEFORE the fix this is expected to FAIL (the bug); AFTER it must pass.
//   L2  3-D input: weight grad from autograd::linear == from autograd::matmul, bit for bit
//       (fp32 and bf16 weights).
//   L3  CONTROL: input grad linear == matmul, bit for bit (never affected by the bug;
//       passes both BEFORE and AFTER -- if it fails, the comparison itself is broken).
//   L4  2-D input (Case 1, never affected): weight and input grads linear == matmul.
//   L5  3-D input with bias: weight grad == matmul path bit for bit, and the bias grad
//       matches column sums of the upstream grad (host double reference; tolerance
//       1e-5 relative for fp32, 1e-2 for bf16 because the bias grad is produced in bf16).
//
// Build + run (from Tensor-Implementations/):
//   make run-cuda FILE=Tests/kernel_tests/test_linear_backward_3d.cu
// =============================================================================

#include <cuda_runtime.h>
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
static void verdict(const char* id, bool ok, const std::string& d) {
  std::printf("%s %s  %s\n", ok ? "PASS" : "FAIL", id, d.c_str());
  if (!ok) g_fail++;
}
static const DeviceIndex kCuda(Device::CUDA, 0);

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f32_to_bf16_rne(float f) {
  uint32_t u = f2u(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
static float bf16_to_f32(uint16_t b) { return u2f(static_cast<uint32_t>(b) << 16); }

// Host data stored as the exact values the device will see.
struct HostT {
  std::vector<float> v;        // exact values
  std::vector<uint8_t> raw;    // bytes to upload in the tensor's dtype
};
static HostT host_rand(size_t n, bool bf16, std::mt19937& g) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  HostT h; h.v.resize(n); h.raw.resize(n * (bf16 ? 2 : 4));
  for (size_t i = 0; i < n; i++) {
    float f = nd(g);
    if (bf16) {
      uint16_t b = f32_to_bf16_rne(f); f = bf16_to_f32(b);
      std::memcpy(&h.raw[2 * i], &b, 2);
    } else {
      std::memcpy(&h.raw[4 * i], &f, 4);
    }
    h.v[i] = f;
  }
  return h;
}
static Tensor dev_tensor(const Shape& s, Dtype d, const HostT& h, bool req_grad) {
  Tensor t(s, d, kCuda, req_grad);
  CUDA_CHECK(cudaMemcpy(t.data(), h.raw.data(), h.raw.size(), cudaMemcpyHostToDevice));
  return t;
}
static std::vector<uint32_t> grad_bits(const Tensor& t, size_t n) {
  Tensor g = t.grad_view();
  if (g.dtype() != Dtype::Float32) { std::printf("  (grad dtype is not fp32)\n"); return {}; }
  std::vector<uint32_t> h(n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h.data(), g.data(), n * 4, cudaMemcpyDeviceToHost));
  return h;
}
static size_t diff(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  if (a.size() != b.size() || a.empty()) return (size_t)-1;
  size_t d = 0;
  for (size_t i = 0; i < a.size(); i++) d += (a[i] != b[i]);
  return d;
}

// One comparison: autograd::linear vs autograd::matmul on identical copies.
static void run_case(bool bf16, bool three_d, bool with_bias) {
  const Dtype dt = bf16 ? Dtype::Bfloat16 : Dtype::Float32;
  const int64_t Bn = 2, T = 32, C = 64, F = 96;
  const int64_t rows = three_d ? Bn * T : 48;
  std::mt19937 g(1000 + (bf16 ? 1 : 0) + (three_d ? 10 : 0) + (with_bias ? 100 : 0));
  const Shape xs = three_d ? Shape{{Bn, T, C}} : Shape{{rows, C}};
  const Shape gs = three_d ? Shape{{Bn, T, F}} : Shape{{rows, F}};
  const HostT hx = host_rand(rows * C, bf16, g), hw = host_rand(C * F, bf16, g),
              hg = host_rand(rows * F, bf16, g), hb = host_rand(F, bf16, g);

  // linear path
  Tensor x1 = dev_tensor(xs, dt, hx, true), w1 = dev_tensor(Shape{{C, F}}, dt, hw, true);
  Tensor b1 = with_bias ? dev_tensor(Shape{{F}}, dt, hb, true) : Tensor();
  Tensor g1 = dev_tensor(gs, dt, hg, false);
  Tensor y1 = OwnTensor::autograd::linear(x1, w1, b1);
  y1.backward(&g1);
  CUDA_CHECK(cudaDeviceSynchronize());

  // matmul path (reference): no bias -- the weight/input grads do not depend on it
  Tensor x2 = dev_tensor(xs, dt, hx, true), w2 = dev_tensor(Shape{{C, F}}, dt, hw, true);
  Tensor g2 = dev_tensor(gs, dt, hg, false);
  Tensor y2 = OwnTensor::autograd::matmul(x2, w2);
  y2.backward(&g2);
  CUDA_CHECK(cudaDeviceSynchronize());

  char tag[64];
  std::snprintf(tag, sizeof tag, "%s %s%s", bf16 ? "bf16" : "fp32", three_d ? "3-D" : "2-D",
                with_bias ? " +bias" : "");
  const std::string T0 = tag;

  const bool has_w = w1.has_grad();
  if (three_d && !with_bias)
    verdict("L1", has_w, T0 + (has_w ? "  weight has a gradient" : "  weight has NO gradient (the bug)"));

  const auto gw1 = has_w ? grad_bits(w1, C * F) : std::vector<uint32_t>{};
  const auto gw2 = grad_bits(w2, C * F);
  const auto gx1 = grad_bits(x1, rows * C), gx2 = grad_bits(x2, rows * C);
  const size_t dw = has_w ? diff(gw1, gw2) : (size_t)-1, dx = diff(gx1, gx2);
  char buf[160];
  std::snprintf(buf, sizeof buf, "  weight-grad mismatches linear vs matmul: %s",
                has_w ? std::to_string(dw).c_str() : "n/a (no grad)");
  const char* wid = three_d ? (with_bias ? "L5" : "L2") : "L4";
  verdict(wid, has_w && dw == 0, T0 + buf);
  std::snprintf(buf, sizeof buf, "  input-grad mismatches linear vs matmul: %zu", dx);
  verdict(three_d ? "L3" : "L4", dx == 0, T0 + buf);

  if (with_bias) {
    bool ok = b1.has_grad();
    double worst = 0.0;
    if (ok) {
      const auto gb = grad_bits(b1, F);
      ok = !gb.empty();
      for (int64_t f = 0; ok && f < F; f++) {
        double ref = 0.0;
        for (int64_t r = 0; r < rows; r++) ref += (double)hg.v[r * F + f];
        const double got = (double)u2f(gb[f]);
        const double rel = std::fabs(got - ref) / std::max(1.0, std::fabs(ref));
        worst = std::max(worst, rel);
      }
      ok = ok && worst <= (bf16 ? 1e-2 : 1e-5);
    }
    std::snprintf(buf, sizeof buf, "  bias grad vs column sums: worst rel err %.3g", worst);
    verdict("L5", ok, T0 + (b1.has_grad() ? buf : "  bias has NO gradient"));
  }
}

int main() {
  int dev = 0; cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDevice(&dev)); CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::printf("device %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);
  for (bool bf16 : {false, true}) {
    run_case(bf16, /*three_d=*/true,  /*bias=*/false);
    run_case(bf16, /*three_d=*/false, /*bias=*/false);
    run_case(bf16, /*three_d=*/true,  /*bias=*/true);
  }
  std::printf("\nRESULT %s  (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
