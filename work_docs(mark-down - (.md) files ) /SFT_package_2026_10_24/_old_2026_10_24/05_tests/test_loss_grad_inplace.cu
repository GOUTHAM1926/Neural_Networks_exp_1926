// =============================================================================
// test_loss_grad_inplace.cu  (2026-10-17)
//
// The trainer's lm_head + loss, through autograd, at the SFT micro-batch:
//   logits = matmul(X [8, 4096, 2304] bf16, W [2304, 50304] bf16)   (requires grad: X and W)
//   loss   = sparse_cross_entropy_loss(logits, targets, 65535, &denom [, grad_into_logits])
//   loss.backward(&seed)
// with the REAL stage-0 targets of one micro-batch (default 458, the median: 7,577 trained rows).
//
// One mode per process (the allocator's peaks only grow), the same sequence in every mode:
//   out : the gradient of the logits is a new tensor (grad_into_logits = false, the default)
//   in  : the gradient is written over the logits (grad_into_logits = true)
// Built with -DLEGACY_LOSS_API it uses the OLD 4-argument loss op, to run against a library
// from before 2026-10-17 item 3 (where the CUDA backward also allocated the gradient twice).
//
// Prints one RESULT line: the loss, checksums of W.grad and X.grad (64-bit, over the raw
// bytes), the backward time, and the allocator's active / allocated / reserved peaks.
// Expected (work doc 45.37): the same loss and the same checksums in every mode and library;
// 'in' one logits-sized tensor (8 x 4096 x 50304 x 2 B = 3,144 MiB) below 'out' wherever the
// backward of the loss sets the peak; the old library up to one more logits-sized tensor.
// The forward + backward runs twice (rep 1, rep 2); both reps must give the same checksums.
//
// usage: test_loss_grad_inplace <labels.bin> <out|in> [micro-batch, default 458]
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
#include "device/CachingCudaAllocator.h"
#include "autograd/operations/MatrixOps.h"
#ifdef LEGACY_LOSS_API
namespace OwnTensor { namespace autograd {
Tensor sparse_cross_entropy_loss(const Tensor &logits, const Tensor &targets, int64_t ignore_index,
                                 const Tensor *denom);
}}
#else
#include "autograd/operations/LossOps.h"
#endif

using namespace OwnTensor;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int64_t kB = 8, kT = 4096, kD = 2304, kV = 50304, kN = kB * kT;
static const DeviceIndex kCuda(Device::CUDA, 0);

static uint16_t bf16_rne(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
static Tensor upload(const Shape &s, Dtype d, const void *p, size_t bytes, bool req_grad) {
  Tensor t(s, d, kCuda, req_grad);
  CK(cudaMemcpy(t.data(), p, bytes, cudaMemcpyHostToDevice));
  return t;
}
static uint64_t checksum_dev(const void *p, size_t bytes) {   // sum of 32-bit words x (index | 1), mod 2^64
  std::vector<uint32_t> h(bytes / 4);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
  uint64_t acc = 0;
  for (size_t i = 0; i < h.size(); ++i) acc += (uint64_t)h[i] * (uint64_t)(i | 1);
  return acc;
}
static size_t esize(Dtype d) { return d == Dtype::Float32 ? 4 : 2; }

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <labels.bin> <out|in> [micro-batch]\n", argv[0]); return 1; }
  const std::string mode = argv[2];
  const int64_t mb = argc > 3 ? std::atoll(argv[3]) : 458;
#ifdef LEGACY_LOSS_API
  if (mode != "out") { std::fprintf(stderr, "the legacy build has only mode 'out'\n"); return 1; }
#endif
  if (mode != "out" && mode != "in") { std::fprintf(stderr, "mode must be out or in\n"); return 1; }

  // ---- targets: one real micro-batch of stage-0 labels (65535 = ignore) ----
  std::vector<uint16_t> tg(kN);
  {
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    if (std::fseek(f, (long)(mb * kN * 2), SEEK_SET) != 0 || std::fread(tg.data(), 2, kN, f) != (size_t)kN) {
      std::fprintf(stderr, "cannot read micro-batch %ld\n", (long)mb); return 1;
    }
    std::fclose(f);
  }
  int64_t trained = 0;
  for (uint16_t t : tg) trained += (t != 65535);
  const float denom = (float)std::max<int64_t>(trained, 1), one = 1.0f;

  // ---- X ~ N(0,1), W ~ N(0, 1/D) so the logits are ~ N(0,1); fixed seeds ----
  std::vector<uint16_t> hx(kN * kD), hw(kD * kV);
  { std::mt19937 g(11); std::normal_distribution<float> n(0.f, 1.f); for (auto &v : hx) v = bf16_rne(n(g)); }
  { std::mt19937 g(12); std::normal_distribution<float> n(0.f, 1.f / std::sqrt((float)kD)); for (auto &v : hw) v = bf16_rne(n(g)); }

  Tensor X = upload(Shape{{kB, kT, kD}}, Dtype::Bfloat16, hx.data(), hx.size() * 2, true);
  Tensor W = upload(Shape{{kD, kV}}, Dtype::Bfloat16, hw.data(), hw.size() * 2, true);
  Tensor targets = upload(Shape{{kB, kT}}, Dtype::UInt16, tg.data(), tg.size() * 2, false);
  Tensor den = upload(Shape{{1}}, Dtype::Float32, &denom, 4, false);
  Tensor seed = upload(Shape{{1}}, Dtype::Float32, &one, 4, false);

  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | mode %s%s | micro-batch %ld, %ld of %ld rows trained\n", prop.name, mode.c_str(),
#ifdef LEGACY_LOSS_API
              " (legacy 4-argument loss op)",
#else
              "",
#endif
              (long)mb, (long)trained, (long)kN);

  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  uint64_t cs_w[2] = {0, 0}, cs_x[2] = {0, 0}; float loss_v[2] = {0, 0}, bwd_ms[2] = {0, 0};
  for (int rep = 0; rep < 2; ++rep) {
    if (rep == 1) { X.zero_grad(); W.zero_grad(); }
    Tensor logits = autograd::matmul(X, W);                      // [8, 4096, 50304] bf16
#ifdef LEGACY_LOSS_API
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets, 65535, &den);
#else
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets, 65535, &den, mode == "in");
#endif
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    loss.backward(&seed);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&bwd_ms[rep], a, b));
    CK(cudaMemcpy(&loss_v[rep], loss.data(), 4, cudaMemcpyDeviceToHost));
    Tensor gw = W.grad_view(), gx = X.grad_view();
    cs_w[rep] = checksum_dev(gw.data(), (size_t)gw.numel() * esize(gw.dtype()));
    cs_x[rep] = checksum_dev(gx.data(), (size_t)gx.numel() * esize(gx.dtype()));
    std::printf("rep %d: loss %.6f | W.grad %s checksum %016llx | X.grad %s checksum %016llx | backward %.3f ms\n", rep + 1,
                loss_v[rep], gw.dtype() == Dtype::Float32 ? "fp32" : "bf16", (unsigned long long)cs_w[rep],
                gx.dtype() == Dtype::Float32 ? "fp32" : "bf16", (unsigned long long)cs_x[rep], bwd_ms[rep]);
  }
  const auto st = CachingCUDAAllocator::instance().get_stats();
  const bool same = cs_w[0] == cs_w[1] && cs_x[0] == cs_x[1] && loss_v[0] == loss_v[1];
  std::printf("RESULT mode=%s legacy=%d loss=%.6f wgrad_cs=%016llx xgrad_cs=%016llx bwd_ms=%.3f active_peak_MiB=%.1f "
              "allocated_peak_MiB=%.1f reserved_peak_MiB=%.1f allocated_peak_B=%zu reps_identical=%d\n",
              mode.c_str(),
#ifdef LEGACY_LOSS_API
              1,
#else
              0,
#endif
              loss_v[0], (unsigned long long)cs_w[0], (unsigned long long)cs_x[0], bwd_ms[1], st.active_peak / 1048576.0,
              st.allocated_peak / 1048576.0, st.reserved_peak / 1048576.0, st.allocated_peak, same ? 1 : 0);
  return same ? 0 : 3;
}
