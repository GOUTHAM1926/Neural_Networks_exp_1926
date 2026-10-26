// =============================================================================
// test_loss_pretrain_ab.cu  (2026-10-21)
//
// OLD vs NEW library A/B for the pretraining branch dist_side_gautam_lottn, loss part:
//   NEW = 16-byte loads in the loss forward and backward, the gradient allocated once and
//         written over the logits (LossKernels.cu, LossBackward.cpp);
//   OLD = BLUCCL as-is (Tensor-Implementations 02d0d18).
// ONE binary, run once per library: LD_LIBRARY_PATH picks the library. It calls only what both
// libraries have (the 2-argument loss op and the out-of-place stats backward entry point).
//
// Part 1, KERNEL: the two loss kernels alone, N = 8 x 4096 rows, V = 50304, uint16 targets, no
// ignored rows (pretraining), on fixed pseudo-random logits:
//   F  forward with stats   cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, uint16_t>
//   B  backward with stats  cuda::sparse_ce_backward_cuda_vec_with_stats<T, uint16_t>
//   F then B, --iters times after --warmup times, CUDA events around each call: median and min.
//   Then checksums of the last call's loss, saved max, saved sum and gradient.
// Part 2, AUTOGRAD: the trainer's lm_head + loss (train/production_zero/bluscript_zero_ckpt.cpp:
// 811-812 and 2759-2808):
//   logits = to_dtype(matmul(X [8, 4096, 2304] bf16, W [2304, 50304] bf16), f32 | bf16)
//   loss   = sparse_cross_entropy_loss(logits, targets);  to_dtype(loss, f32).backward(1/32)
//   3 reps (rep 1 warms up): loss bits, checksums of W.grad and X.grad, the loss forward time, the
//   backward time, and the caching allocator's peaks (one dtype per process: the peaks only grow).
// Targets: real tokens -- the next-token targets of N + 1 tokens of a uint16 shard from --offset.
//
// Expected: every checksum and loss bit pattern equal for OLD and NEW (same arithmetic in the same
// order); NEW faster in F (and B for bf16); NEW's allocated peak lower by up to two gradient-sized
// tensors (fp32: 2 x 6,288 MiB; bf16: 2 x 3,144 MiB) where the loss backward sets the peak.
//
// usage: test_loss_pretrain_ab <f32|bf16> <tokens.bin> [--offset TOKENS] [--warmup N] [--iters N]
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
#include "dtype/Types.h"
#include "ops/helpers/LossKernels.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/LossOps.h"

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
static uint64_t checksum_dev(const void *p, size_t bytes) {   // sum of 32-bit words x (index | 1), mod 2^64
  std::vector<uint32_t> h(bytes / 4);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
  uint64_t acc = 0;
  for (size_t i = 0; i < h.size(); ++i) acc += (uint64_t)h[i] * (uint64_t)(i | 1);
  return acc;
}
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// fixed pseudo-random logits in [-8, 8): a hash of the element index, the same on every run
__global__ void fill_logits(float *f, uint16_t *h, int64_t n) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ static_cast<uint32_t>(i >> 32) * 40503u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    const float v = (static_cast<float>(x >> 8) * (1.0f / 16777216.0f) - 0.5f) * 16.0f;
    if (f) f[i] = v;
    if (h) h[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
  }
}

template <typename T>
static void kernel_part(const char *dname, const uint16_t *d_tgt, int warmup, int iters) {
  const size_t nel = (size_t)kN * kV;
  T *d_logits = nullptr, *d_grad = nullptr;
  float *d_loss = nullptr, *d_max = nullptr, *d_sum = nullptr, *d_gout = nullptr, *d_den = nullptr;
  CK(cudaMalloc(&d_logits, nel * sizeof(T)));
  CK(cudaMalloc(&d_grad, nel * sizeof(T)));
  CK(cudaMalloc(&d_loss, 4)); CK(cudaMalloc(&d_max, kN * 4)); CK(cudaMalloc(&d_sum, kN * 4));
  CK(cudaMalloc(&d_gout, 4)); CK(cudaMalloc(&d_den, 4));
  if (sizeof(T) == 4) fill_logits<<<4096, 256>>>(reinterpret_cast<float *>(d_logits), nullptr, (int64_t)nel);
  else                fill_logits<<<4096, 256>>>(nullptr, reinterpret_cast<uint16_t *>(d_logits), (int64_t)nel);
  CK(cudaGetLastError());
  const float gout = 1.0f / 32.0f, den = (float)kN;
  CK(cudaMemcpy(d_gout, &gout, 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_den, &den, 4, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());

  auto F = [&] { OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, uint16_t>(
                     d_logits, d_tgt, d_loss, d_max, d_sum, kN, kV, (cudaStream_t)0); };
  auto B = [&] { OwnTensor::cuda::sparse_ce_backward_cuda_vec_with_stats<T, uint16_t>(
                     d_logits, d_tgt, d_grad, d_max, d_sum, kN, kV, d_gout, d_den, (cudaStream_t)0); };
  for (int i = 0; i < warmup; ++i) { F(); B(); }
  CK(cudaDeviceSynchronize());
  cudaEvent_t e[4]; for (auto &x : e) CK(cudaEventCreate(&x));
  std::vector<float> tf, tb;
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(e[0])); F(); CK(cudaEventRecord(e[1]));
    CK(cudaEventRecord(e[2])); B(); CK(cudaEventRecord(e[3]));
    CK(cudaEventSynchronize(e[3]));
    float a, b; CK(cudaEventElapsedTime(&a, e[0], e[1])); CK(cudaEventElapsedTime(&b, e[2], e[3]));
    tf.push_back(a); tb.push_back(b);
  }
  CK(cudaGetLastError());
  float loss = 0; CK(cudaMemcpy(&loss, d_loss, 4, cudaMemcpyDeviceToHost));
  const uint64_t cs_max = checksum_dev(d_max, kN * 4), cs_sum = checksum_dev(d_sum, kN * 4),
                 cs_grad = checksum_dev(d_grad, nel * sizeof(T));
  std::printf("KERNEL dtype=%s F_med_ms=%.4f F_min_ms=%.4f B_med_ms=%.4f B_min_ms=%.4f iters=%d loss=%.6f "
              "loss_bits=%08x max_cs=%016llx sum_cs=%016llx grad_cs=%016llx\n",
              dname, median(tf), *std::min_element(tf.begin(), tf.end()), median(tb),
              *std::min_element(tb.begin(), tb.end()), iters, loss, bits(loss), (unsigned long long)cs_max,
              (unsigned long long)cs_sum, (unsigned long long)cs_grad);
  for (auto &x : e) CK(cudaEventDestroy(x));
  CK(cudaFree(d_logits)); CK(cudaFree(d_grad)); CK(cudaFree(d_loss)); CK(cudaFree(d_max)); CK(cudaFree(d_sum));
  CK(cudaFree(d_gout)); CK(cudaFree(d_den));
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <f32|bf16> <tokens.bin> [--offset TOKENS] [--warmup N] [--iters N]\n", argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  if (mode != "f32" && mode != "bf16") { std::fprintf(stderr, "mode must be f32 or bf16\n"); return 1; }
  int64_t offset = 0; int warmup = 50, iters = 200;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--offset") offset = std::atoll(argv[i + 1]);
    else if (a == "--warmup") warmup = std::atoi(argv[i + 1]);
    else if (a == "--iters") iters = std::atoi(argv[i + 1]);
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
  }

  // ---- targets: the next-token targets of N + 1 real tokens ----
  std::vector<uint16_t> tok(kN + 1);
  {
    FILE *f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    if (std::fseek(f, (long)(offset * 2), SEEK_SET) != 0 || std::fread(tok.data(), 2, kN + 1, f) != (size_t)(kN + 1)) {
      std::fprintf(stderr, "cannot read %ld tokens at offset %ld\n", (long)(kN + 1), (long)offset); return 1;
    }
    std::fclose(f);
  }
  std::vector<uint16_t> tg(tok.begin() + 1, tok.end());
  for (uint16_t t : tg)
    if (t >= kV) { std::fprintf(stderr, "token %u >= vocab %ld\n", (unsigned)t, (long)kV); return 1; }

  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | mode %s | targets %s @ token %ld | N %ld V %ld | warmup %d iters %d\n", prop.name,
              mode.c_str(), argv[2], (long)offset, (long)kN, (long)kV, warmup, iters);

  // ---- part 1: the kernels alone (plain cudaMalloc, outside the caching allocator) ----
  {
    uint16_t *d_tgt = nullptr;
    CK(cudaMalloc(&d_tgt, kN * 2));
    CK(cudaMemcpy(d_tgt, tg.data(), kN * 2, cudaMemcpyHostToDevice));
    if (mode == "f32") kernel_part<float>("f32", d_tgt, warmup, iters);
    else               kernel_part<bfloat16_t>("bf16", d_tgt, warmup, iters);
    CK(cudaFree(d_tgt));
  }

  // ---- part 2: the trainer's lm_head + loss through autograd ----
  // X ~ N(0,1), W ~ N(0, 1/D) so the logits are ~ N(0,1); fixed seeds (as test_loss_grad_inplace.cu)
  std::vector<uint16_t> hx(kN * kD), hw(kD * kV);
  { std::mt19937 g(11); std::normal_distribution<float> n(0.f, 1.f); for (auto &v : hx) v = bf16_rne(n(g)); }
  { std::mt19937 g(12); std::normal_distribution<float> n(0.f, 1.f / std::sqrt((float)kD)); for (auto &v : hw) v = bf16_rne(n(g)); }
  auto upload = [](const Shape &s, Dtype d, const void *p, size_t bytes, bool req_grad) {
    Tensor t(s, d, kCuda, req_grad);
    CK(cudaMemcpy(t.data(), p, bytes, cudaMemcpyHostToDevice));
    return t;
  };
  Tensor X = upload(Shape{{kB, kT, kD}}, Dtype::Bfloat16, hx.data(), hx.size() * 2, true);
  Tensor W = upload(Shape{{kD, kV}}, Dtype::Bfloat16, hw.data(), hw.size() * 2, true);
  Tensor targets = upload(Shape{{kB, kT}}, Dtype::UInt16, tg.data(), tg.size() * 2, false);
  const float scale = 1.0f / 32.0f;   // the trainer's grad_scale = 1 / GRAD_ACCUM (32 micro-steps on one GPU)
  Tensor seed = upload(Shape{{1}}, Dtype::Float32, &scale, 4, false);
  const Dtype logit_dtype = mode == "f32" ? Dtype::Float32 : Dtype::Bfloat16;

  cudaEvent_t f0, f1, b0, b1;
  CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1)); CK(cudaEventCreate(&b0)); CK(cudaEventCreate(&b1));
  constexpr int kReps = 3;
  uint64_t cs_w[kReps], cs_x[kReps]; float loss_v[kReps], fwd_ms[kReps], bwd_ms[kReps];
  for (int rep = 0; rep < kReps; ++rep) {
    if (rep > 0) { X.zero_grad(); W.zero_grad(); }
    Tensor logits = autograd::to_dtype(autograd::matmul(X, W), logit_dtype);   // as model.forward returns them
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(f0));
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    CK(cudaEventRecord(f1));
    Tensor loss32 = autograd::to_dtype(loss, Dtype::Float32);                  // as the trainer seeds it
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(b0));
    loss32.backward(&seed);
    CK(cudaEventRecord(b1));
    CK(cudaEventSynchronize(b1));
    CK(cudaEventElapsedTime(&fwd_ms[rep], f0, f1));
    CK(cudaEventElapsedTime(&bwd_ms[rep], b0, b1));
    CK(cudaMemcpy(&loss_v[rep], loss32.data(), 4, cudaMemcpyDeviceToHost));
    Tensor gw = W.grad_view(), gx = X.grad_view();
    cs_w[rep] = checksum_dev(gw.data(), (size_t)gw.numel() * (gw.dtype() == Dtype::Float32 ? 4 : 2));
    cs_x[rep] = checksum_dev(gx.data(), (size_t)gx.numel() * (gx.dtype() == Dtype::Float32 ? 4 : 2));
    std::printf("rep %d: loss %.6f (%08x) | W.grad %s %016llx | X.grad %s %016llx | loss fwd %.3f ms | backward %.3f ms\n",
                rep + 1, loss_v[rep], bits(loss_v[rep]), gw.dtype() == Dtype::Float32 ? "fp32" : "bf16",
                (unsigned long long)cs_w[rep], gx.dtype() == Dtype::Float32 ? "fp32" : "bf16",
                (unsigned long long)cs_x[rep], fwd_ms[rep], bwd_ms[rep]);
  }
  bool same = true;
  for (int r = 1; r < kReps; ++r)
    same &= cs_w[r] == cs_w[0] && cs_x[r] == cs_x[0] && bits(loss_v[r]) == bits(loss_v[0]);
  const auto st = CachingCUDAAllocator::instance().get_stats();
  std::printf("AUTOGRAD dtype=%s loss=%.6f loss_bits=%08x wgrad_cs=%016llx xgrad_cs=%016llx fwd_ms=%.3f bwd_ms=%.3f "
              "active_peak_MiB=%.1f allocated_peak_MiB=%.1f reserved_peak_MiB=%.1f reps_identical=%d\n",
              mode.c_str(), loss_v[0], bits(loss_v[0]), (unsigned long long)cs_w[0], (unsigned long long)cs_x[0],
              0.5f * (fwd_ms[1] + fwd_ms[2]), 0.5f * (bwd_ms[1] + bwd_ms[2]), st.active_peak / 1048576.0,
              st.allocated_peak / 1048576.0, st.reserved_peak / 1048576.0, same ? 1 : 0);
  return same ? 0 : 3;
}
