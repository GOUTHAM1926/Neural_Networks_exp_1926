// ============================================================================
// bench_sparse_ce_sft.cu -- standalone timing of the SFT sparse cross-entropy
// loss kernels (forward + backward) at the SFT micro-batch, v1 padding.
//
//   shape   : B=8, T=4096 -> N=32768 rows, V=50304, bf16 logits, uint16 targets,
//             ignore label 65535 (prompt + pad positions), as BLU_ALL_BF16=1 runs.
//   calls   : the same library entry points autograd::sparse_cross_entropy_loss
//             reaches on H200 (LossOps.cpp / LossBackward.cpp):
//     forward  cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<bf16,u16>
//              = sparse_ce_forward_kernel_vec_save_stats + 2x sum_reduction_kernel
//     backward cuda::sparse_ce_backward_cuda_vec_with_stats<bf16,u16>
//              = sparseCENormalize_from_stats
//   cases   : real micro-batches from labels.bin (first / min / median / max
//             trained fraction), every row trained with masking off (the
//             pretraining path), and every row masked (the zero-write floor).
//   timing  : CUDA events around every call, warm-up first, median/min/mean, for
//             F    forward save_stats alone          (trainer, training micro-batch)
//             B    backward with stats alone         (trainer)
//             F+B  the two back to back, one timing  (trainer pair)
//             Fv   forward without stats             (what validation runs)
//             Fv+Bo  no-stats forward + a backward that recomputes max/sum
//                    (reference: what saving the stats buys)
//   checks  : after timing, per case -- loss finite; the number of rows the
//             forward actually computed (saved_sum != 1) equals the trained
//             count; a masked row's gradient is bit-zero; a trained row's
//             gradient sums to ~0 and is negative at its target; the no-stats
//             path gives the same loss and the same gradient row.
//
// Build (from Tensor-Implementations/, library already built):
//   make -s -n run-cuda FILE=Tests/kernel_tests/bench_sparse_ce_sft.cu   (prints the nvcc line)
// Run:
//   ./bench_sparse_ce_sft <labels.bin> [--case NAME] [--iters N] [--warmup N]
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dtype/Types.h"
#include "ops/helpers/LossKernels.h"

using OwnTensor::bfloat16_t;

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,  \
              cudaGetErrorString(e_));                                         \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int64_t kB = 8, kT = 4096, kN = kB * kT, kV = 50304;
static constexpr uint16_t kIgnore = 65535;
static constexpr double kHbmPeak = 4.8e12;  // H200 SXM HBM3e spec, bytes/s

// Deterministic uniform [-4, 4) logits. The values do not change the work the
// kernels do (one pass over each trained row, exp per element).
__global__ void fill_logits(__nv_bfloat16 *p, int64_t n, uint32_t seed) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    float u = (h >> 8) * (1.0f / 16777216.0f);
    p[i] = __float2bfloat16_rn((u - 0.5f) * 8.0f);
  }
}

struct Case {
  std::string name, desc;
  std::vector<uint16_t> tgt;  // [kN]
  int64_t ignore = kIgnore;   // -1 = masking off
  int64_t trained = 0;
};

struct Stat { double med, min, mean; };
static Stat stats_of(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  double s = 0;
  for (float x : v) s += x;
  return {v[v.size() / 2], v.front(), s / v.size()};
}

static float bf16_bits_to_float(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char **argv) {
  std::string labels_path, only;
  int iters = 50, warmup = 5;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--case" && i + 1 < argc) only = argv[++i];
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else labels_path = a;
  }
  if (labels_path.empty()) {
    fprintf(stderr, "usage: %s <labels.bin> [--case NAME] [--iters N] [--warmup N]\n", argv[0]);
    return 1;
  }

  // ---- labels.bin: uint16 [R, T], 65535 = ignore -----------------------------
  FILE *f = fopen(labels_path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", labels_path.c_str()); return 1; }
  std::vector<uint16_t> lab;
  {
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    lab.resize(bytes / 2);
    if (fread(lab.data(), 2, lab.size(), f) != lab.size()) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
  }
  const int64_t R = (int64_t)lab.size() / kT, nb = R / kB;
  if (nb < 1 || (int64_t)lab.size() % kT) { fprintf(stderr, "labels.bin is not [R,%ld]\n", (long)kT); return 1; }
  for (uint16_t x : lab)
    if (x != kIgnore && x >= kV) { fprintf(stderr, "label %u >= V\n", x); return 1; }

  std::vector<int64_t> cnt(nb, 0);
  for (int64_t b = 0; b < nb; ++b)
    for (int64_t i = 0; i < kN; ++i) cnt[b] += lab[b * kN + i] != kIgnore;
  std::vector<int64_t> order(nb);
  for (int64_t b = 0; b < nb; ++b) order[b] = b;
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return cnt[a] < cnt[b]; });
  int64_t total = 0;
  for (int64_t c : cnt) total += c;

  auto real_case = [&](const char *name, int64_t b, const char *why) {
    Case c;
    c.name = name;
    c.tgt.assign(lab.begin() + b * kN, lab.begin() + (b + 1) * kN);
    c.trained = cnt[b];
    char d[160];
    snprintf(d, sizeof d, "%s: micro-batch %ld (rows %ld-%ld)", why, (long)b, (long)(b * kB), (long)(b * kB + kB - 1));
    c.desc = d;
    return c;
  };
  std::vector<Case> cases;
  cases.push_back(real_case("real_first", 0, "the first micro-batch of the run"));
  cases.push_back(real_case("real_min", order.front(), "fewest trained tokens"));
  cases.push_back(real_case("real_median", order[nb / 2], "median trained tokens"));
  cases.push_back(real_case("real_max", order.back(), "most trained tokens"));
  {
    Case c = real_case("nomask", 0, "masking OFF, every row trained (pretraining path)");
    for (int64_t i = 0; i < kN; ++i)
      if (c.tgt[i] == kIgnore) c.tgt[i] = (uint16_t)((i * 7919) % kV);
    c.ignore = -1;
    c.trained = kN;
    cases.push_back(c);
  }
  {
    Case c;
    c.name = "all_masked";
    c.desc = "every row masked: backward only writes zeros (floor)";
    c.tgt.assign(kN, kIgnore);
    c.trained = 0;
    cases.push_back(c);
  }

  // ---- device -----------------------------------------------------------------
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  size_t freeb = 0, totb = 0;
  CK(cudaMemGetInfo(&freeb, &totb));
  printf("device      : %s, %d SMs, %.1f GiB total, %.1f GiB free\n", prop.name,
         prop.multiProcessorCount, totb / 1073741824.0, freeb / 1073741824.0);
  printf("shape       : B=%ld T=%ld N=%ld V=%ld, logits bf16, targets uint16, ignore %u\n",
         (long)kB, (long)kT, (long)kN, (long)kV, kIgnore);
  printf("labels      : %s, %ld rows, %ld micro-batches, mean trained %.4f\n",
         labels_path.c_str(), (long)R, (long)nb, (double)total / (nb * kN));
  printf("timing      : %d warm-up + %d timed calls per pass, CUDA events per call\n\n", warmup, iters);

  const size_t logit_bytes = (size_t)kN * kV * sizeof(bfloat16_t);
  bfloat16_t *d_logits = nullptr, *d_grad = nullptr;
  uint16_t *d_tgt = nullptr;
  float *d_loss = nullptr, *d_max = nullptr, *d_sum = nullptr, *d_gout = nullptr, *d_denom = nullptr;
  CK(cudaMalloc(&d_logits, logit_bytes));
  CK(cudaMalloc(&d_grad, logit_bytes));
  CK(cudaMalloc(&d_tgt, kN * sizeof(uint16_t)));
  CK(cudaMalloc(&d_loss, sizeof(float)));
  CK(cudaMalloc(&d_max, kN * sizeof(float)));
  CK(cudaMalloc(&d_sum, kN * sizeof(float)));
  CK(cudaMalloc(&d_gout, sizeof(float)));
  CK(cudaMalloc(&d_denom, sizeof(float)));
  fill_logits<<<4 * prop.multiProcessorCount, 256>>>(reinterpret_cast<__nv_bfloat16 *>(d_logits),
                                                      (int64_t)kN * kV, 1234u);
  CK(cudaGetLastError());
  const float one = 1.0f;
  CK(cudaMemcpy(d_gout, &one, sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());

  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  std::vector<cudaEvent_t> ev(2 * iters);
  for (auto &e : ev) CK(cudaEventCreate(&e));

  struct Row { std::string name; double frac; Stat F, B, FB, Fv, FvBo; double gF, gB; bool ok; };
  std::vector<Row> rows;
  int failures = 0;

  for (Case &c : cases) {
    if (!only.empty() && only != c.name) continue;
    CK(cudaMemcpy(d_tgt, c.tgt.data(), kN * sizeof(uint16_t), cudaMemcpyHostToDevice));
    const float denom = (float)std::max<int64_t>(c.trained, 1);
    CK(cudaMemcpy(d_denom, &denom, sizeof(float), cudaMemcpyHostToDevice));

    // Trainer path (training micro-batch, grad on):  F = forward save_stats, B = backward with stats.
    // Reference path: Fv = forward WITHOUT stats (what validation runs, grad off),
    //                 Bo = backward WITHOUT stats (recomputes max/sum: Reduce + Normalize kernels).
    auto F = [&]() {
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<bfloat16_t, uint16_t>(
          d_logits, d_tgt, d_loss, d_max, d_sum, kN, kV, st, c.ignore, d_denom);
    };
    auto B = [&]() {
      OwnTensor::cuda::sparse_ce_backward_cuda_vec_with_stats<bfloat16_t, uint16_t>(
          d_logits, d_tgt, d_grad, d_max, d_sum, kN, kV, d_gout, d_denom, st, c.ignore);
    };
    auto Fv = [&]() {
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec<bfloat16_t, uint16_t>(
          d_logits, d_tgt, d_loss, kN, kV, st, c.ignore, d_denom);
    };
    auto Bo = [&]() {
      OwnTensor::cuda::sparse_cross_entropy_backward_cuda<bfloat16_t, uint16_t>(
          d_logits, d_tgt, d_grad, kN, kV, d_gout, d_denom, st, c.ignore);
    };
    auto time_it = [&](auto &&body) {
      for (int w = 0; w < warmup; ++w) body();
      CK(cudaStreamSynchronize(st));
      std::vector<float> t(iters);
      for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(ev[2 * i], st)); body(); CK(cudaEventRecord(ev[2 * i + 1], st));
      }
      CK(cudaStreamSynchronize(st));
      CK(cudaGetLastError());
      for (int i = 0; i < iters; ++i) CK(cudaEventElapsedTime(&t[i], ev[2 * i], ev[2 * i + 1]));
      return stats_of(t);
    };
    const Stat sF = time_it(F);                                   // forward alone
    const Stat sB = time_it(B);                                   // backward alone (stats from the last F)
    const Stat sFB = time_it([&]() { F(); B(); });                // trainer pair, back to back
    const Stat sFv = time_it(Fv);                                 // validation forward
    const Stat sFvBo = time_it([&]() { Fv(); Bo(); });            // no-stats pair (recompute in backward)

    // ---- checks (after timing, so they cannot perturb it) --------------------
    // 1) trainer path once more, then inspect its outputs.
    F(); B();
    CK(cudaStreamSynchronize(st));
    float loss = 0;
    CK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> hsum(kN);
    CK(cudaMemcpy(hsum.data(), d_sum, kN * sizeof(float), cudaMemcpyDeviceToHost));
    int64_t computed = 0;
    for (float s : hsum) computed += (s != 1.0f);
    int64_t first_masked = -1, first_trained = -1;
    for (int64_t i = 0; i < kN; ++i) {
      bool masked = (c.ignore >= 0 && c.tgt[i] == c.ignore);
      if (masked && first_masked < 0) first_masked = i;
      if (!masked && first_trained < 0) first_trained = i;
    }
    std::vector<uint16_t> g(kV), gref(kV);
    bool masked_zero = true, trained_ok = true;
    double gsum = 0, gtgt = 0;
    if (first_masked >= 0) {
      CK(cudaMemcpy(g.data(), d_grad + first_masked * kV, kV * 2, cudaMemcpyDeviceToHost));
      for (uint16_t b : g) masked_zero &= (b & 0x7fff) == 0;
    }
    if (first_trained >= 0) {
      CK(cudaMemcpy(gref.data(), d_grad + first_trained * kV, kV * 2, cudaMemcpyDeviceToHost));
      for (int64_t k = 0; k < kV; ++k) gsum += bf16_bits_to_float(gref[k]);
      gtgt = bf16_bits_to_float(gref[c.tgt[first_trained]]);
      trained_ok = gtgt < 0 && std::fabs(gsum) < 0.05 * std::fabs(gtgt) + 1e-12;
    }
    const bool loss_ok = std::isfinite(loss) && (c.trained == 0 ? loss == 0.0f : loss > 0.0f);
    const bool count_ok = computed == c.trained;
    // 2) the no-stats path must give the same loss and the same gradient row.
    Fv(); Bo();
    CK(cudaStreamSynchronize(st));
    float loss_v = 0;
    CK(cudaMemcpy(&loss_v, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    double gdiff = 0;
    if (first_trained >= 0) {
      CK(cudaMemcpy(g.data(), d_grad + first_trained * kV, kV * 2, cudaMemcpyDeviceToHost));
      for (int64_t k = 0; k < kV; ++k)
        gdiff = std::max(gdiff, std::fabs((double)bf16_bits_to_float(g[k]) - bf16_bits_to_float(gref[k])));
    }
    const bool same_ok = std::fabs(loss_v - loss) <= 1e-4 * std::fabs(loss) + 1e-6 &&
                         gdiff <= 1e-2 * std::fabs(gtgt) + 1e-12;
    const bool ok = loss_ok && count_ok && masked_zero && trained_ok && same_ok;
    failures += !ok;

    // ---- bytes the kernels must move (HBM) ---------------------------------
    const double L = (double)kN * kV * 2;                  // whole logits (or grad) tensor
    const double Ltr = (double)c.trained * kV * 2;         // trained rows only
    const double bF = Ltr + (double)kN * (2 + 12 + 4);     // read trained rows; per-row loss/max/sum
    const double bB = Ltr + L + (double)kN * (2 + 8);      // read trained rows; write the whole grad
    const double bFv = Ltr + (double)kN * (2 + 4 + 4);
    const double bBo = 2 * Ltr + L + (double)kN * (2 + 8); // Reduce pass re-reads trained rows
    auto gbs = [](double bytes, const Stat &s) { return bytes / (s.med * 1e-3) / 1e9; };
    const double frac = (double)c.trained / kN;

    printf("== %-12s %s\n", c.name.c_str(), c.desc.c_str());
    printf("   trained rows : %ld of %ld (%.2f%%), ignore_index %ld\n", (long)c.trained, (long)kN, 100 * frac, (long)c.ignore);
    auto line = [&](const char *what, const Stat &s, double bytes) {
      const double g_ = gbs(bytes, s);
      printf("   %-26s median %8.4f  min %8.4f  mean %8.4f ms | %6.3f GB -> %5.0f GB/s = %5.1f%% of 4.8 TB/s\n",
             what, s.med, s.min, s.mean, bytes / 1e9, g_, 100 * g_ * 1e9 / kHbmPeak);
    };
    line("F   fwd save_stats", sF, bF);
    line("B   bwd with stats", sB, bB);
    line("F+B trainer pair", sFB, bF + bB);
    line("Fv  fwd no stats (val)", sFv, bFv);
    line("Fv+Bo no-stats pair", sFvBo, bFv + bBo);
    printf("   checks       : loss %.6f %s | rows computed %ld (expect %ld) %s | masked row %ld grad bit-zero %s | "
           "trained row %ld: g[target] %.3e, sum %.3e %s | no-stats path: loss %.6f, max|dg| %.2e %s  => %s\n\n",
           loss, loss_ok ? "ok" : "BAD", (long)computed, (long)c.trained, count_ok ? "ok" : "BAD",
           (long)first_masked, first_masked < 0 ? "n/a" : (masked_zero ? "ok" : "BAD"),
           (long)first_trained, gtgt, gsum, first_trained < 0 ? "n/a" : (trained_ok ? "ok" : "BAD"),
           loss_v, gdiff, same_ok ? "ok" : "BAD", ok ? "PASS" : "FAIL");
    rows.push_back({c.name, frac, sF, sB, sFB, sFv, sFvBo, gbs(bF, sF), gbs(bB, sB), ok});
  }

  printf("==================================================================================================\n");
  printf("SUMMARY  (%s, B=8 T=4096 V=50304 bf16, median of %d calls)\n", prop.name, iters);
  printf("==================================================================================================\n");
  printf("%-12s %8s | %9s %6s | %9s %6s | %9s | %9s | %9s | %6s\n", "case", "trained", "F ms", "%peak",
         "B ms", "%peak", "F+B ms", "Fv ms", "Fv+Bo ms", "check");
  for (auto &r : rows)
    printf("%-12s %7.2f%% | %9.4f %5.1f%% | %9.4f %5.1f%% | %9.4f | %9.4f | %9.4f | %6s\n", r.name.c_str(),
           100 * r.frac, r.F.med, 100 * r.gF * 1e9 / kHbmPeak, r.B.med, 100 * r.gB * 1e9 / kHbmPeak, r.FB.med,
           r.Fv.med, r.FvBo.med, r.ok ? "PASS" : "FAIL");
  printf("F = forward save_stats, B = backward with stats (the trainer's pair); Fv = forward without stats "
         "(validation); Bo = backward that recomputes the stats\n");
  for (auto &r : rows)
    if (r.name == "real_median")
      printf("\nper optimizer step on 1 GPU (64 rows = 8 micro-batches, median batch): 8 x F+B %.4f = %.2f ms\n",
             r.FB.med, 8 * r.FB.med);
  printf("%s (%d case%s failed)\n", failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASS", failures,
         failures == 1 ? "" : "s");

  for (auto &e : ev) cudaEventDestroy(e);
  cudaStreamDestroy(st);
  cudaFree(d_logits); cudaFree(d_grad); cudaFree(d_tgt); cudaFree(d_loss);
  cudaFree(d_max); cudaFree(d_sum); cudaFree(d_gout); cudaFree(d_denom);
  return failures ? 3 : 0;
}
