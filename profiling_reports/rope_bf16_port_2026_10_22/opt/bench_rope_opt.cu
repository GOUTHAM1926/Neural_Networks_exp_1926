// =============================================================================
// bench_rope_opt.cu (2026-10-22): the optimized kernels (opt/FusedRMSRoPE_bf16_opt.cu, namespace ropt) against the
// original ones (FusedRMSRoPE_bf16.cu), on the same inputs:
//   1. BIT-IDENTITY of every output: forward q_out, k_out, q_rstd, k_rstd; backward dq, dk, the dgamma partials;
//      reduce dgamma_q, dgamma_k. Any differing element is a FAIL, and its timing does not count.
//   2. TIME of each kernel, original vs optimized, without and with packed positions (CUDA events around `reps`
//      back-to-back launches, 2 s warm-up, the variants alternated in each round, median of `rounds`).
// Shape: T 4096, 18 query / 6 KV heads, head dim 128, NeoX, eps 1e-6; B = 1, 2, 4, 8.
// build: nvcc -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a -I<Tensor-Implementations>/include
//        -I<dir with FusedRMSRoPE_bf16.cu> -I<dir with FusedRMSRoPE_bf16_opt.cu> -o bench_rope_opt bench_rope_opt.cu
// usage: CUDA_VISIBLE_DEVICES=<free GPU> ./bench_rope_opt [--reps 200] [--rounds 9] [--B 1,2,4,8]
// =============================================================================
#include "FusedRMSRoPE_bf16.cu"
#include "FusedRMSRoPE_bf16_opt.cu"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

namespace orig = OwnTensor::cuda;
namespace ropt = OwnTensor::cuda::ropt;
using bf16 = __nv_bfloat16;

static constexpr int kT = 4096, kHq = 18, kHkv = 6, kHD = 128, kH = kHq + kHkv;
static constexpr float kEps = 1e-6f;

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }
template <typename T> static T* dev(const std::vector<T>& h) {
  T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <typename T> static T* dev_zero(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <typename T> static int64_t diff(const T* a, const T* b, size_t n) {   // number of differing elements (bit compare)
  std::vector<T> ha(n), hb(n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
  int64_t d = 0; for (size_t i = 0; i < n; ++i) d += std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0; return d;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
// debug: the first differing bf16 elements of a head-major [.., hd] tensor, with the head dimension and both values
static void show_diff(const char* nm, const bf16* a, const bf16* b, size_t n, int hd) {
  std::vector<uint16_t> ha(n), hb(n);
  CK(cudaMemcpy(ha.data(), a, n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, n * 2, cudaMemcpyDeviceToHost));
  int shown = 0; int64_t lo = 0, hi = 0;
  for (size_t i = 0; i < n; ++i) if (ha[i] != hb[i]) {
    ((int)(i % hd) < hd / 2 ? lo : hi)++;
    if (shown < 6) { std::printf("    %s[%zu] dim %3d: original %.8g optimized %.8g\n", nm, i, (int)(i % hd), bf2f(ha[i]), bf2f(hb[i])); ++shown; }
  }
  std::printf("    %s: %lld differing in dims < %d (a*cos - b*sin), %lld in dims >= %d (a*sin + b*cos)\n",
              nm, (long long)lo, hd / 2, (long long)hi, hd / 2);
}

int main(int argc, char** argv) {
  int reps = 200, rounds = 9; std::string Bs = "1,2,4,8";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--reps") reps = std::atoi(argv[i + 1]);
    if (std::string(argv[i]) == "--rounds") rounds = std::atoi(argv[i + 1]);
    if (std::string(argv[i]) == "--B") Bs = argv[i + 1];
  }
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | T %d, %d / %d heads, hd %d, NeoX | %d launches per round, median of %d rounds\n",
              prop.name, kT, kHq, kHkv, kHD, reps, rounds);
  bool all_ok = true;
  for (size_t s = 0; s <= Bs.size();) {
    size_t e = Bs.find(',', s); if (e == std::string::npos) e = Bs.size();
    const int B = std::atoi(Bs.substr(s, e - s).c_str()); s = e + 1;
    const size_t nq = (size_t)B * kT * kHq * kHD, nk = (size_t)B * kT * kHkv * kHD;
    const size_t nrq = (size_t)B * kHq * kT, nrk = (size_t)B * kHkv * kT;
    const size_t npq = (size_t)B * kHq * BF16_BWD_T_CHUNKS * kHD, npk = (size_t)B * kHkv * BF16_BWD_T_CHUNKS * kHD;
    uint64_t st = 0x9E3779B97F4A7C15ull + B;
    auto next = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
    auto rnd = [&]() { return (float)((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; };
    auto bf_vec = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) x = f2bf(rnd()); return v; };
    std::vector<float> hg(kHD), hcs((size_t)kT * kHD);
    for (auto& x : hg) x = 1.0f + 0.1f * rnd();
    for (auto& x : hcs) x = rnd();
    std::vector<int32_t> hpos((size_t)B * kT);
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < kT;) { int len = 1 + (int)(next() % 1500); if (t + len > kT) len = kT - t;
        for (int i = 0; i < len; ++i) hpos[(size_t)b * kT + t + i] = i; t += len; }
    bf16 *q = (bf16*)dev(bf_vec(nq)), *k = (bf16*)dev(bf_vec(nk)), *dzq = (bf16*)dev(bf_vec(nq)), *dzk = (bf16*)dev(bf_vec(nk));
    float *qg = dev(hg), *kg = dev(hg), *cs = dev(hcs);
    int32_t* pos = dev(hpos);
    // outputs: [0] = original, [1] = optimized
    bf16 *qo[2], *ko[2], *dq[2], *dk[2]; float *qr[2], *kr[2], *qp[2], *kp[2], *dqg[2], *dkg[2];
    for (int v = 0; v < 2; ++v) {
      qo[v] = dev_zero<bf16>(nq); ko[v] = dev_zero<bf16>(nk); dq[v] = dev_zero<bf16>(nq); dk[v] = dev_zero<bf16>(nk);
      qr[v] = dev_zero<float>(nrq); kr[v] = dev_zero<float>(nrk); qp[v] = dev_zero<float>(npq); kp[v] = dev_zero<float>(npk);
      dqg[v] = dev_zero<float>(kHD); dkg[v] = dev_zero<float>(kHD);
    }
    const dim3 fgrid((kT + 31) / 32, kH, B), fblock(256), bgrid(BF16_BWD_T_CHUNKS, kH, B), bblock(32);
    // the original kernels, exactly as their launchers call them
    auto fwd_o = [&](const int32_t* p) {
      if (p) orig::fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, true ><<<fgrid, fblock>>>(q, k, qo[0], ko[0], qg, kg, cs, qr[0], kr[0], kHq, kHkv, kT, 0, kEps, p);
      else   orig::fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, false><<<fgrid, fblock>>>(q, k, qo[0], ko[0], qg, kg, cs, qr[0], kr[0], kHq, kHkv, kT, 0, kEps, nullptr);
    };
    auto bwd_o = [&](const int32_t* p) {
      if (p) orig::fused_qknorm_rope_bwd_bf16_tm<true, kHD, true ><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[0], dk[0], qp[0], kp[0], kHq, kHkv, kT, 0, p);
      else   orig::fused_qknorm_rope_bwd_bf16_tm<true, kHD, false><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[0], dk[0], qp[0], kp[0], kHq, kHkv, kT, 0, nullptr);
    };
    auto red_o = [&]() {
      orig::reduce_dgamma_bf16_tm<<<1, 128>>>(qp[0], dqg[0], B * kHq * BF16_BWD_T_CHUNKS, kHD);
      orig::reduce_dgamma_bf16_tm<<<1, 128>>>(kp[0], dkg[0], B * kHkv * BF16_BWD_T_CHUNKS, kHD);
    };
    // the optimized kernels (the backward reads the ORIGINAL forward's rstd, so it is compared on equal inputs)
    auto fwd_n = [&](const int32_t* p) {
      if (p) ropt::fwd_neox<kHD, 4, 8, true ><<<fgrid, fblock>>>(q, k, qo[1], ko[1], qg, kg, cs, qr[1], kr[1], kHq, kHkv, kT, 0, kEps, p);
      else   ropt::fwd_neox<kHD, 4, 8, false><<<fgrid, fblock>>>(q, k, qo[1], ko[1], qg, kg, cs, qr[1], kr[1], kHq, kHkv, kT, 0, kEps, nullptr);
    };
    auto bwd_n = [&](const int32_t* p) {
      if (p) ropt::bwd<true, kHD, true ><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[1], dk[1], qp[1], kp[1], kHq, kHkv, kT, 0, p);
      else   ropt::bwd<true, kHD, false><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[1], dk[1], qp[1], kp[1], kHq, kHkv, kT, 0, nullptr);
    };
    auto red_n = [&]() {
      ropt::reduce_dgamma2<32><<<dim3((kHD + 31) / 32, 2), 32>>>(qp[1], dqg[1], B * kHq * BF16_BWD_T_CHUNKS,
                                                                 kp[1], dkg[1], B * kHkv * BF16_BWD_T_CHUNKS, kHD);
    };

    // ---- 1. bit-identity, without and with positions ----
    for (int withpos = 0; withpos < 2; ++withpos) {
      const int32_t* p = withpos ? pos : nullptr;
      fwd_o(p); bwd_o(p); red_o(); fwd_n(p); bwd_n(p); red_n(); CK(cudaGetLastError());
      struct { const char* nm; int64_t bad; size_t n; } r[] = {
        {"q_out", diff(qo[0], qo[1], nq), nq}, {"k_out", diff(ko[0], ko[1], nk), nk},
        {"q_rstd", diff(qr[0], qr[1], nrq), nrq}, {"k_rstd", diff(kr[0], kr[1], nrk), nrk},
        {"dq", diff(dq[0], dq[1], nq), nq}, {"dk", diff(dk[0], dk[1], nk), nk},
        {"dgamma partials q", diff(qp[0], qp[1], npq), npq}, {"dgamma partials k", diff(kp[0], kp[1], npk), npk},
        {"dgamma_q", diff(dqg[0], dqg[1], kHD), (size_t)kHD}, {"dgamma_k", diff(dkg[0], dkg[1], kHD), (size_t)kHD}};
      int64_t bad = 0; std::string which;
      for (auto& x : r) { bad += x.bad; if (x.bad) which += std::string(" ") + x.nm + "(" + std::to_string(x.bad) + "/" + std::to_string(x.n) + ")"; }
      std::printf("CHECK B=%d %-12s optimized vs original, 10 outputs: %s\n", B, withpos ? "positions" : "no positions",
                  bad ? ("DIFFERENT" + which + "  <-- FAIL").c_str() : "all bit-identical");
      all_ok = all_ok && bad == 0;
      if (bad && std::getenv("ROPT_DEBUG")) show_diff("q_out", qo[0], qo[1], nq, kHD);
    }

    // ---- 2. time ----
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto round_us = [&](auto&& body) {
      CK(cudaEventRecord(e0)); for (int i = 0; i < reps; ++i) body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return 1000.0f * ms / reps;
    };
    { const auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(2)) { fwd_o(pos); fwd_n(pos); bwd_o(pos); bwd_n(pos); red_o(); red_n(); CK(cudaDeviceSynchronize()); } }
    enum { FO0, FN0, FO1, FN1, BO0, BN0, BO1, BN1, RO, RN, NV };
    std::vector<float> t[NV];
    for (int r = 0; r < rounds; ++r) {
      t[FO0].push_back(round_us([&] { fwd_o(nullptr); })); t[FN0].push_back(round_us([&] { fwd_n(nullptr); }));
      t[FO1].push_back(round_us([&] { fwd_o(pos); }));     t[FN1].push_back(round_us([&] { fwd_n(pos); }));
      t[BO0].push_back(round_us([&] { bwd_o(nullptr); })); t[BN0].push_back(round_us([&] { bwd_n(nullptr); }));
      t[BO1].push_back(round_us([&] { bwd_o(pos); }));     t[BN1].push_back(round_us([&] { bwd_n(pos); }));
      t[RO].push_back(round_us([&] { red_o(); }));         t[RN].push_back(round_us([&] { red_n(); }));
    }
    float m[NV]; for (int i = 0; i < NV; ++i) m[i] = median(t[i]);
    auto line = [&](const char* nm, float o, float n) {
      std::printf("  %-34s original %8.1f us | optimized %8.1f us | %5.2fx\n", nm, o, n, o / n);
    };
    std::printf("TIME B=%d\n", B);
    line("forward, no positions (pretraining)", m[FO0], m[FN0]);
    line("forward, positions (SFT)", m[FO1], m[FN1]);
    line("backward, no positions", m[BO0], m[BN0]);
    line("backward, positions", m[BO1], m[BN1]);
    line("dgamma reduce (q + k)", m[RO], m[RN]);
    line("TOTAL pretraining (fwd+bwd+reduce)", m[FO0] + m[BO0] + m[RO], m[FN0] + m[BN0] + m[RN]);
    line("TOTAL SFT (fwd+bwd+reduce)", m[FO1] + m[BO1] + m[RO], m[FN1] + m[BN1] + m[RN]);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    for (int v = 0; v < 2; ++v)
      for (void* x : std::initializer_list<void*>{qo[v], ko[v], dq[v], dk[v], qr[v], kr[v], qp[v], kp[v], dqg[v], dkg[v]}) CK(cudaFree(x));
    for (void* x : std::initializer_list<void*>{q, k, dzq, dzk, qg, kg, cs, pos}) CK(cudaFree(x));
    if (e >= Bs.size()) break;
  }
  std::printf("%s\n", all_ok ? "RESULT: every optimized output bit-identical to the original; timings above"
                             : "RESULT: FAIL, some optimized outputs differ (see CHECK lines); timings do not count");
  return all_ok ? 0 : 3;
}
