// =============================================================================
// bench_rope_v2.cu (2026-10-22): the round-2 kernels (opt/FusedRMSRoPE_bf16_v2.cu, namespace ropt2) against the
// original (FusedRMSRoPE_bf16.cu), every template VARIANT separately:
//   forward : fwd2 with HPW (heads per warp) = 2, 4, 6, 12, 24
//   backward: bwd2 with (HPB heads per block, STAGE tokens per cos/sin stage) = (8,8) (12,8) (24,8) (24,16)
//   reduce  : reduce3 with UNROLL = 32, 64, 128
// For each variant, without and with packed positions: 1. BIT-IDENTITY of its outputs against the original kernel's
// (forward: q_out, k_out, q_rstd, k_rstd; backward: dq, dk, both dgamma partial sets; reduce: dgamma_q, dgamma_k);
// a variant with any differing element is marked FAIL and its time does not count. 2. TIME against the original,
// alternated in every round (CUDA events, `reps` launches per round, median of `rounds`, 2 s warm-up).
// Shape: T 4096, 18 query / 6 KV heads (24 heads), head dim 128, NeoX, eps 1e-6.
// build: nvcc -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a -I<Tensor-Implementations>/include
//        -I<dir with FusedRMSRoPE_bf16.cu> -I<dir with FusedRMSRoPE_bf16_v2.cu> -o bench_rope_v2 bench_rope_v2.cu
// usage: CUDA_VISIBLE_DEVICES=<free GPU> ./bench_rope_v2 [--reps 200] [--rounds 9] [--B 1,2,4,8]
// =============================================================================
#include "FusedRMSRoPE_bf16.cu"
#include "FusedRMSRoPE_bf16_v2.cu"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

namespace orig = OwnTensor::cuda;
namespace v2 = OwnTensor::cuda::ropt2;
using bf16 = __nv_bfloat16;

static constexpr int kT = 4096, kHq = 18, kHkv = 6, kHD = 128, kH = kHq + kHkv;
static constexpr float kEps = 1e-6f;

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }
template <typename T> static T* dev(const std::vector<T>& h) {
  T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <typename T> static T* dev_zero(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <typename T> static int64_t diff(const T* a, const T* b, size_t n) {
  std::vector<T> ha(n), hb(n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
  int64_t d = 0; for (size_t i = 0; i < n; ++i) d += std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0; return d;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

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
  size_t s0 = 0;
  while (s0 <= Bs.size()) {
    size_t e = Bs.find(',', s0); if (e == std::string::npos) e = Bs.size();
    const int B = std::atoi(Bs.substr(s0, e - s0).c_str()); s0 = e + 1;
    const size_t nq = (size_t)B * kT * kHq * kHD, nk = (size_t)B * kT * kHkv * kHD;
    const size_t nrq = (size_t)B * kHq * kT, nrk = (size_t)B * kHkv * kT;
    const int rq = B * kHq * BF16_BWD_T_CHUNKS, rk = B * kHkv * BF16_BWD_T_CHUNKS;
    const size_t npq = (size_t)rq * kHD, npk = (size_t)rk * kHD;
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
    // [0] = original, [1] = variant under test
    bf16 *qo[2], *ko[2], *dq[2], *dk[2]; float *qr[2], *kr[2], *qp[2], *kp[2], *dqg[2], *dkg[2];
    for (int v = 0; v < 2; ++v) {
      qo[v] = dev_zero<bf16>(nq); ko[v] = dev_zero<bf16>(nk); dq[v] = dev_zero<bf16>(nq); dk[v] = dev_zero<bf16>(nk);
      qr[v] = dev_zero<float>(nrq); kr[v] = dev_zero<float>(nrk); qp[v] = dev_zero<float>(npq); kp[v] = dev_zero<float>(npk);
      dqg[v] = dev_zero<float>(kHD); dkg[v] = dev_zero<float>(kHD);
    }
    // ---- the original kernels, as their launchers call them ----
    const dim3 fgrid((kT + 31) / 32, kH, B), fblock(256), bgrid(BF16_BWD_T_CHUNKS, kH, B), bblock(32);
    auto fwd_o = [&](const int32_t* p) {
      if (p) orig::fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, true ><<<fgrid, fblock>>>(q, k, qo[0], ko[0], qg, kg, cs, qr[0], kr[0], kHq, kHkv, kT, 0, kEps, p);
      else   orig::fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, false><<<fgrid, fblock>>>(q, k, qo[0], ko[0], qg, kg, cs, qr[0], kr[0], kHq, kHkv, kT, 0, kEps, nullptr);
    };
    auto bwd_o = [&](const int32_t* p) {
      if (p) orig::fused_qknorm_rope_bwd_bf16_tm<true, kHD, true ><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[0], dk[0], qp[0], kp[0], kHq, kHkv, kT, 0, p);
      else   orig::fused_qknorm_rope_bwd_bf16_tm<true, kHD, false><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[0], dk[0], qp[0], kp[0], kHq, kHkv, kT, 0, nullptr);
    };
    auto red_o = [&]() {
      orig::reduce_dgamma_bf16_tm<<<1, 128>>>(qp[0], dqg[0], rq, kHD);
      orig::reduce_dgamma_bf16_tm<<<1, 128>>>(kp[0], dkg[0], rk, kHD);
    };
    // ---- the variants (each writes into set [1]; the backward reads the ORIGINAL forward's rstd, the reduce the
    //      ORIGINAL backward's partials, so every kernel is compared on identical inputs) ----
    struct Var { std::string kind, name; std::function<void(const int32_t*)> run; };
    std::vector<Var> vars;
    auto add_fwd = [&](const char* nm, auto kernel_nopos, auto kernel_pos, int hpw) {
      const long warps = (long)B * kT * ((kH + hpw - 1) / hpw);
      const dim3 grid((unsigned)((warps + 7) / 8));
      vars.push_back({"forward", nm, [=](const int32_t* p) {
        if (p) kernel_pos<<<grid, 256>>>(q, k, qo[1], ko[1], qg, kg, cs, qr[1], kr[1], B, kHq, kHkv, kT, 0, kEps, p);
        else   kernel_nopos<<<grid, 256>>>(q, k, qo[1], ko[1], qg, kg, cs, qr[1], kr[1], B, kHq, kHkv, kT, 0, kEps, nullptr);
      }});
    };
    add_fwd("fwd2 HPW 2",  v2::fwd2<kHD, 2,  false>, v2::fwd2<kHD, 2,  true>, 2);
    add_fwd("fwd2 HPW 4",  v2::fwd2<kHD, 4,  false>, v2::fwd2<kHD, 4,  true>, 4);
    add_fwd("fwd2 HPW 6",  v2::fwd2<kHD, 6,  false>, v2::fwd2<kHD, 6,  true>, 6);
    add_fwd("fwd2 HPW 12", v2::fwd2<kHD, 12, false>, v2::fwd2<kHD, 12, true>, 12);
    add_fwd("fwd2 HPW 24", v2::fwd2<kHD, 24, false>, v2::fwd2<kHD, 24, true>, 24);
    auto add_bwd = [&](const char* nm, auto kernel_nopos, auto kernel_pos, int hpb) {
      const dim3 grid(BF16_BWD_T_CHUNKS, (kH + hpb - 1) / hpb, B);
      vars.push_back({"backward", nm, [=](const int32_t* p) {
        if (p) kernel_pos<<<grid, hpb * 32>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[1], dk[1], qp[1], kp[1], kHq, kHkv, kT, 0, p);
        else   kernel_nopos<<<grid, hpb * 32>>>(dzq, dzk, q, k, qg, kg, cs, qr[0], kr[0], dq[1], dk[1], qp[1], kp[1], kHq, kHkv, kT, 0, nullptr);
      }});
    };
    add_bwd("bwd2 HPB 8 STAGE 8",   v2::bwd2<true, kHD, 8,  8,  false>, v2::bwd2<true, kHD, 8,  8,  true>, 8);
    add_bwd("bwd2 HPB 12 STAGE 8",  v2::bwd2<true, kHD, 12, 8,  false>, v2::bwd2<true, kHD, 12, 8,  true>, 12);
    add_bwd("bwd2 HPB 24 STAGE 8",  v2::bwd2<true, kHD, 24, 8,  false>, v2::bwd2<true, kHD, 24, 8,  true>, 24);
    add_bwd("bwd2 HPB 24 STAGE 16", v2::bwd2<true, kHD, 24, 16, false>, v2::bwd2<true, kHD, 24, 16, true>, 24);
    auto add_red = [&](const char* nm, auto kernel) {
      vars.push_back({"reduce", nm, [=](const int32_t*) {
        kernel<<<dim3((kHD + 31) / 32, 2), 32>>>(qp[0], dqg[1], rq, kp[0], dkg[1], rk, kHD);
      }});
    };
    add_red("reduce3 UNROLL 32",  v2::reduce3<32>);
    add_red("reduce3 UNROLL 64",  v2::reduce3<64>);
    add_red("reduce3 UNROLL 128", v2::reduce3<128>);

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto round_us = [&](auto&& body) {
      CK(cudaEventRecord(e0)); for (int i = 0; i < reps; ++i) body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return 1000.0f * ms / reps;
    };
    { const auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(2)) { fwd_o(pos); bwd_o(pos); red_o(); for (auto& v : vars) v.run(pos); CK(cudaDeviceSynchronize()); } }

    std::printf("B=%d  (time: original | variant | speed-up; CHECK = all outputs of that kernel bit-identical)\n", B);
    for (int withpos = 0; withpos < 2; ++withpos) {
      const int32_t* p = withpos ? pos : nullptr;
      std::printf("  -- %s --\n", withpos ? "packed positions (SFT)" : "no positions (pretraining)");
      fwd_o(p); bwd_o(p); red_o(); CK(cudaDeviceSynchronize());     // set [0] = the reference outputs of this case
      for (auto& v : vars) {
        v.run(p); CK(cudaGetLastError());
        int64_t bad = 0;
        if (v.kind == "forward")  bad = diff(qo[0], qo[1], nq) + diff(ko[0], ko[1], nk) + diff(qr[0], qr[1], nrq) + diff(kr[0], kr[1], nrk);
        if (v.kind == "backward") bad = diff(dq[0], dq[1], nq) + diff(dk[0], dk[1], nk) + diff(qp[0], qp[1], npq) + diff(kp[0], kp[1], npk);
        if (v.kind == "reduce")   bad = diff(dqg[0], dqg[1], kHD) + diff(dkg[0], dkg[1], kHD);
        std::vector<float> to, tn;
        for (int r = 0; r < rounds; ++r) {
          if (v.kind == "forward")  to.push_back(round_us([&] { fwd_o(p); }));
          if (v.kind == "backward") to.push_back(round_us([&] { bwd_o(p); }));
          if (v.kind == "reduce")   to.push_back(round_us([&] { red_o(); }));
          tn.push_back(round_us([&] { v.run(p); }));
        }
        const float mo = median(to), mn = median(tn);
        std::printf("  %-9s %-22s %8.1f | %8.1f us | %5.2fx  CHECK %s\n", v.kind.c_str(), v.name.c_str(), mo, mn, mo / mn,
                    bad ? ("FAIL (" + std::to_string(bad) + " elements differ)").c_str() : "bit-identical");
        all_ok = all_ok && bad == 0;
      }
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    for (int v = 0; v < 2; ++v)
      for (void* x : std::initializer_list<void*>{qo[v], ko[v], dq[v], dk[v], qr[v], kr[v], qp[v], kp[v], dqg[v], dkg[v]}) CK(cudaFree(x));
    for (void* x : std::initializer_list<void*>{q, k, dzq, dzk, qg, kg, cs, pos}) CK(cudaFree(x));
    if (e >= Bs.size()) break;
  }
  std::printf("%s\n", all_ok ? "RESULT: every variant bit-identical to the original; timings above"
                             : "RESULT: some variants FAIL (see CHECK); their timings do not count");
  return all_ok ? 0 : 3;
}
