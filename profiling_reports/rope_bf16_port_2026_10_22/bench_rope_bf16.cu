// =============================================================================
// bench_rope_bf16.cu  (2026-10-22): kernel-only timing of the bf16 fused QK-RMSNorm + RoPE (Kruthik's kernels with the
// stage-B per-chat positions), EACH KERNEL ON ITS OWN, without and with positions:
//   fwd   fused_qknorm_rope_fwd_bf16_tm   grid (T/32, Hq+Hkv, B) x 256 threads
//   bwd   fused_qknorm_rope_bwd_bf16_tm   grid (32, Hq+Hkv, B) x 32 threads (one warp walks T/32 tokens)
//   red   reduce_dgamma_bf16_tm x 2       ONE block of 128 threads each (q and k dgamma partials)
// The stage-B test timed forward + backward together through the op layer (allocations and checks included), so it
// could not say where the positions cost is. This one times the kernels alone: CUDA events around `reps` back-to-back
// launches = mean per launch; `rounds` such rounds, alternating no-positions / positions, report the median.
// It also prints the achieved DRAM bandwidth (bytes the kernel must move / time) against the H200 peak (4.8 TB/s).
// Sanity check first: identity positions must give the same bits as no positions (forward and backward).
//
// It compiles the kernel file itself (no libtensor), so it never touches a library a run is using:
//   nvcc -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a -I<Tensor-Implementations>/include -I<dir holding
//        FusedRMSRoPE_bf16.cu> -o bench_rope_bf16 bench_rope_bf16.cu
// usage: CUDA_VISIBLE_DEVICES=<free GPU> ./bench_rope_bf16 [--reps 200] [--rounds 9]
// Shape: T 4096, 18 query / 6 KV heads, head dim 128, NeoX, eps 1e-6 (the SFT model); B = 4 and 8.
// =============================================================================
#include "FusedRMSRoPE_bf16.cu"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

using namespace OwnTensor::cuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int kT = 4096, kHq = 18, kHkv = 6, kHD = 128, kH = kHq + kHkv;
static constexpr float kEps = 1e-6f, kPeakGBs = 4800.0f;   // H200 HBM3e peak

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }

template <typename T> static T* dev(const std::vector<T>& h) {
  T* d; CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <typename T> static T* dev_empty(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0, n * sizeof(T))); return d; }
template <typename T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  int reps = 200, rounds = 9;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--reps") reps = std::atoi(argv[i + 1]);
    if (std::string(argv[i]) == "--rounds") rounds = std::atoi(argv[i + 1]);
  }
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s (%d SMs) | T %d, %d / %d heads, hd %d, NeoX | %d launches per round, median of %d rounds\n",
              prop.name, prop.multiProcessorCount, kT, kHq, kHkv, kHD, reps, rounds);
  bool sane = true;

  for (int B : {4, 8}) {
    const size_t nq = (size_t)B * kT * kHq * kHD, nk = (size_t)B * kT * kHkv * kHD;
    uint64_t s = 0x9E3779B97F4A7C15ull + B;
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    auto rnd = [&]() { return (float)((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f; };
    auto bf_vec = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) x = f2bf(rnd()); return v; };
    std::vector<float> hg(kHD), hcs((size_t)kT * kHD);
    for (auto& x : hg) x = 1.0f + 0.1f * rnd();
    for (auto& x : hcs) x = rnd();                       // timing and bit-equality do not depend on the cache values
    // packed positions as SftDataLoader builds them: chats of 1..1500 tokens, each from 0, then the pad tail from 0
    std::vector<int32_t> hpos((size_t)B * kT), hid((size_t)B * kT);
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < kT;) { int len = 1 + (int)(next() % 1500); if (t + len > kT) len = kT - t;
        for (int i = 0; i < len; ++i) hpos[(size_t)b * kT + t + i] = i; t += len; }
      for (int t = 0; t < kT; ++t) hid[(size_t)b * kT + t] = t;
    }
    bf16 *q = (bf16*)dev(bf_vec(nq)), *k = (bf16*)dev(bf_vec(nk)), *dzq = (bf16*)dev(bf_vec(nq)), *dzk = (bf16*)dev(bf_vec(nk));
    bf16 *qo = dev_empty<bf16>(nq), *ko = dev_empty<bf16>(nk), *dq = dev_empty<bf16>(nq), *dk = dev_empty<bf16>(nk);
    float *qg = dev(hg), *kg = dev(hg), *cs = dev(hcs);
    float *qr = dev_empty<float>((size_t)B * kHq * kT), *kr = dev_empty<float>((size_t)B * kHkv * kT);
    float *qpart = dev_empty<float>((size_t)B * kHq * BF16_BWD_T_CHUNKS * kHD), *kpart = dev_empty<float>((size_t)B * kHkv * BF16_BWD_T_CHUNKS * kHD);
    float *dqg = dev_empty<float>(kHD), *dkg = dev_empty<float>(kHD);
    int32_t *pos = dev(hpos), *ident = dev(hid);

    const dim3 fgrid((kT + 31) / 32, kH, B), fblock(256), bgrid(BF16_BWD_T_CHUNKS, kH, B), bblock(32);
    auto fwd = [&](const int32_t* p) {
      if (p) fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, true ><<<fgrid, fblock>>>(q, k, qo, ko, qg, kg, cs, qr, kr, kHq, kHkv, kT, 0, kEps, p);
      else   fused_qknorm_rope_fwd_bf16_tm<true, kHD, 4, 8, false><<<fgrid, fblock>>>(q, k, qo, ko, qg, kg, cs, qr, kr, kHq, kHkv, kT, 0, kEps, nullptr);
    };
    auto bwd = [&](const int32_t* p) {
      if (p) fused_qknorm_rope_bwd_bf16_tm<true, kHD, true ><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr, kr, dq, dk, qpart, kpart, kHq, kHkv, kT, 0, p);
      else   fused_qknorm_rope_bwd_bf16_tm<true, kHD, false><<<bgrid, bblock>>>(dzq, dzk, q, k, qg, kg, cs, qr, kr, dq, dk, qpart, kpart, kHq, kHkv, kT, 0, nullptr);
    };
    auto red = [&]() {
      reduce_dgamma_bf16_tm<<<1, 128>>>(qpart, dqg, B * kHq * BF16_BWD_T_CHUNKS, kHD);
      reduce_dgamma_bf16_tm<<<1, 128>>>(kpart, dkg, B * kHkv * BF16_BWD_T_CHUNKS, kHD);
    };

    // ---- sanity: identity positions == no positions, forward and backward, bit for bit ----
    fwd(nullptr); bwd(nullptr); red();
    auto f0 = host((uint16_t*)qo, nq); auto b0 = host((uint16_t*)dq, nq); auto g0 = host(dqg, kHD);
    fwd(ident); bwd(ident); red();
    auto f1 = host((uint16_t*)qo, nq); auto b1 = host((uint16_t*)dq, nq); auto g1 = host(dqg, kHD);
    const bool ok = f0 == f1 && b0 == b1 && std::memcmp(g0.data(), g1.data(), kHD * 4) == 0;
    std::printf("SANITY B=%d identity positions vs none: %s\n", B, ok ? "bit-identical (q_out, dq, dgamma_q)" : "DIFFERENT  <-- FAIL");
    sane = sane && ok;
    fwd(pos); bwd(pos); red(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // ---- timing ----
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto round_ms = [&](auto&& body) {
      for (int i = 0; i < 10; ++i) body();
      CK(cudaEventRecord(e0)); for (int i = 0; i < reps; ++i) body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / reps;
    };
    // warm-up: ~2 s of all kernels first, so the GPU clocks have settled before the first measured round
    { const auto w0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - w0 < std::chrono::seconds(2)) { fwd(nullptr); fwd(pos); bwd(nullptr); bwd(pos); red(); CK(cudaDeviceSynchronize()); } }
    std::vector<float> f_no, f_pos, b_no, b_pos, r_all;
    for (int r = 0; r < rounds; ++r) {           // alternate, so clock drift hits both sides alike
      f_no.push_back(round_ms([&] { fwd(nullptr); })); f_pos.push_back(round_ms([&] { fwd(pos); }));
      b_no.push_back(round_ms([&] { bwd(nullptr); })); b_pos.push_back(round_ms([&] { bwd(pos); }));
      r_all.push_back(round_ms([&] { red(); }));
    }
    CK(cudaGetLastError());
    const double act = (double)B * kT * kH * kHD * 2;          // bytes of q+k (bf16) for all tokens and heads
    const double stat = (double)B * kT * kH * 4;               // rstd bytes
    const double fbytes = 2 * act + stat;                      // fwd: read q,k + write q,k + write rstd
    const double bbytes = 3 * act + stat + (double)B * kH * BF16_BWD_T_CHUNKS * kHD * 4;   // read dz,q,k + write dq,dk + rstd + partials
    auto row = [&](const char* nm, float no, float po, double bytes) {
      std::printf("  %-28s | no positions %8.1f us (%5.0f GB/s, %3.0f%% of peak) | positions %8.1f us | %+7.1f us (%+5.1f%%)\n",
                  nm, 1e3 * no, bytes / (no * 1e6), 100.0 * bytes / (no * 1e6) / kPeakGBs, 1e3 * po, 1e3 * (po - no), 100.0 * (po - no) / no);
    };
    const float fn = median(f_no), fp = median(f_pos), bn = median(b_no), bp = median(b_pos), rr = median(r_all);
    std::printf("B=%d\n", B);
    row("forward kernel", fn, fp, fbytes);
    row("backward kernel", bn, bp, bbytes);
    std::printf("  %-28s | %8.1f us for both (no positions involved; ONE block of 128 threads each, %d + %d rows)\n",
                "dgamma reduce (q + k)", 1e3 * rr, B * kHq * BF16_BWD_T_CHUNKS, B * kHkv * BF16_BWD_T_CHUNKS);
    std::printf("  %-28s | no positions %8.1f us | positions %8.1f us | %+7.1f us (%+5.1f%%)\n", "total fwd + bwd + reduce",
                1e3 * (fn + bn + rr), 1e3 * (fp + bp + rr), 1e3 * (fp + bp - fn - bn), 100.0 * (fp + bp - fn - bn) / (fn + bn + rr));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    for (void* p : std::initializer_list<void*>{q, k, dzq, dzk, qo, ko, dq, dk, qg, kg, cs, qr, kr, qpart, kpart, dqg, dkg, pos, ident}) CK(cudaFree(p));
  }
  std::printf("%s\n", sane ? "RESULT: sanity checks pass; timings above" : "RESULT: SANITY FAIL (timings not trustworthy)");
  return sane ? 0 : 3;
}
