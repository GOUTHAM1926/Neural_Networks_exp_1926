// =============================================================================
// bench_attn_ab.cu  (2026-10-21)
//
// OLD vs NEW library A/B for the pretraining branch dist_side_gautam_lottn, attention part: the causal GQA forward and
// backward at the production shape (train/production_zero/bluscript_zero_ckpt_config.yaml: B 8, 18 query / 6 KV heads,
// T 4096, head_dim 128), through the raw library calls both libraries have:
//   cuda::gqa_flash_attn_forward_qkv   OLD: v62 as in BLUCCL (GQA_fwd_sm90.cu dbce3e46)   NEW: QPU (7e4197ea)
//   cuda::gqa_flash_attn_backward_qkv  OLD: Vj1d (the default before 2026-10-15)          NEW: Vj2 (the default since)
// NEW's calls have one more parameter (const GqaChatPlan* chats = nullptr), so the symbols differ: the SAME source is built
// twice, once against each tree. Correctness is NOT judged here (test_varlen_attention.cu does that against fp64 truth);
// this prints the time and checksums of O, LSE, dQ, dK, dV (O, LSE, dK, dV must repeat bit for bit inside a run).
//   timing: --warmup calls, then --iters calls of F, then of B, CUDA events per call: median, min, and TFLOP/s by FA's
//   formula (causal fwd = B*Hq*2*T*(T/2)*(2*hd), bwd = 2.5 x fwd: hopper/benchmark_attn.py:62-73, :387).
// usage: bench_attn_ab [--warmup N] [--iters N] [--B b] [--T t]
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dtype/Types.h"
#include "ops/helpers/AttentionKernels.h"

using OwnTensor::bfloat16_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

// fixed pseudo-random bf16 values, about N(0, 1) (sum of 4 uniforms), a hash of (seed, index)
__global__ void fill_bf16(uint16_t *p, int64_t n, uint32_t seed) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    float s = 0.f;
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ (seed * 4u + j) * 2246822519u ^ static_cast<uint32_t>(i >> 32);
      x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
      s += static_cast<float>(x >> 8) * (1.0f / 16777216.0f) - 0.5f;
    }
    p[i] = __bfloat16_as_ushort(__float2bfloat16_rn(s * 1.7320508f));   // 4 uniforms on [-0.5, 0.5): variance 1/3
  }
}
static uint64_t checksum_dev(const void *p, size_t bytes) {   // sum of 32-bit words x (index | 1), mod 2^64
  std::vector<uint32_t> h(bytes / 4);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
  uint64_t acc = 0;
  for (size_t i = 0; i < h.size(); ++i) acc += (uint64_t)h[i] * (uint64_t)(i | 1);
  return acc;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char **argv) {
  int warmup = 20, iters = 200, B = 8, T = 4096;
  const int Hq = 18, Hkv = 6, hd = 128;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--warmup") warmup = std::atoi(argv[i + 1]);
    else if (a == "--iters") iters = std::atoi(argv[i + 1]);
    else if (a == "--B") B = std::atoi(argv[i + 1]);
    else if (a == "--T") T = std::atoi(argv[i + 1]);
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
  }
  const size_t nq = (size_t)B * Hq * T * hd, nkv = (size_t)B * Hkv * T * hd, nl = (size_t)B * Hq * T;
  uint16_t *q, *k, *v, *o, *dout, *dq, *dk, *dv; float *lse;
  CK(cudaMalloc(&q, nq * 2)); CK(cudaMalloc(&k, nkv * 2)); CK(cudaMalloc(&v, nkv * 2)); CK(cudaMalloc(&o, nq * 2));
  CK(cudaMalloc(&dout, nq * 2)); CK(cudaMalloc(&dq, nq * 2)); CK(cudaMalloc(&dk, nkv * 2)); CK(cudaMalloc(&dv, nkv * 2));
  CK(cudaMalloc(&lse, nl * 4));
  fill_bf16<<<2048, 256>>>(q, nq, 1); fill_bf16<<<2048, 256>>>(k, nkv, 2); fill_bf16<<<2048, 256>>>(v, nkv, 3);
  fill_bf16<<<2048, 256>>>(dout, nq, 4);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  auto bf = [](uint16_t *p) { return reinterpret_cast<bfloat16_t *>(p); };
  auto F = [&] { OwnTensor::cuda::gqa_flash_attn_forward_qkv(bf(q), bf(k), bf(v), bf(o), lse, B, Hq, Hkv, T, hd, true); };
  auto Bw = [&] { OwnTensor::cuda::gqa_flash_attn_backward_qkv(bf(q), bf(k), bf(v), bf(o), bf(dout), lse, bf(dq), bf(dk), bf(dv),
                                                                B, Hq, Hkv, T, hd, true); };
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | B %d Hq %d Hkv %d T %d hd %d causal | warmup %d iters %d\n", prop.name, B, Hq, Hkv, T, hd, warmup, iters);

  F(); Bw(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
  uint64_t cs[2][5];
  for (int r = 0; r < 2; ++r) {   // two passes: O, LSE, dK, dV must repeat bit for bit (dQ is reduced with atomics)
    F(); Bw();
    cs[r][0] = checksum_dev(o, nq * 2); cs[r][1] = checksum_dev(lse, nl * 4); cs[r][2] = checksum_dev(dq, nq * 2);
    cs[r][3] = checksum_dev(dk, nkv * 2); cs[r][4] = checksum_dev(dv, nkv * 2);
  }
  for (int i = 0; i < warmup; ++i) { F(); Bw(); }
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> tf, tb;
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(a)); F(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); tf.push_back(ms);
  }
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(a)); Bw(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); tb.push_back(ms);
  }
  CK(cudaGetLastError());
  const double fwd_flop = (double)B * Hq * 2.0 * T * (T / 2.0) * (2.0 * hd), bwd_flop = 2.5 * fwd_flop;
  const float fm = median(tf), bm = median(tb);
  std::printf("ATTN F_med_ms=%.4f F_min_ms=%.4f F_tflops=%.1f B_med_ms=%.4f B_min_ms=%.4f B_tflops=%.1f "
              "o_cs=%016llx lse_cs=%016llx dq_cs=%016llx dk_cs=%016llx dv_cs=%016llx repeat_o_lse_dk_dv=%d repeat_dq=%d\n",
              fm, *std::min_element(tf.begin(), tf.end()), fwd_flop / (fm * 1e9), bm, *std::min_element(tb.begin(), tb.end()),
              bwd_flop / (bm * 1e9), (unsigned long long)cs[0][0], (unsigned long long)cs[0][1], (unsigned long long)cs[0][2],
              (unsigned long long)cs[0][3], (unsigned long long)cs[0][4],
              (cs[0][0] == cs[1][0] && cs[0][1] == cs[1][1] && cs[0][3] == cs[1][3] && cs[0][4] == cs[1][4]) ? 1 : 0,
              cs[0][2] == cs[1][2] ? 1 : 0);
  return 0;
}
