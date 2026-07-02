// test_bgemm_sm103_bf16_nn_cluster.cu
// Benchmark and accuracy test: BluBridge SM103a BF16 batched NN Cluster GEMM vs
// cuBLAS. Target: NVIDIA B300 (SM 10.3a) — tcgen05.mma.cta_group::2.kind::f16
//
// Usage:
//   ./build/test_bgemm_sm103_bf16_nn_cluster
//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <numeric>
#include <string>
#include <vector>

#include "../src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_bf16.cuh"

// ─── error helpers
// ────────────────────────────────────────────────────────────

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t e = (call);                                                 \
    if (e != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)e, __FILE__,          \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ─── data init
// ────────────────────────────────────────────────────────────────

static void fill_rand_bf16(std::vector<__nv_bfloat16> &v, unsigned seed) {
  srand(seed);
  for (auto &x : v)
    x = __float2bfloat16((float(rand()) / RAND_MAX) * 2.f - 1.f);
}

// ─── accuracy check
// ───────────────────────────────────────────────────────────

static double max_rel_err(const std::vector<__nv_bfloat16> &ref,
                          const std::vector<__nv_bfloat16> &out) {
  double max_e = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double r = __bfloat162float(ref[i]);
    double o = __bfloat162float(out[i]);
    double e = std::abs(r - o) / std::max(1e-5, std::abs(r));
    max_e = std::max(max_e, e);
  }
  return max_e;
}

// ─── benchmark harness
// ────────────────────────────────────────────────────────

struct Result {
  double blublas_tflops;
  double cublas_tflops;
  double rel_err;
  bool pass;
};

static Result run_test(int M, int N, int K, int batchCount, float alpha,
                       float beta, bool use_256_tile, cublasHandle_t cublas,
                       cudaStream_t stream) {
  const long long elA = (long long)batchCount * M * K;
  const long long elB = (long long)batchCount * K * N;
  const long long elC = (long long)batchCount * M * N;

  std::vector<__nv_bfloat16> hA(elA), hB(elB),
      hC_ref(elC, __float2bfloat16(0.f));

  fill_rand_bf16(hA, 42);
  fill_rand_bf16(hB, 99);

  __nv_bfloat16 *dA, *dB, *dC_ref, *dC_blu;
  CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dC_blu, elC * sizeof(__nv_bfloat16)));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC_ref, 0, elC * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(dC_blu, 0, elC * sizeof(__nv_bfloat16)));

  // ── cuBLAS reference (batched NN sgemm) ────────
  {
    float fa = alpha, fb = beta;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &fa, dB, CUDA_R_16BF, N,
        (long long)K * N, dA, CUDA_R_16BF, K, (long long)M * K, &fb, dC_ref,
        CUDA_R_16BF, N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Warmup BluBridge (5 iters)
  for (int w = 0; w < 5; ++w) {
    if (use_256_tile)
      mycublasBgemmSM103_bf16_nn_cluster_256x256x64(
          nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
    else
      mycublasBgemmSM103_bf16_nn_cluster_128x128x64(
          nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());

  // Timed BluBridge (20 iters)
  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < 20; ++i) {
    if (use_256_tile)
      mycublasBgemmSM103_bf16_nn_cluster_256x256x64(
          nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
    else
      mycublasBgemmSM103_bf16_nn_cluster_128x128x64(
          nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
  }
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaEventSynchronize(t1));
  CHECK_CUDA(cudaGetLastError());
  float blu_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&blu_ms, t0, t1));

  // Warmup + timed cuBLAS (20 iters)
  for (int w = 0; w < 5; ++w) {
    float fa = alpha, fb = beta;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &fa, dB, CUDA_R_16BF, N,
        (long long)K * N, dA, CUDA_R_16BF, K, (long long)M * K, &fb, dC_ref,
        CUDA_R_16BF, N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < 20; ++i) {
    float fa = alpha, fb = beta;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &fa, dB, CUDA_R_16BF, N,
        (long long)K * N, dA, CUDA_R_16BF, K, (long long)M * K, &fb, dC_ref,
        CUDA_R_16BF, N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float cub_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&cub_ms, t0, t1));

  // Accuracy
  std::vector<__nv_bfloat16> hC_ref_h(elC), hC_blu_h(elC);
  CHECK_CUDA(cudaMemcpy(hC_ref_h.data(), dC_ref, elC * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hC_blu_h.data(), dC_blu, elC * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  double rel_err = max_rel_err(hC_ref_h, hC_blu_h);

  double flops = 2.0 * (double)M * N * K * batchCount;
  double iters = 20.0;
  double blu_tfl = flops * iters / (blu_ms * 1e-3) * 1e-12;
  double cub_tfl = flops * iters / (cub_ms * 1e-3) * 1e-12;

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC_ref));
  CHECK_CUDA(cudaFree(dC_blu));
  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));

  return {blu_tfl, cub_tfl, rel_err, rel_err < 0.05};
}

// ─── cuBLAS-only profile runner (for nsys kernel-name capture) ───
// Skips BluBLAS entirely — used to sample shapes BluBLAS doesn't handle
// (skinny GEMV, tiny K, split-K territory) so nsys sees the kernels
// cuBLAS picks. Runs 20 timed iters of the NN cuBLAS call.
static double run_cublas_only_nn(int M, int N, int K, int batchCount,
                                 cublasHandle_t cublas, cudaStream_t stream) {
  const long long elA = (long long)batchCount * M * K;
  const long long elB = (long long)batchCount * K * N;
  const long long elC = (long long)batchCount * M * N;

  std::vector<__nv_bfloat16> hA(elA), hB(elB);
  fill_rand_bf16(hA, 42);
  fill_rand_bf16(hB, 99);

  __nv_bfloat16 *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dC, elC * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, elC * sizeof(__nv_bfloat16)));

  float fa = 1.0f, fb = 0.0f;
  for (int w = 0; w < 5; ++w) {
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &fa, dB, CUDA_R_16BF, N,
        (long long)K * N, dA, CUDA_R_16BF, K, (long long)M * K, &fb, dC,
        CUDA_R_16BF, N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < 20; ++i) {
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &fa, dB, CUDA_R_16BF, N,
        (long long)K * N, dA, CUDA_R_16BF, K, (long long)M * K, &fb, dC,
        CUDA_R_16BF, N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float cub_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&cub_ms, t0, t1));

  double flops = 2.0 * (double)M * N * K * batchCount;
  double cub_tfl = flops * 20.0 / (cub_ms * 1e-3) * 1e-12;

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));
  return cub_tfl;
}

int main() {
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

  printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

  if (prop.major != 10 || prop.minor != 3) {
    printf(
        "[SKIP] SM103 BF16 NN Cluster kernel requires SM 10.3 (B300/sm_103a).\n"
        "       This device is SM %d.%d (%s).\n"
        "       Tests skipped.\n",
        prop.major, prop.minor, prop.name);
    return EXIT_SUCCESS;
  }

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cublasHandle_t cublas;
  CHECK_CUBLAS(cublasCreate(&cublas));
  CHECK_CUBLAS(cublasSetStream(cublas, stream));

  struct Shape {
    int M, N, K, batch;
    const char *label;
  };

  // --- 128x128x64 Cluster Configurations ---
  const std::vector<Shape> shapes_128 = {{128, 128, 64, 1, "tiny"},
                                         {256, 256, 128, 1, "small"},
                                         {512, 512, 128, 1, "medium"},
                                         {1024, 1024, 128, 1, "large"},
                                         {2048, 2048, 256, 1, "xlarge"},
                                         {4096, 4096, 512, 1, "llm-style"},
                                         {1024, 1024, 1024, 1, "square-1024"},
                                         {2048, 2048, 2048, 1, "square-2048"},
                                         {4096, 4096, 4096, 1, "square-4096"}};

  // --- 256x256x64 Cluster Configurations (Requires M/N multiples of 256) ---
  const std::vector<Shape> shapes_256 = {
      {256, 256, 128, 1, "small-256"},
      {512, 512, 128, 1, "medium-256"},
      {1024, 1024, 128, 1, "large-256"},
      {2048, 2048, 256, 1, "xlarge-256"},
      {4096, 4096, 512, 1, "llm-style-256"},
      {2048, 2048, 2048, 1, "square-2048-256"},
      {4096, 4096, 4096, 1, "square-4096-256"}};

  printf("\n=== Testing 128x128x64 Cluster Kernel (cta_group::2) ===\n");
  printf("%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %s\n", "Shape", "M", "N",
         "K", "Batch", "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
  printf("%s\n", std::string(85, '-').c_str());

  int passed = 0, total = (int)shapes_128.size();
  for (const auto &s : shapes_128) {
    Result r =
        run_test(s.M, s.N, s.K, s.batch, 1.0f, 0.0f, false, cublas, stream);
    printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %s\n", s.label, s.M,
           s.N, s.K, s.batch, r.blublas_tflops, r.cublas_tflops, r.rel_err,
           r.pass ? "PASS" : "FAIL");
    if (r.pass)
      ++passed;
  }
  printf("128x128x64 Results: %d / %d PASSED\n", passed, total);

  printf("\n=== Testing 256x256x64 Cluster Kernel (cta_group::2) ===\n");
  printf("%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %s\n", "Shape", "M", "N",
         "K", "Batch", "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
  printf("%s\n", std::string(85, '-').c_str());

  int passed_256 = 0, total_256 = (int)shapes_256.size();
  for (const auto &s : shapes_256) {
    Result r =
        run_test(s.M, s.N, s.K, s.batch, 1.0f, 0.0f, true, cublas, stream);
    printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %s\n", s.label, s.M,
           s.N, s.K, s.batch, r.blublas_tflops, r.cublas_tflops, r.rel_err,
           r.pass ? "PASS" : "FAIL");
    if (r.pass)
      ++passed_256;
  }
  printf("256x256x64 Results: %d / %d PASSED\n\n", passed_256, total_256);

  // ─── cuBLAS-only Profile Sweep (for nsys kernel-name capture) ───
  // These shapes are outside BluBLAS's alignment envelope. We run ONLY
  // cuBLAS on them so nsys can capture the kernel cuBLAS picks per shape.
  // Targets: skinny GEMV, small-batch decode, tiny-K, split-K, unaligned.
  const std::vector<Shape> shapes_profile = {
      {1, 4096, 4096, 1, "decode-1"},   // GEMV: expect gemv2N/T
      {8, 4096, 4096, 1, "decode-8"},   // small M: expect gemmSN_NN
      {32, 4096, 4096, 1, "decode-32"}, // small M: expect gemmSN_NN
      {64, 4096, 4096, 1, "decode-64"}, // SN/full-tile boundary
      {1, 32000, 4096, 1, "lm-head-1"}, // GEMV huge N: LM logits
      {1024, 1024, 32, 1, "tiny-K-32"}, // small K: expect gemmk1
      {64, 64, 16384, 1, "split-K"},    // fat K: expect splitKreduce
      {129, 129, 128, 1, "unaligned"},  // 128+1 edge (M,N off-tile)
  };
  printf(
      "\n=== cuBLAS-only Profile Sweep NN (nsys capture — no BluBLAS) ===\n");
  printf("%-20s %6s %6s %6s %5s  %-8s\n", "Shape", "M", "N", "K", "Batch",
         "cuBLAS");
  printf("%s\n", std::string(65, '-').c_str());
  for (const auto &s : shapes_profile) {
    double tfl = run_cublas_only_nn(s.M, s.N, s.K, s.batch, cublas, stream);
    printf("%-20s %6d %6d %6d %5d  %7.2f T\n", s.label, s.M, s.N, s.K, s.batch,
           tfl);
  }
  printf("\n");

  // ─── Qwen Sweep — actual matmul shapes from Qwen 2.5 + 3 (M=4096) ───
  // Configs verified from HuggingFace on 2026-07-01.
  const std::vector<Shape> shapes_qwen = {
      // Qwen 2.5-0.5B  hidden=896   KV_dim=128   I=4864   vocab=151936
      {4096, 896, 896, 1, "q25-0.5b-Q"},
      {4096, 128, 896, 1, "q25-0.5b-K"},
      {4096, 4864, 896, 1, "q25-0.5b-Gate"},
      {4096, 896, 4864, 1, "q25-0.5b-Down"},
      {4096, 151936, 896, 1, "q25-0.5b-LMhead"},
      // Qwen 2.5-1.5B  hidden=1536  KV_dim=256   I=8960
      {4096, 1536, 1536, 1, "q25-1.5b-Q"},
      {4096, 256, 1536, 1, "q25-1.5b-K"},
      {4096, 8960, 1536, 1, "q25-1.5b-Gate"},
      {4096, 1536, 8960, 1, "q25-1.5b-Down"},
      {4096, 151936, 1536, 1, "q25-1.5b-LMhead"},
      // Qwen 2.5-3B   hidden=2048  KV_dim=256   I=11008
      {4096, 2048, 2048, 1, "q25-3b-Q"},
      {4096, 256, 2048, 1, "q25-3b-K"},
      {4096, 11008, 2048, 1, "q25-3b-Gate"},
      {4096, 2048, 11008, 1, "q25-3b-Down"},
      {4096, 151936, 2048, 1, "q25-3b-LMhead"},
      // Qwen 2.5-7B   hidden=3584  KV_dim=512   I=18944  vocab=152064
      {4096, 3584, 3584, 1, "q25-7b-Q"},
      {4096, 512, 3584, 1, "q25-7b-K"},
      {4096, 18944, 3584, 1, "q25-7b-Gate"},
      {4096, 3584, 18944, 1, "q25-7b-Down"},
      {4096, 152064, 3584, 1, "q25-7b-LMhead"},
      // Qwen 3-0.6B   hidden=1024  Q_dim=2048   KV_dim=1024  I=3072 NON-SQUARE
      // Q/O
      {4096, 2048, 1024, 1, "q3-0.6b-Q"},
      {4096, 1024, 1024, 1, "q3-0.6b-K"},
      {4096, 1024, 2048, 1, "q3-0.6b-O"},
      {4096, 3072, 1024, 1, "q3-0.6b-Gate"},
      {4096, 1024, 3072, 1, "q3-0.6b-Down"},
      {4096, 151936, 1024, 1, "q3-0.6b-LMhead"},
      // Qwen 3-1.7B   hidden=2048  Q_dim=2048   KV_dim=1024  I=6144
      {4096, 2048, 2048, 1, "q3-1.7b-Q"},
      {4096, 1024, 2048, 1, "q3-1.7b-K"},
      {4096, 6144, 2048, 1, "q3-1.7b-Gate"},
      {4096, 2048, 6144, 1, "q3-1.7b-Down"},
      {4096, 151936, 2048, 1, "q3-1.7b-LMhead"},
      // Qwen 3-4B     hidden=2560  Q_dim=4096   KV_dim=1024  I=9728 NON-SQUARE
      // Q/O
      {4096, 4096, 2560, 1, "q3-4b-Q"},
      {4096, 1024, 2560, 1, "q3-4b-K"},
      {4096, 2560, 4096, 1, "q3-4b-O"},
      {4096, 9728, 2560, 1, "q3-4b-Gate"},
      {4096, 2560, 9728, 1, "q3-4b-Down"},
      {4096, 151936, 2560, 1, "q3-4b-LMhead"},
      // Qwen 3-8B     hidden=4096  Q_dim=4096   KV_dim=1024  I=12288
      {4096, 4096, 4096, 1, "q3-8b-Q"},
      {4096, 1024, 4096, 1, "q3-8b-K"},
      {4096, 12288, 4096, 1, "q3-8b-Gate"},
      {4096, 4096, 12288, 1, "q3-8b-Down"},
      {4096, 151936, 4096, 1, "q3-8b-LMhead"},
  };
  printf(
      "=== BluBLAS vs cuBLAS Qwen Sweep NN (M=4096, 8 models × ~5 ops) ===\n");
  printf("%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %-6s  %s\n", "Shape", "M",
         "N", "K", "Batch", "BluBLAS", "cuBLAS", "MaxRelErr", "Ratio",
         "Status");
  printf("%s\n", std::string(95, '-').c_str());
  int qwen_pass = 0, qwen_total = (int)shapes_qwen.size();
  for (const auto &s : shapes_qwen) {
    Result r =
        run_test(s.M, s.N, s.K, s.batch, 1.0f, 0.0f, false, cublas, stream);
    double ratio = r.cublas_tflops > 0 ? r.blublas_tflops / r.cublas_tflops : 0;
    printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %5.1f%%  %s\n",
           s.label, s.M, s.N, s.K, s.batch, r.blublas_tflops, r.cublas_tflops,
           r.rel_err, ratio * 100.0, r.pass ? "PASS" : "FAIL");
    if (r.pass)
      ++qwen_pass;
  }
  printf("Qwen Sweep NN Results: %d / %d PASSED\n\n", qwen_pass, qwen_total);

  CHECK_CUBLAS(cublasDestroy(cublas));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return (passed == total && passed_256 == total_256) ? EXIT_SUCCESS
                                                      : EXIT_FAILURE;
}
