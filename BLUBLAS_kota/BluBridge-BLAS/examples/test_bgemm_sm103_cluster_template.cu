// test_bgemm_sm103_cluster_template.cu
// Comprehensive accuracy and benchmark test suite for the unified SM103a
// Cluster GEMM template. Target: NVIDIA B300 (SM 10.3a) —
// tcgen05.mma.cta_group::2.kind::f16
//
// Usage:
//   make run TEST=examples/test_bgemm_sm103_cluster_template.cu
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <numeric>
#include <string>
#include <vector>

// Include the unified SM103 cluster template
#include "../src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh"

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

// Worst-element info from the last max_rel_err call (printed on FAIL so we
// can SEE whether a failure is real corruption or a near-zero cancellation
// artifact).
static size_t g_err_idx = 0;
static float g_err_ref = 0.f, g_err_out = 0.f;

static double max_rel_err(const std::vector<__nv_bfloat16> &ref,
                          const std::vector<__nv_bfloat16> &out) {
  double max_e = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double r = __bfloat162float(ref[i]);
    double o = __bfloat162float(out[i]);
    // Denominator floor 1.0 (was 1e-5): outputs are O(sqrt(K)) ~ 10..100, but
    // a few elements are near-zero by CANCELLATION, and there cuBLAS vs our
    // kernel legitimately differ by accumulation ORDER (different tiles split
    // K differently) — tiny absolute diffs that explode as RELATIVE error
    // (seen on the 622M-element LM-heads: identical dumps yet rel>5%). Real
    // corruption (wrong tile/swizzle/barrier) shifts elements by O(10-100) and
    // still trips 5% against floor 1.0. DeepGEMM avoids the same artifact by
    // using a norm-based diff instead of per-element rel-err.
    double e = std::abs(r - o) / std::max(1.0, std::abs(r));
    if (e > max_e) {
      max_e = e;
      g_err_idx = i;
      g_err_ref = (float)r;
      g_err_out = (float)o;
    }
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

// Global progress counter — how many test CASES (run_test_case calls) have
// finished. Printed as a leading [n] on every result row so a long run shows
// live progress (and you can tell it's not hung). Set g_cases_total in main to
// enable an [n/total] readout.
static int g_cases_done = 0;
static int g_cases_total = 0;

// Generic runner for any Config and Layout
template <typename Config, Sm103Layout Layout, bool VerticalWave = false,
          bool MulticastA = false, bool DynamicCluster = false>
static Result run_test_case(int M, int N, int K, int batchCount, float alpha,
                            float beta, cublasHandle_t cublas,
                            cudaStream_t stream) {
  // Determine dimensions based on layout
  long long elA = 0;
  long long elB = 0;
  const long long elC = (long long)batchCount * M * N;

  if (Layout == Sm103Layout::NN) {
    elA = (long long)batchCount * M * K;
    elB = (long long)batchCount * K * N;
  } else if (Layout == Sm103Layout::NT) {
    elA = (long long)batchCount * M * K;
    elB = (long long)batchCount * N * K;
  } else { // TN
    elA = (long long)batchCount * K * M;
    elB = (long long)batchCount * K * N;
  }

  std::vector<__nv_bfloat16> hA(elA), hB(elB), hC_init(elC);
  fill_rand_bf16(hA, 42);
  fill_rand_bf16(hB, 99);

  // For beta != 0, initialize C with random values
  if (beta != 0.0f) {
    fill_rand_bf16(hC_init, 77);
  } else {
    std::fill(hC_init.begin(), hC_init.end(), __float2bfloat16(0.f));
  }

  __nv_bfloat16 *dA, *dB, *dC_ref, *dC_blu;
  CHECK_CUDA(cudaMalloc(&dA, elA * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dB, elB * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&dC_blu, elC * sizeof(__nv_bfloat16)));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC_ref, hC_init.data(), elC * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC_blu, hC_init.data(), elC * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  // ── cuBLAS reference ────────
  {
    float fa = alpha, fb = beta;
    cublasOperation_t transA, transB;
    long long lda, ldb, strideA, strideB;

    if (Layout == Sm103Layout::NN) {
      transA = CUBLAS_OP_N;
      transB = CUBLAS_OP_N;
      lda = K;
      ldb = N;
      strideA = (long long)M * K;
      strideB = (long long)K * N;
    } else if (Layout == Sm103Layout::NT) {
      transA = CUBLAS_OP_N;
      transB = CUBLAS_OP_T;
      lda = K;
      ldb = K;
      strideA = (long long)M * K;
      strideB = (long long)N * K;
    } else { // TN
      transA = CUBLAS_OP_T;
      transB = CUBLAS_OP_N;
      lda = M;
      ldb = N;
      strideA = (long long)K * M;
      strideB = (long long)K * N;
    }

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        cublas, transB, transA, // Swap due to column-major vs row-major
        N, M, K, &fa, dB, CUDA_R_16BF, (int)ldb, strideB, dA, CUDA_R_16BF,
        (int)lda, strideA, &fb, dC_ref, CUDA_R_16BF, N, (long long)M * N,
        batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  // Run exactly one iteration of BluBridge for accuracy checking
  launch_bgemm_sm103_cluster<Config, Layout, VerticalWave, MulticastA,
                             DynamicCluster>(dA, dB, dC_blu, M, N, K,
                                             batchCount, alpha, beta, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());

  // Accuracy Check
  std::vector<__nv_bfloat16> hC_ref_h(elC), hC_blu_h(elC);
  CHECK_CUDA(cudaMemcpy(hC_ref_h.data(), dC_ref, elC * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hC_blu_h.data(), dC_blu, elC * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  double rel_err = max_rel_err(hC_ref_h, hC_blu_h);
  if (rel_err >= 0.05) {
    printf("  [DEBUG FAIL] worst element: idx=%zu (row %lld, col %lld)  "
           "ref=%.6f  out=%.6f  absdiff=%.6f\n",
           g_err_idx, (long long)(g_err_idx / N), (long long)(g_err_idx % N),
           g_err_ref, g_err_out, std::abs(g_err_ref - g_err_out));
    printf("  [DEBUG FAIL] First 10 elements:\n");
    printf("    Ref (cuBLAS): ");
    for (int i = 0; i < std::min(10, (int)elC); ++i)
      printf("%.4f ", __bfloat162float(hC_ref_h[i]));
    printf("\n    Blu (Template): ");
    for (int i = 0; i < std::min(10, (int)elC); ++i)
      printf("%.4f ", __bfloat162float(hC_blu_h[i]));
    printf("\n");
  }

  // Warmup BluBridge Cluster Template (5 iters)
  for (int w = 0; w < 5; ++w) {
    launch_bgemm_sm103_cluster<Config, Layout, VerticalWave, MulticastA,
                               DynamicCluster>(dA, dB, dC_blu, M, N, K,
                                               batchCount, alpha, beta, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());

  // Timed BluBridge Cluster Template (20 iters)
  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));
  CHECK_CUDA(cudaEventRecord(t0, stream));
  for (int i = 0; i < 20; ++i) {
    launch_bgemm_sm103_cluster<Config, Layout, VerticalWave, MulticastA,
                               DynamicCluster>(dA, dB, dC_blu, M, N, K,
                                               batchCount, alpha, beta, stream);
  }
  CHECK_CUDA(cudaEventRecord(t1, stream));
  CHECK_CUDA(cudaEventSynchronize(t1));
  CHECK_CUDA(cudaGetLastError());
  float blu_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&blu_ms, t0, t1));

  // Warmup + timed cuBLAS (20 iters)
  {
    float fa = alpha, fb = beta;
    cublasOperation_t transA, transB;
    long long lda, ldb, strideA, strideB;

    if (Layout == Sm103Layout::NN) {
      transA = CUBLAS_OP_N;
      transB = CUBLAS_OP_N;
      lda = K;
      ldb = N;
      strideA = (long long)M * K;
      strideB = (long long)K * N;
    } else if (Layout == Sm103Layout::NT) {
      transA = CUBLAS_OP_N;
      transB = CUBLAS_OP_T;
      lda = K;
      ldb = K;
      strideA = (long long)M * K;
      strideB = (long long)N * K;
    } else { // TN
      transA = CUBLAS_OP_T;
      transB = CUBLAS_OP_N;
      lda = M;
      ldb = N;
      strideA = (long long)K * M;
      strideB = (long long)K * N;
    }

    for (int w = 0; w < 5; ++w) {
      CHECK_CUBLAS(cublasGemmStridedBatchedEx(
          cublas, transB, transA, N, M, K, &fa, dB, CUDA_R_16BF, (int)ldb,
          strideB, dA, CUDA_R_16BF, (int)lda, strideA, &fb, dC_ref, CUDA_R_16BF,
          N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) {
      CHECK_CUBLAS(cublasGemmStridedBatchedEx(
          cublas, transB, transA, N, M, K, &fa, dB, CUDA_R_16BF, (int)ldb,
          strideB, dA, CUDA_R_16BF, (int)lda, strideA, &fb, dC_ref, CUDA_R_16BF,
          N, (long long)M * N, batchCount, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
  }
  float cub_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&cub_ms, t0, t1));

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

  ++g_cases_done; // one more test case finished (progress readout)
  return {blu_tfl, cub_tfl, rel_err, rel_err < 0.05};
}

// Struct to define dynamic test shape
struct Shape {
  int M, N, K, batch;
  float alpha, beta;
  const char *label;
};

// Layout name helper
static const char *layout_name(Sm103Layout layout) {
  switch (layout) {
  case Sm103Layout::NN:
    return "NN";
  case Sm103Layout::NT:
    return "NT";
  case Sm103Layout::TN:
    return "TN";
  }
  return "Unknown";
}

template <typename Config, Sm103Layout Layout, bool VerticalWave = false,
          bool MulticastA = false, bool DynamicCluster = false>
static bool run_test_suite_for_layout_config(const std::vector<Shape> &shapes,
                                             cublasHandle_t cublas,
                                             cudaStream_t stream) {
  printf("\n=== Testing Layout: %s | Config: %dx%dx%d (cta_group::2, %s wave, "
         "clusterN=%d, mcastA=%d, dyn=%d) ===\n",
         layout_name(Layout), Config::BM, Config::BN, Config::BK,
         VerticalWave ? "vertical" : "horizontal", Config::CLUSTER_N,
         (int)MulticastA, (int)DynamicCluster);
  printf("%-18s %5s %5s %5s %5s %5s %5s  %-9s  %-9s  %-7s  %-10s  %-6s  %s\n",
         "Shape", "M", "N", "K", "Batch", "alpha", "beta", "Template", "cuBLAS",
         "%cuBLAS", "MaxRelErr", "Status", "wall(ms)");
  printf("%s\n", std::string(116, '-').c_str());
  fflush(stdout); // flush the suite header NOW so a hang on the first kernel
                  // still shows which suite/layout we're in (frontier visible)

  int passed = 0;
  int total = 0;
  double suite_wall_ms = 0.0;
  for (const auto &s : shapes) {
    // 256x256x64 config requires M & N to be multiples of 256
    if (Config::BM == 256 && (s.M % 512 != 0 || s.N % 256 != 0)) {
      // Skip shapes that are not compatible with 256x256 config
      // Note: cluster height is 2*BM = 512, cluster width is BN = 256
      continue;
    }

    total++;
    // Per-test wall-clock (whole run_test_case: alloc + cuBLAS ref + accuracy
    // check + warmups + 20 timed iters of both). Includes host-side work, so
    // it's much larger than the pure-kernel Template/cuBLAS TFLOPS above.
    auto t_start = std::chrono::steady_clock::now();
    Result r =
        run_test_case<Config, Layout, VerticalWave, MulticastA, DynamicCluster>(
            s.M, s.N, s.K, s.batch, s.alpha, s.beta, cublas, stream);
    auto t_end = std::chrono::steady_clock::now();
    double wall_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    suite_wall_ms += wall_ms;

    printf("[%3d/%3d] %-18s %5d %5d %5d %5d %5.1f %5.1f  %8.2f T  %8.2f T  "
           "%5.1f%%  "
           "%9.2e  %-6s  %8.1f\n",
           g_cases_done, g_cases_total, s.label, s.M, s.N, s.K, s.batch,
           s.alpha, s.beta, r.blublas_tflops, r.cublas_tflops,
           r.cublas_tflops > 0 ? r.blublas_tflops / r.cublas_tflops * 100 : 0.0,
           r.rel_err, r.pass ? "PASS" : "FAIL", wall_ms);
    fflush(stdout); // flush so progress is visible live during long runs

    if (r.pass)
      ++passed;
  }
  printf("Result: %d / %d PASSED   (suite wall time: %.1f ms)\n", passed, total,
         suite_wall_ms);
  fflush(stdout); // flush the suite Result so the log frontier is always exact
  return passed == total;
}

// ─── Qwen 42-shape sweep — P1 impact + dispatcher routing data
// ──────────────── For each real Qwen matmul shape (M=4096), run the 128×256
// tile in BOTH horizontal and vertical wave and compare to cuBLAS. Tells us,
// per shape, whether vertical helps (→ dispatcher routes it to vertical) and
// how close we are to cuBLAS. Correctness is validated by the suites above;
// here we focus on throughput, but still flag any accuracy miss.
template <typename Config, Sm103Layout Layout>
static void run_qwen_sweep(const std::vector<Shape> &shapes,
                           cublasHandle_t cublas, cudaStream_t stream) {
  printf("\n=== Qwen 42-shape sweep (%s, %dx%d tile): horizontal vs vertical "
         "vs cuBLAS ===\n",
         layout_name(Layout), Config::BM, Config::BN);
  printf("%-20s %-6s %7s %6s   %8s %6s   %8s  %-6s  %8s  %s\n", "Shape", "wave",
         "N", "K", "TFLOPS", "%cuB", "cuBLAS", "Status", "wall(ms)", "best");
  printf("%s\n", std::string(96, '-').c_str());
  int vwins = 0, total = 0;
  double h_sum = 0, v_sum = 0, c_sum = 0;
  double sweep_wall_ms = 0.0; // total sweep wall-clock (whole run_test_case
                              // incl. host-side work, like the suites)
  for (const auto &s : shapes) {
    // One printed row PER test case (own [n/total], own TFLOPS/%/Status/wall)
    // — no combined row hiding the losing wave's numbers. Winner marked on the
    // vert row once both are known.
    auto t0 = std::chrono::steady_clock::now();
    Result h = run_test_case<Config, Layout, false>(s.M, s.N, s.K, 1, 1.0f,
                                                    0.0f, cublas, stream);
    auto t1 = std::chrono::steady_clock::now();
    double h_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[%3d/%3d] %-20s %-6s %7d %6d   %7.1fT %5.1f%%   %7.1fT  %-6s  "
           "%8.1f\n",
           g_cases_done, g_cases_total, s.label, "horiz", s.N, s.K,
           h.blublas_tflops, h.blublas_tflops / h.cublas_tflops * 100,
           h.cublas_tflops, h.pass ? "ok" : "FAIL", h_ms);
    fflush(stdout);
    auto t2 = std::chrono::steady_clock::now();
    Result v = run_test_case<Config, Layout, true>(s.M, s.N, s.K, 1, 1.0f, 0.0f,
                                                   cublas, stream);
    auto t3 = std::chrono::steady_clock::now();
    double v_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    bool vbest = v.blublas_tflops >= h.blublas_tflops;
    printf("[%3d/%3d] %-20s %-6s %7d %6d   %7.1fT %5.1f%%   %7.1fT  %-6s  "
           "%8.1f  best=%s\n",
           g_cases_done, g_cases_total, s.label, "vert", s.N, s.K,
           v.blublas_tflops, v.blublas_tflops / v.cublas_tflops * 100,
           v.cublas_tflops, v.pass ? "ok" : "FAIL", v_ms,
           vbest ? "vert" : "horiz");
    fflush(stdout);
    sweep_wall_ms += h_ms + v_ms;
    if (vbest)
      ++vwins;
    ++total;
    h_sum += h.blublas_tflops / h.cublas_tflops;
    v_sum += v.blublas_tflops / v.cublas_tflops;
    c_sum += std::max(h.blublas_tflops, v.blublas_tflops) / v.cublas_tflops;
  }
  printf("%s\n", std::string(96, '-').c_str());
  printf("vertical wins: %d / %d   avg%% of cuBLAS: horiz %.1f%%, vert %.1f%%, "
         "best-of-2 %.1f%%   (sweep wall time: %.1f s)\n",
         vwins, total, h_sum / total * 100, v_sum / total * 100,
         c_sum / total * 100, sweep_wall_ms / 1000.0);
}

int main() {
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

  printf("====================================================================="
         "====\n");
  printf("SM103 Cluster GEMM Unified Template Test Suite\n");
  printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("====================================================================="
         "====\n");

  if (prop.major != 10 || prop.minor != 3) {
    printf("[SKIP] SM103 Cluster GEMM template requires SM 10.3 (Blackwell "
           "Ultra/sm_103a).\n"
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

  // Test cases for 128x128x64 configuration (requires M % 256 == 0, N % 128 ==
  // 0, K % 64 == 0) Note: for BM=128, cluster height is 2*BM = 256. So M must
  // be a multiple of 256.
  const std::vector<Shape> shapes_128 = {
      // M, N, K, batch, alpha, beta, label
      {256, 128, 64, 1, 1.0f, 0.0f, "tiny"},
      {256, 256, 128, 1, 1.0f, 0.0f, "small"},
      {512, 512, 128, 1, 1.0f, 0.0f, "medium"},
      {1024, 1024, 256, 1, 1.0f, 0.0f, "large"},
      // Multi-batch test
      {512, 512, 128, 4, 1.0f, 0.0f, "batched-4"},
      {256, 256, 128, 8, 1.0f, 0.0f, "batched-8"},
      // Alpha/Beta scaling and accumulation tests
      {512, 512, 128, 1, 1.5f, 0.0f, "alpha-scaling"},
      {256, 256, 128, 1, 1.0f, 0.5f, "beta-accumulate"},
      {512, 512, 128, 2, 1.2f, 0.8f, "alpha-beta-batch"},
      // ── Comparison shapes vs legacy _cluster.cu B0 results (M=4096) ──
      {4096, 4096, 4096, 1, 1.0f, 0.0f, "cmp-square4096"},
      {4096, 12288, 4096, 1, 1.0f, 0.0f, "cmp-wideN-gate"},
      {4096, 1024, 4096, 1, 1.0f, 0.0f, "cmp-skinny-K"}};

  // Test cases for 256x256x64 configuration (requires M % 512 == 0, N % 256 ==
  // 0, K % 64 == 0)
  const std::vector<Shape> shapes_256 = {
      {512, 256, 128, 1, 1.0f, 0.0f, "tiny-256"},
      {512, 512, 128, 1, 1.0f, 0.0f, "small-256"},
      {1024, 1024, 256, 1, 1.0f, 0.0f, "medium-256"},
      // Multi-batch test
      {512, 512, 128, 4, 1.0f, 0.0f, "batched-4-256"},
      // Alpha/Beta scaling and accumulation tests
      {512, 512, 128, 1, 1.5f, 0.5f, "alpha-beta-256"},
      // ── Comparison shapes vs legacy _cluster.cu B0 results (M=4096) ──
      {4096, 4096, 4096, 1, 1.0f, 0.0f, "cmp-square4096"},
      {4096, 12288, 4096, 1, 1.0f, 0.0f, "cmp-wideN-gate"},
      {4096, 1024, 4096, 1, 1.0f, 0.0f, "cmp-skinny-K"}};

  bool all_passed = true;

  // Total test cases for the [done/total] progress readout. Keep in sync if the
  // suites below change: 128x128 (×3 layouts) + 128x256 (×3) + P1-vertical (×3)
  // + P2 2x2 (×3) + P2-M2 (×3) + Qwen sweep (42 shapes × 2 waves × 2 tiles).
  // shapes_2x2 has 7 shapes, qwen_shapes has 42 (defined below).
  // 2×2 (4-CTA) cluster suites are OPT-IN via RUN_2X2=1 (and RUN_2X2_MCAST=1
  // for the A-multicast suite). Both are VALIDATED bit-exact on B300
  // (2026-07-04, §30.11-30.12) — kept opt-in only to keep the default run
  // short.
  const bool run_2x2 = (getenv("RUN_2X2") != nullptr);
  const bool run_2x2_mcast = (getenv("RUN_2X2_MCAST") != nullptr);
  const bool run_2x2f = (getenv("RUN_2X2F") != nullptr);
  const bool run_1x2 = (getenv("RUN_1X2") != nullptr);
  const bool run_1x2b = (getenv("RUN_1X2B") != nullptr);
  const bool run_1x2c = (getenv("RUN_1X2C") != nullptr);
  g_cases_total = (int)shapes_128.size() * 3 + (int)shapes_256.size() * 3 * 2 +
                  (run_2x2 ? 7 * 3 : 0) +
                  (run_2x2 && run_2x2_mcast ? 7 * 3 : 0) +
                  (run_2x2f ? 7 * 3 * 2 : 0) + (run_1x2 ? 7 * 3 * 2 : 0) +
                  (run_1x2b ? 7 * 2 : 0) + (run_1x2c ? 7 * 2 : 0) + 42 * 2 * 2;

  // 1. NN Layout
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::NN>(
          shapes_128, cublas, stream);
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NN>(
          shapes_256, cublas, stream);

  // 2. NT Layout
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::NT>(
          shapes_128, cublas, stream);
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NT>(
          shapes_256, cublas, stream);

  // 3. TN Layout
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::TN>(
          shapes_128, cublas, stream);
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::TN>(
          shapes_256, cublas, stream);

  // 4. P1 variant: 128×256 tile + VERTICAL wave (cuBLAS `128x256_2x1_v`).
  // Same tile/cluster as the 256 suites above, VerticalWave=true. Compare the
  // TFLOPS here vs the horizontal 256 suites on the same shapes to isolate the
  // wave-order effect (expected to help the M=4096 wide-N shapes via L2 reuse).
  printf("\n########## P1: 128x256 tile, VERTICAL wave (2x1) ##########\n");
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NN,
                                       true>(shapes_256, cublas, stream);
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NT,
                                       true>(shapes_256, cublas, stream);
  all_passed &=
      run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::TN,
                                       true>(shapes_256, cublas, stream);

  // ── P2: 128×256 tile, 2×2 cluster (4-CTA) — Milestone 1 (redundant-A) ──
  // Correctness-first. Shapes have M%256==0 and N%512==0 so both N-blocks of
  // each cluster are valid (no partial-tile OOB). If these pass bit-exact, the
  // 4-CTA cluster + per-pair MMA + scheduler N-split are correct (§30).
  const std::vector<Shape> shapes_2x2 = {
      {512, 512, 128, 1, 1.0f, 0.0f, "2x2-small"},
      {1024, 1024, 256, 1, 1.0f, 0.0f, "2x2-medium"},
      {2048, 2048, 512, 1, 1.0f, 0.0f, "2x2-large"},
      {512, 512, 128, 4, 1.0f, 0.0f, "2x2-batched"},
      {512, 512, 128, 1, 1.5f, 0.5f, "2x2-alpha-beta"},
      {4096, 4096, 4096, 1, 1.0f, 0.0f, "2x2-square4096"},
      {4096, 12288, 4096, 1, 1.0f, 0.0f, "2x2-wideN"}};
  if (run_2x2) {
    printf("\n########## P2: 128x256 tile, 2x2 cluster (4-CTA) ##########\n");
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                                   Sm103Layout::NN>(
        shapes_2x2, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                                   Sm103Layout::NT>(
        shapes_2x2, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                                   Sm103Layout::TN>(
        shapes_2x2, cublas, stream);

    // P2 Milestone 2: A-MULTICAST ON (load A once, multicast across N —
    // halves A DRAM traffic). VALIDATED bit-exact on B300 after the mbar
    // parity fix (leader mbar for even-parity routing, §30.11). Opt-in via
    // RUN_2X2_MCAST=1 only to keep the default run short.
    if (getenv("RUN_2X2_MCAST") != nullptr) {
      printf("\n########## P2-M2: 128x256 2x2 cluster, A-MULTICAST ON "
             "##########\n");
      all_passed &=
          run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                           Sm103Layout::NN, false, true>(
              shapes_2x2, cublas, stream);
      all_passed &=
          run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                           Sm103Layout::NT, false, true>(
              shapes_2x2, cublas, stream);
      all_passed &=
          run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                           Sm103Layout::TN, false, true>(
              shapes_2x2, cublas, stream);
    } else {
      printf("\n########## P2-M2 (A-MULTICAST) SKIPPED — redundant-A 2x2 above "
             "is correct; multicast is a deferred DRAM-opt (RUN_2X2_MCAST=1 to "
             "test) ##########\n");
    }
  } else {
    printf("\n########## P2 2x2 (4-CTA) SUITES SKIPPED (set RUN_2X2=1 to "
           "include) ##########\n");
  }

  // ── P3: "2x2f" — DynamicCluster (preferred 4-CTA, fallback 2-CTA). The
  // cuBLAS crown-jewel launch policy (§31): 100%% SM fill vs rigid-4's 89%%.
  // M1: CLC stealing disabled for this variant (one tile-unit per cluster).
  // Runs redundant-A AND A-multicast (adaptive masks) — both must be
  // bit-exact. Needs CUDA >= 12.8 on the server. RUN_2X2F=1 to include.
  if (run_2x2f) {
    printf("\n########## P3: 128x256 2x2f (dynamic cluster, pref 4 / fb 2) "
           "##########\n");
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::NN, false, false, true>(
            shapes_2x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::NT, false, false, true>(
            shapes_2x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::TN, false, false, true>(
            shapes_2x2, cublas, stream);
    printf("\n########## P3-M2: 2x2f + A-MULTICAST (adaptive) ##########\n");
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::NN, false, true, true>(
            shapes_2x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::NT, false, true, true>(
            shapes_2x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_128x256_2x2,
                                         Sm103Layout::TN, false, true, true>(
            shapes_2x2, cublas, stream);
  } else {
    printf("\n########## P3 2x2f (dynamic cluster) SKIPPED — RUN_2X2F=1 to "
           "include ##########\n");
  }

  // ── P4: 1x2 cluster, cta_group::1 (single-CTA MMA), 128x192 bring-up ──
  // First cta_group::1 path (CLUSTER_SIZE=1 knob, DeepGEMM-style; 1x2_plan.md).
  // 2 CTAs along N share A; barriers fully symmetric (no parity routing).
  // Covers the machinery for the nvjet 160x192_1x2 shapes (swap-AB next).
  // Shapes: M%128==0 (cluster row = 1*BM = 128), N%384==0 for full 1x2 units.
  if (run_1x2) {
    const std::vector<Shape> shapes_1x2 = {
        {128, 384, 128, 1, 1.0f, 0.0f, "1x2-tiny"},
        {512, 768, 256, 1, 1.0f, 0.0f, "1x2-small"},
        {1024, 1152, 512, 1, 1.0f, 0.0f, "1x2-medium"},
        {512, 768, 128, 4, 1.0f, 0.0f, "1x2-batched"},
        {512, 768, 128, 1, 1.5f, 0.5f, "1x2-alpha-beta"},
        {4096, 2048, 1024, 1, 1.0f, 0.0f, "1x2-qwen-q06"},
        {4096, 2048, 2048, 1, 1.0f, 0.0f, "1x2-qwen-q3b"}};
    printf("\n########## P4: 1x2 cluster (cta_group::1), 128x192 bring-up "
           "##########\n");
    all_passed &= run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                                   Sm103Layout::NN>(
        shapes_1x2, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                                   Sm103Layout::NT>(
        shapes_1x2, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                                   Sm103Layout::TN>(
        shapes_1x2, cublas, stream);
    printf("\n########## P4-M2: 1x2 + A-MULTICAST ##########\n");
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                         Sm103Layout::NN, false, true>(
            shapes_1x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                         Sm103Layout::NT, false, true>(
            shapes_1x2, cublas, stream);
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_1x2_128x192,
                                         Sm103Layout::TN, false, true>(
            shapes_1x2, cublas, stream);
  } else {
    printf("\n########## P4 1x2 (cta_group::1) SKIPPED — RUN_1X2=1 to include "
           "##########\n");
  }

  // ── P5: swap-AB B1 — C^T compute, composite M'=192 MMA + transposed
  // epilogue (1x2_plan.md R7). TN only (the 3 Qwen Q-projections are TN).
  // Same C-tile as P4 (128x192) so P4-vs-P5 isolates the swap machinery.
  if (run_1x2b) {
    const std::vector<Shape> shapes_1x2b = {
        {128, 384, 128, 1, 1.0f, 0.0f, "1x2b-tiny"},
        {512, 768, 256, 1, 1.0f, 0.0f, "1x2b-small"},
        {1024, 1152, 512, 1, 1.0f, 0.0f, "1x2b-medium"},
        {512, 768, 128, 4, 1.0f, 0.0f, "1x2b-batched"},
        {512, 768, 128, 1, 1.5f, 0.5f, "1x2b-alpha-beta"},
        {4096, 2048, 1024, 1, 1.0f, 0.0f, "1x2b-qwen-q06"},
        {4096, 2048, 2048, 1, 1.0f, 0.0f, "1x2b-qwen-q3b"}};
    printf("\n########## P5: 1x2 swap-AB B1 (C^T, composite M'=192) "
           "##########\n");
    all_passed &= run_test_suite_for_layout_config<Sm103Config_1x2swap_128x192,
                                                   Sm103Layout::TN>(
        shapes_1x2b, cublas, stream);
    printf("\n########## P5-M2: swap-AB + A-MULTICAST ##########\n");
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_1x2swap_128x192,
                                         Sm103Layout::TN, false, true>(
            shapes_1x2b, cublas, stream);
  } else {
    printf("\n########## P5 swap-AB SKIPPED — RUN_1X2B=1 to include "
           "##########\n");
  }

  // ── P6: B2 — the real nvjet 160x192_1x2 tile (swap-AB + 64B-swizzle
  // 160-wide operand; 1x2_plan.md R7). TN only. Includes non-multiple-of-160
  // M shapes (TMA OOB fill / clipped store) + the 3 Qwen Q-projections.
  if (run_1x2c) {
    const std::vector<Shape> shapes_1x2c = {
        {160, 384, 128, 1, 1.0f, 0.0f, "1x2c-tiny"},
        {320, 768, 256, 1, 1.0f, 0.0f, "1x2c-small"},
        {192, 384, 128, 1, 1.0f, 0.0f,
         "1x2c-oobM"}, // partial M-tile; M%32==0 (64B-swizzle tmap
                       // precondition, R7a)
        {480, 768, 128, 4, 1.0f, 0.0f, "1x2c-batched"},
        {320, 768, 128, 1, 1.5f, 0.5f, "1x2c-alpha-beta"},
        {4096, 2048, 1024, 1, 1.0f, 0.0f, "1x2c-qwen-q06"},
        {4096, 2048, 2048, 1, 1.0f, 0.0f, "1x2c-qwen-q3b"}};
    printf("\n########## P6: 1x2 swap-AB B2 (160x192, 64B-swizzle operand) "
           "##########\n");
    all_passed &= run_test_suite_for_layout_config<Sm103Config_1x2swap_160x192,
                                                   Sm103Layout::TN>(
        shapes_1x2c, cublas, stream);
    printf("\n########## P6-M2: B2 + A-MULTICAST ##########\n");
    all_passed &=
        run_test_suite_for_layout_config<Sm103Config_1x2swap_160x192,
                                         Sm103Layout::TN, false, true>(
            shapes_1x2c, cublas, stream);
  } else {
    printf("\n########## P6 B2 (160x192) SKIPPED — RUN_1X2C=1 to include "
           "##########\n");
  }

  // ── Full Qwen matmul-shape sweep (42 shapes, M=4096) ──
  // (N, K) from analyzations.md §16.3. Runs 128×256 horizontal vs vertical vs
  // cuBLAS to measure P1 impact per shape + collect dispatcher routing data.
  const std::vector<Shape> qwen_shapes = {
      {4096, 896, 896, 1, 1, 0, "q25-0.5b-Q"},
      {4096, 128, 896, 1, 1, 0, "q25-0.5b-K"},
      {4096, 4864, 896, 1, 1, 0, "q25-0.5b-Gate"},
      {4096, 896, 4864, 1, 1, 0, "q25-0.5b-Down"},
      {4096, 151936, 896, 1, 1, 0, "q25-0.5b-LMhead"},
      {4096, 1536, 1536, 1, 1, 0, "q25-1.5b-Q"},
      {4096, 256, 1536, 1, 1, 0, "q25-1.5b-K"},
      {4096, 8960, 1536, 1, 1, 0, "q25-1.5b-Gate"},
      {4096, 1536, 8960, 1, 1, 0, "q25-1.5b-Down"},
      {4096, 151936, 1536, 1, 1, 0, "q25-1.5b-LMhead"},
      {4096, 2048, 2048, 1, 1, 0, "q25-3b-Q"},
      {4096, 256, 2048, 1, 1, 0, "q25-3b-K"},
      {4096, 11008, 2048, 1, 1, 0, "q25-3b-Gate"},
      {4096, 2048, 11008, 1, 1, 0, "q25-3b-Down"},
      {4096, 151936, 2048, 1, 1, 0, "q25-3b-LMhead"},
      {4096, 3584, 3584, 1, 1, 0, "q25-7b-Q"},
      {4096, 512, 3584, 1, 1, 0, "q25-7b-K"},
      {4096, 18944, 3584, 1, 1, 0, "q25-7b-Gate"},
      {4096, 3584, 18944, 1, 1, 0, "q25-7b-Down"},
      {4096, 152064, 3584, 1, 1, 0, "q25-7b-LMhead"},
      {4096, 2048, 1024, 1, 1, 0, "q3-0.6b-Q"},
      {4096, 1024, 1024, 1, 1, 0, "q3-0.6b-K"},
      {4096, 1024, 2048, 1, 1, 0, "q3-0.6b-O"},
      {4096, 3072, 1024, 1, 1, 0, "q3-0.6b-Gate"},
      {4096, 1024, 3072, 1, 1, 0, "q3-0.6b-Down"},
      {4096, 151936, 1024, 1, 1, 0, "q3-0.6b-LMhead"},
      {4096, 2048, 2048, 1, 1, 0, "q3-1.7b-Q"},
      {4096, 1024, 2048, 1, 1, 0, "q3-1.7b-K"},
      {4096, 6144, 2048, 1, 1, 0, "q3-1.7b-Gate"},
      {4096, 2048, 6144, 1, 1, 0, "q3-1.7b-Down"},
      {4096, 151936, 2048, 1, 1, 0, "q3-1.7b-LMhead"},
      {4096, 4096, 2560, 1, 1, 0, "q3-4b-Q"},
      {4096, 1024, 2560, 1, 1, 0, "q3-4b-K"},
      {4096, 2560, 4096, 1, 1, 0, "q3-4b-O"},
      {4096, 9728, 2560, 1, 1, 0, "q3-4b-Gate"},
      {4096, 2560, 9728, 1, 1, 0, "q3-4b-Down"},
      {4096, 151936, 2560, 1, 1, 0, "q3-4b-LMhead"},
      {4096, 4096, 4096, 1, 1, 0, "q3-8b-Q"},
      {4096, 1024, 4096, 1, 1, 0, "q3-8b-K"},
      {4096, 12288, 4096, 1, 1, 0, "q3-8b-Gate"},
      {4096, 4096, 12288, 1, 1, 0, "q3-8b-Down"},
      {4096, 151936, 4096, 1, 1, 0, "q3-8b-LMhead"}};

  printf("\n########## Qwen 42-shape sweep (P1 impact) ##########\n");
  // 128×256 tile (BN=256) — good for wide-N, wastes half on skinny-N.
  run_qwen_sweep<Sm103Config_256x256x64, Sm103Layout::TN>(qwen_shapes, cublas,
                                                          stream);
  // 128×128 tile (BN=128) — no N-waste on skinny shapes (N=128 → 1 tile,
  // N=256 → 2). Compare per shape to pick best-of-tiles for the dispatcher.
  run_qwen_sweep<Sm103Config_128x128x64, Sm103Layout::TN>(qwen_shapes, cublas,
                                                          stream);
  // (64×128 / BM=64 removed — BM<128 is invalid for this cta_group::2 epilogue;
  //  see §28.7. Skinny-N needs a cta_group::1 variant, not a BM=64 tile.)

  printf("\n==================================================================="
         "======\n");
  if (all_passed) {
    printf("ALL TESTS PASSED SUCCESSFULLY!\n");
  } else {
    printf("SOME TESTS FAILED!\n");
  }
  printf("====================================================================="
         "====\n");

  CHECK_CUBLAS(cublasDestroy(cublas));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
