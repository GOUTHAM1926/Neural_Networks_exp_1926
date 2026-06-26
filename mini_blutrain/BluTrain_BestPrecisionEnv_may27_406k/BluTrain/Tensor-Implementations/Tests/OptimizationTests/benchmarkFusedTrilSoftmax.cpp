#include <iostream>
#include <vector>
#include <limits>

#include "TensorLib.h"           // pulls in TensorOps.h → tril()
#include "ops/FusedKernels.cuh"
#include "ops/helpers/ActivationKernels.h"
#include "utils/KernelUtils.cuh"

using namespace OwnTensor;

// =============================================================================
// Config — change these to match your attention dims
// =============================================================================
static const int64_t ROWS         = 8192 * 6;   // B * H  (8 batches × 1024 seq len)
static const int64_t COLS         = 1024;   // W      (seq len)
static const int64_t H            = 1024;   // per-batch height for fused kernel
static const int64_t W            = 1024;   // per-batch width  for fused kernel
static const int     WARMUP_ITERS = 10;
static const int     BENCH_ITERS  = 100;

// =============================================================================
// L2 flush buffer — allocate once, reuse across both benchmarks
// =============================================================================
static void* make_flush_buf(size_t& out_size) {
  int device  = 0;
  int l2_size = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
  printf("L2 cache size: %d MB\n", l2_size / (1024 * 1024));
  out_size = static_cast<size_t>(l2_size) * 2;
  void* buf = nullptr;
  CUDA_CHECK(cudaMalloc(&buf, out_size));
  CUDA_CHECK(cudaMemset(buf, 0, out_size));   // note: buf, NOT &buf
  return buf;
}

// =============================================================================
// Benchmark: fused tril + softmax in a single kernel
// =============================================================================
float benchmark_fused(const Tensor& input, Tensor& output,
                      void* d_flush, size_t flush_size) {
  const float NEG_INF = -std::numeric_limits<float>::infinity();

  // Warm-up
  for (int i = 0; i < WARMUP_ITERS; ++i)
    fused_tril_softmax(input.data<float>(), output.data<float>(),
                           0, NEG_INF, H, W, ROWS, COLS);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> times(BENCH_ITERS);
  for (int i = 0; i < BENCH_ITERS; ++i) {
    // Flush L2 between iterations to get cold-cache timing
    CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));  // note: d_flush, NOT &d_flush
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    fused_tril_softmax(input.data<float>(), output.data<float>(),
                           0, NEG_INF, H, W, ROWS, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&times[i], start, stop);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float avg = 0.f;
  for (float t : times) avg += t;
  return avg / BENCH_ITERS;
}

// =============================================================================
// Benchmark: separate tril kernel then softmax kernel, same input data
// =============================================================================
float benchmark_separate(const Tensor& input,
                         Tensor& tril_out, Tensor& sm_out,
                         void* d_flush, size_t flush_size) {
  const double NEG_INF_D = -std::numeric_limits<double>::infinity();

  // Warm-up
  for (int i = 0; i < WARMUP_ITERS; ++i) {
    tril_out = tril(input, 0, NEG_INF_D);   // tril() dispatches to cuda_tril_tensor
    cuda::softmax_forward_cuda(tril_out.data<float>(), sm_out.data<float>(),
                               ROWS, COLS);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float> times(BENCH_ITERS);
  for (int i = 0; i < BENCH_ITERS; ++i) {
    // Flush L2 between iterations
    CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start);
    // Both kernels run sequentially on the default stream
    tril_out = tril(input, 0, NEG_INF_D);
    cuda::softmax_forward_cuda(tril_out.data<float>(), sm_out.data<float>(),
                               ROWS, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&times[i], start, stop);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  float avg = 0.f;
  for (float t : times) avg += t;
  return avg / BENCH_ITERS;
}

// =============================================================================
// main
// =============================================================================
int main() {
  const size_t mb = (ROWS * COLS * sizeof(float)) / (1024 * 1024);
  printf("Matrix : [%ld x %ld]  (%zu MB)\n", ROWS, COLS, mb);
  printf("Batches: %ld x [%ld x %ld]\n", ROWS / H, H, W);
  printf("Warmup : %d   Bench iters: %d\n\n", WARMUP_ITERS, BENCH_ITERS);

  size_t flush_size = 0;
  void*  d_flush    = make_flush_buf(flush_size);

  TensorOptions opts =
      TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);

  // All tensors share the same input — ensures an identical data set
  Tensor input    = Tensor::rand<float>(Shape{{ROWS, COLS}}, opts);
  Tensor fused_out  = Tensor::empty(Shape{{ROWS, COLS}}, opts);
  Tensor tril_out   = Tensor::empty(Shape{{ROWS, COLS}}, opts);
  Tensor sm_out     = Tensor::empty(Shape{{ROWS, COLS}}, opts);

  printf("--- Fused tril+softmax (single kernel) ---\n");
  float fused_ms = benchmark_fused(input, fused_out, d_flush, flush_size);
  printf("\033[32m  avg: %.4f ms\033[0m\n\n", fused_ms);

  printf("--- Separate tril then softmax (two kernels, same input) ---\n");
  float sep_ms = benchmark_separate(input, tril_out, sm_out, d_flush, flush_size);
  printf("\033[32m  avg: %.4f ms\033[0m\n\n", sep_ms);

  printf("=== Summary ===\n");
  printf("  Fused:    %.4f ms\n", fused_ms);
  printf("  Separate: %.4f ms\n", sep_ms);
  if (fused_ms < sep_ms)
    printf("\033[32m  Fused is %.2fx faster\033[0m\n", sep_ms / fused_ms);
  else
    printf("\033[31m  Separate is %.2fx faster (fused needs more work)\033[0m\n",
           fused_ms / sep_ms);

  cudaFree(d_flush);
  return 0;
}
