#define C10_CUDA_NO_CMAKE_CONFIGURE_FILE
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>

#include "TensorLib.h"
#include "ops/helpers/LossKernels.h"
#include "utils/KernelUtils.cuh"

using namespace OwnTensor;

// =============================================================================
// Config
// =============================================================================
static const int64_t BATCH_SIZE  = 4096;
static const int64_t VOCAB_SIZE  = 32768;  // divisible by 4 — hits the aligned cp.async path
static const int     WARMUP_ITERS = 10;
static const int     BENCH_ITERS  = 100;

// =============================================================================
// Helpers
// =============================================================================
static void* make_flush_buf(size_t& out_size) {
    int device = 0, l2_size = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
    printf("L2 cache size: %d MB\n", l2_size / (1024 * 1024));
    out_size = (size_t)l2_size * 2;
    void* buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, out_size));
    CUDA_CHECK(cudaMemset(buf, 0, out_size));
    return buf;
}

static float mean_ms(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.f) / (float)v.size();
}

// =============================================================================
// Benchmark: non-vec  (sparse_cross_entropy_forward_cuda)
// =============================================================================
float benchmark_nonvec(
    const float* d_logits, const int32_t* d_targets, float* d_loss,
    void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
            d_logits, d_targets, d_loss, BATCH_SIZE, VOCAB_SIZE, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
            d_logits, d_targets, d_loss, BATCH_SIZE, VOCAB_SIZE, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// Benchmark: vec  (sparse_cross_entropy_forward_cuda_vec)
// =============================================================================
float benchmark_vec(
    const float* d_logits, const int32_t* d_targets, float* d_loss,
    void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
            d_logits, d_targets, d_loss, BATCH_SIZE, VOCAB_SIZE, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
            d_logits, d_targets, d_loss, BATCH_SIZE, VOCAB_SIZE, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// Benchmark: PyTorch  (cross_entropy_loss forward only, Reduction::Sum)
//
//   Our kernels output sum(per-sample losses) — same as Reduction::Sum in PyTorch.
//   No backward needed: the custom kernels are forward-only here, so this is
//   a fair apples-to-apples scope (raw logits → scalar loss).
// =============================================================================
float benchmark_pytorch(
    const at::Tensor& pt_logits,
    const at::Tensor& pt_targets,
    void* d_flush, size_t flush_size)
{
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto loss = at::cross_entropy_loss(pt_logits, pt_targets,
                                           /*weight=*/{},
                                           at::Reduction::Sum);
        (void)loss;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        auto loss = at::cross_entropy_loss(pt_logits, pt_targets,
                                           /*weight=*/{},
                                           at::Reduction::Sum);
        (void)loss;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// main
// =============================================================================
int main() {
    size_t n_elems = (size_t)BATCH_SIZE * VOCAB_SIZE;
    size_t mb      = n_elems * sizeof(float) / (1024 * 1024);

    printf("Sparse CE Forward — Three-way Benchmark\n");
    printf("  Batch size  : %ld\n",   BATCH_SIZE);
    printf("  Vocab size  : %ld\n",   VOCAB_SIZE);
    printf("  Logits      : [%ld x %ld]  (%zu MB)\n", BATCH_SIZE, VOCAB_SIZE, mb);
    printf("  Warmup      : %d   Bench iters: %d\n\n", WARMUP_ITERS, BENCH_ITERS);

    // -------------------------------------------------------------------------
    // Shared host data — same inputs for all three kernels
    // -------------------------------------------------------------------------
    srand(42);
    std::vector<float>   h_logits(n_elems);
    std::vector<int32_t> h_targets(BATCH_SIZE);

    for (size_t i = 0; i < n_elems; ++i)
        h_logits[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    for (int64_t i = 0; i < BATCH_SIZE; ++i)
        h_targets[i] = rand() % (int)VOCAB_SIZE;

    // -------------------------------------------------------------------------
    // Device buffers  (logits and targets are read-only, shared across kernels)
    // -------------------------------------------------------------------------
    float   *d_logits, *d_loss_nonvec, *d_loss_vec;
    int32_t *d_targets;

    CUDA_CHECK(cudaMalloc(&d_logits,     n_elems    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_nonvec, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_vec,   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets,    BATCH_SIZE * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_logits,  h_logits.data(),  n_elems    * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), BATCH_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // PyTorch tensors — built from the same host data, no grad needed
    // -------------------------------------------------------------------------
    auto pt_f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto pt_i64_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);

    at::Tensor pt_logits = at::tensor(h_logits, pt_f32_opts).reshape({BATCH_SIZE, VOCAB_SIZE});

    std::vector<int64_t> h_targets64(BATCH_SIZE);
    for (int64_t i = 0; i < BATCH_SIZE; ++i) h_targets64[i] = h_targets[i];
    at::Tensor pt_targets = at::tensor(h_targets64, pt_i64_opts);

    // -------------------------------------------------------------------------
    // L2 flush buffer
    // -------------------------------------------------------------------------
    size_t flush_size = 0;
    void*  d_flush    = make_flush_buf(flush_size);
    printf("\n");

    // -------------------------------------------------------------------------
    // Run benchmarks
    // -------------------------------------------------------------------------
    printf("--- [1] Non-vec  (sparse_cross_entropy_forward_cuda) ---\n");
    float nonvec_ms = benchmark_nonvec(d_logits, d_targets, d_loss_nonvec, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", nonvec_ms);

    printf("--- [2] Vec  (sparse_cross_entropy_forward_cuda_vec) ---\n");
    float vec_ms = benchmark_vec(d_logits, d_targets, d_loss_vec, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", vec_ms);

    printf("--- [3] PyTorch  (cross_entropy_loss, Reduction::Sum) ---\n");
    float pt_ms = benchmark_pytorch(pt_logits, pt_targets, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", pt_ms);

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    printf("=== Summary ===\n");
    printf("  [1] Non-vec            : %8.4f ms\n", nonvec_ms);
    printf("  [2] Vec                : %8.4f ms\n", vec_ms);
    printf("  [3] PyTorch (fwd only) : %8.4f ms\n", pt_ms);
    printf("\n");

    // Vec vs non-vec
    if (vec_ms < nonvec_ms)
        printf("\033[32m  Vec is %.2fx faster than Non-vec\033[0m\n",
               nonvec_ms / vec_ms);
    else
        printf("\033[33m  Vec is %.2fx slower than Non-vec\033[0m\n",
               vec_ms / nonvec_ms);

    // Best custom vs PyTorch
    float best_ms = (vec_ms < nonvec_ms) ? vec_ms : nonvec_ms;
    const char* best_name = (vec_ms < nonvec_ms) ? "Vec" : "Non-vec";
    if (best_ms < pt_ms)
        printf("\033[32m  %s is %.2fx faster than PyTorch\033[0m\n",
               best_name, pt_ms / best_ms);
    else
        printf("\033[31m  PyTorch is %.2fx faster than best custom (%s)\033[0m\n",
               best_ms / pt_ms, best_name);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_logits);
    cudaFree(d_loss_nonvec);
    cudaFree(d_loss_vec);
    cudaFree(d_targets);
    cudaFree(d_flush);
    cudaDeviceReset();
    return 0;
}
