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
static const int64_t BATCH_SIZE   = 4096;
// VOCAB_SIZE must be divisible by 2048 for the vec kernel:
//   - Reduce kernel loop step = bdim*4 = 1024; elementsPerBlock = V/2 must be a
//     multiple of 1024, so V % 2048 == 0.  This also satisfies the 16-byte
//     alignment requirement (2048 % 8 == 0) and the normalize kernel (V % 1024 == 0).
// 32768 = 16 * 2048 (LLaMA-style vocab, safely divisible).
static const int64_t VOCAB_SIZE   = 32768;
static const float   HOST_SCALE   = 1.0f / (float)BATCH_SIZE;
static const float   GRAD_OUT_VAL = 1.0f;
static const int     WARMUP_ITERS = 10;
static const int     BENCH_ITERS  = 100;

static_assert(VOCAB_SIZE % 2048 == 0, "VOCAB_SIZE must be divisible by 2048 for the vec kernel");

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
// Benchmark: single-kernel  (sparse_cross_entropy_backward_cuda)
// =============================================================================
float benchmark_single_kernel(
    const float* d_logits, const int32_t* d_targets, float* d_grad,
    const float* d_grad_out, void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_grad, 0, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));
        cuda::sparse_cross_entropy_backward_cuda<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
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
        cuda::sparse_cross_entropy_backward_cuda<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// Benchmark: two-kernel  (sparse_ce_backward_cuda)
// =============================================================================
float benchmark_two_kernel(
    const float* d_logits, const int32_t* d_targets, float* d_grad,
    const float* d_grad_out, void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_grad, 0, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));
        cuda::sparse_ce_backward_cuda<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
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
        cuda::sparse_ce_backward_cuda<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// Benchmark: two-kernel vectorized  (sparse_ce_backward_cuda_vec)
// =============================================================================
float benchmark_vec_kernel(
    const float* d_logits, const int32_t* d_targets, float* d_grad,
    const float* d_grad_out, void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(d_grad, 0, BATCH_SIZE * VOCAB_SIZE * sizeof(float)));
        cuda::sparse_ce_backward_cuda_vec<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
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
        cuda::sparse_ce_backward_cuda_vec<float, int32_t>(
            d_logits, d_targets, d_grad,
            BATCH_SIZE, VOCAB_SIZE, d_grad_out, HOST_SCALE, nullptr);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return mean_ms(times);
}

// =============================================================================
// Benchmark: PyTorch  (cross_entropy_loss + backward)
//
//   With Reduction::Mean, grad = (softmax(logits) - one_hot(targets)) / N.
//   This matches host_scale = 1/N and grad_output = 1.0 used by the custom
//   kernels.  Both our kernels and PyTorch recompute softmax from raw logits,
//   so forward + backward is the correct apples-to-apples scope.
// =============================================================================
float benchmark_pytorch(
    at::Tensor logits_leaf,
    const at::Tensor& pt_targets,
    void* d_flush, size_t flush_size)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        if (logits_leaf.grad().defined()) logits_leaf.grad().zero_();
        auto loss = at::cross_entropy_loss(logits_leaf, pt_targets,
                                           /*weight=*/{},
                                           at::Reduction::Mean);
        loss.backward();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(BENCH_ITERS);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_size));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (logits_leaf.grad().defined()) logits_leaf.grad().zero_();

        cudaEventRecord(start);
        auto loss = at::cross_entropy_loss(logits_leaf, pt_targets,
                                           /*weight=*/{},
                                           at::Reduction::Mean);
        loss.backward();
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

    printf("Sparse CE Backward — Four-way Benchmark\n");
    printf("  Batch size  : %ld\n",   BATCH_SIZE);
    printf("  Vocab size  : %ld\n",   VOCAB_SIZE);
    printf("  Logits      : [%ld x %ld]  (%zu MB)\n", BATCH_SIZE, VOCAB_SIZE, mb);
    printf("  host_scale  : %.6f\n",  HOST_SCALE);
    printf("  grad_output : %.1f\n",  GRAD_OUT_VAL);
    printf("  Warmup      : %d   Bench iters: %d\n\n", WARMUP_ITERS, BENCH_ITERS);

    // -------------------------------------------------------------------------
    // Shared host data — same inputs go to all four kernels
    // -------------------------------------------------------------------------
    srand(42);
    std::vector<float>   h_logits(n_elems);
    std::vector<int32_t> h_targets(BATCH_SIZE);

    for (size_t i = 0; i < n_elems; ++i)
        h_logits[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    for (int64_t i = 0; i < BATCH_SIZE; ++i)
        h_targets[i] = rand() % (int)VOCAB_SIZE;

    // -------------------------------------------------------------------------
    // Device buffers (logits and targets are read-only, shared across kernels)
    // -------------------------------------------------------------------------
    float   *d_logits, *d_grad_single, *d_grad_two, *d_grad_vec, *d_grad_out;
    int32_t *d_targets;

    CUDA_CHECK(cudaMalloc(&d_logits,      n_elems    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_single, n_elems    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_two,    n_elems    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_vec,    n_elems    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_out,    sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets,     BATCH_SIZE * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_logits,  h_logits.data(),  n_elems    * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), BATCH_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_out, &GRAD_OUT_VAL,   sizeof(float),               cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // PyTorch tensors — built from the same host data
    // -------------------------------------------------------------------------
    auto pt_f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto pt_i64_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);

    at::Tensor pt_logits = at::tensor(h_logits, pt_f32_opts).reshape({BATCH_SIZE, VOCAB_SIZE});

    std::vector<int64_t> h_targets64(BATCH_SIZE);
    for (int64_t i = 0; i < BATCH_SIZE; ++i) h_targets64[i] = h_targets[i];
    at::Tensor pt_targets = at::tensor(h_targets64, pt_i64_opts);

    // Leaf variable — same values, grad tracking enabled
    at::Tensor logits_leaf = pt_logits.clone().detach().requires_grad_(true);

    // -------------------------------------------------------------------------
    // L2 flush buffer
    // -------------------------------------------------------------------------
    size_t flush_size = 0;
    void*  d_flush    = make_flush_buf(flush_size);
    printf("\n");

    // -------------------------------------------------------------------------
    // Run benchmarks
    // -------------------------------------------------------------------------
    printf("--- [1] Single-kernel  (sparse_cross_entropy_backward_cuda) ---\n");
    float single_ms = benchmark_single_kernel(
        d_logits, d_targets, d_grad_single, d_grad_out, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", single_ms);

    printf("--- [2] Two-kernel  (sparse_ce_backward_cuda) ---\n");
    float two_ms = benchmark_two_kernel(
        d_logits, d_targets, d_grad_two, d_grad_out, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", two_ms);

    printf("--- [3] Two-kernel vec  (sparse_ce_backward_cuda_vec) ---\n");
    float vec_ms = benchmark_vec_kernel(
        d_logits, d_targets, d_grad_vec, d_grad_out, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", vec_ms);

    printf("--- [4] PyTorch  (cross_entropy_loss fwd + backward) ---\n");
    float pt_ms = benchmark_pytorch(logits_leaf, pt_targets, d_flush, flush_size);
    printf("\033[32m  avg: %.4f ms\033[0m\n\n", pt_ms);

    // -------------------------------------------------------------------------
    // Summary table
    // -------------------------------------------------------------------------
    printf("=== Summary ===\n");
    printf("  [1] Single-kernel       : %8.4f ms\n", single_ms);
    printf("  [2] Two-kernel          : %8.4f ms\n", two_ms);
    printf("  [3] Two-kernel vec      : %8.4f ms\n", vec_ms);
    printf("  [4] PyTorch fwd+bwd     : %8.4f ms\n", pt_ms);
    printf("\n");

    // Best custom vs PyTorch
    float mins[3]       = { single_ms, two_ms, vec_ms };
    const char* names[] = { "Single-kernel", "Two-kernel", "Two-kernel-vec" };
    int best_idx = 0;
    for (int i = 1; i < 3; ++i)
        if (mins[i] < mins[best_idx]) best_idx = i;
    float best_ms = mins[best_idx];

    if (best_ms < pt_ms)
        printf("\033[32m  %s is %.2fx faster than PyTorch\033[0m\n",
               names[best_idx], pt_ms / best_ms);
    else
        printf("\033[31m  PyTorch is %.2fx faster than best custom (%s)\033[0m\n",
               best_ms / pt_ms, names[best_idx]);

    // Vec vs non-vec two-kernel
    if (vec_ms < two_ms)
        printf("\033[32m  Vec is %.2fx faster than non-vec two-kernel\033[0m\n",
               two_ms / vec_ms);
    else
        printf("\033[33m  Vec is %.2fx slower than non-vec two-kernel\033[0m\n",
               vec_ms / two_ms);

    // Single vs best two-kernel variant
    float best_two = (vec_ms < two_ms) ? vec_ms : two_ms;
    const char* best_two_name = (vec_ms < two_ms) ? "Two-kernel-vec" : "Two-kernel";
    if (single_ms < best_two)
        printf("\033[32m  Single-kernel is %.2fx faster than %s\033[0m\n",
               best_two / single_ms, best_two_name);
    else
        printf("\033[33m  %s is %.2fx faster than Single-kernel\033[0m\n",
               best_two_name, single_ms / best_two);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    cudaFree(d_logits);
    cudaFree(d_grad_single);
    cudaFree(d_grad_two);
    cudaFree(d_grad_vec);
    cudaFree(d_grad_out);
    cudaFree(d_targets);
    cudaFree(d_flush);
    cudaDeviceReset();
    return 0;
}
