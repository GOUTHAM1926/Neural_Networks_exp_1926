/**
 * testSparseCEForward.cu
 *
 * Accuracy test for sparse CE forward kernel variants:
 *   1. Non-vec  – sparse_cross_entropy_forward_cuda
 *   2. Vec      – sparse_cross_entropy_forward_cuda_vec  (cp.async + per-thread staging)
 *
 * Both functions output the SUM of per-sample losses (not mean).
 * CPU reference computes the same sum for ground truth.
 *
 * NOTE: The vec kernel has no __syncthreads() inside its vectorized loop and has
 * a built-in runtime alignment fallback, so it is safe for any vocab_size.
 *
 * Build & run
 * -----------
 *   make run-snippet FILE=Tests/OptimizationTests/testSparseCEForward.cu
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "ops/helpers/LossKernels.h"
#include "utils/KernelUtils.cuh"

using namespace OwnTensor;

// ============================================================================
// Helpers
// ============================================================================

#define GPU_CHECK(call) do {                                             \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// Comparison on the normalised (per-sample mean) loss to keep tolerances
// independent of batch size.
static bool compare_scalar(float ref_sum, float got_sum, int64_t batch_size,
                            float abs_tol, float rel_tol, const char* label)
{
    float ref  = ref_sum / (float)batch_size;
    float got  = got_sum / (float)batch_size;
    float diff = std::fabs(ref - got);
    float rel  = diff / (std::fabs(ref) + 1e-6f);
    bool  pass = (diff <= abs_tol || rel <= rel_tol);

    if (pass)
        printf("\033[32m[PASS] %-50s  ref=%.6f  got=%.6f  diff=%.2e  rel=%.2e\033[0m\n",
               label, ref, got, diff, rel);
    else
        printf("\033[31m[FAIL] %-50s  ref=%.6f  got=%.6f  diff=%.2e (tol=%.2e)"
               "  rel=%.2e (tol=%.2e)\033[0m\n",
               label, ref, got, diff, abs_tol, rel, rel_tol);
    return pass;
}

// ============================================================================
// CPU reference  (numerically stable log-softmax loss)
// ============================================================================

static float cpu_sparse_ce_forward_ref(
    const float*   logits,
    const int32_t* targets,
    int64_t        batch_size,
    int64_t        vocab_size
) {
    float total = 0.f;
    for (int64_t i = 0; i < batch_size; ++i) {
        const float* row = logits + i * vocab_size;
        float max_val = *std::max_element(row, row + vocab_size);
        float sum_exp = 0.f;
        for (int64_t j = 0; j < vocab_size; ++j)
            sum_exp += std::exp(row[j] - max_val);
        float loss = std::log(sum_exp) + max_val - row[targets[i]];
        total += loss;
    }
    return total;
}

// ============================================================================
// Test harness
// ============================================================================

struct TestConfig {
    const char* name;
    int64_t     batch_size;
    int64_t     vocab_size;
};

static bool run_test(const TestConfig& cfg, std::mt19937& rng)
{
    printf("\n[TEST] %-42s  batch=%ld  vocab=%ld\n",
           cfg.name, cfg.batch_size, cfg.vocab_size);

    const float abs_tol = 1e-3f;
    const float rel_tol = 1e-3f;

    int64_t logit_n = cfg.batch_size * cfg.vocab_size;

    // ------------------------------------------------------------------
    // Host data
    // ------------------------------------------------------------------
    std::vector<float>   h_logits(logit_n);
    std::vector<int32_t> h_targets(cfg.batch_size);

    std::uniform_real_distribution<float>  logit_dist(-5.f, 5.f);
    std::uniform_int_distribution<int32_t> tgt_dist(0, (int32_t)(cfg.vocab_size - 1));

    for (float&   v : h_logits)  v = logit_dist(rng);
    for (int32_t& t : h_targets) t = tgt_dist(rng);

    // ------------------------------------------------------------------
    // CPU reference
    // ------------------------------------------------------------------
    float ref_sum = cpu_sparse_ce_forward_ref(
        h_logits.data(), h_targets.data(), cfg.batch_size, cfg.vocab_size);

    // ------------------------------------------------------------------
    // GPU allocations
    // ------------------------------------------------------------------
    float   *d_logits, *d_loss_nonvec, *d_loss_vec;
    int32_t *d_targets;

    GPU_CHECK(cudaMalloc(&d_logits,     logit_n        * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_nonvec, sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_vec,   sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_targets,    cfg.batch_size * sizeof(int32_t)));

    GPU_CHECK(cudaMemcpy(d_logits,  h_logits.data(),
                         logit_n * sizeof(float),          cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_targets, h_targets.data(),
                         cfg.batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 1. Non-vec kernel
    // ------------------------------------------------------------------
    GPU_CHECK(cudaMemset(d_loss_nonvec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
        d_logits, d_targets, d_loss_nonvec,
        cfg.batch_size, cfg.vocab_size, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    float h_loss_nonvec = 0.f;
    GPU_CHECK(cudaMemcpy(&h_loss_nonvec, d_loss_nonvec, sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // 2. Vec kernel
    // ------------------------------------------------------------------
    GPU_CHECK(cudaMemset(d_loss_vec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
        d_logits, d_targets, d_loss_vec,
        cfg.batch_size, cfg.vocab_size, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    float h_loss_vec = 0.f;
    GPU_CHECK(cudaMemcpy(&h_loss_vec, d_loss_vec, sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Comparisons  (all on per-sample mean for size-independent thresholds)
    // ------------------------------------------------------------------
    bool ok1 = compare_scalar(ref_sum,       h_loss_nonvec, cfg.batch_size,
                               abs_tol, rel_tol, "non-vec  vs CPU ref");
    bool ok2 = compare_scalar(ref_sum,       h_loss_vec,    cfg.batch_size,
                               abs_tol, rel_tol, "vec      vs CPU ref");
    bool ok3 = compare_scalar(h_loss_nonvec, h_loss_vec,    cfg.batch_size,
                               abs_tol, rel_tol, "vec      vs non-vec");

    if (cfg.batch_size == 1) {
        printf("  single-sample: ref=%.6f  non-vec=%.6f  vec=%.6f\n",
               ref_sum, h_loss_nonvec, h_loss_vec);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_loss_nonvec));
    GPU_CHECK(cudaFree(d_loss_vec));
    GPU_CHECK(cudaFree(d_targets));

    return ok1 && ok2 && ok3;
}

// ============================================================================
// Edge-case: batch_size == 0  (should be a no-op, not a crash)
// ============================================================================
static bool test_empty_batch()
{
    printf("\n[TEST] %-42s\n", "empty batch (batch_size=0)");
    float *d_loss, *d_dummy_logits;
    int32_t *d_dummy_tgt;
    GPU_CHECK(cudaMalloc(&d_loss,        sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_dummy_logits, sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_dummy_tgt,    sizeof(int32_t)));

    cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
        d_dummy_logits, d_dummy_tgt, d_loss, 0, 512, nullptr);
    cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
        d_dummy_logits, d_dummy_tgt, d_loss, 0, 512, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    GPU_CHECK(cudaFree(d_loss));
    GPU_CHECK(cudaFree(d_dummy_logits));
    GPU_CHECK(cudaFree(d_dummy_tgt));
    printf("\033[32m[PASS] %-50s\033[0m\n", "empty batch – no crash");
    return true;
}

// ============================================================================
// Edge-case: batch_size == 1 (single sample, full per-sample visibility)
// ============================================================================
static bool test_single_sample()
{
    printf("\n[TEST] %-42s\n", "single sample (batch_size=1)");
    const int64_t V = 8;
    float h_logits[] = { 0.1f, 2.3f, -1.4f, 0.7f, 1.1f, -0.6f, 3.2f, 0.5f };
    int32_t h_tgt = 6;  // target = argmax, so loss should be small

    float ref = cpu_sparse_ce_forward_ref(h_logits, &h_tgt, 1, V);

    float   *d_logits, *d_loss_nonvec, *d_loss_vec;
    int32_t *d_tgt;
    GPU_CHECK(cudaMalloc(&d_logits,     V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_nonvec, sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_vec,   sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_tgt,         sizeof(int32_t)));
    GPU_CHECK(cudaMemcpy(d_logits, h_logits, V * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_tgt,    &h_tgt,   sizeof(int32_t),   cudaMemcpyHostToDevice));

    GPU_CHECK(cudaMemset(d_loss_nonvec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
        d_logits, d_tgt, d_loss_nonvec, 1, V, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    GPU_CHECK(cudaMemset(d_loss_vec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
        d_logits, d_tgt, d_loss_vec, 1, V, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    float h_nonvec = 0.f, h_vec = 0.f;
    GPU_CHECK(cudaMemcpy(&h_nonvec, d_loss_nonvec, sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(&h_vec,    d_loss_vec,    sizeof(float), cudaMemcpyDeviceToHost));

    printf("  ref=%.6f  non-vec=%.6f  vec=%.6f\n", ref, h_nonvec, h_vec);

    bool ok1 = compare_scalar(ref,     h_nonvec, 1, 1e-4f, 1e-4f, "single non-vec vs CPU ref");
    bool ok2 = compare_scalar(ref,     h_vec,    1, 1e-4f, 1e-4f, "single vec     vs CPU ref");
    bool ok3 = compare_scalar(h_nonvec, h_vec,   1, 1e-5f, 1e-5f, "single vec     vs non-vec");

    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_loss_nonvec));
    GPU_CHECK(cudaFree(d_loss_vec));
    GPU_CHECK(cudaFree(d_tgt));
    return ok1 && ok2 && ok3;
}

// ============================================================================
// Edge-case: uniform logits (softmax = 1/V, loss = log(V))
// ============================================================================
static bool test_uniform_logits()
{
    printf("\n[TEST] %-42s\n", "uniform logits  (loss should equal log(V))");
    const int64_t B = 8, V = 64;

    std::vector<float>   h_logits(B * V, 0.f);  // all zeros → uniform softmax
    std::vector<int32_t> h_tgt(B, 0);
    float expected_per_sample = std::log((float)V);
    float expected_sum        = expected_per_sample * (float)B;

    float ref = cpu_sparse_ce_forward_ref(h_logits.data(), h_tgt.data(), B, V);
    printf("  expected_sum=%.6f  cpu_ref=%.6f\n", expected_sum, ref);

    float   *d_logits, *d_loss_nonvec, *d_loss_vec;
    int32_t *d_tgt;
    GPU_CHECK(cudaMalloc(&d_logits,     B * V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_nonvec, sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_loss_vec,   sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_tgt,         B * sizeof(int32_t)));
    GPU_CHECK(cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_tgt,    h_tgt.data(),    B * sizeof(int32_t),   cudaMemcpyHostToDevice));

    GPU_CHECK(cudaMemset(d_loss_nonvec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda<float, int32_t>(
        d_logits, d_tgt, d_loss_nonvec, B, V, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    GPU_CHECK(cudaMemset(d_loss_vec, 0, sizeof(float)));
    cuda::sparse_cross_entropy_forward_cuda_vec<float, int32_t>(
        d_logits, d_tgt, d_loss_vec, B, V, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    float h_nonvec = 0.f, h_vec = 0.f;
    GPU_CHECK(cudaMemcpy(&h_nonvec, d_loss_nonvec, sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(&h_vec,    d_loss_vec,    sizeof(float), cudaMemcpyDeviceToHost));

    printf("  non-vec=%.6f  vec=%.6f\n", h_nonvec, h_vec);

    bool ok1 = compare_scalar(ref,     h_nonvec, B, 1e-4f, 1e-4f, "uniform non-vec vs CPU ref");
    bool ok2 = compare_scalar(ref,     h_vec,    B, 1e-4f, 1e-4f, "uniform vec     vs CPU ref");
    bool ok3 = compare_scalar(h_nonvec, h_vec,   B, 1e-5f, 1e-5f, "uniform vec     vs non-vec");

    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_loss_nonvec));
    GPU_CHECK(cudaFree(d_loss_vec));
    GPU_CHECK(cudaFree(d_tgt));
    return ok1 && ok2 && ok3;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    printf("========================================================\n");
    printf("  sparse_ce_forward – Accuracy Test Suite\n");
    printf("  Kernels: non-vec | vec (cp.async per-thread staging)\n");
    printf("========================================================\n");

    std::mt19937 rng(42);

    static const TestConfig configs[] = {
        // name                               B       V
        { "tiny   B=1   V=4",                1,      4  },
        { "tiny   B=2   V=4",                2,      4  },
        { "tiny   B=4   V=8",                4,      8  },
        { "small  B=8   V=16",               8,      16 },
        { "small  B=16  V=256",             16,     256 },
        { "med    B=64  V=512",             64,     512 },
        { "med    B=128 V=1024",           128,    1024 },
        { "large  B=512 V=2048",           512,    2048 },
        { "large  B=1024 V=4096",         1024,    4096 },
        { "xlarge B=32  V=50257 (GPT-2)",   32,   50257 },  // non-multiple of 4 — fallback path
        { "xlarge B=32  V=32768",           32,   32768 },  // aligned path
        { "xlarge B=64  V=49152",           64,   49152 },
    };

    bool all_pass = true;
    for (const auto& cfg : configs)
        all_pass &= run_test(cfg, rng);

    all_pass &= test_empty_batch();
    all_pass &= test_single_sample();
    all_pass &= test_uniform_logits();

    printf("\n========================================================\n");
    if (all_pass)
        printf("\033[32m  ALL TESTS PASSED\033[0m\n");
    else
        printf("\033[31m  SOME TESTS FAILED\033[0m\n");
    printf("========================================================\n");

    return all_pass ? 0 : 1;
}
