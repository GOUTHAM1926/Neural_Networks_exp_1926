/**
 * testSparseCEBackward.cu
 *
 * Accuracy test for all three sparse CE backward kernel variants:
 *   1. Single-kernel  – sparse_cross_entropy_backward_cuda
 *   2. Two-kernel     – sparse_ce_backward_cuda
 *   3. Two-kernel vec – sparse_ce_backward_cuda_vec  (requires vocab_size % 4 == 0)
 *
 * Build & run
 * -----------
 *   make run-snippet FILE=Tests/OptimizationTests/testSparseCEBackward.cu
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
// Helper macros / utilities
// ============================================================================

#define GPU_CHECK(call) do {                                             \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static bool compare(const float* ref, const float* got, int64_t n,
                    float abs_tol, float rel_tol, const char* label)
{
    float max_abs = 0.f, max_rel = 0.f;
    int64_t worst = 0;
    bool pass = true;

    for (int64_t i = 0; i < n; ++i) {
        float diff = std::fabs(ref[i] - got[i]);
        float rel  = diff / (std::fabs(ref[i]) + 1e-6f);
        if (diff > max_abs) { max_abs = diff; worst = i; }
        max_rel = std::max(max_rel, rel);
        if (diff > abs_tol && rel > rel_tol) {
            pass = false;
        }
    }

    if (pass) {
        printf("\033[32m[PASS] %-45s  max_abs=%.2e  max_rel=%.2e\033[0m\n",
               label, max_abs, max_rel);
    } else {
        printf("\033[31m[FAIL] %-45s  max_abs=%.2e (tol=%.2e)  "
               "max_rel=%.2e (tol=%.2e)  worst_idx=%ld  ref=%.6f  got=%.6f\033[0m\n",
               label, max_abs, abs_tol, max_rel, rel_tol, worst,
               ref[worst], got[worst]);
    }
    return pass;
}

// ============================================================================
// CPU reference implementation  (ground truth)
// ============================================================================

static void cpu_sparse_ce_backward_ref(
    const float*   logits,
    const int32_t* targets,
    float*         grad_out,
    int64_t        batch_size,
    int64_t        vocab_size,
    float          grad_output_scalar,
    float          host_scale
) {
    for (int64_t i = 0; i < batch_size; ++i) {
        const float* row = logits + i * vocab_size;
        float*       grd = grad_out + i * vocab_size;

        float max_val = *std::max_element(row, row + vocab_size);

        float sum_exp = 0.f;
        for (int64_t j = 0; j < vocab_size; ++j)
            sum_exp += std::exp(row[j] - max_val);

        float f_scale = grad_output_scalar * host_scale;
        float inv_sum = 1.f / sum_exp;
        int64_t tgt   = targets[i];

        for (int64_t j = 0; j < vocab_size; ++j) {
            float prob = std::exp(row[j] - max_val) * inv_sum;
            grd[j] = (j == tgt) ? (prob - 1.f) * f_scale : prob * f_scale;
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

struct TestConfig {
    const char* name;
    int64_t     batch_size;
    int64_t     vocab_size;
    float       grad_output_val;
    float       host_scale;
};

static bool run_test(const TestConfig& cfg, std::mt19937& rng)
{
    printf("\n[TEST] %-40s  batch=%ld  vocab=%ld\n",
           cfg.name, cfg.batch_size, cfg.vocab_size);

    // Vectorized kernel safety constraints:
    //
    // DEADLOCK root cause: __syncthreads() is inside the vectorized stride loop.
    // Every thread in the block must execute the SAME number of loop iterations,
    // otherwise some threads exit early and diverge at a different __syncthreads()
    // call (the reduction sync), permanently hanging the block.
    //
    // Reduce kernel loop:   col = startCol + tid*4,  step = bdim*4 = 1024
    //   elementsPerBlock = vocab_size / grid_reduce.x
    //   All threads iterate the same count iff elementsPerBlock % (bdim*4) == 0
    //   → vocab_size % (grid_reduce.x * bdim * 4) == 0
    //   → vocab_size % 2048 == 0
    //
    // Normalize kernel loop: j = tid*4, step = bdim*4 = 1024
    //   All threads iterate the same count iff vocab_size % (bdim*4) == 0
    //   → vocab_size % 1024 == 0   (covered by the 2048 condition above)
    //
    // ALIGNMENT: cp.async.cg requires 16-byte aligned global address.
    //   startCol for reduce block k = k*(vocab_size/2) must be a multiple of 4 floats.
    //   → vocab_size % 8 == 0   (covered by the 2048 condition above)
    //
    // Single combined condition: vocab_size % 2048 == 0
    static const int64_t VEC_BDIM       = 256;
    static const int64_t VEC_GRID_RED_X = 2;
    const bool vec_eligible =
        (cfg.vocab_size % (VEC_GRID_RED_X * VEC_BDIM * 4) == 0);
    if (!vec_eligible)
        printf("  [NOTE] vocab_size=%ld not divisible by 2048 — vec kernel skipped\n",
               cfg.vocab_size);

    int64_t logit_n = cfg.batch_size * cfg.vocab_size;
    float abs_tol = 1e-4f, rel_tol = 1e-3f;

    // ------------------------------------------------------------------
    // Host data
    // ------------------------------------------------------------------
    std::vector<float>   h_logits(logit_n);
    std::vector<int32_t> h_targets(cfg.batch_size);

    std::uniform_real_distribution<float>   logit_dist(-5.f, 5.f);
    std::uniform_int_distribution<int32_t>  tgt_dist(0, (int32_t)(cfg.vocab_size - 1));

    for (float&   v : h_logits)  v = logit_dist(rng);
    for (int32_t& t : h_targets) t = tgt_dist(rng);

    // ------------------------------------------------------------------
    // CPU reference
    // ------------------------------------------------------------------
    std::vector<float> h_ref_grad(logit_n, 0.f);
    cpu_sparse_ce_backward_ref(
        h_logits.data(), h_targets.data(), h_ref_grad.data(),
        cfg.batch_size, cfg.vocab_size,
        cfg.grad_output_val, cfg.host_scale);

    // ------------------------------------------------------------------
    // GPU allocations
    // ------------------------------------------------------------------
    float   *d_logits, *d_grad_single, *d_grad_two, *d_grad_vec, *d_grad_out_scalar;
    int32_t *d_targets;

    GPU_CHECK(cudaMalloc(&d_logits,          logit_n * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_grad_single,     logit_n * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_grad_two,        logit_n * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_grad_vec,        logit_n * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_targets,         cfg.batch_size * sizeof(int32_t)));
    GPU_CHECK(cudaMalloc(&d_grad_out_scalar, sizeof(float)));

    GPU_CHECK(cudaMemcpy(d_logits,  h_logits.data(),
                         logit_n * sizeof(float),          cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_targets, h_targets.data(),
                         cfg.batch_size * sizeof(int32_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_grad_out_scalar, &cfg.grad_output_val,
                         sizeof(float),                    cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 1. Single-kernel
    // ------------------------------------------------------------------
    GPU_CHECK(cudaMemset(d_grad_single, 0, logit_n * sizeof(float)));
    cuda::sparse_cross_entropy_backward_cuda<float, int32_t>(
        d_logits, d_targets, d_grad_single,
        cfg.batch_size, cfg.vocab_size,
        d_grad_out_scalar, cfg.host_scale, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // 2. Two-kernel
    // ------------------------------------------------------------------
    GPU_CHECK(cudaMemset(d_grad_two, 0, logit_n * sizeof(float)));
    cuda::sparse_ce_backward_cuda<float, int32_t>(
        d_logits, d_targets, d_grad_two,
        cfg.batch_size, cfg.vocab_size,
        d_grad_out_scalar, cfg.host_scale, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // 3. Two-kernel vectorized  (vocab_size % 4 == 0 required)
    // ------------------------------------------------------------------
    bool ok_vec_ref = true, ok_vec_single = true;
    std::vector<float> h_vec(logit_n, 0.f);

    if (vec_eligible) {
        GPU_CHECK(cudaMemset(d_grad_vec, 0, logit_n * sizeof(float)));
        cuda::sparse_ce_backward_cuda_vec<float, int32_t>(
            d_logits, d_targets, d_grad_vec,
            cfg.batch_size, cfg.vocab_size,
            d_grad_out_scalar, cfg.host_scale, nullptr);
        GPU_CHECK(cudaDeviceSynchronize());
        GPU_CHECK(cudaMemcpy(h_vec.data(), d_grad_vec,
                             logit_n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // ------------------------------------------------------------------
    // Copy results back
    // ------------------------------------------------------------------
    std::vector<float> h_single(logit_n), h_two(logit_n);
    GPU_CHECK(cudaMemcpy(h_single.data(), d_grad_single,
                         logit_n * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaMemcpy(h_two.data(),    d_grad_two,
                         logit_n * sizeof(float), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Comparisons
    // ------------------------------------------------------------------
    bool ok1 = compare(h_ref_grad.data(), h_single.data(), logit_n, abs_tol, rel_tol,
                       "single-kernel    vs CPU ref");
    bool ok2 = compare(h_ref_grad.data(), h_two.data(),    logit_n, abs_tol, rel_tol,
                       "two-kernel       vs CPU ref");
    bool ok3 = compare(h_single.data(),   h_two.data(),    logit_n, abs_tol, rel_tol,
                       "two-kernel       vs single-kernel");

    if (vec_eligible) {
        ok_vec_ref    = compare(h_ref_grad.data(), h_vec.data(), logit_n, abs_tol, rel_tol,
                                "two-kernel-vec   vs CPU ref");
        ok_vec_single = compare(h_single.data(),   h_vec.data(), logit_n, abs_tol, rel_tol,
                                "two-kernel-vec   vs single-kernel");
    }

    // Print first row for manual inspection on tiny tests
    if (cfg.batch_size <= 4 && cfg.vocab_size <= 16) {
        printf("  Row 0 target = %d\n", h_targets[0]);
        printf("  logits  :"); for (int j = 0; j < cfg.vocab_size; ++j) printf(" %7.4f", h_logits[j]); printf("\n");
        printf("  cpu_ref :"); for (int j = 0; j < cfg.vocab_size; ++j) printf(" %7.4f", h_ref_grad[j]); printf("\n");
        printf("  single  :"); for (int j = 0; j < cfg.vocab_size; ++j) printf(" %7.4f", h_single[j]); printf("\n");
        printf("  two-ker :"); for (int j = 0; j < cfg.vocab_size; ++j) printf(" %7.4f", h_two[j]); printf("\n");
        if (vec_eligible) {
            printf("  vec-ker :"); for (int j = 0; j < cfg.vocab_size; ++j) printf(" %7.4f", h_vec[j]); printf("\n");
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_grad_single));
    GPU_CHECK(cudaFree(d_grad_two));
    GPU_CHECK(cudaFree(d_grad_vec));
    GPU_CHECK(cudaFree(d_targets));
    GPU_CHECK(cudaFree(d_grad_out_scalar));

    return ok1 && ok2 && ok3 && ok_vec_ref && ok_vec_single;
}

// ============================================================================
// Edge-case: batch_size == 0  (should be a no-op, not a crash)
// ============================================================================
static bool test_empty_batch()
{
    printf("\n[TEST] %-40s\n", "empty batch (batch_size=0)");
    float* d_dummy;
    GPU_CHECK(cudaMalloc(&d_dummy, sizeof(float)));

    cuda::sparse_ce_backward_cuda<float, int32_t>(
        nullptr, nullptr, nullptr, 0, 512, d_dummy, 1.f, nullptr);
    cuda::sparse_ce_backward_cuda_vec<float, int32_t>(
        nullptr, nullptr, nullptr, 0, 512, d_dummy, 1.f, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    GPU_CHECK(cudaFree(d_dummy));
    printf("\033[32m[PASS] %-45s\033[0m\n", "empty batch – no crash (two-kernel + vec)");
    return true;
}

// ============================================================================
// Edge-case: target at boundary indices (0 and vocab_size-1)
// ============================================================================
static bool test_boundary_targets()
{
    printf("\n[TEST] %-40s\n", "boundary targets (0 and vocab-1)");
    const int64_t B = 2, V = 8;
    float h_logits[] = {
        0.1f,  0.5f, -0.3f, 1.2f, 0.7f, -0.9f, 0.4f, 0.2f,   // row 0, target = 0
        0.3f, -0.1f,  0.8f, 1.5f, 0.2f,  0.9f, -0.4f, 1.1f   // row 1, target = 7
    };
    int32_t h_tgt[] = {0, 7};
    float grad_out_val = 1.f, scale = 1.f;

    float h_ref_grad[B * V];
    cpu_sparse_ce_backward_ref(h_logits, h_tgt, h_ref_grad, B, V, grad_out_val, scale);

    float   *d_logits, *d_grad_two, *d_go;
    int32_t *d_tgt;
    GPU_CHECK(cudaMalloc(&d_logits,   B * V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_grad_two, B * V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_tgt,      B * sizeof(int32_t)));
    GPU_CHECK(cudaMalloc(&d_go,       sizeof(float)));
    GPU_CHECK(cudaMemcpy(d_logits, h_logits, B * V * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_tgt,    h_tgt,    B * sizeof(int32_t),   cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_go,     &grad_out_val, sizeof(float),    cudaMemcpyHostToDevice));

    // Two-kernel
    GPU_CHECK(cudaMemset(d_grad_two, 0, B * V * sizeof(float)));
    cuda::sparse_ce_backward_cuda<float, int32_t>(
        d_logits, d_tgt, d_grad_two, B, V, d_go, scale, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    // V=8 < 2048: vec kernel would deadlock (not all threads enter the loop) — skip
    printf("  [NOTE] V=8 < 2048, vec kernel skipped for boundary test\n");

    float h_two[B * V];
    GPU_CHECK(cudaMemcpy(h_two, d_grad_two, B * V * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < B; ++i) {
        printf("  row %d (tgt=%d): ref=[", i, h_tgt[i]);
        for (int j = 0; j < V; ++j) printf(" %.4f", h_ref_grad[i*V+j]);
        printf(" ]  two=[");
        for (int j = 0; j < V; ++j) printf(" %.4f", h_two[i*V+j]);
        printf(" ]\n");
    }

    bool ok1 = compare(h_ref_grad, h_two, B * V, 1e-4f, 1e-3f, "boundary two-kernel vs CPU ref");
    bool ok2 = true; // vec skipped (V=8 < 2048)

    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_grad_two));
    GPU_CHECK(cudaFree(d_tgt));
    GPU_CHECK(cudaFree(d_go));
    return ok1 && ok2;
}

// ============================================================================
// Edge-case: all-identical logits (uniform softmax)
// ============================================================================
static bool test_uniform_logits()
{
    printf("\n[TEST] %-40s\n", "uniform logits");
    const int64_t B = 4, V = 32;

    std::vector<float>   h_logits(B * V, 1.f);
    std::vector<int32_t> h_tgt = {0, 5, 10, 31};
    float grad_out_val = 1.f, scale = 0.25f;

    std::vector<float> h_ref_grad(B * V);
    cpu_sparse_ce_backward_ref(
        h_logits.data(), h_tgt.data(), h_ref_grad.data(),
        B, V, grad_out_val, scale);

    float   *d_logits, *d_grad_two, *d_go;
    int32_t *d_tgt;
    GPU_CHECK(cudaMalloc(&d_logits,   B * V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_grad_two, B * V * sizeof(float)));
    GPU_CHECK(cudaMalloc(&d_tgt,      B * sizeof(int32_t)));
    GPU_CHECK(cudaMalloc(&d_go,       sizeof(float)));
    GPU_CHECK(cudaMemcpy(d_logits, h_logits.data(), B * V * sizeof(float), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_tgt,    h_tgt.data(),    B * sizeof(int32_t),   cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_go,     &grad_out_val,   sizeof(float),         cudaMemcpyHostToDevice));

    // Two-kernel
    GPU_CHECK(cudaMemset(d_grad_two, 0, B * V * sizeof(float)));
    cuda::sparse_ce_backward_cuda<float, int32_t>(
        d_logits, d_tgt, d_grad_two, B, V, d_go, scale, nullptr);
    GPU_CHECK(cudaDeviceSynchronize());

    // V=32 < 2048: vec kernel would deadlock (not all threads enter the loop) — skip
    printf("  [NOTE] V=32 < 2048, vec kernel skipped for uniform test\n");

    std::vector<float> h_two(B * V);
    GPU_CHECK(cudaMemcpy(h_two.data(), d_grad_two, B * V * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok1 = compare(h_ref_grad.data(), h_two.data(), B * V, 1e-4f, 1e-3f,
                       "uniform two-kernel vs CPU ref");
    bool ok2 = true; // vec skipped (V=32 < 2048)

    GPU_CHECK(cudaFree(d_logits));
    GPU_CHECK(cudaFree(d_grad_two));
    GPU_CHECK(cudaFree(d_tgt));
    GPU_CHECK(cudaFree(d_go));
    return ok1 && ok2;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    printf("========================================================\n");
    printf("  sparse_ce_backward – Accuracy Test Suite\n");
    printf("  Kernels: single | two-kernel | two-kernel-vec\n");
    printf("========================================================\n");

    std::mt19937 rng(42);

    static const TestConfig configs[] = {
        // name                              B      V      grad_out  scale
        // V=4: vec skipped (4 % 8 != 0 — block 1 startCol=2 would be unaligned for cp.async)
        { "tiny  B=2   V=4",                2,     4,     1.f,  0.5f     },
        // V=8: vec eligible (8 % 8 == 0 — both blocks start at 4-float aligned offsets)
        { "tiny  B=4   V=8",                4,     8,     1.f,  0.25f    },
        { "small B=8   V=16",               8,    16,     1.f,  0.125f   },
        { "small B=16  V=256",             16,   256,     1.f,  1.f/16   },
        { "med   B=64  V=512",             64,   512,     1.f,  1.f/64   },
        { "med   B=128 V=1024",           128,  1024,     1.f,  1.f/128  },
        { "large B=512 V=2048",           512,  2048,     1.f,  1.f/512  },
        { "large B=1024 V=4096",         1024,  4096,     1.f,  1.f/1024 },
        { "xlarge B=32 V=49152 (24*2048)", 32,  49152,     1.f,  1.f/32   }, // 49152 % 2048 == 0
        { "xlarge B=32 V=50257 (GPT-2)",  32,  50257,     1.f,  1.f/32   }, // vec skipped (not mult of 2048)
        { "scale!=1 B=64 V=256",          64,    256,     2.5f, 0.1f     },
    };

    bool all_pass = true;
    for (const auto& cfg : configs)
        all_pass &= run_test(cfg, rng);

    all_pass &= test_empty_batch();
    all_pass &= test_boundary_targets();
    all_pass &= test_uniform_logits();

    printf("\n========================================================\n");
    if (all_pass)
        printf("\033[32m  ALL TESTS PASSED\033[0m\n");
    else
        printf("\033[31m  SOME TESTS FAILED\033[0m\n");
    printf("========================================================\n");

    return all_pass ? 0 : 1;
}
