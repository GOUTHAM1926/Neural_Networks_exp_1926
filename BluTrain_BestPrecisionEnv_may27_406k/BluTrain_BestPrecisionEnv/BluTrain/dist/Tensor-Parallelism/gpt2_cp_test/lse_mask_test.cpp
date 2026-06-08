// =============================================================================
// LSE Masking Consistency Test (Test B)
//
// Verifies that the LSE computed from `tril(scores, 0, -1e9)` in SDPAOp.h
// matches the LSE implied by `fused_tril_softmax(scores, 0)`.
//
// Build:
//   Add to Makefile or compile directly against your library.
//   Example: make lse_test  (after adding the target)
//
// Run:
//   ./lse_test
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/TensorOps.h"

using namespace OwnTensor;

int main() {
    std::cout << "=== LSE Masking Consistency Test ===" << std::endl;

    cudaSetDevice(0);

    // Small scores tensor [1, 1, 4, 4] for easy inspection
    int64_t B = 1, H = 1, T = 4, D = 4;
    Shape scores_shape({{B, H, T, T}});
    TensorOptions opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(DeviceIndex(Device::CUDA, 0));

    // Create scores with known values
    Tensor scores = Tensor::randn<float>(scores_shape, opts, 42, 1.0f);

    // =====================================================
    // Path 1: Standalone LSE (what SDPAOp.h does)
    //   tril(scores, 0, -1e9) -> max -> exp -> sum -> log
    // =====================================================
    int64_t last_dim = scores.ndim() - 1;
    Tensor scores_masked_1e9 = OwnTensor::tril(scores, 0, -1e9);
    Tensor max1 = reduce_max(scores_masked_1e9, {last_dim}, true);
    Tensor exp1 = OwnTensor::exp(scores_masked_1e9 - max1);
    Tensor sum1 = reduce_sum(exp1, {last_dim}, true);
    Tensor lse_standalone = max1 + OwnTensor::log(sum1);

    // =====================================================
    // Path 2: LSE derived from fused_tril_softmax probs
    //   fused_tril_softmax(scores, 0) uses mask_value = 0.0 (default!)
    //   Then: LSE = scores_masked - log(probs)  for any unmasked position
    //   Or equivalently: log(sum(exp(scores_masked)))
    // =====================================================

    // First test: fused_tril_softmax with DEFAULT mask value (0.0)
    Tensor probs_default = autograd::fused_tril_softmax(scores, 0);

    // To get LSE from probs:
    // For the fused kernel with mask=0.0, the effective masked scores are:
    //   S_eff[i] = (i <= row_idx) ? score[i] : 0.0
    // So we replicate that
    Tensor scores_masked_zero = OwnTensor::tril(scores, 0, 0.0);
    Tensor max2 = reduce_max(scores_masked_zero, {last_dim}, true);
    Tensor exp2 = OwnTensor::exp(scores_masked_zero - max2);
    Tensor sum2 = reduce_sum(exp2, {last_dim}, true);
    Tensor lse_from_fused_default = max2 + OwnTensor::log(sum2);

    // =====================================================
    // Path 3: fused_tril_softmax with CORRECT mask value (-1e9)
    // =====================================================
    // This is what fused_tril_softmax SHOULD use for causal attention
    // Note: we pass -1e9 explicitly via the autograd version
    Tensor scores_mut = scores;
    Tensor probs_correct = autograd::fused_tril_softmax(scores_mut, 0, -1e9);

    // =====================================================
    // Print results to CPU for comparison
    // =====================================================
    Tensor lse_standalone_cpu = lse_standalone.to_cpu();
    Tensor lse_fused_default_cpu = lse_from_fused_default.to_cpu();

    Tensor probs_default_cpu = probs_default.to_cpu();
    Tensor probs_correct_cpu = probs_correct.to_cpu();

    std::cout << "\n--- LSE values (per query row) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    float* lse1_ptr = lse_standalone_cpu.data<float>();
    float* lse2_ptr = lse_fused_default_cpu.data<float>();

    float max_diff = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        float v1 = lse1_ptr[t];
        float v2 = lse2_ptr[t];
        float diff = std::abs(v1 - v2);
        max_diff = std::max(max_diff, diff);
        std::cout << "  Row " << t
                  << ": LSE_standalone(mask=-1e9) = " << v1
                  << ", LSE_fused(mask=0.0) = " << v2
                  << ", diff = " << diff << std::endl;
    }

    std::cout << "\nMax LSE difference: " << max_diff << std::endl;

    if (max_diff > 0.01f) {
        std::cout << "\n>>> MISMATCH DETECTED! <<<" << std::endl;
        std::cout << "The standalone LSE (mask=-1e9) and fused_tril_softmax LSE (mask=0.0)" << std::endl;
        std::cout << "are INCONSISTENT. This means the SDPAMerger receives wrong weights." << std::endl;
        std::cout << "\nFIX: Change SDPAOp.h line 85 to pass -1e9 as the mask value:" << std::endl;
        std::cout << "  attn_probs = autograd::fused_tril_softmax(scores, 0, -1e9);" << std::endl;
        std::cout << "Or change the standalone LSE path to use mask=0.0 (but this is wrong for attention)." << std::endl;
    } else {
        std::cout << "\n>>> MATCH <<<" << std::endl;
        std::cout << "LSE values are consistent. The masking issue is NOT the root cause." << std::endl;
    }

    // Also print some probs to show the difference
    std::cout << "\n--- Attention probs row 0 (first 4 cols) ---" << std::endl;
    float* p_default = probs_default_cpu.data<float>();
    float* p_correct = probs_correct_cpu.data<float>();
    std::cout << "  fused(mask=0.0):  ";
    for (int i = 0; i < T; ++i) std::cout << p_default[i] << " ";
    std::cout << std::endl;
    std::cout << "  fused(mask=-1e9): ";
    for (int i = 0; i < T; ++i) std::cout << p_correct[i] << " ";
    std::cout << std::endl;
    std::cout << "  (Row 0 should have prob 1.0 on col 0 and 0.0 elsewhere for causal)" << std::endl;

    return 0;
}
