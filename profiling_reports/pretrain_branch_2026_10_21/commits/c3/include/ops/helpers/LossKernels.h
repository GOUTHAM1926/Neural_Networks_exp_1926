#pragma once

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

// ============================================================================
// DESIGN NOTE — why loss_output and grad_output are fp32, never T
//
// loss_output and grad_output are ALWAYS fp32, whatever dtype the logits are.
// They are one element each, so fp32 costs nothing, and it lets a caller pass
// bf16 logits (the big tensor, where the memory and bandwidth win is) while every
// consumer of the loss still reads a plain float.
//
// THE BUG THIS REMOVES
//   LossBackward.cpp used to bind grad_output's C++ type from the LOGITS:
//       using LogitT = ...decltype(logits_ptr)...;
//       ..._with_stats<LogitT, ...>(..., grad_out_ptr, ...);
//   The trainer hands backward() an fp32 grad_scale. With bf16 logits that fp32
//   value was read as bf16 -- silently, because Tensor::data<T>() is a raw
//   reinterpret_cast with no dtype check. Wrong gradient scale, nothing reported.
//   It predates the SFT work (present in a4c80c1); bf16 logits used to crash in
//   the forward first, which is why it never fired.
//
// THE TWO VALID DESIGNS, and who uses which
//   (1) ASSUME the type, but CHECK it.  <- PyTorch
//       aten/src/ATen/native/cuda/Loss.cu:567-580 binds scalar_t from
//       input.scalar_type() and then reads grad_output.const_data_ptr<scalar_t>().
//       Same assumption we made -- but const_data_ptr calls check_type()
//       (aten/src/ATen/templates/TensorMethods.cpp:22) which THROWS on mismatch.
//   (2) DON'T ASSUME.  <- Liger, Megatron
//       Liger multiplies tensors and lets Triton read the pointer's own dtype
//       (ops/cross_entropy.py:483, ops/utils.py:135 `tl.load(grad_output_ptr)`).
//       Megatron does grad_input.mul_(grad_output.unsqueeze(dim=-1))
//       (core/tensor_parallel/cross_entropy.py:19). Neither asserts a dtype.
//
//   We were doing NEITHER -- assuming without checking, the one combination
//   nobody ships. Declaring these two as float* puts us in category (2), reached
//   through C++ types instead of a dynamic framework: the type is in the
//   signature, so a mismatch is a COMPILE error, not a runtime throw.
//
// MEASURED EFFECT of making these two fp32 (Tests/kernel_tests/
// test_sparse_ce_ignore_index.cu, RTX 3060 sm_86, 2026-09-23). Unlike the memory
// figures quoted in bluscript_zero_wsd.cpp, these ARE measured:
//   bf16 loss error  1.565e-03 -> 1.127e-05   (139x smaller)
//   fp16 loss error  2.007e-04 -> 8.629e-06   ( 23x smaller)
// The old loss scalar was stored as T, so the final value was rounded to the
// logits dtype: 369.1512 came back as 370.0000, a whole bf16 ulp. Keeping the
// scalar in fp32 removes that last rounding -- it now reads 369.1480.
// fp32 callers are unaffected (the store was already float->float).
//
// WHY NOT PyTorch'S ROUTE (option 1) HERE
//   Adding check_type() to Tensor::data<T>() would protect the whole framework,
//   not just this interface. It was rejected FOR NOW, not on merit:
//     - 840 data<T>() call sites, and 59 of them are inside loops
//       (e.g. LossBackward.cpp:273 `saved_targets_.data<int64_t>()[i]` per row),
//       so the check would be paid per iteration on paths that have none today
//     - 38 sites reinterpret ON PURPOSE (data<__nv_bfloat16>, data<__half>,
//       data<half>) so the tensor's bytes can be fed to CUDA intrinsics; a strict
//       check fails to compile there. PyTorch hit the same wall and solved it with
//       an alias allowance (check_type permits a quantized tensor to be read as its
//       underlying type), so we would need an equivalent alias table first
//     - dtype is a RUNTIME property, so a check only surfaces mismatches on paths
//       that actually execute -- it cannot be validated by a test suite that
//       covers one kernel
//   Also worth stating plainly: option 1 DETECTS this bug at runtime, option 2
//   makes it UNREPRESENTABLE at compile time. Compile-time wins where it applies.
//
// RESEARCH / FUTURE TASK — implement and test option 1
//   Add check_type() to Tensor::data<T>() framework-wide, PyTorch-style:
//     a. build the alias table first (bfloat16_t <-> __nv_bfloat16,
//        float16_t <-> __half/half), mirroring PyTorch's isQIntType allowance
//     b. hoist the 59 in-loop data<T>() calls out of their loops first, the way
//        PyTorch uses accessors, so the check is paid once per kernel not per element
//     c. keep it out-of-line like PyTorch does ("Purposely not defined here to
//        avoid inlining", ATen/core/TensorBase.h:650)
//     d. gate it on a FULL-repo build plus the widest test run we have, not on the
//        loss suite -- the failure mode is a runtime throw in code we do not cover
//   Doing it inside loss work would land an 840-site framework change on a branch
//   whose coverage is one cross-entropy kernel.
// ============================================================================

/**
 * @brief Count targets that are NOT ignore_index -- the SFT loss denominator.
 *
 * This is PyTorch's `total_weight` (LossNLL.cpp:288). The forward divides the
 * summed loss by it and the backward divides the gradient by the SAME device
 * scalar, so the two passes can never disagree. Result is clamped to >= 1 so an
 * all-masked batch cannot divide by zero.
 *
 * With ignore_index = -1 the count equals batch_size exactly, so pretraining is
 * unaffected.
 *
 * @param out_count [1] device float, written by this call.
 */
template <typename T_idx>
void count_valid_targets_cuda(const T_idx *targets, int64_t batch_size,
                              int64_t ignore_index, float *out_count,
                              cudaStream_t stream);



template <typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec(
    const T *logits, const T_idx *targets, float *loss_output, int64_t batch_size,
    int64_t vocab_size, cudaStream_t stream,
    int64_t ignore_index =
        -1, // SFT : target value meaning "skip this row" ; -1 = masking off
    const float *denom =
        nullptr // device scalar the summed loss is divided by; nullptr = no divide
);


/**
 * @brief Two-kernel sparse CE backward — vectorized (cp.async + float4 stores).
 * Requires vocab_size % 4 == 0.
 */
template <typename T, typename Tidx>
void sparse_ce_backward_cuda_vec(const T *logits, const Tidx *targets,
                                 T *grad_logits, int64_t batch_size,
                                 int64_t vocab_size, const float *grad_output,
                                 const float *denom, cudaStream_t stream,
                                 int64_t ignore_index = -1);

/**
 * @brief Stats-aware sparse CE forward. Same as ..._impl_vec but ALSO writes
 * per-row (max, sum) to the supplied buffers — these get saved in the
 * autograd node so backward can skip the Reduce kernel.
 *
 * saved_max / saved_sum: pre-allocated [batch_size] float buffers (caller
 * owns).
 */
template <typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec_save_stats(
    const T *logits, const T_idx *targets, float *loss_output, float *saved_max,
    float *saved_sum, int64_t batch_size, int64_t vocab_size,
    cudaStream_t stream,
    int64_t ignore_index = -1, // SFT: "skip this row"; -1 = masking off
    const float *denom = nullptr); // device scalar to divide the summed loss by

/**
 * @brief Stats-aware sparse CE backward. Reuses (max, sum) saved by forward
 * and runs ONLY the Normalize kernel — Reduce kernel skipped, ~3.7 ms / call
 * faster than sparse_ce_backward_cuda_vec.
 */
template <typename T, typename Tidx>
void sparse_ce_backward_cuda_vec_with_stats(
    const T *logits, const Tidx *targets, T *grad_logits,
    const float *saved_max, const float *saved_sum, int64_t batch_size,
    int64_t vocab_size, const float *grad_output, const float *denom,
    cudaStream_t stream,
    int64_t ignore_index =
        -1); // SFT : target value meaning "skip this row"; -1 = masking off

/**
 * @brief The same stats-aware backward, with the gradient written OVER the
 * logits (logits_grad is read, then overwritten). No second [rows x vocab]
 * buffer; the gradient is bit-identical to the out-of-place call's. Only for a
 * caller that never needs the logits again (2026-10-17; since 2026-10-21 the
 * only path autograd::sparse_cross_entropy_loss takes on the CUDA training path).
 */
template <typename T, typename Tidx>
void sparse_ce_backward_cuda_vec_with_stats_inplace(
    T *logits_grad, const Tidx *targets, const float *saved_max,
    const float *saved_sum, int64_t batch_size, int64_t vocab_size,
    const float *grad_output, const float *denom, cudaStream_t stream,
    int64_t ignore_index = -1);


/**
 * @brief Compute gradient for Sparse Cross Entropy with Logits.
 *
 * grad = (softmax(logits) - targets_one_hot) * scale
 */
template <typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T *logits, const T_idx *targets, T *grad_logits, int64_t batch_size,
    int64_t vocab_size, const float *grad_output, const float *denom,
    cudaStream_t stream,
    int64_t ignore_index =
        -1); // SFT : target value meaning "skip this row"; -1 = masking off

// Categorical Cross Entropy (Targets are probabilities [N, C])
// Forward: loss = -1/N * sum(target * log(clip(pred)))
void categorical_cross_entropy_forward_cuda(const float *predictions,
                                            const float *targets,
                                            float *loss_output,
                                            int64_t batch_size,
                                            int64_t num_classes);

// Backward: grad = -1/N * grad_output * target / pred
void categorical_cross_entropy_backward_cuda(
    const float *grad_output, const float *predictions, const float *targets,
    float *grad_input, int64_t batch_size, int64_t num_classes);

// MSE Loss
// Forward: loss = mean((pred - target)^2)
void mse_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel);

// Backward: grad = 2/N * (pred - target) * grad_output
void mse_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel);

// MAE Loss
// Forward: loss = mean(|pred - target|)
void mae_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel);

// Backward: grad = 1/N * sign(pred - target) * grad_output
void mae_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel);

// Binary Cross Entropy (BCE)
// Forward: loss = -mean(target * log(clip(pred)) + (1-target) *
// log(1-clip(pred)))
void bce_loss_forward_cuda(const float *predictions, const float *targets,
                           float *loss_output, int64_t numel);

// Backward: grad = 1/N * (-target/pred + (1-target)/(1-pred)) * grad_output
void bce_loss_backward_cuda(const float *grad_output, const float *predictions,
                            const float *targets, float *grad_input,
                            int64_t numel);

} // namespace cuda
} // namespace OwnTensor
