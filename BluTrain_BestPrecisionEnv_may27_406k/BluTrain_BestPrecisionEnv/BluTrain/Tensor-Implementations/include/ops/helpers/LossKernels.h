#pragma once

#include "core/Tensor.h"
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

/**
 * @brief Compute sparse cross entropy loss (forward pass).
 *
 * loss = sum(-log(softmax(logits)[targets])) / batch_size
 * Uses numerically stable log-softmax computation.
 *
 * @param logits [batch_size, vocab_size] - Input logits
 * @param targets [batch_size] - Target indices
 * @param loss_output [1] - Output scalar loss (sum of all sample losses)
 * @param batch_size Number of samples
 * @param vocab_size Number of classes
 * @param stream CUDA stream for asynchronous execution
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

/**
 * @brief Vectorized sparse CE forward (cp.async 128-bit loads, per-thread staging).
 * Safe for any vocab_size — has runtime alignment check with scalar fallback.
 * Output is the sum of per-sample losses (same as sparse_cross_entropy_forward_cuda).
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_vec(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

/**
 * @brief Two-kernel sparse CE backward — vectorized (cp.async + float4 stores).
 * Requires vocab_size % 4 == 0.
 */
template<typename T, typename Tidx>
void sparse_ce_backward_cuda_vec(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
);

/**
 * @brief Stats-aware sparse CE forward. Same as ..._impl_vec but ALSO writes
 * per-row (max, sum) to the supplied buffers — these get saved in the
 * autograd node so backward can skip the Reduce kernel.
 *
 * saved_max / saved_sum: pre-allocated [batch_size] float buffers (caller owns).
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda_impl_vec_save_stats(
    const T* logits,
    const T_idx* targets,
    T* loss_output,
    float* saved_max,
    float* saved_sum,
    int64_t batch_size,
    int64_t vocab_size,
    cudaStream_t stream
);

/**
 * @brief Stats-aware sparse CE backward. Reuses (max, sum) saved by forward
 * and runs ONLY the Normalize kernel — Reduce kernel skipped, ~3.7 ms / call
 * faster than sparse_ce_backward_cuda_vec.
 */
template<typename T, typename Tidx>
void sparse_ce_backward_cuda_vec_with_stats(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    const float* saved_max,
    const float* saved_sum,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
);

/**
 * @brief Two-kernel sparse CE backward (partial online-softmax reduction + normalize).
 */
template<typename T, typename Tidx>
void sparse_ce_backward_cuda(
    const T* logits,
    const Tidx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
);

/**
 * @brief Compute gradient for Sparse Cross Entropy with Logits.
 *
 * grad = (softmax(logits) - targets_one_hot) * scale
 */
template<typename T, typename T_idx>
void sparse_cross_entropy_backward_cuda(
    const T* logits,
    const T_idx* targets,
    T* grad_logits,
    int64_t batch_size,
    int64_t vocab_size,
    const T* grad_output,
    float host_scale,
    cudaStream_t stream
);

// Categorical Cross Entropy (Targets are probabilities [N, C])
// Forward: loss = -1/N * sum(target * log(clip(pred)))
void categorical_cross_entropy_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t batch_size,
    int64_t num_classes
);

// Backward: grad = -1/N * grad_output * target / pred
void categorical_cross_entropy_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t batch_size,
    int64_t num_classes
);

// MSE Loss
// Forward: loss = mean((pred - target)^2)
void mse_loss_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t numel
);

// Backward: grad = 2/N * (pred - target) * grad_output
void mse_loss_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t numel
);

// MAE Loss
// Forward: loss = mean(|pred - target|)
void mae_loss_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t numel
);

// Backward: grad = 1/N * sign(pred - target) * grad_output
void mae_loss_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t numel
);

// Binary Cross Entropy (BCE)
// Forward: loss = -mean(target * log(clip(pred)) + (1-target) * log(1-clip(pred)))
void bce_loss_forward_cuda(
    const float* predictions,
    const float* targets,
    float* loss_output,
    int64_t numel
);

// Backward: grad = 1/N * (-target/pred + (1-target)/(1-pred)) * grad_output
void bce_loss_backward_cuda(
    const float* grad_output,
    const float* predictions,
    const float* targets,
    float* grad_input,
    int64_t numel
);

} // namespace cuda
} // namespace OwnTensor
