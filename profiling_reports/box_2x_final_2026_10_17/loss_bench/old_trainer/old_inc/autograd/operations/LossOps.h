#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware MSE loss
 */
Tensor mse_loss(const Tensor &predictions, const Tensor &targets);

/**
 * @brief Autograd-aware MAE loss
 */
Tensor mae_loss(const Tensor &predictions, const Tensor &targets);

/**
 * @brief Autograd-aware binary cross entropy loss
 */
Tensor binary_cross_entropy(const Tensor &predictions, const Tensor &targets);

/**
 * @brief Autograd-aware categorical cross entropy loss
 */
Tensor categorical_cross_entropy(const Tensor &predictions,
                                 const Tensor &targets);
/*
 * @param ignore_index Target value marking a row excluded from the loss.
 *        -1 (default) matches no valid target, so masking is OFF and
 *        pretraining behaviour is unchanged. For uint16 targets with
 *        vocab 50304, use 65535.
 * @param denom Optional denominator. nullptr (default) divides by batch_size,
 *        exactly as before. When supplied it must be a 1-element device tensor
 *        holding the GLOBAL valid-token count for the whole optimizer step
 *        (summed over grad-accum micro-batches AND all-reduced across ranks),
 *        matching torchtitan's `loss = loss / global_valid_tokens`. Passing a
 *        Tensor rather than a scalar keeps the count on device and avoids a
 *        CPU sync in the training loop.
 */
Tensor sparse_cross_entropy_loss(const Tensor &logits, const Tensor &targets,
                                 int64_t ignore_index = -1,
                                 const Tensor *denom = nullptr);

} // namespace autograd
} // namespace OwnTensor