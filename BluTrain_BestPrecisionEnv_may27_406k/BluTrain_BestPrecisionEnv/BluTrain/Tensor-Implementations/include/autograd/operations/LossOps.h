#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware MSE loss
 */
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware MAE loss
 */
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware binary cross entropy loss
 */
Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware categorical cross entropy loss
 */
Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets);

/**
 * @brief Autograd-aware sparse cross entropy loss with logits
 * 
 * Computes softmax internally and cross entropy against integer class indices.
 * 
 * @param logits Raw logits [batch_size, num_classes]
 * @param targets Integer class indices [batch_size]
 * @return Scalar loss tensor
 */
Tensor sparse_cross_entropy_loss(const Tensor& logits, const Tensor& targets);

} // namespace autograd
} // namespace OwnTensor