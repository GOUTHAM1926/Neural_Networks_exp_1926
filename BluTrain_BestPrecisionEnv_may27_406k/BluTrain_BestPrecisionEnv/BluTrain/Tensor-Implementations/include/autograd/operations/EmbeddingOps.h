#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Autograd-aware embedding lookup
 * 
 * Looks up rows from weight matrix using indices.
 * 
 * @param weight Weight matrix [vocab_size, embed_dim]
 * @param indices Token indices [B, T] (int32 or int64)
 * @return Embeddings [B, T, embed_dim]
 */
Tensor embedding(const Tensor& weight, const Tensor& indices, int padding_idx = -1);

} // namespace autograd
} // namespace OwnTensor