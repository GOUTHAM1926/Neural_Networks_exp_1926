#pragma once

#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * Flash Attention 2 — fused causal/non-causal scaled dot-product attention.
 *
 * Q, K, V : [BH, T, d]  where BH = batch * heads
 * Returns  : O [BH, T, d]
 *
 * Avoids materialising the full [BH, T, T] score matrix by tiling Q/K/V
 * and maintaining an online softmax (running max + sum).
 * Saves the log-sum-exp tensor L [BH, T] in the backward node so the
 * backward pass can recompute softmax weights without storing them.
 */
void launch_flash_attn(
    const float* Q, const float* K, const float* V, float* O, float* L,
    int B, int H, int N, int d, bool causal
);

Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                       bool causal = true);

} // namespace autograd
} // namespace OwnTensor
