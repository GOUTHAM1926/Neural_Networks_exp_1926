#pragma once
// =============================================================================
// common.h — Shared utilities for GPT-2 training
//   - GPT-2 style weight initialization
//   - Cosine LR scheduler with warmup
// =============================================================================

#include <cmath>
#include <cstdint>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "nn/NN.h"

using namespace OwnTensor;

namespace gpt2 {

// =============================================================================
// GPT-2 style init: copy N(0, std) into an existing nn::Linear's weight
// =============================================================================
inline void init_linear_gpt2(nn::Linear& layer, float std = 0.02f,
                              uint64_t seed = 1234, bool req_grad = true) {
    auto shape = layer.weight.shape();
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor init_data = Tensor::randn<float>(shape, opts, seed, std);
    layer.weight.copy_(init_data);
    layer.weight.set_requires_grad(req_grad);

    if (layer.bias.is_valid()) {
        Tensor bias_init = Tensor::zeros(layer.bias.shape(), opts);
        layer.bias.copy_(bias_init);
        layer.bias.set_requires_grad(req_grad);
    }
}

// =============================================================================
// Cosine learning rate schedule with linear warmup
// =============================================================================
inline float get_lr(int step, float max_lr, float min_lr,
                    int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) /
                        static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

}  // namespace gpt2
