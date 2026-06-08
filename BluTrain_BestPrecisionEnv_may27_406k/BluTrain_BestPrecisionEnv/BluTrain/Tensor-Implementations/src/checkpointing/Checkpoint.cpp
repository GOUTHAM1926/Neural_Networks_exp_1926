
#include "checkpointing/CheckpointNode.h"
#include "checkpointing/GradMode.h"
#include "autograd/Variable.h"
#include "dtype/Dtype.h"
#include "nn/NN.h"
#include <algorithm>
#include <iostream>

namespace OwnTensor {
namespace autograd {

variable_list checkpoint(
    std::function<variable_list(const variable_list&)> fn,
    const variable_list& inputs,
    bool offload_to_cpu) {
    
    // 1. Check if any input requires gradient.
    // If none do, we don't need to checkpoint at all.
    bool any_requires_grad = std::any_of(inputs.begin(), inputs.end(),
        [](const Tensor& t) { return t.requires_grad(); });
    
    if (!any_requires_grad || !GradMode::is_enabled()) {
        return fn(inputs);
    }

    // 2. Capture current RNG state and run the function in NO_GRAD mode.
    // The capture must be the last operation before fn(inputs) so that no
    // intervening code advances the RNG between capture and execution.
    variable_list outputs;
    RNGState rng_state;
    {
        NoGradGuard guard;
        rng_state = RNG::get_state();  // capture INSIDE the no-grad scope
        outputs = fn(inputs);
    }

    // 4. Validate outputs before creating the CheckpointNode.
    // An empty output list means no output tensor will ever hold an edge to the
    // CheckpointNode, so it would be garbage-collected immediately and no
    // gradient would ever be computed.
    if (outputs.empty()) {
        throw std::runtime_error(
            "checkpoint: fn returned an empty variable_list. "
            "The checkpointed function must produce at least one output tensor.");
    }

    // 5. Create the CheckpointNode.
    // This node will handle the recomputation during the backward pass.
    auto checkpoint_node = std::make_shared<CheckpointNode>(
        std::move(fn),
        inputs,
        std::move(rng_state),
        outputs.size(),
        offload_to_cpu
    );

    // 6. Connect differentiable outputs to the CheckpointNode.
    // Only floating-point (and complex) outputs can carry gradients — setting
    // requires_grad=true on an integer, bool, or index tensor is incorrect and
    // will cause downstream autograd ops to crash or miscompute.
    auto is_differentiable_dtype = [](Dtype d) -> bool {
        return d == Dtype::Float16  || d == Dtype::Bfloat16 ||
               d == Dtype::Float32  || d == Dtype::Float64  ||
               d == Dtype::Float4_e2m1 || d == Dtype::Float4_e2m1_2x ||
               d == Dtype::Complex32 || d == Dtype::Complex64 || d == Dtype::Complex128;
    };

    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i].unsafeGetTensorImpl()) continue;
        if (!is_differentiable_dtype(outputs[i].dtype())) continue;
        outputs[i].set_requires_grad(true);
        impl::set_gradient_edge(outputs[i], Edge(checkpoint_node, static_cast<uint32_t>(i)));
    }

    return outputs;
}

variable_list checkpoint_sequential(
    std::shared_ptr<nn::Sequential> model,
    int segments,
    const variable_list& inputs,
    bool offload_to_cpu) {
    
    const auto& modules = model->modules();
    int num_modules = static_cast<int>(modules.size());
    
    if (segments <= 0) {
        throw std::invalid_argument("segments must be greater than 0");
    }

    if (segments > num_modules) {
        segments = num_modules;
    }

    // segments==1 means the entire model is one segment, which is always the
    // "last segment" and therefore runs without checkpointing (BUG-7 design).
    // Warn so the caller knows they are getting no memory benefit.
    if (segments == 1) {
        // Use stderr so this surfaces without requiring a logging framework.
        // To checkpoint a single-segment model use checkpoint() directly.
        std::cerr << "[checkpoint_sequential] WARNING: segments=1 means the entire "
                     "model runs as the last segment and is NOT checkpointed. "
                     "Use at least segments=2, or call checkpoint() directly.\n";
    }

    int modules_per_segment = num_modules / segments;
    int remainder = num_modules % segments;

    variable_list current_inputs = inputs;
    int start_idx = 0;

    for (int i = 0; i < segments; ++i) {
        int segment_size = modules_per_segment + (i < remainder ? 1 : 0);
        int end_idx = start_idx + segment_size;

        std::vector<std::shared_ptr<nn::Module>> segment_modules(
            modules.begin() + start_idx,
            modules.begin() + end_idx
        );
        // The convention for Sequential checkpointing: segment_inputs[0] is the
        // "main" tensor that flows through each module's forward(). Any additional
        // tensors in segment_inputs (e.g. encoder outputs, attention masks, position
        // ids) are passed through unchanged so they remain available to later
        // segments and are not silently dropped.
        auto segment_fn = [sm = std::move(segment_modules)](const variable_list& segment_inputs) {
            if (segment_inputs.empty()) {
                throw std::runtime_error(
                    "checkpoint_sequential: segment received an empty input list");
            }
            Tensor x = segment_inputs[0];
            for (size_t j = 0; j < sm.size(); ++j) {
                x = sm[j]->forward(x);
            }
            // Return the transformed main tensor followed by all pass-through tensors.
            variable_list result;
            result.reserve(segment_inputs.size());
            result.push_back(std::move(x));
            for (size_t j = 1; j < segment_inputs.size(); ++j) {
                result.push_back(segment_inputs[j]);
            }
            return result;
        };

        if (i < segments - 1) {
            current_inputs = checkpoint(segment_fn, current_inputs, offload_to_cpu);
        } else {
            // Last segment: run directly without checkpointing. Its activations stay
            // alive naturally until backward starts, so checkpointing would only force
            // a redundant recomputation with zero memory benefit (BUG-7 fix).
            current_inputs = segment_fn(current_inputs);
        }

        start_idx = end_idx;
    }

    return current_inputs;
}

} // namespace autograd
} // namespace OwnTensor