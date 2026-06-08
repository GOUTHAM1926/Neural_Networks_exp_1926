#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
    
std::atomic<bool> Node::use_arena_{true};

variable_list Node::operator()(variable_list&& inputs) {
    // Execute pre-hooks (can modify inputs)
    variable_list processed_inputs = std::move(inputs);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& hook : pre_hooks_) {
            processed_inputs = hook(processed_inputs);
        }
    }
    
    // Apply the backward function
    variable_list outputs = apply(std::move(processed_inputs));
    
    // Execute post-hooks (read-only, for logging/debugging)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& hook : post_hooks_) {
            hook(processed_inputs, outputs);
        }
    }
    
    return outputs;
}

} // namespace OwnTensor
