#include "autograd/ops_template.h"
#include "autograd/backward/GradAccumulator.h"
#include "core/AutogradMeta.h"

namespace OwnTensor {
namespace autograd {

Edge get_grad_edge(const Tensor& tensor) {
    if (tensor.grad_fn()) {
        // Non-leaf: connect to existing grad_fn
        return make_edge(tensor.grad_fn(), tensor.output_nr());
    } else if (tensor.requires_grad()) {
        // Leaf: get or create GradAccumulator from AutogradMeta
        TensorImpl* impl = tensor.unsafeGetTensorImpl();
        if (impl->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(impl->autograd_meta());
            
            // Try to get existing accumulator
            auto accumulator = meta->grad_accumulator_.lock();
            if (!accumulator) {
                // Create new one and cache it
                accumulator = std::make_shared<GradAccumulator>(impl);
                meta->grad_accumulator_ = accumulator;
            }
            return make_edge(accumulator, 0);
        }
    }
    return Edge{};  // No gradient needed
}

} // namespace autograd
} // namespace OwnTensor
