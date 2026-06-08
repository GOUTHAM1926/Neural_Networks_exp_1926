#include "autograd/backward/GradAccumulator.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"


namespace OwnTensor {
namespace autograd {


std::vector<GradAccumulator*> GradAccumulator::pool_;
std::mutex GradAccumulator::pool_mutex_;
std::atomic<int> GradAccumulator::global_id_counter_{1};

GradAccumulator::GradAccumulator(TensorImpl* impl)
    : Node(1), leaf_impl_(impl) {
    name_ = "GradAccumulator";
}

void GradAccumulator::reset(TensorImpl* impl) {
    leaf_impl_ = intrusive_ptr<TensorImpl>(impl);
 
    clear_edges(); 
    // Reset edge to empty/invalid if any
}

std::shared_ptr<GradAccumulator> GradAccumulator::make(TensorImpl* impl) {
    // Temporarily disabled pooling for debugging
    GradAccumulator* ptr = new GradAccumulator(impl);
    
    // Return shared_ptr with default deleter (delete ptr)
    return std::shared_ptr<GradAccumulator>(ptr);
}

std::vector<Tensor> GradAccumulator::apply(std::vector<Tensor>&& grads) {
    // fprintf(stderr, "DEBUG: GradAccumulator::apply EMPTY\n");
    // return {};

    // Assuming grads contains a single grad_output for this leaf
    // And leaf_impl_ is the TensorImpl for which we are accumulating gradients
    Tensor grad_output = std::move(grads[0]); // Take ownership of the grad

    // Accumulate gradient into leaf tensor
    if (leaf_impl_->has_autograd_meta()) {
        auto* meta = static_cast<AutogradMeta*>(leaf_impl_->autograd_meta());
        
        // Use optimized accumulation (1 lock instead of 3)
        meta->accumulate_grad(std::move(grad_output));
        
        meta->trigger_post_acc_hooks(meta->grad());
    }

    // Leaf nodes don't propagate gradients further up the graph
    return {};
}

} // namespace autograd
} // namespace OwnTensor
