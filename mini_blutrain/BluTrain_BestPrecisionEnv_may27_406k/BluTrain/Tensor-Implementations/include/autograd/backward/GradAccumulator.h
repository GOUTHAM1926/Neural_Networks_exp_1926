#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Gradient accumulator for leaf tensors
 * 
 * This node accumulates gradients into a leaf tensor's AutogradMeta.
 * It's used as the terminal node in the backward graph for parameters.
 */
class GradAccumulator : public Node {
private:
    intrusive_ptr<TensorImpl> leaf_impl_;  // Owning pointer to keep leaf alive during backward
    std::string name_; // Cached name with metadata
    
public:
    explicit GradAccumulator(TensorImpl* impl);
    
    // Pool factory method
    static std::shared_ptr<GradAccumulator> make(TensorImpl* impl);
    
    const char* name() const override { return name_.c_str(); }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    // Reset state for pooling
    void reset(TensorImpl* impl);

private:
   // Thread-safe pool
   static std::vector<GradAccumulator*> pool_;
   static std::mutex pool_mutex_;
   static std::atomic<int> global_id_counter_;
};

} // namespace autograd
} // namespace OwnTensor
