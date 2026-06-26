#pragma once

#include "autograd/Node.h"
#include "autograd/SavedVariable.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for embedding lookup
 * 
 * Forward: output[b,t,:] = weight[indices[b,t], :]
 * Backward: Scatter-add grad_output into weight gradient by indices
 */
class EmbeddingBackward : public Node {
private:
    SavedVariable saved_indices_;
    int64_t vocab_size_;
    int64_t embed_dim_;
    int padding_idx_;
    
public:
    EmbeddingBackward(const Tensor& indices, int64_t vocab_size, int64_t embed_dim, int padding_idx = -1);
    
    const char* name() const override { return "EmbeddingBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    void release_saved_variables() override { saved_indices_.reset(); }
};

} // namespace autograd
} // namespace OwnTensor