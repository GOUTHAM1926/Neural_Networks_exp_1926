#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for transpose: tensor.transpose(dim0, dim1)
 */
class TransposeBackward : public Node {
private:
    int dim0_;
    int dim1_;
    
public:
    TransposeBackward(int dim0, int dim1);
    
    const char* name() const override { return "TransposeBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
} // namespace OwnTensor
