#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {


class Make_shards_inplace_axis_Backward: public Node {
private:
    size_t num_shards_;
    int64_t axis_;
    Shape input_shape_;
    TensorOptions options_;   // device/dtype for creating zero-fill grads when a shard has no gradient

public:
    // options must carry the device+dtype of the input tensor so that zero
    // replacement gradients are allocated on the correct device.
    Make_shards_inplace_axis_Backward(size_t num_shards, int64_t axis,
                                      const Shape& input_shape,
                                      const TensorOptions& options);
    
    const char* name() const override { return "make_shards_inplace_axis_Backward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace autograd
}