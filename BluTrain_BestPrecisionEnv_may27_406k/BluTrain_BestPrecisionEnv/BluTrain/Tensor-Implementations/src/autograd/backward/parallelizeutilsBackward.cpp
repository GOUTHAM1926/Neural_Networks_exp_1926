#include "autograd/backward/parallelizeutilsBackward.h"

namespace OwnTensor {
namespace autograd {

Make_shards_inplace_axis_Backward::Make_shards_inplace_axis_Backward(size_t num_shards, int64_t axis,
                                                                     const Shape& input_shape,
                                                                     const TensorOptions& options)
    : Node(1), num_shards_(num_shards), axis_(axis), input_shape_(input_shape), options_(options) {}

std::vector<Tensor> Make_shards_inplace_axis_Backward::apply(std::vector<Tensor>&& grads) {
    bool all_empty = true;
    for (const auto& g : grads) {
        if (g.is_valid()) {
            all_empty = false;
            break;
        }
    }
    if (all_empty) {
        return {Tensor()};
    }

    Shape shard_shape = input_shape_;
    shard_shape.dims[axis_] = input_shape_.dims[axis_] / static_cast<int64_t>(num_shards_);

    std::vector<Tensor> valid_grads;
    valid_grads.reserve(num_shards_);
    for (size_t i = 0; i < num_shards_; ++i) {
        if (i < grads.size() && grads[i].is_valid()) {
            valid_grads.push_back(grads[i]);
        } else {
            // Use the options stored during forward pass to ensure correct device
            valid_grads.push_back(Tensor::zeros(shard_shape, options_));
        }
    }

    Tensor combined_grad = Tensor::cat(valid_grads, axis_);
    return {combined_grad};
}

} // namespace autograd
} // namespace OwnTensor
