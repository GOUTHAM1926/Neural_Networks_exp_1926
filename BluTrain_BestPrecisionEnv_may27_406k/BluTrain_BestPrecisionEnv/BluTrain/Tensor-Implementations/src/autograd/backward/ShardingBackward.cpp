#include "autograd/backward/ShardingBackward.h"
#include "core/Tensor.h"
#include <numeric>

namespace OwnTensor {
namespace autograd {

variable_list ShardingBackward::apply(variable_list&& grads) {
    if (grads.empty()) {
        return {Tensor()};
    }

    // Step 1: Ensure all shards have valid gradients. 
    // If a shard doesn't have a gradient, we treat it as zeros.
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

    // Filter out invalid tensors for concatenation (replace with zeros of appropriate shape)
    std::vector<Tensor> valid_grads;
    valid_grads.reserve(num_shards_);

    if (is_custom_) {
        for (size_t i = 0; i < num_shards_; ++i) {
            if (grads[i].is_valid()) {
                valid_grads.push_back(grads[i]);
            } else {
                valid_grads.push_back(Tensor::zeros(shard_shapes_[i], grads[0].opts()));
            }
        }
    } else {
        size_t shard_elems = std::accumulate(input_shape_.dims.begin(), input_shape_.dims.end(), 1LL, std::multiplies<int64_t>()) / num_shards_;
        Shape shard_shape = Shape({{1, static_cast<int64_t>(shard_elems)}});
        
        // Find a valid opt from existing grads for zero creation
        TensorOptions first_valid_opts;
        for (const auto& g : grads) {
            if (g.is_valid()) {
                first_valid_opts = g.opts();
                break;
            }
        }

        for (size_t i = 0; i < num_shards_; ++i) {
            if (grads[i].is_valid()) {
                valid_grads.push_back(grads[i]);
            } else {
                valid_grads.push_back(Tensor::zeros(shard_shape, first_valid_opts));
            }
        }
    }

    // Step 2: Concatenate gradients.
    Tensor combined_grad = Tensor::flatten_concat(valid_grads);

    // Step 3: Reshape back to original input shape if it was not just a flat vector
    if (combined_grad.shape().dims != input_shape_.dims) {
        combined_grad = combined_grad.reshape(input_shape_);
    }

    return {combined_grad};
}

} // namespace autograd
} // namespace OwnTensor