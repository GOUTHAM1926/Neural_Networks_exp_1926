#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for sharding operations.
 * 
 * This node handles multiple outputs (the shards). It collects gradients from
 * all shards and concatenates them to form the gradient of the original tension.
 */
class ShardingBackward : public Node {
private:
    Shape input_shape_;
    size_t num_shards_;
    bool is_custom_;
    std::vector<Shape> shard_shapes_; // For custom sharding

public:
    /// Constructor for equal sharding
    ShardingBackward(const Shape& input_shape, size_t num_shards)
        : Node(num_shards), input_shape_(input_shape), num_shards_(num_shards), is_custom_(false) {}

    /// Constructor for custom sharding
    ShardingBackward(const Shape& input_shape, const std::vector<Shape>& shard_shapes)
        : Node(shard_shapes.size()), input_shape_(input_shape), 
          num_shards_(shard_shapes.size()), is_custom_(true), shard_shapes_(shard_shapes) {}
    
    const char* name() const override { return "ShardingBackward"; }
    
    variable_list apply(variable_list&& grads) override;
    
    void release_saved_variables() override {
        shard_shapes_.clear();
    }
};

} // namespace autograd
} // namespace OwnTensor