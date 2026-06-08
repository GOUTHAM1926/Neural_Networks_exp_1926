#pragma once

#include "core/Tensor.h"
#include "autograd/Node.h"
#include <functional>
#include <vector>

namespace OwnTensor {

namespace nn { class Sequential; }

namespace autograd {

/**
 * @brief Checkpoint a model block to save memory during training.
 * 
 * This function implements gradient checkpointing (trading compute for memory).
 * It runs the provided function in no_grad mode during the forward pass,
 * and recomputes the activations during the backward pass.
 * 
 * @param fn The function/block to checkpoint. It should take a variable_list 
 *           and return a variable_list.
 * @param inputs The input tensors to the block.
 * @return The output tensors of the block.
 * 
 * ## Usage Example
 * ```cpp
 * auto my_block = [](const variable_list& inputs) {
 *     return variable_list{relu(linear(inputs[0], w, b))};
 * };
 * auto outputs = checkpoint(my_block, {input_tensor});
 * ```
 */

variable_list checkpoint(
    std::function<variable_list(const variable_list&)> fn,
    const variable_list& inputs,
    bool offload_to_cpu = false);

/**
 * @brief Checkpoint a sequential model by splitting it into segments.
 * 
 * @note This implementation currently assumes a single tensor flows through the 
 *       sequence. It uses only the first element of the input `variable_list` 
 *       for each segment.
 * 
 * @param model The sequential model to checkpoint.
 * @param segments Number of segments to split the model into.
 * @param inputs Input tensors to the model.
 * @return Output tensors of the model.
 */
variable_list checkpoint_sequential(
    std::shared_ptr<nn::Sequential> model,
    int segments,
    const variable_list& inputs,
    bool offload_to_cpu = false);

} // namespace autograd
} // namespace OwnTensor