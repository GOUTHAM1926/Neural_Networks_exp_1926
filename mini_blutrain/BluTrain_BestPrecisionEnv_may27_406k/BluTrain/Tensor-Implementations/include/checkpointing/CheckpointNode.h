#pragma once

#include "autograd/Node.h"
#include "autograd/SavedVariable.h"
#include "checkpointing/RNG.h"
#include "device/Device.h"
#include <functional>
#include <vector>
#include <iostream>

namespace OwnTensor {
namespace autograd {

/**
 * @brief A specialized Node for gradient checkpointing.
 * 
 * CheckpointNode stores the forward function and its inputs. During the backward
 * pass, it re-runs the forward function to recreate the local computational graph
 * and then triggers a local backward pass.
 */
class CheckpointNode : public Node {
public:
    /**
     * @brief Construct a CheckpointNode.
     * 
     * @param forward_fn The function to re-run during recomputation.
     * @param inputs The inputs to the forward function.
     * @param rng_state The RNG state captured during the initial forward pass.
     */
    CheckpointNode(
        std::function<variable_list(const variable_list&)> forward_fn,
        const variable_list& inputs,
        RNGState rng_state,
        size_t num_outputs,
        bool offload_to_cpu);

    /**
     * @brief The core recomputation logic.
     * 
     * This method is called by the autograd Engine during the backward pass.
     * It performs:
     * 1. RNG state restoration.
     * 2. Re-running the forward pass with gradients enabled.
     * 3. Triggering a local backward pass on the recomputed graph.
     * 4. Collecting gradients for the original inputs.
     * 
     * @param grads Gradients with respect to the outputs of the checkpointed block.
     * @return Gradients with respect to the inputs of the checkpointed block.
     */
    variable_list apply(variable_list&& grads) override;

    const char* name() const override { return "CheckpointNode"; }
    size_t num_outputs() const { return num_outputs_; }   // TODO: override

private:
    /// The forward function to re-run.
    std::function<variable_list(const variable_list&)> forward_fn_;
    
    /// Inputs saved for recomputation (includes versioning checks).
    std::vector<SavedVariable> saved_inputs_;
    
    /// RNG state to ensure deterministic recomputation.
    RNGState rng_state_;
    
    /// Whether any input requires gradient.
    std::vector<bool> input_requires_grad_;

    /// Original devices of inputs (for restoration after CPU offloading)
    std::vector<DeviceIndex> input_devices_;

    /// Number of outputs of the checkpointed block.
    size_t num_outputs_;

    /// Whether to offload inputs to CPU during forward pass.
    bool offload_to_cpu_;

    /// Set to true after release_saved_variables() fires; prevents double-release.
    bool released_ = false;

    /**
     * @brief Release saved variables to break reference cycles and free memory.
     */
    void release_saved_variables() override;

    /**
     * @brief Self-contained local backward pass over the recomputed graph.
     *
     * Runs a sequential BFS + execution loop using only local data structures.
     * No global singletons (NodeTaskPool, BackwardContext) are touched, which
     * eliminates the re-entrant engine call that caused BUG-4.
     *
     * @param outputs         Recomputed outputs of forward_fn_ (roots of local graph).
     * @param grads           Incoming gradients from the outer engine (one per output).
     * @param local_seq_floor Minimum sequence number of any GradAccumulator created for
     *                        the recompute_input leaf tensors. Any Node whose sequence_nr()
     *                        is strictly less than this value was created during the outer
     *                        forward pass and belongs to the outer computation graph.
     *                        The BFS will not traverse into those nodes, preventing
     *                        LocalCleanupGuard from releasing outer-engine saved state.
     */
    static void run_local_backward(
        const variable_list& outputs,
        const variable_list& grads,
        uint64_t local_seq_floor);
};

} // namespace autograd
} // namespace OwnTensor