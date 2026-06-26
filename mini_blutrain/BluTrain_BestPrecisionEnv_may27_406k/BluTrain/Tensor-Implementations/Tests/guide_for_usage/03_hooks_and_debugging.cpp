/**
 * @file 03_hooks_and_debugging.cpp
 * @brief GUIDE: Gradient Hooks and Debugging
 * 
 * This guide demonstrates:
 * - Registering gradient hooks on tensors
 * - Pre/post hooks on backward nodes
 * - Using AnomalyMode for debugging
 * - SavedVariable for version checking
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/03_hooks_and_debugging.cpp
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/Node.h"
#include "autograd/Functions.h"
#include "autograd/Hooks.h"
#include "autograd/AnomalyMode.h"
#include "autograd/SavedVariable.h"
#include <iostream>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║        GUIDE: Gradient Hooks and Debugging           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // =========================================================================
    // 1. Registering Tensor Hooks
    // =========================================================================
    std::cout << "=== 1. Registering Tensor Hooks ===\n\n";
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor x = Tensor::randn<float>(Shape{{2, 2}}, opts);
    
    // Register a gradient hook on tensor
    // This hook will be called when gradient is computed
    int hook_call_count = 0;
    x.register_hook(make_pre_hook([&hook_call_count](const Tensor& grad) {
        hook_call_count++;
        std::cout << "  [HOOK] Gradient hook called! Count: " << hook_call_count << "\n";
        std::cout << "  [HOOK] Gradient values: ";
        const float* g = grad.data<float>();
        for (size_t i = 0; i < 4; i++) std::cout << g[i] << " ";
        std::cout << "\n";
        return grad;  // Return gradient (can modify it here!)
    }));
    
    std::cout << "Registered pre-hook on tensor x\n\n";

    // =========================================================================
    // 2. Node Pre/Post Hooks
    // =========================================================================
    std::cout << "=== 2. Node Pre/Post Hooks ===\n\n";
    
    // Create a backward node directly for demonstration
    Tensor dummy_a = Tensor::zeros(Shape{{2, 2}}, opts);
    Tensor dummy_b = Tensor::zeros(Shape{{2, 2}}, opts);
    auto add_node = std::make_shared<AddBackward>(dummy_a.shape(), dummy_b.shape());
    
    // Pre-hook: Called before apply(), can modify inputs
    add_node->register_pre_hook([](variable_list& inputs) {
        std::cout << "  [PRE-HOOK] About to compute backward for AddBackward\n";
        std::cout << "  [PRE-HOOK] Number of gradient inputs: " << inputs.size() << "\n";
        return inputs;  // Pass through (could modify here)
    });
    
    // Post-hook: Called after apply(), for logging/debugging
    add_node->register_post_hook([](const variable_list& inputs, const variable_list& outputs) {
        std::cout << "  [POST-HOOK] AddBackward completed\n";
        std::cout << "  [POST-HOOK] Produced " << outputs.size() << " output gradients\n";
    });
    
    std::cout << "Registered pre-hook and post-hook on AddBackward node\n";
    std::cout << "Now executing node with operator()...\n\n";
    
    // Execute the node (this triggers hooks)
    Tensor grad_in = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    variable_list result = (*add_node)({grad_in});
    
    std::cout << "\nResult: " << result.size() << " output gradients\n\n";

    // =========================================================================
    // 3. AnomalyMode for Debugging
    // =========================================================================
    std::cout << "=== 3. AnomalyMode for Debugging ===\n\n";
    
    std::cout << "AnomalyMode is_enabled(): " 
              << (AnomalyMode::is_enabled() ? "true" : "false") << "\n";
    
    // Enable anomaly detection (provides better error messages)
    AnomalyMode::set_enabled(true);
    std::cout << "Enabled AnomalyMode\n";
    std::cout << "AnomalyMode is_enabled(): " 
              << (AnomalyMode::is_enabled() ? "true" : "false") << "\n\n";
    
    // Use AnomalyMetadata for storing debug info
    AnomalyMetadata meta;
    meta.set_context("Created during forward pass of layer 3");
    std::cout << "AnomalyMetadata context: " << meta.context() << "\n\n";
    
    AnomalyMode::set_enabled(false);  // Disable for normal operation

    // =========================================================================
    // 4. SavedVariable for Safe Tensor Storage
    // =========================================================================
    std::cout << "=== 4. SavedVariable for Safe Tensor Storage ===\n\n";
    
    Tensor saved_tensor = Tensor::randn<float>(Shape{{3, 3}}, opts);
    
    // Save the tensor (stores version for in-place detection)
    SavedVariable saved(saved_tensor, false);  // false = not an output
    
    std::cout << "Saved tensor with version: " << saved.saved_version() << "\n";
    std::cout << "was_leaf: " << (saved.was_leaf() ? "true" : "false") << "\n";
    std::cout << "is_output: " << (saved.is_output() ? "true" : "false") << "\n\n";
    
    // Unpack (validates version hasn't changed)
    Tensor unpacked = saved.unpack();
    std::cout << "Successfully unpacked saved tensor!\n\n";
    
    // Demonstrate in-place detection
    std::cout << "Now simulating in-place modification...\n";
    increment_version(saved_tensor);  // Simulate in-place op
    
    std::cout << "After increment_version(), trying to unpack...\n";
    try {
        saved.unpack();
        std::cout << "ERROR: Should have thrown!\n";
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Caught expected error: " << e.what() << "\n";
    }

    std::cout << "\n✅ Hooks and debugging guide complete!\n\n";
    return 0;
}
