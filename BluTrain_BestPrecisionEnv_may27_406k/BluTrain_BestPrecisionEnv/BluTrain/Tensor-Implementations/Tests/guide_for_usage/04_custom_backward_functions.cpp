/**
 * @file 04_custom_backward_functions.cpp
 * @brief GUIDE: Writing Custom Backward Functions
 * 
 * This guide shows how to create custom autograd operations:
 * - Creating a custom Node subclass
 * - Implementing the apply() method
 * - Pattern for connecting to the computational graph
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/04_custom_backward_functions.cpp
 */

#include "core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/Variable.h"
#include "autograd/SavedVariable.h"
#include "autograd/AutogradOps.h"
#include <iostream>

using namespace OwnTensor;

// ============================================================================
// EXAMPLE 1: Simple Gradient Scaler
// ============================================================================
class ScaleGradBackward : public Node {
private:
    float scale_;
    
public:
    ScaleGradBackward(float scale) : Node(1), scale_(scale) {}
    
    const char* name() const override { return "ScaleGradBackward"; }
    
    variable_list apply(variable_list&& grads) override {
        const Tensor& grad_output = grads[0];
        
        size_t n = grad_output.numel();
        Tensor result(grad_output.shape(), TensorOptions());
        
        const float* in = grad_output.data<float>();
        float* out = const_cast<float*>(result.data<float>());
        
        for (size_t i = 0; i < n; i++) {
            out[i] = in[i] * scale_;
        }
        
        return {result};
    }
};

// ============================================================================
// EXAMPLE 2: Identity with Debugging
// ============================================================================
class DebugBackward : public Node {
private:
    std::string message_;
    
public:
    DebugBackward(const std::string& msg) : Node(1), message_(msg) {}
    
    const char* name() const override { return "DebugBackward"; }
    
    variable_list apply(variable_list&& grads) override {
        std::cout << "  [DEBUG] " << message_ << "\n";
        return grads;  // Pass through unchanged
    }
};

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║       GUIDE: Custom Backward Functions               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // Test 1: Custom Gradient Scaler
    std::cout << "=== 1. Custom Gradient Scaler ===\n\n";
    
    auto scale_node = std::make_shared<ScaleGradBackward>(2.0f);
    std::cout << "Created: " << scale_node->name() << "\n";
    std::cout << "This node multiplies gradients by 2.0\n\n";
    
    Tensor grad = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    std::cout << "Input gradient (all 1s): ";
    for (size_t i = 0; i < 4; i++) std::cout << grad.data<float>()[i] << " ";
    std::cout << "\n";
    
    variable_list result = scale_node->apply({grad});
    std::cout << "Output gradient (scaled by 2): ";
    for (size_t i = 0; i < 4; i++) std::cout << result[0].data<float>()[i] << " ";
    std::cout << "\n\n";

    // Test 2: Debug Node (pass-through with logging)
    std::cout << "=== 2. Debug Node ===\n\n";
    
    auto debug_node = std::make_shared<DebugBackward>("Processing gradient...");
    std::cout << "Calling debug_node->apply():\n";
    variable_list debug_result = debug_node->apply({grad});
    std::cout << "Gradient passed through unchanged\n\n";

    // Summary
    std::cout << "=== PATTERN: Creating Custom Backward ===\n\n";
    std::cout << "class MyBackward : public Node {\n";
    std::cout << "private:\n";
    std::cout << "    SavedVariable saved_input_;  // Optional: save tensors\n";
    std::cout << "public:\n";
    std::cout << "    MyBackward() : Node(1) {}   // num_inputs = 1\n";
    std::cout << "    std::string name() const override { return \"MyBackward\"; }\n";
    std::cout << "    variable_list apply(variable_list&& grads) override {\n";
    std::cout << "        // Compute and return gradient\n";
    std::cout << "        return {computed_grad};\n";
    std::cout << "    }\n";
    std::cout << "};\n\n";
    
    std::cout << "Then in forward function:\n";
    std::cout << "  auto grad_fn = std::make_shared<MyBackward>(...);\n";
    std::cout << "  Edge edge = impl::gradient_edge(input);\n";
    std::cout << "  grad_fn->set_next_edge(0, edge);\n";
    std::cout << "  output.set_grad_fn(grad_fn);\n";

    std::cout << "\n✅ Custom backward functions guide complete!\n\n";
    return 0;
}
