/**
 * @file 02_autograd_basics.cpp
 * @brief GUIDE: Automatic Differentiation Basics
 * 
 * This guide demonstrates the autograd system:
 * - Enabling gradient tracking
 * - Forward pass through operations
 * - Backward pass (gradient computation)
 * - Accessing computed gradients
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/02_autograd_basics.cpp
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/Engine.h"
#include <iostream>

using namespace OwnTensor;

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║       GUIDE: Automatic Differentiation Basics        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // =========================================================================
    // 1. Creating Tensors with Gradient Tracking
    // =========================================================================
    std::cout << "=== 1. Creating Tensors with Gradient Tracking ===\n\n";
    
    // Enable gradient tracking with TensorOptions
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    Tensor x = Tensor::randn<float>(Shape{{2, 2}}, opts);
    Tensor w = Tensor::randn<float>(Shape{{2, 2}}, opts);
    
    std::cout << "x.requires_grad(): " << (x.requires_grad() ? "true" : "false") << "\n";
    std::cout << "x.is_leaf(): " << (x.is_leaf() ? "true" : "false") << "\n\n";

    // =========================================================================
    // 2. Forward Pass with Autograd Operations
    // =========================================================================
    std::cout << "=== 2. Forward Pass with Autograd Operations ===\n\n";
    
    // Use autograd:: namespace for gradient-tracked operations
    Tensor y = autograd::mul(x, w);  // y = x * w
    
    std::cout << "y = autograd::mul(x, w)\n";
    std::cout << "y.is_leaf(): " << (y.is_leaf() ? "true" : "false") << "\n";
    std::cout << "y.grad_fn(): " << (y.grad_fn() ? y.grad_fn()->name() : "none") << "\n\n";
    
    // Chain operations
    Tensor z = autograd::add(y, w);  // z = y + w
    std::cout << "z = autograd::add(y, w)\n";
    std::cout << "z.grad_fn(): " << (z.grad_fn() ? z.grad_fn()->name() : "none") << "\n\n";
    
    // Reduce to scalar for backward
    Tensor loss = autograd::mean(z);
    std::cout << "loss = autograd::mean(z)\n";
    std::cout << "loss.grad_fn(): " << (loss.grad_fn() ? loss.grad_fn()->name() : "none") << "\n";
    std::cout << "loss value: " << loss.data<float>()[0] << "\n\n";

    // =========================================================================
    // 3. Backward Pass (Gradient Computation)
    // =========================================================================
    std::cout << "=== 3. Backward Pass ===\n\n";
    
    // Compute gradients
    autograd::backward(loss);
    
    std::cout << "Called autograd::backward(loss)\n\n";
    
    // Check if gradients were computed
    std::cout << "x.owns_grad(): " << (x.owns_grad() ? "true" : "false") << "\n";
    std::cout << "w.owns_grad(): " << (w.owns_grad() ? "true" : "false") << "\n\n";

    // =========================================================================
    // 4. Accessing Gradients
    // =========================================================================
    std::cout << "=== 4. Accessing Gradients ===\n\n";
    
    if (x.owns_grad()) {
        const float* x_grad = x.grad<float>();
        std::cout << "x.grad: ";
        for (size_t i = 0; i < x.numel(); i++) {
            std::cout << x_grad[i] << " ";
        }
        std::cout << "\n";
    }
    
    if (w.owns_grad()) {
        const float* w_grad = w.grad<float>();
        std::cout << "w.grad: ";
        for (size_t i = 0; i < w.numel(); i++) {
            std::cout << w_grad[i] << " ";
        }
        std::cout << "\n";
    }
    
    // =========================================================================
    // 5. Gradient View (Alternative Access)
    // =========================================================================
    std::cout << "\n=== 5. Gradient View ===\n\n";
    
    if (x.owns_grad()) {
        Tensor x_grad_tensor = x.grad_view();
        std::cout << "x.grad_view() numel: " << x_grad_tensor.numel() << "\n";
    }

    std::cout << "\n✅ Autograd basics guide complete!\n\n";
    return 0;
}
