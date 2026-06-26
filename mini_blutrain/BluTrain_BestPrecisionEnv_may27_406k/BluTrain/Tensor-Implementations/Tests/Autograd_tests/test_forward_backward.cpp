#include "core/Tensor.h"
#include <iostream>
#include <iomanip>

using namespace OwnTensor;

int main() {
    std::cout << "\n=== Forward & Backward Pass Infrastructure Test ===\n";
    std::cout << "Simulating: a = w @ b, x = W_2 @ a, then backward\n\n";
    
    // ------------------------------------------------------------------------
    // SETUP & FORWARD PASS (Simulated)
    // ------------------------------------------------------------------------
    std::cout << "--- FORWARD PASS (Simulated) ---\n\n";
    
    std::cout << "Creating tensors with requires_grad=true...\n";
    TensorOptions grad_opts = TensorOptions{}.with_dtype(Dtype::Float32);
    grad_opts.requires_grad = true;
    
    // Create weight matrices
    Tensor w(Shape{{3, 4}}, grad_opts);      // w: (3, 4)
    Tensor b(Shape{{4, 2}}, grad_opts);      // b: (4, 2)
    Tensor W_2(Shape{{2, 3}}, grad_opts);    // W_2: (2, 3)
    Tensor a(Shape{{3, 2}}, grad_opts);      // a = w @ b would be (3, 2)
    Tensor x(Shape{{2, 2}}, grad_opts);      // x = W_2 @ a^T would be (2, 2)
    
    // Set some data
    std::vector<float> w_data(w.numel(), 1.0f);
    std::vector<float> b_data(b.numel(), 0.5f);
    std::vector<float> W_2_data(W_2.numel(), 0.8f);
    std::vector<float> a_data(a.numel(), 2.0f);   // Simulated result of w @ b
    std::vector<float> x_data(x.numel(), 3.0f);   // Simulated result of W_2 @ a
    
    w.set_data(w_data);
    b.set_data(b_data);
    W_2.set_data(W_2_data);
    a.set_data(a_data);
    x.set_data(x_data);
    
    std::cout << "Created 5 tensors (w, b, W_2, a, x) with requires_grad=true\n";
    std::cout << "Forward computation simulated\n";
    
    // ------------------------------------------------------------------------
    // BACKWARD PASS (Manual gradient propagation)
    // ------------------------------------------------------------------------
    std::cout << "\n--- BACKWARD PASS (Manual) ---\n\n";
    std::cout << "Demonstrating gradient storage via AutogradMeta...\n\n";
    
    // Step 1: Set output gradient
    std::cout << "1. Setting dx (output gradient from loss)...\n";
    std::vector<float> dx_data(x.numel(), 1.0f);
    x.set_grad(dx_data);
    std::cout << " dx = 1.0 (gradient flows backward from here)\n";
    
    // Step 2: Backprop through x = W_2 @ a
    std::cout << "\n2. Backprop through x = W_2 @ a:\n";
    
    // Gradient for W_2: dW_2 = dx @ a^T
    std::vector<float> dW_2_data(W_2.numel(), 0.5f);  // dummy computation
    W_2.set_grad(dW_2_data);
    std::cout << " dW_2 computed and stored\n";
    
    // Gradient for a: da = W_2^T @ dx
    std::vector<float> da_data(a.numel(), 0.4f);  // dummy
    a.set_grad(da_data);
    std::cout << " da computed and stored\n";
    
    // Step 3: Backprop through a = w @ b
    std::cout << "\n3. Backprop through a = w @ b:\n";
    
    // Gradient for w: dw = da @ b^T
    std::vector<float> dw_data(w.numel(), 0.3f);  // dummy
    w.set_grad(dw_data);
    std::cout << " dw computed and stored\n";
    
    // Gradient for b: db = w^T @ da
    std::vector<float> db_data(b.numel(), 0.2f);  // dummy
    b.set_grad(db_data);
    std::cout << " db computed and stored\n";
    
    // ------------------------------------------------------------------------
    // VERIFICATION
    // ------------------------------------------------------------------------
    std::cout << "\n--- VERIFICATION ---\n\n";
    
    bool all_passed = true;
    

    auto check_grad = [&](Tensor& t, const std::string& name, float expected) {
        if (t.owns_grad()) {
            float* grad = t.grad<float>(); 
            if (grad != nullptr && std::abs(grad[0] - expected) < 1e-5f) {
                std::cout<< name << " gradient: " << std::fixed << std::setprecision(2) 
                          << grad[0] << "\n";
                return true;
            }
        }
        std::cout << name << " gradient missing or incorrect!\n";
        all_passed = false;
        return false;
    };
    
    check_grad(x, "x ", 1.0f);
    check_grad(W_2, "W_2", 0.5f);
    check_grad(a, "a ", 0.4f);
    check_grad(w, "w ", 0.3f);
    check_grad(b, "b ", 0.2f);
    
    // Test grad_view() functionality
    std::cout << "\nTesting grad_view() access:\n";
    try {
        Tensor w_grad_tensor = w.grad_view();
        if (w_grad_tensor.shape() == w.shape()) {
            std::cout << "w.grad_view() returns correct shape\n";
        } else {
            std::cout << "w.grad_view() shape mismatch\n";
            all_passed = false;
        }
    } catch (const std::exception& e) {
        std::cout << "w.grad_view() failed: " << e.what() << "\n";
        all_passed = false;
    }
    
    // ------------------------------------------------------------------------
    // SUMMARY
    // ------------------------------------------------------------------------
    std::cout << "\n=== SUMMARY ===\n\n";
    if (all_passed) {
        std::cout << "✅ ALL TESTS PASSED!\n\n";
        std::cout << "Successfully demonstrated:\n";
        std::cout << " 1. Gradient storage infrastructure (AutogradMeta)\n";
        std::cout << " 2. All tensors track requires_grad\n";
        std::cout << " 3. Gradients stored correctly via set_grad()\n";
        std::cout << " 4. grad<T>() retrieves gradient pointers\n";
        std::cout << " 5. grad_view() creates gradient tensors\n";
        std::cout << "\nInfrastructure READY for automatic backward() implementation!\n";
        std::cout << "Next step: Implement autograd engine to compute gradients automatically.\n";
    } else {
        std::cout << "❌ SOME TESTS FAILED\n";
        return 1;
    }
    
    std::cout << "\n";
    return 0;
}
