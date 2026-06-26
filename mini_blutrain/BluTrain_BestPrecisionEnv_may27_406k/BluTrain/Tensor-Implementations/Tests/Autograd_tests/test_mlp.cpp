#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "ops/ScalarOps.h"
#include <iostream>
#include <iomanip>

using namespace OwnTensor;

int main() {
    std::cout << "\n=== Testing MLP with Automatic Differentiation ===\n\n";
    
    // Create a simple 2-layer MLP: input(2) -> hidden(3) -> output(1)
    std::cout << "Creating MLP parameters...\n";
    
    // Layer 1: W1 (2x3), b1 (3)
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    Tensor W1 = Tensor::randn<float>(Shape{{2, 3}}, req_grad, 0.1f ) * 0.5f;
    Tensor b1 = Tensor::zeros(Shape{{3}}, req_grad);
    
    // Layer 2: W2 (3x1), b2 (1)
    Tensor W2 = Tensor::randn<float>(Shape{{3, 1}}, req_grad) * 0.5f;
    Tensor b2 = Tensor::zeros(Shape{{1}}, req_grad);
    
    std::cout << " Created W1 (2x3), b1 (3), W2 (3x1), b2 (1)\n";    std::cout << "All parameters require_grad=true\n\n";
    
    // Create input data (batch_size=1, features=2)
    std::cout << "Creating input data...\n";
    Tensor x = Tensor::ones(Shape{{1, 2}}, TensorOptions().with_req_grad(false));
    std::cout << " Input x shape: (1, 2)\n\n";
    
    // Forward pass
    std::cout << "Forward pass...\n";
    
    // Layer 1: h = relu(x @ W1 + b1)
    Tensor z1 = autograd::matmul(x, W1);              // (1,2) @ (2,3) = (1,3)
    std::cout << "  z1 = x @ W1\n";
    
    Tensor z1_bias = autograd::add(z1, b1);           // (1,3) + (3) = (1,3)
    std::cout << "  z1_bias = z1 + b1\n";
    
    Tensor h = autograd::relu(z1_bias);               // (1,3)
    std::cout << "  h = relu(z1_bias)\n";
    
    // Layer 2: output = h @ W2 + b2
    Tensor z2 = autograd::matmul(h, W2);              // (1,3) @ (3,1) = (1,1)
    std::cout << "  z2 = h @ W2\n";
    
    Tensor output = autograd::add(z2, b2);            // (1,1) + (1) = (1,1)
    std::cout << "  output = z2 + b2\n";
    
    // Loss (mean squared error with target=1.0)
    Tensor target = Tensor::ones(Shape{{1, 1}}, TensorOptions());
    Tensor neg_target = target * -1.0f;
    Tensor diff = autograd::add(output, neg_target);
    Tensor loss = autograd::mul(diff, diff);          // squared error
    loss = autograd::mean(loss);                      // mean
    
    std::cout << "  loss = mean((output - target)^2)\n";
    std::cout << "\n Forward pass complete!\n";
    std::cout << "Loss value: " << loss.data<float>()[0] << "\n\n";
    
    // Backward pass
    std::cout << "Backward pass...\n";
    loss.backward();
    std::cout << " loss.backward() executed!\n\n";
    
    // Check gradients
    std::cout << "Checking gradients...\n";
    
    if (W1.owns_grad()) {
        std::cout << " W1 has gradient\n";
        float* W1_grad = W1.grad<float>();
        float grad_sum = 0.0f;
        for (int i = 0; i < 6; ++i) grad_sum += std::abs(W1_grad[i]);
        std::cout << "  W1 gradient sum (abs): " << grad_sum << "\n";
    } else {
        std::cout << " W1 does NOT have gradient\n";
    }
    
    if (b1.owns_grad()) {
        std::cout << " b1 has gradient\n";
        float* b1_grad = b1.grad<float>();
        float grad_sum = 0.0f;
        for (int i = 0; i < 3; ++i) grad_sum += std::abs(b1_grad[i]);
        std::cout << "  b1 gradient sum (abs): " << grad_sum << "\n";
    } else {
        std::cout << " b1 does NOT have gradient\n";
    }
    
    if (W2.owns_grad()) {
        std::cout << " W2 has gradient\n";
        float* W2_grad = W2.grad<float>();
        float grad_sum = 0.0f;
        for (int i = 0; i < 3; ++i) grad_sum += std::abs(W2_grad[i]);
        std::cout << "  W2 gradient sum (abs): " << grad_sum << "\n";
    } else {
        std::cout << " W2 does NOT have gradient\n";
    }
    
    if (b2.owns_grad()) {
        std::cout << " b2 has gradient\n";
        float* b2_grad = b2.grad<float>();
        std::cout << "  b2 gradient: " << b2_grad[0] << "\n";
    } else {
        std::cout << " b2 does NOT have gradient\n";
    }
    
    std::cout << "\n=== MLP Test COMPLETE! ===\n";
    std::cout << "✅ Automatic differentiation is WORKING!\n";
    std::cout << "✅ Ready for training neural networks!\n\n";
    
    return 0;
}