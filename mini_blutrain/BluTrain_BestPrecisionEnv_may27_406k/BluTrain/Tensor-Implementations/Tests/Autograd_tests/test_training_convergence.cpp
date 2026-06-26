#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace OwnTensor;

// test_training_convergence.cpp
// Verify that a simple MLP can learn a function (loss should decrease over epochs)

// Simple MLP: input(2) -> hidden(4) -> output(1)
// Task: Learn y = x1 + x2 (sum of inputs)

int main() {
    std::cout << "Starting training convergence test..." << std::endl;
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    TensorOptions no_grad = TensorOptions().with_req_grad(false);
    
    // Initialize weights (small values)
    Tensor W1 = Tensor::randn<float>(Shape{{2, 4}}, req_grad);
    Tensor b1 = Tensor::zeros(Shape{{4}}, req_grad);
    Tensor W2 = Tensor::randn<float>(Shape{{4, 1}}, req_grad);
    Tensor b2 = Tensor::zeros(Shape{{1}}, req_grad);
    
    // Training data: y = x1 + x2
    // Single sample: x = [1, 2], y = 3
    Tensor x = Tensor::ones(Shape{{1, 2}}, no_grad);
    x.data<float>()[0] = 1.0f;
    x.data<float>()[1] = 2.0f;
    
    Tensor target = Tensor::ones(Shape{{1, 1}}, no_grad);
    target.data<float>()[0] = 3.0f;  // Expected: 1 + 2 = 3
    
    float learning_rate = 0.01f;
    int epochs = 100;
    
    std::vector<float> losses;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // IMPORTANT: Zero gradients before each forward pass
        W1.zero_grad();
        W2.zero_grad();
        b1.zero_grad();
        b2.zero_grad();

        // Forward pass
        Tensor h1 = autograd::matmul(x, W1);       // (1,2) @ (2,4) = (1,4)
        Tensor h1_bias = autograd::add(h1, b1);    // (1,4) + (4) = (1,4)
        Tensor h1_act = autograd::relu(h1_bias);   // ReLU
        
        Tensor out = autograd::matmul(h1_act, W2); // (1,4) @ (4,1) = (1,1)
        Tensor output = autograd::add(out, b2);    // (1,1) + (1) = (1,1)
        
        // MSE Loss: (output - target)^2
        // Create neg_one tensor for subtraction
        Tensor neg_one = Tensor::ones(Shape{{1, 1}}, no_grad);
        neg_one.data<float>()[0] = -1.0f;
        Tensor neg_target = autograd::mul(target, neg_one);
        Tensor diff = autograd::add(output, neg_target);
        Tensor loss = autograd::mul(diff, diff);
        loss = autograd::mean(loss);
        
        float loss_val = loss.data<float>()[0];
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cerr << "NaN/Inf loss detected at epoch " << epoch << "!" << std::endl;
            break;
        }
        losses.push_back(loss_val);
        
        // Backward pass
        loss.backward();
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << ": Loss = " << loss_val << ", Output = " << output.data<float>()[0] << std::endl;
            if (epoch == 0 || epoch == 20) {
                std::cout << "  W1 grad[0]: " << (W1.owns_grad() ? W1.grad<float>()[0] : -1.0f) << std::endl;
                std::cout << "  W2 grad[0]: " << (W2.owns_grad() ? W2.grad<float>()[0] : -1.0f) << std::endl;
            }
        }
        
        // SGD update (manual)
        // Note: This is a simplified update. In practice, you'd use an optimizer.
        // W = W - lr * grad
        if (W1.owns_grad()) {
            float* w1_data = W1.data<float>();
            float* w1_grad = W1.grad<float>();
            for (int64_t i = 0; i < W1.numel(); ++i) {
                w1_data[i] -= learning_rate * w1_grad[i];
            }
        }
        if (W2.owns_grad()) {
            float* w2_data = W2.data<float>();
            float* w2_grad = W2.grad<float>();
            for (int64_t i = 0; i < W2.numel(); ++i) {
                w2_data[i] -= learning_rate * w2_grad[i];
            }
        }
        if (b1.owns_grad()) {
            float* b1_data = b1.data<float>();
            float* b1_grad = b1.grad<float>();
            for (int64_t i = 0; i < b1.numel(); ++i) {
                b1_data[i] -= learning_rate * b1_grad[i];
            }
        }
        if (b2.owns_grad()) {
            float* b2_data = b2.data<float>();
            float* b2_grad = b2.grad<float>();
            for (int64_t i = 0; i < b2.numel(); ++i) {
                b2_data[i] -= learning_rate * b2_grad[i];
            }
        }
    }
    
    // Verification: Loss should decrease
    float initial_loss = losses[0];
    float final_loss = losses.back();
    
    std::cout << "\nInitial Loss: " << initial_loss << std::endl;
    std::cout << "Final Loss: " << final_loss << std::endl;
    
    if (final_loss < initial_loss * 0.5f) {
        std::cout << "PASS: Loss decreased significantly (convergence achieved)" << std::endl;
    } else if (final_loss < initial_loss) {
        std::cout << "PASS: Loss decreased (training is working, but slow)" << std::endl;
    } else {
        std::cerr << "FAIL: Loss did not decrease! Training is broken." << std::endl;
        return 1;
    }
    
    std::cout << "\nTraining convergence test completed!" << std::endl;
    return 0;
}