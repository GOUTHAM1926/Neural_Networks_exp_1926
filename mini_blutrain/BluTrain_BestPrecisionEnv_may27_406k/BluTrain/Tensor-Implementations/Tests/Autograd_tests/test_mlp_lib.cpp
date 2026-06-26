#include "core/Tensor.h"
#include "nn/NN.h"
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace OwnTensor;
using namespace OwnTensor::nn;

// ============================================================================
// Main Test using Library Modules
// ============================================================================

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "--- Starting End-to-End CPU Training ---\n";
        std::cout << "========================================\n\n";

        const int batch_size = 8;
        const int in_features = 16;
        const int hidden_features = 32;
        const int out_features = 4;
        const float learning_rate = 0.01f;

        // Create Model using Library
        Sequential model_cpu({
            new Linear(in_features, hidden_features),
            new ReLU(),
            new Linear(hidden_features, out_features)
        });
        std::cout << "CPU Model created successfully.\n";

        // Create Data
        auto cpu_opts = TensorOptions().with_device(Device::CPU).with_dtype(Dtype::Float32);
        
        Tensor input = Tensor::randn<float>(Shape{{batch_size, in_features}}, cpu_opts);
        // Note: zeros creates expected output for testing
        Tensor labels = Tensor::zeros(Shape{{batch_size, out_features}}, cpu_opts);
        
        std::cout << "Created Input data: (" << batch_size << ", " << in_features << ")\n";
        
        // --- Initial Pass ---
        Tensor output = model_cpu(input);
        
        // Use library mse_loss
        Tensor loss = mse_loss(output, labels);
        loss.backward();

        float initial_loss_val = loss.data<float>()[0];
        std::cout << "Initial Loss (CPU): " << initial_loss_val << std::endl;
        
        // Verify Gradients
        // model_cpu.parameters()[0] is the weight of first Linear layer
        // Store parameters in a local variable to keep them alive! (Fix for dangling ref)
        auto params = model_cpu.parameters();
        const auto& w1 = params[0];
        
        assert(w1.grad() != nullptr && "Weight should own gradient");
        
        float grad_sum_cpu = 0.0f;
        const float* grad_data = w1.grad<float>(); // Use const float*
        size_t numel = w1.numel();
        for (size_t i=0; i<numel; ++i) {
            grad_sum_cpu += std::abs(grad_data[i]);
        }
        
        std::cout << "Gradient sum for Layer 1 weights: " << grad_sum_cpu << "\n";
        assert(grad_sum_cpu > 0.0f && "FATAL: Gradients are zero after first backward pass!");
        std::cout << "PASS: Gradients were computed successfully on CPU.\n";

        // --- SGD Step ---
        for (auto& param : model_cpu.parameters()) {
            if (param.requires_grad() && param.grad() != nullptr) {
                // param.data() -= lr * param.grad()
                float* p_data = param.data<float>();
                const float* g_data = param.grad<float>(); // Use const float*
                size_t n = param.numel();
                
                for (size_t i=0; i<n; ++i) {
                    p_data[i] -= learning_rate * g_data[i];
                }
            }
        }
        std::cout << "Performed one SGD step.\n";
        
        // --- Second Pass ---
        model_cpu.zero_grad();
        Tensor new_output = model_cpu(input);
        Tensor new_loss = mse_loss(new_output, labels);
        float new_loss_val = new_loss.data<float>()[0];
        std::cout << "New Loss (CPU): " << new_loss_val << std::endl;

        assert(new_loss_val < initial_loss_val && "FATAL: Loss did not decrease after training step!");
        std::cout << "PASS: Loss decreased on CPU, indicating successful training.\n";

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: An exception occurred during the test: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n✅ All end-to-end training tests passed successfully!\n";
    return 0;
}
