#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

void test_adamw_weight_decay_gpu() {
    std::cout << "===========================================================" << std::endl;
    std::cout << "Testing AdamW Weight Decay Isolation on GPU" << std::endl;
    std::cout << "===========================================================" << std::endl;

    // 1. Setup Parameter
    float initial_val = 1.0f;
    Tensor p = Tensor::full(Shape{{10}}, TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)), initial_val);
    p.set_requires_grad(true);

    // 2. Setup Gradient (ZERO)
    // We want to isolate weight decay. If grad is 0, Adam update (m, v) part should be 0.
    // Only weight decay should affect the parameter.
    Tensor g = Tensor::zeros(p.shape(), TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, 0)));
    p.set_grad(g);

    // 3. Setup Optimizer
    float lr = 0.1f;
    float wd = 0.1f;
    // beta1, beta2, eps don't matter if grad is 0
    AdamW optim({p}, lr, 0.9f, 0.999f, 1e-8f, wd);

    std::cout << "Initial Value: " << initial_val << std::endl;
    std::cout << "Learning Rate: " << lr << std::endl;
    std::cout << "Weight Decay:  " << wd << std::endl;
    std::cout << "Expected Decay: value * (1 - lr * wd)" << std::endl;
    std::cout << "Expected Value: " << initial_val * (1.0f - lr * wd) << std::endl;

    // 4. Step
    optim.step();

    // 5. Verify
    Tensor p_cpu = p.to_cpu();
    float* data = p_cpu.data<float>();
    float actual_val = data[0];

    std::cout << "Actual Value:   " << actual_val << std::endl;
    
    float expected_val = initial_val * (1.0f - lr * wd);
    float diff = std::abs(actual_val - expected_val);

    if (diff < 1e-5f) {
        std::cout << "[PASS] Weight decay applied correctly." << std::endl;
    } else {
        std::cout << "[FAIL] Weight decay mismatch! Diff: " << diff << std::endl;
        // Check if it matches coupled weight decay (Adam style)
        // Adam style: grad += wd * p.
        // grad_eff = 0 + 0.1 * 1.0 = 0.1.
        // m = (1-0.9)*0.1 = 0.01.
        // v = (1-0.999)*0.1^2 = 0.001 * 0.01 = 0.00001.
        // m_hat = 0.01 / (1-0.9) = 0.1.
        // v_hat = 0.00001 / (1-0.999) = 0.01.
        // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 0.1 / (0.1) = 0.1.
        // new_val = 1.0 - 0.1 = 0.9.
        
        // Wait!
        // AdamW: 1.0 * (1 - 0.1*0.1) = 1.0 * 0.99 = 0.99.
        
        // If it was standard Adam: parameter would become 0.9.
        if (std::abs(actual_val - 0.9f) < 1e-5f) {
             std::cout << "[INFO] Result matches standard Adam (coupled) behavior, not AdamW!" << std::endl;
        }
        exit(1);
    }
}

int main() {
    try {
        test_adamw_weight_decay_gpu();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
