#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <iomanip>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "nn/optimizer/LossScaler.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

void test_sgd_scaling() {
    std::cout << "Testing SGD Scaling..." << std::endl;
    // Param = 1.0
    Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32));
    p.set_requires_grad(true);
    
    // Manually set grad tensor
    Tensor g = Tensor::full(p.shape(), p.opts(), 2.0f);
    p.set_grad(g);
    
    // Optimizer: SGD, lr=1.0
    SGDOptimizer optim({p}, 1.0f);
    
    // Loss Scale = 2.0
    LossScaler sgd_scaler(2.0f);
    optim.set_scaler(&sgd_scaler);
    
    // Step
    // Update should be: p = p - lr * (grad / scale)
    // p = 1.0 - 1.0 * (2.0 / 2.0) = 0.0
    optim.step();
    
    float val = p.to_cpu().data<float>()[0];
    std::cout << "  Param value after step (expected 0.0): " << val << std::endl;
    
    if (std::abs(val - 0.0f) < 1e-6) {
        std::cout << "  [PASS] SGD Scaling" << std::endl;
    } else {
        std::cout << "  [FAIL] SGD Scaling. Got " << val << std::endl;
        exit(1);
    }
}

void test_adam_scaling() {
    std::cout << "Testing Adam Scaling..." << std::endl;
    
    // Param = 1.0
    Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32));
    p.set_requires_grad(true);
    
    // Grad = 1.0
    Tensor g = Tensor::full(p.shape(), p.opts(), 1.0f);
    p.set_grad(g);
    
    // Optimizer: Adam, lr=1.0, betas=(0.9, 0.999), eps=1e-8
    // First step:
    // Loss Scale = 2.0
    // Real Grad = 1.0 / 2.0 = 0.5
    // m = (1-0.9)*0.5 = 0.05
    // v = (1-0.999)*(0.5^2) = 0.001 * 0.25 = 0.00025
    // m_hat = 0.05 / (1-0.9) = 0.5
    // v_hat = 0.00025 / (1-0.999) = 0.25
    // step = lr * m_hat / (sqrt(v_hat) + eps)
    // step = 1.0 * 0.5 / (0.5 + 1e-8) approx 1.0
    // p_new = 1.0 - 1.0 = 0.0
    
    Adam optim({p}, 1.0f);
    LossScaler adam_scaler(2.0f);
    optim.set_scaler(&adam_scaler);
    optim.step();
    
    float val = p.to_cpu().data<float>()[0];
    std::cout << "  Param value after step (expected approx 0.0): " << val << std::endl;
    
    if (std::abs(val - 0.0f) < 1e-4) {
        std::cout << "  [PASS] Adam Scaling" << std::endl;
    } else {
        std::cout << "  [FAIL] Adam Scaling. Got " << val << std::endl;
        exit(1);
    }
}

void test_overflow_check() {
    std::cout << "Testing Overflow Check..." << std::endl;
    Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32));
    p.set_requires_grad(true);
    
    // Set Inf grad
    Tensor g = Tensor::full(p.shape(), p.opts(), std::numeric_limits<float>::infinity());
    p.set_grad(g);
    
    LossScaler scaler;
    bool overflow = scaler.check_overflow({p});
    
    if (overflow) {
        std::cout << "  [PASS] Overflow detected" << std::endl;
    } else {
        std::cout << "  [FAIL] Overflow NOT detected" << std::endl;
        exit(1);
    }
    
    // Verify param grad was NOT modified
    if (std::isinf(p.grad_view().data<float>()[0])) {
         std::cout << "  [PASS] Gradients not modified" << std::endl;
    } else {
         std::cout << "  [FAIL] Gradients were modified!" << std::endl;
         exit(1);
    }
}

int main() {
    try {
        test_sgd_scaling();
        test_adam_scaling();
        test_overflow_check();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
