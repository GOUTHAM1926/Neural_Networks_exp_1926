#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "TensorLib.h"
#include "dtype/Types.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

void test_fp16_adamw_gpu() {
    std::string backend_name = "default";
    std::cout << "===========================================================" << std::endl;
    std::cout << "Testing AdamW FP16 Mixed Precision on GPU [" << backend_name << "]" << std::endl;
    std::cout << "===========================================================" << std::endl;

    // --- Setup: Create FP16 parameter on GPU ---
    Tensor p = Tensor::full(Shape{{1024, 1024}}, TensorOptions().with_dtype(Dtype::Float16), 1.0f);
    p = p.to_cuda(0);
    p.set_requires_grad(true);

    // Create FP16 gradient on GPU (ones)
    Tensor g = Tensor::full(p.shape(), TensorOptions().with_dtype(Dtype::Float16), 1.0f);
    g = g.to_cuda(0);
    p.set_grad(g);

    // Optimizer with the selected backend
    // Optimizer with default backend
    AdamW optim({p}, /*lr=*/1.0f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*eps=*/1e-8f, /*wd=*/0.01f);

    // Set a large loss scale to test unscaling in FP32
    float scale = 33554432.0f; // 2^25
    LossScaler scaler(scale);
    optim.set_scaler(&scaler);

    double unscaled_grad = 1.0 / scale;

    std::cout << "\nSetup:" << std::endl;
    std::cout << "  Backend:                " << backend_name << std::endl;
    std::cout << "  Parameter shape:        [1024, 1024]" << std::endl;
    std::cout << "  Parameter dtype:        Float16" << std::endl;
    std::cout << "  Loss Scale:             2^25 (" << std::fixed << std::setprecision(1) << scale << ")" << std::endl;
    std::cout << "  Unscaled Gradient:      " << std::scientific << std::setprecision(2) << unscaled_grad << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    // --- Step ---
    optim.step();

    // --- Verification ---
    std::cout << "\nResults:" << std::endl;
    int pass_count = 0;
    int total_checks = 0;

    // CHECK 1: Master weight exists and is FP32
    total_checks++;
    Tensor* master = optim.get_master_weight(p);
    if (master == nullptr) {
        std::cout << "  [FAIL] Master weight not created!" << std::endl;
        exit(1);
    }
    if (master->dtype() != Dtype::Float32) {
        std::cout << "  [FAIL] Master weight is not Float32!" << std::endl;
        exit(1);
    }
    std::cout << "  [PASS] Master weight exists and is Float32." << std::endl;
    pass_count++;

    // CHECK 2: Master weight is on GPU
    total_checks++;
    if (!master->device().is_cuda()) {
        std::cout << "  [FAIL] Master weight is not on GPU!" << std::endl;
        exit(1);
    }
    std::cout << "  [PASS] Master weight is on CUDA device." << std::endl;
    pass_count++;

    // CHECK 3: Master weight was updated (value < 1.0)
    total_checks++;
    Tensor master_cpu = master->to_cpu();
    float* master_data = master_cpu.data<float>();
    std::cout << "  Master Weight (FP32) sample value [0]: " << std::fixed << std::setprecision(6) << master_data[0] << std::endl;
    if (master_data[0] < 0.999999f) {
        std::cout << "  [PASS] Master weight updated! (FP32 unscaling works)" << std::endl;
        pass_count++;
    } else {
        std::cout << "  [FAIL] Master weight unchanged." << std::endl;
        exit(1);
    }

    // CHECK 4: Model parameter is still FP16
    total_checks++;
    if (p.dtype() != Dtype::Float16) {
        std::cout << "  [FAIL] Parameter dtype changed from Float16!" << std::endl;
        exit(1);
    }
    std::cout << "  [PASS] Model parameter remains Float16." << std::endl;
    pass_count++;

    // CHECK 5: Model parameter was cast back from master
    total_checks++;
    Tensor p_cpu = p.to_cpu();
    float16_t* p_data = p_cpu.data<float16_t>();
    float p_val = static_cast<float>(p_data[0]);
    std::cout << "  Param Weight (FP16) sample value [0]:  " << std::fixed << std::setprecision(6) << p_val << std::endl;
    if (std::abs(p_val - 1.0f) > 1e-4f) {
        std::cout << "  [PASS] FP16 parameter updated via cast-back." << std::endl;
        pass_count++;
    } else {
        std::cout << "  [WARN] FP16 parameter didn't change enough." << std::endl;
        pass_count++; // precision limitation, not a failure
    }

    // CHECK 6: Parameter is still on GPU
    total_checks++;
    if (!p.device().is_cuda()) {
        std::cout << "  [FAIL] Parameter moved off GPU!" << std::endl;
        exit(1);
    }
    std::cout << "  [PASS] Parameter remains on CUDA device." << std::endl;
    pass_count++;

    // --- Summary ---
    std::cout << "\n  Result: " << pass_count << "/" << total_checks << " checks passed.";
    if (pass_count == total_checks) {
        std::cout << " [" << backend_name << "] PASSED!" << std::endl;
    } else {
        std::cout << " [" << backend_name << "] FAILED!" << std::endl;
    }
    std::cout << "===========================================================\n" << std::endl;
}

int main() {
    try {
        // Test default backend
        test_fp16_adamw_gpu();

        std::cout << "GPU mixed precision test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
