#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "TensorLib.h"
#include "dtype/Types.h" // Needed for float16_t
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

void test_fp16_adamw_update() {
    std::cout << "Testing AdamW FP16 Mixed Precision Update..." << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    
    // Initialize FP16 Parameter
    Tensor p = Tensor::full(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float16), 1.0f);
    p.set_requires_grad(true);
    
    // Initialize FP16 Gradient
    Tensor g = Tensor::full(p.shape(), TensorOptions().with_dtype(Dtype::Float16), 1.0f);
    p.set_grad(g);
    
    // Setup Optimizer
    AdamW optim({p}, 1.0f);
    
    // Set Loss Scale via LossScaler
    float scale = 33554432.0f; // 2^25
    LossScaler scaler(scale);
    optim.set_scaler(&scaler);

    // Print setup info
    std::cout << "Setup:" << std::endl;
    std::cout << "  Parameter Value (FP16): 1.0" << std::endl;
    std::cout << "  Gradient Value (FP16):  1.0" << std::endl;
    std::cout << "  Loss Scale:             2^25 (" << std::fixed << std::setprecision(1) << scale << ")" << std::endl;
    
    double unscaled_grad = 1.0 / scale;
    std::cout << "  Unscaled Gradient:      " << std::scientific << std::setprecision(2) << unscaled_grad << std::endl;
    std::cout << "  FP16 Min Subnormal:     ~5.96e-08" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Hypothesis:" << std::endl;
    std::cout << "  If unscaling happens in FP16: " << unscaled_grad << " < 5.96e-08 -> Underflows to 0.0 -> No Update." << std::endl;
    std::cout << "  If unscaling happens in FP32: " << unscaled_grad << " is preserved -> Valid Update." << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    
    // 5. Step
    optim.step();
    
    // 6. Verification
    std::cout << "Results:" << std::endl;
    
    // Verify Master Weight exists and updated
    Tensor* master = optim.get_master_weight(p);
    if (master == nullptr) {
        std::cout << "  [FAIL] Master weight not created!" << std::endl;
        exit(1);
    }
    
    if (master->dtype() != Dtype::Float32) {
        std::cout << "  [FAIL] Master weight is not Float32!" << std::endl;
        exit(1);
    }
    
    // p is CPU, so master is CPU. Access data directly.
    float master_val = master->data<float>()[0];
    std::cout << "  Master Weight (FP32) after step: " << std::fixed << std::setprecision(6) << master_val << std::endl;
    
    if (master_val < 0.999999f) {
        std::cout << "  [PASS] Master weight updated! (Value < 1.0)" << std::endl;
        std::cout << "         This proves unscaling occurred in FP32 precision." << std::endl;
    } else {
        std::cout << "  [FAIL] Master weight unchanged (Value == 1.0). Unscaling likely underflowed in FP16." << std::endl;
        exit(1);
    }
    
    // B. Verify Parameter is still FP16
    if (p.dtype() != Dtype::Float16) {
        std::cout << "  [FAIL] Parameter dtype changed! Expected Float16." << std::endl;
        exit(1);
    }
    
    // Convert back to float to check value
    float p_val = static_cast<float>(p.data<float16_t>()[0]);
    std::cout << "  Param Weight (FP16) after step:  " << std::fixed << std::setprecision(6) << p_val << std::endl;
    
    if (std::abs(p_val - 1.0f) > 1e-4) {
         std::cout << "  [PASS] Parameter correctly updated on cast-back." << std::endl;
    } else {
         std::cout << "  [WARN] Parameter value didn't change enough for FP16 representation?" << std::endl;
    }
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main() {
    try {
        test_fp16_adamw_update();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
