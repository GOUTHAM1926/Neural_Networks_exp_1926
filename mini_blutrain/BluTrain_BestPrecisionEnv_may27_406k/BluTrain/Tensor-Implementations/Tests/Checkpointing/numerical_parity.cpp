#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/TrigonometryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/ActivationOps.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/TensorOps.h"
#include "device/DeviceCore.h"
#include "core/Tensor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// Helper to compare tensors
bool all_close(const Tensor& a, const Tensor& b, float tol = 1e-5) {
    if (a.shape() != b.shape()) return false;
    // Calculation of diff for verification (not on grad tape)
    Tensor diff = OwnTensor::abs(a - b); 
    float max_diff = *OwnTensor::reduce_max(diff).to_cpu().data<float>();
    return max_diff < tol;
}

// A deep block of operations
variable_list deep_mlp_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    for (int i = 0; i < 5; ++i) {
        x = autograd::sin(x);
        x = autograd::relu(x);
        // Use a dummy weight for "linear-ish" op
        Tensor w = Tensor::full(x.shape(), x.opts(), 0.5f);
        x = autograd::mul(x, w);
        x = autograd::add(x, Tensor::full(x.shape(), x.opts(), 0.1f));
    }
    return {x};
}

void run_parity_test(DeviceIndex device) {
    std::string dev_name = device.is_cpu() ? "CPU" : "GPU";
    std::cout << "Running Numerical Parity Test on " << dev_name << "...\n";

    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::rand<float>(Shape{{64, 64}}, opts);
    
    // 1. Without Checkpointing
    Tensor out1 = deep_mlp_block({x})[0];
    autograd::backward(autograd::sum(out1));
    Tensor grad_no_cp = x.grad_view().clone();
    
    x.zero_grad();
    
    // 2. With Checkpointing
    variable_list out2 = checkpoint(deep_mlp_block, {x});
    autograd::backward(autograd::sum(out2[0]));
    Tensor grad_cp = x.grad_view().clone();
    
    if (all_close(grad_no_cp, grad_cp)) {
        std::cout << "  - Parity check PASSED\n";
    } else {
        std::cerr << "  - Parity check FAILED!\n";
        exit(1);
    }
}

int main() {
    try {
        run_parity_test(DeviceIndex(Device::CPU)); // CPU
        
        #ifdef WITH_CUDA
        if (OwnTensor::device::cuda_available()) {
            run_parity_test(DeviceIndex(Device::CUDA, 0)); // GPU
        } else {
            std::cout << "CUDA not available, skipping GPU test.\n";
        }
        #endif
        
        std::cout << "All parity tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
