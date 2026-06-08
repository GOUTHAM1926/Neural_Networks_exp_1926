#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReductionOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/TensorOps.h"
#include "ops/ScalarOps.h"
#include "device/AllocationTracker.h"
#include "device/DeviceCore.h"
#include "core/Tensor.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

/**
 * 20M Parameter MLP Block
 * -----------------------
 * Hidden Size (H) = 2048
 * Matrix W (HxH) = 2048 * 2048 = 4,194,304 params (~16 MB @ float32)
 * 5 Layers = 5 * 4.19M = 20,971,520 params (~83.8 MB @ float32)
 */
struct MLP20M {
    std::vector<Tensor> weights;
    std::vector<Tensor> biases;

    MLP20M(DeviceIndex device) {
        TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
        for (int i = 0; i < 5; ++i) {
            // Initialize with small random values
            weights.push_back(Tensor::rand<float>(Shape{{2048, 2048}}, opts) * 0.01f);
            biases.push_back(Tensor::full(Shape{{2048}}, opts, 0.1f));
        }
    }

    variable_list forward(const variable_list& inputs) {
        Tensor x = inputs[0]; // Expect [B, 2048]
        for (int i = 0; i < 5; ++i) {
            // Linear operation: x @ W + b
            x = autograd::linear(x, weights[i], biases[i]);
            x = autograd::relu(x);
        }
        return {x};
    }
};

void run_20m_test(DeviceIndex device) {
    std::string dev_name = device.is_cpu() ? "CPU" : "GPU";
    int dev_id = device.is_cpu() ? -1 : device.index;
    
    std::cout << "--- Scale Test: 20M Parameter MLP on " << dev_name << " ---\n";

    // Batch size = 1024, Hidden = 2048
    // One activation = 1024 * 2048 * 4 bytes = 8 MB
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{1024, 2048}}, opts);
    
    MLP20M model(device);
    AllocationTracker& tracker = AllocationTracker::instance();
    tracker.init("memory_test_20m.csv");

    size_t fwd_mem_standard = 0;

    // 1. Without Checkpointing
    // tracker.reset_peak();
    // size_t start_mem = tracker.get_current_allocated(dev_id);
    // {
    //     std::cout << "Running WITHOUT checkpointing...\n";
    //     auto out = model.forward({x})[0];
    //     fwd_mem_standard = tracker.get_current_allocated(dev_id);
    //     std::cout << "  Persistent Activation Memory: " << (fwd_mem_standard - start_mem) / 1024 << " KB\n";
        
    //     autograd::backward(autograd::sum(out));
    // }
    // x.zero_grad();

    // 2. With Checkpointing
    tracker.reset_peak();
    size_t start_mem_cp = tracker.get_current_allocated(dev_id);
    {
        std::cout << "Running WITH checkpointing...\n";
        // Define a wrapper for model.forward to pass to checkpoint
        auto cp_fn = [&](const variable_list& inputs) { return model.forward(inputs); };
        
        variable_list out_cp = checkpoint(cp_fn, {x});
        size_t fwd_mem_cp = tracker.get_current_allocated(dev_id);
        
        std::cout << "  Persistent Activation Memory: " << (fwd_mem_cp - start_mem_cp) / 1024 << " KB\n";
        
        // size_t baseline_diff = fwd_mem_standard - start_mem;
        // size_t cp_diff = fwd_mem_cp - start_mem_cp;
        
        // if (baseline_diff > 0) {
            // std::cout << "  Savings: " << (baseline_diff - cp_diff) / 1024 << " KB\n";
        // }
        std::cout<<"Backward pass...";
        autograd::backward(autograd::sum(out_cp[0]));
        std::cout<<"Done!\n";
    }
    std::cout << "---------------------------------------------------\n\n";
}

int main() {
    try {
        run_20m_test(DeviceIndex(Device::CPU));
        
        // #ifdef WITH_CUDA
        // if (OwnTensor::device::cuda_available()) {
            // run_20m_test(DeviceIndex(Device::CUDA, 0));
        // }
        // #endif
    } catch (const std::exception& e) {
        std::cerr << "Scale Test Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
