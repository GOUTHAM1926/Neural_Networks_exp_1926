#include <iostream>
#include <vector>
#include <cassert>
#include "core/Tensor.h"
#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "checkpointing/Checkpointing.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;

// Simple module for testing
class TestModule : public nn::Module {
public:
    Tensor w0;
    Tensor w1;

    TestModule() {
        // Parameter on GPU 0
        w0 = Tensor::full(Shape{{2, 2}}, TensorOptions().with_device(DeviceIndex(Device::CUDA, 0)), 0.0f);
        // Parameter on GPU 1
        w1 = Tensor::full(Shape{{2, 2}}, TensorOptions().with_device(DeviceIndex(Device::CUDA, 1)), 1.0f);
        
        register_parameter(w0);
        register_parameter(w1);
    }

    Tensor forward(const Tensor& input) override {
        return input;
    }
};

void test_multi_gpu_save_load() {
    std::cout << "Running Multi-GPU Checkpoint Test..." << std::endl;

    TestModule model;
    
    // Create a dummy optimizer (SGD)
    nn::SGDOptimizer optimizer(model.parameters(), 0.01f);

    CheckpointManager manager("test_checkpoints", "multi_gpu");
    
    // Initial values
    std::cout << "Initial state:" << std::endl;
    std::cout << "  w0 (GPU 0) sum: " << autograd::sum(model.w0.to_cpu()).data<float>()[0] << std::endl;
    std::cout << "  w1 (GPU 1) sum: " << autograd::sum(model.w1.to_cpu()).data<float>()[0] << std::endl;

    // Save checkpoint
    manager.save(1, model, optimizer, 0.5f);
    std::cout << "Saved checkpoint at step 1." << std::endl;

    // Modify values
    model.w0.fill(2.0f);
    model.w1.fill(3.0f);
    std::cout << "Modified state:" << std::endl;
    std::cout << "  w0 (GPU 0) sum: " << autograd::sum(model.w0.to_cpu()).data<float>()[0] << std::endl;
    std::cout << "  w1 (GPU 1) sum: " << autograd::sum(model.w1.to_cpu()).data<float>()[0] << std::endl;

    // Load checkpoint
    int step = 0;
    float loss = 0.0f;
    bool success = manager.load_latest(model, optimizer, step, loss);
    
    assert(success);
    assert(step == 1);
    
    float w0_sum = autograd::sum(model.w0.to_cpu()).data<float>()[0];
    float w1_sum = autograd::sum(model.w1.to_cpu()).data<float>()[0];
    
    std::cout << "Restored state:" << std::endl;
    std::cout << "  w0 (GPU 0) sum: " << w0_sum << std::endl;
    std::cout << "  w1 (GPU 1) sum: " << w1_sum << std::endl;

    if (w0_sum == 0.0f && w1_sum == 4.0f) {
        std::cout << "✔ SUCCESS: Multi-GPU checkpointing works correctly." << std::endl;
    } else {
        std::cout << "✘ FAILURE: Multi-GPU checkpointing failed validation." << std::endl;
        std::cout << "Expected w0 sum 0.0, got " << w0_sum << std::endl;
        std::cout << "Expected w1 sum 4.0, got " << w1_sum << std::endl;
        exit(1);
    }
}

int main() {
    try {
        test_multi_gpu_save_load();
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
