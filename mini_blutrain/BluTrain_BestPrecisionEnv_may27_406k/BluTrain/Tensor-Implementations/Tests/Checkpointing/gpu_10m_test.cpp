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
#include <chrono>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

/**
 * 10M Parameter MLP Block for GPU
 * -------------------------------
 * Hidden Size (H) = 1024
 * Matrix W (HxH) = 1024 * 1024 = 1,048,576 params (~4 MB @ float32)
 * 10 Layers = 10 * 1.04M = 10,485,760 params (~41.9 MB @ float32)
 */
struct MLPGPU10M {
    std::vector<Tensor> weights;
    std::vector<Tensor> biases;

    MLPGPU10M(DeviceIndex device) {
        TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
        for (int i = 0; i < 10; ++i) {
            weights.push_back(Tensor::rand<float>(Shape{{1024, 1024}}, opts) * 0.01f);
            biases.push_back(Tensor::full(Shape{{1024}}, opts, 0.1f));
        }
    }

    variable_list forward(const variable_list& inputs) {
        Tensor x = inputs[0]; // Expect [B, 1024]
        for (int i = 0; i < 10; ++i) {
            x = autograd::linear(x, weights[i], biases[i]);
            x = autograd::relu(x);
        }
        return {x};
    }
};

void run_gpu_test(DeviceIndex device) {
    if (!device.is_cuda()) {
        std::cerr << "This test requires a CUDA device.\n";
        return;
    }
    
    std::cout << "--- GPU Scale Test: 10M Parameter MLP (CUDA:" << device.index << ") ---\n";

    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{1024, 1024}}, opts);
    
    MLPGPU10M model(device);
    AllocationTracker& tracker = AllocationTracker::instance();
    tracker.init("memory_test_gpu_10m.csv");

    int dev_id = device.index;
    
    // Warmup
    tracker.reset_peak();
    x.zero_grad();
    
    // --- Manual Checkpointing Test ---
    std::cout << "Running WITH checkpointing...\n\n";
    
    size_t start_mem = tracker.get_current_allocated(dev_id);
    
    // 1. Forward Pass
    auto start_time_fwd = std::chrono::high_resolution_clock::now();
    
    auto cp_fn = [&](const variable_list& inputs) { return model.forward(inputs); };
    variable_list out_cp = checkpoint(cp_fn, {x});

    #ifdef WITH_CUDA
    cudaDeviceSynchronize();
    #endif
    auto end_time_fwd = std::chrono::high_resolution_clock::now();
    
    size_t mem_after_fwd = tracker.get_current_allocated(dev_id);
    double fwd_time_ms = std::chrono::duration<double, std::milli>(end_time_fwd - start_time_fwd).count();
    
    std::cout << "[Forward Pass]\n";
    std::cout << "  Time:   " << fwd_time_ms << " ms\n";
    std::cout << "  Memory: " << (mem_after_fwd - start_mem) / 1024 << " KB (Persistent)\n\n";

    // 2. Backward Pass
    auto start_time_bwd = std::chrono::high_resolution_clock::now();
    
    autograd::backward(autograd::sum(out_cp[0]));

    #ifdef WITH_CUDA
    cudaDeviceSynchronize();
    #endif
    auto end_time_bwd = std::chrono::high_resolution_clock::now();
    
    double bwd_time_ms = std::chrono::duration<double, std::milli>(end_time_bwd - start_time_bwd).count();
    size_t peak_mem_during_bwd = tracker.get_peak_allocated(dev_id);
    
    std::cout << "[Backward Pass]\n";
    std::cout << "  Time:   " << bwd_time_ms << " ms\n";
    // Peak memory during backward includes recomputed activations
    std::cout << "  Peak Memory delta: " << (peak_mem_during_bwd - start_mem) / 1024 << " KB\n";  
    
    std::cout << "------------------------------------------------------\n";
}

int main() {
    try {
        #ifdef WITH_CUDA
        if (OwnTensor::device::cuda_available()) {
            run_gpu_test(DeviceIndex(Device::CUDA, 0));
        } else {
            std::cerr << "CUDA is not available, cannot run GPU test.\n";
            return 1;
        }
        #else
        std::cerr << "Not compiled with CUDA support.\n";
        return 1;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "GPU Scale Test Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
