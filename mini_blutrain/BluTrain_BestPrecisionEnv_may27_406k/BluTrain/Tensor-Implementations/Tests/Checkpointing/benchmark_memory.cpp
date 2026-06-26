#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "device/Device.h"
#include "device/AllocationTracker.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// A computationally intensive block to simulate a model layer
variable_list model_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor w = Tensor::randn<float>(x.shape(), x.opts(), 42); // Weights
    
    // Several operations to increase activation memory usage if not checkpointed
    // But checkpointing saves inputs.
    // The input 'x' is what is saved.
    // So the memory saving comes from moving 'x' (and potential other inputs) to CPU.
    
    x = autograd::matmul(x, w);
    x = autograd::relu(x);
    x = autograd::matmul(x, w);
    return {x};
}

void benchmark_memory(int size) {
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Memory Benchmark Model Size: " << size << "x" << size << "\n";
    std::cout << "--------------------------------------------------\n";

    DeviceIndex device(Device::CUDA, 0);
    if (!OwnTensor::device::cuda_available()) {
        std::cout << "CUDA not available, skipping GPU memory test.\n";
        return;
    }

    Tensor x = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(device).with_req_grad(true));
    
    // lambda to run benchmark
    auto run_test = [&](bool offload, const std::string& label) {
        // Clear previous stats
        AllocationTracker::instance().reset_peak();
        
        {
            // Run sequence of checkpoints
            // x -> chk -> x2 -> chk -> x3 ...
            Tensor curr = x;
            for(int i=0; i<5; ++i) {
                variable_list out = checkpoint(model_block, {curr}, offload);
                curr = out[0];
            }
            autograd::backward(autograd::sum(curr));
            x.zero_grad();
        }
        
        if (device.is_cuda()) {
            cudaDeviceSynchronize();
        }
        
        size_t peak_bytes = AllocationTracker::instance().get_peak_allocated(0); // Device 0
        double peak_mb = peak_bytes / (1024.0 * 1024.0);
        
        std::cout << std::left << std::setw(30) << label 
                  << ": Peak GPU Mem = " << std::fixed << std::setprecision(2) << peak_mb << " MB\n";
                  
        // Reset for next run
        AllocationTracker::instance().reset_peak();
    };

    run_test(false, "GPU Checkpointing (No Offload)");
    run_test(true,  "CPU Offloading Checkpointing");
}

int main() {
    try {
        AllocationTracker::instance().init("mem_benchmark.csv");
        
        std::cout << "==================================================\n";
        std::cout << "   CPU Offloading Memory Benchmark\n";
        std::cout << "==================================================\n";
        
        // Ensure AllocationTracker tracks peaks properly
        // It's usually enabled by default in debug builds or if integrated deep in allocator.
        
        // Small Model
        benchmark_memory(512);
        
        // Medium Model
        benchmark_memory(1024);
        
        // Large Model
        benchmark_memory(2048);
        
        // Extra Large
        benchmark_memory(4096);

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
