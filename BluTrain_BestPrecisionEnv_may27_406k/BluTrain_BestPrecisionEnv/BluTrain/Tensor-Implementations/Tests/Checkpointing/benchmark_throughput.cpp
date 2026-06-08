#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "device/Device.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// A computationally intensive block to simulate a model layer
variable_list model_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    // Matmul: (N, N) x (N, N) -> O(N^3)
    // We create a weight matrix on the fly (in real models it's a parameter)
    Tensor w = Tensor::randn<float>(x.shape(), x.opts(), 42); 
    
    x = autograd::matmul(x, w);
    x = autograd::relu(x);
    x = autograd::matmul(x, w); // Another one
    return {x};
}

void benchmark_size(int size) {
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Benchmarking Model Size: " << size << "x" << size << "\n";
    std::cout << "--------------------------------------------------\n";

    DeviceIndex device(Device::CUDA, 0);
    if (!OwnTensor::device::cuda_available()) {
        std::cout << "CUDA not available, running on CPU (results may be different)...\n";
        device = DeviceIndex(Device::CPU);
    }

    Tensor x = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(device).with_req_grad(true));
    
    int iterations = 10;
    
    // lambda to run benchmark
    auto run_test = [&](bool offload, const std::string& label) {
        // Warmup
        {
            Tensor curr = x;
            for(int k=0; k<5; ++k) {
                variable_list out = checkpoint(model_block, {curr}, offload);
                curr = out[0];
            }
            autograd::backward(autograd::sum(curr));
            x.zero_grad();
        }

        // Sync device
        if (device.is_cuda()) {
            cudaDeviceSynchronize();
        }

        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            Tensor curr = x;
            for(int k=0; k<5; ++k) {
                variable_list out = checkpoint(model_block, {curr}, offload);
                curr = out[0];
            }
            autograd::backward(autograd::sum(curr));
            x.zero_grad();
        }
        
        if (device.is_cuda()) {
            cudaDeviceSynchronize();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time = duration.count() / iterations;
        
        std::cout << std::left << std::setw(25) << label 
                  << ": " << std::fixed << std::setprecision(2) << avg_time << " ms / iter\n";
    };

    run_test(false, "GPU Checkpointing (No Offload)");
    run_test(true,  "CPU Offloading Checkpointing");
}

int main() {
    try {
        std::cout << "==================================================\n";
        std::cout << "   CPU Offloading Throughput Benchmark\n";
        std::cout << "==================================================\n";
        
        // Small Model
        benchmark_size(512);
        
        // Medium Model
        benchmark_size(1024);
        
        // Large Model
        benchmark_size(2048);

        // Extra Large (if memory allows)
        // benchmark_size(4096); 

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
