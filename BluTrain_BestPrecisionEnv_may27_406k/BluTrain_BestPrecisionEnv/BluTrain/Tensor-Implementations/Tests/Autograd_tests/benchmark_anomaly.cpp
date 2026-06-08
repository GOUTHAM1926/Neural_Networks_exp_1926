#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/AnomalyMode.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;

// Simple timer class
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() { // returns ms
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

void run_benchmark(bool anomaly_enabled, int iterations, int size) {
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "BENCHMARK: AnomalyMode = " << (anomaly_enabled ? "ON" : "OFF") << std::endl;
    std::cout << "Workload: " << iterations << " iterations of " << size << "x" << size << " MatMul on CUDA" << std::endl;

    autograd::AnomalyMode::set_enabled(anomaly_enabled);

    // Warmup
    if (OwnTensor::device::cuda_available()) {
        Tensor w = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(DeviceIndex(Device::CUDA)));
    }

    Tensor A = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(DeviceIndex(Device::CUDA)).with_req_grad(true));
    Tensor B = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(DeviceIndex(Device::CUDA)).with_req_grad(true));

    std::vector<double> forward_times;
    std::vector<double> backward_times;
    Timer timer;

    // To ensure we measure graph building overhead, we create new nodes every iteration
    // by using the same leaf tensors but creating new graph structure
    for (int i = 0; i < iterations; ++i) {
        // Forward
        // Flush CUDA before timing
        cudaDeviceSynchronize();
        timer.start();
        
        // C = A * B + A
        Tensor C = autograd::matmul(A, B);
        Tensor D = autograd::add(C, A); 
        Tensor result = autograd::relu(D);

        cudaDeviceSynchronize();
        forward_times.push_back(timer.stop());

        // Backward
        // Create a grad tensor to start backward
        Tensor grad = Tensor::ones(result.shape(), TensorOptions().with_device(DeviceIndex(Device::CUDA)));
        
        // Clear grads from previous iteration
        A.zero_grad();
        B.zero_grad();

        cudaDeviceSynchronize();
        timer.start();
        result.backward(&grad); // Pass grad explicitly
        cudaDeviceSynchronize();
        backward_times.push_back(timer.stop());
    }

    // Statistics
    double avg_fwd = std::reduce(forward_times.begin(), forward_times.end()) / iterations;
    double avg_bwd = std::reduce(backward_times.begin(), backward_times.end()) / iterations;
    double total_time = avg_fwd + avg_bwd;
    // Approximating throughput as "Wait-free Ops per second" (inverse of latency)
    double throughput = 1000.0 / total_time; 

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Results:" << std::endl;
    std::cout << "  Avg Forward Latency:  " << avg_fwd << " ms" << std::endl;
    std::cout << "  Avg Backward Latency: " << avg_bwd << " ms" << std::endl;
    std::cout << "  Avg Total per iter:   " << total_time << " ms" << std::endl;
    std::cout << "  Est. Throughput:      " << throughput << " iter/sec" << std::endl;
}

int main() {
    if (!OwnTensor::device::cuda_available()) {
        std::cerr << "CUDA unavailable, cannot run GPU benchmark." << std::endl;
        return 1;
    }

    int SIZE = 1024;
    int ITERATIONS = 50; 

    std::cout << "==================================================" << std::endl;
    std::cout << "       AnomalyMode Performance Benchmark          " << std::endl;
    std::cout << "==================================================" << std::endl;

    // Global Warmup
    std::cout << "Warming up GPU..." << std::endl;
    run_benchmark(false, 10, SIZE); // Warmup run
    std::cout << "Warmup complete.\n" << std::endl;

    // Run with AnomalyMode ON first (to see if it's slow)
    run_benchmark(true, ITERATIONS, SIZE);

    // Run without AnomalyMode OFF second
    run_benchmark(false, ITERATIONS, SIZE);

    std::cout << "==================================================" << std::endl;
    return 0;
}
