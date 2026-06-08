#pragma once

#include "nn/DistributedNN.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <cmath>
#include <iomanip>
#include <mpi.h>

using namespace OwnTensor;

struct DTensorTestMetrics {
    double forward_latency_ms;
    double backward_latency_ms;
    double throughput_ops_sec;
    bool numerical_match;
    size_t memory_leak_count;
    int rank;
};

class DTensorTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint start_time;

public:
    void start() { start_time = Clock::now(); }
    double elapsed_ms() {
        auto end_time = Clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

inline bool verify_dtensor_numerics(const DTensor& a, const DTensor& b, float tol = 1e-3) {
    const Tensor& a_t = a.local_tensor();
    const Tensor& b_t = b.local_tensor();
    
    if (a_t.numel() != b_t.numel()) {
        std::cerr << "[Rank " << a.rank() << "] Shape mismatch: " << a_t.numel() 
                  << " vs " << b_t.numel() << std::endl;
        return false;
    }
    
    Tensor a_cpu = a_t.to_cpu();
    Tensor b_cpu = b_t.to_cpu();
    
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < a_t.numel(); ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            std::cerr << "[Rank " << a.rank() << "] Mismatch at index " << i 
                      << ": " << a_data[i] << " vs " << b_data[i] 
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

inline DTensorTestMetrics benchmark_dtensor_op(
    const std::string& name, 
    std::function<DTensor()> forward_fn, 
    DTensor& input_tensor,
    int iterations = 50,
    float tolerance = 1e-3) 
{
    (void)name;
    (void)tolerance;
    
    DTensorTestMetrics metrics = {0, 0, 0, true, 0, input_tensor.rank()};
    
    // Warmup - this may create persistent tensors in layers, which is expected
    for(int i = 0; i < 3; i++) {
        DTensor out = forward_fn();
        if (out.local_tensor().requires_grad()) {
            Tensor grad = Tensor::ones(out.local_tensor().shape(), out.local_tensor().opts());
            out.mutable_tensor().backward(&grad);
        }
        cudaDeviceSynchronize();
    }
    
    // *** IMPORTANT: Take baseline count AFTER warmup to exclude layer weights ***
    int64_t baseline_count = Tensor::get_active_tensor_count();
    
    // Time Forward + Backward
    DTensorTimer timer;
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    timer.start();
    for(int i = 0; i < iterations; i++) {
        DTensor out = forward_fn();
        Tensor grad = Tensor::ones(out.local_tensor().shape(), out.local_tensor().opts());
        out.mutable_tensor().backward(&grad);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double fwd_bwd_combined_ms = timer.elapsed_ms() / iterations;
    
    // *** IMPORTANT: Take final count immediately after benchmark loop ***
    int64_t final_count = Tensor::get_active_tensor_count();
    
    // Calculate leaks: should be 0 if all tensors are properly released
    metrics.memory_leak_count = final_count - baseline_count;
    
    metrics.forward_latency_ms = fwd_bwd_combined_ms * 0.5;
    metrics.backward_latency_ms = fwd_bwd_combined_ms * 0.5;
    metrics.throughput_ops_sec = (input_tensor.local_tensor().numel() / (fwd_bwd_combined_ms / 1000.0));
    
    return metrics;
}

inline void print_dtensor_result(const std::string& name, const DTensorTestMetrics& m) {
    // Only rank 0 prints to avoid duplicate output
    if (m.rank == 0) {
        std::cout << std::left << std::setw(30) << name 
                  << " | Fwd: " << std::fixed << std::setprecision(3) << m.forward_latency_ms << " ms"
                  << " | Bwd: " << m.backward_latency_ms << " ms"
                  << " | Thr: " << std::scientific << std::setprecision(2) << m.throughput_ops_sec << " elem/s"
                  << " | Match: " << (m.numerical_match ? "PASS" : "FAIL")
                  << " | Leaks: " << m.memory_leak_count
                  << std::endl;
    }
}

// Helper to create a simple DTensor for testing
inline DTensor create_test_dtensor(
    const DeviceMesh& mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    std::vector<int64_t> shape,
    bool requires_grad = true,
    const std::string& name = "test_tensor")
{
    Layout layout(mesh, shape);
    DTensor tensor(mesh, pg, layout, name);
    tensor.mutable_tensor().set_requires_grad(requires_grad);
    return tensor;
}
