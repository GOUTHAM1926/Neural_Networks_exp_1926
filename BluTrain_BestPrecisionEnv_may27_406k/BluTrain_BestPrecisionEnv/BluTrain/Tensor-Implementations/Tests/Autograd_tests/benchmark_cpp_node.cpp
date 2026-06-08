#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "autograd/CppNode.h"
#include "autograd/AutogradContext.h"
#include "autograd/AutogradOps.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// CppNode implementation of Square
class CppSquare : public CppNode<CppSquare> {
public:
    static variable_list forward(AutogradContext* ctx, const Tensor& x) {
        ctx->save_for_backward({x});
        return {autograd::mul(x, x)};
    }
    static variable_list backward(AutogradContext* ctx, const variable_list& grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor two = Tensor::full(saved[0].shape(), saved[0].opts(), 2.0f);
        return {autograd::mul(autograd::mul(two, saved[0]), grad_outputs[0])};
    }
};

void run_benchmark(int size, int iterations) {
    if (!OwnTensor::device::cuda_available()) {
         std::cout << "CUDA not available, running on CPU..." << std::endl;
    }
    DeviceIndex device = OwnTensor::device::cuda_available() ? DeviceIndex(Device::CUDA) : DeviceIndex(Device::CPU);

    std::cout << "\nBenchmark: Native Square vs CppNode Square" << std::endl;
    std::cout << "Workload: " << size << "x" << size << " Tensor, " << iterations << " iterations" << std::endl;

    Tensor x = Tensor::randn<float>(Shape{{size, size}}, TensorOptions().with_device(device).with_req_grad(true));
    Timer timer;

    // -------------------------------------------------------------------------
    // Native
    // -------------------------------------------------------------------------
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        Tensor y = autograd::square(x);
        Tensor grad = Tensor::ones(y.shape(), y.opts());
        y.backward(&grad);
    }
    cudaDeviceSynchronize();
    double native_time = timer.stop();

    // -------------------------------------------------------------------------
    // CppNode
    // -------------------------------------------------------------------------
    cudaDeviceSynchronize();
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        Tensor y = CppSquare::apply(x)[0];
        Tensor grad = Tensor::ones(y.shape(), y.opts());
        y.backward(&grad);
    }
    cudaDeviceSynchronize();
    double cppnode_time = timer.stop();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Results:" << std::endl;
    std::cout << "  Native Avg:  " << native_time / iterations << " ms" << std::endl;
    std::cout << "  CppNode Avg: " << cppnode_time / iterations << " ms" << std::endl;
    std::cout << "  Overhead:    " << (cppnode_time - native_time) / native_time * 100.0 << " %" << std::endl;
}

int main() {
    int size = 1024;
    int iters = 100;
    
    run_benchmark(size, iters);
    
    return 0;
}
