#include "core/Tensor.h"
#include "autograd/Engine.h"
#include "ops/TensorOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/BinaryBackward.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "device/DeviceCore.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// Helper function to compare tensors
bool tensors_close(const Tensor& a, const Tensor& b, float atol = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    if (a.dtype() != b.dtype()) return false;
    
    // Sync to CPU for comparison if needed
    Tensor a_cpu = a.is_cuda() ? a.to_cpu() : a;
    Tensor b_cpu = b.is_cuda() ? b.to_cpu() : b;
    
    auto a_data = a_cpu.data<float>();
    auto b_data = b_cpu.data<float>();
    
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > atol) {
            std::cout << "Mismatch at index " << i << ": " 
                      << a_data[i] << " vs " << b_data[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Helper wrappers for testing autograd
Tensor add_autograd(const Tensor& a, const Tensor& b) {
    auto fwd = [](const Tensor& x, const Tensor& y) { return x + y; };
    return autograd::make_binary_op<autograd::AddBackward>(a, b, fwd, a.shape(), b.shape());
}



// Test 1: Simple arithmetic operations
void test_simple_arithmetic(DeviceIndex device) {
    std::cout << "\n=== Test 1: Simple Arithmetic (" << (device.is_cuda() ? "GPU" : "CPU") << ") ===" << std::endl;
    
    TensorOptions opts;
    opts.device = device;
    
    // Create input tensors
    auto x = Tensor::ones(Shape{{3, 3}}, opts);
    x.set_requires_grad(true);
    // Use Tensor::full for scalar multiplication since operator* isn't overloaded
    auto y = Tensor::full(Shape{{3, 3}}, opts, 2.0f);
    
    // Forward: z = x + y, loss = sum(z)
    auto z = add_autograd(x, y);
    auto loss = autograd::sum(z);
    
    // Test Sequential Mode
    autograd::set_execution_mode(autograd::ExecutionMode::SEQUENTIAL);
    x.zero_grad();
    loss.backward();
    auto grad_seq = x.grad_view().clone();
    
    std::cout << "Sequential gradient:" << std::endl;
    grad_seq.display();
    
    // Test Parallel Mode
    // Must re-run forward pass as backward pass consumes the graph (releases saved variables)
    autograd::set_execution_mode(autograd::ExecutionMode::PARALLEL);
    x.zero_grad();
    
    // Re-run forward
    auto z_par = add_autograd(x, y);
    auto loss_par = autograd::sum(z_par);
    
    loss_par.backward();
    auto grad_par = x.grad_view().clone();
    
    std::cout << "Parallel gradient:" << std::endl;
    grad_par.display();
    
    // Compare
    bool match = tensors_close(grad_seq, grad_par);
    std::cout << "Gradients match: " << (match ? "PASS" : "FAIL") << std::endl;
}

// Test 2: Matrix multiplication
void test_matmul(DeviceIndex device) {
    std::cout << "\n=== Test 2: Matrix Multiplication (" << (device.is_cuda() ? "GPU" : "CPU") << ") ===" << std::endl;
    
    TensorOptions opts;
    opts.device = device;
    
    auto x = Tensor::randn(Shape{{4, 3}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    auto w = Tensor::randn(Shape{{3, 5}}, opts, 43, 1.0f);
    w.set_requires_grad(true);
    
    auto out = matmul(x, w);
    auto loss = autograd::sum(out);
    
    // Sequential
    autograd::set_execution_mode(autograd::ExecutionMode::SEQUENTIAL);
    x.zero_grad();
    w.zero_grad();
    loss.backward();
    auto x_grad_seq = x.grad_view().clone();
    auto w_grad_seq = w.grad_view().clone();
    
    std::cout << "Sequential x gradient sum: " << autograd::sum(x_grad_seq).to_cpu().data<float>()[0] << std::endl;
    std::cout << "Sequential w gradient sum: " << autograd::sum(w_grad_seq).to_cpu().data<float>()[0] << std::endl;
    
    // Parallel
    // Re-run forward
    autograd::set_execution_mode(autograd::ExecutionMode::PARALLEL);
    x.zero_grad();
    w.zero_grad();
    
    auto out_par = matmul(x, w);
    auto loss_par = autograd::sum(out_par);
    
    loss_par.backward();
    auto x_grad_par = x.grad_view().clone();
    auto w_grad_par = w.grad_view().clone();
    
    std::cout << "Parallel x gradient sum: " << autograd::sum(x_grad_par).to_cpu().data<float>()[0] << std::endl;
    std::cout << "Parallel w gradient sum: " << autograd::sum(w_grad_par).to_cpu().data<float>()[0] << std::endl;
    
    bool x_match = tensors_close(x_grad_seq, x_grad_par);
    bool w_match = tensors_close(w_grad_seq, w_grad_par);
    std::cout << "X gradients match: " << (x_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "W gradients match: " << (w_match ? "PASS" : "FAIL") << std::endl;
}

// Test 3: MLP (Multi-layer Perceptron)
void test_mlp(DeviceIndex device) {
    std::cout << "\n=== Test 3: MLP (" << (device.is_cuda() ? "GPU" : "CPU") << ") ===" << std::endl;
    
    TensorOptions opts;
    opts.device = device;
    
    // Input
    auto x = Tensor::randn(Shape{{2, 4}}, opts, 42, 1.0f);
    x.set_requires_grad(true);
    
    // Weights
    auto w1 = Tensor::randn(Shape{{4, 8}}, opts, 43, 1.0f);
    w1.set_requires_grad(true);
    auto b1 = Tensor::randn(Shape{{8}}, opts, 44, 1.0f);
    b1.set_requires_grad(true);
    auto w2 = Tensor::randn(Shape{{8, 1}}, opts, 45, 1.0f);
    w2.set_requires_grad(true);
    auto b2 = Tensor::randn(Shape{{1}}, opts, 46, 1.0f);
    b2.set_requires_grad(true);
    
    // Forward
    auto h1 = add_autograd(matmul(x, w1), b1);
    auto h1_relu = relu(h1);
    auto out = add_autograd(matmul(h1_relu, w2), b2);
    auto loss = autograd::sum(out);
    
    // Sequential
    autograd::set_execution_mode(autograd::ExecutionMode::SEQUENTIAL);
    x.zero_grad();
    w1.zero_grad();
    b1.zero_grad();
    w2.zero_grad();
    b2.zero_grad();
    loss.backward();
    
    auto x_grad_seq = x.grad_view().clone();
    auto w1_grad_seq = w1.grad_view().clone();
    auto b1_grad_seq = b1.grad_view().clone();
    auto w2_grad_seq = w2.grad_view().clone();
    auto b2_grad_seq = b2.grad_view().clone();
    
    std::cout << "Sequential gradients - x: " << autograd::sum(x_grad_seq).to_cpu().data<float>()[0] 
              << ", w1: " << autograd::sum(w1_grad_seq).to_cpu().data<float>()[0] 
              << ", b1: " << autograd::sum(b1_grad_seq).to_cpu().data<float>()[0]
              << ", w2: " << autograd::sum(w2_grad_seq).to_cpu().data<float>()[0]
              << ", b2: " << autograd::sum(b2_grad_seq).to_cpu().data<float>()[0] << std::endl;
    
    // Parallel
    autograd::set_execution_mode(autograd::ExecutionMode::PARALLEL);
    x.zero_grad();
    w1.zero_grad();
    b1.zero_grad();
    w2.zero_grad();
    b2.zero_grad();
    
    // Re-run forward
    auto h1_par = add_autograd(matmul(x, w1), b1);
    auto h1_relu_par = relu(h1_par);
    auto out_par = add_autograd(matmul(h1_relu_par, w2), b2);
    auto loss_par = autograd::sum(out_par);
    
    loss_par.backward();
    
    auto x_grad_par = x.grad_view().clone();
    auto w1_grad_par = w1.grad_view().clone();
    auto b1_grad_par = b1.grad_view().clone();
    auto w2_grad_par = w2.grad_view().clone();
    auto b2_grad_par = b2.grad_view().clone();
    
    std::cout << "Parallel gradients - x: " << autograd::sum(x_grad_par).to_cpu().data<float>()[0] 
              << ", w1: " << autograd::sum(w1_grad_par).to_cpu().data<float>()[0] 
              << ", b1: " << autograd::sum(b1_grad_par).to_cpu().data<float>()[0]
              << ", w2: " << autograd::sum(w2_grad_par).to_cpu().data<float>()[0]
              << ", b2: " << autograd::sum(b2_grad_par).to_cpu().data<float>()[0] << std::endl;
    
    bool matches = tensors_close(x_grad_seq, x_grad_par) &&
                   tensors_close(w1_grad_seq, w1_grad_par) &&
                   tensors_close(b1_grad_seq, b1_grad_par) &&
                   tensors_close(w2_grad_seq, w2_grad_par) &&
                   tensors_close(b2_grad_seq, b2_grad_par);
    
    std::cout << "All gradients match: " << (matches ? "PASS" : "FAIL") << std::endl;
}

// Test 4: Default mode test (should be SEQUENTIAL)
void test_default_mode() {
    std::cout << "\n=== Test 4: Default Mode ===" << std::endl;
    
    // Don't set mode explicitly - should default to SEQUENTIAL
    auto mode = autograd::get_execution_mode();
    std::cout << "Default mode is: " 
              << (mode == autograd::ExecutionMode::SEQUENTIAL ? "SEQUENTIAL" : "PARALLEL")
              << std::endl;
    
    bool pass = (mode == autograd::ExecutionMode::SEQUENTIAL);
    std::cout << "Default mode test: " << (pass ? "PASS" : "FAIL") << std::endl;
}

// Test 5: Mode switching
void test_mode_switching() {
    std::cout << "\n=== Test 5: Mode Switching ===" << std::endl;
    
    autograd::set_execution_mode(autograd::ExecutionMode::SEQUENTIAL);
    auto mode1 = autograd::get_execution_mode();
    std::cout << "After setting SEQUENTIAL: " 
              << (mode1 == autograd::ExecutionMode::SEQUENTIAL ? "SEQUENTIAL" : "PARALLEL")
              << std::endl;
    
    autograd::set_execution_mode(autograd::ExecutionMode::PARALLEL);
    auto mode2 = autograd::get_execution_mode();
    std::cout << "After setting PARALLEL: " 
              << (mode2 == autograd::ExecutionMode::PARALLEL ? "PARALLEL" : "SEQUENTIAL")
              << std::endl;
    
    autograd::set_execution_mode(autograd::ExecutionMode::SEQUENTIAL);
    auto mode3 = autograd::get_execution_mode();
    std::cout << "After setting SEQUENTIAL again: " 
              << (mode3 == autograd::ExecutionMode::SEQUENTIAL ? "SEQUENTIAL" : "PARALLEL")
              << std::endl;
    
    bool pass = (mode1 == autograd::ExecutionMode::SEQUENTIAL) &&
                (mode2 == autograd::ExecutionMode::PARALLEL) &&
                (mode3 == autograd::ExecutionMode::SEQUENTIAL);
    
    std::cout << "Mode switching test: " << (pass ? "PASS" : "FAIL") << std::endl;
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "  Execution Mode Test Suite" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        test_default_mode();
        test_mode_switching();
        
        std::vector<DeviceIndex> devices;
        devices.push_back(DeviceIndex(Device::CPU));
        
        if (OwnTensor::device::cuda_available()) {
            std::cout << "CUDA is available! Adding GPU tests." << std::endl;
            devices.push_back(DeviceIndex(Device::CUDA, 0));
        } else {
            std::cout << "CUDA is not available. Skipping GPU tests." << std::endl;
        }
        
        for (const auto& dev : devices) {
            test_simple_arithmetic(dev);
            test_matmul(dev);
            test_mlp(dev);
        }
        
        std::cout << "\n====================================" << std::endl;
        std::cout << "  All tests completed!" << std::endl;
        std::cout << "====================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
