#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include <iostream>

using namespace OwnTensor;

// test_memory_leak.cpp
// Goal: Verify no memory leaks by repeatedly creating and destroying graphs.

void run_iteration() {
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    
    // Create tensors
    Tensor t1 = Tensor::randn<float>(Shape{{10, 10}}, req_grad);
    Tensor t2 = Tensor::randn<float>(Shape{{10, 10}}, req_grad);
    
    // Build graph: v3 = (t1 + t2) * t1
    Tensor v1 = autograd::add(t1, t2);
    Tensor v3 = autograd::mul(v1, t1);
    
    // Backward
    Tensor loss = autograd::mean(v3);
    loss.backward();
}

int main() {
    std::cout << "Starting memory leak test..." << std::endl;
    
    const int iterations = 1000;
    for (int i = 0; i <= iterations; ++i) {
        run_iteration();
        if (i % 100 == 0) {
            std::cout << "Iteration " << i << " completed." << std::endl;
        }
    }
    
    std::cout << "Memory leak test completed. constructing and destructing graphs repeatedly." << std::endl;
    std::cout << "Please run with valgrind or ASAN to verify 0 leaks." << std::endl;
    
    return 0;
}