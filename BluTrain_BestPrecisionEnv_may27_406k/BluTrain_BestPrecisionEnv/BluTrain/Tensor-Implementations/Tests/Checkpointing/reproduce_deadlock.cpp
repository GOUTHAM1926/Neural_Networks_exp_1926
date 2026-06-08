#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// A simple function that sleeps during recomputation
variable_list heavy_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    // This is run during recompute in apply()
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return {autograd::mul(x, Tensor::full(x.shape(), x.opts(), 1.01f))};
}

void test_parallel_deadlock() {
    std::cout << "Testing Parallel Deadlock (Reproduction)...\n";
    
    set_execution_mode(ExecutionMode::PARALLEL);
    
    int num_threads = std::thread::hardware_concurrency();
    std::cout << "Threads in pool: " << num_threads << "\n";
    
    int num_checkpoints = num_threads + 2;
    std::cout << "Creating " << num_checkpoints << " checkpoint nodes...\n";
    
    std::vector<Tensor> inputs;
    for (int i = 0; i < num_checkpoints; ++i) {
        inputs.push_back(Tensor::ones(Shape{{1}}, TensorOptions().with_req_grad(true)));
    }
    
    std::vector<Tensor> outputs;
    for (int i = 0; i < num_checkpoints; ++i) {
        outputs.push_back(checkpoint(heavy_fn, {inputs[i]})[0]);
    }
    
    Tensor loss = outputs[0];
    for (size_t i = 1; i < outputs.size(); ++i) {
        loss = autograd::add(loss, outputs[i]);
    }
    
    std::cout << "Starting backward pass. This SHOULD deadlock.\n";
    
    auto start = std::chrono::steady_clock::now();
    bool finished = false;
    std::thread t([&]() {
        try {
            autograd::backward(loss);
            finished = true;
        } catch (...) {
            std::cout << "Caught exception in backward\n";
        }
    });
    
    // Wait for 5 seconds
    for (int i = 0; i < 50; ++i) {
        if (finished) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (!finished) {
        std::cout << "The bug is reproduced: Nested engine calls have saturated the thread pool.\n";
        exit(0); // Exit success because we proved the bug
    } else {
        std::cout << "Backward finished WITHOUT deadlock. This is unexpected for the bug repro.\n";
    }
    
    if (t.joinable()) t.join();
}

int main() {
    test_parallel_deadlock();
    return 0;
}
