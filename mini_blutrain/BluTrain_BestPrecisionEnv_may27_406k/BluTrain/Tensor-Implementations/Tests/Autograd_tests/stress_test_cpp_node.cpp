#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cassert>
#include <cmath>
#include "autograd/CppNode.h"
#include "autograd/AutogradContext.h"
#include "autograd/AutogradOps.h"
#include "core/Tensor.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

class StressNode : public CppNode<StressNode> {
public:
    static variable_list forward(AutogradContext* ctx, const Tensor& x) {
        ctx->save_for_backward({x});
        // result = x * 2
        return {autograd::mul(x, Tensor::full(x.shape(), x.opts(), 2.0f))};
    }
    
    static variable_list backward(AutogradContext* ctx, const variable_list& grad_outputs) {
        auto saved = ctx->get_saved_variables();
        // grad_input = grad_output * 2
        return {autograd::mul(grad_outputs[0], Tensor::full(saved[0].shape(), saved[0].opts(), 2.0f))};
    }
};

void test_race_conditions() {
    std::cout << "Testing Concurrent CppNode Applications..." << std::endl;
    
    std::atomic<int> errors{0};
    const int NUM_THREADS = 16;
    const int ITERATIONS = 1000;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            for (int k = 0; k < ITERATIONS; ++k) {
                // Each thread applies the node on its own tensors
                Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), static_cast<float>(i));
                auto results = StressNode::apply(x);
                Tensor y = results[0];
                
                if (std::abs(y.data<float>()[0] - static_cast<float>(i) * 2.0f) > 1e-4) {
                    errors++;
                    break;
                }
                
                y.backward();
                if (std::abs(x.grad<float>()[0] - 2.0f) > 1e-4) {
                    errors++;
                    break;
                }
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    if (errors > 0) {
        std::cerr << "FAIL: Parallel CppNode applications failed with " << errors << " errors" << std::endl;
        exit(1);
    }
    std::cout << "PASS: Concurrent CppNode (" << NUM_THREADS << " threads, " << ITERATIONS << " iters)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running CppNode Stress Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_race_conditions();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All CppNode stress tests PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
