#include "checkpointing/Checkpoint.h"
#include "checkpointing/RNG.h"
#include "autograd/Engine.h"
#include "autograd/Node.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "ops/UnaryOps/Trigonometry.h" // For sin
#include "ops/UnaryOps/Arithmetics.h" // For abs
#include "ops/UnaryOps/Reduction.h" // For max
#include "autograd/operations/TrigonometryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/BinaryOps.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// Define helper for basic_fn
Tensor my_sin(const Tensor& t) {
    return autograd::sin(t);
}

bool all_close(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.shape() != b.shape()) return false;
    Tensor diff = OwnTensor::abs(a - b);
    return *OwnTensor::reduce_max(diff).data<float>() < tol;
}

// Simple function: x * y + sin(x)
variable_list basic_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor y = inputs[1];
    return {autograd::add(autograd::mul(x, y), my_sin(x))};
}

// Dropout-like function for RNG test
variable_list dropout_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    // CheckpointNode restores the state.
    Tensor mask = Tensor::rand<float>(x.shape(), TensorOptions()); 
    return {autograd::mul(x, mask)};
}


void test_basic_checkpoint() {
    std::cout << "Testing Basic Checkpointing...\n";
    
    Tensor x = Tensor::ones(Shape{{10, 10}}, TensorOptions().with_req_grad(true));
    Tensor y = Tensor::full(Shape{{10, 10}},  TensorOptions().with_req_grad(true), 2.0f);
    
    // 1. Normal execution
    {
        Tensor out = basic_fn({x, y})[0];
        autograd::backward(autograd::sum(out));
    }
    Tensor x_grad_normal = x.grad_view().clone();
    Tensor y_grad_normal = y.grad_view().clone();
    
    x.zero_grad();
    y.zero_grad();
    
    // 2. Checkpointed execution
    {
        variable_list out = checkpoint(basic_fn, {x, y});
        autograd::backward(autograd::sum(out[0]));
    }
    
    assert(all_close(x.grad_view(), x_grad_normal));
    assert(all_close(y.grad_view(), y_grad_normal));
    
    std::cout << "Basic Checkpointing Passed!\n";
}

void test_rng_restore() {
    std::cout << "Testing RNG Restoration...\n";
    
    Tensor x = Tensor::ones(Shape{{10, 10}}, TensorOptions().with_req_grad(true));
    
    // 1. Normal execution
    RNG::set_seed(42);
    Tensor out_normal = dropout_fn({x})[0];
    autograd::backward(autograd::sum(out_normal));
    Tensor grad_normal = x.grad_view().clone();
    
    x.zero_grad();
    
    // 2. Checkpointed execution
    RNG::set_seed(42); // Reset seed to ensure same starting point
    variable_list out_cp = checkpoint(dropout_fn, {x});
    
    // The forward output should be identical because we reset the seed
    assert(all_close(out_normal, out_cp[0]));
    
    // Now the tricky part: CheckpointNode re-runs the forward function.
    // If it doesn't restore RNG, it will generate a DIFFERENT mask during recompute,
    // leading to different gradients.
    autograd::backward(autograd::sum(out_cp[0]));
    
    assert(all_close(x.grad_view(), grad_normal));
    
    std::cout << "RNG Restoration Passed!\n";
}

int main() {
    try {
        test_basic_checkpoint();
        test_rng_restore();
        std::cout << "All tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
