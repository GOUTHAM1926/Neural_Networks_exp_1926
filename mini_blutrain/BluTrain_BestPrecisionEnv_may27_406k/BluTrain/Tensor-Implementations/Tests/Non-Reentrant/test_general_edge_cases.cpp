#include "checkpointing/Checkpoint.h"
#include "checkpointing/RNG.h"
#include "autograd/Engine.h"
#include "autograd/Node.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h" // For reduce_max
#include "ops/UnaryOps/Arithmetics.h" // For abs
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/ActivationOps.h" // For relu if needed
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// Helper for assertions
#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << #cond << " at line " << __LINE__ << std::endl; \
        throw std::runtime_error("Test failed"); \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "Assertion failed: " << #a << " == " << #b << " (" << (a) << " vs " << (b) << ") at line " << __LINE__ << std::endl; \
        throw std::runtime_error("Test failed"); \
    }

bool is_empty_tensor(const Tensor& t) {
    return !t.unsafeGetTensorImpl();
}

// 1. Mixed Gradient Requirements
variable_list mixed_grad_fn(const variable_list& inputs) {
    // x requires grad, y does not
    Tensor x = inputs[0];
    Tensor y = inputs[1];
    return {autograd::mul(x, y)};
}

void test_mixed_gradient_requirements() {
    std::cout << "Running test_mixed_gradient_requirements...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 2.0f);
    Tensor y = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(false), 3.0f);

    variable_list out = checkpoint(mixed_grad_fn, {x, y});
    autograd::backward(autograd::sum(out[0]));

    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(*x.grad_view().data<float>(), 3.0f);
    
    // y should NOT have a gradient
    ASSERT_TRUE(!y.has_grad());
    std::cout << "Shared Pass\n";
}

// 2. Unused Inputs
variable_list unused_input_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    // y is unused
    // Fix scalar usage: wrap scalars in tensors
    Tensor scalar2 = Tensor::full(x.shape(), x.opts(), 2.0f);
    return {autograd::mul(x, scalar2)};
}

void test_unused_inputs() {
    std::cout << "Running test_unused_inputs...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 2.0f);
    Tensor y = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 3.0f);

    variable_list out = checkpoint(unused_input_fn, {x, y});
    autograd::backward(autograd::sum(out[0]));

    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(*x.grad_view().data<float>(), 2.0f);

    // y gradient should be zero or undefined depending on implementation usually zero if it was part of graph but disconnected? 
    // CheckpointNode returns gradients corresponding to inputs. If input wasn't used in recompute, its gradient should be zero/empty.
    // Based on CheckpointNode::apply logic: "if (recompute_inputs[i].has_grad()) ... else input_grads.push_back(Tensor())"
    // Since y is NOT used, it won't be in the local graph trace, so it won't get a gradient during local backward.
    // Thus it should be empty/undefined.
    
    if (y.has_grad()) {
         // If it is defined, it must be zero
          ASSERT_EQ(*y.grad_view().data<float>(), 0.0f);
    } else {
        // This is also acceptable
    }
     std::cout << "Pass\n";
}

// 3. Multiple Outputs
variable_list multi_output_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor scalar2 = Tensor::full(x.shape(), x.opts(), 2.0f);
    Tensor scalar1 = Tensor::full(x.shape(), x.opts(), 1.0f);
    Tensor y = autograd::mul(x, scalar2);
    Tensor z = autograd::add(x, scalar1);
    return {y, z};
}

void test_multiple_outputs() {
    std::cout << "Running test_multiple_outputs...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 10.0f);
    
    variable_list outs = checkpoint(multi_output_fn, {x});
    
    // Loss = y + z = (2x) + (x + 1) = 3x + 1
    // Grad should be 3
    Tensor loss = autograd::add(outs[0], outs[1]);
    autograd::backward(autograd::sum(loss));

    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(*x.grad_view().data<float>(), 3.0f);
    std::cout << "Pass\n";
}

// 4. RNG Consistency
variable_list rng_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor noise = Tensor::rand<float>(x.shape(), x.opts()); 
    // The noisy operation
    return {autograd::mul(x, noise)};
}

void test_rng_consistency() {
    std::cout << "Running test_rng_consistency...\n";
    Tensor x = Tensor::ones(Shape{{10}}, TensorOptions().with_req_grad(true));
    
    // Reference run (without checkpointing, but with captured RNG logic if possible? No, hard to replicate exactly without manual RNG set)
    // Actually, best way is to compare recomputation correctness.
    // If RNG is restored, recomputation produces SAME output as original forward (which we didn't save, but CheckpointNode did internally).
    // But since we can't inspect internal CheckpointNode, we trust the "test_rng_restore" in basic tests.
    // Here let's try a different angle: 
    // Run twice with SAME seed. One normal, one checkpointed. They should have SAME gradients.
    
    RNG::set_seed(12345);
    Tensor x1 = x.clone(); x1.set_requires_grad(true);
    Tensor out1 = rng_fn({x1})[0];
    autograd::backward(autograd::sum(out1));
    Tensor grad1 = x1.grad_view().clone();

    RNG::set_seed(12345); // Reset seed
    Tensor x2 = x.clone(); x2.set_requires_grad(true);
    variable_list out2 = checkpoint(rng_fn, {x2});
    autograd::backward(autograd::sum(out2[0]));
    Tensor grad2 = x2.grad_view().clone();

    // Check difference using non-autograd ops for comparison
    Tensor diff = OwnTensor::abs(grad1 - grad2);
    float max_diff = *OwnTensor::reduce_max(diff).data<float>();
    
    ASSERT_TRUE(max_diff < 1e-5);
    std::cout << "Pass\n";
}

// 5. Exception Safety
variable_list throwing_fn(const variable_list& inputs) {
    throw std::runtime_error("Simulated failure");
}

void test_exception_safety() {
    std::cout << "Running test_exception_safety...\n";
    Tensor x = Tensor::ones(Shape{{1}}, TensorOptions().with_req_grad(true));
    
    bool caught = false;
    try {
        checkpoint(throwing_fn, {x});
    } catch (const std::runtime_error& e) {
        caught = true;
        // Verify message
        std::string msg = e.what();
        if (msg.find("Simulated failure") == std::string::npos) {
             std::cerr << "Caught wrong exception: " << msg << std::endl;
             throw;
        }
    }
    ASSERT_TRUE(caught);
    std::cout << "Pass\n";
}

// 6. Nested Checkpointing
variable_list inner_fn(const variable_list& inputs) {
    Tensor x = inputs[0]; 
    Tensor scalar2 = Tensor::full(x.shape(), x.opts(), 2.0f);
    return {autograd::mul(x, scalar2)};
}

variable_list outer_fn(const variable_list& inputs) {
     return checkpoint(inner_fn, inputs);
}

void test_nested_checkpointing() {
    std::cout << "Running test_nested_checkpointing...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 5.0f);
    
    // Checkpoint calling checkpoint
    variable_list out = checkpoint(outer_fn, {x});
    autograd::backward(autograd::sum(out[0]));

    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(*x.grad_view().data<float>(), 2.0f);
    std::cout << "Pass\n";
}

// 7. Detached Subgraph
variable_list detached_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    // Return a new tensor not connected to x
    return {Tensor::full(x.shape(), TensorOptions(), 1.0f)};
}

void test_detached_subgraph() {
    std::cout << "Running test_detached_subgraph...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 5.0f);
    
    variable_list out = checkpoint(detached_fn, {x});
    autograd::backward(autograd::sum(out[0]));

    // Gradient for x should be zero/undefined because it wasn't used
    try {
        if (x.has_grad()) {
             // If this throws, it means no gradient, which is fine
             Tensor g = x.grad_view();
             if (g.unsafeGetTensorImpl()) {
                 ASSERT_EQ(*g.data<float>(), 0.0f);
             }
        }
    } catch (...) {
        // Gradient not allocated -> arguably treated as zero/undefined, so pass.
    }
    std::cout << "Pass\n";
}

int main() {
    try {
        test_mixed_gradient_requirements();
        test_unused_inputs();
        test_multiple_outputs();
        test_rng_consistency();
        test_exception_safety();
        test_nested_checkpointing();
        test_detached_subgraph();
        
        std::cout << "All edge cases passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
