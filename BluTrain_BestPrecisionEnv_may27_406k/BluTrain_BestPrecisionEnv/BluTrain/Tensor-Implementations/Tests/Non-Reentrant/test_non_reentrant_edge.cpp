#include "checkpointing/Checkpoint.h"
#include "checkpointing/RNG.h"
#include "autograd/Engine.h"
#include "autograd/Node.h"
#include "autograd/Hooks.h"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/ActivationOps.h"
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << #cond << " at line " << __LINE__ << std::endl; \
        throw std::runtime_error("Test failed"); \
    }

// ------------------------------------------------------------------
// Test 1: In-Place Modification Safeguard
// ------------------------------------------------------------------
variable_list inplace_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor y = autograd::add(x, Tensor::full(x.shape(), x.opts(), 1.0f));
    return {y};
}

void test_inplace_modification_throws() {
    std::cout << "Running test_inplace_modification_throws...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 2.0f);
    variable_list out;
    bool caught = false;
    try {
        out = checkpoint(inplace_fn, {x});
        
        std::cout << "x version before bump: " << x.unsafeGetTensorImpl()->version() << "\n";
        if (x.unsafeGetTensorImpl()) {
            x.unsafeGetTensorImpl()->bump_version(); 
        }
        std::cout << "x version after bump: " << x.unsafeGetTensorImpl()->version() << "\n";

        autograd::backward(autograd::sum(out[0]));
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        if (msg.find("modified by an inplace operation") == std::string::npos) {
            std::cerr << "Caught wrong exception: " << msg << std::endl;
            throw;
        }
    }
    ASSERT_TRUE(caught);
    std::cout << "Pass: Caught inplace modification flawlessly!\n";
}

// ------------------------------------------------------------------
// Test 2: DDP Hook Firing Synchronously
// ------------------------------------------------------------------
variable_list ddp_fn(const variable_list& inputs) {
    Tensor x = inputs[0];
    Tensor scalar2 = Tensor::full(x.shape(), x.opts(), 3.0f);
    return {autograd::mul(x, scalar2)};
}

void test_ddp_hook_firing() {
    std::cout << "Running test_ddp_hook_firing (DDP Simulator)...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 10.0f);
    
    bool hook_fired = false;
    x.register_post_acc_hook(make_post_acc_hook([&hook_fired](const Tensor& grad) {
        hook_fired = true;
        std::cout << "  [DDP Hook Intercept] Gradient ready. Value: " << grad.data<float>()[0] << std::endl;
        std::cout << "  [DDP Hook Intercept] Launching All-Reduce stream...\n";
    }));

    variable_list out = checkpoint(ddp_fn, {x});
    autograd::backward(autograd::sum(out[0]));

    std::cout << "x.has_grad() = " << x.has_grad() << "\n";
    if (x.has_grad()) {
        std::cout << "x.grad()[0] = " << x.grad_view().data<float>()[0] << "\n";
    }

    ASSERT_TRUE(hook_fired);
    ASSERT_TRUE(x.has_grad());
    std::cout << "Pass: Outer DDP Hooks fired exactly inline with local backward pass!\n";
}

// ------------------------------------------------------------------
// Test 3: Complex Nested Reentrancy Stress Test
// ------------------------------------------------------------------
variable_list level3(const variable_list& inputs) {
    return {autograd::mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f))};
}
variable_list level2(const variable_list& inputs) {
    return checkpoint(level3, inputs);
}
variable_list level1(const variable_list& inputs) {
    return checkpoint(level2, inputs);
}

void test_reentrancy_stress() {
    std::cout << "Running test_reentrancy_stress...\n";
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 5.0f);
    
    variable_list out = checkpoint(level1, {x});
    autograd::backward(autograd::sum(out[0]));

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(x.grad_view().data<float>()[0] == 2.0f);
    std::cout << "Pass: Reentrancy stress sequence executed gracefully!\n";
}

int main() {
    try {
        std::cout << "=== Non-Reentrant Edge Case Validations ===\n\n";
        test_inplace_modification_throws();
        test_ddp_hook_firing();
        test_reentrancy_stress();
        std::cout << "\nAll Non-Reentrant Core checks passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Fail: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
