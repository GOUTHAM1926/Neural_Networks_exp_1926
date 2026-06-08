#include <iostream>
#include <cassert>
#include <vector>
#include "core/Tensor.h"
#include "autograd/Variable.h"
#include "autograd/Hooks.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/Node.h"

using namespace OwnTensor;

/**
 * ## HOOKS USAGE GUIDE
 * 
 * 1. Post-Accumulation Hooks (Tensor level):
 *    - Attached to leaf tensors (parameters).
 *    - Triggered AFTER the gradient is fully accumulated.
 *    - Suitable for DDP synchronization (all-reduce).
 *    - Example: `x.register_post_acc_hook(make_post_acc_hook([](const Tensor& g){...}))`
 * 
 * 2. Node-Level Hooks (Graph level):
 *    - Attached to Backward nodes (grad_fn).
 *    - `register_pre_hook`: Can modify input gradients before the node fires.
 *    - `register_post_hook` / `register_hook`: Read-only access after node fires.
 *    - Example: `y.grad_fn()->register_hook([](const variable_list& in, const variable_list& out){...})`
 */

int post_acc_called = 0;
bool node_hook_called = false;

void test_advanced_hooks() {
    std::cout << "Testing Advanced Hooks Implementation..." << std::endl;
    
    // 1. Setup shared parameter
    Tensor x = Tensor::ones(Shape{{1}}, TensorOptions().with_req_grad(true));
    
    // Register two post-acc hooks
    x.register_post_acc_hook(std::make_unique<LambdaPostAccHook>([](const Tensor& grad) {
        std::cout << "[PostAcc] Hook 1 fired. Grad value: " << grad.data<float>()[0] << std::endl;
        post_acc_called++;
    }));
    
    x.register_post_acc_hook(std::make_unique<LambdaPostAccHook>([](const Tensor& grad) {
        std::cout << "[PostAcc] Hook 2 fired." << std::endl;
        post_acc_called++;
    }));
    
    // 2. Build graph y = x * x + x
    // x is used twice, should accumulate 2*x + 1
    Tensor x_sq = autograd::mul(x, x);
    
    // Register node-level hook on mul_node
    if (auto mul_fn = x_sq.grad_fn()) {
        mul_fn->register_hook([](const variable_list& in, const variable_list& out) {
            std::cout << "[NodeHook] MulBackward executed." << std::endl;
            node_hook_called = true;
        });
    }
    
    Tensor y = autograd::add(x_sq, x);
    Tensor loss = autograd::sum(y);
    
    // 3. Backward
    std::cout << "Starting backward..." << std::endl;
    loss.backward();
    
    // 4. Verify
    // x = 1.0, grad = 2*1 + 1 = 3.0
    float expected_grad = 3.0f;
    float actual_grad = x.grad_view().data<float>()[0];
    
    std::cout << "Actual Grad: " << actual_grad << std::endl;
    assert(std::abs(actual_grad - expected_grad) < 1e-5);
    
    // Check post_acc_hooks: Should be called exactly once per hook (Total 2)
    // even though x was used twice.
    if (post_acc_called == 2) {
        std::cout << "SUCCESS: Post-accumulation hooks fired correctly (exactly once per hook)." << std::endl;
    } else {
        std::cout << "FAILURE: Post-accumulation hooks fired " << post_acc_called << " times!" << std::endl;
        exit(1);
    }
    
    if (node_hook_called) {
        std::cout << "SUCCESS: Node-level hook fired correctly." << std::endl;
    } else {
        std::cout << "FAILURE: Node-level hook did NOT fire!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        test_advanced_hooks();
        std::cout << "\nAll Hook tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
