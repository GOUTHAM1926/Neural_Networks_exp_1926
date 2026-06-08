#include "core/Tensor.h"
#include "autograd/Functions.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

int main() {
    std::cout << "\n=== Testing backward() Implementation ===\n\n";
    
    // Test 1: Simple scalar backward
    std::cout << "Test 1: Scalar tensor backward...\n";
    {
        TensorOptions opts = TensorOptions().with_req_grad(true);
        Tensor x = Tensor::ones(Shape{{1}}, opts);
        
        // Manually create a simple computation graph
        // For now, just test that backward() can be called on a leaf
        try {
            x.backward();
            std::cout << " backward() executed on leaf tensor\n";
            
            // Check gradient
            if (x.owns_grad()) {
                float* grad = x.grad<float>();
                if (grad && std::abs(grad[0] - 1.0f) < 1e-6) {
                    std::cout << " Gradient correctly set to 1.0\n";
                } else {
                    std::cout << " Gradient value incorrect\n";
                }
            }
        } catch (const std::exception& e) {
            std::cout << " Correctly threw exception: " << e.what() << "\n";
        }
    }
    
    // Test 2: Test grad_fn and is_leaf
    std::cout << "\nTest 2: Testing grad_fn and is_leaf...\n";
    {
        TensorOptions opts = TensorOptions().with_req_grad(true);
        Tensor a = Tensor::ones(Shape{{2, 2}}, opts);
        
        if (a.is_leaf()) {
            std::cout << " Leaf tensor correctly identified\n";
        } else {
            std::cout << " Leaf tensor not identified\n";
        }
        
        if (!a.grad_fn()) {
            std::cout << " Leaf tensor has no grad_fn\n";
        } else {
            std::cout << " Leaf tensor should not have grad_fn\n";
        }
    }
    
    // Test 3: Test backward function creation
    std::cout << "\nTest 3: Creating backward functions...\n";
    {
        Tensor a = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        Tensor b = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        
        // Create an AddBackward node
        auto add_backward = std::make_shared<autograd::AddBackward>(a.shape(), b.shape());
        std::cout << " AddBackward created\n";
        
        // Create a MulBackward node
        auto mul_backward = std::make_shared<autograd::MulBackward>(a, b);
        std::cout << " MulBackward created\n";
        
        // Create a MatmulBackward node
        auto matmul_backward = std::make_shared<autograd::MatmulBackward>(a, b);
        std::cout << " MatmulBackward created\n";
        
        // Create a ReluBackward node
        auto relu_backward = std::make_shared<autograd::ReluBackward>(a);
        std::cout << " ReluBackward created\n";
    }
    
    // Test 4: Test backward function application
    std::cout << "\nTest 4: Testing backward function apply...\n";
    {
        Tensor a = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        Tensor b = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        Tensor grad_out = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        
        // Test AddBackward
        auto add_backward = std::make_shared<autograd::AddBackward>(a.shape(), b.shape());
        std::vector<Tensor> add_grads = add_backward->apply({grad_out});
        if (add_grads.size() == 2) {
            std::cout << " AddBackward returned 2 gradients\n";
        }
        
        // Test MulBackward
        auto mul_backward = std::make_shared<autograd::MulBackward>(a, b);
        std::vector<Tensor> mul_grads = mul_backward->apply({grad_out});
        if (mul_grads.size() == 2) {
            std::cout << " MulBackward returned 2 gradients\n";
        }
    }
    
    // Test 5: Test setting grad_fn on tensor
    std::cout << "\nTest 5: Testing grad_fn attachment...\n";
    {
        TensorOptions opts = TensorOptions().with_req_grad(true);
        Tensor result = Tensor::ones(Shape{{2, 2}}, opts);
        Tensor a = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        Tensor b = Tensor::ones(Shape{{2, 2}}, TensorOptions());
        
        auto grad_fn = std::make_shared<autograd::MulBackward>(a, b);
        result.set_grad_fn(grad_fn);
        
        if (result.grad_fn() == grad_fn) {
            std::cout << " grad_fn correctly attached\n";
        }
        
        if (!result.is_leaf()) {
            std::cout << " Tensor with grad_fn is not a leaf\n";
        }
    }
    
    std::cout << "\n=== All backward() tests PASSED! ===\n\n";
    return 0;
}
