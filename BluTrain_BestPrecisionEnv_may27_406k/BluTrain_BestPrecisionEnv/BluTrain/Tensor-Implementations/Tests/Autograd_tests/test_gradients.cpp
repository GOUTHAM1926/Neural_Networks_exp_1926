#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

// test_gradients.cpp
// Goal: Verify gradient correctness for f(x) = x^3 + 2x at x = 2.
// f'(x) = 3x^2 + 2. f'(2) = 12 + 2 = 14.

int main() {
    std::cout << "Starting gradient correctness test..." << std::endl;
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    
    // x = 2.0
    Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
    x = autograd::add(x, x); // x = 2.0. (Using ops to set value to keep it simple if explicit set is hard)
    // Actually Tensor::ones creates new tensor. We want a leaf 'x' initialized to 2.
    // Let's create x = 1 then x.data()[0] = 2.0 provided we can access data.
    // test_mlp.cpp uses .data<float>().
    
    Tensor x_leaf = Tensor::ones(Shape{{1, 1}}, req_grad);
    x_leaf.data<float>()[0] = 2.0f;
    
    std::cout << "x = " << x_leaf.data<float>()[0] << std::endl;
    
    // Debugging step-by-step
    
    // 1. Check x^2
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor y = autograd::mul(x, x);
        y.backward();
        std::cout << "d(x^2)/dx at x=2. Expected: 4. Computed: " << x.grad<float>()[0] << std::endl;
    }

    // 2. Check x^3 via reuse
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor x2 = autograd::mul(x, x);
        Tensor y = autograd::mul(x2, x);
        y.backward();
        std::cout << "d(x^3)/dx at x=2. Expected: 12. Computed: " << x.grad<float>()[0] << std::endl;
    }
    
    // 3. Check 2x
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor two = Tensor::ones(Shape{{1, 1}}, TensorOptions()); two.data<float>()[0] = 2.0f;
        Tensor y = autograd::mul(x, two);
        y.backward();
        std::cout << "d(2x)/dx. Expected: 2. Computed: " << x.grad<float>()[0] << std::endl;
    }
    
    // 4. Check Accumulation: x + x
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor y = autograd::add(x, x);
        y.backward();
        std::cout << "d(x+x)/dx. Expected: 2. Computed: " << x.grad<float>()[0] << std::endl;
    }

    // 5. Check Diamond Add: x*x + x
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor x2 = autograd::mul(x, x);
        Tensor y = autograd::add(x2, x);
        y.backward();
        // d(x^2 + x)/dx = 2x + 1 = 5
        std::cout << "d(x^2 + x)/dx at x=2. Expected: 5. Computed: " << x.grad<float>()[0] << std::endl;
    }
    // 6. Matmul: d(x @ W)/dx where x=[[1,2]], W=[[1],[1]]
    // dy/dx = W^T = [[1,1]]
    {
        Tensor x = Tensor::ones(Shape{{1, 2}}, req_grad);
        x.data<float>()[0] = 1.0f;
        x.data<float>()[1] = 2.0f;
        Tensor W = Tensor::ones(Shape{{2, 1}}, TensorOptions());
        Tensor y = autograd::matmul(x, W);
        y.backward();
        std::cout << "d(x@W)/dx. Expected: [1,1]. Computed: [" 
                  << x.grad<float>()[0] << "," << x.grad<float>()[1] << "]" << std::endl;
    }

    // 7. ReLU: d(relu(x))/dx at x=-1 should be 0
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = -1.0f;
        Tensor y = autograd::relu(x);
        Tensor loss = autograd::mean(y);
        loss.backward();
        std::cout << "d(relu(-1))/dx. Expected: 0. Computed: " << x.grad<float>()[0] << std::endl;
    }

    // 8. ReLU at positive: d(relu(x))/dx at x=2 should be 1
    {
        Tensor x = Tensor::ones(Shape{{1, 1}}, req_grad);
        x.data<float>()[0] = 2.0f;
        Tensor y = autograd::relu(x);
        Tensor loss = autograd::mean(y);
        loss.backward();
        std::cout << "d(relu(2))/dx. Expected: 1. Computed: " << x.grad<float>()[0] << std::endl;
    }

    return 0;
}