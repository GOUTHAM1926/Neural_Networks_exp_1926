#include "core/Tensor.h"
#include "autograd/operations/ArithmeticsOps.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace OwnTensor;

int main() {
    std::cout << "Testing Autograd Arithmetics..." << std::endl;
    TensorOptions req_grad = TensorOptions().with_req_grad(true);

    // 1. Square: f(x) = x^2, f'(x) = 2x. At x=3, f'(3)=6
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 3.0f);
        Tensor y = autograd::square(x);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "square(3) grad: " << grad << " (expected 6)" << std::endl;
        assert(std::abs(grad - 6.0f) < 1e-5);
    }

    // 2. Sqrt: f(x) = sqrt(x), f'(x) = 1/(2sqrt(x)). At x=4, f'(4)=1/4=0.25
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 4.0f);
        Tensor y = autograd::sqrt(x);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "sqrt(4) grad: " << grad << " (expected 0.25)" << std::endl;
        assert(std::abs(grad - 0.25f) < 1e-5);
    }
    
    // 3. Neg: f(x) = -x, f'(x) = -1
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 10.0f);
        Tensor y = autograd::neg(x);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "neg(10) grad: " << grad << " (expected -1)" << std::endl;
        assert(std::abs(grad + 1.0f) < 1e-5);
    }

    // 4. Abs: f(x) = |x|. At x=-2, f'(-2) = -1.
    {
        Tensor x = Tensor::full(Shape{}, req_grad, -2.0f);
        Tensor y = autograd::abs(x);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "abs(-2) grad: " << grad << " (expected -1)" << std::endl;
        assert(std::abs(grad + 1.0f) < 1e-5);
    }
    
    // 5. Reciprocal: f(x) = 1/x, f'(x) = -1/x^2. At x=2, f'(2) = -0.25
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor y = autograd::reciprocal(x);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "reciprocal(2) grad: " << grad << " (expected -0.25)" << std::endl;
        assert(std::abs(grad + 0.25f) < 1e-5);
    }

    // 6. Pow: f(x) = x^3, f'(x) = 3x^2. At x=2, f'(2) = 12
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor y = autograd::pow(x, 3.0f);
        y.backward();
        float grad = x.grad<float>()[0];
        std::cout << "pow(2, 3) grad: " << grad << " (expected 12)" << std::endl;
        assert(std::abs(grad - 12.0f) < 1e-5);
    }

    std::cout << "All Arithmetics tests passed!" << std::endl;
    return 0;
}
