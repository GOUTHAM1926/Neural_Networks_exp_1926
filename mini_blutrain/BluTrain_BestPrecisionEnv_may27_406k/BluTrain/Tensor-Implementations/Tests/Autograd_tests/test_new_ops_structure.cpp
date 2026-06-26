#include "core/Tensor.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ArithmeticsOps.h"
#include "autograd/operations/TrigonometryOps.h"
#include "autograd/operations/ExponentsOps.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>

using namespace OwnTensor;

/**
 * @brief Helper function to check gradients
 */
void check_grad(float computed, float expected, const std::string& name) {
    if (std::abs(computed - expected) < 1e-4) {
        std::cout << "[PASS] " << name << ": " << computed << std::endl;
    } else {
        std::cout << "[FAIL] " << name << ": Computed=" << computed << ", Expected=" << expected << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing New Autograd Operations Structure" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);

    // ============================================================
    // 1. Binary Operations
    // ============================================================
    std::cout << "\n[1] Binary Operations" << std::endl;
    
    // Test Add
    {
        Tensor a = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor b = Tensor::full(Shape{}, req_grad, 3.0f);
        Tensor c = autograd::add(a, b); // c = a + b = 5
        c.backward();
        check_grad(a.grad<float>()[0], 1.0f, "d(a+b)/da");
        check_grad(b.grad<float>()[0], 1.0f, "d(a+b)/db");
    }

    // Test Sub
    {
        Tensor a = Tensor::full(Shape{}, req_grad, 5.0f);
        Tensor b = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor c = autograd::sub(a, b); // c = a - b = 3
        c.backward();
        check_grad(a.grad<float>()[0], 1.0f, "d(a-b)/da");
        check_grad(b.grad<float>()[0], -1.0f, "d(a-b)/db");
    }

    // Test Mul
    {
        Tensor a = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor b = Tensor::full(Shape{}, req_grad, 4.0f);
        Tensor c = autograd::mul(a, b); // c = a * b = 8
        c.backward();
        check_grad(a.grad<float>()[0], 4.0f, "d(a*b)/da (=b)");
        check_grad(b.grad<float>()[0], 2.0f, "d(a*b)/db (=a)");
    }
    
    // Test Div
    {
        Tensor a = Tensor::full(Shape{}, req_grad, 6.0f);
        Tensor b = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor c = autograd::div(a, b); // c = a / b = 3
        c.backward();
        // d(a/b)/da = 1/b = 0.5
        // d(a/b)/db = -a/b^2 = -6/4 = -1.5
        check_grad(a.grad<float>()[0], 0.5f, "d(a/b)/da");
        check_grad(b.grad<float>()[0], -1.5f, "d(a/b)/db");
    }

    // ============================================================
    // 2. Arithmetic Operations
    // ============================================================
    std::cout << "\n[2] Arithmetic Operations" << std::endl;

    // Test Square
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 3.0f);
        Tensor y = autograd::square(x); // y = x^2 = 9
        y.backward();
        check_grad(x.grad<float>()[0], 6.0f, "d(x^2)/dx (=2x)");
    }
    
    // Test Sqrt
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 4.0f);
        Tensor y = autograd::sqrt(x); // y = 2
        y.backward();
        // dy/dx = 1/(2sqrt(x)) = 1/4 = 0.25
        check_grad(x.grad<float>()[0], 0.25f, "d(sqrt(x))/dx");
    }
    
    // Test Neg
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 10.0f);
        Tensor y = autograd::neg(x);
        y.backward();
        check_grad(x.grad<float>()[0], -1.0f, "d(-x)/dx");
    }
    
    // Test Abs
    {
        Tensor x = Tensor::full(Shape{}, req_grad, -5.0f);
        Tensor y = autograd::abs(x);
        y.backward();
        // dy/dx = sign(x) = -1
        check_grad(x.grad<float>()[0], -1.0f, "d(|x|)/dx (at x=-5)");
    }
    
    // Test Reciprocal
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor y = autograd::reciprocal(x); // y = 1/x = 0.5
        y.backward();
        // dy/dx = -1/x^2 = -1/4 = -0.25
        check_grad(x.grad<float>()[0], -0.25f, "d(1/x)/dx");
    }
    
    // Test Pow
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 3.0f);
        Tensor y = autograd::pow(x, 3.0f); // y = 27
        y.backward();
        // dy/dx = 3x^2 = 3*9 = 27
        check_grad(x.grad<float>()[0], 27.0f, "d(x^3)/dx");
    }

    // ============================================================
    // 3. Trigonometry Operations
    // ============================================================
    std::cout << "\n[3] Trigonometry Operations" << std::endl;
    
    // Test Sin
    {
        float val = M_PI / 2.0f; // 90 deg
        Tensor x = Tensor::full(Shape{}, req_grad, val);
        Tensor y = autograd::sin(x); // sin(pi/2) = 1
        y.backward();
        // dy/dx = cos(x) = cos(pi/2) = 0
        check_grad(x.grad<float>()[0], std::cos(val), "d(sin)/dx");
    }

    // Test Cos
    {
        float val = 0.0f;
        Tensor x = Tensor::full(Shape{}, req_grad, val);
        Tensor y = autograd::cos(x); // cos(0) = 1
        y.backward();
        // dy/dx = -sin(x) = 0
        check_grad(x.grad<float>()[0], -std::sin(val), "d(cos)/dx");
    }
    
    // Test Tan
    {
        float val = M_PI / 4.0f; // 45 deg
        Tensor x = Tensor::full(Shape{}, req_grad, val);
        Tensor y = autograd::tan(x); // tan(45) = 1
        y.backward();
        // dy/dx = 1/cos^2(x) = 1/cos^2(45) = 1/(0.5) = 2
        float expected = 1.0f / (std::cos(val) * std::cos(val));
        check_grad(x.grad<float>()[0], expected, "d(tan)/dx");
    }
    
    // Test Tanh
    {
        float val = 0.0f;
        Tensor x = Tensor::full(Shape{}, req_grad, val);
        Tensor y = autograd::tanh(x); // tanh(0) = 0
        y.backward();
        // dy/dx = 1 - tanh^2(x) = 1 - 0 = 1
        check_grad(x.grad<float>()[0], 1.0f, "d(tanh)/dx");
    }

    // ============================================================
    // 4. Exponent Operations
    // ============================================================
    std::cout << "\n[4] Exponent Operations" << std::endl;
    
    // Test Exp
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 1.0f);
        Tensor y = autograd::exp(x); // exp(1) = 2.718...
        y.backward();
        // dy/dx = exp(x) = e
        check_grad(x.grad<float>()[0], std::exp(1.0f), "d(exp)/dx");
    }
    
    // Test Log
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 2.0f);
        Tensor y = autograd::log(x); // ln(2)
        y.backward();
        // dy/dx = 1/x = 0.5
        check_grad(x.grad<float>()[0], 0.5f, "d(ln)/dx");
    }
    
    // Test Exp2
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 3.0f);
        Tensor y = autograd::exp2(x); // 2^3 = 8
        y.backward();
        // dy/dx = 2^x * ln(2) = 8 * 0.693...
        check_grad(x.grad<float>()[0], 8.0f * M_LN2, "d(2^x)/dx");
    }
    
    // Test Log10
    {
        Tensor x = Tensor::full(Shape{}, req_grad, 10.0f);
        Tensor y = autograd::log10(x); // 1
        y.backward();
        // dy/dx = 1/(x * ln(10))
        check_grad(x.grad<float>()[0], 1.0f / (10.0f * M_LN10), "d(log10)/dx");
    }

    std::cout << "\nAll checked tests passed." << std::endl;
    return 0;
}
