/**
 * @file 01_tensor_basics.cpp
 * @brief GUIDE: Basic Tensor Operations
 * 
 * This guide demonstrates fundamental tensor operations:
 * - Creating tensors (zeros, ones, randn, from data)
 * - Basic math operations (+, -, *, /)
 * - Shape and dimension handling
 * - Data access patterns
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/01_tensor_basics.cpp
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;

void print_tensor(const std::string& name, const Tensor& t) {
    std::cout << name << ": shape=" << t.ndim() << "D [";
    for (int64_t i = 0; i < t.ndim(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << t.shape().dims[i];
    }
    std::cout << "], numel=" << t.numel() << std::endl;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║          GUIDE: Basic Tensor Operations              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // =========================================================================
    // 1. Creating Tensors
    // =========================================================================
    std::cout << "=== 1. Creating Tensors ===\n\n";
    
    // Method 1: Zeros
    Tensor zeros = Tensor::zeros(Shape{{3, 4}}, TensorOptions());
    print_tensor("zeros(3,4)", zeros);
    
    // Method 2: Ones
    Tensor ones = Tensor::ones(Shape{{2, 3}}, TensorOptions());
    print_tensor("ones(2,3)", ones);
    
    // Method 3: Random normal
    Tensor randn = Tensor::randn<float>(Shape{{2, 2}}, TensorOptions());
    print_tensor("randn(2,2)", randn);
    std::cout << "  Values: ";
    const float* data = randn.data<float>();
    for (size_t i = 0; i < randn.numel(); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Method 4: With requires_grad for autograd
    Tensor with_grad = Tensor::randn<float>(Shape{{3, 3}}, 
                                      TensorOptions().with_req_grad(true));
    print_tensor("with_grad(3,3)", with_grad);
    std::cout << "  requires_grad: " << (with_grad.requires_grad() ? "true" : "false") << "\n\n";

    // =========================================================================
    // 2. Basic Math Operations
    // =========================================================================
    std::cout << "=== 2. Basic Math Operations ===\n\n";
    
    Tensor a = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    Tensor b = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    
    // Element-wise addition
    Tensor c = autograd::add(a, b);
    std::cout << "autograd::add(a, b) (ones + ones):\n  ";
    const float* c_data = c.data<float>();
    for (size_t i = 0; i < c.numel(); i++) {
        std::cout << c_data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Element-wise multiplication
    Tensor d = autograd::mul(a, b);
    std::cout << "autograd::mul(a, b) (ones * ones):\n  ";
    const float* d_data = d.data<float>();
    for (size_t i = 0; i < d.numel(); i++) {
        std::cout << d_data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Scalar multiplication (manual approach)
    Tensor e = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    float* e_data = const_cast<float*>(e.data<float>());
    for (size_t i = 0; i < e.numel(); i++) {
        e_data[i] = 2.5f;  // Simulating a * 2.5
    }
    std::cout << "Creating tensor with value 2.5:\n  ";
    for (size_t i = 0; i < e.numel(); i++) {
        std::cout << e_data[i] << " ";
    }
    std::cout << "\n\n";

    // =========================================================================
    // 3. Matrix Multiplication
    // =========================================================================
    std::cout << "=== 3. Matrix Multiplication ===\n\n";
    
    Tensor m1 = Tensor::ones(Shape{{2, 3}}, TensorOptions());
    Tensor m2 = Tensor::ones(Shape{{3, 4}}, TensorOptions());
    
    Tensor result = autograd::matmul(m1, m2);
    print_tensor("matmul(2x3, 3x4)", result);
    std::cout << "  First row: ";
    const float* r_data = result.data<float>();
    for (size_t i = 0; i < 4; i++) {
        std::cout << r_data[i] << " ";
    }
    std::cout << "\n\n";

    // =========================================================================
    // 4. Reductions
    // =========================================================================
    std::cout << "=== 4. Reductions ===\n\n";
    
    Tensor vals = Tensor::randn<float>(Shape{{3, 3}}, TensorOptions());
    
    // Sum
    Tensor s = autograd::sum(vals);
    std::cout << "sum(randn(3,3)): " << s.data<float>()[0] << "\n";
    
    // Mean
    Tensor m = autograd::mean(vals);
    std::cout << "mean(randn(3,3)): " << m.data<float>()[0] << "\n\n";

    // =========================================================================
    // 5. Transpose
    // =========================================================================
    std::cout << "=== 5. Transpose ===\n\n";
    
    Tensor orig = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions());
    Tensor transposed = orig.t();
    
    print_tensor("original", orig);
    print_tensor("transposed", transposed);
    
    std::cout << "\n✅ Basic tensor operations guide complete!\n\n";
    return 0;
}
