#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "core/Tensor.h"
#include "nn/optimizer/Optim.h"


using namespace OwnTensor;
using namespace OwnTensor::nn;


bool approx_equal(float a, float b, float tol = 1e-5) {
   return std::abs(a - b) < tol;
}


void test_sgd_plain() {
   std::cout << "Testing Plain SGD..." << std::endl;
   Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
   // p = [1.0]
   p.set_grad(Tensor::full(Shape{{1}}, TensorOptions(), 0.1f));
   // grad = [0.1]
  
   SGDOptimizer optim({p}, 0.1f); // lr = 0.1
   optim.step();
  
   // p_new = p - lr * grad = 1.0 - 0.1 * 0.1 = 0.99
   float val = *p.data<float>();
   if (approx_equal(val, 0.99f)) {
       std::cout << "[PASS] Plain SGD" << std::endl;
   } else {
       std::cout << "[FAIL] Plain SGD: expected 0.99, got " << val << std::endl;
   }
}


void test_sgd_momentum() {
   std::cout << "Testing SGD with Momentum..." << std::endl;
   Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
   // p = [1.0]
  
   float lr = 0.1f;
   float momentum = 0.9f;
   SGDOptimizer optim({p}, lr, momentum);
  
   // Step 1
   p.set_grad(Tensor::full(Shape{{1}}, TensorOptions(), 0.1f));
   // v = 0
   // v = 0.9 * 0 + 0.1 = 0.1
   // p = 1.0 - 0.1 * 0.1 = 0.99
   optim.step();
   if (!approx_equal(*p.data<float>(), 0.99f)) {
        std::cout << "[FAIL] SGD Momentum Step 1: expected 0.99, got " << *p.data<float>() << std::endl;
        return;
   }

   // Step 2
   p.set_grad(Tensor::full(Shape{{1}}, TensorOptions(), 0.1f));
   // v = 0.9 * 0.1 + 0.1 = 0.09 + 0.1 = 0.19
   // p = 0.99 - 0.1 * 0.19 = 0.99 - 0.019 = 0.971
   optim.step();
  
   if (approx_equal(*p.data<float>(), 0.971f)) {
       std::cout << "[PASS] SGD Momentum Step 2" << std::endl;
   } else {
       std::cout << "[FAIL] SGD Momentum Step 2: expected 0.971, got " << *p.data<float>() << std::endl;
   }
}

void test_sgd_weight_decay() {
   std::cout << "Testing SGD with Weight Decay..." << std::endl;
   Tensor p = Tensor::ones(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
   // p = [1.0]
  
   float lr = 0.1f;
   float wd = 0.1f;
   SGDOptimizer optim({p}, lr, 0.0f, wd);
  
   p.set_grad(Tensor::full(Shape{{1}}, TensorOptions(), 0.1f));
   // grad = 0.1
   // grad = grad + wd * p = 0.1 + 0.1 * 1.0 = 0.2
  
   optim.step();
  
   // p = p - lr * grad = 1.0 - 0.1 * 0.2 = 1.0 - 0.02 = 0.98
  
   if (approx_equal(*p.data<float>(), 0.98f)) {
       std::cout << "[PASS] SGD Weight Decay" << std::endl;
   } else {
       std::cout << "[FAIL] SGD Weight Decay: expected 0.98, got " << *p.data<float>() << std::endl;
   }
}

int main() {
   test_sgd_plain();
   test_sgd_momentum();
   test_sgd_weight_decay();
   return 0;
}
