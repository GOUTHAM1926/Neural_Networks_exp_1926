/**
* @file test_sgd_vs_pytorch.cpp
* @brief Compares OwnTensor's SGD optimizer variations against PyTorch.
*
* Tests: Vanilla SGD, SGD+Momentum, SGD+WeightDecay, SGD+Momentum+WeightDecay
* Uses same inputs as Adam test for consistency.
*/
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>


#include "TensorLib.h"


using namespace OwnTensor;


// Print tensor values to stdout with 7 decimal precision
void print_tensor(const std::string& name, const Tensor& t, int max_elements = 16) {
   Tensor tc = t.to_cpu();
   const float* d = tc.data<float>();
   int n = std::min(max_elements, (int)tc.numel());
  
   std::cout << name << " [" << tc.numel() << " elements, showing first " << n << "]:" << std::endl;
   for (int i = 0; i < n; ++i) {
       std::cout << std::fixed << std::setprecision(7) << d[i];
       if (i < n - 1) std::cout << ", ";
       if ((i + 1) % 8 == 0) std::cout << std::endl;
   }
   if (n % 8 != 0) std::cout << std::endl;
}


void test_sgd(const std::string& name, float lr, float momentum, float weight_decay) {
   std::cout << "\n========================================" << std::endl;
   std::cout << "=== SGD (" << name << "): OwnTensor ===" << std::endl;
   std::cout << "========================================\n" << std::endl;
  
   std::cout << "Hyperparameters:" << std::endl;
   std::cout << "  lr = " << lr << std::endl;
   std::cout << "  momentum = " << momentum << std::endl;
   std::cout << "  weight_decay = " << weight_decay << std::endl;
   std::cout << std::endl;
  
   // Create weight tensor: [0.1, 0.2, ..., 1.6]
   TensorOptions opts = TensorOptions()
       .with_dtype(Dtype::Float32)
       .with_req_grad(true);
  
   Tensor W = Tensor::zeros(Shape{{4, 4}}, opts);
   float* w_data = W.data<float>();
   for (int i = 0; i < 16; ++i) {
       w_data[i] = 0.1f * (i + 1);
   }
   W.set_requires_grad(true);
  
   std::cout << "--- Initial Weights ---" << std::endl;
   print_tensor("W_initial", W);
  
   // Fixed gradient [1.0, 1.0, ..., 1.0]
   Tensor grad = Tensor::ones(Shape{{4, 4}}, TensorOptions().with_dtype(Dtype::Float32));
  
   std::cout << "\n--- Gradient (fixed) ---" << std::endl;
   print_tensor("grad", grad);
  
   // Create SGD optimizer
   nn::SGDOptimizer sgd({W}, lr, momentum, weight_decay);
  
   // Set gradient and step
   W.set_grad(grad);
   sgd.step();
  
   std::cout << "\n--- After Step 1 ---" << std::endl;
   print_tensor("W", W);
}


int main() {
   std::cout << "================================================================" << std::endl;
   std::cout << "    OwnTensor SGD Optimizer vs PyTorch Accuracy Test" << std::endl;
   std::cout << "================================================================" << std::endl;
  
   float lr = 0.1f;
  
   // Test all SGD variations
   test_sgd("Vanilla", lr, 0.0f, 0.0f);
   test_sgd("Momentum", lr, 0.9f, 0.0f);
   test_sgd("WeightDecay", lr, 0.0f, 0.01f);
   test_sgd("Momentum+WeightDecay", lr, 0.9f, 0.01f);
  
   std::cout << "\n================================================================" << std::endl;
   std::cout << "    Now run: python3 test_sgd_vs_pytorch.py" << std::endl;
   std::cout << "================================================================" << std::endl;
  
   return 0;
}
