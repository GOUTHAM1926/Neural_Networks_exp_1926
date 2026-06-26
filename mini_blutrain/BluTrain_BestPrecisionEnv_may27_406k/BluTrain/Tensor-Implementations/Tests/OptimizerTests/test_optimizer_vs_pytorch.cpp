
/**
* @file test_optimizer_vs_pytorch.cpp
* @brief Compares OwnTensor's Adam/AdamW optimizer output against PyTorch (CPU and GPU).
*
* This test:
* 1. Creates a weight tensor with fixed values
* 2. Sets a fixed gradient
* 3. Runs optimizer step(s) on CPU and GPU
* 4. Prints results to 7 decimal places for comparison with PyTorch
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


void test_adam_vs_pytorch(int num_steps, bool use_gpu) {
   std::string device_str = use_gpu ? "GPU" : "CPU";
  
   std::cout << "\n========================================" << std::endl;
   std::cout << "=== Adam Optimizer (" << device_str << "): OwnTensor vs PyTorch ===" << std::endl;
   std::cout << "========================================\n" << std::endl;
  
   // Hyperparameters (must match Python)
   float lr = 0.001f;
   float beta1 = 0.9f;
   float beta2 = 0.999f;
   float eps = 1e-8f;
   float weight_decay = 0.01f;
  
   std::cout << "Hyperparameters:" << std::endl;
   std::cout << "  lr = " << lr << std::endl;
   std::cout << "  beta1 = " << beta1 << std::endl;
   std::cout << "  beta2 = " << beta2 << std::endl;
   std::cout << "  eps = " << eps << std::endl;
   std::cout << "  weight_decay = " << weight_decay << std::endl;
   std::cout << "  num_steps = " << num_steps << std::endl;
   std::cout << "  device = " << device_str << std::endl;
   std::cout << std::endl;
  
   // Create weight tensor with specific values (reproducible)
   TensorOptions opts = TensorOptions()
       .with_dtype(Dtype::Float32)
       .with_req_grad(true);
  
   // Create 4x4 tensor with known values [0.1, 0.2, 0.3, ..., 1.6]
   Tensor W = Tensor::zeros(Shape{{4, 4}}, opts);
   float* w_data = W.data<float>();
   for (int i = 0; i < 16; ++i) {
       w_data[i] = 0.1f * (i + 1);  // 0.1, 0.2, ..., 1.6
   }
   W.set_requires_grad(true);
  
   // Move to GPU if requested
   if (use_gpu) {
       W = W.to_cuda(0);
       W.set_requires_grad(true);
   }
  
   std::cout << "--- Initial Weights ---" << std::endl;
   print_tensor("W_initial", W);
  
   // Set fixed gradient [1.0, 1.0, ..., 1.0]
   Tensor grad = Tensor::ones(Shape{{4, 4}}, TensorOptions().with_dtype(Dtype::Float32));
   if (use_gpu) {
       grad = grad.to_cuda(0);
   }
  
   std::cout << "\n--- Gradient (fixed) ---" << std::endl;
   print_tensor("grad", grad);
  
   // Create Adam optimizer
   nn::Adam adam({W}, lr, beta1, beta2, eps, weight_decay);
  
   // Run multiple steps
   for (int step = 1; step <= num_steps; ++step) {
       // Set the gradient manually
       W.set_grad(grad);
      
       // Optimizer step
       adam.step();
      
       std::cout << "\n--- After Step " << step << " ---" << std::endl;
       print_tensor("W", W);
   }
}


void test_adamw_vs_pytorch(int num_steps, bool use_gpu) {
   std::string device_str = use_gpu ? "GPU" : "CPU";
  
   std::cout << "\n========================================" << std::endl;
   std::cout << "=== AdamW Optimizer (" << device_str << "): OwnTensor vs PyTorch ===" << std::endl;
   std::cout << "========================================\n" << std::endl;
  
   // Hyperparameters (must match Python)
   float lr = 0.001f;
   float beta1 = 0.9f;
   float beta2 = 0.999f;
   float eps = 1e-8f;
   float weight_decay = 0.01f;
  
   std::cout << "Hyperparameters:" << std::endl;
   std::cout << "  lr = " << lr << std::endl;
   std::cout << "  beta1 = " << beta1 << std::endl;
   std::cout << "  beta2 = " << beta2 << std::endl;
   std::cout << "  eps = " << eps << std::endl;
   std::cout << "  weight_decay = " << weight_decay << std::endl;
   std::cout << "  num_steps = " << num_steps << std::endl;
   std::cout << "  device = " << device_str << std::endl;
   std::cout << std::endl;
  
   // Create weight tensor with specific values (reproducible)
   TensorOptions opts = TensorOptions()
       .with_dtype(Dtype::Float32)
       .with_req_grad(true);
  
   // Create 4x4 tensor with known values [0.1, 0.2, 0.3, ..., 1.6]
   Tensor W = Tensor::zeros(Shape{{4, 4}}, opts);
   float* w_data = W.data<float>();
   for (int i = 0; i < 16; ++i) {
       w_data[i] = 0.1f * (i + 1);  // 0.1, 0.2, ..., 1.6
   }
   W.set_requires_grad(true);
  
   // Move to GPU if requested
   if (use_gpu) {
       W = W.to_cuda(0);
       W.set_requires_grad(true);
   }
  
   std::cout << "--- Initial Weights ---" << std::endl;
   print_tensor("W_initial", W);
  
   // Set fixed gradient [1.0, 1.0, ..., 1.0]
   Tensor grad = Tensor::ones(Shape{{4, 4}}, TensorOptions().with_dtype(Dtype::Float32));
   if (use_gpu) {
       grad = grad.to_cuda(0);
   }
  
   std::cout << "\n--- Gradient (fixed) ---" << std::endl;
   print_tensor("grad", grad);
  
   // Create AdamW optimizer
   nn::AdamW adamw({W}, lr, beta1, beta2, eps, weight_decay);
  
   // Run multiple steps
   for (int step = 1; step <= num_steps; ++step) {
       // Set the gradient manually
       W.set_grad(grad);
      
       // Optimizer step
       adamw.step();
      
       std::cout << "\n--- After Step " << step << " ---" << std::endl;
       print_tensor("W", W);
   }
}


int main() {
   int num_steps = 2;
  
   std::cout << "================================================================" << std::endl;
   std::cout << "    OwnTensor Optimizer vs PyTorch Accuracy Test" << std::endl;
   std::cout << "================================================================" << std::endl;
  
   // CPU Tests
   std::cout << "\n############### CPU TESTS ###############" << std::endl;
   test_adam_vs_pytorch(num_steps, false);
   test_adamw_vs_pytorch(num_steps, false);
  
   // GPU Tests
   std::cout << "\n############### GPU TESTS ###############" << std::endl;
   test_adam_vs_pytorch(num_steps, true);
   test_adamw_vs_pytorch(num_steps, true);
  
   std::cout << "\n================================================================" << std::endl;
   std::cout << "    Now run: python3 test_optimizer_vs_pytorch.py" << std::endl;
   std::cout << "    to compare with PyTorch results" << std::endl;
   std::cout << "================================================================" << std::endl;
  
   return 0;
}
