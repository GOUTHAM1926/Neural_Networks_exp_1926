
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <cstdlib>
#include <ctime>


#include "TensorLib.h"


using namespace OwnTensor;
using namespace OwnTensor::autograd;


// A complex MLP with various activation functions to test optimizer stability and convergence
// Architecture:
// Input (32) -> Linear(64) -> GELU -> Linear(32) -> GELU -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(Out)
// Using Sparse Cross Entropy Loss (Logits + Integer Targets)


void run_training(std::string optim_name, int epochs = 200, bool use_gpu = false) {
  std::cout << "Starting " << optim_name << " training";
  if (use_gpu) std::cout << " (GPU)";
  std::cout << "..." << std::endl;


  // Reset seed for identical initialization
  std::srand(42);


  int B = 16;
  int In = 32;
  int Out = 10;


  // Create parameters on CPU first for reproducible initialization.
  // IMPORTANT: Scale BEFORE setting requires_grad.
  // Since commit 8583811, operator*(tensor, scalar) routes through
  // autograd::mul() when requires_grad=true, creating a non-leaf tensor.
  // Non-leaf tensors don't accumulate .grad, breaking CPU training.
  TensorOptions cpu_init_opts = TensorOptions().with_dtype(Dtype::Float32);
  TensorOptions cpu_bias_opts = TensorOptions()
      .with_req_grad(true)
      .with_dtype(Dtype::Float32);


  // Kaiming-style scaling (* 0.25f), then set requires_grad to stay leaf
  Tensor W1 = Tensor::randn<float>(Shape{{In, 64}}, cpu_init_opts) * 0.25f;
  W1.set_requires_grad(true);
  Tensor b1 = Tensor::zeros(Shape{{1, 64}}, cpu_bias_opts);
  Tensor W2 = Tensor::randn<float>(Shape{{64, 32}}, cpu_init_opts) * 0.25f;
  W2.set_requires_grad(true);
  Tensor b2 = Tensor::zeros(Shape{{1, 32}}, cpu_bias_opts);
  Tensor W3 = Tensor::randn<float>(Shape{{32, 32}}, cpu_init_opts) * 0.25f;
  W3.set_requires_grad(true);
  Tensor b3 = Tensor::zeros(Shape{{1, 32}}, cpu_bias_opts);
  Tensor W4 = Tensor::randn<float>(Shape{{32, 32}}, cpu_init_opts) * 0.25f;
  W4.set_requires_grad(true);
  Tensor b4 = Tensor::zeros(Shape{{1, 32}}, cpu_bias_opts);
  Tensor W5 = Tensor::randn<float>(Shape{{32, Out}}, cpu_init_opts) * 0.25f;
  W5.set_requires_grad(true);
  Tensor b5 = Tensor::zeros(Shape{{1, Out}}, cpu_bias_opts);


  // Move to GPU if needed
  if (use_gpu) {
      W1 = W1.to_cuda(0); W1.set_requires_grad(true);
      b1 = b1.to_cuda(0); b1.set_requires_grad(true);
      W2 = W2.to_cuda(0); W2.set_requires_grad(true);
      b2 = b2.to_cuda(0); b2.set_requires_grad(true);
      W3 = W3.to_cuda(0); W3.set_requires_grad(true);
      b3 = b3.to_cuda(0); b3.set_requires_grad(true);
      W4 = W4.to_cuda(0); W4.set_requires_grad(true);
      b4 = b4.to_cuda(0); b4.set_requires_grad(true);
      W5 = W5.to_cuda(0); W5.set_requires_grad(true);
      b5 = b5.to_cuda(0); b5.set_requires_grad(true);
  }


  // Parameter tensors for optimizers
  std::vector<Tensor> params = {W1, b1, W2, b2, W3, b3, W4, b4, W5, b5};


  // Optimizer selection
  std::unique_ptr<nn::SGDOptimizer> sgd_optim;
  std::unique_ptr<nn::Adam> adam_optim;
  std::unique_ptr<nn::AdamW> adamw_optim;


  bool use_sgd = false;
  bool use_adam = false;
  bool use_adamw = false;


  if (optim_name.find("SGD") == 0) {
      // SGDOptimizer now uses pointer-based API like Adam/AdamW
      float momentum = 0.0f;
      float weight_decay = 0.0f;


      if (optim_name.find("Momentum") != std::string::npos) {
          momentum = 0.9f;
      }
      if (optim_name.find("WD") != std::string::npos) {
          weight_decay = 0.001f; // Reduced WD for stability
      }


      sgd_optim = std::make_unique<nn::SGDOptimizer>(params, 0.001f, momentum, weight_decay);
      use_sgd = true;
  }
  else if (optim_name == "Adam") {
      // Adam with weight_decay=0.01
      adam_optim = std::make_unique<nn::Adam>(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
      use_adam = true;
  }
  else if (optim_name == "Adam_NoWD") {
      // Adam without weight decay
      adam_optim = std::make_unique<nn::Adam>(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
      use_adam = true;
  }
  else if (optim_name == "AdamW") {
      // AdamW
      adamw_optim = std::make_unique<nn::AdamW>(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
      use_adamw = true;
  }


  // Data: Create Random Integer Targets for Sparse Cross Entropy
  TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32); // Input is Float
  TensorOptions cpu_index_opts = TensorOptions().with_dtype(Dtype::Int32); // Targets are Int32


  Tensor X_cpu = Tensor::randn<float>(Shape{{B, In}}, cpu_opts);
  // Random Integer Targets [0, Out)
  Tensor Y_cpu = Tensor::zeros(Shape{{B}}, cpu_index_opts);
  int32_t* y_data = Y_cpu.data<int32_t>();
  for(int i = 0; i < B; ++i) {
      y_data[i] = static_cast<int32_t>(std::rand() % Out);
  }


  // Move to GPU if needed
  Tensor X = use_gpu ? X_cpu.to_cuda(0) : X_cpu;
  Tensor Y = use_gpu ? Y_cpu.to_cuda(0) : Y_cpu;


  std::cout << std::setw(10) << "Epoch" << std::setw(20) << "Loss" << std::endl;
  std::cout << "------------------------------" << std::endl;


  for (int epoch = 0; epoch <= epochs; ++epoch) {
      // Forward pass using autograd operations
      Tensor L1 = gelu(add(autograd::matmul(X, W1), b1));
      Tensor L2 = gelu(add(autograd::matmul(L1, W2), b2));
      Tensor L3 = relu(add(autograd::matmul(L2, W3), b3));
      Tensor L4 = relu(add(autograd::matmul(L3, W4), b4));
      Tensor logits = add(autograd::matmul(L4, W5), b5);


      // Loss - using SPARSE cross entropy (expects Logits and Integer Index Targets)
      // NOTE: We do NOT apply Softmax here. LogSumExp is inside the loss.
      Tensor loss = sparse_cross_entropy_loss(logits, Y);


      float loss_val = loss.to_cpu().data<float>()[0];


      if (epoch % 10 == 0) {
           std::cout << std::setw(10) << epoch << std::setw(20) << loss_val << std::endl;
      }


      // Zero gradients and backward
      if (use_sgd) {
          sgd_optim->zero_grad();
      } else if (use_adam) {
          adam_optim->zero_grad();
      } else if (use_adamw) {
          adamw_optim->zero_grad();
      }


      backward(loss);


      // Optimizer step
      if (use_sgd) {
          sgd_optim->step();
      } else if (use_adam) {
          adam_optim->step();
      } else if (use_adamw) {
          adamw_optim->step();
      }
  }

  // Calculate total weight norm
  float total_norm_sq = 0.0f;
  for (auto& p : params) {
      if (p.ndim() > 1) {
          Tensor norm_sq = autograd::sum(autograd::mul(p, p));
          total_norm_sq += norm_sq.to_cpu().data<float>()[0];
      }
  }
  std::cout << "Final Total Weight L2 Norm: " << std::sqrt(total_norm_sq) << std::endl;
}

int main() {
  std::cout << "========== CPU Tests ==========" << std::endl;
  // Reduced epochs because convergence should be faster
  run_training("SGD", 200, false);  // Vanilla SGD
  std::cout << std::endl;
  run_training("SGD_WD", 200, false);  // SGD + Weight Decay
  std::cout << std::endl;
  run_training("SGD_Momentum", 200, false);  // SGD + Momentum
  std::cout << std::endl;
  run_training("SGD_Momentum_WD", 200, false);  // SGD + Momentum + Weight Decay
  std::cout << std::endl;
  run_training("Adam", 200, false);
  std::cout << std::endl;
  run_training("Adam_NoWD", 200, false);
  std::cout << std::endl;
  run_training("AdamW", 200, false);
  std::cout << std::endl;

  std::cout << "========== GPU Tests ==========" << std::endl;
  run_training("Adam", 200, true);
  std::cout << std::endl;
  run_training("Adam_NoWD", 200, true);
  std::cout << std::endl;
  run_training("AdamW", 200, true);
  std::cout << std::endl;

  return 0;
}