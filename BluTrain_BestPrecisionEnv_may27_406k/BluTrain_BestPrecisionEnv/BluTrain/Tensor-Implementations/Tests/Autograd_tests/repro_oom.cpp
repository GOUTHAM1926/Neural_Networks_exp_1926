#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/MatrixOps.h"
#include "nn/optimizer/Optim.h"
#include <iostream>
#include <vector>
#include <chrono>
#include "device/DeviceCore.h"

using namespace OwnTensor;

// repro_oom.cpp
// Simulates a Mini-GPT training loop to check for memory leaks

int main() {
    std::cout << "Starting OOM reproduction test (Mini-GPT, 500 steps)..." << std::endl;
    std::cout << "Initial Active Tensors: " << Tensor::get_active_tensor_count() << std::endl;

    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_req_grad(true);

    bool use_cuda = OwnTensor::device::cuda_available();
    if (use_cuda) {
        opts = opts.with_device(Device::CUDA);
        std::cout << "Running on CUDA" << std::endl;
    } else {
        std::cout << "Running on CPU" << std::endl;
    }

    // Shapes like GPT-2 small
    int64_t B = 8;
    int64_t T = 64;
    int64_t C = 256;
    int64_t vocab = 1000;
    
    DeviceIndex target_device = use_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU);
    
    // Pos indices
    std::vector<int64_t> pos_data(T);
    for (int i = 0; i < T; ++i) pos_data[i] = i;
    Tensor pos_cpu = Tensor(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
    pos_cpu.set_data(pos_data);
    Tensor pos = pos_cpu.to(target_device);

    int64_t start_tensors = Tensor::get_active_tensor_count();
    
    {
        // Model parameters (Mini GPT style)
        Tensor wte = Tensor::randn<float>(Shape{{vocab, C}}, opts, 1234, 0.02f);
        Tensor wpe = Tensor::randn<float>(Shape{{T, C}}, opts, 2345, 0.02f);
        
        // MLP params
        int64_t hidden = C * 4;
        Tensor ln_gamma = Tensor::ones(Shape{{C}}, opts);
        Tensor ln_beta = Tensor::zeros(Shape{{C}}, opts);
        Tensor W_up = Tensor::randn<float>(Shape{{C, hidden}}, opts, 3456, 0.02f);
        Tensor b_up = Tensor::zeros(Shape{{hidden}}, opts);
        Tensor W_down = Tensor::randn<float>(Shape{{hidden, C}}, opts, 4567, 0.02f);
        Tensor b_down = Tensor::zeros(Shape{{C}}, opts);
        
        // Final LN
        Tensor ln_f_gamma = Tensor::ones(Shape{{C}}, opts);
        Tensor ln_f_beta = Tensor::zeros(Shape{{C}}, opts);

        // Inputs (constant for simplicity, but new tensor each iter)
        std::vector<int64_t> tok_data(B * T);
        std::vector<int64_t> tgt_data(B * T);
        for (int i = 0; i < B * T; ++i) {
            tok_data[i] = i % vocab;
            tgt_data[i] = (i + 1) % vocab;
        }
        
        int64_t steps = 10000; // Let's run 1000 steps to be sure

        // Optimizer
        std::vector<Tensor> params = {wte, wpe, ln_gamma, ln_beta, W_up, b_up, W_down, b_down, ln_f_gamma, ln_f_beta};
        nn::Adam optimizer(params, 1e-3f);

        for (int step = 0; step < steps; ++step) {
            // Zero grad
            optimizer.zero_grad();

            // Create inputs
            Tensor tokens_cpu = Tensor(Shape{{B, T}}, TensorOptions().with_dtype(Dtype::Int64));
            tokens_cpu.set_data(tok_data);
            Tensor tokens = tokens_cpu.to(target_device);
            
            Tensor targets_cpu = Tensor(Shape{{B * T}}, TensorOptions().with_dtype(Dtype::Int64));
            targets_cpu.set_data(tgt_data);
            Tensor targets = targets_cpu.to(target_device);

            // Forward Pass (Mini GPT)
            Tensor tok_emb = autograd::embedding(wte, tokens);
            Tensor pos_emb = autograd::embedding(wpe, pos);
            Tensor x = autograd::add(tok_emb, pos_emb);
            x = x.reshape(Shape{{B * T, C}});
            
            // Block
            Tensor h = autograd::layer_norm(x, ln_gamma, ln_beta, C);
            h = autograd::matmul(h, W_up);
            h = autograd::add(h, b_up);
            h = autograd::gelu(h);
            h = autograd::matmul(h, W_down);
            h = autograd::add(h, b_down);
            x = autograd::add(x, h);
            
            // Final
            x = autograd::layer_norm(x, ln_f_gamma, ln_f_beta, C);
            Tensor logits = autograd::matmul(x, wte.t());
            
            // Loss
            Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
            
            // Backward
            loss.backward();

            // Optimizer Step
            optimizer.step();

            if (step % 200 == 0) {
                int64_t current_tensors = Tensor::get_active_tensor_count();
                std::cout << "Step " << step << " | Active Tensors: " << current_tensors 
                        << " | Net Change: " << (current_tensors - start_tensors) << std::endl;
            }
        }
    }

    int64_t final_tensors = Tensor::get_active_tensor_count();
    std::cout << "Final Active Tensors: " << final_tensors << std::endl;
    std::cout << "Net Leak: " << (final_tensors - start_tensors) << std::endl;

    if (final_tensors > start_tensors) { 
        std::cerr << "FAIL: Possible memory leak detected!" << std::endl;
        return 1;
    } else {
        std::cout << "PASS: No leaks detected." << std::endl;
        return 0;
    }
}
