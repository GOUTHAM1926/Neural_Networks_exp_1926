/**
 * @file gpt2wotying.cpp
 * @brief GPT-2 training script WITHOUT weight tying
 * 
 * This script implements GPT-2 training using custom tensor library with autograd support.
 * Architecture: Token Embedding -> Position Embedding -> MLP x n_layers -> Linear -> Cross Entropy
 * 
 * Key difference from gpt2_test.cpp: 
 * - Separate weights for token embedding and output projection (no weight tying)
 * - Token embedding uses forward() with gradient tracking
 * - Output projection uses its own trainable weight matrix
 */

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/optimizer/Optim.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "checkpointing/Checkpoint.h"

// Dataloader
#include "dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 8192;  // GPT-2 vocab size (padded to 64)
    int64_t n_embd = 384;
    int64_t n_layers = 3;
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding {
public:
    Tensor weight;  // [vocab_size, n_embd]
    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device, uint64_t seed = 1234)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        // Initialize weight with small normal distribution
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
    }
    
    // Forward: indices [B, T] -> embeddings [B, T, C]
    // Standard forward with gradient tracking through embedding
    Tensor forward(const Tensor& indices) {
        // Use autograd-aware embedding function for proper gradient flow
        return autograd::embedding(weight, indices);
    }
    
    std::vector<Tensor*> parameters() {
        return {&weight};
    }
    
private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};

// =============================================================================
// MLP Block
// =============================================================================

// Helper: Initialize nn::Linear weights with GPT-2 style (std=0.02)
// Creates tensors directly on the target device
void init_linear_gpt2(nn::Linear& layer, DeviceIndex device, float std = 0.02f, uint64_t seed = 1234) {
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    auto shape = layer.weight.shape();
    layer.weight = Tensor::randn<float>(shape, opts, seed, std);
    if (layer.bias.is_valid()) {
        layer.bias = Tensor::zeros(layer.bias.shape(), opts);
        // layer.bias = Tensor::randn<float>(layer.bias.shape(), opts, seed, std);
    }
    // std::cout << "WEIGHT: " << std::endl;
    // layer.weight.display();
    // std::cout << "BIAS: " << std::endl;
    // layer.bias.display();
}

class MLP {
public:
    nn::LayerNorm ln;       // LayerNorm before MLP
    nn::Linear fc_up;       // Linear(n_embd, 4*n_embd)
    nn::Linear fc_down;     // Linear(4*n_embd, n_embd)
    
    MLP(int64_t n_embd, int n_layers, DeviceIndex device, uint64_t seed = 1234)
        : n_embd_(n_embd), 
          ln(n_embd),
          fc_up(n_embd, 4 * n_embd, true),
          fc_down(4 * n_embd, n_embd, true)
    {
        // GPT-2 style initialization - create tensors directly on target device
        init_linear_gpt2(fc_up, device, 0.02f, seed);


        
        // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(fc_down, device, 0.02f * scale, seed + 1);
        
        // Move LayerNorm to device
        ln.to(device);
    }
    
    // Forward: x [B, T, C] -> [B, T, C]
    Tensor forward(const Tensor& x) {
        // Pre-Norm: ln(x)
        Tensor h = ln.forward(x);
        
        // Up projection + GELU + Down projection
        h = fc_up.forward(h);
        h = autograd::gelu(h);
        h = fc_down.forward(h);
        
        return h;
    }
    
    std::vector<Tensor*> parameters() {
        return {
            &fc_up.weight, &fc_up.bias,
            &fc_down.weight, &fc_down.bias,
            &ln.weight, &ln.bias
        };
    }
    
private:
    int64_t n_embd_;
};

// =============================================================================
// GPT Model (WITHOUT Weight Tying)
// =============================================================================

class GPT {
public:
    GPTConfig config;
    Embedding wte;  // Token embedding
    Embedding wpe;  // Position embedding
    std::vector<MLP> mlps;
    nn::LayerNorm ln_f; // Final LayerNorm
    Tensor W_out;  // Separate output projection weight [n_embd, vocab_size]

    GPT(GPTConfig cfg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg), 
          wte(cfg.vocab_size, cfg.n_embd, device, seed),
          wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);
        
        // Create MLP blocks
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.emplace_back(MLP(cfg.n_embd, cfg.n_layers, device, seed + 200 + i * 10));
        }
        
        // Separate output projection weight (no weight tying)
        // Shape: [n_embd, vocab_size] to compute: hidden @ W_out = logits
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        // Use same initialization as token embeddings (std=0.02)
        W_out = Tensor::randn<float>(Shape{{cfg.n_embd, cfg.vocab_size}}, opts, seed + 1000, 0.02f);
    }
    
    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    Tensor forward(const Tensor& idx) {
        auto shape = idx.shape().dims;
        int64_t B = shape[0];
        int64_t T = shape[1];
        
        // Create position indices [T]
        Tensor pos = Tensor(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(idx.device()));
        {
            Tensor pos_cpu(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
            int64_t* pos_data = pos_cpu.data<int64_t>();
            for (int64_t i = 0; i < T; ++i) {
                pos_data[i] = i;
            }
            if (idx.device().is_cuda()) {
                pos = pos_cpu.to(idx.device());
            } else {
                pos = pos_cpu;
            }
        }
        
        // Get embeddings [B, T, C]
        // Use regular forward() - both embeddings now track gradients independently
        Tensor tok_emb = wte.forward(idx);  // [B, T, C] - gradients flow through embedding
        Tensor pos_emb = wpe.forward(pos);  // [1, T, C] - broadcasts
        
        // Add embeddings
        Tensor x = autograd::add(tok_emb, pos_emb);
        
        // Apply MLP blocks with gradient checkpointing
        for (auto& mlp : mlps) {
            auto mlp_fn = [&mlp](const variable_list& inputs) -> variable_list {
                return {mlp.forward(inputs[0])};
            };
            x = autograd::checkpoint(mlp_fn, {x})[0];
        }
        
        // Final normalization
        x = ln_f.forward(x);
        
        // Final projection to vocab size [B, T, vocab_size]
        // Uses separate W_out instead of wte.weight.t()

        Tensor logits = autograd::matmul(x, W_out);
        
        return logits;
    }
    
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        
        // Token and position embeddings
        for (auto* p : wte.parameters()) params.push_back(p);
        for (auto* p : wpe.parameters()) params.push_back(p);
        
        // MLP blocks
        for (auto& mlp : mlps) {
            for (auto* p : mlp.parameters()) params.push_back(p);
        }
        
        // Final LN
        params.push_back(&ln_f.weight);
        params.push_back(&ln_f.bias);
        
        // Output projection (separate from token embedding)
        params.push_back(&W_out);
        
        return params;
    }
    
    int64_t count_params() {
        int64_t total = 0;
        std::cout << "Total tensors: " << parameters().size() << std::endl;
        for (auto* p : parameters()) {
            total += p->numel();
        }
        return total;
    }
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main() {
    try {
        std::cout << "=== GPT-2 Training Script (WITHOUT Weight Tying) ===" << std::endl;
        
        // Configuration
        GPTConfig config;
        config.context_length = 1024;
        config.vocab_size = 8192;
        config.n_embd = 384;
        config.n_layers = 6;
        
        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int global_batch = 65536;  // Global batch size
        const int grad_accum_steps = global_batch / (B * T);
        
        const float max_lr = 2e-4f;  
        const float min_lr = max_lr * 0.1f;
        const int warmup_steps = 62;
        const int max_steps = 617;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  global_batch: " << global_batch << std::endl;
        std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
        std::cout << "  Weight Tying: DISABLED" << std::endl;
        
        // Set device - GPU-1 for training
        int gpu_device = 0;  // Use GPU-1
        int rank = 0;        // Rank for dataloader (0 for single-GPU training)
        int world_size = 1;  // Single GPU
        DeviceIndex device(Device::CUDA, gpu_device);
        cudaSetDevice(gpu_device);
        
        std::cout << "\nInitializing model on CUDA device " << gpu_device << "..." << std::endl;
        
        // Create model
        GPT model(config, device);
        
        // Print parameter count
        int64_t num_params = model.count_params();
        std::cout << "Number of parameters: " << num_params << std::endl;
        std::cout << "(Note: More params than weight-tied version due to separate W_out)" << std::endl;
        
        // Get all parameters
        auto params_ptrs = model.parameters();
        std::vector<Tensor> params;
        for (auto* p : params_ptrs) {
            params.push_back(*p);
        }
        // Create optimizer
        nn::Adam optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        // Create data loaders
        std::string data_root = "/home/blubridge-029/agtensor/tensor/Tests/training/data";
        DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, true, 100000000, gpu_device);
        DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true, 100000000, gpu_device);
        
        std::cout << "\nStarting training..." << std::endl;
        
        // Create CSV log file
        std::ofstream log_file("log_new_pull_loss1.csv");
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
        
        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
        
        for (int step = 0; step < max_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Validation every 100 steps
            if (step % 100 == 0 || step == max_steps - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    Tensor x = batch.input.to(device);
                    Tensor y = batch.target.to(device);
                    
                    Tensor logits = model.forward(x);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                    
                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }
                
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }
            
            // Training step

            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            // Optimized: Accumulate loss on GPU to avoid CPU syncs
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                Batch batch = train_loader.next_batch();
                Tensor x = batch.input.to(device);
                Tensor y = batch.target.to(device);
                
                // Forward
                Tensor logits = model.forward(x);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                
                // Accumulate loss on GPU (no graph tracking needed for logging)
                loss_accum_gpu = loss_accum_gpu + loss;
                
                // Backward with scaling
                Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                loss.backward(&grad_scale);
            }
            
            // Synchronize ONCE after all micro-steps
            cudaDeviceSynchronize();
            
            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
            
            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }
            
            // Clip gradients
            float norm = nn::clip_grad_norm_(params, 1.0f);
            cudaDeviceSynchronize();
            
            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);
            
            // Optimizer step
            optimizer.step();
            cudaDeviceSynchronize();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
            
            // Print training info
            std::cout << "step " << std::setw(5) << step 
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                      << " | lr " << std::scientific << std::setprecision(4) << lr 
                      << " | norm: " << std::fixed << std::setprecision(4) << norm 
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                      << std::endl;
            
            // Log metrics to CSV
            log_file << step << "," 
                     << loss_accum << ","
                     << val_loss_accum_log << ","
                     << lr << ","
                     << norm << ","
                     << (dt * 1000.0) << ","
                     << tokens_per_sec << "\n";
            log_file.flush();
            val_loss_accum_log = -1.0f;  // Reset for next iteration
        }
        
        log_file.close();
        std::cout << "\nTraining log saved to: log_no_weight_tying.csv" << std::endl;
        
        std::cout << "\n=== Training Complete ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
