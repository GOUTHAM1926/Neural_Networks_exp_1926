#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <nvtx3/nvToolsExt.h> 
// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/LossOps.h"
#include "checkpointing/Checkpoint.h"
#include "checkpointing/Checkpointing.h"
#include "checkpointing/GradMode.h"
#include "mlp/activation.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
// Dataloader
#include "/home/blubridge-029/tensor/Tensor-Implementation/Tests/training/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
  int64_t context_length = 1024;
  int64_t vocab_size = 50304; // GPT-2 vocab size
  int64_t n_embd = 384;
  int64_t n_layers = 3;
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding : public nn::Module {
public:
  Tensor weight; // [vocab_size, n_embd]
  Embedding() = default;
  Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device,
            uint64_t seed = 1234)
      : vocab_size_(vocab_size), embed_dim_(embed_dim) {
    // Initialize weight with small normal distribution
    TensorOptions opts = TensorOptions()
                             .with_dtype(Dtype::Float32)
                             .with_device(device)
                             .with_req_grad(true);
    weight =
        Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);

    register_parameter(weight);
  }

  // Forward: indices [B, T] -> embeddings [B, T, C]
  // Standard forward with gradient tracking through embedding
  Tensor forward(const Tensor &indices) override {
    // Use autograd-aware embedding function for proper gradient flow
    return autograd::embedding(weight, indices);
  }

private:
  int64_t vocab_size_;
  int64_t embed_dim_;
};

// =============================================================================
// MLP Block
// =============================================================================

// Helper: Initialize nn::Linear weights with GPT-2 style (std=0.02)
void init_linear_gpt2(nn::Linear &layer, float std = 0.02f,
                      uint64_t seed = 1234, bool req_grad = true) {
  // IMPORTANT: Do NOT replace layer.weight with a new tensor!
  // nn::Linear already registered its weight in params_.
  // We must copy data INTO the existing weight to preserve parameter identity.

  auto shape = layer.weight.shape();
  TensorOptions opts =
      TensorOptions().with_dtype(Dtype::Float32); // CPU, no grad
  Tensor init_data = Tensor::randn<float>(shape, opts, seed, std);

  // Copy into existing weight (both on CPU at this point)
  layer.weight.copy_(init_data);
  layer.weight.set_requires_grad(req_grad);

  if (layer.bias.is_valid()) {
    Tensor bias_init = Tensor::zeros(layer.bias.shape(), opts);
    layer.bias.copy_(bias_init);
    layer.bias.set_requires_grad(req_grad);
  }
}

class MLP : public nn::Module {
public:
  nn::LayerNorm ln;   // LayerNorm before MLP
  nn::Linear fc_up;   // Linear(n_embd, 4*n_embd)
  nn::Linear fc_down; // Linear(4*n_embd, n_embd)

  MLP(int64_t n_embd, int n_layers, DeviceIndex device, uint64_t seed = 1234)
      : ln(n_embd), fc_up(n_embd, 4 * n_embd, true),
        fc_down(4 * n_embd, n_embd, true), n_embd_(n_embd) {
    // GPT-2 style initialization on CPU (preserves params_ identity)
    init_linear_gpt2(fc_up, 0.02f, seed);

    // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
    float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
    init_linear_gpt2(fc_down, 0.02f * scale, seed + 1);

    // Move everything to device (uses to_cuda_ which modifies in-place)
    fc_up.to(device);
    fc_down.to(device);
    ln.to(device);

    register_module(ln);
    register_module(fc_up);
    register_module(fc_down);
  }

  // Forward: x [B, T, C] -> [B, T, C]
  Tensor forward(const Tensor &x) override {
    // Pre-Norm: ln(x)
    Tensor h = ln.forward(x);

    // Up projection + GELU + Down projection
    h = fc_up.forward(h);
    h = autograd::gelu(h);
    h = fc_down.forward(h);

    // Residual connection: x + MLP(x)
    return autograd::add(x, h);
  }

private:
  int64_t n_embd_;
};

// =============================================================================
// GPT Model (WITHOUT Weight Tying)
// =============================================================================

class GPT : public nn::Module {
public:
  GPTConfig config;
  Embedding wte; // Token embedding [vocab_size, n_embd]
  Embedding wpe; // Position embedding
  std::shared_ptr<nn::Sequential> mlps;
  nn::LayerNorm ln_f; // Final LayerNorm
  nn::Linear
      lm_head; // Separate output projection [n_embd, vocab_size], bias=False

  GPT(GPTConfig cfg, DeviceIndex device, uint64_t seed = 1234)
      : config(cfg), wte(cfg.vocab_size, cfg.n_embd, device, seed),
        wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
        ln_f(cfg.n_embd), lm_head(cfg.n_embd, cfg.vocab_size, false) {
    ln_f.to(device);

    // Create MLP blocks and add to Sequential
    mlps = std::make_shared<nn::Sequential>();
    for (int i = 0; i < cfg.n_layers; ++i) {
      mlps->add(std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device,
                                      seed + 200 + i * 10));
    }

    // Initialize lm_head on CPU (preserves params_ identity), then move to GPU
    // NO weight tying — lm_head has its own independent weight
    init_linear_gpt2(lm_head, 0.02f, seed + 1000, true);
    lm_head.to(device);

    // Optimization: cache position tensor once (avoids re-creating + H2D
    // transfer every forward)
    Tensor pos_cpu(Shape{{1, cfg.context_length}},
                   TensorOptions().with_dtype(Dtype::Int64));
    int64_t *pos_data = pos_cpu.data<int64_t>();
    for (int64_t i = 0; i < cfg.context_length; ++i) {
      pos_data[i] = i;
    }
    cached_pos_ = pos_cpu.to(device);

    register_module(wte);
    register_module(wpe);
    register_module(*mlps);
    register_module(ln_f);
    register_module(lm_head);
  }

  // Forward: indices [B, T] -> logits [B, T, vocab_size]
  Tensor forward(const Tensor &idx) override {
    // Get embeddings [B, T, C]
    Tensor tok_emb = wte.forward(idx);         // [B, T, C]
    Tensor pos_emb = wpe.forward(cached_pos_); // [1, T, C] - broadcasts

    // Add embeddings
    Tensor x = autograd::add(tok_emb, pos_emb);

    // Apply MLP blocks (Sequential handles the loop and residual is inside MLP
    // now)
    x = mlps->forward(x);

    // Final normalization
    x = ln_f.forward(x);

    // Output projection via separate lm_head (NOT weight-tied with wte)
    Tensor logits = lm_head.forward(x); // [B, T, vocab_size]

    return logits;
  }

private:
  Tensor cached_pos_; // [1, T] position indices, cached on GPU
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps,
             int max_steps) {
  if (step < warmup_steps) {
    return max_lr * static_cast<float>(step + 1) /
           static_cast<float>(warmup_steps);
  }
  if (step > max_steps) {
    return min_lr;
  }
  float decay_ratio = static_cast<float>(step - warmup_steps) /
                      static_cast<float>(max_steps - warmup_steps);
  float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
  return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main(int argc, char *argv[]) {
  try {
    std::cout << "=== GPT-2 Training Script (WITHOUT Weight Tying) ==="
              << std::endl;

    // Configuration
    GPTConfig config;
    config.context_length = 1024;
    config.vocab_size = 50304;
    config.n_embd = 768;
    config.n_layers = 10;

    // Training hyperparameters
    const int B = 8;                // Batch size
    const int T = 1024;             // Sequence length
    const int global_batch = 65536; // Global batch size
    const int grad_accum_steps = global_batch / (B * T);

    const float max_lr = 1e-4f;
    const float min_lr = max_lr * 0.1f;
    const int warmup_steps = 1;
    const int max_steps = 10;
    const bool use_async_checkpointing = true; // Toggle this to compare performance!

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
    int gpu_device = 1; // Use GPU-1
    int rank = 0;       // Rank for dataloader (0 for single-GPU training)
    int world_size = 1; // Single GPU
    DeviceIndex device(Device::CUDA, gpu_device);
    cudaSetDevice(gpu_device);

    std::cout << "\nInitializing model on CUDA device " << gpu_device << "..."
              << std::endl;

    // Create model
    GPT model(config, device);

    // Print parameter count
    auto params = model.parameters();
    int64_t num_params = 0;
    for (auto &p : params)
      num_params += p.numel();

    std::cout << "Number of parameters: " << num_params << std::endl;
    std::cout
        << "(Note: More params than weight-tied version due to separate W_out)"
        << std::endl;

    // Get all parameters
    // auto params = model.parameters(); // Already got above

    // Create optimizer
    nn::AdamW optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);

    // Create data loaders
    std::string data_root =
        "/home/blubridge-029/tensor/Tensor-Implementation/data";
    DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root,
                                true, 100000000, gpu_device);
    DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true,
                              100000000, gpu_device);

    // =============================================================================
    // Checkpoint Management
    // =============================================================================
    CheckpointManager manager("checkpoints_gpt", "gpt_model", 5, 0, false, use_async_checkpointing);
    manager.register_signal_handler();
    manager.set_save_intervals(2); // Save every 50 steps

    int start_step = 0;
    float latest_loss = 0.0f;

    // Auto-resume if checkpoint exists
    if (manager.load_latest(model, optimizer, start_step, latest_loss)) {
      std::cout << "[Main] Resumed from step " << start_step
                << " (latest loss: " << latest_loss << ")" << std::endl;
      // start_step remains as is, loop will start from step (e.g. if saved at
      // 49, next is 50) Checkpointing manager saves the step it JUST completed.
    }

    std::cout << "\nStarting training..." << std::endl;

    // Create CSV log file
    std::ofstream log_file("benchmark_gpt_test.csv");
    log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
    log_file << std::fixed << std::setprecision(6);

    // Declare once here so the stop_requested path (before the inner reset) sees
    // the actual loss from the previous step, not a stale 0.0f from a shadow.
    float loss_accum = 0.0f;
    // Bug 2 fix: grad_scale is a constant GPU tensor — create it once before the
    // training loop (not as `static`) so its destructor runs before CUDA teardown.
    Tensor grad_scale =
        Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                     1.0f / static_cast<float>(grad_accum_steps));
    float val_loss_accum_log = -1.0f;
    for (int step = start_step; step < max_steps; ++step) {
      // Check for graceful stop request (Ctrl+C)
      if (CheckpointManager::stop_requested) {
        std::cout << "\n[Main] Stop requested. Saving checkpoint at step "
                  << step << "..." << std::endl;
        manager.save(step, model, optimizer, loss_accum);
        break;
      }

      auto t0 = std::chrono::high_resolution_clock::now();

      // Validation every 100 steps
      if (step % 100 == 0 || step == max_steps - 1) {
        val_loader.reset();
        float val_loss_accum = 0.0f;
        int val_loss_steps = 20;

        // Bug 3 fix: disable gradient tracking during validation — forward pass
        // would otherwise build and immediately discard a full autograd graph,
        // wasting GPU memory and compute on every validation batch.
        autograd::NoGradGuard no_grad;
        for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
          Batch batch = val_loader.next_batch();
          // Tensors already on GPU from dataloader — no .to(device) needed

          Tensor logits = model.forward(batch.input);
          Tensor loss =
              autograd::sparse_cross_entropy_loss(logits, batch.target);

          Tensor loss_cpu = loss.to_cpu();
          val_loss_accum +=
              loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
        }

        std::cout << "validation loss: " << std::fixed << std::setprecision(4)
                  << val_loss_accum << std::endl;
        val_loss_accum_log = val_loss_accum;
      }

      // Training step

      optimizer.zero_grad();
      // Bug 1 fix: was `float loss_accum = 0.0f` — re-declaring here shadowed the
      // outer variable, so the stop_requested path always saved loss=0.0f.
      loss_accum = 0.0f;

      // Accumulate loss on GPU to avoid per-micro-step CPU sync
      Tensor loss_accum_gpu =
          Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

      for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
        Batch batch = train_loader.next_batch();
        // Tensors already on GPU from dataloader — no .to(device) needed

        // Forward
        Tensor logits = model.forward(batch.input);
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);

        // Accumulate detached loss on GPU (no autograd graph, no CPU sync)
        loss_accum_gpu = loss_accum_gpu + loss.detach();

        // Backward with scaling
        loss.backward(&grad_scale);
      }

      // ONE sync after all micro-steps complete
      {
        Tensor loss_cpu = loss_accum_gpu.to_cpu();
        loss_accum =
            loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
      }

      // NaN detection - early exit if training goes unstable
      if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
        std::cerr << "ERROR: NaN/Inf detected in loss at step " << step
                  << std::endl;
        log_file.close();
        return 1;
      }

      // Clip gradients
      float norm = nn::clip_grad_norm_(params, 1.0f);
      // cudaDeviceSynchronize();

      // Update learning rate
      float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
      optimizer.set_lr(lr);

      // Optimizer step
      optimizer.step();

      auto t1 = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(t1 - t0).count();

      // Compute throughput
      int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
      double tokens_per_sec = static_cast<double>(tokens_processed) / dt;

      // Print training info
      std::cout << "step " << std::setw(5) << step << " | loss: " << std::fixed
                << std::setprecision(6) << loss_accum << " | lr "
                << std::scientific << std::setprecision(4) << lr
                << " | norm: " << std::fixed << std::setprecision(4) << norm
                << " | dt: " << std::fixed << std::setprecision(2)
                << (dt * 1000.0) << "ms" << " | tok/sec: " << std::fixed
                << std::setprecision(2) << tokens_per_sec << std::endl;

      // Log metrics to CSV
      log_file << step << "," << loss_accum << "," << val_loss_accum_log << ","
               << lr << "," << norm << "," << (dt * 1000.0) << ","
               << tokens_per_sec << "\n";
      log_file.flush();
      val_loss_accum_log = -1.0f; // Reset for next iteration
      // Periodic checkpointer step
      if (step % 2 == 0) {
          std::string range_name = use_async_checkpointing ? "checkpoint_staging_async" : "checkpoint_full_sync";
          nvtxRangePushA(range_name.c_str());
          manager.step(step, model, optimizer, loss_accum);
          nvtxRangePop();
      }
    }

    log_file.close();
    std::cout << "\nTraining log saved to: benchmark_gpt_test.csv" << std::endl;

    std::cout << "\n=== Training Complete ===" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}