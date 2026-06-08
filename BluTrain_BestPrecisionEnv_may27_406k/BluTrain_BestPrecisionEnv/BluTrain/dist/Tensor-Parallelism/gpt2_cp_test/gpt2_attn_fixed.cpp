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
#include "checkpointing/GradMode.h"
#include "autograd/operations/TrilOps.h"

#include "checkpointing/Checkpointing.h"
#include "device/CachingCudaAllocator.h"

// Dataloader
#include "DataLoader.h"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 768;
    int64_t n_layers = 6;
    int64_t n_heads = 12;        // GPT-2 default: 12 heads
    bool weight_tying = true;    // Flag for weight tying
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding : public nn::Module {
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

        register_parameter(weight);
    }

    // Forward: indices [B, T] -> embeddings [B, T, C]
    Tensor forward(const Tensor& indices) override {
        return autograd::embedding(weight, indices);
    }

private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};

// =============================================================================
// Helper: Initialize nn::Linear weights with GPT-2 style (std=0.02)
// =============================================================================

void init_linear_gpt2(nn::Linear& layer, float std = 0.02f, uint64_t seed = 1234, bool req_grad=true) {
    // IMPORTANT: Do NOT replace layer.weight with a new tensor!
    // nn::Linear already registered its weight in params_.
    // We must copy data INTO the existing weight to preserve parameter identity.

    auto shape = layer.weight.shape();
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);  // CPU, no grad
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

// =============================================================================
// Multi-Head Causal Self-Attention (FIXED)
// =============================================================================

class Attention : public nn::Module {
public:
    nn::LayerNorm ln;        // Pre-norm LayerNorm
    nn::Linear c_attn;       // QKV projection: [n_embd] -> [3 * n_embd]
    nn::Linear c_proj;       // Output projection: [n_embd] -> [n_embd]

    Attention(int64_t n_embd, int n_heads, int n_layers, DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd),
          c_attn(n_embd, 3 * n_embd, true),
          c_proj(n_embd, n_embd, true),
          n_embd_(n_embd),
          n_heads_(n_heads),
          head_dim_(n_embd / n_heads)
    {
        // GPT-2 style init for qkv projection
        init_linear_gpt2(c_attn, 0.02f, seed);

        // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(c_proj, 0.02f * scale, seed + 1);

        // Pre-compute attention scale on GPU once
        scale_ = Tensor::full(Shape{{1}}, TensorOptions().with_dtype(Dtype::Float32).with_device(device),
                              1.0f / std::sqrt(static_cast<float>(head_dim_)));

        ln.to(device);
        c_attn.to(device);
        c_proj.to(device);

        register_module(ln);
        register_module(c_attn);
        register_module(c_proj);
    }

    Tensor forward(const Tensor& x) override {
        int64_t B = x.shape().dims[0];
        int64_t T = x.shape().dims[1];
        int64_t C = x.shape().dims[2];

        // Pre-Norm
        Tensor h = ln.forward(x);

        // QKV Projection
        Tensor qkv = c_attn.forward(h);

        std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = inp[0];
        Tensor k = inp[1];
        Tensor v = inp[2];

        q = autograd::transpose( autograd::reshape(q, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
        k = autograd::transpose( autograd::reshape(k, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
        v = autograd::transpose( autograd::reshape(v, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);

        // Scaled Dot-Product Attention
        Tensor attn_weights = autograd::matmul( autograd::mul(q, scale_), autograd::transpose(k, -2, -1));

        float neg_inf = -std::numeric_limits<float>::infinity();
        Tensor masked = autograd::tril(attn_weights, 0, neg_inf);

        Tensor attn_probs = autograd::softmax(masked);

        Tensor attn_out = autograd::matmul(attn_probs, v);

        Tensor merged = autograd::reshape(
                            autograd::transpose(attn_out, 1, 2),
                            Shape({{B, T, C}}));

        Tensor proj = c_proj.forward(merged);

        // Residual connection
        return autograd::add(x, proj);
    }

private:
    int64_t n_embd_;
    int64_t n_heads_;
    int64_t head_dim_;
    Tensor scale_;  // Pre-computed 1/sqrt(head_dim), reused across forward calls
};

// =============================================================================
// MLP Block
// =============================================================================

class MLP : public nn::Module {
public:
    nn::LayerNorm ln;       // LayerNorm before MLP
    nn::Linear fc_up;       // Linear(n_embd, 4*n_embd)
    nn::Linear fc_down;     // Linear(4*n_embd, n_embd)

    MLP(int64_t n_embd, int n_layers, DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd),
          fc_up(n_embd, 4 * n_embd, true),
          fc_down(4 * n_embd, n_embd, true),
          n_embd_(n_embd)
    {
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
    Tensor forward(const Tensor& x) override {
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
// GPT Model
// =============================================================================

class GPT : public nn::Module {
public:
    GPTConfig config;
    Embedding wte;  // Token embedding [vocab_size, n_embd]
    Embedding wpe;  // Position embedding
    std::vector<std::shared_ptr<Attention>> attn_blocks;
    std::vector<std::shared_ptr<MLP>> mlp_blocks;
    nn::LayerNorm ln_f; // Final LayerNorm
    std::shared_ptr<nn::Linear> lm_head;  // Output projection [n_embd, vocab_size], bias=False
                                            // When weight_tying: shares wte.weight (transposed view)
                                            // When no weight_tying: independent weight

    GPT(GPTConfig cfg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg),
          wte(cfg.vocab_size, cfg.n_embd, device, seed),
          wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);

        // Create Attention + MLP blocks (interleaved per layer)
        for (int i = 0; i < cfg.n_layers; ++i) {
            auto a = std::make_shared<Attention>(cfg.n_embd, cfg.n_heads, cfg.n_layers, device, seed + 200 + i * 10);
            auto m = std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device, seed + 200 + i * 10);
            attn_blocks.push_back(a);
            mlp_blocks.push_back(m);
            register_module(a.get());
            register_module(m.get());
        }

        // Initialize lm_head
        if (config.weight_tying) {
            // Weight tying: create lm_head with default constructor, then
            // assign its weight to a transposed view of wte.weight.
            // NoGradGuard prevents TransposeBackward from being attached,
            // so the view is a LEAF tensor. MatmulBackward sends gradients
            // directly to AccumulateGrad(view) — no TransposeBackward overhead.
            lm_head = std::make_shared<nn::Linear>();
            {
                autograd::NoGradGuard no_grad;
                lm_head->weight = wte.weight.transpose(0, 1);  // [n_embd, vocab_size] leaf view
            }
            lm_head->weight.set_requires_grad(true);  // Ensure it's tracked by autograd
        } else {
            // Independent lm_head with its own weight
            lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
            init_linear_gpt2(*lm_head, 0.02f, seed + 1000, true);
            lm_head->to(device);
        }

        // Optimization: cache position tensor once (avoids re-creating + H2D transfer every forward)
        Tensor pos_cpu(Shape{{1, cfg.context_length}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* pos_data = pos_cpu.data<int64_t>();
        for (int64_t i = 0; i < cfg.context_length; ++i) {
            pos_data[i] = i;
        }
        cached_pos_ = pos_cpu.to(device);

        register_module(wte);
        register_module(wpe);
        // attn_blocks and mlp_blocks already registered in the loop above
        register_module(ln_f);
        if (!config.weight_tying && lm_head) {
            register_module(lm_head.get());
        }
    }

    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    Tensor forward(const Tensor& idx) override {
        // Get embeddings [B, T, C]
        // std::cout << "Started GPT forward" << std::endl;
        // auto stats = CachingCUDAAllocator::instance().get_stats();
        // std::cout << "Stats reserved before: " << stats.allocated_current / (1024 * 1024) << std::endl;
        Tensor tok_emb = wte.forward(idx);      // [B, T, C]

        int64_t T = idx.shape().dims[1];
        // std::cout << "Token embedding forward completed" << std::endl;
        // Slice position indices to match T
        Tensor pos_flat = autograd::reshape(cached_pos_, Shape({{config.context_length}}));
        Tensor pos_sliced = pos_flat.slice(0, T);
        Tensor pos_indices = autograd::reshape(pos_sliced, Shape({{1, T}}));
        // std::cout << "Pos embedding forward started" << std::endl;
        Tensor pos_emb = wpe.forward(pos_indices);  // [1, T, C] - broadcasts
        // std::cout << "Pos embedding forward completed" << std::endl;
        // Add embeddings
        Tensor x = autograd::add(tok_emb, pos_emb);
        // std::cout << "add embedding completed" << std::endl;
        // Transformer blocks: interleave Attention + MLP per layer
        for (int i = 0; i < config.n_layers; ++i) {
            // std::cout << "Transformer forward start: " << i << std::endl;
            x = attn_blocks[i]->forward(x);  // pre-norm + multi-head attention + residual
            x = mlp_blocks[i]->forward(x);   // pre-norm + FFN + residual
            // std::cout << "Transformer forward completed: " << i << std::endl;
        }
        // std::cout << "final norm started "<< std::endl;
        // Final normalization
        x = ln_f.forward(x);
        // std::cout << "final norm ended "<< std::endl;
        // std::cout << "lm_head started "<< std::endl;

        // Output projection
        // std::cout << "Came Here" << std::endl;
        Tensor logits = lm_head->forward(x);  // [B, T, vocab_size]
        // std::cout << "Came Here twice" << std::endl;
        // std::cout << "lm_head completed "<< std::endl;
        // std::cout << "Stats reserved after: " << stats.allocated_current / (1024 * 1024) << std::endl;


        return logits;
    }

private:
    Tensor cached_pos_;  // [1, T] position indices, cached on GPU
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float MAX_LR, float MIN_LR, int WARMUP_STEPS, int MAX_STEPS) {
    if (step < WARMUP_STEPS) {
        return MAX_LR * static_cast<float>(step + 1) / static_cast<float>(WARMUP_STEPS);

    }
    if (step > MAX_STEPS) {
        return MIN_LR;
    }
    float decay_ratio = static_cast<float>(step - WARMUP_STEPS) / static_cast<float>(MAX_STEPS - WARMUP_STEPS);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return MIN_LR + coeff * (MAX_LR - MIN_LR);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main() {
    try {
        std::cout << "=== GPT-2 Training Script (Fixed Attention) ===" << std::endl;

        // Configuration
        GPTConfig config;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 768;
        config.n_layers = 6;
        config.n_heads = 12;       // Proper multi-head attention
        config.weight_tying = true; // Toggle weight tying here

        // Training hyperparameters
        const int B = 4;           // Batch size
        const int T = 1024;        // Sequence length
        const int GLOBAL_BATCH = 65536;  // Global batch size
        const int GRAD_ACCUM_STEPS = GLOBAL_BATCH / (B * T);

        const float MAX_LR = 6e-4f;
        const float MIN_LR = MAX_LR * 0.1f;
        const int WARMUP_STEPS = 2501;
        const int MAX_STEPS = 25010;

        const int VAL_FREQ = 1000;
        const int TOK_GEN_FREQ = 10000;
        const int CKPT_FREQ = 5000;

        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_heads: " << config.n_heads << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  head_dim: " << (config.n_embd / config.n_heads) << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  GLOBAL_BATCH: " << GLOBAL_BATCH << std::endl;
        std::cout << "  GRAD_ACCUM_STEPS: " << GRAD_ACCUM_STEPS << std::endl;
        std::cout << "  Weight Tying: " << (config.weight_tying ? "ENABLED" : "DISABLED") << std::endl;

        // Set device - GPU-0 for training
        int gpu_device = 0;  // Use GPU-0
        int rank = 0;        // Rank for dataloader (0 for single-GPU training)
        int world_size = 1;  // Single GPU
        DeviceIndex device(Device::CUDA, gpu_device);
        cudaSetDevice(gpu_device);

        std::cout << "\nInitializing model on CUDA device " << gpu_device << "..." << std::endl;

        // Create model
        GPT model(config, device);

        // Print parameter count
        auto params = model.parameters();
        int64_t num_params = 0;
        for(auto& p : params) num_params += p.numel();

        std::cout << "Number of parameters: " << num_params << std::endl;
        if (!config.weight_tying) {
            std::cout << "(Note: More params than weight-tied version due to separate lm_head)" << std::endl;
        }

        // Create optimizer
        nn::AdamW optimizer(params, MAX_LR, 0.9f, 0.95f, 1e-8f, 0.1f);

        // Create data loaders
        std::string data_root = "/home/blubridge-035/Desktop/Backup/parallelism/data/edufineweb";
        DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, true, 100000000, gpu_device);
        DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true, 100000000, gpu_device);

        std::cout << "\nStarting training..." << std::endl;

        CheckpointManager ckpt_manager("checkpoints", "gpt2", 5);


        ckpt_manager.set_save_intervals(CKPT_FREQ); // Save every CKPT_FREQ steps

        int start_step = 0;
        float latest_loss = 0.0f;

        // Auto-resume if checkpoint exists
        if (ckpt_manager.load_latest(model, optimizer, start_step, latest_loss)) {
            std::cout << "[Resume] Continuing from step " << start_step << " with loss " << latest_loss << std::endl;

            // Re-align dataloader: skip all batches consumed in steps 0...start_step
            size_t batches_to_skip = static_cast<size_t>(start_step + 1) * GRAD_ACCUM_STEPS;
            std::cout << "[Resume] Skipping " << batches_to_skip << " batches..." << std::endl;
            train_loader.skip_batches(batches_to_skip);

            start_step++;
        }

        // Create CSV log file
        std::ofstream log_file("../Training_Logs/master1_best_bigmodel_prompt.csv");
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);

        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step

        for (int step = start_step; step < MAX_STEPS; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // Validation every VAL_FREQ steps
            if (step % VAL_FREQ == 0 || step == MAX_STEPS - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;

                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    // Tensors already on GPU from dataloader — no .to(device) needed

                    Tensor logits = model.forward(batch.input);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);

                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }

                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }

            // token generation
            if(step % TOK_GEN_FREQ == 0 || step == MAX_STEPS - 1) {
                std::cout << "--- Generating tokens at step " << step << " ---" << std::endl;
                int num_return_sequence = 4;
                int max_length = 60;

                Tensor xgen = Tensor(Shape({{num_return_sequence, 4}}),
                                           TensorOptions().with_dtype(Dtype::Int64).with_device(device));
                std::vector<int64_t> xgen_tokens = {17, 10, 17, 28,17, 10, 17, 28,17, 10, 17, 28,17, 10, 17, 28};
                xgen.set_data(xgen_tokens);

                uint64_t gen_seed = 42 + rank;
                while (xgen.shape().dims[1] < max_length) {
                    Tensor logits;
                    logits = model.forward(xgen);

                    int64_t B = logits.shape().dims[0];
                    int64_t T = logits.shape().dims[1];
                    int64_t V = logits.shape().dims[2];

                    // Select last time step: logits[:, -1, :] -> (B, V)
                    Tensor gather_idx = Tensor::full(Shape({{B, 1, V}}),
                                                     TensorOptions().with_dtype(Dtype::Int64).with_device(device),
                                                     static_cast<float>(T - 1));

                    Tensor last_logits_3d = OwnTensor::gather(logits, 1, gather_idx);

                    Tensor scaled_logits = last_logits_3d;
                    Tensor probs = OwnTensor::autograd::softmax(scaled_logits, -1);

                    // topk(50, -1) -> (B, 1, 50)
                    auto topk_res = probs.topk(50, -1);
                    Tensor topk_probs = topk_res.first;   // (B, 1, 50)
                    Tensor topk_indices = topk_res.second; // (B, 1, 50)

                    Tensor topk_probs_2d = OwnTensor::autograd::reshape(topk_probs, Shape({{B, 50}}));
                    // topk_probs_2d.display();

                    Tensor ix = Tensor::multinomial(topk_probs_2d, 1, /*replacement=*/false,
                                                     /*seed=*/gen_seed++); // (B, 1)

                    Tensor topk_indices_2d = OwnTensor::autograd::reshape(topk_indices, Shape({{B, 50}}));

                    Tensor next_token = OwnTensor::gather(topk_indices_2d, 1, ix);

                    xgen = Tensor::cat({xgen, next_token}, 1);
                }

                // Print generated tokens
                Tensor xgen_cpu = xgen.to_cpu();
                int64_t* data = xgen_cpu.data<int64_t>();
                int64_t B = xgen.shape().dims[0];
                int64_t T = xgen.shape().dims[1];

                for (int i = 0; i < B; ++i) {
                    std::cout << "sample" << i << "= \"";
                    for (int j = 0; j < T; ++j) {
                        std::cout << data[i * T + j] << " ";
                    }
                    std::cout << "\""<<std::endl;
                }
            }



            // Training step

            optimizer.zero_grad();
            // When weight tying, lm_head->weight is NOT registered as a parameter,
            // so we must zero its gradient manually
            if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
                model.lm_head->weight.zero_grad();
            }
            float loss_accum = 0.0f;

            // Cache grad_scale outside the loop — same value every micro-step
            static Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                                     1.0f / static_cast<float>(GRAD_ACCUM_STEPS));

            // Accumulate loss on GPU to avoid per-micro-step CPU sync
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

            for (int micro_step = 0; micro_step < GRAD_ACCUM_STEPS; ++micro_step) {
                Batch batch = train_loader.next_batch();

                // Forward
                Tensor logits = model.forward(batch.input);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);

                // Accumulate detached loss on GPU (no autograd graph, no CPU sync)
                loss_accum_gpu = loss_accum_gpu + loss.detach();

                // Backward with scaling
                loss.backward(&grad_scale);
            }

            // Weight tying: manually accumulate lm_head's gradient into wte.weight
            // lm_head->weight.grad is [n_embd, vocab_size] — transpose to [vocab_size, n_embd]
            // and add to wte.weight.grad so the optimizer sees the combined gradient
            if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
                Tensor lm_grad_T = model.lm_head->weight.grad_view().transpose(0, 1).contiguous();
                Tensor wte_grad = model.wte.weight.grad_view();  // [vocab, embd]
                // In-place accumulate: wte_grad += lm_grad_transposed
                wte_grad += lm_grad_T;
            }

            // ONE sync after all micro-steps complete
            {
                Tensor loss_cpu = loss_accum_gpu.to_cpu();
                loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
            }

            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }


            // Clip gradients
            float norm = nn::clip_grad_norm_(params, 1.0f);

            // Update learning rate
            float lr = get_lr(step, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS);
            optimizer.set_lr(lr);

            // Optimizer step
            optimizer.step();

            // Checkpointing
            if(step == MAX_STEPS - 2){
                ckpt_manager.save(step, model, optimizer, loss_accum);
            }

            ckpt_manager.step(step, model, optimizer, loss_accum);

            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();

            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS;
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


        std::cout << "\n=== Training Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
