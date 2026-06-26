        
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
#include <filesystem>

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/optimizer/Optim.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "dnn/DistributedNN.h"


// Dataloader
#include "Data_Loader/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

int rank, world_size;

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void start_timer() { cudaEventRecord(start); }
    float get_elapsed_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    double get_elapsed_seconds() {
        return get_elapsed_ms() / 1000.0;
    }
};

struct GPTConfig {
    int64_t batch_size = 8;
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 384;
    int64_t n_layers = 3;
    bool weight_tying = false; // Added
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
    // Standard forward with gradient tracking through embedding
    Tensor forward(const Tensor& indices) override {
        // Use autograd-aware embedding function for proper gradient flow
        return autograd::embedding(weight, indices);
    }

    void clear_own_params() { params_.clear(); }

private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};

// =============================================================================
// MLP Block
// =============================================================================

// Helper: Initialize nn::Linear weights with GPT-2 style (std=0.02)
// Creates tensors directly on the target device
// void init_linear_gpt2(dnn::DColumnLinear& layer, DeviceIndex device, float std = 0.02f, uint64_t seed = 1234) {
//     TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
//                                         .with_device(device)
//                                         .with_req_grad(true);
//     auto shape = layer.weight.shape();
//     layer.weight = Tensor::randn<float>(shape, opts, seed, std);
//     if (layer.bias.is_valid()) {
//         layer.bias = Tensor::zeros(layer.bias.shape(), opts);
//         // layer.bias = Tensor::randn<float>(layer.bias.shape(), opts, seed, std);
//     }
//     // std::cout << "WEIGHT: " << std::endl;
//     // layer.weight.display();
//     // std::cout << "BIAS: " << std::endl;
//     // layer.bias.display();
// }

class MLP : public dnn::DModule {
public:
    nn::LayerNorm ln;       // LayerNorm before MLP
    dnn::DColumnLinear fc_up;       // Linear(n_embd, 4*n_embd)
    dnn::DRowLinear fc_down;     // Linear(4*n_embd, n_embd)
    dnn::DGeLU gelu;

    MLP(GPTConfig config, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg,  DeviceIndex device, uint64_t seed = 1234)
        : ln(config.n_embd),
          // Correct Arguments: weight_data, use_bias, sd, seed, sync_input, use_backward_hook
          fc_up(mesh, pg, config.batch_size, config.context_length, config.n_embd, 4 * config.n_embd, {}, true, 0.02f, seed, false, false),

          // DRowLinear: weight_data, use_bias, sd, seed, sync_output, with_autograd
          fc_down(mesh, pg, config.batch_size, config.context_length,  4 * config.n_embd, config.n_embd, {}, true, 0.02f * (1.0f / std::sqrt(2.0f * static_cast<float>(config.n_layers))), seed, false, false)

    {
        // GPT-2 style initialization - create tensors directly on target device
        // init_linear_gpt2(fc_up, device, 0.02f, seed);



        // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
        // float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(config.n_layers));
        // init_linear_gpt2(fc_down, device, 0.02f * scale, seed + 1);

        // Move LayerNorm to device
        ln.to(device);

        register_module(ln);
        register_module(fc_up);
        register_module(fc_down);
    }

    using dnn::DModule::register_module;

    void register_module(nn::LayerNorm& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }

    // Forward: x [B, T, C] -> [B, T, C]
    DTensor forward( DTensor& x) {

        // Backward-only sync via Hook is now handled by fc_up (use_backward_hook=true)

        // Pre-Norm: ln(x)
        DTensor h(x.get_device_mesh(), x.get_pg(), x.get_layout(), "h_intermediate");
        h.mutable_tensor() = ln.forward(x.mutable_tensor());

        // Up projection + GELU + Down projection
        h = fc_up.forward(h);
        h = gelu.forward(h);
        h = fc_down.forward(h);

        // Residual connection: x + MLP(x)
        h.mutable_tensor() = autograd::add(x.mutable_tensor(), h.mutable_tensor());
        return h;
    }

private:
    int64_t n_embd_;
};

// =============================================================================
// GPT Model (WITHOUT Weight Tying)
// =============================================================================

class GPT : public dnn::DModule {
public:
    GPTConfig config;
    DeviceMesh &mesh;
    std::shared_ptr<ProcessGroupNCCL> pg_ptr; // Store pg for forward layout
    Embedding wte;  // Token embedding
    Embedding wpe;  // Position embedding
    dnn::DSequential mlps;
    nn::LayerNorm ln_f; // Final LayerNorm
    std::shared_ptr<nn::Linear> lm_head;  // Output projection [n_embd, vocab_size], bias=False
    // Tensor& W_out;
    // Tensor pos;
    Tensor logits;
    Tensor cached_pos_; // Position indices cached on GPU

    // Component timing (accumulated per step, reset after printing)
    double t_tok_emb = 0, t_pos_emb = 0, t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
    CudaTimer timer_tok_emb, timer_pos_emb, timer_mlp, timer_ln_f, timer_lm_head;

    GPT(GPTConfig cfg, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg),
          mesh(mesh),
          pg_ptr(pg),
          wte(cfg.vocab_size, cfg.n_embd, device, seed = 1234),
          wpe(cfg.context_length, cfg.n_embd, device),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);


        // Create MLP blocks and add to Sequential
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.add(std::make_shared<MLP>(config, mesh, pg, device, 1234));
        }


        if (!config.weight_tying) {
            lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
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
        register_module(mlps);
        register_module(ln_f);
        if (lm_head)
            register_module(*lm_head);

        // Initialize W_out
        // TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
        //                                   .with_device(device)
        //                                   .with_req_grad(true);
        // W_out = Tensor::randn<float>(Shape{{config.n_embd, config.vocab_size}}, opts, seed + 500, 0.02f);
        // W_out = wte.weight.t();
        // register_parameter(&W_out);
    }

    // using dnn::DModule::register_module;
    // void register_module(nn::LayerNorm& m) {
    //     register_parameter(&m.weight);
    //     if(m.bias.is_valid()) register_parameter(&m.bias);zz
    // void register_module(Embedding& m) {
    //     register_parameter(&m.weight);
    // }

    using dnn::DModule::register_module;

    void register_module(Embedding& m) {
        register_parameter(&m.weight);
    }

    void register_module(nn::LayerNorm& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }

    void register_module(nn::Linear& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }

    void reset_timing() {
        t_tok_emb = t_pos_emb = t_mlp = t_ln_f = t_lm_head = 0;
    }

    void print_timing(int rank) {
        if (rank == 0) {
            std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1) << (t_tok_emb * 1000.0) << "ms"
                      << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                      << " | mlps: " << (t_mlp * 1000.0) << "ms"
                      << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                      << " | lm_head: " << (t_lm_head * 1000.0) << "ms"
                      << std::endl;
        }
    }

    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    Tensor forward(Tensor& idx) {
        auto shape = idx.shape().dims;
        int64_t B = shape[0];
        int64_t T = shape[1];

        // --- Token Embedding ---
        timer_tok_emb.start_timer();
        Tensor tok_emb = wte.forward(idx);  // [B, T, C]

        t_tok_emb += timer_tok_emb.get_elapsed_seconds();

        // --- Position Embedding ---
        timer_pos_emb.start_timer();
        Tensor pos_flat = autograd::reshape(cached_pos_, Shape({{config.context_length}}));
        Tensor pos_sliced = pos_flat.slice(0, T);
        Tensor pos_indices = autograd::reshape(pos_sliced, Shape({{1, T}}));

        Tensor pos_emb = wpe.forward(pos_indices);  // [1, T, C] - broadcasts

        t_pos_emb += timer_pos_emb.get_elapsed_seconds();

        // Add embeddings — create DTensor with actual B,T dimensions
        Layout input_layout(mesh, std::vector<int64_t>{B, T, config.n_embd});
        DTensor current_x(mesh, pg_ptr, input_layout, "x_combined");
        current_x.mutable_tensor() = autograd::add(tok_emb, pos_emb);

        current_x.register_backward_node();

        // --- MLP Blocks ---
        timer_mlp.start_timer();
        current_x = mlps.forward(current_x);

        t_mlp += timer_mlp.get_elapsed_seconds();

        current_x.sync();

        // --- Final LayerNorm ---
        timer_ln_f.start_timer();
        current_x.mutable_tensor() = ln_f.forward(current_x.mutable_tensor());

        t_ln_f += timer_ln_f.get_elapsed_seconds();

        // --- LM Head ---
        timer_lm_head.start_timer();

        if (lm_head) {
            logits = lm_head->forward(current_x.mutable_tensor());
        } else {
            // Weight tying: use wte.weight^T as output projection
            logits = autograd::matmul(current_x.mutable_tensor(), wte.weight.t());
        }

        t_lm_head += timer_lm_head.get_elapsed_seconds();

        return logits;
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
// Tiktoken Decoder (via Python popen)
// =============================================================================

std::string decode_tokens_tiktoken(const std::vector<int64_t>& tokens) {
    std::string cmd = "python3 -c \"import tiktoken; enc = tiktoken.get_encoding('gpt2'); print(enc.decode([";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) cmd += ",";
        cmd += std::to_string(tokens[i]);
    }
    cmd += "]))\" 2>/dev/null";

    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
    }
    if (!result.empty() && result.back() == '\n') result.pop_back();
    return result;
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(rank == 0){
        std::cout << "=== GPT-2 Tensor Parallel Training Script ===" << std::endl;
    }
    try {

        // Configuration
        GPTConfig config;
        config.batch_size = 8;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 384;
        config.n_layers = 3;
        config.weight_tying = true;

        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int global_batch = 65536;  // Global batch size
        const int grad_accum_steps = global_batch / (B * T);

        const float max_lr = 1e-4f;
        const float min_lr = max_lr * 0.1f;
        const int VAL_FREQ = 100;
        const int TOK_GEN_FREQ = 1000;

        if(rank == 0) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  global_batch: " << global_batch << std::endl;
        std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
        std::cout << "  Weight Tying: " << (config.weight_tying ? "ENABLED" : "DISABLED") << std::endl;
        }
        // Set device - GPU-0 for training
        // int gpu_device = 0;  // Use GPU-0
        // int rank = 0;        // Rank for dataloader (0 for single-GPU training)
        // int world_size = 2;


        DeviceIndex device(Device::CUDA, rank);
        cudaSetDevice(rank);

        std::vector<int> ranks_vec(world_size);
        for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
        DeviceMesh mesh({world_size}, ranks_vec);
        auto pg = mesh.get_process_group(0);

        std::cout << "\nInitializing model on CUDA device "<< device.index << "..." << std::endl;

        // Create model
        GPT model(config, mesh, pg, device);

        // Print parameter count
        std::vector<DTensor*> params = model.parameters();
        int64_t num_params = 0;
        for(auto& p : params) num_params += p->mutable_tensor().numel();
        const int max_steps = num_params / global_batch * 5;
        // const int max_steps = 1;
        const int warmup_steps = max_steps / 10;
        // const int warmup_steps = 0;

        if(rank == 0){
            std::cout << "Number of parameters: " << num_params << std::endl;
            std::cout << "Number of steps: " << max_steps << std::endl;
            std::cout << "Number of warmup_steps: " << warmup_steps << std::endl;
        }

        // std::cout << "(Note: More params than weight-tied version due to separate W_out)" << std::endl;

        // Get all parameters
        // auto params = model.parameters(); // Already got above

        // Create optimizer
        dnn::AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);

        // Create data loaders
        std::string data_root = "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data/";
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true, 100000000);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true, 100000000);

        CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim;

        if(rank == 0){
            std::cout << "\nStarting training..." << std::endl;
        }

        // Create CSV log file
        // Enable dynamic log filename generation
        std::string log_filename;
        std::string config_filename;
        std::ofstream log_file;

        if (rank == 0) {
            std::filesystem::create_directories("TP_MLP_Training_logs");
            int log_idx = 1;
            while (true) {
                log_filename = "TP_MLP_Training_logs/TP_MLP_Training_log" +
                               std::to_string(log_idx) + ".csv";
                bool exists = std::filesystem::exists(log_filename);

                if (!exists) {
                  break;
                }
                log_idx++;
            }
            std::cout << "Saving logs to: " << log_filename << std::endl;

            // Save configuration
            config_filename = "TP_MLP_Training_logs/TP_MLP_Training_log" + std::to_string(log_idx) + "_config.txt";
            std::ofstream config_file(config_filename);
            config_file << "Configuration:\n";
            config_file << "  Batch_size: " << B << "\n";
            config_file << "  context_length: " << config.context_length << "\n";
            config_file << "  n_embd: " << config.n_embd << "\n";
            config_file << "  vocab_size: " << config.vocab_size << "\n";
            config_file << "  n_layers: " << config.n_layers << "\n";
            config_file << "  global_batch: " << global_batch << "\n";
            config_file << "  grad_accum_steps: " << grad_accum_steps << "\n";
            config_file << "  Number of parameters: " << num_params << "\n";
            config_file << "  Max Learning Rate: " << max_lr << "\n";
            config_file << "  Min Learning Rate: " << min_lr << "\n";
            config_file << "  Number of steps: " << max_steps << "\n";
            config_file << "  Number of warmup_steps: " << warmup_steps << "\n";
            config_file.close();

            log_file.open(log_filename);
            if (!log_file.is_open()) {
                std::cerr << "ERROR: Could not open log file " << log_filename << " for writing!" << std::endl;
                std::exit(1);
            }
            log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim, timer_tok_emb, timer_pos_emb, timer_mlp, timer_ln_f, timer_lm_head \n";
            log_file << std::fixed << std::setprecision(6);
        }

        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step


        for (int step = 0; step < max_steps; ++step) {
            try {
            timer_step.start_timer();

            // Validation every VAL_FREQ steps
            if (step % VAL_FREQ == 0 || step == max_steps - 1) {
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

            // token generation (skip step 0 — random weights produce NaN in softmax,
            // and a rank-0-only exception would deadlock NCCL collectives)
            if(step > 0 && (step % TOK_GEN_FREQ == 0 || step == max_steps - 1)) {
                if (rank == 0) std::cout << "--- Generating tokens at step " << step << " ---" << std::endl;
                int num_return_sequence = 4;
                int max_length = 60;

                Tensor xgen = Tensor(Shape({{num_return_sequence, 8}}),
                                           TensorOptions().with_dtype(Dtype::Int64).with_device(device));
                std::vector<int64_t> xgen_tokens = {15496, 11, 314, 1101, 281, 9552, 2746, 11,15496, 11, 314, 1101, 281, 9552, 2746, 11,15496, 11, 314, 1101, 281, 9552, 2746, 11,15496, 11, 314, 1101, 281, 9552, 2746, 11};
                xgen.set_data(xgen_tokens);

                while (xgen.shape().dims[1] < max_length) {
                    Tensor logits = model.forward(xgen);

                    int64_t Bg = logits.shape().dims[0];
                    int64_t Tg = logits.shape().dims[1];
                    int64_t Vg = logits.shape().dims[2];

                    Tensor gather_idx = Tensor::full(Shape({{Bg, 1, Vg}}),
                                                     TensorOptions().with_dtype(Dtype::Int64).with_device(device),
                                                     static_cast<float>(Tg - 1));

                    Tensor last_logits_3d = OwnTensor::gather(logits, 1, gather_idx);
                    Tensor probs = OwnTensor::autograd::softmax(last_logits_3d, -1);

                    auto topk_res = probs.topk(50, -1);
                    Tensor topk_probs = topk_res.first;
                    Tensor topk_indices = topk_res.second;

                    Tensor topk_probs_2d = OwnTensor::autograd::reshape(topk_probs, Shape({{Bg, 50}}));
                    Tensor ix;
                    if (rank == 0) {
                        try {
                            ix = Tensor::multinomial(topk_probs_2d, 1);
                        } catch (...) {
                            // Fallback: pick index 0 (most probable from topk) if probs are NaN/Inf
                            ix = Tensor::zeros(Shape({{Bg, 1}}), TensorOptions().with_dtype(Dtype::Int64).with_device(device));
                        }
                    } else {
                        // allocate ix on other ranks for broadcast
                        ix = Tensor::zeros(Shape({{Bg, 1}}), TensorOptions().with_dtype(Dtype::Int64).with_device(device));
                    }

                    // Sync index to avoid divergence in MP environment
                    if (world_size > 1) {
                        ix = ix.contiguous();
                        pg->broadcast(ix.data<int64_t>(), ix.data<int64_t>(), ix.numel(), OwnTensor::Dtype::Int64, 0, true);
                    }

                    Tensor topk_indices_2d = OwnTensor::autograd::reshape(topk_indices, Shape({{Bg, 50}}));
                    Tensor next_token = OwnTensor::gather(topk_indices_2d, 1, ix);

                    xgen = Tensor::cat({xgen, next_token}, 1);
                }

                if (rank == 0) {
                    Tensor xgen_cpu = xgen.to_cpu();
                    int64_t* data = xgen_cpu.data<int64_t>();
                    int64_t Bg = xgen.shape().dims[0];
                    int64_t Tg = xgen.shape().dims[1];

                    // Append generated tokens to config file
                    std::ofstream config_append(config_filename, std::ios::app);
                    config_append << "\n--- Generated Tokens at Step " << step << " ---\n";

                    for (int i = 0; i < Bg; ++i) {
                        // Collect token IDs into a vector and a printable string
                        std::vector<int64_t> sample_tokens;
                        std::string token_str;
                        for (int j = 0; j < Tg; ++j) {
                            sample_tokens.push_back(data[i * Tg + j]);
                            if (j > 0) token_str += " ";
                            token_str += std::to_string(data[i * Tg + j]);
                        }

                        // Print raw IDs
                        std::cout << "Sample " << i << " [IDs]: " << token_str << std::endl;

                        // Decode via tiktoken popen
                        std::string decoded = decode_tokens_tiktoken(sample_tokens);
                        std::cout << "Sample " << i << " [Text]: " << decoded << std::endl;

                        // Log both to config file
                        config_append << "Sample " << i << " [IDs]: " << token_str << "\n";
                        config_append << "Sample " << i << " [Text]: " << decoded << "\n";
                    }
                    config_append.close();
                }
            }

            // Training step with component timing
            double time_data = 0, time_forward = 0, time_loss = 0, time_backward = 0, time_allreduce = 0, time_clip = 0, time_optim = 0;

            optimizer.zero_grad();
            float loss_accum = 0.0f;

            // Optimized: Accumulate loss on GPU to avoid CPU syncs
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

            static Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                                     1.0f / static_cast<float>(grad_accum_steps));

            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {

                // --- Data Loading ---
                timer_data.start_timer();
                Batch batch = train_loader.next_batch();
                Tensor x = batch.input.to(device).as_type(Dtype::Int64);
                Tensor y = batch.target.to(device).as_type(Dtype::Int64);

                time_data += timer_data.get_elapsed_seconds();

                // --- Forward Pass ---
                timer_fwd.start_timer();
                Tensor logits = model.forward(x);

                time_forward += timer_fwd.get_elapsed_seconds();

                // --- Loss Computation ---
                timer_loss.start_timer();
                Tensor loss = autograd::sparse_cross_entropy_loss(logits,y);

                loss_accum_gpu = loss_accum_gpu + loss.detach();

                time_loss += timer_loss.get_elapsed_seconds();

                // --- Backward Pass ---
                timer_bwd.start_timer();
                loss.backward(&grad_scale);
                time_backward += timer_bwd.get_elapsed_seconds();
            }


        // TEMP DEBUGGING: Print parameter gradients to check cross-rank sync
            // if ((grad_accum_steps + 1) % 1 == 0) {
            //   for (int r = 0; r < world_size; ++r) {
            //     if (rank == r) {
            //       std::cout << "\n=== DEBUG: Parameter Gradients at Step " << step
            //                 << " [Rank " << rank << "] ===" << std::endl;
            //       for (auto *p : params) {
            //         if (p->mutable_tensor().has_grad()) {
            //           std::cout << "\nParam: " << p->name()
            //                     << " | size: " << p->mutable_tensor().numel()
            //                     << std::endl;
            //           try {
            //             p->mutable_tensor().display();
            //           } catch (const std::exception &e) {
            //             std::cout << "  Error displaying grad: " << e.what()
            //                       << std::endl;
            //           }
            //         } else {
            //           std::cout << "Param: " << p->name() << " | NO GRADIENT"
            //                     << std::endl;
            //         }
            //       }
            //       std::cout << "=================================================\n"
            //                 << std::endl;
            //     }
            //     MPI_Barrier(MPI_COMM_WORLD);
            //   }
            // }


            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);

            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }

            // --- Gradient Clipping ---
            timer_clip.start_timer();
            float norm = dnn::clip_grad_norm_dtensor_nccl(params, 1.0f, pg);

            time_clip = timer_clip.get_elapsed_seconds();

            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);

            // --- Optimizer Step ---
            timer_optim.start_timer();
            optimizer.step(params);
            time_optim = timer_optim.get_elapsed_seconds();

            double dt = timer_step.get_elapsed_seconds();

            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
            long long total_sec = static_cast<long long>((max_steps - step) * dt);

           int h = total_sec / 3600;
           int m = (total_sec % 3600) / 60;
           int s = total_sec % 60;



            // Print training info
        if(rank == 0){
            std::cout << "step " << std::setw(5) << step
                        << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                        << " | lr " << std::scientific << std::setprecision(4) << lr
                        << " | norm: " << std::fixed << std::setprecision(4) << norm
                        << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                        << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                        << " |       Time Left: " << std::setfill('0') << std::setprecision(2) << h << " hrs : "<< m <<" mins "
                        << std::endl;


            // Component timing breakdown (in ms)
            std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1) << (time_data * 1000.0) << "ms"
                      << " | fwd: " << (time_forward * 1000.0) << "ms"
                      << " | loss: " << (time_loss * 1000.0) << "ms"
                      << " | bwd: " << (time_backward * 1000.0) << "ms"
                      << " | clip: " << (time_clip * 1000.0) << "ms"
                      << " | optim: " << (time_optim * 1000.0) << "ms"
                    //   << " | gradients allreduce: " << std::fixed << std::setprecision(1) << (time_allreduce * 1000.0) << "ms"
                      << std::endl;

            // Layer-level timing breakdown
            model.print_timing(rank);

            // Log metrics to CSV
            log_file << step << ","
            << loss_accum << ","
            << val_loss_accum_log << ","
            << lr << ","
            << norm << ","
            << (dt * 1000.0) << ","
            << tokens_per_sec << ","
            << (time_data * 1000.0) << ","
            << (time_forward * 1000.0) << ","
            << (time_loss * 1000.0) << ","
            << (time_backward * 1000.0) << ","
            << (time_clip * 1000.0) << ","
            << (time_optim * 1000.0) << ","
            << (model.t_tok_emb * 1000.0) << ","
            << (model.t_pos_emb * 1000.0) << ","
            << (model.t_mlp * 1000.0) << ","
            << (model.t_ln_f * 1000.0) << ","
            << (model.t_lm_head * 1000.0) << "\n";

            log_file.flush();
            model.reset_timing();
        }
        val_loss_accum_log = -1.0f;  // Reset for next iteration
            } catch (const std::exception& e) {
                std::cerr << "EXCEPTION IN RANK " << rank << " AT STEP " << step << ": " << e.what() << std::endl;
                std::exit(1);
            }
    }
    if(rank == 0){
            log_file.close();
            std::cout << "\nTraining log saved to: " << log_filename << std::endl;

            std::cout << "\n=== Training Complete ===" << std::endl;
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
