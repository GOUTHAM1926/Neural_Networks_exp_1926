#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/LossOps.h"
#include "checkpointing/GradMode.h"
#include "dnn/DistributedNN.h"
#include "mlp/activation.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "communication/include/ProcessGroupNCCL.h"
#include "tensor/dtensor.h"

// DataLoader (same path as gpt2_tp_test)
#include "Data_Loader/dl_test.cpp"

// Context Parallel
#include "gpt2_cp_test/context_parallel/ContextParallel.h"

#include "autograd/GraphRecorder.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// =============================================================================
// CudaTimer
// =============================================================================

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
  double get_elapsed_seconds() { return get_elapsed_ms() / 1000.0; }
};

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
  int64_t batch_size = 8;
  int64_t context_length = 1024;
  int64_t vocab_size = 50304;
  int64_t n_embd = 384;
  int64_t n_layers = 3;
  int64_t n_heads = 1;
  bool weight_tying = true;
  bool load_balancing = false;
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
  float coeff =
      0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * decay_ratio));
  return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Tiktoken Decoder (via Python popen)
// =============================================================================

std::string decode_tokens_tiktoken(const std::vector<int64_t> &tokens) {
  std::string cmd =
      "python3 -c \"import tiktoken; enc = tiktoken.get_encoding('gpt2'); "
      "print(enc.decode([";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0)
      cmd += ",";
    cmd += std::to_string(tokens[i]);
  }
  cmd += "]))\" 2>/dev/null";

  std::string result;
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe) {
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
      result += buffer;
    pclose(pipe);
  }
  if (!result.empty() && result.back() == '\n')
    result.pop_back();
  return result;
}

// =============================================================================
// Embedding Layer
// =============================================================================

class Embedding : public nn::Module {
public:
  Tensor weight;
  Embedding() = default;
  Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device,
            uint64_t seed = 1234)
      : vocab_size_(vocab_size), embed_dim_(embed_dim) {
    TensorOptions opts = TensorOptions()
                             .with_dtype(Dtype::Float32)
                             .with_device(device)
                             .with_req_grad(true);
    weight =
        Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
    register_parameter(weight);
  }

  Tensor forward(const Tensor &indices) override {
    return autograd::embedding(weight, indices);
  }

private:
  int64_t vocab_size_;
  int64_t embed_dim_;
};

// =============================================================================
// Helper: init linear weights
// =============================================================================

void init_linear_gpt2(nn::Linear &layer, float std_val = 0.02f,
                      uint64_t seed = 1234, bool req_grad = true) {
  auto shape = layer.weight.shape();
  TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);
  Tensor init_data = Tensor::randn<float>(shape, opts, seed, std_val);
  layer.weight.copy_(init_data);
  layer.weight.set_requires_grad(req_grad);

  if (layer.bias.is_valid()) {
    Tensor bias_init = Tensor::zeros(layer.bias.shape(), opts);
    layer.bias.copy_(bias_init);
    layer.bias.set_requires_grad(req_grad);
  }
}

// =============================================================================
// Context Parallel Attention Block
// =============================================================================

class CPAttention : public nn::Module {
public:
  nn::LayerNorm ln;
  nn::Linear c_attn;
  nn::Linear c_proj;
  CudaTimer timer_attn;
  double t_attn = 0.0;

  CPAttention(int64_t n_embd, int64_t n_heads, int64_t n_layers,
              DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,
              const DeviceMesh &mesh, uint64_t seed = 1234,
              bool load_balancing = false)
      : ln(n_embd), c_attn(n_embd, 3 * n_embd, true),
        c_proj(n_embd, n_embd, true), n_embd_(n_embd), n_heads_(n_heads),
        head_dim_(n_embd / n_heads) {
    init_linear_gpt2(c_attn, 0.02f, seed);
    float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
    init_linear_gpt2(c_proj, proj_std, seed + 1);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    // load_balance=false: HeadTail permutation breaks causal masking because
    // interleaved Q positions make the simple tril mask incorrect.
    // With load_balance=false, contiguous chunks allow correct causal behavior:
    //   self-chunk: tril causal mask
    //   past chunks (source_rank < rank_): full attention
    //   future chunks (source_rank > rank_): skipped
    cp_ = std::make_shared<ContextParallel>(
        mesh, pg, attn_scale, /*is_causal=*/true, RotatorType::AlltoAll,
        /*load_balance=*/load_balancing);

    ln.to(device);
    c_attn.to(device);
    c_proj.to(device);

    register_module(ln);
    register_module(c_attn);
    register_module(c_proj);
  }

  Tensor forward(const Tensor &x) override {
    int64_t B = x.shape().dims[0];
    int64_t T = x.shape().dims[1];
    int64_t C = x.shape().dims[2];

    Tensor h = ln.forward(x);

    Tensor qkv = c_attn.forward(h);
    std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
    Tensor q = inp[0];
    Tensor k = inp[1];
    Tensor v = inp[2];

    // Reshape to [B, H, T, D]
    q = autograd::transpose(
        autograd::reshape(q, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
    k = autograd::transpose(
        autograd::reshape(k, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
    v = autograd::transpose(
        autograd::reshape(v, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);

    // Context Parallel SDPA
    timer_attn.start_timer();
    Tensor attn_out = cp_->forward_cp(q, k, v);
    t_attn += timer_attn.get_elapsed_seconds();

    // Reshape back [B, T, C]
    Tensor merged = autograd::reshape(autograd::transpose(attn_out, 1, 2),
                                      Shape({{B, T, C}}));

    Tensor proj = c_proj.forward(merged);
    return autograd::add(x, proj);
  }

  void reset_t_attn() { t_attn = 0.0; }

private:
  int64_t n_embd_;
  int64_t n_heads_;
  int64_t head_dim_;
  std::shared_ptr<ContextParallel> cp_;
};

// =============================================================================
// MLP Block
// =============================================================================

class MLP : public nn::Module {
public:
  nn::LayerNorm ln;
  nn::Linear fc_up;
  nn::Linear fc_down;

  MLP(int64_t n_embd, int64_t n_layers, DeviceIndex device,
      uint64_t seed = 1234)
      : ln(n_embd), fc_up(n_embd, 4 * n_embd, true),
        fc_down(4 * n_embd, n_embd, true) {
    init_linear_gpt2(fc_up, 0.02f, seed);
    float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
    init_linear_gpt2(fc_down, proj_std, seed + 1);

    ln.to(device);
    fc_up.to(device);
    fc_down.to(device);

    register_module(ln);
    register_module(fc_up);
    register_module(fc_down);
  }

  Tensor forward(const Tensor &x) override {
    Tensor h = ln.forward(x);
    h = fc_up.forward(h);
    h = autograd::gelu(h);
    h = fc_down.forward(h);
    return autograd::add(x, h);
  }
};

// =============================================================================
// GPT Model with Context Parallel Attention
// =============================================================================

class GPT : public nn::Module {
public:
  GPTConfig config;
  Embedding wte;
  Embedding wpe;
  std::vector<std::shared_ptr<CPAttention>> attn_blocks;
  std::vector<std::shared_ptr<MLP>> mlp_blocks;
  nn::LayerNorm ln_f;
  std::shared_ptr<nn::Linear> lm_head;

  // Component timers
  double t_tok_emb = 0, t_pos_emb = 0, t_attn = 0;
  double t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
  CudaTimer timer_tok_emb, timer_pos_emb, timer_attn_block;
  CudaTimer timer_mlp, timer_ln_f, timer_lm_head;

  GPT(GPTConfig cfg, DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,
      const DeviceMesh &mesh, uint64_t seed = 1234)
      : config(cfg), wte(cfg.vocab_size, cfg.n_embd, device, seed),
        wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
        ln_f(cfg.n_embd) {
    ln_f.to(device);

    for (int i = 0; i < cfg.n_layers; ++i) {
      auto a = std::make_shared<CPAttention>(
          cfg.n_embd, cfg.n_heads, cfg.n_layers, device, pg, mesh,
          seed + 200 + i * 10,
          // seed + 200 + static_cast<uint64_t>(i) * 10,
          cfg.load_balancing);
      auto m = std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device,
                                     seed + 200 + i * 10);
      // seed + 300 + static_cast<uint64_t>(i) * 10);
      attn_blocks.push_back(a);
      mlp_blocks.push_back(m);
      register_module(a.get());
      register_module(m.get());
    }

    if (cfg.weight_tying) {
      lm_head = std::make_shared<nn::Linear>();
      {
        autograd::NoGradGuard no_grad;
        lm_head->weight = wte.weight.transpose(0, 1);
      }
      lm_head->weight.set_requires_grad(true);
    } else {
      lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
      init_linear_gpt2(*lm_head, 0.02f, seed + 1000, true);
      lm_head->to(device);
      register_module(lm_head.get());
    }

    Tensor pos_cpu(Shape{{1, cfg.context_length}},
                   TensorOptions().with_dtype(Dtype::Int64));
    int64_t *pos_data = pos_cpu.data<int64_t>();
    for (int64_t i = 0; i < cfg.context_length; ++i)
      pos_data[i] = i;
    cached_pos_ = pos_cpu.to(device);

    register_module(wte);
    register_module(wpe);
    register_module(ln_f);
  }

  void reset_timing() {
    t_tok_emb = t_pos_emb = t_attn = t_mlp = t_ln_f = t_lm_head = 0.0;
    for (auto &a : attn_blocks)
      a->reset_t_attn();
  }

  void collect_attn_timing() {
    t_attn = 0.0;
    for (auto &a : attn_blocks)
      t_attn += a->t_attn;
  }

  void print_timing(int rank) const {
    if (rank == 0) {
      std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1)
                << (t_tok_emb * 1000.0) << "ms"
                << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                << " | attn_cp: " << (t_attn * 1000.0) << "ms"
                << " | mlp: " << (t_mlp * 1000.0) << "ms"
                << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                << " | lm_head: " << (t_lm_head * 1000.0) << "ms" << std::endl;
    }
  }

  Tensor forward(const Tensor &idx) override {
    int64_t T = idx.shape().dims[1];

    // Token embedding
    timer_tok_emb.start_timer();
    Tensor tok_emb = wte.forward(idx);
    t_tok_emb += timer_tok_emb.get_elapsed_seconds();

    // Position embedding
    timer_pos_emb.start_timer();
    Tensor pos_flat =
        autograd::reshape(cached_pos_, Shape({{config.context_length}}));
    Tensor pos_sliced = pos_flat.slice(0, T);
    Tensor pos_idx = autograd::reshape(pos_sliced, Shape({{1, T}}));
    Tensor pos_emb = wpe.forward(pos_idx);
    t_pos_emb += timer_pos_emb.get_elapsed_seconds();

    Tensor x = autograd::add(tok_emb, pos_emb);

    // Attention + MLP blocks
    for (int i = 0; i < config.n_layers; ++i) {
      x = attn_blocks[i]->forward(x);
      // attn timing tracked inside CPAttention; collect at end

      timer_mlp.start_timer();
      x = mlp_blocks[i]->forward(x);
      t_mlp += timer_mlp.get_elapsed_seconds();
    }

    // Final LayerNorm
    timer_ln_f.start_timer();
    x = ln_f.forward(x);
    t_ln_f += timer_ln_f.get_elapsed_seconds();

    // LM Head
    timer_lm_head.start_timer();
    Tensor logits = lm_head->forward(x);
    t_lm_head += timer_lm_head.get_elapsed_seconds();

    return logits;
  }

private:
  Tensor cached_pos_;
};

// =============================================================================
// Main Training Loop
// =============================================================================

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    std::cout << "=== GPT-2 Context Parallel Training Script ===" << std::endl;
  }

  try {
    // Configuration
    GPTConfig config;
    config.batch_size = 4;
    config.context_length = 1024;
    config.vocab_size = 50304;
    config.n_embd = 384;
    config.n_layers = 3;
    config.n_heads = 6;
    config.weight_tying = false;
    config.load_balancing = true;

    const int B = static_cast<int>(config.batch_size);
    const int T = static_cast<int>(config.context_length);
    const int global_batch = 65536;
    const int grad_accum_steps = global_batch / (B * T);

    const float max_lr = 6e-4f;
    const float min_lr = max_lr * 0.1f;
    const int VAL_FREQ = 100;
    const int TOK_GEN_FREQ = 100;

    if (rank == 0) {
      std::cout << "Configuration:\n";
      std::cout << "  vocab_size: " << config.vocab_size << "\n";
      std::cout << "  context_length: " << config.context_length << "\n";
      std::cout << "  n_embd: " << config.n_embd << "\n";
      std::cout << "  n_layers: " << config.n_layers << "\n";
      std::cout << "  n_heads: " << config.n_heads << "\n";
      std::cout << "  B=" << B << ", T=" << T << "\n";
      std::cout << "  world_size: " << world_size << "\n";
    }

    // Device + Process Group
    DeviceIndex device(Device::CUDA, rank);
    cudaSetDevice(rank);

    std::vector<int> ranks_vec(world_size);

    for (int i = 0; i < world_size; i++)
      ranks_vec[i] = i;
    DeviceMesh mesh({world_size}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    if (rank == 0) {
      std::cout << "\nInitializing model on CUDA device " << rank << "...\n";
    }

    // Model
    GPT model(config, device, pg, mesh, /*seed=*/1234);

    // Parameter count
    auto params = model.parameters();
    int64_t num_params = 0;
    int64_t num_params_gpu = 0;
    for (auto &p : params) {
      num_params += p.numel();
      num_params_gpu += p.numel();
    }

    // const int max_steps    = (static_cast<int>(num_params) / global_batch ) *
    // 5;
    const int max_steps = 6768;
    // const int max_steps    = 1;
    const int warmup_steps = max_steps / 10;

    if (rank == 0) {
      std::cout << "Parameters: " << num_params << "\n";
      std::cout << "Parameters per GPU: " << num_params_gpu << "\n";
      std::cout << "max_steps: " << max_steps << "\n";
      std::cout << "warmup_steps: " << warmup_steps << "\n";
    }

    // Optimizer
    nn::AdamW optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);

    // Data loaders (same path as gpt2_tp_test)
    std::string data_root =
        "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data/";
    DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true, 100000000,
                                rank);
    DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true, 100000000,
                              rank);

    // Restore CUDA device context after DataLoader init
    // (DataLoader constructor may switch active device via tensor allocation)
    cudaSetDevice(rank);

    // Step timers
    CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd;
    CudaTimer timer_clip, timer_optim;

    if (rank == 0) {
      std::cout << "\nStarting training...\n";
    }

    // CSV log + config file setup
    std::string log_filename, config_filename;
    std::ofstream log_file;

    if (rank == 0) {
      std::filesystem::create_directories("CP_Training_logs");
      int log_idx = 1;
      while (true) {
        log_filename = "CP_Training_logs/CP_Training_log" +
                       std::to_string(log_idx) + ".csv";
        if (!std::filesystem::exists(log_filename))
          break;
        log_idx++;
      }
      std::cout << "Saving logs to: " << log_filename << "\n";

      config_filename = "CP_Training_logs/CP_Training_log" +
                        std::to_string(log_idx) + "_config.txt";
      std::ofstream config_file(config_filename);
      config_file << "Configuration:\n";
      config_file << "  Batch_size: " << B << "\n";
      config_file << "  context_length: " << config.context_length << "\n";
      config_file << "  n_embd: " << config.n_embd << "\n";
      config_file << "  n_heads: " << config.n_heads << "\n";
      config_file << "  vocab_size: " << config.vocab_size << "\n";
      config_file << "  n_layers: " << config.n_layers << "\n";
      config_file << "  global_batch: " << global_batch << "\n";
      config_file << "  grad_accum_steps: " << grad_accum_steps << "\n";
      config_file << "  world_size: " << world_size << "\n";
      config_file << "  Parameters: " << num_params << "\n";
      config_file << "  Parameters per GPU: " << num_params_gpu << "\n";
      config_file << "  Max Learning Rate: " << max_lr << "\n";
      config_file << "  Min Learning Rate: " << min_lr << "\n";
      config_file << "  max_steps: " << max_steps << "\n";
      config_file << "  warmup_steps: " << warmup_steps << "\n";

      // Initial GPU memory
      size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);
      double used_mb =
          static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);
      double total_mb = static_cast<double>(total_mem) / (1024.0 * 1024.0);
      config_file << "  GPU Memory Used (rank 0): " << std::fixed
                  << std::setprecision(1) << used_mb << " MB / " << total_mb
                  << " MB\n";
      config_file.close();

      log_file.open(log_filename);
      if (!log_file.is_open()) {
        std::cerr << "ERROR: Could not open log file " << log_filename << "\n";
        std::exit(1);
      }
      log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
                  "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,"
                  "timer_optim,timer_tok_emb,timer_pos_emb,timer_attn_cp,"
                  "timer_mlp,timer_ln_f,timer_lm_head,mem_gpu_mb\n";
      log_file << std::fixed << std::setprecision(6);
    }

    float val_loss_log = -1.0f;

    for (int step = 0; step < max_steps; ++step) {
      try {
        timer_step.start_timer();

        // ---- Validation every VAL_FREQ steps ----
        if (step % VAL_FREQ == 0 || step == max_steps - 1) {
          val_loader.reset();
          float val_loss_accum = 0.0f;
          const int val_steps = 5;

          for (int v = 0; v < val_steps; ++v) {
            Batch vbatch = val_loader.next_batch();
            Tensor vx = vbatch.input.to(device).as_type(Dtype::Int64);
            Tensor vy = vbatch.target.to(device).as_type(Dtype::Int64);
            Tensor vlogits = model.forward(vx);
            Tensor vloss = autograd::sparse_cross_entropy_loss(vlogits, vy);
            Tensor vloss_cpu = vloss.to_cpu();
            val_loss_accum +=
                vloss_cpu.data<float>()[0] / static_cast<float>(val_steps);
          }

          if (rank == 0) {
            std::cout << "validation loss: " << std::fixed
                      << std::setprecision(4) << val_loss_accum << "\n";
          }
          val_loss_log = val_loss_accum;
        }

        // ---- Token generation every TOK_GEN_FREQ steps ----
        if (step > 0 && (step % TOK_GEN_FREQ == 0 || step == max_steps - 1)) {
          if (rank == 0) {
            std::cout << "--- Generating tokens at step " << step << " ---\n";
          }

          const int num_return_seq = 2;
          const int max_length = 32;

          Tensor xgen(
              Shape({{num_return_seq, 8}}),
              TensorOptions().with_dtype(Dtype::Int64).with_device(device));
          std::vector<int64_t> xgen_tokens = {
              15496, 11, 314, 1101, 281, 9552, 2746, 11,
              15496, 11, 314, 1101, 281, 9552, 2746, 11};
          xgen.set_data(xgen_tokens);

          while (xgen.shape().dims[1] < max_length) {
            // Pad sequence to multiple of world_size for CP
            int64_t seq_len = xgen.shape().dims[1];
            int64_t padded_len =
                ((seq_len + world_size - 1) / world_size) * world_size;
            Tensor xgen_fwd = xgen;
            if (padded_len != seq_len) {
              int64_t Bg_pad = xgen.shape().dims[0];
              Tensor pad = Tensor::zeros(
                  Shape({{Bg_pad, padded_len - seq_len}}), xgen.opts());
              xgen_fwd = Tensor::cat({xgen, pad}, 1);
            }
            Tensor gen_logits = model.forward(xgen_fwd);

            int64_t Bg = gen_logits.shape().dims[0];
            int64_t Tg = gen_logits.shape().dims[1];
            int64_t Vg = gen_logits.shape().dims[2];

            // Use original seq_len-1 as last real token position
            Tensor gather_idx = Tensor::full(
                Shape({{Bg, 1, Vg}}),
                TensorOptions().with_dtype(Dtype::Int64).with_device(device),
                static_cast<float>(seq_len - 1));

            Tensor last_logits = OwnTensor::gather(gen_logits, 1, gather_idx);
            Tensor probs = autograd::softmax(last_logits, -1);

            auto topk_res = probs.topk(50, -1);
            Tensor topk_probs = topk_res.first;
            Tensor topk_idx = topk_res.second;

            Tensor topk_probs_2d =
                autograd::reshape(topk_probs, Shape({{Bg, 50}}));

            Tensor ix;
            if (rank == 0) {
              try {
                ix = Tensor::multinomial(topk_probs_2d, 1);
              } catch (...) {
                ix = Tensor::zeros(Shape({{Bg, 1}}),
                                   TensorOptions()
                                       .with_dtype(Dtype::Int64)
                                       .with_device(device));
              }
            } else {
              ix = Tensor::zeros(
                  Shape({{Bg, 1}}),
                  TensorOptions().with_dtype(Dtype::Int64).with_device(device));
            }

            if (world_size > 1) {
              ix = ix.contiguous();
              pg->broadcast(ix.data<int64_t>(), ix.data<int64_t>(), ix.numel(),
                            Dtype::Int64, 0, true);
            }

            Tensor topk_idx_2d = autograd::reshape(topk_idx, Shape({{Bg, 50}}));
            Tensor next_token = OwnTensor::gather(topk_idx_2d, 1, ix);
            xgen = Tensor::cat({xgen, next_token}, 1);
          }

          if (rank == 0) {
            Tensor xgen_cpu = xgen.to_cpu();
            int64_t *data = xgen_cpu.data<int64_t>();
            int64_t Bg = xgen.shape().dims[0];
            int64_t Tg = xgen.shape().dims[1];

            std::ofstream cfg_app(config_filename, std::ios::app);
            cfg_app << "\n--- Generated Tokens at Step " << step << " ---\n";

            for (int i = 0; i < static_cast<int>(Bg); ++i) {
              std::vector<int64_t> sample_tokens;
              std::string token_str;
              for (int j = 0; j < static_cast<int>(Tg); ++j) {
                sample_tokens.push_back(data[i * Tg + j]);
                if (j > 0)
                  token_str += " ";
                token_str += std::to_string(data[i * Tg + j]);
              }
              std::string decoded = decode_tokens_tiktoken(sample_tokens);
              std::cout << "Sample " << i << " [IDs]: " << token_str << "\n";
              std::cout << "Sample " << i << " [Text]: " << decoded << "\n";
              cfg_app << "Sample " << i << " [IDs]: " << token_str << "\n";
              cfg_app << "Sample " << i << " [Text]: " << decoded << "\n";
            }
            cfg_app.close();
          }
        }

        // ---- Training step ----
        double time_data = 0, time_forward = 0, time_loss = 0;
        double time_backward = 0, time_clip = 0, time_optim = 0;

        optimizer.zero_grad();
        if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
          model.lm_head->weight.zero_grad();
        }
        model.reset_timing();

        Tensor loss_accum_gpu =
            Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

        static Tensor grad_scale =
            Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                         1.0f / static_cast<float>(grad_accum_steps));

        for (int micro = 0; micro < grad_accum_steps; ++micro) {
          // Data
          timer_data.start_timer();
          Batch batch = train_loader.next_batch();
          Tensor x_in = batch.input.to(device).as_type(Dtype::Int64);
          Tensor y_in = batch.target.to(device).as_type(Dtype::Int64);
          time_data += timer_data.get_elapsed_seconds();

          // Record autograd graph on step 0, micro 0, rank 0
          std::unique_ptr<autograd::GraphRecordGuard> graph_guard;
          if (step == 0 && micro == 0 && rank == 0) {
              graph_guard =
              std::make_unique<autograd::GraphRecordGuard>(true);
          }

          // Forward
          timer_fwd.start_timer();
          Tensor logits = model.forward(x_in);
          time_forward += timer_fwd.get_elapsed_seconds();

          // Loss
          timer_loss.start_timer();
          Tensor loss = autograd::sparse_cross_entropy_loss(logits, y_in);
          loss_accum_gpu = loss_accum_gpu + loss.detach();
          time_loss += timer_loss.get_elapsed_seconds();

          // Backward
          timer_bwd.start_timer();
          loss.backward(&grad_scale);
          time_backward += timer_bwd.get_elapsed_seconds();

          // Guard destructor auto-prints forward + backward sequences
          // graph_guard.reset();
        }

        // Weight tying: accumulate lm_head grad into wte
        if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
          Tensor lm_grad_T =
              model.lm_head->weight.grad_view().transpose(0, 1).contiguous();
          Tensor wte_grad = model.wte.weight.grad_view();
          wte_grad += lm_grad_T;
        }

        // Collect attention timing from all layers
        model.collect_attn_timing();

        // TEMP DEBUGGING: Print parameter gradients to check cross-rank sync
        // if ((grad_accum_steps + 1) % 1 == 0 && rank == 0) {
        //   for (int r = 0; r < world_size; ++r) {
        //     if (rank == r) {
        //       std::cout << "\n=== DEBUG: Parameter Gradients at Step " << step
        //                 << " [Rank " << rank << "] ===" << std::endl;
        //       for (auto &p : params) {
        //         if (p.has_grad()) {
        //           std::cout << "\nParam size: " << p.numel() << std::endl;
        //           try {
        //             // p.grad_view().display();
        //             p.display();
        //           } catch (const std::exception &e) {
        //             std::cout << "  Error displaying grad: " << e.what()
        //                       << std::endl;
        //           }
        //         } else {
        //           std::cout << "Param | NO GRADIENT" << std::endl;
        //         }
        //       }
        //       std::cout << "=================================================\n"
        //                 << std::endl;
        //     }
        //   }
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // Loss to CPU
        Tensor loss_cpu = loss_accum_gpu.to_cpu();
        float loss_scalar =
            loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);

        if (std::isnan(loss_scalar) || std::isinf(loss_scalar)) {
          std::cerr << "ERROR: NaN/Inf at step " << step << "\n";
          if (rank == 0)
            log_file.close();
          MPI_Finalize();
          return 1;
        }

        // Gradient clipping
        timer_clip.start_timer();
        float norm = nn::clip_grad_norm_(params, 1.0f);
        time_clip = timer_clip.get_elapsed_seconds();

        // LR schedule + optimizer step
        float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
        optimizer.set_lr(lr);

        timer_optim.start_timer();
        optimizer.step();
        time_optim = timer_optim.get_elapsed_seconds();

        double dt = timer_step.get_elapsed_seconds();

        // Throughput + time left
        int64_t tokens_processed =
            static_cast<int64_t>(B) * T * grad_accum_steps;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
        long long total_sec = static_cast<long long>((max_steps - step) * dt);
        int h = static_cast<int>(total_sec / 3600);
        int m = static_cast<int>((total_sec % 3600) / 60);

        // GPU memory
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        double used_mb =
            static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);

        if (rank == 0) {
          std::cout << "step " << std::setw(5) << step
                    << " | loss: " << std::fixed << std::setprecision(6)
                    << loss_scalar << " | lr " << std::scientific
                    << std::setprecision(4) << lr << " | norm: " << std::fixed
                    << std::setprecision(4) << norm << " | dt: " << std::fixed
                    << std::setprecision(2) << (dt * 1000.0) << "ms"
                    << " | tok/sec: " << std::fixed << std::setprecision(1)
                    << tokens_per_sec << " | mem: " << std::fixed
                    << std::setprecision(0) << used_mb << "MB"
                    << " | Time Left: " << std::setfill('0') << std::setw(2)
                    << h << " hrs : " << std::setw(2) << m << " mins"
                    << std::setfill(' ') << "\n";

          std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1)
                    << (time_data * 1000.0) << "ms"
                    << " | fwd: " << (time_forward * 1000.0) << "ms"
                    << " | loss: " << (time_loss * 1000.0) << "ms"
                    << " | bwd: " << (time_backward * 1000.0) << "ms"
                    << " | clip: " << (time_clip * 1000.0) << "ms"
                    << " | optim: " << (time_optim * 1000.0) << "ms\n";

          model.print_timing(rank);

          // CSV
          log_file << step << "," << loss_scalar << "," << val_loss_log << ","
                   << lr << "," << norm << "," << (dt * 1000.0) << ","
                   << tokens_per_sec << "," << (time_data * 1000.0) << ","
                   << (time_forward * 1000.0) << "," << (time_loss * 1000.0)
                   << "," << (time_backward * 1000.0) << ","
                   << (time_clip * 1000.0) << "," << (time_optim * 1000.0)
                   << "," << (model.t_tok_emb * 1000.0) << ","
                   << (model.t_pos_emb * 1000.0) << ","
                   << (model.t_attn * 1000.0) << "," << (model.t_mlp * 1000.0)
                   << "," << (model.t_ln_f * 1000.0) << ","
                   << (model.t_lm_head * 1000.0) << "," << used_mb << "\n";
          log_file.flush();
        }

        val_loss_log = -1.0f;

      } catch (const std::exception &e) {
        std::cerr << "EXCEPTION RANK " << rank << " STEP " << step << ": "
                  << e.what() << "\n";
        std::exit(1);
      }
    }

    if (rank == 0) {
      log_file.close();
      std::cout << "\nTraining log saved to: " << log_filename << "\n";
      std::cout << "\n=== Context Parallel Training Complete ===\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}