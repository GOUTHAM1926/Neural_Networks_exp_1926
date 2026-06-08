#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cstring>
#include <unordered_map>

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
#include "ops/FusedKernels.cuh"
#include "autograd/backward/FusedTrilSoftmaxBackward.h"

#include "checkpointing/Checkpointing.h"
#include "device/CachingCudaAllocator.h"
#include "device/AllocationTracker.h"

// Dataloader
#include "DataLoader.h"
#include "dist/distributed.h"
#include "autograd/GraphRecorder.h"

// HellaSwag in-training eval (mirrors gpt2.py:507-538).
#include "hellaswag_eval.h"

using namespace OwnTensor;

// Runtime toggle: when true, Attention::forward uses the packed-QKV fused
// SDPA path (scaled_dot_product_attention_packed) — skipping shard,
// reshape+transpose of Q/K/V on the way in, and transpose+reshape of the
// merged output on the way out. When false, the original unfused path is
// used. Set by the USE_PACKED_SDPA env var in main().
static bool g_use_packed_sdpa = false;

// Runtime toggle for the unfused path only: when true, reshape qkv to
// [B,T,H,3*d] (free view on contiguous qkv) and then shard axis 3 — avoids
// the 3 contiguous copies that the original shard-axis-2 → reshape route
// forces. Ignored when g_use_packed_sdpa is true. Set by USE_QKV_RESHAPE_FIRST.
static bool g_use_qkv_reshape_first = false;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 768;
    int64_t n_layers = 12;
    int64_t n_heads = 12;       // GPT-2 124M
    bool weight_tying = true;     // Flag for weight tying
    bool checkpointing = false;
    bool if_log = true;
};

// =============================================================================
// Single shared advancing RNG (mirrors gpt2.py's torch.manual_seed(1234) +
// apply(_init_weights) — one stream advanced across every parameter draw).
// =============================================================================
//
// Tensor::randn(shape, opts, seed, std) constructs a fresh std::mt19937 / cuRAND
// state on each call, so its per-call seed cannot share state across draws.
// Bypass it for parameter init: draw all init values from one std::mt19937 here
// and copy_ each draw into the parameter (CPU→device handled by copy_, which is
// already how init_linear_gpt2 worked). LayerNorms are not touched (gpt2.py's
// _init_weights doesn't either — they keep the default weight=1, bias=0).
//
// Advancement order matches PyTorch's apply(_init_weights) traversal:
//   wte → wpe → for each block: (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)
//        → lm_head    (overwrites wte when weight_tying=true)

struct InitRng {
    std::mt19937 gen;
    explicit InitRng(uint64_t seed) : gen(static_cast<uint32_t>(seed)) {}

    // CPU float32 tensor drawn from N(0, std) using the shared advancing stream.
    Tensor normal(const Shape& shape, float std) {
        Tensor t(shape, TensorOptions().with_dtype(Dtype::Float32));
        std::normal_distribution<float> dist(0.0f, std);
        float* data = t.data<float>();
        size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
        return t;
    }
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding : public nn::Module {
public:
    Tensor weight;  // [vocab_size, n_embd]
    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device, InitRng& rng)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        // Allocate the parameter on device WITHOUT req_grad first, fill it via
        // copy_ from a CPU draw on the shared RNG stream, THEN flip on req_grad
        // — matches init_linear_gpt2's order so copy_ never runs against a
        // leaf with grad-tracking already enabled. Done this way (rather than
        // constructing via Tensor::randn) so a single std::mt19937 advances
        // across every parameter init in the model.
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device);
        weight = Tensor(Shape{{vocab_size, embed_dim}}, opts);
        Tensor init_data = rng.normal(Shape{{vocab_size, embed_dim}}, 0.02f);
        weight.copy_(init_data);
        weight.set_requires_grad(true);

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

void init_linear_gpt2(nn::Linear& layer, float std, InitRng& rng, bool req_grad=true) {
    // IMPORTANT: Do NOT replace layer.weight with a new tensor!
    // nn::Linear already registered its weight in params_.
    // We must copy data INTO the existing weight to preserve parameter identity.
    Tensor init_data = rng.normal(layer.weight.shape(), std);
    layer.weight.copy_(init_data);
    layer.weight.set_requires_grad(req_grad);

    if (layer.bias.is_valid()) {
        Tensor bias_init = Tensor::zeros(layer.bias.shape(),
                                         TensorOptions().with_dtype(Dtype::Float32));
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
    
    Attention(int64_t n_embd, int n_heads, int n_layers, DeviceIndex device, InitRng& rng)
        : ln(n_embd),
          c_attn(n_embd, 3 * n_embd, true),
          c_proj(n_embd, n_embd, true),
          n_embd_(n_embd),
          n_heads_(n_heads),
          head_dim_(n_embd / n_heads)
    {
        // Init order matches gpt2.py CausalSelfAttention.apply(_init_weights):
        //   c_attn (std=0.02), then c_proj (std=0.02 * (2L)^-0.5  — NANOGPT_SCALE_INIT).
        init_linear_gpt2(c_attn, 0.02f, rng);
        const float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(c_proj, 0.02f * scale, rng);

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

        Tensor merged;
        if (g_use_packed_sdpa) {
            // Packed-QKV fused SDPA: kernel reads Q/K/V via strided pointers
            // into qkv directly, output is already [B, T, C] — no shard,
            // no reshape, no transpose on either side.
            merged = autograd::scaled_dot_product_attention_packed(
                    qkv, n_heads_, /*is_causal=*/true, /*dropout_p=*/0.0f,
                    autograd::SDPBackend::MemoryEfficient);
        } else {
            Tensor q, k, v;
            if (g_use_qkv_reshape_first) {
                // Reshape-first: qkv is contiguous, so reshape to [B,T,H,3d]
                // is a free view. Then shard axis 3 into 3 × [B,T,H,d] views
                // (non-contiguous, but downstream stride-aware ops handle it).
                Tensor qkv_heads = autograd::reshape(qkv, Shape({{B, T, n_heads_, 3 * head_dim_}}));
                std::vector<Tensor> inp = qkv_heads.make_shards_inplace_axis(3, 3);
                q = autograd::transpose(inp[0], 1, 2);
                k = autograd::transpose(inp[1], 1, 2);
                v = autograd::transpose(inp[2], 1, 2);
            } else {
                // Original baseline: shard axis 2 → reshape forces 3 contiguous copies.
                std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
                q = autograd::transpose( autograd::reshape(inp[0], Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
                k = autograd::transpose( autograd::reshape(inp[1], Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
                v = autograd::transpose( autograd::reshape(inp[2], Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
            }

            Tensor attn_out = autograd::scaled_dot_product_attention(
                    q, k, v, /*is_causal=*/true, 0.0f, autograd::SDPBackend::MemoryEfficient);

            merged = autograd::reshape(
                                autograd::transpose(attn_out, 1, 2),
                                Shape({{B, T, C}}));
        }

        // Output projection
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
    
    MLP(int64_t n_embd, int n_layers, DeviceIndex device, InitRng& rng)
        : ln(n_embd),
          fc_up(n_embd, 4 * n_embd, true),
          fc_down(4 * n_embd, n_embd, true),
          n_embd_(n_embd)
    {
        // Init order matches gpt2.py MLP.apply(_init_weights):
        //   c_fc / fc_up   (std=0.02), then
        //   c_proj/fc_down (std=0.02 * (2L)^-0.5 — NANOGPT_SCALE_INIT).
        init_linear_gpt2(fc_up, 0.02f, rng);
        const float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(fc_down, 0.02f * scale, rng);

        // Move everything to device (uses to_cuda_ which modifies in-place)
        fc_up.to(device);
        fc_down.to(device);
        ln.to(device);

        register_module(ln);
        register_module(fc_up);
        register_module(fc_down);
    }
    
    // Forward: x [B, T, C] -> [B, T, C]
    // Tensor forward(const Tensor& x) override {
    //     // Pre-Norm: ln(x)
    //     Tensor h = ln.forward(x);
        
    //     // Up projection + GELU + Down projection
    //     // Fused: matmul(h, weight) then bias+gelu in one kernel (saves 1 kernel fwd + 1 bwd)
    //     h = autograd::matmul(h, fc_up.weight);
    //     h = autograd::fused_bias_gelu(h, fc_up.bias);
    //     h = fc_down.forward(h);
        
    //     // Residual connection: x + MLP(x)
    //     return autograd::add(x, h);
    // }
    
    Tensor forward(const Tensor& x) override {
        // Pre-Norm: ln(x)
        Tensor h = ln.forward(x);
        
        // Up projection + GELU + Down projection
        // Fused: matmul(h, weight) then bias+gelu in one kernel (saves 1 kernel fwd + 1 bwd)
        // h = autograd::matmul(h, fc_up.weight);
        // h = autograd::fused_bias_gelu(h, fc_up.bias);
        // h = fc_down.forward(h);

        h = fc_up.forward(h);
        h = autograd::gelu(h );
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
private:
    // MUST be declared before wte/wpe — C++ initializes members in declaration
    // order, and wte/wpe's initializers reference init_rng_, so it has to exist
    // first. Holds the single shared std::mt19937 advanced through every
    // parameter init in gpt2.py's _init_weights traversal order.
    InitRng init_rng_;
public:
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
          // Single torch.manual_seed(1234)-style stream advanced through every
          // parameter draw in gpt2.py's apply(_init_weights) traversal order:
          //   wte → wpe → per block (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)
          //              → lm_head  (overwrites wte when weight_tying=true)
          // (LayerNorms left at default weight=1 / bias=0 — _init_weights in
          // gpt2.py doesn't touch them either.)
          init_rng_(seed),
          wte(cfg.vocab_size, cfg.n_embd, device, init_rng_),
          wpe(cfg.context_length, cfg.n_embd, device, init_rng_),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);

        // Each block advances the same RNG: c_attn → attn.c_proj → mlp.c_fc → mlp.c_proj.
        for (int i = 0; i < cfg.n_layers; ++i) {
            auto a = std::make_shared<Attention>(cfg.n_embd, cfg.n_heads, cfg.n_layers, device, init_rng_);
            auto m = std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device, init_rng_);
            attn_blocks.push_back(a);
            mlp_blocks.push_back(m);
            register_module(a.get());
            register_module(m.get());
        }

        if (!config.weight_tying) {
            // Untied: lm_head is its own parameter — final draw from the same stream,
            // std=0.02 (no NANOGPT_SCALE_INIT, matching gpt2.py's lm_head init).
            lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
            init_linear_gpt2(*lm_head, 0.02f, init_rng_, true);
            lm_head->to(device);
        } else {
            // Tied: in gpt2.py, `self.transformer.wte.weight = self.lm_head.weight`
            // makes both names refer to the same Parameter, and apply(_init_weights)
            // walks lm_head LAST → its Linear normal_(std=0.02) draw OVERWRITES the
            // earlier wte Embedding init in the shared parameter. Replicate by
            // drawing one more value from the shared stream and copy_-ing into
            // wte.weight (preserves tensor identity → registered param + DDP stay
            // valid).
            Tensor lm_head_init = init_rng_.normal(wte.weight.shape(), 0.02f);
            wte.weight.copy_(lm_head_init);
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

        // Zero-copy view for position indices (avoids 3 intermediate tensor allocations)
        Tensor pos_indices = (T == config.context_length)
            ? cached_pos_                          // [1, context_length] — reuse directly
            : cached_pos_.narrow_view(1, 0, T);    // [1, T] view — no copy
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
        Tensor logits;
        if (config.weight_tying) {
            // Tied: reuse wte.weight transposed. autograd::transpose keeps TransposeBackward
            // attached so gradient flows back to wte.weight.AccumulateGrad automatically.
            Tensor w_T = autograd::transpose(wte.weight, 0, 1);
            logits = autograd::matmul(x, w_T);
        } else {
            logits = lm_head->forward(x);  // [B, T, vocab_size]
        }
        // std::cout << "Came Here twice" << std::endl;
        // std::cout << "lm_head completed "<< std::endl;
        // std::cout << "Stats reserved after: " << stats.allocated_current / (1024 * 1024) << std::endl;

        
        return logits;
    }

private:
    Tensor cached_pos_;  // [1, T] position indices, cached on GPU
};

// =============================================================================
// Init parity loader: read gpt2_init.bin produced by dump_init.py and copy_
// values straight into the model's parameter tensors. Lets the C++ trainer
// start from the EXACT same point in parameter space as gpt2.py at seed=1234,
// removing RNG-stream divergence as a source of drift in like-for-like
// comparisons.
//
// Wire format (must match dump_init.py):
//   magic         : 4 bytes "GP2I"
//   version       : uint32 (=1)
//   num_tensors   : uint32
//   per tensor:
//     name_len    : uint16
//     name        : utf-8 bytes
//     dtype       : uint8 (0 = float32)
//     ndim        : uint8
//     dims        : int64 * ndim
//     payload     : float32 * numel
//
// Names use PyTorch's nesting (h.{i}.ln_1, h.{i}.attn.c_attn, ...). The C++
// model has a different structure (LayerNorms live INSIDE Attention/MLP, not
// in a Block), so the loader maps PyTorch names → the matching C++ tensors.
//   ln_1     ↔ attn_blocks[i]->ln       (pre-attn norm)
//   c_attn   ↔ attn_blocks[i]->c_attn
//   c_proj   ↔ attn_blocks[i]->c_proj   (in attn block)
//   ln_2     ↔ mlp_blocks[i]->ln        (pre-mlp norm)
//   mlp.c_fc ↔ mlp_blocks[i]->fc_up
//   mlp.c_proj ↔ mlp_blocks[i]->fc_down
// Linear weights are already transposed in the file ([in, out]) — the C++
// tensor lib stores Linear as [in, out], so this is a flat memcpy.
// =============================================================================
static void load_init_bin(GPT& model, const std::string& path, bool is_master) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_init_bin: cannot open " + path);

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GP2I", 4) != 0)
        throw std::runtime_error("load_init_bin: bad magic in " + path);

    uint32_t version = 0, num_tensors = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_tensors), 4);
    if (version != 1u)
        throw std::runtime_error("load_init_bin: unsupported version");

    // Build name → Tensor& map for the C++ model.
    std::unordered_map<std::string, Tensor*> name_to_tensor;
    name_to_tensor.reserve(2 + 12 * static_cast<size_t>(model.config.n_layers) + 2);
    name_to_tensor["wte.weight"] = &model.wte.weight;
    name_to_tensor["wpe.weight"] = &model.wpe.weight;
    name_to_tensor["ln_f.weight"] = &model.ln_f.weight;
    name_to_tensor["ln_f.bias"]   = &model.ln_f.bias;
    for (int i = 0; i < model.config.n_layers; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        auto& a = *model.attn_blocks[i];
        auto& m = *model.mlp_blocks[i];
        name_to_tensor[p + "ln_1.weight"]        = &a.ln.weight;
        name_to_tensor[p + "ln_1.bias"]          = &a.ln.bias;
        name_to_tensor[p + "attn.c_attn.weight"] = &a.c_attn.weight;
        name_to_tensor[p + "attn.c_attn.bias"]   = &a.c_attn.bias;
        name_to_tensor[p + "attn.c_proj.weight"] = &a.c_proj.weight;
        name_to_tensor[p + "attn.c_proj.bias"]   = &a.c_proj.bias;
        name_to_tensor[p + "ln_2.weight"]        = &m.ln.weight;
        name_to_tensor[p + "ln_2.bias"]          = &m.ln.bias;
        name_to_tensor[p + "mlp.c_fc.weight"]    = &m.fc_up.weight;
        name_to_tensor[p + "mlp.c_fc.bias"]      = &m.fc_up.bias;
        name_to_tensor[p + "mlp.c_proj.weight"]  = &m.fc_down.weight;
        name_to_tensor[p + "mlp.c_proj.bias"]    = &m.fc_down.bias;
    }

    size_t loaded = 0, total_floats = 0;
    std::vector<float> buf;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint16_t name_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), 2);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint8_t dtype = 0, ndim = 0;
        f.read(reinterpret_cast<char*>(&dtype), 1);
        f.read(reinterpret_cast<char*>(&ndim), 1);
        if (dtype != 0)
            throw std::runtime_error("load_init_bin: only float32 supported, " + name);

        std::vector<int64_t> dims(ndim);
        for (uint8_t d = 0; d < ndim; ++d) {
            f.read(reinterpret_cast<char*>(&dims[d]), 8);
        }
        size_t numel = 1;
        for (auto d : dims) numel *= static_cast<size_t>(d);
        buf.resize(numel);
        f.read(reinterpret_cast<char*>(buf.data()), numel * sizeof(float));
        if (!f) throw std::runtime_error("load_init_bin: read failed at " + name);

        auto it = name_to_tensor.find(name);
        if (it == name_to_tensor.end()) {
            if (is_master) std::cerr << "[init-load] WARN: unknown tensor name '" << name
                                     << "' — skipping" << std::endl;
            continue;
        }
        Tensor* dst = it->second;

        // Shape sanity. Both libs store the same logical layout per the
        // load-side mapping above (Embedding [vocab,emb], Linear [in,out]
        // post-transpose, LayerNorm [d]).
        const auto& dst_shape = dst->shape().dims;
        if (static_cast<int64_t>(dst_shape.size()) != static_cast<int64_t>(ndim)) {
            throw std::runtime_error("load_init_bin: ndim mismatch for " + name);
        }
        for (size_t d = 0; d < dst_shape.size(); ++d) {
            if (dst_shape[d] != dims[d]) {
                std::ostringstream oss;
                oss << "load_init_bin: shape mismatch for " << name
                    << " — file says [";
                for (size_t k = 0; k < dims.size(); ++k) oss << dims[k] << (k + 1 < dims.size() ? "," : "");
                oss << "], model has [";
                for (size_t k = 0; k < dst_shape.size(); ++k) oss << dst_shape[k] << (k + 1 < dst_shape.size() ? "," : "");
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }

        // CPU staging tensor → copy_ pushes to the destination's device.
        Shape shape;
        shape.dims = std::vector<int64_t>(dims.begin(), dims.end());
        Tensor cpu_t(shape, TensorOptions().with_dtype(Dtype::Float32));
        cpu_t.set_data<float>(buf.data(), numel);
        // copy_ requires the destination's req_grad to be off when the source
        // is a fresh leaf tensor (mirrors how init_linear_gpt2 turns req_grad
        // OFF, copy_'s, then turns it back ON). Embedding/Linear weights have
        // req_grad already true here. Toggle around the copy_.
        const bool was_req = dst->requires_grad();
        if (was_req) dst->set_requires_grad(false);
        dst->copy_(cpu_t);
        if (was_req) dst->set_requires_grad(true);

        ++loaded;
        total_floats += numel;
    }

    // Sanity: lm_head.weight was NOT in the file (tied case). Untied case
    // would need a separate dump entry; flag if model is in untied mode and
    // we loaded an init that didn't include lm_head — caller can decide.
    if (!model.config.weight_tying && model.lm_head) {
        if (is_master) std::cerr << "[init-load] WARN: weight_tying=false but file "
                                    "contains no lm_head.weight — lm_head retains its "
                                    "in-process random init." << std::endl;
    }

    if (is_master) {
        std::cout << "[init-load] loaded " << loaded << " / " << num_tensors
                  << " tensors (" << total_floats << " floats, "
                  << (total_floats * sizeof(float)) / (1024 * 1024) << " MiB) from "
                  << path << std::endl;
    }
}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const bool is_master = (rank == 0);

    // Runtime SDPA path selector. Default: unfused (baseline).
    // Set USE_PACKED_SDPA=1 to enable scaled_dot_product_attention_packed.
    if (const char* env = std::getenv("USE_PACKED_SDPA")) {
        g_use_packed_sdpa = (env[0] == '1');
    }
    // Within the unfused path: USE_QKV_RESHAPE_FIRST=1 reshapes qkv to
    // [B,T,H,3d] (free view) before sharding axis 3, eliminating the 3
    // contiguous copies the original shard-axis-2 → reshape route forces.
    if (const char* env = std::getenv("USE_QKV_RESHAPE_FIRST")) {
        g_use_qkv_reshape_first = (env[0] == '1');
    }
    if (is_master) {
        std::cout << "Attention path: ";
        if (g_use_packed_sdpa) {
            std::cout << "PACKED-QKV (fused SDPA)";
        } else if (g_use_qkv_reshape_first) {
            std::cout << "UNFUSED + reshape-first (free view, shard axis 3)";
        } else {
            std::cout << "UNFUSED baseline (shard axis 2 + reshape + transpose + SDPA)";
        }
        std::cout << std::endl;
    }

        // autograd::GraphRecordGuard niggesh(true, "jebam.txt");
        // autograd::g_shape_debug = true;
    
    
    // ---- One GPU per rank ----
    cudaSetDevice(rank);
    DeviceIndex device(Device::CUDA, rank);
    // Skip process group / DDP entirely on single-GPU to avoid the wrapper's overhead.
    std::shared_ptr<ProcessGroupNCCL> pg;
    if (world_size > 1) {
        pg = init_process_group(world_size, rank);
    }
    // Configuration
    GPTConfig config;
    config.context_length = 1024;
    config.vocab_size = 50304;
    config.n_embd = 768;
    config.n_layers = 12;
    config.n_heads = 12;       // Proper multi-head attention
    config.weight_tying = true;
    config.checkpointing = true; // Toggle weight tying Here
    config.if_log = true;
    
    // Training hyperparameters (NanoGPT speedrun — matches gpt2.py)
    const int B = 16;          // Micro batch size (per rank)
    const int T = 1024;        // Sequence length
    const int GLOBAL_BATCH = 524288;  // 2**19, ~0.5M tokens (total across ranks)
    const int GRAD_ACCUM_STEPS = GLOBAL_BATCH / (B * T * world_size);

    const float MAX_LR = 6e-4f;
    const float MIN_LR = MAX_LR * 0.1f;

    // const int WARMUP_STEPS = 715;
    // const int MAX_STEPS = 19073;  // ~1 epoch on 10B tokens at 0.5M batch

    const int WARMUP_STEPS = 715;
    const int MAX_STEPS = 19073;  // ~1 epoch on 10B tokens at 0.5M batch

    const int VAL_FREQ = 250;
    const int TOK_GEN_FREQ = 5000;
    const int CKPT_FREQ = 1300;

    // Target loss thresholds for speedrun timing (NanoGPT target)
    const float TARGET_TRAIN_LOSS = 3.28f;
    const float TARGET_VAL_LOSS = 3.28f;
    if(is_master){
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
    }
    
    std::cout << "\nInitializing model on CUDA device " << rank << "..." << std::endl;
    
    // Create model
    GPT model(config, device);

    // Optional init parity: if INIT_FROM_BIN points at a gpt2_init.bin produced
    // by dump_init.py (same seed, same vocab, weight_tying=true), overwrite
    // the in-process random init with the dumped values. This makes the C++
    // trainer start from the EXACT same point in parameter space as gpt2.py
    // — eliminates RNG-stream divergence between std::mt19937 and torch's RNG
    // as a source of drift in like-for-like comparisons. See dump_init.py and
    // like_for_like_plan.txt for the full protocol.
    if (const char* p = std::getenv("INIT_FROM_BIN")) {
        if (p[0] != '\0') {
            try {
                load_init_bin(model, p, is_master);
            } catch (const std::exception& e) {
                if (is_master) std::cerr << "[init-load] FATAL: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            // Init-parity DIAG: dump first 8 floats of two parameters so we
            // can byte-compare against dump_init.py's manifest. Print on
            // master only — every rank loaded the same file so values are
            // identical. wte.weight is contiguous on device → to_cpu() then
            // pull the first 8 floats. h.0.attn.c_attn.weight is stored
            // [in,out] in the C++ lib so its first 8 floats are the same
            // ones gpt2.py prints from the transposed weight (gpt2.py:354).
            if (is_master) {
                Tensor wte_cpu = model.wte.weight.to_cpu();
                const float* w = wte_cpu.data<float>();
                std::cout << "[DIAG] wte.weight first 8: ";
                for (int i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << w[i] << " ";
                std::cout << std::endl;

                Tensor c0_cpu = model.attn_blocks[0]->c_attn.weight.to_cpu();
                const float* c = c0_cpu.data<float>();
                std::cout << "[DIAG] h.0.attn.c_attn.weight first 8: ";
                for (int i = 0; i < 8; ++i)
                    std::cout << std::fixed << std::setprecision(6) << c[i] << " ";
                std::cout << std::endl;
            }
        }
    }

    // Print parameter count
    auto params = model.parameters();
    int64_t num_params = 0;
    for(auto& p : params) num_params += p.numel();

    

    // === DEBUG: print wte.weight gradient on each AccumulateGrad call ===
    // With weight_tying=true, wte.weight has TWO backward paths feeding it:
    //   (a) tok_emb path  : embedding_backward scatters into wte.weight.grad
    //   (b) lm_head path  : matmul backward -> transpose backward -> wte.weight.grad
    // GradAccumulator is invoked once per path; the post-acc hook fires after
    // each accumulation. So we expect the hook to fire twice per backward:
    //   1st fire : grad == (whichever path finished first — usually lm_head,
    //              since its backward chain is much shorter)
    //   2nd fire : grad == sum of both paths (verifies accumulation is wired)
    // Capped at 2 prints so micro-steps / later steps don't drown stdout.
    // if (is_master && config.weight_tying) {
    //     auto debug_counter = std::make_shared<int>(0);
    //     model.wte.weight.register_post_acc_hook(make_post_acc_hook(
    //         [debug_counter](const Tensor& grad) {
    //             if (*debug_counter >= 2) return;
    //             (*debug_counter)++;
    //             std::cout << "\n[WT-GRAD-DEBUG] wte.weight accumulation #"
    //                       << *debug_counter
    //                       << ((*debug_counter == 1)
    //                             ? " — first path landed (likely lm_head contribution alone):"
    //                             : " — second path landed (should be SUM of tok_emb + lm_head):")
    //                       << std::endl;
    //             grad.display();
    //         }
    //     ));
    // }

    std::unique_ptr<DataParallel> ddp_model;
    if (world_size > 1) {
        DDP_Options ddp_opts;
        ddp_opts = ddp_opts
            .with_process_group(pg)
            .with_world_size(world_size)
            .with_bucket_data(/*bucket=*/true, /*bucket_size_bytes=*/25 * 1024 * 1024)
            .with_grad_view(false);

        ddp_model = std::make_unique<DataParallel>(&model, ddp_opts, /*init_sync=*/false);
        // DDP broadcast just overwrote wte.weight with rank 0's values —
    }

    if(is_master) std::cout << "Number of parameters: " << num_params << std::endl;
    if (!config.weight_tying && is_master) {
        std::cout << "(Note: More params than weight-tied version due to separate lm_head)" << std::endl;
    }
    
    // Optim groups (mirrors gpt2.py:configure_optimizers).
    //   decay_params   : dim >= 2  → matmul weights + embeddings, weight_decay=0.1
    //   nodecay_params : dim  < 2  → biases + LayerNorm scales,    weight_decay=0.0
    // Same selection rule PyTorch uses: `p.dim() >= 2`.
    const float WEIGHT_DECAY = 0.1f;
    std::vector<Tensor> decay_params, nodecay_params;
    decay_params.reserve(params.size());
    nodecay_params.reserve(params.size());
    int64_t n_decay = 0, n_nodecay = 0;
    for (const auto& p : params) {
        if (p.shape().dims.size() >= 2) {
            decay_params.push_back(p);
            n_decay += p.numel();
        } else {
            nodecay_params.push_back(p);
            n_nodecay += p.numel();
        }
    }
    if (is_master) {
        std::cout << "num decayed parameter tensors: " << decay_params.size()
                  << ", with " << n_decay << " parameters" << std::endl;
        std::cout << "num non-decayed parameter tensors: " << nodecay_params.size()
                  << ", with " << n_nodecay << " parameters" << std::endl;
    }

    // PyTorch's torch.optim.AdamW takes per-group `weight_decay` directly via
    // optim_groups; the C++ AdamW here is single-wd-per-instance and two
    // back-to-back instances race on multi_tensor_adam_cuda's static pinned
    // host metadata buffers, so we replicate the per-group behavior with one
    // AdamW(wd=0) plus a manual stepweight-decay pass on the decay group only.
    //
    // Bit-equivalence to PyTorch's _single_tensor_adamw:
    //   PyTorch (per-step, decay group):
    //     param.mul_(1 - lr * wd)              ← stepweight decay, FIRST
    //     exp_avg.mul_(beta1).add_(grad, ...)  ← moments depend only on grad
    //     exp_avg_sq.mul_(beta2).addcmul_(...)
    //     param.addcdiv_(exp_avg, denom, -lr / (1-beta1^t))
    //   C++ (per-step, decay group):
    //     for p in decay_params: p *= (1 - lr * wd)   ← stepweight decay, FIRST
    //     optimizer.step()  (wd=0 inside → only the moment update + addcdiv)
    // Reordering the stepweight outside step() is bit-identical because the
    // Adam moment update reads only `grad`, never `param`.
    nn::AdamW optimizer(params, MAX_LR, 0.9f, 0.95f, 1e-8f, /*weight_decay=*/0.0f);
    
    // Create data loaders
    std::string data_root = "/mnt/volgrp03/3rd_floor/edu_fineweb10B_bin";
    DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, true, 100000000, rank);
    DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true, 100000000, rank);

    // Data identity diagnostic: dump first 32 input tokens from rank 0's first
    // batch, then reset so training itself is unaffected. Compare against the
    // PyTorch run's matching dump to confirm both pipelines see identical bytes.
    if (is_master) {
        Batch diag = train_loader.next_batch();
        Tensor diag_cpu = diag.input.to_cpu();
        const uint16_t* p = diag_cpu.data<uint16_t>();
        std::cout << "[DIAG] first 32 input tokens (rank 0): ";
        for (int i = 0; i < 32; ++i) std::cout << p[i] << " ";
        std::cout << std::endl;
    }
    train_loader.reset();

    // Load HellaSwag val once. Every rank loads (file is small, ~10 MB) so the
    // eval loop can shard examples by index without a separate broadcast.
    // If the binary is missing the eval is skipped with a warning instead of
    // aborting training.
    std::vector<hellaswag::Example> hellaswag_examples;
    try {
        hellaswag_examples = hellaswag::load_binary("hellaswag/hellaswag_val.bin");
        if (is_master)
            std::cout << "loaded " << hellaswag_examples.size()
                      << " HellaSwag val examples" << std::endl;
    } catch (const std::exception& e) {
        if (is_master)
            std::cerr << "[WARN] HellaSwag eval disabled: " << e.what()
                      << "\n[WARN] Run: python prep_hellaswag.py" << std::endl;
    }

    std::cout << "\nStarting training..." << std::endl;



    CheckpointManager ckpt_manager("checkpoints", "gpt2", 5, rank, false, false);


    
    // No interval-based auto-saves: the val block below does an explicit
    // `.save()` matching gpt2.py's structure (save lives in the val block,
    // before the optimizer step). This guarantees `gpt2_step_N.ckpt` and the
    // step-N eval reflect the SAME model state.

    int start_step = 0;
    float latest_loss = 0.0f;

    // Auto-resume if checkpoint exists. With the new save semantics
    // (checkpoint at step N == model state BEFORE iteration N's training),
    // resuming means: skip N*GRAD_ACCUM_STEPS batches consumed by steps 0..N-1
    // and start the loop at step N. No `start_step++`.
    if (config.checkpointing && ckpt_manager.load_latest(model, optimizer, start_step, latest_loss)) {
        std::cout << "[Resume] Continuing from step " << start_step << " with loss " << latest_loss << std::endl;
        size_t batches_to_skip = static_cast<size_t>(start_step) * GRAD_ACCUM_STEPS;
        if (batches_to_skip > 0) {
            std::cout << "[Resume] Skipping " << batches_to_skip << " batches..." << std::endl;
            train_loader.skip_batches(batches_to_skip);
        }
    }
        
    // Create CSV log file (master only — avoid multi-rank truncation races)
    std::ofstream log_file;
    if(is_master && config.if_log) {
        log_file.open("gpt2_may16_3xtf32_fullrun_sundayeve_Rformula_no3xtf32.csv");
        log_file << "step,loss,val_loss,hellaswag_acc,lr,grad_norm,dt_ms,tok_per_sec,elapsed_min\n";
        log_file << std::fixed << std::setprecision(6);
        log_file.flush();  // ensure header survives if training is killed mid-step
    }


    float val_loss_accum_log = -1.0f;     // -1 indicates no validation this step
    float hellaswag_acc_log = -1.0f;      // -1 indicates no hellaswag eval this step

    // Pre-allocate training loop tensors (avoids per-step allocation)
    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                        1.0f / static_cast<float>(GRAD_ACCUM_STEPS));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

    // Target loss tracking state
    bool train_loss_reached = false;
    bool val_loss_reached = false;
    auto training_start = std::chrono::high_resolution_clock::now();

    for (int step = start_step; step < MAX_STEPS; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // Validation every VAL_FREQ steps
        if (step % VAL_FREQ == 0 || step == MAX_STEPS - 1) {
            autograd::NoGradGuard no_grad;  // No backward graph needed for validation
            val_loader.reset();
            float val_loss_accum = 0.0f;
            int val_loss_steps = 20;

            for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                Batch batch = val_loader.next_batch();

                Tensor logits = model.forward(batch.input);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
                
                Tensor loss_cpu = loss.to_cpu();
                val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
            }
            
            // Average validation loss across all ranks (skip the collective on single-GPU)
            float global_val_loss = val_loss_accum;
            if (world_size > 1) {
                MPI_Allreduce(&val_loss_accum, &global_val_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                global_val_loss /= world_size;
            }

            if(is_master) std::cout << "validation loss: " << std::fixed << std::setprecision(4) << global_val_loss << std::endl;
            val_loss_accum_log = global_val_loss;

            // Check if target val loss reached
            if (!val_loss_reached && global_val_loss <= TARGET_VAL_LOSS && is_master) {
                val_loss_reached = true;
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_min = std::chrono::duration<double>(now - training_start).count() / 60.0;
                std::cout << ">>> TARGET VAL LOSS " << TARGET_VAL_LOSS
                            << " reached at step " << step
                            << " | val_loss=" << std::fixed << std::setprecision(6) << global_val_loss
                            << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed_min << " min"
                            << std::endl;
                if(config.if_log) log_file << "# TARGET_VAL_LOSS=" << TARGET_VAL_LOSS
                            << " reached at step " << step
                            << " val_loss=" << global_val_loss
                            << " elapsed_min=" << elapsed_min << "\n";
                if(config.if_log) log_file.flush();
            }

            // HellaSwag eval (matches gpt2.py:507-538). Sharded across ranks
            // by example index, counts MPI-allreduced, master logs accuracy.
            // Stash into hellaswag_acc_log so this step's CSV row carries the
            // value; non-eval steps write -1 in that column.
            if (!hellaswag_examples.empty()) {
                double hella_acc = hellaswag::evaluate(
                    model, device, rank, world_size,
                    hellaswag_examples, is_master, step);
                hellaswag_acc_log = static_cast<float>(hella_acc);
            }

            // Checkpoint save (matches gpt2.py:494-505 — save lives inside
            // the val block, BEFORE the iteration's optimizer.step()). This
            // way gpt2_step_N.ckpt is the same model state the eval at step
            // N just measured: load it back, re-run hellaswag.py, get the
            // same accuracy reported above.
            if (config.checkpointing && (step % CKPT_FREQ == 0 || step == MAX_STEPS - 1)) {
                try {
                    ckpt_manager.save(step, model, optimizer, val_loss_accum_log);
                } catch (const std::exception& e) {
                    if (is_master)
                        std::cerr << "[WARN] checkpoint save at step " << step
                                  << " failed: " << e.what() << std::endl;
                }
            }
        }
        
        // token generation — run on master only so non-master ranks don't enter
        // the same path concurrently (rules out per-rank races + isolates the
        // crash to a single rank's stderr).
        // if(is_master && (step % TOK_GEN_FREQ == 0 || step == MAX_STEPS - 1)) {
        //         autograd::NoGradGuard no_grad;  // No backward graph needed for generation
        //         std::cout << "--- Generating tokens at step " << step << " ---" << std::endl;
        //         int num_return_sequence = 4;
        //         int max_length = 60;

        //         Tensor xgen = Tensor(Shape({{num_return_sequence, 5}}),
        //                                    TensorOptions().with_dtype(Dtype::Int64).with_device(device));
        //         static const std::vector<int64_t> xgen_tokens = {31373, 616, 1438, 318, 11, 31373, 616, 1438, 318, 11, 31373, 616, 1438, 318, 11, 31373, 616, 1438, 318, 11};
        //         xgen.set_data(xgen_tokens);

        //         uint64_t gen_seed = 42 + rank;
        //         while (xgen.shape().dims[1] < max_length) {
        //             Tensor logits = model.forward(xgen);

        //             int64_t B = logits.shape().dims[0];
        //             int64_t T = logits.shape().dims[1];

        //             // Zero-copy view: logits[:, -1, :] -> (B, 1, V)
        //             // Replaces gather which allocated a [B, 1, V] Int64 index tensor per step
        //             // NOTE: softmax CUDA fast-path assumes contiguous row-major layout;
        //             // narrow_view leaves dim-0 stride = T*V, so we MUST materialize a
        //             // contiguous copy before softmax — otherwise rows 1..B-1 read
        //             // neighbouring batches' logits and can surface NaN/Inf in probs.
        //             Tensor last_logits_3d = logits.narrow_view(1, T - 1, 1).contiguous();

        //             // ---- Top-k sampling computed on CPU for numerical robustness ----
        //             // The fused CUDA softmax kernel (SoftmaxKernels.cu) occasionally
        //             // produces NaN for one row when applied over V=50304; since generation
        //             // is cheap (B*V floats per step), we do top-k + softmax on the CPU
        //             // directly from logits. This sidesteps the kernel bug entirely.
        //             constexpr int TOPK = 50;
        //             Tensor logits_cpu = last_logits_3d.to_cpu();           // [B,1,V] float32
        //             const float* lg = logits_cpu.data<float>();
        //             int64_t V = last_logits_3d.shape().dims.back();

        //             // Output: next-token indices (B, 1) Int64 on CPU
        //             Tensor next_token_cpu(Shape{{B, 1}},
        //                                   TensorOptions().with_dtype(Dtype::Int64));
        //             int64_t* nt = next_token_cpu.data<int64_t>();

        //             std::mt19937 rng(static_cast<uint32_t>(gen_seed++));
        //             std::uniform_real_distribution<float> uni(0.0f, 1.0f);

        //             for (int64_t b = 0; b < B; ++b) {
        //                 const float* row = lg + b * V;

        //                 // Partial sort for top-k by logit value
        //                 std::vector<int64_t> idx(V);
        //                 std::iota(idx.begin(), idx.end(), 0);
        //                 std::partial_sort(
        //                     idx.begin(), idx.begin() + TOPK, idx.end(),
        //                     [row](int64_t a, int64_t b) { return row[a] > row[b]; });

        //                 // Numerically-stable softmax over the top-k only
        //                 float mx = row[idx[0]];
        //                 double sum = 0.0;
        //                 std::array<double, TOPK> w{};
        //                 for (int k = 0; k < TOPK; ++k) {
        //                     w[k] = std::exp(static_cast<double>(row[idx[k]] - mx));
        //                     sum += w[k];
        //                 }
        //                 // Sample via inverse-CDF
        //                 double r = static_cast<double>(uni(rng)) * sum;
        //                 double acc = 0.0;
        //                 int chosen = TOPK - 1;
        //                 for (int k = 0; k < TOPK; ++k) {
        //                     acc += w[k];
        //                     if (r <= acc) { chosen = k; break; }
        //                 }
        //                 nt[b] = idx[chosen];
        //             }

        //             Tensor next_token = next_token_cpu.to(device);

        //             xgen = Tensor::cat({xgen, next_token}, 1);
        //         }

        //         // Print generated tokens
        //         Tensor xgen_cpu = xgen.to_cpu();
        //         int64_t* data = xgen_cpu.data<int64_t>();
        //         int64_t B = xgen.shape().dims[0];
        //         int64_t T = xgen.shape().dims[1];

        //         for (int i = 0; i < B; ++i) {
        //             std::cout << "sample" << i << "= \"";
        //             for (int j = 0; j < T; ++j) {
        //                 std::cout << data[i * T + j] << " ";
        //             }
        //             std::cout << "\""<<std::endl;
        //         }
        //     }



        // Training step

        optimizer.zero_grad();
        // optimizer.zero_grad();
        // optimizer.zero_grad();  // old comment
        // for(auto& param : params){
        //     if (void* g = param.grad()) {                    // skip null grads on first step
        //         cudaMemsetAsync(g, 0, param.nbytes());
        //     }
        // }
        cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[step " << step << "] CUDA sticky error after zero_grad: "
                << cudaGetErrorString(err) << std::endl;
    }


        float loss_accum = 0.0f;

        // Reset pre-allocated GPU accumulator in-place (no new allocation)
        loss_accum_gpu *= 0.0f;
        if (ddp_model) ddp_model->no_sync();
        for (int micro_step = 0; micro_step < GRAD_ACCUM_STEPS; ++micro_step) {
            Batch batch = train_loader.next_batch();

            Tensor logits = ddp_model ? ddp_model->forward(batch.input)
                                      : model.forward(batch.input);
            Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);

            // In-place accumulate on GPU (no new tensor allocation per micro-step)
            loss_accum_gpu += loss.detach();
            if (ddp_model && micro_step == GRAD_ACCUM_STEPS - 1) ddp_model->sync();
            // Backward with scaling
            loss.backward(&grad_scale);
        }
        
        // ONE sync after all micro-steps complete
        {
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            float local_loss = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
            if (world_size > 1) {
                MPI_Allreduce(&local_loss, &loss_accum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                loss_accum /= world_size;
            } else {
                loss_accum = local_loss;
            }
        }
        
        // NaN detection - early exit if training goes unstable
        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
            if(config.if_log) log_file.close();
            return 1;
        }

        // Check if target train loss reached
        if (!train_loss_reached && loss_accum <= TARGET_TRAIN_LOSS && is_master) {
            train_loss_reached = true;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_min = std::chrono::duration<double>(now - training_start).count() / 60.0;
            std::cout << ">>> TARGET TRAIN LOSS " << TARGET_TRAIN_LOSS
                        << " reached at step " << step
                        << " | train_loss=" << std::fixed << std::setprecision(6) << loss_accum
                        << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed_min << " min"
                        << std::endl;
            if(config.if_log) log_file << "# TARGET_TRAIN_LOSS=" << TARGET_TRAIN_LOSS
                        << " reached at step " << step
                        << " train_loss=" << loss_accum
                        << " elapsed_min=" << elapsed_min << "\n";
            if(config.if_log) log_file.flush();
        }

        // for(auto& p : params){
        //     p.grad_view().display();
        // }


        // Clip gradients
        float norm = nn::clip_grad_norm_(params, 1.0f);
        
        // Update learning rate
        float lr = get_lr(step, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS);
        optimizer.set_lr(lr);

        // Stepweight decay on the wd=0.1 group only (decay_params, dim >= 2).
        // Bit-identical to PyTorch AdamW with optim_groups[{wd:0.1},{wd:0.0}]
        // because the Adam moment update reads only `grad`, never `param` —
        // see the constructor-site comment for the line-by-line equivalence.
        // The nodecay group needs no work: factor would be 1.0.
        const float wd_factor = 1.0f - lr * WEIGHT_DECAY;
        for (auto& p : decay_params) p *= wd_factor;

        // Optimizer step (wd=0 inside; pure Adam update on every param).
        optimizer.step();

        // (No checkpoint save here — saves live in the val block above so the
        // checkpoint state matches the eval state, mirroring gpt2.py.)
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        
        // Compute throughput
        int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS * world_size;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
        
        // Print training info
        if(is_master) std::cout << "step " << std::setw(5) << step 
                    << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                    << " | lr " << std::scientific << std::setprecision(4) << lr 
                    << " | norm: " << std::fixed << std::setprecision(4) << norm 
                    << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                    << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                    << std::endl;
        
        // Log metrics to CSV
        double elapsed_total_min = std::chrono::duration<double>(t1 - training_start).count() / 60.0;
        if(is_master && config.if_log) log_file << step << ","
                    << loss_accum << ","
                    << val_loss_accum_log << ","
                    << hellaswag_acc_log << ","
                    << lr << ","
                    << norm << ","
                    << (dt * 1000.0) << ","
                    << tokens_per_sec << ","
                    << elapsed_total_min << "\n";
        if(is_master && config.if_log) log_file.flush();
        val_loss_accum_log = -1.0f;     // Reset for next iteration
        hellaswag_acc_log = -1.0f;

        if (step == 0) {
            CachingCUDAAllocator::instance().empty_cache();
        }
    }
    if(is_master && config.if_log) log_file.close();
    
    
    if(is_master) std::cout << "\n=== Training Complete ===" << std::endl;

    // CachingCUDAAllocator::instance().print_memory_summary();

    // Flush any in-flight async save and reset the manager BEFORE MPI_Finalize
    // so the destructor's rename / cuda-stream-destroy can't fight with MPI's
    // teardown. Wrapped in try/catch as belt-and-braces against a stale
    // rename surfacing during shutdown.
    try {
        ckpt_manager.wait_for_completion();
    } catch (const std::exception& e) {
        if (is_master)
            std::cerr << "[WARN] checkpoint flush at shutdown: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}