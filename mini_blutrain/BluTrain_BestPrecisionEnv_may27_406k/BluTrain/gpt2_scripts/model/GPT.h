#pragma once
// =============================================================================
// GPT.h — Standard GPT-2 Model (single GPU / DDP)
//
// Uses nn::Module-based Attention and MLP blocks.
// Attention backend is selected via config::AttentionType.
// =============================================================================

#include <cstdint>
#include <memory>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "checkpointing/GradMode.h"
#include "checkpointing/Checkpoint.h"

#include "../config/ConfigParser.h"
#include "common.h"
#include "Embedding.h"
#include "Attention.h"
#include "MLP.h"

using namespace OwnTensor;

namespace gpt2 {

class GPT : public nn::Module {
public:
    config::TrainConfig train_cfg;
    Embedding wte;
    Embedding wpe;
    std::vector<std::shared_ptr<Attention>> attn_blocks;
    std::vector<std::shared_ptr<MLP>> mlp_blocks;
    nn::LayerNorm ln_f;
    std::shared_ptr<nn::Linear> lm_head;

    GPT(const config::TrainConfig& cfg, DeviceIndex device, uint64_t seed = 1337)
        : train_cfg(cfg),
          wte(cfg.vocab_size, cfg.n_embd, device, seed),
          wpe(cfg.context_length(), cfg.n_embd, device, seed + 100),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);

        for (int i = 0; i < cfg.n_layers; ++i) {
            auto a = std::make_shared<Attention>(
                cfg.n_embd, cfg.n_heads, cfg.n_layers,
                cfg.attention, device, seed + 200 + i * 10);
            auto m = std::make_shared<MLP>(
                cfg.n_embd, cfg.n_layers, device, seed + 200 + i * 10);
            attn_blocks.push_back(a);
            mlp_blocks.push_back(m);
            register_module(a.get());
            register_module(m.get());
        }

        // LM head: weight-tied or independent
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
        }

        // Cache position indices on GPU
        Tensor pos_cpu(Shape{{1, cfg.context_length()}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* pos_data = pos_cpu.data<int64_t>();
        for (int64_t i = 0; i < cfg.context_length(); ++i) pos_data[i] = i;
        cached_pos_ = pos_cpu.to(device);

        register_module(wte);
        register_module(wpe);
        register_module(ln_f);
        if (!cfg.weight_tying && lm_head) {
            register_module(lm_head.get());
        }
    }

    Tensor forward(const Tensor& idx) override {
        Tensor tok_emb = wte.forward(idx);
        int64_t T = idx.shape().dims[1];

        Tensor pos_indices = (T == train_cfg.context_length())
            ? cached_pos_
            : cached_pos_.narrow_view(1, 0, T);
        Tensor pos_emb = wpe.forward(pos_indices);

        Tensor x = autograd::add(tok_emb, pos_emb);

        for (int i = 0; i < train_cfg.n_layers; ++i) {
            if (train_cfg.activation_checkpointing && autograd::GradMode::is_enabled()) {
                auto& attn = attn_blocks[i];
                auto& mlp  = mlp_blocks[i];
                auto attn_fn = [&attn](const variable_list& ins) -> variable_list {
                    return { attn->forward(ins[0]) };
                };
                auto mlp_fn = [&mlp](const variable_list& ins) -> variable_list {
                    return { mlp->forward(ins[0]) };
                };
                x = autograd::checkpoint(attn_fn, {x})[0];
                x = autograd::checkpoint(mlp_fn,  {x})[0];
            } else {
                x = attn_blocks[i]->forward(x);
                x = mlp_blocks[i]->forward(x);
            }
        }

        x = ln_f.forward(x);
        Tensor logits = lm_head->forward(x);
        return logits;
    }

private:
    Tensor cached_pos_;
};

}  // namespace gpt2
