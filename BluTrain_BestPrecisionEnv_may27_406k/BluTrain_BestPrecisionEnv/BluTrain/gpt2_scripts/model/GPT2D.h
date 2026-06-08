#pragma once
// =============================================================================
// GPT2D.h — GPT-2 Model with 2D Parallelism (DDP + TP)
//
// Uses VPEmbedding (vocab-parallel), TpAttention (column/row-parallel),
// TpMLP (column/row-parallel), and column-parallel LM head.
// =============================================================================

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/NN.h"
#include "checkpointing/GradMode.h"
#include "checkpointing/Checkpoint.h"
#include "../../dist/distributed.h"

#include "../config/ConfigParser.h"
#include "common.h"
#include "Embedding.h"
#include "Attention.h"
#include "MLP.h"
#include "../parallelism/tp_ops.h"

using namespace OwnTensor;

namespace gpt2 {

// =============================================================================
// VPEmbedding implementation (needs ProcessGroupNCCL, defined here)
// =============================================================================

inline VPEmbedding::VPEmbedding(int64_t vocab_size, int64_t n_embd,
                                 std::shared_ptr<ProcessGroupNCCL>& tp_pg,
                                 DeviceIndex device, uint64_t seed)
    : n_embd_(n_embd), tp_pg_(tp_pg)
{
    int tp_size = tp_pg->get_worldsize();
    int tp_rank = tp_pg->get_rank();
    local_v_ = vocab_size / tp_size;
    vocab_start_ = tp_rank * local_v_;
    vocab_end_ = (tp_rank == tp_size - 1) ? vocab_size : vocab_start_ + local_v_;
    local_v_ = vocab_end_ - vocab_start_;

    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor full = Tensor::randn<float>(Shape{{vocab_size, n_embd}}, cpu_opts, seed, 0.02f);
    weight = full.narrow(0, vocab_start_, local_v_).contiguous();
    weight = weight.to(device);
    weight.set_requires_grad(true);
}

inline Tensor VPEmbedding::forward(const Tensor& indices) {
    auto [partial, work] = forward_async(indices);
    tp::allreduce_wait(work);
    return partial;
}

inline std::pair<Tensor, std::shared_ptr<Work>> VPEmbedding::forward_async(
    const Tensor& indices) {
    Tensor idx_i32 = indices.as_type(Dtype::Int32);
    Tensor mask_ge = (idx_i32 >= static_cast<int32_t>(vocab_start_)).as_type(Dtype::Float32);
    Tensor mask_lt = (idx_i32 < static_cast<int32_t>(vocab_end_)).as_type(Dtype::Float32);
    Tensor mask = autograd::mul(mask_ge, mask_lt);

    Tensor local_idx = (idx_i32 - static_cast<int32_t>(vocab_start_));
    Tensor local_safe = (local_idx.as_type(Dtype::Float32) * mask).as_type(Dtype::UInt16);

    Tensor embeds = autograd::embedding(weight, local_safe);
    std::vector<int64_t> mask_dims = mask.shape().dims;
    mask_dims.push_back(1);
    Tensor mask_3d = autograd::reshape(mask, Shape{mask_dims});
    Tensor partial = autograd::mul(embeds, mask_3d);

    auto work = tp::allreduce_launch(partial, tp_pg_);
    return {partial, work};
}

// =============================================================================
// GPT2D: 2D Parallel GPT-2 model
// =============================================================================

class GPT2D {
public:
    config::TrainConfig train_cfg;
    VPEmbedding wte;
    PosEmbedding wpe;
    std::vector<TpAttention> attn_blocks;
    std::vector<TpMLP> mlp_blocks;
    nn::LayerNorm ln_f;
    Tensor w_lm_head;  // Column-parallel LM head: [n_embd, vocab/tp]

    GPT2D(const config::TrainConfig& cfg, DeviceMesh& mesh,
          std::shared_ptr<ProcessGroupNCCL>& tp_pg,
          DeviceIndex device, uint64_t seed = 1234)
        : train_cfg(cfg),
          wte(cfg.vocab_size, cfg.n_embd, tp_pg, device, seed),
          wpe(cfg.context_length(), cfg.n_embd, device, seed + 100),
          ln_f(cfg.n_embd),
          mesh_(mesh), tp_pg_(tp_pg)
    {
        int tp_size = tp_pg->get_worldsize();
        int tp_rank = tp_pg->get_rank();
        local_vocab_ = cfg.vocab_size / tp_size;

        ln_f.to(device);

        for (int i = 0; i < cfg.n_layers; ++i) {
            attn_blocks.emplace_back(cfg.n_embd, cfg.n_heads, cfg.n_layers,
                                     cfg.attention, tp_pg, device,
                                     seed + 200 + i * 10);
            mlp_blocks.emplace_back(cfg.n_embd, cfg.n_layers, tp_pg, device,
                                    seed + 200 + i * 10 + 5);
        }

        // Column-parallel LM head
        {
            TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
            Tensor w_full = Tensor::randn<float>(
                Shape{{cfg.n_embd, cfg.vocab_size}}, cpu_opts, seed + 1000, 0.02f);
            w_lm_head = w_full.narrow(1, tp_rank * local_vocab_, local_vocab_)
                             .contiguous().to(device);
            w_lm_head.set_requires_grad(true);
        }

        // Cache position indices
        Tensor pos_cpu(Shape{{1, cfg.context_length()}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* pos_data = pos_cpu.data<int64_t>();
        for (int64_t i = 0; i < cfg.context_length(); ++i) pos_data[i] = i;
        cached_pos_ = pos_cpu.to(device);
    }

    DTensor forward(const Tensor& idx) {
        int64_t B = idx.shape().dims[0];
        int64_t T = idx.shape().dims[1];

        // Overlapped: token embedding AR + position embedding
        auto [tok_emb, tok_ar_work] = wte.forward_async(idx);
        Tensor pos_indices = (T == train_cfg.context_length())
            ? cached_pos_
            : cached_pos_.narrow_view(1, 0, T);
        Tensor pos_emb = wpe.forward(pos_indices);
        tp::allreduce_wait(tok_ar_work);

        Tensor x = autograd::add(tok_emb, pos_emb);

        for (int i = 0; i < train_cfg.n_layers; ++i) {
            if (train_cfg.activation_checkpointing && autograd::GradMode::is_enabled()) {
                auto& attn = attn_blocks[i];
                auto& mlp  = mlp_blocks[i];
                auto attn_fn = [&attn](const variable_list& ins) -> variable_list {
                    Tensor in = ins[0];
                    return { attn.forward(in) };
                };
                auto mlp_fn = [&mlp](const variable_list& ins) -> variable_list {
                    Tensor in = ins[0];
                    return { mlp.forward(in) };
                };
                x = autograd::checkpoint(attn_fn, {x})[0];
                x = autograd::checkpoint(mlp_fn,  {x})[0];
            } else {
                x = attn_blocks[i].forward(x);
                x = mlp_blocks[i].forward(x);
            }
        }

        x = ln_f.forward(x);

        Tensor logits_local = tp::column_parallel_matmul(x, w_lm_head, tp_pg_);

        Layout logits_layout(mesh_, {B, T, local_vocab_});
        DTensor logits_dt(mesh_, tp_pg_, logits_layout, "logits_2d");
        logits_dt.mutable_tensor() = logits_local;

        return logits_dt;
    }

    std::vector<Tensor> collect_params() {
        std::vector<Tensor> params;
        params.push_back(wte.weight);
        params.push_back(wpe.weight);
        for (int i = 0; i < train_cfg.n_layers; ++i) {
            for (auto* p : attn_blocks[i].param_ptrs()) params.push_back(*p);
            for (auto* p : mlp_blocks[i].param_ptrs()) params.push_back(*p);
        }
        params.push_back(ln_f.weight);
        params.push_back(ln_f.bias);
        params.push_back(w_lm_head);
        return params;
    }

    void zero_grad() {
        for (auto& p : collect_params()) {
            if (p.requires_grad() && p.has_grad()) p.zero_grad();
        }
    }

private:
    DeviceMesh& mesh_;
    std::shared_ptr<ProcessGroupNCCL> tp_pg_;
    int64_t local_vocab_ = 0;
    Tensor cached_pos_;
};

}  // namespace gpt2
