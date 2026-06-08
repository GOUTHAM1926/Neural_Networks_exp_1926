#pragma once
// =============================================================================
// Attention.h — Multi-Head Causal Self-Attention variants
//
//   - Attention:    Standard module (single GPU / DDP) with configurable backend
//   - TpAttention:  Tensor-parallel attention (column-parallel QKV, row-parallel proj)
//
// The attention backend (normal, fmha, fused_tril_softmax) is selected via
// config::AttentionType passed at construction time.
// =============================================================================

#include <cmath>
#include <limits>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/TrilOps.h"
#include "nn/NN.h"
#include "checkpointing/GradMode.h"

#include "../config/ConfigParser.h"
#include "common.h"
#include "../parallelism/tp_ops.h"

using namespace OwnTensor;

namespace gpt2 {

// =============================================================================
// Standard Attention (single GPU / DDP)
// =============================================================================
class Attention : public nn::Module {
public:
    nn::LayerNorm ln;
    nn::Linear c_attn;   // QKV: [n_embd] -> [3 * n_embd]
    nn::Linear c_proj;   // Output: [n_embd] -> [n_embd]

    Attention(int64_t n_embd, int n_heads, int n_layers,
              config::AttentionType attn_type, DeviceIndex device,
              uint64_t seed = 1234)
        : ln(n_embd),
          c_attn(n_embd, 3 * n_embd, true),
          c_proj(n_embd, n_embd, true),
          n_embd_(n_embd),
          n_heads_(n_heads),
          head_dim_(n_embd / n_heads),
          attn_type_(attn_type)
    {
        init_linear_gpt2(c_attn, 0.02f, seed);
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(c_proj, 0.02f * scale, seed + 1);

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
        Tensor q = inp[0], k = inp[1], v = inp[2];

        q = autograd::transpose(autograd::reshape(q, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);

        // Attention computation — dispatch based on config
        Tensor attn_out;
        switch (attn_type_) {
            case config::AttentionType::FMHA:
                // std::cout << "CALLED FMHA" << std::endl;
                attn_out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::MemoryEfficient);
                break;

            case config::AttentionType::NORMAL:
                // std::cout << "CALLED NORMAL" << std::endl;
                attn_out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::Math);
                break;

            case config::AttentionType::FUSED_TRIL_SOFTMAX: {
                // std::cout << "CALLED FUSED TRIL SOFTMAX" << std::endl;
                float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(head_dim_));
                Tensor scale_t = Tensor::full(Shape{{1}},
                    TensorOptions().with_dtype(Dtype::Float32).with_device(x.device()),
                    inv_sqrt);
                Tensor attn = autograd::matmul(autograd::mul(q, scale_t),
                                               autograd::transpose(k, -2, -1));
                float neg_inf = -std::numeric_limits<float>::infinity();
                Tensor probs = autograd::fused_tril_softmax(attn, 0, neg_inf);
                attn_out = autograd::matmul(probs, v);
                break;
            }

            case config::AttentionType::FLASH:
                // Placeholder: falls back to FMHA for now
                // std::cout << "CALLED FLASH ATTENTION" << std::endl;
                attn_out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::MemoryEfficient);
                break;
        }

        Tensor merged = autograd::reshape(
            autograd::transpose(attn_out, 1, 2), Shape({{B, T, C}}));

        // Output projection + residual
        Tensor proj = c_proj.forward(merged);
        return autograd::add(x, proj);
    }

private:
    int64_t n_embd_;
    int64_t n_heads_;
    int64_t head_dim_;
    config::AttentionType attn_type_;
};

// =============================================================================
// Tensor-Parallel Attention (column-parallel QKV, row-parallel output proj)
// =============================================================================
class TpAttention {
public:
    nn::LayerNorm ln;
    Tensor w_qkv, b_qkv;    // Column-parallel: [n_embd, 3 * local_embd]
    Tensor w_proj, b_proj;   // Row-parallel: [local_embd, n_embd]

    TpAttention() : ln(1) {}

    TpAttention(int64_t n_embd, int64_t n_heads, int64_t n_layers,
                config::AttentionType attn_type,
                std::shared_ptr<ProcessGroupNCCL>& tp_pg,
                DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd), tp_pg_(tp_pg), attn_type_(attn_type)
    {
        int tp_size = tp_pg->get_worldsize();
        int tp_rank = tp_pg->get_rank();
        n_embd_ = n_embd;
        n_heads_ = n_heads;
        head_dim_ = n_embd / n_heads;
        local_heads_ = n_heads / tp_size;
        local_embd_ = local_heads_ * head_dim_;

        TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);

        // Column-parallel QKV
        {
            Tensor w_full = Tensor::randn<float>(
                Shape{{n_embd_, 3 * n_embd_}}, cpu_opts, seed, 0.02f);
            int64_t shard = 3 * local_embd_;
            w_qkv = w_full.narrow(1, tp_rank * shard, shard).contiguous().to(device);
            w_qkv.set_requires_grad(true);

            Tensor b_full = Tensor::zeros(Shape{{3 * n_embd_}}, cpu_opts);
            b_qkv = b_full.narrow(0, tp_rank * shard, shard).contiguous().to(device);
            b_qkv.set_requires_grad(true);
        }

        // Row-parallel output projection
        {
            float scale = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
            Tensor w_full = Tensor::randn<float>(
                Shape{{n_embd_, n_embd_}}, cpu_opts, seed + 1, scale);
            w_proj = w_full.narrow(0, tp_rank * local_embd_, local_embd_).contiguous().to(device);
            w_proj.set_requires_grad(true);

            b_proj = Tensor::zeros(Shape{{n_embd_}}, cpu_opts).to(device);
            b_proj.set_requires_grad(true);
        }

        scale_t_ = Tensor::full(Shape{{1}},
            TensorOptions().with_dtype(Dtype::Float32).with_device(device),
            1.0f / std::sqrt(static_cast<float>(head_dim_)));

        ln.to(device);
    }

    Tensor forward(Tensor& x) {
        int64_t B = x.shape().dims[0];
        int64_t T = x.shape().dims[1];

        Tensor h = ln.forward(x);

        // Column-parallel QKV with overlapped backward AR
        Tensor qkv = tp::column_parallel_addmm(b_qkv, h, w_qkv, tp_pg_);

        auto splits = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = splits[0], k = splits[1], v = splits[2];

        q = autograd::transpose(autograd::reshape(q, Shape({{B, T, local_heads_, head_dim_}})), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape({{B, T, local_heads_, head_dim_}})), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape({{B, T, local_heads_, head_dim_}})), 1, 2);

        // Attention computation — dispatch based on config
        Tensor out;
        switch (attn_type_) {
            case config::AttentionType::FMHA:
                out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::MemoryEfficient);
                break;

            case config::AttentionType::NORMAL:
                out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::Math);
                break;

            case config::AttentionType::FUSED_TRIL_SOFTMAX:
            default: {
                Tensor attn = autograd::matmul(autograd::mul(q, scale_t_),
                                               autograd::transpose(k, -2, -1));
                float neg_inf = -std::numeric_limits<float>::infinity();
                Tensor probs = autograd::fused_tril_softmax(attn, 0, neg_inf);
                out = autograd::matmul(probs, v);
                break;
            }

            case config::AttentionType::FLASH:
                out = autograd::scaled_dot_product_attention(
                    q, k, v, true, 0.0f, autograd::SDPBackend::MemoryEfficient);
                break;
        }

        Tensor merged = autograd::reshape(autograd::transpose(out, 1, 2),
                                          Shape({{B, T, local_embd_}}));

        // Row-parallel output projection + all-reduce + bias + residual
        Tensor proj = autograd::matmul(merged, w_proj);
        tp::allreduce_inplace(proj, tp_pg_);
        proj = autograd::add(proj, b_proj);
        return autograd::add(x, proj);
    }

    std::vector<Tensor*> param_ptrs() {
        return {&ln.weight, &ln.bias, &w_qkv, &b_qkv, &w_proj, &b_proj};
    }

private:
    std::shared_ptr<ProcessGroupNCCL> tp_pg_;
    config::AttentionType attn_type_ = config::AttentionType::FUSED_TRIL_SOFTMAX;
    int64_t n_embd_ = 0, n_heads_ = 0, head_dim_ = 0;
    int64_t local_heads_ = 0, local_embd_ = 0;
    Tensor scale_t_;
};

}  // namespace gpt2
