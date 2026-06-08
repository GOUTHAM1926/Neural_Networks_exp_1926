#pragma once
// =============================================================================
// MLP.h — Feed-Forward Network blocks
//   - MLP:    Standard FFN (single GPU / DDP)
//   - TpMLP:  Tensor-parallel FFN (column-parallel up, row-parallel down)
// =============================================================================

#include <cmath>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "mlp/activation.h"
#include "nn/NN.h"

#include "common.h"
#include "../parallelism/tp_ops.h"

using namespace OwnTensor;

namespace gpt2 {

// =============================================================================
// Standard MLP (single GPU / DDP)
// =============================================================================
class MLP : public nn::Module {
public:
    nn::LayerNorm ln;
    nn::Linear fc_up;    // [n_embd, 4*n_embd]
    nn::Linear fc_down;  // [4*n_embd, n_embd]

    MLP(int64_t n_embd, int n_layers, DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd),
          fc_up(n_embd, 4 * n_embd, true),
          fc_down(4 * n_embd, n_embd, true),
          n_embd_(n_embd)
    {
        init_linear_gpt2(fc_up, 0.02f, seed);
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(fc_down, 0.02f * scale, seed + 1);

        fc_up.to(device);
        fc_down.to(device);
        ln.to(device);

        register_module(ln);
        register_module(fc_up);
        register_module(fc_down);
    }

    Tensor forward(const Tensor& x) override {
        Tensor h = ln.forward(x);
        h = autograd::matmul(h, fc_up.weight);
        h = autograd::fused_bias_gelu(h, fc_up.bias);
        h = fc_down.forward(h);
        return autograd::add(x, h);
    }

private:
    int64_t n_embd_;
};

// =============================================================================
// Tensor-Parallel MLP (column-parallel up, row-parallel down)
// =============================================================================
class TpMLP {
public:
    nn::LayerNorm ln;
    Tensor w_up, b_up;      // Column-parallel: [n_embd, 4*n_embd/tp]
    Tensor w_down, b_down;  // Row-parallel: [4*n_embd/tp, n_embd]

    TpMLP() : ln(1) {}

    TpMLP(int64_t n_embd, int64_t n_layers,
          std::shared_ptr<ProcessGroupNCCL>& tp_pg,
          DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd), tp_pg_(tp_pg), n_embd_(n_embd)
    {
        int tp_size = tp_pg->get_worldsize();
        int tp_rank = tp_pg->get_rank();
        int64_t inner = 4 * n_embd;
        local_inner_ = inner / tp_size;

        TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);

        // Column-parallel up projection
        {
            Tensor w_full = Tensor::randn<float>(
                Shape{{n_embd, inner}}, cpu_opts, seed, 0.02f);
            w_up = w_full.narrow(1, tp_rank * local_inner_, local_inner_).contiguous().to(device);
            w_up.set_requires_grad(true);

            Tensor b_full = Tensor::zeros(Shape{{inner}}, cpu_opts);
            b_up = b_full.narrow(0, tp_rank * local_inner_, local_inner_).contiguous().to(device);
            b_up.set_requires_grad(true);
        }

        // Row-parallel down projection
        {
            float scale = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
            Tensor w_full = Tensor::randn<float>(
                Shape{{inner, n_embd}}, cpu_opts, seed + 1, scale);
            w_down = w_full.narrow(0, tp_rank * local_inner_, local_inner_).contiguous().to(device);
            w_down.set_requires_grad(true);

            b_down = Tensor::zeros(Shape{{n_embd}}, cpu_opts).to(device);
            b_down.set_requires_grad(true);
        }

        ln.to(device);
    }

    Tensor forward(Tensor& x) {
        Tensor h = ln.forward(x);

        // Column-parallel up projection with overlapped backward AR
        h = tp::column_parallel_matmul(h, w_up, tp_pg_);
        h = autograd::gelu(h);

        // Row-parallel down projection + all-reduce + bias + residual
        h = autograd::matmul(h, w_down);
        tp::allreduce_inplace(h, tp_pg_);
        h = autograd::add(h, b_down);
        return autograd::add(x, h);
    }

    std::vector<Tensor*> param_ptrs() {
        return {&ln.weight, &ln.bias, &w_up, &b_up, &w_down, &b_down};
    }

private:
    std::shared_ptr<ProcessGroupNCCL> tp_pg_;
    int64_t n_embd_ = 0, local_inner_ = 0;
};

}  // namespace gpt2
