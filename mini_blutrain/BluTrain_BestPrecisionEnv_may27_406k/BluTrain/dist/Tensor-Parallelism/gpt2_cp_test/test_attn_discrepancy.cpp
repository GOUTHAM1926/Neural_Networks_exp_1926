// =============================================================================
// Attention Discrepancy Test
//
// Uses the EXACT Attention class from gpt2/gpt2_attn_fixed.cpp and the
// EXACT CPAttention class from DTensor/gpt2_cp_test/gpt2_cp_test.cpp
// with identical configurations and seeds.
//
// Compares forward output at multiple levels:
//   1. Full attention block output (after LN, QKV, SDPA, projection, residual)
//   2. Pre-projection attention output (before c_proj and residual)
//   3. Raw SDPA output (Q*K^T*V after softmax)
// =============================================================================

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <mpi.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "checkpointing/GradMode.h"
#include "autograd/operations/TrilOps.h"
#include "communication/include/ProcessGroupNCCL.h"
#include "ops/UnaryOps/Reduction.h"
#include "dnn/DistributedNN.h"
#include "tensor/dtensor.h"

#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// =============================================================================
// Helper: init linear weights (exactly as in both source files)
// =============================================================================
void init_linear_gpt2(nn::Linear& layer, float std_val = 0.02f,
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
// Helper: print error stats
// =============================================================================
void print_error_stats(const std::string& label, Tensor& a, Tensor& b, int rank) {
    Tensor diff = a - b;
    Tensor abs_diff = OwnTensor::abs(diff);

    Tensor max_err = abs_diff;
    for (int64_t d = abs_diff.ndim() - 1; d >= 0; --d) {
        max_err = reduce_max(max_err, {d}, false);
    }

    Tensor mean_err = abs_diff;
    for (int64_t d = abs_diff.ndim() - 1; d >= 0; --d) {
        mean_err = reduce_sum(mean_err, {d}, false);
    }

    float max_val = max_err.to_cpu().data<float>()[0];
    float mean_val = mean_err.to_cpu().data<float>()[0] / static_cast<float>(a.numel());

    std::cout << "  [Rank " << rank << "] [" << label << "] Max Error: " << std::scientific << max_val
              << "  Mean Error: " << mean_val << std::fixed << std::endl;
}

void print_first_n(const std::string& label, Tensor& t, int n, int rank) {
    Tensor cpu = t.to_cpu();
    float* data = cpu.data<float>();
    int64_t total = cpu.numel();
    int count = std::min(n, static_cast<int>(total));
    std::cout << "  [Rank " << rank << "] [" << label << "] first " << count << ": ";
    for (int i = 0; i < count; ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i];
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

// Print n values starting at a flat offset into the tensor
void print_at_offset(const std::string& label, Tensor& t, int64_t offset, int n, int rank) {
    Tensor cpu = t.to_cpu();
    float* data = cpu.data<float>();
    int64_t total = cpu.numel();
    int count = std::min(n, static_cast<int>(total - offset));
    if (count <= 0) { std::cout << "  [Rank " << rank << "] [" << label << "] offset out of range" << std::endl; return; }
    std::cout << "  [Rank " << rank << "] [" << label << "] @offset=" << offset << " (" << count << "): ";
    for (int i = 0; i < count; ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[offset + i];
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

// =============================================================================
// EXACT copy of Attention from gpt2/gpt2_attn_fixed.cpp
// =============================================================================
class StdAttention : public nn::Module {
public:
    nn::LayerNorm ln;        // Pre-norm LayerNorm
    nn::Linear c_attn;       // QKV projection: [n_embd] -> [3 * n_embd]
    nn::Linear c_proj;       // Output projection: [n_embd] -> [n_embd]

    // Diagnostic: saved intermediates from last forward
    Tensor last_h;           // LN output [B, T, C]
    Tensor last_q;           // [B, H, T, D]
    Tensor last_k;           // [B, H, T, D]
    Tensor last_v;           // [B, H, T, D]
    Tensor last_attn_out;    // SDPA output [B, H, T, D]
    Tensor last_merged;      // reshaped [B, T, C] before c_proj
    Tensor last_proj;        // c_proj output [B, T, C]

    StdAttention(int64_t n_embd, int n_heads, int n_layers, DeviceIndex device, uint64_t seed = 1234)
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
        last_h = h;

        // QKV Projection
        Tensor qkv = c_attn.forward(h);

        std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = inp[0];
        Tensor k = inp[1];
        Tensor v = inp[2];

        q = autograd::transpose( autograd::reshape(q, Shape{{B, T, n_heads_, head_dim_}}), 1, 2);
        k = autograd::transpose( autograd::reshape(k, Shape{{B, T, n_heads_, head_dim_}}), 1, 2);
        v = autograd::transpose( autograd::reshape(v, Shape{{B, T, n_heads_, head_dim_}}), 1, 2);
        last_q = q;
        last_k = k;
        last_v = v;

        // Scaled Dot-Product Attention
        Tensor attn_weights = autograd::matmul( autograd::mul(q, scale_), autograd::transpose(k, -2, -1));

        float neg_inf = -std::numeric_limits<float>::infinity();

        Tensor attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);

        Tensor attn_out = autograd::matmul(attn_probs, v);
        last_attn_out = attn_out;

        Tensor merged = autograd::reshape(
                            autograd::transpose(attn_out, 1, 2),
                            Shape{{B, T, C}});
        last_merged = merged;

        Tensor proj = c_proj.forward(merged);
        last_proj = proj;

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
// EXACT copy of CPAttention from DTensor/gpt2_cp_test/gpt2_cp_test.cpp
// =============================================================================
class CPAttention : public nn::Module {
public:
    nn::LayerNorm  ln;
    nn::Linear     c_attn;
    nn::Linear     c_proj;

    // Diagnostic: saved intermediates from last forward
    Tensor last_h;           // LN output [B, T, C]
    Tensor last_q;           // [B, H, T, D]
    Tensor last_k;           // [B, H, T, D]
    Tensor last_v;           // [B, H, T, D]
    Tensor last_attn_out;    // CP SDPA output [B, H, T, D]
    Tensor last_merged;      // reshaped [B, T, C] before c_proj
    Tensor last_proj;        // c_proj output [B, T, C]

    CPAttention(int64_t n_embd, int64_t n_heads, int64_t n_layers,
                DeviceIndex device,
                std::shared_ptr<ProcessGroupNCCL> pg,
                const DeviceMesh& mesh,
                uint64_t seed = 1234,
                bool load_balancing = false)
        : ln(n_embd),
          c_attn(n_embd, 3 * n_embd, true),
          c_proj(n_embd, n_embd, true),
          n_embd_(n_embd),
          n_heads_(n_heads),
          head_dim_(n_embd / n_heads)
    {
        init_linear_gpt2(c_attn, 0.02f, seed);
        float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(c_proj, proj_std, seed + 1);

        float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        cp_ = std::make_shared<ContextParallel>(
            mesh, pg, attn_scale, /*is_causal=*/true,
            RotatorType::AlltoAll, /*load_balance=*/load_balancing);

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

        Tensor h = ln.forward(x);
        last_h = h;

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
        last_q = q;
        last_k = k;
        last_v = v;

        // Context Parallel SDPA
        Tensor attn_out = cp_->forward_cp(q, k, v);
        last_attn_out = attn_out;

        // Reshape back [B, T, C]
        Tensor merged = autograd::reshape(
                            autograd::transpose(attn_out, 1, 2),
                            Shape({{B, T, C}}));
        last_merged = merged;

        Tensor proj = c_proj.forward(merged);
        last_proj = proj;
        return autograd::add(x, proj);
    }

private:
    int64_t n_embd_;
    int64_t n_heads_;
    int64_t head_dim_;
    std::shared_ptr<ContextParallel> cp_;
};

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DeviceIndex device(Device::CUDA, rank);
    cudaSetDevice(rank);

    std::vector<int> ranks_vec(world_size);
    for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
    DeviceMesh mesh({world_size}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    // =========================================================================
    // Use EXACT same config as both production files
    // =========================================================================
    int64_t n_embd = 384;
    int64_t n_heads = 6;
    int64_t n_layers = 3;
    int64_t head_dim = n_embd / n_heads;  // = 64

    // T must be divisible by world_size for CP
    int64_t B = 2;
    int64_t T = 64;  // Small but divisible by world_size
    int64_t C = n_embd;

    // Use same seed as production: seed=1234 (default)
    uint64_t seed = 1234;

    if (rank == 0) {
        std::cout << "=== Attention Discrepancy Test (Exact Production Classes) ===" << std::endl;
        std::cout << "World Size: " << world_size
                  << ", B=" << B << ", T=" << T << ", C=" << C << std::endl;
        std::cout << "n_heads=" << n_heads << ", head_dim=" << head_dim
                  << ", n_layers=" << n_layers << ", seed=" << seed << std::endl;
        std::cout << std::endl;
    }

    // =========================================================================
    // Create both attention blocks with IDENTICAL seed + config
    // =========================================================================
    StdAttention std_attn(n_embd, n_heads, n_layers, device, seed);
    CPAttention cp_attn(n_embd, n_heads, n_layers, device, pg, mesh, seed, false);

    // =========================================================================
    // Verify weights are identical
    // =========================================================================
    if (rank == 0) {
        std::cout << "--- Verifying Weight Identity ---" << std::endl;

        // Compare c_attn weights
        Tensor diff_cattn = std_attn.c_attn.weight - cp_attn.c_attn.weight;
        Tensor abs_diff_cattn = OwnTensor::abs(diff_cattn);
        Tensor max_cattn = abs_diff_cattn;
        for (int64_t d = abs_diff_cattn.ndim() - 1; d >= 0; --d) {
            max_cattn = reduce_max(max_cattn, {d}, false);
        }
        std::cout << "  c_attn weight max diff: " << std::scientific
                  << max_cattn.to_cpu().data<float>()[0] << std::endl;

        // Compare c_proj weights
        Tensor diff_cproj = std_attn.c_proj.weight - cp_attn.c_proj.weight;
        Tensor abs_diff_cproj = OwnTensor::abs(diff_cproj);
        Tensor max_cproj = abs_diff_cproj;
        for (int64_t d = abs_diff_cproj.ndim() - 1; d >= 0; --d) {
            max_cproj = reduce_max(max_cproj, {d}, false);
        }
        std::cout << "  c_proj weight max diff: " << std::scientific
                  << max_cproj.to_cpu().data<float>()[0] << std::endl;

        // Compare LN weights
        Tensor diff_ln_w = std_attn.ln.weight - cp_attn.ln.weight;
        Tensor abs_diff_ln = OwnTensor::abs(diff_ln_w);
        Tensor max_ln = abs_diff_ln;
        for (int64_t d = abs_diff_ln.ndim() - 1; d >= 0; --d) {
            max_ln = reduce_max(max_ln, {d}, false);
        }
        std::cout << "  ln weight max diff: " << std::scientific
                  << max_ln.to_cpu().data<float>()[0] << std::endl;

        std::cout << std::fixed << std::endl;
    }

    // =========================================================================
    // Create identical input on ALL ranks
    // =========================================================================
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(device).with_req_grad(true);
    Tensor x = Tensor::randn<float>(Shape{{B, T, C}}, opts, 999, 1.0f);

    // =========================================================================
    // Forward both
    // =========================================================================
    if (rank == 0) std::cout << "--- Forward Pass ---" << std::endl;

    Tensor out_std = std_attn.forward(x);
    Tensor out_cp = cp_attn.forward(x);

    // =========================================================================
    // Direct SDPA test: bypass ContextParallel, manually shard + SDPA
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- Direct SDPA Test (bypass ContextParallel) ---" << std::endl;

        // Use standard Q, K, V (known correct) and manually shard
        // IMPORTANT: Q from autograd::transpose has non-contiguous strides
        // [24576, 64, 384, 1] (transposed [B,T,H,D] -> [B,H,T,D])
        // Make BOTH contiguous first so flat offsets = logical [b,h,t,d] positions
        Tensor std_q_c = std_attn.last_q.contiguous();  // [B=2, H=6, T=64, D=64] contiguous
        Tensor std_k_c = std_attn.last_k.contiguous();
        Tensor std_v_c = std_attn.last_v.contiguous();

        // Shard the CONTIGUOUS Q along dim 2 (T), take rank 0's chunk
        std::vector<Tensor> q_chunks = std_q_c.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2);
        std::vector<Tensor> k_chunks = std_k_c.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2);
        std::vector<Tensor> v_chunks = std_v_c.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2);

        std::cout << "  q_chunks[0] shape: ["
                  << q_chunks[0].shape().dims[0] << ","
                  << q_chunks[0].shape().dims[1] << ","
                  << q_chunks[0].shape().dims[2] << ","
                  << q_chunks[0].shape().dims[3] << "]" << std::endl;

        Tensor local_q = q_chunks[0].contiguous();
        Tensor local_k = k_chunks[0].contiguous();
        Tensor local_v = v_chunks[0].contiguous();

        int64_t D = head_dim;
        int64_t H = n_heads;
        int64_t T_local = T / world_size;

        // Compare local_q (sharded from contiguous) vs std_q_c (contiguous full)
        // at t=31, h=0 — both now have contiguous strides so flat offsets are valid
        int64_t off_local_31 = 0 * (H * T_local * D) + 0 * (T_local * D) + 31 * D;
        int64_t off_full_31  = 0 * (H * T * D) + 0 * (T * D) + 31 * D;
        std::cout << "\n  sharded_q[b0,h0,t31] vs std_q[b0,h0,t31] (both contiguous):" << std::endl;
        print_at_offset("local_q[b0,h0,t31]", local_q,  off_local_31, 8, rank);
        print_at_offset("std_q_c[b0,h0,t31]", std_q_c, off_full_31,  8, rank);

        // h=1, t=0
        int64_t off_local_h1_t0 = 0 * (H * T_local * D) + 1 * (T_local * D) + 0 * D;
        int64_t off_full_h1_t0  = 0 * (H * T * D) + 1 * (T * D) + 0 * D;
        std::cout << "\n  sharded_q[b0,h1,t0] vs std_q[b0,h1,t0] (both contiguous):" << std::endl;
        print_at_offset("local_q[b0,h1,t0]", local_q,  off_local_h1_t0, 8, rank);
        print_at_offset("std_q_c[b0,h1,t0]", std_q_c, off_full_h1_t0,  8, rank);

        // Now: does the CP code shard the TRANSPOSED (non-contiguous) Q?
        // In forward_cp, q_work = q (transposed) -> make_shards_inplace_axis -> contiguous()
        // Let's replicate EXACTLY what forward_cp does:
        Tensor transposed_q = std_attn.last_q;  // non-contiguous [B,H,T,D] from transpose
        std::vector<Tensor> t_q_chunks = transposed_q.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2);
        Tensor cp_local_q = t_q_chunks[0].contiguous();

        std::cout << "\n  CP path: shard transposed Q, then contiguous():" << std::endl;
        print_at_offset("cp_local_q[b0,h0,t31]", cp_local_q, off_local_31, 8, rank);
        print_at_offset("std_q_c[b0,h0,t31]",    std_q_c,    off_full_31,  8, rank);

        std::cout << "\n  CP path h=1:" << std::endl;
        print_at_offset("cp_local_q[b0,h1,t0]", cp_local_q, off_local_h1_t0, 8, rank);
        print_at_offset("std_q_c[b0,h1,t0]",    std_q_c,    off_full_h1_t0,  8, rank);

        // Direct SDPA on CORRECTLY sharded data (from contiguous Q)
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        SDPAResult correct_result = sdpa_forward(local_q, local_k, local_v, true, attn_scale);

        // Direct SDPA on CP-path sharded data (from transposed Q)
        Tensor cp_local_k = std_attn.last_k.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2)[0].contiguous();
        Tensor cp_local_v = std_attn.last_v.make_shards_inplace_axis(
            static_cast<size_t>(world_size), 2)[0].contiguous();
        SDPAResult cp_result = sdpa_forward(cp_local_q, cp_local_k, cp_local_v, true, attn_scale);

        Tensor std_attn_out_c = std_attn.last_attn_out.contiguous();
        std::cout << "\n  SDPA on correct-sharded vs CP-sharded at t=31:" << std::endl;
        print_at_offset("correct_sdpa[b0,h0,t31]", correct_result.out, off_local_31, 8, rank);
        print_at_offset("cp_sdpa[b0,h0,t31]",      cp_result.out,      off_local_31, 8, rank);
        print_at_offset("std_sdpa_c[b0,h0,t31]",   std_attn_out_c,     off_full_31,  8, rank);

        std::cout << "\n  Direct SDPA numel: " << correct_result.out.numel()
                  << " (expected " << B * H * T_local * D << ")" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Layered Comparison: pinpoint where the divergence starts
    // =========================================================================
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 0: LN output ---" << std::endl;
        print_first_n("std_h", std_attn.last_h, 8, rank);
        print_first_n("cp_h",  cp_attn.last_h,  8, rank);
        print_error_stats("LN output", std_attn.last_h, cp_attn.last_h, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 1: Q, K, V [B,H,T,D] ---" << std::endl;
        print_first_n("std_q", std_attn.last_q, 8, rank);
        print_first_n("cp_q",  cp_attn.last_q,  8, rank);
        print_error_stats("Q", std_attn.last_q, cp_attn.last_q, rank);

        print_first_n("std_k", std_attn.last_k, 8, rank);
        print_first_n("cp_k",  cp_attn.last_k,  8, rank);
        print_error_stats("K", std_attn.last_k, cp_attn.last_k, rank);

        print_first_n("std_v", std_attn.last_v, 8, rank);
        print_first_n("cp_v",  cp_attn.last_v,  8, rank);
        print_error_stats("V", std_attn.last_v, cp_attn.last_v, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 2: SDPA output [B,H,T,D] ---" << std::endl;
        std::cout << "  std_attn_out shape: [" << std_attn.last_attn_out.shape().dims[0]
                  << "," << std_attn.last_attn_out.shape().dims[1]
                  << "," << std_attn.last_attn_out.shape().dims[2]
                  << "," << std_attn.last_attn_out.shape().dims[3] << "]" << std::endl;
        std::cout << "  cp_attn_out shape:  [" << cp_attn.last_attn_out.shape().dims[0]
                  << "," << cp_attn.last_attn_out.shape().dims[1]
                  << "," << cp_attn.last_attn_out.shape().dims[2]
                  << "," << cp_attn.last_attn_out.shape().dims[3] << "]" << std::endl;

        print_first_n("std_attn_out", std_attn.last_attn_out, 8, rank);
        print_first_n("cp_attn_out",  cp_attn.last_attn_out,  8, rank);
        print_error_stats("SDPA output", std_attn.last_attn_out, cp_attn.last_attn_out, rank);

        // Compare specific positions in SDPA output [B=2, H=6, T=64, D=64]
        // Layout: [b, h, t, d] -> offset = b*(H*T*D) + h*(T*D) + t*D + d
        int64_t H = n_heads;
        int64_t D = head_dim;

        // Position 0, Head 0 (rank 0 chunk, no merger) — should be V[0]
        int64_t off_t0 = 0 * (H * T * D) + 0 * (T * D) + 0 * D;
        std::cout << "\n  Pos 0 (rank0 chunk, no merger):" << std::endl;
        print_at_offset("std[b0,h0,t0]", std_attn.last_attn_out, off_t0, 8, rank);
        print_at_offset("cp[b0,h0,t0]",  cp_attn.last_attn_out,  off_t0, 8, rank);

        // Position 31, Head 0 (last in rank 0 chunk, no merger)
        int64_t off_t31 = 0 * (H * T * D) + 0 * (T * D) + 31 * D;
        std::cout << "\n  Pos 31 (last in rank0 chunk, no merger):" << std::endl;
        print_at_offset("std[b0,h0,t31]", std_attn.last_attn_out, off_t31, 8, rank);
        print_at_offset("cp[b0,h0,t31]",  cp_attn.last_attn_out,  off_t31, 8, rank);

        // Position 32, Head 0 (first in rank 1 chunk, USES merger)
        int64_t off_t32 = 0 * (H * T * D) + 0 * (T * D) + 32 * D;
        std::cout << "\n  Pos 32 (first in rank1 chunk, USES merger):" << std::endl;
        print_at_offset("std[b0,h0,t32]", std_attn.last_attn_out, off_t32, 8, rank);
        print_at_offset("cp[b0,h0,t32]",  cp_attn.last_attn_out,  off_t32, 8, rank);

        // Position 63, Head 0 (last in rank 1 chunk, USES merger)
        int64_t off_t63 = 0 * (H * T * D) + 0 * (T * D) + 63 * D;
        std::cout << "\n  Pos 63 (last in rank1 chunk, USES merger):" << std::endl;
        print_at_offset("std[b0,h0,t63]", std_attn.last_attn_out, off_t63, 8, rank);
        print_at_offset("cp[b0,h0,t63]",  cp_attn.last_attn_out,  off_t63, 8, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 3: merged [B,T,C] (after transpose+reshape, before c_proj) ---" << std::endl;
        print_first_n("std_merged", std_attn.last_merged, 8, rank);
        print_first_n("cp_merged",  cp_attn.last_merged,  8, rank);
        print_error_stats("merged", std_attn.last_merged, cp_attn.last_merged, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 4: c_proj output [B,T,C] ---" << std::endl;
        print_first_n("std_proj", std_attn.last_proj, 8, rank);
        print_first_n("cp_proj",  cp_attn.last_proj,  8, rank);
        print_error_stats("c_proj", std_attn.last_proj, cp_attn.last_proj, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n--- Layer 5: Final output (after residual) ---" << std::endl;
    }
    print_first_n("out_std", out_std, 8, rank);
    print_first_n("out_cp", out_cp, 8, rank);
    print_error_stats("out_std vs out_cp", out_std, out_cp, rank);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
