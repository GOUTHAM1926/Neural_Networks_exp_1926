// =============================================================================
// Attention Block Gradient Comparison Test
//
// Compares CP attention block vs standard attention block gradient-by-gradient.
// Uses EXACT same class definitions, weight initialization, and config as:
//   - gpt2_cp_test.cpp  (CPAttention)
//   - gpt2_attn_fixed.cpp (Attention)
//
// Build:  make attn_block_grad_test
// Run:    mpirun -np 2 ./attn_block_grad_test_exec
//
// Output: per-param gradient max/mean abs diff and first-mismatch positions
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "mlp/activation.h"
#include "checkpointing/GradMode.h"
#include "tensor/dtensor.h"
#include "communication/include/ProcessGroupNCCL.h"

#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// =============================================================================
// Config — must match gpt2_cp_test.cpp and gpt2_attn_fixed.cpp exactly
// =============================================================================
static const int64_t N_EMBD   = 384;
static const int64_t N_HEADS  = 6;
static const int64_t N_LAYERS = 3;       // needed for proj_std scaling
static const int64_t B        = 4;
static const int64_t T        = 32;      // shorter than 1024 for speed; still divisible by 2
static const uint64_t SEED    = 1234 + 200;  // matches layer-0 seed in both training files

// =============================================================================
// Weight initialization — exact copy from gpt2_cp_test.cpp / gpt2_attn_fixed.cpp
// =============================================================================
static void init_linear_gpt2(nn::Linear& layer, float std_val, uint64_t seed, bool req_grad = true) {
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
// CPAttention — exact copy from gpt2_cp_test.cpp
// =============================================================================
class CPAttention : public nn::Module {
public:
    nn::LayerNorm ln;
    nn::Linear c_attn;
    nn::Linear c_proj;

    CPAttention(int64_t n_embd, int64_t n_heads, int64_t n_layers,
                DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,
                const DeviceMesh& mesh, uint64_t seed, bool load_balancing)
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
            mesh, pg, attn_scale, true, RotatorType::AlltoAll, load_balancing);

        ln.to(device);
        c_attn.to(device);
        c_proj.to(device);

        register_module(ln);
        register_module(c_attn);
        register_module(c_proj);
    }

    Tensor forward(const Tensor& x) override {
        int64_t Bd = x.shape().dims[0];
        int64_t Td = x.shape().dims[1];
        int64_t Cd = x.shape().dims[2];

        Tensor h = ln.forward(x);
        Tensor qkv = c_attn.forward(h);
        std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = inp[0], k = inp[1], v = inp[2];

        q = autograd::transpose(autograd::reshape(q, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);

        Tensor attn_out = cp_->forward_cp(q, k, v);

        Tensor merged = autograd::reshape(
            autograd::transpose(attn_out, 1, 2), Shape({{Bd, Td, Cd}}));
        Tensor proj = c_proj.forward(merged);
        return autograd::add(x, proj);
    }

private:
    int64_t n_embd_, n_heads_, head_dim_;
    std::shared_ptr<ContextParallel> cp_;
};

// =============================================================================
// Standard Attention — exact copy from gpt2_attn_fixed.cpp
// =============================================================================
class Attention : public nn::Module {
public:
    nn::LayerNorm ln;
    nn::Linear c_attn;
    nn::Linear c_proj;

    Attention(int64_t n_embd, int64_t n_heads, int64_t n_layers, DeviceIndex device, uint64_t seed)
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

        scale_ = Tensor::full(Shape({{1}}),
                              TensorOptions().with_dtype(Dtype::Float32).with_device(device),
                              1.0f / std::sqrt(static_cast<float>(head_dim_)));

        ln.to(device);
        c_attn.to(device);
        c_proj.to(device);

        register_module(ln);
        register_module(c_attn);
        register_module(c_proj);
    }

    Tensor forward(const Tensor& x) override {
        int64_t Bd = x.shape().dims[0];
        int64_t Td = x.shape().dims[1];
        int64_t Cd = x.shape().dims[2];

        Tensor h = ln.forward(x);
        Tensor qkv = c_attn.forward(h);
        std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
        Tensor q = inp[0], k = inp[1], v = inp[2];

        q = autograd::transpose(autograd::reshape(q, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape({{Bd, Td, n_heads_, head_dim_}})), 1, 2);

        Tensor attn_weights = autograd::matmul(
            autograd::mul(q, scale_), autograd::transpose(k, -2, -1));

        float neg_inf = -std::numeric_limits<float>::infinity();
        Tensor attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
        Tensor attn_out = autograd::matmul(attn_probs, v);

        Tensor merged = autograd::reshape(
            autograd::transpose(attn_out, 1, 2), Shape({{Bd, Td, Cd}}));
        Tensor proj = c_proj.forward(merged);
        return autograd::add(x, proj);
    }

private:
    int64_t n_embd_, n_heads_, head_dim_;
    Tensor scale_;
};

// =============================================================================
// Helpers
// =============================================================================
static float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < a.numel(); ++i)
        m = std::max(m, std::abs(pa[i] - pb[i]));
    return m;
}

static float mean_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    double s = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i)
        s += std::abs(pa[i] - pb[i]);
    return static_cast<float>(s / static_cast<double>(a.numel()));
}

// Print first few mismatches with index decode
static void print_mismatches(const Tensor& ref, const Tensor& got,
                             const char* label, float tol, int max_print) {
    const float* r = ref.data<float>();
    const float* g = got.data<float>();
    int printed = 0;
    for (int64_t i = 0; i < ref.numel() && printed < max_print; ++i) {
        float d = std::abs(r[i] - g[i]);
        if (d > tol) {
            std::cout << "    " << label << "[" << i << "]"
                      << " ref=" << std::setw(10) << r[i]
                      << " got=" << std::setw(10) << g[i]
                      << " diff=" << std::scientific << d << std::endl;
            printed++;
        }
    }
}

// Copy src tensor data into dst (both on GPU, same shape)
static void copy_weights(Tensor& dst, const Tensor& src) {
    size_t bytes = static_cast<size_t>(src.numel()) * sizeof(float);
    cudaMemcpy(dst.data<float>(), src.data<float>(), bytes, cudaMemcpyDeviceToDevice);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0)
            std::cerr << "Requires exactly 2 GPUs. Run: mpirun -np 2 ./attn_block_grad_test_exec\n";
        MPI_Finalize();
        return 1;
    }

    cudaSetDevice(rank);
    DeviceIndex device(Device::CUDA, rank);

    std::vector<int> ranks_vec = {0, 1};
    DeviceMesh mesh({2}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    if (rank == 0) {
        std::cout << "=== Attention Block Gradient Test ===" << std::endl;
        std::cout << "Config: B=" << B << " T=" << T
                  << " n_embd=" << N_EMBD << " n_heads=" << N_HEADS
                  << " n_layers=" << N_LAYERS << " seed=" << SEED << std::endl;
        std::cout << "head_dim=" << (N_EMBD / N_HEADS) << std::endl;
    }

    // -------------------------------------------------------------------------
    // Build both attention blocks with the SAME seed (guarantees same weights)
    // -------------------------------------------------------------------------
    CPAttention cp_attn(N_EMBD, N_HEADS, N_LAYERS, device, pg, mesh, SEED,
                        /*load_balancing=*/false);

    // Normal attention only needs to run on rank 0, but we construct it on
    // both to avoid diverging control flow — only rank 0 will use it.
    Attention std_attn(N_EMBD, N_HEADS, N_LAYERS, device, SEED);

    // Explicitly copy CP weights into std_attn to guarantee bit-identical params
    // (guards against any RNG state drift between the two constructions)
    if (rank == 0) {
        copy_weights(std_attn.ln.weight,   cp_attn.ln.weight);
        copy_weights(std_attn.ln.bias,     cp_attn.ln.bias);
        copy_weights(std_attn.c_attn.weight, cp_attn.c_attn.weight);
        copy_weights(std_attn.c_attn.bias,   cp_attn.c_attn.bias);
        copy_weights(std_attn.c_proj.weight, cp_attn.c_proj.weight);
        copy_weights(std_attn.c_proj.bias,   cp_attn.c_proj.bias);
    }

    // -------------------------------------------------------------------------
    // Build input x: same data on all ranks (fixed seed)
    // Shape: [B, T, C]
    // -------------------------------------------------------------------------
    Shape x_shape({{B, T, N_EMBD}});
    TensorOptions x_opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(device)
        .with_req_grad(true);

    // Both ranks create the same x (same seed = same data)
    Tensor x_cp  = Tensor::randn<float>(x_shape, x_opts, 42, 1.0f);
    Tensor x_std = Tensor::randn<float>(x_shape, x_opts, 42, 1.0f);  // same seed

    // Upstream gradient: ones (same as using sum loss)
    Tensor dout = Tensor::full(x_shape,
                               TensorOptions().with_dtype(Dtype::Float32).with_device(device),
                               1.0f);

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // CP forward + backward
    // -------------------------------------------------------------------------
    Tensor out_cp = cp_attn.forward(x_cp);
    out_cp.backward(&dout);

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // Standard forward + backward (rank 0 only)
    // -------------------------------------------------------------------------
    Tensor out_std, std_dout;
    if (rank == 0) {
        out_std = std_attn.forward(x_std);
        
        // Print forward pass divergence
        Tensor out_cp_cpu = out_cp.to_cpu();
        Tensor out_std_cpu = out_std.to_cpu();
        float fwd_diff = max_abs_diff(out_cp_cpu, out_std_cpu);
        std::cout << "\nFORWARD PASS MAX DIFF: " << std::scientific << fwd_diff << std::endl;

        std_dout = Tensor::full(x_shape,
                                TensorOptions().with_dtype(Dtype::Float32).with_device(device),
                                1.0f);
        out_std.backward(&std_dout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // Compare gradients (rank 0 only)
    // -------------------------------------------------------------------------
    if (rank == 0) {
        const float tol = 1e-3f;

        struct ParamPair {
            const char* name;
            Tensor* cp_param;
            Tensor* std_param;
        };

        std::vector<ParamPair> pairs = {
            {"ln.weight",      &cp_attn.ln.weight,      &std_attn.ln.weight},
            {"ln.bias",        &cp_attn.ln.bias,        &std_attn.ln.bias},
            {"c_attn.weight",  &cp_attn.c_attn.weight,  &std_attn.c_attn.weight},
            {"c_attn.bias",    &cp_attn.c_attn.bias,    &std_attn.c_attn.bias},
            {"c_proj.weight",  &cp_attn.c_proj.weight,  &std_attn.c_proj.weight},
            {"c_proj.bias",    &cp_attn.c_proj.bias,    &std_attn.c_proj.bias},
        };

        // Also compare x input gradient (flows to wpe)
        bool x_cp_has_grad  = x_cp.has_grad();
        bool x_std_has_grad = x_std.has_grad();

        std::cout << "\n=== Gradient Comparison (CP vs Standard) ===" << std::endl;
        std::cout << std::left << std::setw(16) << "Parameter"
                  << std::setw(12) << "Shape"
                  << std::setw(14) << "MaxAbsDiff"
                  << std::setw(14) << "MeanAbsDiff"
                  << "Status" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        bool all_pass = true;

        for (auto& pp : pairs) {
            if (!pp.cp_param->has_grad()) {
                std::cout << std::setw(16) << pp.name << " NO GRAD (CP)" << std::endl;
                all_pass = false;
                continue;
            }
            if (!pp.std_param->has_grad()) {
                std::cout << std::setw(16) << pp.name << " NO GRAD (STD)" << std::endl;
                all_pass = false;
                continue;
            }

            Tensor cp_g  = pp.cp_param->grad_view().to_cpu();
            Tensor std_g = pp.std_param->grad_view().to_cpu();

            float maxd  = max_abs_diff(cp_g, std_g);
            float meand = mean_abs_diff(cp_g, std_g);
            bool pass = (maxd < tol);
            all_pass &= pass;

            std::cout << std::left << std::setw(16) << pp.name
                      << std::setw(12) << pp.cp_param->numel()
                      << std::setw(14) << std::scientific << maxd
                      << std::setw(14) << meand
                      << (pass ? "PASS" : "FAIL") << std::endl;

            if (!pass) {
                print_mismatches(std_g, cp_g, pp.name, 1e-4f, 8);
            }
        }

        // x gradient (represents what flows to wpe)
        std::cout << std::string(70, '-') << std::endl;
        if (x_cp_has_grad && x_std_has_grad) {
            Tensor cp_dx  = x_cp.grad_view().to_cpu();
            Tensor std_dx = x_std.grad_view().to_cpu();
            float maxd  = max_abs_diff(cp_dx, std_dx);
            float meand = mean_abs_diff(cp_dx, std_dx);
            bool pass = (maxd < tol);
            all_pass &= pass;
            std::cout << std::left << std::setw(16) << "x.grad (->wpe)"
                      << std::setw(12) << x_cp.numel()
                      << std::setw(14) << std::scientific << maxd
                      << std::setw(14) << meand
                      << (pass ? "PASS" : "FAIL") << std::endl;
            if (!pass) {
                print_mismatches(std_dx, cp_dx, "x.grad", 1e-4f, 12);
            }
        } else {
            std::cout << "x.grad: cp_has=" << x_cp_has_grad
                      << " std_has=" << x_std_has_grad << std::endl;
        }

        std::cout << std::string(70, '-') << std::endl;
        std::cout << "\n>>> Overall: " << (all_pass ? "ALL PASS" : "MISMATCH FOUND") << " <<<\n" << std::endl;

        // Print raw grad norms for sanity check
        std::cout << "--- CP grad norms ---" << std::endl;
        for (auto& pp : pairs) {
            if (pp.cp_param->has_grad()) {
                Tensor g = pp.cp_param->grad_view().to_cpu();
                double norm = 0.0;
                const float* p = g.data<float>();
                for (int64_t i = 0; i < g.numel(); ++i) norm += p[i] * p[i];
                std::cout << "  " << pp.name << ": " << std::sqrt(norm) << std::endl;
            }
        }
        std::cout << "--- STD grad norms ---" << std::endl;
        for (auto& pp : pairs) {
            if (pp.std_param->has_grad()) {
                Tensor g = pp.std_param->grad_view().to_cpu();
                double norm = 0.0;
                const float* p = g.data<float>();
                for (int64_t i = 0; i < g.numel(); ++i) norm += p[i] * p[i];
                std::cout << "  " << pp.name << ": " << std::sqrt(norm) << std::endl;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "=== Test complete ===" << std::endl;

    MPI_Finalize();
    return 0;
}
