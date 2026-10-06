//======================================================================
// test_ds_compare.cpp — drive owt::zero::SymmetricShardedOptimizer with the
// EXACT fixtures DeepSpeed ZeRO-1 was driven with, so the two can be compared
// numerically.
//
// WHAT THIS IS FOR
// ----------------
// The in-tree tests check this optimizer against ITSELF (v1 vs v2, W=1 vs W=N,
// sharded vs single-process). That catches internal inconsistency but cannot
// catch a shared misconception — if the AdamW recipe, the bias correction or
// the reduce semantics are wrong in the same way everywhere, every self-
// consistent test still passes. DeepSpeed ZeRO-1 is an independent, widely
// used implementation of the same contract, so agreeing with it is evidence of
// a different kind.
//
// The fixtures come from ds_ref/ds_zero1_ref.py, which drives the real
// DeepSpeed engine with gradients it also dumps to disk. This harness reads
// the same initial weights and the same PER-RANK gradients and replays them —
// no model, no data, no autograd. What remains is the sharded optimizer math
// and the cross-rank reduce, which is exactly what is under test.
//
// Clipping is OFF here, deliberately. DeepSpeed clips on the GLOBAL norm over
// every parameter; this optimizer clips per dtype group against that group's
// own norm. With more than one group the two differ BY CONSTRUCTION. Comparing
// them with clipping on would measure that design difference, not correctness.
//
// Run (must match the world size the fixtures were generated at):
//   mpirun -np 2 ./test_ds_compare --fixtures <dir> --steps 4
//======================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "ProcessGroupNCCL.h"
#include "SymmetricShardedOptimizer.h"
#include "ShardedOptimizer.h"

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
using owt::zero::SymmetricShardedOptimizer;
using owt::zero::ShardedOptimizer;
using owt::zero::ZeroConfig;
using owt::zero::AdamVariant;

namespace {

constexpr uint64_t kMagic = 0x5A45524F31524546ull;   // "ZERO1REF"

int g_rank = 0, g_world = 1;

struct HostTensor {
    std::vector<int64_t> dims;
    std::vector<float>   data;
    size_t numel() const {
        size_t n = 1;
        for (int64_t d : dims) n *= static_cast<size_t>(d);
        return n;
    }
};

std::vector<HostTensor> read_fixture(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }

    uint64_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kMagic) {
        std::fprintf(stderr, "bad magic in %s (got %llx)\n", path.c_str(),
                     static_cast<unsigned long long>(magic));
        std::exit(2);
    }
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    std::vector<HostTensor> out(static_cast<size_t>(n));
    for (auto& t : out) {
        int32_t nd = 0;
        f.read(reinterpret_cast<char*>(&nd), sizeof(nd));
        t.dims.resize(static_cast<size_t>(nd));
        for (auto& d : t.dims) f.read(reinterpret_cast<char*>(&d), sizeof(d));
    }
    for (auto& t : out) {
        t.data.resize(t.numel());
        f.read(reinterpret_cast<char*>(t.data.data()),
               static_cast<std::streamsize>(t.numel() * sizeof(float)));
        if (!f) { std::fprintf(stderr, "truncated %s\n", path.c_str()); std::exit(2); }
    }
    return out;
}

void write_fixture(const std::string& path, const std::vector<Tensor>& ts) {
    std::ofstream f(path, std::ios::binary);
    uint64_t magic = kMagic;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    int32_t n = static_cast<int32_t>(ts.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& t : ts) {
        int32_t nd = static_cast<int32_t>(t.shape().dims.size());
        f.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
        for (int64_t d : t.shape().dims)
            f.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    for (const auto& t : ts) {
        Tensor c = t.to_cpu();
        f.write(reinterpret_cast<const char*>(c.data<float>()),
                static_cast<std::streamsize>(c.numel() * sizeof(float)));
    }
}

std::string arg_str(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return dflt;
}
int arg_int(int argc, char** argv, const char* key, int dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    return dflt;
}
double arg_f(int argc, char** argv, const char* key, double dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return std::atof(argv[i + 1]);
    return dflt;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world);

    const std::string fx   = arg_str(argc, argv, "--fixtures", "");
    const int   steps      = arg_int(argc, argv, "--steps", 4);
    const bool  use_v1     = arg_int(argc, argv, "--v1", 0) != 0;
    const float lr         = static_cast<float>(arg_f(argc, argv, "--lr", 1e-3));
    const float beta1      = static_cast<float>(arg_f(argc, argv, "--beta1", 0.9));
    const float beta2      = static_cast<float>(arg_f(argc, argv, "--beta2", 0.95));
    const float eps        = static_cast<float>(arg_f(argc, argv, "--eps", 1e-8));
    const float wd         = static_cast<float>(arg_f(argc, argv, "--wd", 0.1));
    const float clip       = static_cast<float>(arg_f(argc, argv, "--clip", 0.0));
    if (fx.empty()) {
        if (g_rank == 0) std::fprintf(stderr, "usage: --fixtures <dir> [--steps N] [--v1 1]\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    cudaSetDevice(g_rank);
    OwnTensor::DeviceIndex dev{OwnTensor::Device::CUDA, g_rank};

    std::shared_ptr<ProcessGroupNCCL> pg;
    if (g_world > 1) pg = init_process_group(g_world, g_rank, /*stream=*/0);

    // --- parameters, from the SAME bytes DeepSpeed started with ---
    std::vector<HostTensor> init = read_fixture(fx + "/init.bin");
    std::vector<Tensor> params;
    params.reserve(init.size());
    for (auto& h : init) {
        Shape s; s.dims = h.dims;
        Tensor t = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
        t.set_data(h.data.data(), h.data.size());
        t.set_requires_grad(true);
        params.push_back(t);
    }

    ZeroConfig cfg;
    cfg.variant                      = AdamVariant::ADAMW;
    cfg.lr                           = lr;
    cfg.beta1                        = beta1;
    cfg.beta2                        = beta2;
    cfg.eps                          = eps;
    cfg.weight_decay                 = wd;
    cfg.exclude_1d_from_weight_decay = true;   // matches the two DeepSpeed param groups
    cfg.max_grad_norm                = clip;   // 0 == off; see the header note
    cfg.init_broadcast               = false;  // both ranks already loaded identical bytes
    cfg.verify_consistency           = true;
    // No autograd graph here (gradients are set directly), so there is nothing
    // for backward hooks to hang off and nothing to overlap against. The
    // reference path is what this test exercises.
    cfg.overlap_backward_comm        = false;
    cfg.overlap_param_refresh        = false;
    cfg.verbose                      = false;

    std::unique_ptr<owt::zero::ZeroOptimizerBase> opt;
    if (use_v1) opt = std::make_unique<ShardedOptimizer>(params, pg, cfg);
    else        opt = std::make_unique<SymmetricShardedOptimizer>(params, pg, cfg);

    if (g_rank == 0) {
        std::printf("=== %s vs DeepSpeed ZeRO-1 (W=%d, %zu params, %d steps) ===\n",
                    use_v1 ? "ShardedOptimizer(v1)" : "SymmetricShardedOptimizer(v2)",
                    g_world, params.size(), steps);
        std::printf("    lr=%g betas=(%g,%g) eps=%g wd=%g clip=%g\n",
                    lr, beta1, beta2, eps, wd, clip);
    }

    for (int s = 0; s < steps; ++s) {
        // This rank's own gradients for this step — the same bytes the matching
        // DeepSpeed rank saw. Regenerating them here instead of reading them
        // would risk the two harnesses silently disagreeing on the INPUT, which
        // would look exactly like an optimizer bug.
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/grad_step%d_rank%d.bin", fx.c_str(), s, g_rank);
        std::vector<HostTensor> g = read_fixture(buf);
        if (g.size() != params.size()) {
            std::fprintf(stderr, "rank %d: gradient fixture has %zu tensors, expected %zu\n",
                         g_rank, g.size(), params.size());
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        opt->zero_grad();
        for (size_t i = 0; i < params.size(); ++i)
            params[i].set_grad(g[i].data.data(), g[i].data.size());

        if (s == 0 && g_rank == 0) {
            // The optimizer reads gradients through param.has_grad()/grad_view();
            // if set_grad did not land where it looks, every reduce sees zeros and
            // the whole comparison silently comes out as "AdamW on no gradient".
            size_t with = 0; double n0 = 0.0;
            for (size_t i = 0; i < params.size(); ++i) {
                if (!params[i].has_grad()) continue;
                ++with;
                Tensor h = params[i].grad_view().as_type(Dtype::Float32).to_cpu();
                const float* d = h.data<float>();
                for (int64_t e = 0; e < h.numel(); ++e) n0 += double(d[e]) * d[e];
            }
            std::printf("  [debug] %zu/%zu params report has_grad, ||g||=%.9g\n",
                        with, params.size(), std::sqrt(n0));
        }

        opt->set_lr(lr);
        opt->step();
        opt->drain_param_refresh();

        if (g_rank == 0)
            std::printf("  step %d | grad_norm %.9g\n", s, opt->last_grad_norm());
    }

    char out[512];
    std::snprintf(out, sizeof(out), "%s/cpp_final_rank%d.bin", fx.c_str(), g_rank);
    write_fixture(out, params);
    if (g_rank == 0) std::printf("wrote %s (and peers)\n", out);

    MPI_Finalize();
    return 0;
}
