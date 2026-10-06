//======================================================================
// test_zero_v2_staged.cpp — regression test for the ZeRO-1 v2 staged parameter
// refresh partition (defect D4).
//
// WHAT IT PINS DOWN
// ----------------
// SymmetricShardedOptimizer::adam_and_publish_staged() splits the shard into two
// phases: the buckets in refresh stage 0 (published immediately, because the next
// forward reads them first) and every other bucket. It used to compute phase A as
// the single range [min stage-0 shard_offset, max stage-0 shard_offset+per_rank)
// and phase B as [that max, S_) — which only covers the shard when the stage-0
// buckets are a PREFIX of buckets_.
//
// If a stage-0 parameter sits mid-model, min offset > 0 and the shard range
// [0, min offset) got NO Adam update and NO down-cast into
// shard_master_compute_. That buffer is never written anywhere else on this path,
// so the all-gather published ZEROS for those parameters on every rank.
//
// This is reachable from the trainer: bluscript_zero.cpp pins every parameter it
// cannot identify by pointer to stage 0 (a deliberate "always correct, just not
// overlapped" fallback), and any such parameter that is not the flat prefix
// triggers it.
//
// The test therefore runs TWO stage assignments against the same reference:
//   scenario A — stage 0 = the FIRST parameter          (prefix; always worked)
//   scenario B — stage 0 = a MIDDLE parameter           (non-prefix; the defect)
// Both must match a single-process AdamW step on the mean gradient. Before the
// fix, scenario B fails with an error of order the parameter magnitude itself
// (weights published as zeros); after it, both match to fp32 round-off.
//
// Run:  mpirun -np 2 ./test_zero_v2_staged      (any W >= 2; 4 is better)
//======================================================================

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <memory>
#include <string>

#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "ProcessGroupNCCL.h"
#include "SymmetricShardedOptimizer.h"

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
using owt::zero::SymmetricShardedOptimizer;
using owt::zero::ZeroConfig;

static OwnTensor::DeviceIndex cuda_dev(int idx) {
    return OwnTensor::DeviceIndex{OwnTensor::Device::CUDA, idx};
}

static Tensor make_fp32(const std::vector<int64_t>& dims, const std::vector<float>& init,
                        OwnTensor::DeviceIndex dev) {
    Shape s; s.dims = dims;
    Tensor t = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
    t.set_data(init.data(), init.size());
    t.set_requires_grad(true);
    return t;
}

static std::vector<float> host_of(const Tensor& t) {
    Tensor c = t.to_cpu();
    const float* p = c.data<float>();
    return std::vector<float>(p, p + c.numel());
}

// Four 2-D parameters so the flat order equals the input order (no weight-decay
// group re-sorting), and each is big enough to span several buckets.
static const std::vector<std::vector<int64_t>> kShapes = {
    {512, 256}, {512, 256}, {512, 256}, {512, 256},
};

namespace {

struct Scenario {
    const char*      name;
    std::vector<int> stage_of;   // index-aligned with kShapes
    int              num_stages;
};

// Run `steps` optimizer steps of ZeRO-1 v2 with the given refresh-stage mapping
// and return the worst absolute deviation from a single-process AdamW step on the
// mean gradient. Every rank builds the same reference locally (all per-rank
// gradients are deterministic), so both the numerics AND cross-rank agreement are
// covered: if a rank published a wrong range, its own comparison fails.
float run_scenario(const Scenario& sc, int rank, int W,
                   OwnTensor::DeviceIndex dev,
                   std::shared_ptr<ProcessGroupNCCL> pg,
                   int steps) {
    const float lr = 5e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.05f;

    std::mt19937 rng(2024);                     // identical on every rank
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Tensor> ref, zed;
    for (const auto& sp : kShapes) {
        int64_t n = 1; for (auto d : sp) n *= d;
        std::vector<float> init(n);
        for (auto& x : init) x = nd(rng);
        ref.push_back(make_fp32(sp, init, dev));
        zed.push_back(make_fp32(sp, init, dev));
    }

    OwnTensor::nn::AdamW ref_opt(ref, lr, b1, b2, eps, wd);

    ZeroConfig cfg;
    cfg.lr = lr; cfg.beta1 = b1; cfg.beta2 = b2; cfg.eps = eps;
    cfg.weight_decay = wd;
    cfg.exclude_1d_from_weight_decay = false;   // keep flat order == input order
    cfg.init_broadcast = false;                 // identical init already
    // The path under test. overlap_backward_comm stays OFF so this isolates the
    // staged refresh from the hook-driven reduce.
    cfg.overlap_param_refresh = true;
    cfg.overlap_backward_comm = false;
    // Small buckets so there are many of them and a mid-model stage-0 parameter
    // really does land on a non-prefix bucket set.
    cfg.comm_bucket_bytes = 64 * 1024;
    cfg.verbose = false;

    SymmetricShardedOptimizer zero_opt(zed, pg, cfg);
    zero_opt.set_param_refresh_stages(sc.stage_of, sc.num_stages);

    float worst = 0.f;
    for (int step = 0; step < steps; ++step) {
        for (size_t i = 0; i < zed.size(); ++i) {
            const int64_t n = zed[i].numel();
            std::vector<float> avg(n, 0.f);
            for (int r = 0; r < W; ++r) {
                std::mt19937 g(1000 * (step + 1) + 37 * r + 7 * static_cast<int>(i));
                std::normal_distribution<float> gd(0.f, 0.5f);
                std::vector<float> gi(n);
                for (auto& x : gi) x = gd(g);
                for (int64_t k = 0; k < n; ++k) avg[k] += gi[k] / static_cast<float>(W);
                if (r == rank) {
                    Shape s; s.dims = zed[i].shape().dims;
                    Tensor gt = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32)
                                                                .with_device(dev));
                    gt.set_data(gi.data(), gi.size());
                    zed[i].set_grad(gt);
                }
            }
            Shape s; s.dims = ref[i].shape().dims;
            Tensor at = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32)
                                                        .with_device(dev));
            at.set_data(avg.data(), avg.size());
            ref[i].set_grad(at);
        }

        ref_opt.step();
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        zero_opt.step();
        // step() returns with every all-gather still in flight and the model
        // tensors still holding the PREVIOUS step's weights; the next forward
        // normally installs them stage by stage. There is no forward here, so
        // drain explicitly before reading the parameters.
        zero_opt.drain_param_refresh();
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < ref.size(); ++i) {
            auto hr = host_of(ref[i]);
            auto hz = host_of(zed[i]);
            for (size_t k = 0; k < hr.size(); ++k)
                worst = std::max(worst, std::fabs(hr[k] - hz[k]));
        }
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, W = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &W);

    if (W < 2) {
        if (rank == 0)
            std::printf("[test_zero_v2_staged] SKIP: the staged refresh is only active "
                        "at world size > 1 (run with mpirun -np 2 or more)\n");
        MPI_Finalize();
        return 0;
    }

    int ndev = 0; cudaGetDeviceCount(&ndev);
    const int dev_idx = rank % std::max(1, ndev);
    cudaSetDevice(dev_idx);
    auto dev = cuda_dev(dev_idx);

    std::shared_ptr<ProcessGroupNCCL> pg = init_process_group(W, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    const std::vector<Scenario> scenarios = {
        // Prefix: parameter 0 in stage 0. This always worked; it guards against a
        // regression in the common case.
        {"stage0 = first param (prefix)",  {0, 1, 1, 2}, 3},
        // Non-prefix: parameter 2 in stage 0 while 0 and 1 are in a later stage.
        // This is defect D4's trigger — the shape the trainer's unmatched-parameter
        // fallback produces.
        {"stage0 = middle param (non-prefix)", {1, 1, 0, 2}, 3},
        // Both ends in stage 0 with a gap in the middle: two disjoint stage-0
        // bucket runs, so phase A must handle more than one range.
        {"stage0 = first and last (split)", {0, 1, 1, 0}, 2},
    };

    int fail = 0;
    for (const auto& sc : scenarios) {
        const float worst = run_scenario(sc, rank, W, dev, pg, /*steps=*/3);
        float global = worst;
        MPI_Allreduce(&worst, &global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        const bool bad = !(global < 1e-4f);
        if (bad) fail = 1;
        if (rank == 0) {
            std::printf("[test_zero_v2_staged W=%d] %-38s max|Δparam| = %.3e -> %s\n",
                        W, sc.name, global, bad ? "FAIL" : "PASS");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::printf("[test_zero_v2_staged] %s\n", fail ? "FAILED" : "ALL PASS");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return fail;
}
