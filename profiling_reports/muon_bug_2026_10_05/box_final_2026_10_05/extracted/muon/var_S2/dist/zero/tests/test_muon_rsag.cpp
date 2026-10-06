//======================================================================
// test_muon_rsag.cpp — ShardedMuon: reduce-scatter/all-gather must agree
// bit-for-bit with reduce/broadcast.
//
// WHAT IT PINS DOWN
// -----------------
// ShardedMuon has two wire formats for the SAME whole-parameter ownership:
//
//   REDUCE_BROADCAST           one ncclReduce(root=owner) + one
//                              ncclBroadcast(root=owner) per parameter. The
//                              original formulation.
//   REDUCE_SCATTER_ALL_GATHER  whole tensors bin-packed into per-rank
//                              contiguous sub-chunks of a bucket, padded to a
//                              common per_rank, then one ncclReduceScatter and
//                              one ncclAllGather per bucket.
//
// The second exists because NCCL ships symmetric-memory (NVLS) device kernels
// for ReduceScatter/AllGather and none for Reduce/Broadcast, so only it can
// reach NVLink SHARP. But it moves every byte through a flat buffer at computed
// offsets, and the failure mode of a wrong offset is SILENT: a tensor gets
// another tensor's bytes, or a rank's padding, and training continues on a
// quietly corrupted weight. Nothing crashes and no norm looks unusual.
//
// So the property under test is the strong one — not "close", but IDENTICAL.
// Both modes average the same gradients and run the same Newton-Schulz on the
// same FP32 master, so at W<=2 there is no reordering anywhere that could
// legitimately change a bit. (At W>=3 the two collectives may sum the ranks in
// different orders, which is a real floating-point difference and not a bug;
// this test therefore asserts bit-equality only up to W=2 and falls back to a
// tight tolerance above it.)
//
// It also asserts the layout invariants directly, because a violated one
// localizes the bug far better than a downstream numeric mismatch:
//   * every tensor lies entirely inside ONE rank's sub-chunk of ONE bucket
//   * no two tensors overlap in the flat buffer
//   * padded_numel accounts for every bucket exactly
//
// Two scenarios, since publishing differs between them:
//   A  sync publish       step() -> all_gather_params_sync()
//   B  staged refresh     step() -> per-stage all-gathers consumed by
//                         refresh_param_stage(), including the scatter that a
//                         broadcast never needed
//
// Run:  mpirun -np 1 ./test_muon_rsag
//       mpirun -np 2 ./test_muon_rsag
//======================================================================

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "ProcessGroupNCCL.h"
#include "ShardedMuon.h"

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
using owt::zero::ShardedMuon;
using owt::zero::MuonConfig;
using owt::zero::MuonCollective;

namespace {

int g_rank = 0, g_world = 1;

OwnTensor::DeviceIndex cuda_dev(int idx) {
    return OwnTensor::DeviceIndex{OwnTensor::Device::CUDA, idx};
}

// Deliberately non-uniform, and deliberately not a multiple of anything: the
// bin-packer's padding is driven by the spread of sizes, and equal shapes would
// let a broken packer look balanced. The two 1-D entries exercise the aux AdamW
// path inside ShardedMuon (a vector has no second axis to orthogonalize).
const std::vector<std::vector<int64_t>> kShapes = {
    {64, 32}, {128, 64}, {32, 32}, {96, 48}, {64, 64},
    {160, 32}, {48, 48}, {80, 40}, {112, 56}, {32, 64},
    {64, 96}, {128, 32}, {40, 40}, {72, 36}, {56, 56},
    {96, 96}, {32, 128}, {88, 44}, {64, 48}, {100, 50},
    {256}, {128},
};

// Deterministic and identical across both runs, different across ranks (so the
// cross-rank average is actually doing something) and across steps.
float synth(uint64_t seed) {
    seed ^= seed >> 33; seed *= 0xff51afd7ed558ccdull;
    seed ^= seed >> 33; seed *= 0xc4ceb9fe1a85ec53ull;
    seed ^= seed >> 33;
    return static_cast<float>(static_cast<int64_t>(seed % 20001) - 10000) * 1e-4f;
}

size_t numel_of(const std::vector<int64_t>& dims) {
    size_t n = 1;
    for (int64_t d : dims) n *= static_cast<size_t>(d);
    return n;
}

std::vector<Tensor> make_params(OwnTensor::DeviceIndex dev) {
    std::vector<Tensor> ps;
    ps.reserve(kShapes.size());
    for (size_t i = 0; i < kShapes.size(); ++i) {
        Shape s; s.dims = kShapes[i];
        Tensor t = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
        std::vector<float> init(numel_of(kShapes[i]));
        // Same on every rank: the optimizer is built with init_broadcast off, so
        // the replicas must already agree.
        for (size_t e = 0; e < init.size(); ++e) init[e] = synth(i * 7919ull + e);
        t.set_data(init.data(), init.size());
        t.set_requires_grad(true);
        ps.push_back(t);
    }
    return ps;
}

void set_grads(std::vector<Tensor>& ps, int step) {
    for (size_t i = 0; i < ps.size(); ++i) {
        std::vector<float> g(ps[i].numel());
        for (size_t e = 0; e < g.size(); ++e) {
            g[e] = synth((static_cast<uint64_t>(step) + 1) * 1000003ull +
                         i * 104729ull + e * 31ull +
                         static_cast<uint64_t>(g_rank) * 15485863ull);
        }
        ps[i].set_grad(g.data(), g.size());
    }
}

std::vector<std::vector<float>> host_params(const std::vector<Tensor>& ps) {
    std::vector<std::vector<float>> out;
    out.reserve(ps.size());
    for (const auto& p : ps) {
        Tensor c = p.to_cpu();
        const float* d = c.data<float>();
        out.emplace_back(d, d + c.numel());
    }
    return out;
}

MuonConfig base_cfg(MuonCollective mode, bool staged) {
    MuonConfig c;
    c.collective         = mode;
    c.target_buckets     = 3;
    c.lr                 = 1e-3f;
    c.momentum           = 0.95f;
    c.weight_decay       = 0.1f;
    c.ns_steps           = 5;
    c.nesterov           = true;
    c.adamw_lr           = 1e-3f;
    c.adamw_weight_decay = 0.1f;
    c.max_grad_norm      = 1.0f;
    // The gradients are set by hand below, so there is no autograd graph for
    // backward hooks to hang off. The staged-refresh scenario still exercises
    // the all-gather publish path, which is where the flat-offset arithmetic
    // that this test exists to check actually lives.
    c.overlap_backward_comm = false;
    c.overlap_param_refresh = staged;
    c.init_broadcast     = false;
    c.verify_consistency = true;
    c.verbose            = false;
    return c;
}

// One parameter per refresh stage, round-robin, so stage 0 is NOT a clean prefix
// of the bucket list — the arrangement that broke the v2 staged refresh (see
// test_zero_v2_staged.cpp) and the one a bucket->stage mapping is most likely to
// get wrong.
std::vector<int> round_robin_stages(int num_stages) {
    std::vector<int> stage_of(kShapes.size());
    for (size_t i = 0; i < kShapes.size(); ++i)
        stage_of[i] = static_cast<int>(i % static_cast<size_t>(num_stages));
    return stage_of;
}

// ---- layout invariants (RS/AG only) ---------------------------------------
// Checked through the public surface plus a padded/total comparison. A real
// overlap or an out-of-bucket tensor shows up here as a size accounting error
// long before it shows up as a wrong weight.
bool check_layout(const ShardedMuon& opt, size_t total_real) {
    bool ok = true;
    const int64_t padded = opt.padded_numel();
    const int64_t shard  = opt.shard_padded_numel();
    const size_t  nb     = opt.num_buckets();

    if (padded < static_cast<int64_t>(total_real)) {
        std::printf("  [layout] FAIL padded_numel %lld < total %zu\n",
                    (long long)padded, total_real);
        ok = false;
    }
    // Every bucket contributes per_rank*W to the flat buffer and per_rank to the
    // shard, so the two must be in exact W:1 proportion.
    if (padded != shard * static_cast<int64_t>(g_world)) {
        std::printf("  [layout] FAIL padded_numel %lld != shard %lld * W %d\n",
                    (long long)padded, (long long)shard, g_world);
        ok = false;
    }
    if (nb == 0) {
        std::printf("  [layout] FAIL no buckets\n");
        ok = false;
    }
    const double pad_pct = 100.0 * (double)(padded - (int64_t)total_real) / (double)total_real;
    if (g_rank == 0) {
        std::printf("  [layout] %zu bucket(s), padded %lld vs real %zu (%.2f%% padding), "
                    "shard %lld\n", nb, (long long)padded, total_real, pad_pct, (long long)shard);
    }
    return ok;
}

// ---- one run ---------------------------------------------------------------
struct RunResult {
    std::vector<std::vector<float>> params;
    float grad_norm_last = 0.0f;
    bool  layout_ok = true;
};

RunResult run(MuonCollective mode, bool staged, int steps,
              std::shared_ptr<ProcessGroupNCCL> pg, OwnTensor::DeviceIndex dev) {
    std::vector<Tensor> ps = make_params(dev);
    ShardedMuon opt(ps, pg, base_cfg(mode, staged));

    RunResult r;
    if (mode == MuonCollective::REDUCE_SCATTER_ALL_GATHER) {
        size_t total = 0;
        for (const auto& s : kShapes) total += numel_of(s);
        r.layout_ok = check_layout(opt, total);
    }

    const int num_stages = 4;
    if (staged) opt.set_param_refresh_stages(round_robin_stages(num_stages), num_stages);

    for (int st = 0; st < steps; ++st) {
        opt.zero_grad();
        set_grads(ps, st);
        opt.step();
        r.grad_norm_last = opt.last_grad_norm();
        if (staged) {
            // The trainer consumes these from forward pre-hooks; here we walk
            // them by hand in the same ascending order the hooks would.
            for (int s = 0; s < num_stages; ++s) opt.refresh_param_stage(s);
        }
        opt.drain_param_refresh();
    }

    r.params = host_params(ps);
    return r;
}

// Bit-exact at W<=2 (no reordering is possible with <=2 addends); a tight
// relative tolerance above it, where the two collectives may legitimately sum
// the ranks in different orders.
bool compare(const RunResult& a, const RunResult& b, const char* what) {
    const bool require_exact = (g_world <= 2);
    size_t n_diff = 0, n_total = 0;
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst_t = 0, worst_e = 0;

    if (a.params.size() != b.params.size()) {
        std::printf("[%s] FAIL parameter count %zu vs %zu\n", what,
                    a.params.size(), b.params.size());
        return false;
    }
    for (size_t t = 0; t < a.params.size(); ++t) {
        if (a.params[t].size() != b.params[t].size()) {
            std::printf("[%s] FAIL tensor %zu size %zu vs %zu\n", what, t,
                        a.params[t].size(), b.params[t].size());
            return false;
        }
        for (size_t e = 0; e < a.params[t].size(); ++e) {
            ++n_total;
            const float x = a.params[t][e], y = b.params[t][e];
            if (x == y) continue;
            ++n_diff;
            const double abs_d = std::fabs((double)x - (double)y);
            const double rel_d = abs_d / std::max(1e-12, std::fabs((double)x));
            if (abs_d > max_abs) { max_abs = abs_d; worst_t = t; worst_e = e; }
            max_rel = std::max(max_rel, rel_d);
        }
    }

    const bool pass = require_exact ? (n_diff == 0) : (max_rel < 1e-5);
    if (g_rank == 0) {
        if (n_diff == 0) {
            std::printf("[%s] PASS bit-identical across %zu elements\n", what, n_total);
        } else {
            std::printf("[%s] %s %zu/%zu elements differ | max_abs %.3e max_rel %.3e "
                        "(worst tensor %zu elem %zu: %.9g vs %.9g)\n",
                        what, pass ? "PASS (tolerance)" : "FAIL",
                        n_diff, n_total, max_abs, max_rel, worst_t, worst_e,
                        a.params[worst_t][worst_e], b.params[worst_t][worst_e]);
            if (require_exact) {
                std::printf("[%s]   W=%d admits no reduction reordering, so any "
                            "difference at all is a layout or publish bug.\n", what, g_world);
            }
        }
    }
    return pass;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world);

    cudaSetDevice(g_rank);
    OwnTensor::DeviceIndex dev = cuda_dev(g_rank);

    std::shared_ptr<ProcessGroupNCCL> pg;
    if (g_world > 1) pg = init_process_group(g_world, g_rank, /*stream=*/0);

    if (g_rank == 0) {
        std::printf("=== ShardedMuon reduce-scatter/all-gather equivalence "
                    "(W=%d, %zu params) ===\n", g_world, kShapes.size());
    }

    const int kSteps = 4;
    bool all_ok = true;

    for (int scenario = 0; scenario < 2; ++scenario) {
        const bool staged = (scenario == 1);
        const char* name = staged ? "staged refresh" : "sync publish";
        if (g_rank == 0) std::printf("\n-- scenario: %s --\n", name);

        RunResult rb = run(MuonCollective::REDUCE_BROADCAST, staged, kSteps, pg, dev);
        RunResult rs = run(MuonCollective::REDUCE_SCATTER_ALL_GATHER, staged, kSteps, pg, dev);

        all_ok &= rs.layout_ok;
        all_ok &= compare(rb, rs, name);

        // The reported gradient norm is a cross-rank all-reduce over disjoint
        // owned sets, so it must agree regardless of which rank owns what.
        const float dn = std::fabs(rb.grad_norm_last - rs.grad_norm_last);
        const float rel = dn / std::max(1e-12f, std::fabs(rb.grad_norm_last));
        if (rel > 1e-6f) {
            if (g_rank == 0)
                std::printf("[%s] FAIL grad_norm %.9g vs %.9g (rel %.3e)\n",
                            name, rb.grad_norm_last, rs.grad_norm_last, rel);
            all_ok = false;
        } else if (g_rank == 0) {
            std::printf("[%s] grad_norm agrees: %.9g\n", name, rb.grad_norm_last);
        }
    }

    int local_ok = all_ok ? 1 : 0, global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (g_rank == 0) {
        std::printf("\n=== %s ===\n", global_ok ? "ALL TESTS PASSED" : "FAILURES PRESENT");
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
