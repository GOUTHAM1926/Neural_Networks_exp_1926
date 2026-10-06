//======================================================================
// test_zero_dist.cpp — distributed (W>1) correctness for ZeRO-1.
//
// Exercises the real NCCL reduce-scatter (gradient averaging) + all-gather
// (parameter reconstruction) path. Launch with:  mpirun -np 2 ./test_zero_dist
//
// Correctness contract: with rank r feeding gradient g_r, ZeRO's averaged
// step must equal a single-process AdamW step on mean(g_0..g_{W-1}). Every
// rank computes that reference locally (all g_r are deterministic) and, after
// all-gather, every rank must hold identical, correct parameters.
//======================================================================

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <memory>

#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "ProcessGroupNCCL.h"
#include "ShardedOptimizer.h"

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
using owt::zero::ShardedOptimizer;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, W = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &W);
    int ndev = 0; cudaGetDeviceCount(&ndev);
    cudaSetDevice(rank % std::max(1, ndev));
    auto dev = cuda_dev(rank % std::max(1, ndev));

    std::shared_ptr<ProcessGroupNCCL> pg = init_process_group(W, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    // shapes chosen so total (57) is NOT divisible by W=2 -> exercises padding
    std::vector<std::vector<int64_t>> shapes = {{4,6},{7},{3,5},{11}};
    std::mt19937 rng(2024);   // identical on every rank -> identical init
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<Tensor> ref, zed;
    for (auto& sp : shapes) {
        int64_t n = 1; for (auto d : sp) n *= d;
        std::vector<float> init(n);
        for (auto& x : init) x = nd(rng);
        ref.push_back(make_fp32(sp, init, dev));
        zed.push_back(make_fp32(sp, init, dev));
    }

    const float lr = 5e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.05f;
    OwnTensor::nn::AdamW ref_opt(ref, lr, b1, b2, eps, wd);
    ZeroConfig cfg; cfg.lr = lr; cfg.beta1 = b1; cfg.beta2 = b2; cfg.eps = eps;
    cfg.weight_decay = wd; cfg.verbose = (rank == 0);
    ShardedOptimizer zero_opt(zed, pg, cfg);

    float worst = 0.f;
    for (int step = 0; step < 5; ++step) {
        // deterministic per-rank grads g_r; every rank can build all of them.
        std::vector<std::vector<float>> gr(W);
        for (size_t i = 0; i < zed.size(); ++i) {
            int64_t n = zed[i].numel();
            std::vector<float> avg(n, 0.f);
            for (int r = 0; r < W; ++r) {
                std::mt19937 g(1000 * (step + 1) + 37 * r + 7 * (int)i);
                std::normal_distribution<float> gd(0.f, 0.5f);
                std::vector<float> gi(n);
                for (auto& x : gi) x = gd(g);
                for (int64_t k = 0; k < n; ++k) avg[k] += gi[k] / (float)W;
                if (r == rank) {
                    Shape s; s.dims = zed[i].shape().dims;
                    Tensor gt = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
                    gt.set_data(gi.data(), gi.size());
                    zed[i].set_grad(gt);            // this rank's local grad
                }
            }
            Shape s; s.dims = ref[i].shape().dims;
            Tensor at = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
            at.set_data(avg.data(), avg.size());
            ref[i].set_grad(at);                    // reference: the true mean grad
        }

        ref_opt.step();
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        zero_opt.step();
        cudaDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);

        for (size_t i = 0; i < ref.size(); ++i) {
            auto hr = host_of(ref[i]);
            auto hz = host_of(zed[i]);
            for (size_t k = 0; k < hr.size(); ++k)
                worst = std::max(worst, std::fabs(hr[k] - hz[k]));
        }
    }

    float global_worst = worst;
    MPI_Allreduce(&worst, &global_worst, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    int fail = (global_worst < 1e-4f) ? 0 : 1;
    if (rank == 0) {
        std::printf("[test_zero_dist W=%d] max|Δparam vs mean-grad AdamW| = %.3e -> %s\n",
                    W, global_worst, fail ? "FAIL" : "PASS");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return fail;
}
