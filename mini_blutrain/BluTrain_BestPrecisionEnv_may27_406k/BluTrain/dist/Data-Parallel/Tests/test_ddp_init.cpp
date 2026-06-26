#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include <cmath>
#include "TensorLib.h"
#include "../include/DataParallel.hpp"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * DDP Initialization tests with 2 ranks.
 * Verifies that after DDP construction with init_sync=true,
 * both ranks end up with identical model parameters.
 */

void test_ddp_construction(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(16, 32),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(32, 16)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    TEST_ASSERT(true, "DDP construction with init_sync=true succeeds");
}

void test_ddp_params_synced_across_ranks(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    // Both ranks create models with DIFFERENT random seeds
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    // init_sync=true broadcasts rank 0's params to all ranks
    DataParallel ddp(&model, opts, true);
    pg->blockStream();
    cudaDeviceSynchronize();

    // After sync, both ranks should have identical parameters.
    // Verify by allreducing the parameter sum — if identical, SUM == 2 * value on each rank
    for (size_t pi = 0; pi < model.parameters().size(); pi++) {
        auto param = model.parameters()[pi];
        auto cpu_param = param.to_cpu();
        float* ptr = static_cast<float*>(cpu_param.data());

        // Compute local sum of this parameter
        double local_sum = 0.0;
        for (size_t i = 0; i < cpu_param.numel(); i++) {
            local_sum += ptr[i];
        }

        // AllReduce the local sum across ranks using MPI (independent of NCCL)
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // If both ranks have the same params, global_sum == 2 * local_sum
        TEST_ASSERT_NEAR(global_sum, 2.0 * local_sum, 1e-2,
                         "Param " + std::to_string(pi) + " identical across ranks after DDP init");
    }
}

void test_ddp_construction_no_sync(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(16, 32),
        new OwnTensor::nn::Linear(32, 16)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, false);

    TEST_ASSERT(true, "DDP construction with init_sync=false succeeds");
}

void test_ddp_model_has_grad_tensors(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 4)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    bool has_grad = false;
    for (auto& param : model.parameters()) {
        if (param.requires_grad()) { has_grad = true; break; }
    }
    TEST_ASSERT(has_grad, "Model has at least one parameter requiring grad");

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    // has_synced_params_ is not set internally by the constructor;
    // it requires an explicit set_param_sync(true) call.
    // Verify the flag starts as false (user must opt-in).
    TEST_ASSERT(!ddp.is_param_sync(), "is_param_sync() is false by default (user must set manually)");
}

void test_ddp_multiple_constructions(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    for (int i = 0; i < 3; i++) {
        OwnTensor::nn::Sequential model({
            new OwnTensor::nn::Linear(4, 8),
            new OwnTensor::nn::Linear(8, 4)
        });
        model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

        DDP_Options opts;
        opts.process_group_ = pg;
        opts.rank_ = rank;
        opts.local_rank_ = rank;

        DataParallel ddp(&model, opts, true);
        pg->blockStream();
    }
    TEST_ASSERT(true, "Multiple DDP constructions succeed without error");
}

void test_ddp_opts_accessor(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;
    opts.bucket_size_ = 10 * 1024 * 1024;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    auto retrieved = ddp.ddp_opts();
    TEST_ASSERT_EQ((int)retrieved.rank_, (int)rank, "ddp_opts().rank_ matches");
    TEST_ASSERT_EQ((int)retrieved.local_rank_, (int)rank, "ddp_opts().local_rank_ matches");
    TEST_ASSERT_EQ(retrieved.bucket_size_, (int64_t)(10 * 1024 * 1024), "ddp_opts().bucket_size_ matches");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_ddp_construction(pg, rank);
    test_ddp_params_synced_across_ranks(pg, rank, world_size);
    test_ddp_construction_no_sync(pg, rank);
    test_ddp_model_has_grad_tensors(pg, rank);
    test_ddp_multiple_constructions(pg, rank);
    test_ddp_opts_accessor(pg, rank);

    print_test_summary("DDP Initialization");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
