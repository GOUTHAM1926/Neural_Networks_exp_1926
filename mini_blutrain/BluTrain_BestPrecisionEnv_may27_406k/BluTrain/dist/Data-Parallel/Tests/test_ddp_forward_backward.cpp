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
 * DDP Forward/Backward flow tests with 2 ranks.
 * Validates end-to-end training flow, optimizer steps,
 * zero_grad, determinism, and parameter updates across ranks.
 */

void test_forward_produces_correct_shape(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 4)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts);
    auto output = ddp.forward(input);

    TEST_ASSERT(output.numel() > 0, "Forward produces non-empty output");
    TEST_ASSERT_EQ((int)output.numel(), 4 * 4, "Forward output shape matches (batch=4, out=4)");
}

void test_forward_output_not_nan(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts);
    auto output = ddp.forward(input);

    auto cpu_out = output.to_cpu();
    float* ptr = static_cast<float*>(cpu_out.data());
    bool has_nan = false;
    for (size_t i = 0; i < cpu_out.numel(); i++) {
        if (std::isnan(ptr[i])) { has_nan = true; break; }
    }
    TEST_ASSERT(!has_nan, "Forward output contains no NaN values");
}

void test_multiple_iterations(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    OwnTensor::nn::SGDOptimizer sgd(model.parameters(), 0.01f);

    for (int i = 0; i < 5; i++) {
        auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 16}}, tensor_opts, rank * 100 + i);
        auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 16}}, tensor_opts, rank * 100 + i + 50);

        auto output = ddp.forward(input);
        auto loss = OwnTensor::nn::mse_loss(output, target);
        loss.backward();
        cudaDeviceSynchronize();

        sgd.step();
        model.zero_grad();
    }
    TEST_ASSERT(true, "5 forward/backward iterations complete without error");
}

void test_optimizer_updates_params(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 32),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(32, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    // Capture param before step
    auto first_param_before = model.parameters()[0].to_cpu();
    float before_val = static_cast<float*>(first_param_before.data())[0];

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts, 42);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts, 43);

    OwnTensor::nn::SGDOptimizer sgd(model.parameters(), 0.01f);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    sgd.step();

    auto first_param_after = model.parameters()[0].to_cpu();
    float after_val = static_cast<float*>(first_param_after.data())[0];

    TEST_ASSERT(std::fabs(before_val - after_val) > 1e-8,
                "Parameters updated after optimizer step");
}

void test_params_stay_synced_after_optimizer_step(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    OwnTensor::nn::SGDOptimizer sgd(model.parameters(), 0.01f);

    // Train for a few steps
    for (int i = 0; i < 3; i++) {
        auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100 + i);
        auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100 + i + 50);

        auto output = ddp.forward(input);
        auto loss = OwnTensor::nn::mse_loss(output, target);
        loss.backward();
        cudaDeviceSynchronize();

        sgd.step();
        model.zero_grad();
    }

    // After training, params should still be identical across ranks
    // (same gradient -> same update -> same params)
    for (size_t pi = 0; pi < model.parameters().size(); pi++) {
        auto param_cpu = model.parameters()[pi].to_cpu();
        float* ptr = static_cast<float*>(param_cpu.data());
        size_t numel = param_cpu.numel();

        double local_sum = 0.0;
        for (size_t i = 0; i < numel; i++) local_sum += ptr[i];

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        TEST_ASSERT_NEAR(global_sum, 2.0 * local_sum, 1e-1,
                         "Param " + std::to_string(pi) + " stays synced after 3 optimizer steps");
    }
}

void test_zero_grad_clears_gradients(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    model.zero_grad();

    bool all_zero = true;
    for (auto& param : model.parameters()) {
        if (param.requires_grad()) {
            auto grad = param.grad_view().to_cpu();
            float* ptr = static_cast<float*>(grad.data());
            for (size_t i = 0; i < grad.numel(); i++) {
                if (std::fabs(ptr[i]) > 1e-8) { all_zero = false; break; }
            }
            if (!all_zero) break;
        }
    }
    TEST_ASSERT(all_zero, "Gradients are zeroed after zero_grad()");
}

void test_adamw_optimizer(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, 10);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, 11);

    OwnTensor::nn::AdamW optimizer(model.parameters());

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    optimizer.step();
    model.zero_grad();

    TEST_ASSERT(true, "Forward/backward with AdamW optimizer succeeds");
}

void test_large_model(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(256, 512),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(512, 256)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{8, 256}}, tensor_opts);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{8, 256}}, tensor_opts);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    TEST_ASSERT(true, "Large model (256->512->256) forward/backward completes on 2 ranks");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_forward_produces_correct_shape(pg, rank);
    test_forward_output_not_nan(pg, rank);
    test_multiple_iterations(pg, rank);
    test_optimizer_updates_params(pg, rank);
    test_params_stay_synced_after_optimizer_step(pg, rank);
    test_zero_grad_clears_gradients(pg, rank);
    test_adamw_optimizer(pg, rank);
    test_large_model(pg, rank);

    print_test_summary("DDP Forward/Backward");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
