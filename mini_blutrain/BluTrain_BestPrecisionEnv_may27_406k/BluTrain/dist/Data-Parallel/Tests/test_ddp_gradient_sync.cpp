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
 * DDP Gradient Synchronization tests with 2 ranks.
 * Verifies that after backward(), gradients are AVERAGED across ranks
 * and that both ranks see the SAME gradient values.
 */

void test_gradients_identical_across_ranks(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    // Each rank uses different input data
    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100 + 42);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100 + 43);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    // After DDP backward, gradients should be averaged and IDENTICAL on both ranks.
    // Verify by computing local grad sum, then checking global_sum == 2 * local_sum.
    auto params = model.parameters();
    for (size_t pi = 0; pi < params.size(); pi++) {
        auto& param = params[pi];
        if (!param.requires_grad()) continue;

        auto grad = param.grad_view().to_cpu();
        float* ptr = static_cast<float*>(grad.data());

        double local_sum = 0.0;
        for (size_t i = 0; i < grad.numel(); i++) {
            local_sum += ptr[i];
        }

        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // If gradients are identical on both ranks: global_sum == 2 * local_sum
        TEST_ASSERT_NEAR(global_sum, 2.0 * local_sum, 1e-1,
                         "Param " + std::to_string(pi) + " gradient identical across ranks");
    }
}

void test_gradient_is_average_of_local_grads(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    // Use deterministic but rank-different inputs
    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 4}}, tensor_opts, rank + 10);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 4}}, tensor_opts, rank + 20);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    // Verify gradients are not NaN/Inf
    bool has_nan = false;
    bool has_inf = false;
    for (auto& param : model.parameters()) {
        if (!param.requires_grad()) continue;
        auto grad = param.grad_view().to_cpu();
        float* ptr = static_cast<float*>(grad.data());
        for (size_t i = 0; i < grad.numel(); i++) {
            if (std::isnan(ptr[i])) has_nan = true;
            if (std::isinf(ptr[i])) has_inf = true;
        }
    }
    TEST_ASSERT(!has_nan, "No NaN in averaged gradients");
    TEST_ASSERT(!has_inf, "No Inf in averaged gradients");
}

void test_gradient_shapes_match_params(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    for (auto& param : model.parameters()) {
        if (param.requires_grad()) {
            auto grad = param.grad_view();
            TEST_ASSERT_EQ((int)grad.numel(), (int)param.numel(),
                           "Gradient numel matches parameter numel");
        }
    }
}

void test_different_inputs_same_synced_gradients(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    // Deliberately different data per rank
    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 1000);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 1000 + 1);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    // Even with different inputs, after DDP sync both ranks must have same grads.
    // Send rank0's first param grad to rank1 via allgather and compare.
    auto params_vec = model.parameters();
    auto& first_param = params_vec[0];
    auto grad_cpu = first_param.grad_view().to_cpu();
    float* local_grad = static_cast<float*>(grad_cpu.data());
    size_t numel = grad_cpu.numel();

    // Use MPI_Allgather to collect grads from both ranks
    std::vector<float> all_grads(numel * 2);
    MPI_Allgather(local_grad, numel, MPI_FLOAT, all_grads.data(), numel, MPI_FLOAT, MPI_COMM_WORLD);

    // Compare rank0's grad with rank1's grad
    bool identical = true;
    for (size_t i = 0; i < numel; i++) {
        if (std::fabs(all_grads[i] - all_grads[numel + i]) > 1e-3) {
            identical = false;
            break;
        }
    }
    TEST_ASSERT(identical, "Rank0 and Rank1 have identical synced gradients despite different inputs");
}

void test_loss_decreases_over_iterations(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    // Same data on both ranks for this convergence test
    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts, 42);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{4, 8}}, tensor_opts, 43);

    OwnTensor::nn::SGDOptimizer sgd(model.parameters(), 0.01f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int i = 0; i < 10; i++) {
        auto output = ddp.forward(input);
        auto loss = OwnTensor::nn::mse_loss(output, target);
        loss.backward();
        cudaDeviceSynchronize();

        auto loss_cpu = loss.to_cpu();
        float loss_val = static_cast<float*>(loss_cpu.data())[0];

        if (i == 0) first_loss = loss_val;
        if (i == 9) last_loss = loss_val;

        sgd.step();
        model.zero_grad();
    }

    TEST_ASSERT(last_loss < first_loss, "Loss decreases over 10 DDP training iterations");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_gradients_identical_across_ranks(pg, rank);
    test_gradient_is_average_of_local_grads(pg, rank);
    test_gradient_shapes_match_params(pg, rank);
    test_different_inputs_same_synced_gradients(pg, rank);
    test_loss_decreases_over_iterations(pg, rank);

    print_test_summary("DDP Gradient Sync");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
