#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <cmath>
#include "TensorLib.h"
#include "../include/DataParallel.hpp"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * DDP Bucket tests with 2 ranks.
 * Tests bucket_order_decide logic and DDP bucketing with actual
 * multi-rank gradient synchronization.
 */


// ---- bucket_order_decide tests (pure logic, same on both ranks) ----

void test_bucket_order_single_param(int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto param = OwnTensor::Tensor::randn({{4, 4}}, opts);
    std::vector<OwnTensor::Tensor> params = {param};

    auto buckets = bucket_order_decide(params, {}, 25 * 1024 * 1024);

    TEST_ASSERT(buckets.size() >= 1, "bucket_order_decide returns at least 1 bucket for single param");
    TEST_ASSERT_EQ((int)buckets[0].size(), 1, "Single param bucket contains 1 index");
    TEST_ASSERT_EQ((int)buckets[0][0], 0, "Single param bucket index is 0");
}

void test_bucket_order_multiple_params_fit_one_bucket(int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto p1 = OwnTensor::Tensor::randn({{4, 4}}, opts);
    auto p2 = OwnTensor::Tensor::randn({{4, 4}}, opts);
    auto p3 = OwnTensor::Tensor::randn({{4, 4}}, opts);
    std::vector<OwnTensor::Tensor> params = {p1, p2, p3};

    auto buckets = bucket_order_decide(params, {}, 25 * 1024 * 1024);

    TEST_ASSERT_EQ((int)buckets.size(), 1, "3 small params fit in 1 bucket (25MB)");

    size_t total_indices = 0;
    for (auto& b : buckets) total_indices += b.size();
    TEST_ASSERT_EQ((int)total_indices, 3, "All 3 param indices accounted for");
}

void test_bucket_order_forces_split(int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    size_t tiny_bucket = 64;  // 64 bytes forces splitting

    auto p1 = OwnTensor::Tensor::randn({{100}}, opts);  // 400 bytes
    auto p2 = OwnTensor::Tensor::randn({{100}}, opts);
    std::vector<OwnTensor::Tensor> params = {p1, p2};

    auto buckets = bucket_order_decide(params, {}, tiny_bucket);

    TEST_ASSERT(buckets.size() >= 2, "Small bucket_size forces params into multiple buckets");
}

void test_bucket_order_with_backward_order(int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto p1 = OwnTensor::Tensor::randn({{10}}, opts);
    auto p2 = OwnTensor::Tensor::randn({{10}}, opts);
    auto p3 = OwnTensor::Tensor::randn({{10}}, opts);
    std::vector<OwnTensor::Tensor> params = {p1, p2, p3};

    std::vector<size_t> backward_order = {2, 1, 0};

    auto buckets = bucket_order_decide(params, backward_order, 25 * 1024 * 1024);

    TEST_ASSERT(buckets.size() >= 1, "bucket_order_decide with backward_order returns buckets");

    size_t total_indices = 0;
    for (auto& b : buckets) total_indices += b.size();
    TEST_ASSERT_EQ((int)total_indices, 3, "All indices present with backward_order");
}

void test_bucket_order_empty_bucket_size_throws(int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto p1 = OwnTensor::Tensor::randn({{10}}, opts);
    std::vector<OwnTensor::Tensor> params = {p1};

    TEST_ASSERT_THROWS(
        bucket_order_decide(params, {}, 0),
        "bucket_order_decide with bucket_size=0 throws"
    );
}


// ---- DDP bucket creation with multi-rank gradient verification ----

void test_ddp_bucket_creation_default(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    TEST_ASSERT(true, "DDP bucket creation with default settings succeeds");
}

void test_ddp_small_bucket_gradients_synced(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(16, 64),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(64, 16)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;
    opts.bucket_size_ = 128;  // Force multiple buckets

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 16}}, tensor_opts, rank * 500);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 16}}, tensor_opts, rank * 500 + 1);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    // Verify gradients are synced even with small bucket size
    auto params_vec = model.parameters();
    auto& first_param = params_vec[0];
    auto grad_cpu = first_param.grad_view().to_cpu();
    float* local_grad = static_cast<float*>(grad_cpu.data());
    size_t numel = grad_cpu.numel();

    std::vector<float> all_grads(numel * 2);
    MPI_Allgather(local_grad, numel, MPI_FLOAT, all_grads.data(), numel, MPI_FLOAT, MPI_COMM_WORLD);

    bool identical = true;
    for (size_t i = 0; i < numel; i++) {
        if (std::fabs(all_grads[i] - all_grads[numel + i]) > 1e-3) {
            identical = false;
            break;
        }
    }
    TEST_ASSERT(identical, "Gradients synced across ranks with small bucket_size=128");
}

void test_ddp_large_bucket_size(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(64, 256),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(256, 64)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;
    opts.bucket_size_ = 500 * 1024 * 1024;  // 500MB — all in one bucket

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    TEST_ASSERT(true, "DDP with large bucket_size (500MB) succeeds");
}

void test_ddp_params_count_preserved(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::ReLU(),
        new OwnTensor::nn::Linear(16, 4)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    size_t total_before = 0;
    for (auto& p : model.parameters()) total_before += p.numel();

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    size_t total_after = 0;
    for (auto& p : model.parameters()) total_after += p.numel();

    TEST_ASSERT_EQ((int)total_before, (int)total_after, "Parameter count preserved after DDP wrapping");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    // Pure logic tests
    test_bucket_order_single_param(rank);
    test_bucket_order_multiple_params_fit_one_bucket(rank);
    test_bucket_order_forces_split(rank);
    test_bucket_order_with_backward_order(rank);
    test_bucket_order_empty_bucket_size_throws(rank);

    // Multi-rank bucket tests
    test_ddp_bucket_creation_default(pg, rank);
    test_ddp_small_bucket_gradients_synced(pg, rank);
    test_ddp_large_bucket_size(pg, rank);
    test_ddp_params_count_preserved(pg, rank);

    print_test_summary("DDP Buckets");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
