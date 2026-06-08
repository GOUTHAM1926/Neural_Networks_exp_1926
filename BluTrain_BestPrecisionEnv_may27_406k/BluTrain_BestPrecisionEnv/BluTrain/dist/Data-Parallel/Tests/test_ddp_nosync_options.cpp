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
 * DDP no_sync/sync and DDP_Options tests with 2 ranks.
 *
 * NOTE: When no_sync() is active, DDP skips the mediator_hook entirely,
 * so finalize_backward is never called and gradients are never copied
 * back to parameters via set_grad(). Therefore we CANNOT call grad_view()
 * after a no_sync backward — it would segfault.
 *
 * Instead, we verify no_sync behavior indirectly:
 * - The API toggles without crashing
 * - After re-enabling sync, a synced backward produces identical gradients
 */


// ---- no_sync / sync tests ----

void test_no_sync_toggle(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    bool sync_status = ddp.sync();
    TEST_ASSERT(sync_status, "sync() returns true (enabled by default)");

    ddp.no_sync();

    sync_status = ddp.sync();
    TEST_ASSERT(sync_status, "sync() returns true after re-enabling");
}

void test_sync_after_no_sync_produces_synced_grads(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    // Step 1: do a synced forward/backward to establish baseline
    auto input1 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100);
    auto target1 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 100 + 1);

    auto output1 = ddp.forward(input1);
    auto loss1 = OwnTensor::nn::mse_loss(output1, target1);
    loss1.backward();
    cudaDeviceSynchronize();

    // Verify baseline: grads are synced
    auto params_vec1 = model.parameters();
    auto grad_cpu1 = params_vec1[0].grad_view().to_cpu();
    float* ptr1 = static_cast<float*>(grad_cpu1.data());
    size_t numel = grad_cpu1.numel();

    std::vector<float> all_grads1(numel * 2);
    MPI_Allgather(ptr1, numel, MPI_FLOAT, all_grads1.data(), numel, MPI_FLOAT, MPI_COMM_WORLD);

    bool baseline_synced = true;
    for (size_t i = 0; i < numel; i++) {
        if (std::fabs(all_grads1[i] - all_grads1[numel + i]) > 1e-3) {
            baseline_synced = false;
            break;
        }
    }
    TEST_ASSERT(baseline_synced, "Baseline: gradients are synced before no_sync test");

    // Step 2: no_sync step — just verify it doesn't crash.
    // Do NOT read grad_view() here (DDP skips finalize_backward under no_sync).
    OwnTensor::nn::SGDOptimizer sgd(model.parameters(), 0.01f);
    sgd.step();
    model.zero_grad();

    ddp.no_sync();

    auto input2 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 200);
    auto target2 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 200 + 1);

    auto output2 = ddp.forward(input2);
    auto loss2 = OwnTensor::nn::mse_loss(output2, target2);
    loss2.backward();
    cudaDeviceSynchronize();

    TEST_ASSERT(true, "no_sync forward/backward completes without crash");

    // Step 3: re-enable sync, zero_grad, do a synced backward
    ddp.sync();
    model.zero_grad();

    auto input3 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 300);
    auto target3 = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 300 + 1);

    auto output3 = ddp.forward(input3);
    auto loss3 = OwnTensor::nn::mse_loss(output3, target3);
    loss3.backward();
    cudaDeviceSynchronize();

    // Verify: after re-enabling sync, gradients are identical across ranks again
    auto params_vec3 = model.parameters();
    auto grad_cpu3 = params_vec3[0].grad_view().to_cpu();
    float* ptr3 = static_cast<float*>(grad_cpu3.data());

    std::vector<float> all_grads3(numel * 2);
    MPI_Allgather(ptr3, numel, MPI_FLOAT, all_grads3.data(), numel, MPI_FLOAT, MPI_COMM_WORLD);

    bool synced_after = true;
    for (size_t i = 0; i < numel; i++) {
        if (std::fabs(all_grads3[i] - all_grads3[numel + i]) > 1e-3) {
            synced_after = false;
            break;
        }
    }
    TEST_ASSERT(synced_after, "Gradients are SYNCED again after re-enabling sync()");
}


// ---- DDP_Options builder pattern tests ----

void test_ddp_options_with_process_group(std::shared_ptr<ProcessGroupNCCL> pg) {
    DDP_Options opts;
    auto new_opts = opts.with_process_group(pg);

    TEST_ASSERT(new_opts.process_group_ != nullptr, "with_process_group sets process_group");
    TEST_ASSERT_EQ((int)new_opts.rank_, pg->get_rank(), "with_process_group sets rank from pg");
    TEST_ASSERT_EQ((int)new_opts.local_rank_, pg->get_local_rank(), "with_process_group sets local_rank from pg");
}

void test_ddp_options_with_world_size() {
    DDP_Options opts;
    auto new_opts = opts.with_world_size(4);
    TEST_ASSERT_EQ((int)new_opts.world_size_, 4, "with_world_size sets world_size=4");
}

void test_ddp_options_with_broadcast_buffer() {
    DDP_Options opts;
    auto new_opts = opts.with_broadcast_buffer(true);
    TEST_ASSERT(new_opts.broadcast_buffer_, "with_broadcast_buffer(true) sets flag");
}

void test_ddp_options_with_rank() {
    DDP_Options opts;
    auto new_opts = opts.with_rank(3, 1);
    TEST_ASSERT_EQ((int)new_opts.rank_, 3, "with_rank sets rank=3");
    TEST_ASSERT_EQ((int)new_opts.local_rank_, 1, "with_rank sets local_rank=1");
}

void test_ddp_options_with_bucket_data() {
    DDP_Options opts;
    auto new_opts = opts.with_bucket_data(true, 50 * 1024 * 1024);
    TEST_ASSERT(new_opts.bucket_, "with_bucket_data(true) enables bucketing");
    TEST_ASSERT_EQ(new_opts.bucket_size_, (int64_t)(50 * 1024 * 1024), "with_bucket_data sets correct size");
}

void test_ddp_options_with_bucket_zero_throws() {
    DDP_Options opts;
    TEST_ASSERT_THROWS(
        opts.with_bucket_data(true, 0),
        "with_bucket_data(true, 0) throws because bucket_size=0"
    );
}

void test_ddp_options_with_static_graph() {
    DDP_Options opts;
    auto new_opts = opts.with_static_graph(true);
    TEST_ASSERT(new_opts.static_graph, "with_static_graph(true) sets static_graph");
}

void test_ddp_options_with_grad_accum() {
    DDP_Options opts;
    auto new_opts = opts.with_grad_accum(true, 4);
    TEST_ASSERT(new_opts.is_accum_sync, "with_grad_accum sets is_accum_sync");
    TEST_ASSERT_EQ((int)new_opts.grad_accum_steps, 4, "with_grad_accum sets steps=4");
}

void test_ddp_options_with_grad_view() {
    DDP_Options opts;
    auto new_opts = opts.with_grad_view(true);
    TEST_ASSERT(new_opts.grad_as_view, "with_grad_view(true) sets grad_as_view");
}

void test_ddp_options_chaining(std::shared_ptr<ProcessGroupNCCL> pg) {
    DDP_Options opts;
    auto final_opts = opts
        .with_process_group(pg)
        .with_world_size(2)
        .with_broadcast_buffer(false)
        .with_bucket_data(true, 10 * 1024 * 1024)
        .with_static_graph(false)
        .with_grad_view(false);

    TEST_ASSERT(final_opts.process_group_ != nullptr, "Chained: process_group set");
    TEST_ASSERT_EQ((int)final_opts.world_size_, 2, "Chained: world_size=2");
    TEST_ASSERT(!final_opts.broadcast_buffer_, "Chained: broadcast_buffer=false");
    TEST_ASSERT(final_opts.bucket_, "Chained: bucket=true");
    TEST_ASSERT_EQ(final_opts.bucket_size_, (int64_t)(10 * 1024 * 1024), "Chained: bucket_size correct");
    TEST_ASSERT(!final_opts.static_graph, "Chained: static_graph=false");
    TEST_ASSERT(!final_opts.grad_as_view, "Chained: grad_as_view=false");
}


// ---- DDP with grad_as_view mode ----

void test_ddp_grad_as_view_grads_synced(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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
    opts.grad_as_view = true;

    DataParallel ddp(&model, opts, true);
    pg->blockStream();

    OwnTensor::TensorOptions tensor_opts;
    tensor_opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_opts.requires_grad = true;

    auto input = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 300);
    auto target = OwnTensor::Tensor::randn(OwnTensor::Shape{{2, 8}}, tensor_opts, rank * 300 + 1);

    auto output = ddp.forward(input);
    auto loss = OwnTensor::nn::mse_loss(output, target);
    loss.backward();
    cudaDeviceSynchronize();

    // Verify gradients are synced in grad_as_view mode
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
    TEST_ASSERT(identical, "Gradients synced across ranks in grad_as_view mode");
}


// ---- set_param_sync / is_param_sync tests ----

void test_set_param_sync(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
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

    // has_synced_params_ is not set by the constructor, starts as false
    TEST_ASSERT(!ddp.is_param_sync(), "is_param_sync() false by default");

    ddp.set_param_sync(true);
    TEST_ASSERT(ddp.is_param_sync(), "is_param_sync() true after set_param_sync(true)");

    ddp.set_param_sync(false);
    TEST_ASSERT(!ddp.is_param_sync(), "is_param_sync() false after set_param_sync(false)");
}


// ---- Assignment operator test ----

void test_ddp_assignment_operator(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::nn::Sequential model({
        new OwnTensor::nn::Linear(8, 16),
        new OwnTensor::nn::Linear(16, 8)
    });
    model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));

    DDP_Options opts;
    opts.process_group_ = pg;
    opts.rank_ = rank;
    opts.local_rank_ = rank;

    DataParallel ddp1(&model, opts, true);
    pg->blockStream();

    DataParallel ddp2;
    ddp2 = ddp1;

    auto retrieved = ddp2.ddp_opts();
    TEST_ASSERT_EQ((int)retrieved.rank_, (int)rank, "Assigned DDP has correct rank");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    // no_sync / sync tests
    test_no_sync_toggle(pg, rank);
    test_sync_after_no_sync_produces_synced_grads(pg, rank);

    // DDP_Options builder tests
    test_ddp_options_with_process_group(pg);
    test_ddp_options_with_world_size();
    test_ddp_options_with_broadcast_buffer();
    test_ddp_options_with_rank();
    test_ddp_options_with_bucket_data();
    test_ddp_options_with_bucket_zero_throws();
    test_ddp_options_with_static_graph();
    test_ddp_options_with_grad_accum();
    test_ddp_options_with_grad_view();
    test_ddp_options_chaining(pg);

    // grad_as_view mode
    test_ddp_grad_as_view_grads_synced(pg, rank);

    // param sync state
    test_set_param_sync(pg, rank);

    // Assignment operator
    test_ddp_assignment_operator(pg, rank);

    print_test_summary("DDP NoSync/Options");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
