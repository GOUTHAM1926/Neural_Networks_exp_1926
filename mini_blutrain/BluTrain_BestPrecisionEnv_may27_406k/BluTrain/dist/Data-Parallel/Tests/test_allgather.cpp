#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * AllGather tests with 2 ranks.
 * Each rank contributes its own chunk; after allgather both ranks
 * see the concatenation: [rank0_data | rank1_data].
 */

void test_allgather_basic(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f}));
    else           send.set_data(std::vector<float>({4.0f, 5.0f, 6.0f}));

    // recv must hold world_size * sendcount elements
    auto recv = OwnTensor::Tensor::empty({{1, 3 * world_size}}, opts);

    result_t result = pg->all_gather(send.data(), recv.data(), send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllGather basic returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    // Both ranks see: [1, 2, 3, 4, 5, 6]
    TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "AllGather: [0] == 1.0 (rank0 chunk)");
    TEST_ASSERT_NEAR(cpu_data[1], 2.0f, 1e-4, "AllGather: [1] == 2.0 (rank0 chunk)");
    TEST_ASSERT_NEAR(cpu_data[2], 3.0f, 1e-4, "AllGather: [2] == 3.0 (rank0 chunk)");
    TEST_ASSERT_NEAR(cpu_data[3], 4.0f, 1e-4, "AllGather: [3] == 4.0 (rank1 chunk)");
    TEST_ASSERT_NEAR(cpu_data[4], 5.0f, 1e-4, "AllGather: [4] == 5.0 (rank1 chunk)");
    TEST_ASSERT_NEAR(cpu_data[5], 6.0f, 1e-4, "AllGather: [5] == 6.0 (rank1 chunk)");
}

void test_allgather_single_element(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{1}}, opts);
    float val = (rank == 0) ? 42.0f : 99.0f;
    send.set_data(std::vector<float>({val}));

    auto recv = OwnTensor::Tensor::empty({{1, world_size}}, opts);

    result_t result = pg->all_gather(send.data(), recv.data(), send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllGather single element returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    TEST_ASSERT_NEAR(cpu_data[0], 42.0f, 1e-4, "AllGather single: [0] == 42.0 (rank0)");
    TEST_ASSERT_NEAR(cpu_data[1], 99.0f, 1e-4, "AllGather single: [1] == 99.0 (rank1)");
}

void test_allgather_async(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{2}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({7.0f, 8.0f}));
    else           send.set_data(std::vector<float>({9.0f, 10.0f}));

    auto recv = OwnTensor::Tensor::empty({{1, 2 * world_size}}, opts);

    auto work = pg->all_gather_async(send.data(), recv.data(), send.numel(), send.dtype(), false);
    TEST_ASSERT(work != nullptr, "AllGather async returns valid Work object");

    work->wait();
    pg->blockStream();

    auto cpu_data = read_tensor_to_cpu(recv);
    // [7, 8, 9, 10]
    TEST_ASSERT_NEAR(cpu_data[0],  7.0f, 1e-4, "AllGather async: [0] == 7.0");
    TEST_ASSERT_NEAR(cpu_data[1],  8.0f, 1e-4, "AllGather async: [1] == 8.0");
    TEST_ASSERT_NEAR(cpu_data[2],  9.0f, 1e-4, "AllGather async: [2] == 9.0");
    TEST_ASSERT_NEAR(cpu_data[3], 10.0f, 1e-4, "AllGather async: [3] == 10.0");
}

void test_allgather_large(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int send_count = 1024;
    float fill_val = (rank == 0) ? 1.0f : 2.0f;
    auto send = OwnTensor::Tensor::empty({{send_count}}, opts);
    send.set_data(std::vector<float>(send_count, fill_val));

    auto recv = OwnTensor::Tensor::empty({{1, send_count * world_size}}, opts);

    result_t result = pg->all_gather(send.data(), recv.data(), send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllGather large (1024) returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    // First half = 1.0 (rank0), second half = 2.0 (rank1)
    TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "AllGather large: rank0 chunk first elem == 1.0");
    TEST_ASSERT_NEAR(cpu_data[send_count], 2.0f, 1e-4, "AllGather large: rank1 chunk first elem == 2.0");
    TEST_ASSERT_EQ((int)cpu_data.size(), send_count * world_size, "AllGather large: total size correct");
}

void test_allgather_2d_tensor(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{2, 2}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}));
    else           send.set_data(std::vector<float>({50.0f, 60.0f, 70.0f, 80.0f}));

    auto recv = OwnTensor::Tensor::empty({{1, 4 * world_size}}, opts);

    result_t result = pg->all_gather(send.data(), recv.data(), send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllGather 2D returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    // [10, 20, 30, 40, 50, 60, 70, 80]
    TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "AllGather 2D: [0] == 10.0");
    TEST_ASSERT_NEAR(cpu_data[3], 40.0f, 1e-4, "AllGather 2D: [3] == 40.0");
    TEST_ASSERT_NEAR(cpu_data[4], 50.0f, 1e-4, "AllGather 2D: [4] == 50.0");
    TEST_ASSERT_NEAR(cpu_data[7], 80.0f, 1e-4, "AllGather 2D: [7] == 80.0");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_allgather_basic(pg, rank, world_size);
    test_allgather_single_element(pg, rank, world_size);
    test_allgather_async(pg, rank, world_size);
    test_allgather_large(pg, rank, world_size);
    test_allgather_2d_tensor(pg, rank, world_size);

    print_test_summary("AllGather");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
