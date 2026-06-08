#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * Broadcast tests with 2 ranks.
 * Root (rank 0) sends data; all ranks should receive the SAME data from root.
 */

void test_broadcast_basic(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{4}}, opts);
    // Only root's data matters; rank 1 sets different data to verify overwrite
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    else           send.set_data(std::vector<float>({99.0f, 99.0f, 99.0f, 99.0f}));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    result_t result = pg->broadcast(send.data(), recv.data(), send.numel(), send.dtype(), 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Broadcast basic returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    // Both ranks should see root's data: [1, 2, 3, 4]
    TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "Broadcast: [0] == 1.0 (root data)");
    TEST_ASSERT_NEAR(cpu_data[1], 2.0f, 1e-4, "Broadcast: [1] == 2.0 (root data)");
    TEST_ASSERT_NEAR(cpu_data[2], 3.0f, 1e-4, "Broadcast: [2] == 3.0 (root data)");
    TEST_ASSERT_NEAR(cpu_data[3], 4.0f, 1e-4, "Broadcast: [3] == 4.0 (root data)");
}

void test_broadcast_inplace(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({7.0f, 8.0f, 9.0f}));
    else           tensor.set_data(std::vector<float>({0.0f, 0.0f, 0.0f}));

    auto work = pg->broadcast_inplace(tensor, 0, true);
    pg->blockStream();

    TEST_ASSERT(work != nullptr, "Broadcast inplace returns valid Work object");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // Both ranks should see root's data
    TEST_ASSERT_NEAR(cpu_data[0], 7.0f, 1e-4, "Broadcast inplace: [0] == 7.0");
    TEST_ASSERT_NEAR(cpu_data[1], 8.0f, 1e-4, "Broadcast inplace: [1] == 8.0");
    TEST_ASSERT_NEAR(cpu_data[2], 9.0f, 1e-4, "Broadcast inplace: [2] == 9.0");
}

void test_broadcast_async(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{5}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({100.0f, 200.0f, 300.0f, 400.0f, 500.0f}));
    else           send.set_data(std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));

    auto recv = OwnTensor::Tensor::empty({{5}}, opts);

    auto work = pg->broadcast_async(send.data(), recv.data(), send.numel(), send.dtype(), 0, false);
    TEST_ASSERT(work != nullptr, "Broadcast async returns valid Work object");

    work->wait();
    pg->blockStream();

    auto cpu_data = read_tensor_to_cpu(recv);
    TEST_ASSERT_NEAR(cpu_data[0], 100.0f, 1e-4, "Broadcast async: [0] == 100.0");
    TEST_ASSERT_NEAR(cpu_data[4], 500.0f, 1e-4, "Broadcast async: [4] == 500.0");
}

void test_broadcast_coalesced(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto t1 = OwnTensor::Tensor({{2}}, opts);
    auto t2 = OwnTensor::Tensor({{3}}, opts);

    if (rank == 0) {
        t1.set_data(std::vector<float>({1.0f, 2.0f}));
        t2.set_data(std::vector<float>({3.0f, 4.0f, 5.0f}));
    } else {
        t1.set_data(std::vector<float>({0.0f, 0.0f}));
        t2.set_data(std::vector<float>({0.0f, 0.0f, 0.0f}));
    }

    std::vector<OwnTensor::Tensor> tensor_list = {t1, t2};

    size_t total = t1.numel() + t2.numel();
    auto output = OwnTensor::Tensor({{1, static_cast<int64_t>(total)}}, opts);

    result_t result = pg->broadcast_coalesced(tensor_list, output, 25, 0);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Broadcast coalesced returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(output);
    // Both ranks see root's concatenated data: [1, 2, 3, 4, 5]
    TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "Broadcast coalesced: [0] == 1.0");
    TEST_ASSERT_NEAR(cpu_data[1], 2.0f, 1e-4, "Broadcast coalesced: [1] == 2.0");
    TEST_ASSERT_NEAR(cpu_data[2], 3.0f, 1e-4, "Broadcast coalesced: [2] == 3.0");
    TEST_ASSERT_NEAR(cpu_data[3], 4.0f, 1e-4, "Broadcast coalesced: [3] == 4.0");
    TEST_ASSERT_NEAR(cpu_data[4], 5.0f, 1e-4, "Broadcast coalesced: [4] == 5.0");
}

void test_broadcast_large(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int N = 512 * 512;
    auto send = OwnTensor::Tensor::empty({{512, 512}}, opts);
    if (rank == 0) send.set_data(std::vector<float>(N, 42.0f));
    else           send.set_data(std::vector<float>(N, 0.0f));

    auto recv = OwnTensor::Tensor::empty({{512, 512}}, opts);

    result_t result = pg->broadcast(send.data(), recv.data(), send.numel(), send.dtype(), 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Broadcast large (512x512) returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    TEST_ASSERT_NEAR(cpu_data[0], 42.0f, 1e-4, "Broadcast large: first elem == 42.0");
    TEST_ASSERT_NEAR(cpu_data[N - 1], 42.0f, 1e-4, "Broadcast large: last elem == 42.0");
}

void test_broadcast_inplace_nonempty(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({11.0f, 22.0f, 33.0f, 44.0f}));
    else           tensor.set_data(std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f}));

    auto work = pg->broadcast_inplace(tensor, 0, true);
    pg->blockStream();

    TEST_ASSERT(work != nullptr, "Broadcast inplace non-empty returns valid Work");

    auto cpu_data = read_tensor_to_cpu(tensor);
    TEST_ASSERT_NEAR(cpu_data[0], 11.0f, 1e-4, "Broadcast inplace non-empty: [0] == 11.0");
    TEST_ASSERT_NEAR(cpu_data[3], 44.0f, 1e-4, "Broadcast inplace non-empty: [3] == 44.0");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_broadcast_basic(pg, rank);
    test_broadcast_inplace(pg, rank);
    test_broadcast_async(pg, rank);
    test_broadcast_coalesced(pg, rank);
    test_broadcast_large(pg, rank);
    test_broadcast_inplace_nonempty(pg, rank);

    print_test_summary("Broadcast");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
