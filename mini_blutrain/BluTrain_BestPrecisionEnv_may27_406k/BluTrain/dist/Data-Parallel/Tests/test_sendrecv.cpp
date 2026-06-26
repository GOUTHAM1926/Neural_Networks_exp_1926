#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * SendRecv (point-to-point) tests with 2 ranks.
 * Rank 0 sends to Rank 1 and receives from Rank 1 simultaneously.
 */

void test_sendrecv_cross_rank(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    else           send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    // Each rank sends to the other and receives from the other
    int peer = 1 - rank;
    result_t result = pg->sendrecv(send.data(), recv.data(), peer, peer, send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "SendRecv cross-rank returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // rank0 receives rank1's data
        TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "SendRecv rank0 recv: [0] == 10.0");
        TEST_ASSERT_NEAR(cpu_data[1], 20.0f, 1e-4, "SendRecv rank0 recv: [1] == 20.0");
        TEST_ASSERT_NEAR(cpu_data[2], 30.0f, 1e-4, "SendRecv rank0 recv: [2] == 30.0");
        TEST_ASSERT_NEAR(cpu_data[3], 40.0f, 1e-4, "SendRecv rank0 recv: [3] == 40.0");
    } else {
        // rank1 receives rank0's data
        TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "SendRecv rank1 recv: [0] == 1.0");
        TEST_ASSERT_NEAR(cpu_data[1], 2.0f, 1e-4, "SendRecv rank1 recv: [1] == 2.0");
        TEST_ASSERT_NEAR(cpu_data[2], 3.0f, 1e-4, "SendRecv rank1 recv: [2] == 3.0");
        TEST_ASSERT_NEAR(cpu_data[3], 4.0f, 1e-4, "SendRecv rank1 recv: [3] == 4.0");
    }
}

void test_sendrecv_async_cross_rank(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({5.0f, 6.0f, 7.0f}));
    else           send.set_data(std::vector<float>({50.0f, 60.0f, 70.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    int peer = 1 - rank;
    auto work = pg->sendrecv_async(send.data(), recv.data(), peer, peer, send.numel(), send.dtype(), false);
    TEST_ASSERT(work != nullptr, "SendRecv async returns valid Work object");

    work->wait();
    pg->blockStream();

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        TEST_ASSERT_NEAR(cpu_data[0], 50.0f, 1e-4, "SendRecv async rank0 recv: [0] == 50.0");
        TEST_ASSERT_NEAR(cpu_data[2], 70.0f, 1e-4, "SendRecv async rank0 recv: [2] == 70.0");
    } else {
        TEST_ASSERT_NEAR(cpu_data[0], 5.0f, 1e-4, "SendRecv async rank1 recv: [0] == 5.0");
        TEST_ASSERT_NEAR(cpu_data[2], 7.0f, 1e-4, "SendRecv async rank1 recv: [2] == 7.0");
    }
}

void test_sendrecv_2d_tensor(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{2, 3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    else           send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}));

    auto recv = OwnTensor::Tensor::empty({{2, 3}}, opts);

    int peer = 1 - rank;
    result_t result = pg->sendrecv(send.data(), recv.data(), peer, peer, send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "SendRecv 2D tensor returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "SendRecv 2D rank0: [0] == 10.0");
        TEST_ASSERT_NEAR(cpu_data[5], 60.0f, 1e-4, "SendRecv 2D rank0: [5] == 60.0");
    } else {
        TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "SendRecv 2D rank1: [0] == 1.0");
        TEST_ASSERT_NEAR(cpu_data[5], 6.0f, 1e-4, "SendRecv 2D rank1: [5] == 6.0");
    }
}

void test_sendrecv_large(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int N = 4096;
    float fill_val = (rank == 0) ? 1.0f : 2.0f;
    auto send = OwnTensor::Tensor::empty({{N}}, opts);
    send.set_data(std::vector<float>(N, fill_val));

    auto recv = OwnTensor::Tensor::empty({{N}}, opts);

    int peer = 1 - rank;
    result_t result = pg->sendrecv(send.data(), recv.data(), peer, peer, send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "SendRecv large (4096) returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    float expected = (rank == 0) ? 2.0f : 1.0f;
    TEST_ASSERT_NEAR(cpu_data[0], expected, 1e-4, "SendRecv large: first elem from peer");
    TEST_ASSERT_NEAR(cpu_data[N - 1], expected, 1e-4, "SendRecv large: last elem from peer");
}

void test_send_recv_ranks(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({11.0f, 22.0f, 33.0f}));
    else           send.set_data(std::vector<float>({44.0f, 55.0f, 66.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    int peer = 1 - rank;
    std::vector<int> ranks = {peer};
    result_t result = pg->send_recv_ranks(send.data(), recv.data(), peer, ranks, send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "send_recv_ranks returns pgSuccess");
}

void test_send_recv_ranks_multi_buffer(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f}));
    else           send.set_data(std::vector<float>({4.0f, 5.0f, 6.0f}));

    auto recv1 = OwnTensor::Tensor::empty({{3}}, opts);

    int peer = 1 - rank;
    std::vector<void*> recvbuffs = {recv1.data()};
    std::vector<int> ranks = {peer};

    result_t result = pg->send_recv_ranks(send.data(), recvbuffs, peer, ranks, send.numel(), send.dtype(), true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "send_recv_ranks multi-buffer returns pgSuccess");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_sendrecv_cross_rank(pg, rank);
    test_sendrecv_async_cross_rank(pg, rank);
    test_sendrecv_2d_tensor(pg, rank);
    test_sendrecv_large(pg, rank);
    test_send_recv_ranks(pg, rank);
    test_send_recv_ranks_multi_buffer(pg, rank);

    print_test_summary("SendRecv (Point-to-Point)");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
