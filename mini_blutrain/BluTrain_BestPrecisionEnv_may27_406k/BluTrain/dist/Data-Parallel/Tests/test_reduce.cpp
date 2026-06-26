#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * Reduce tests with 2 ranks.
 * Reduce puts the result only on the root rank (rank 0).
 * Non-root rank's recv buffer content is undefined by NCCL spec,
 * so we only verify on root.
 */

void test_reduce_sum(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f}));
    else           send.set_data(std::vector<float>({4.0f, 5.0f, 6.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    result_t result = pg->reduce(send.data(), recv.data(), send.numel(), send.dtype(), sum, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce SUM returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // SUM on root: [1+4, 2+5, 3+6] = [5, 7, 9]
        TEST_ASSERT_NEAR(cpu_data[0], 5.0f, 1e-4, "Reduce SUM root: [0] == 5.0");
        TEST_ASSERT_NEAR(cpu_data[1], 7.0f, 1e-4, "Reduce SUM root: [1] == 7.0");
        TEST_ASSERT_NEAR(cpu_data[2], 9.0f, 1e-4, "Reduce SUM root: [2] == 9.0");
    }
}

void test_reduce_avg(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f}));
    else           send.set_data(std::vector<float>({30.0f, 40.0f, 50.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    result_t result = pg->reduce(send.data(), recv.data(), send.numel(), send.dtype(), avg, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce AVG returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // AVG on root: [(10+30)/2, (20+40)/2, (30+50)/2] = [20, 30, 40]
        TEST_ASSERT_NEAR(cpu_data[0], 20.0f, 1e-4, "Reduce AVG root: [0] == 20.0");
        TEST_ASSERT_NEAR(cpu_data[1], 30.0f, 1e-4, "Reduce AVG root: [1] == 30.0");
        TEST_ASSERT_NEAR(cpu_data[2], 40.0f, 1e-4, "Reduce AVG root: [2] == 40.0");
    }
}

void test_reduce_max(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 100.0f, -5.0f, 0.0f}));
    else           send.set_data(std::vector<float>({50.0f, 2.0f,  10.0f, 0.0f}));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    result_t result = pg->reduce(send.data(), recv.data(), send.numel(), send.dtype(), max, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce MAX returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // MAX on root: [max(1,50), max(100,2), max(-5,10), max(0,0)] = [50, 100, 10, 0]
        TEST_ASSERT_NEAR(cpu_data[0],  50.0f, 1e-4, "Reduce MAX root: [0] == 50.0");
        TEST_ASSERT_NEAR(cpu_data[1], 100.0f, 1e-4, "Reduce MAX root: [1] == 100.0");
        TEST_ASSERT_NEAR(cpu_data[2],  10.0f, 1e-4, "Reduce MAX root: [2] == 10.0");
        TEST_ASSERT_NEAR(cpu_data[3],   0.0f, 1e-4, "Reduce MAX root: [3] == 0.0");
    }
}

void test_reduce_min(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 100.0f, -5.0f, 0.0f}));
    else           send.set_data(std::vector<float>({50.0f, 2.0f,  10.0f, 0.0f}));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    result_t result = pg->reduce(send.data(), recv.data(), send.numel(), send.dtype(), min, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce MIN returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // MIN on root: [min(1,50), min(100,2), min(-5,10), min(0,0)] = [1, 2, -5, 0]
        TEST_ASSERT_NEAR(cpu_data[0],  1.0f, 1e-4, "Reduce MIN root: [0] == 1.0");
        TEST_ASSERT_NEAR(cpu_data[1],  2.0f, 1e-4, "Reduce MIN root: [1] == 2.0");
        TEST_ASSERT_NEAR(cpu_data[2], -5.0f, 1e-4, "Reduce MIN root: [2] == -5.0");
        TEST_ASSERT_NEAR(cpu_data[3],  0.0f, 1e-4, "Reduce MIN root: [3] == 0.0");
    }
}

void test_reduce_mul(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({2.0f, 3.0f, 4.0f}));
    else           send.set_data(std::vector<float>({5.0f, 6.0f, 7.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    result_t result = pg->reduce(send.data(), recv.data(), send.numel(), send.dtype(), mul, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce MUL returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // MUL on root: [2*5, 3*6, 4*7] = [10, 18, 28]
        TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "Reduce MUL root: [0] == 10.0");
        TEST_ASSERT_NEAR(cpu_data[1], 18.0f, 1e-4, "Reduce MUL root: [1] == 18.0");
        TEST_ASSERT_NEAR(cpu_data[2], 28.0f, 1e-4, "Reduce MUL root: [2] == 28.0");
    }
}

void test_reduce_async(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({5.0f, 10.0f, 15.0f}));
    else           send.set_data(std::vector<float>({1.0f,  2.0f,  3.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    auto work = pg->reduce_async(send.data(), recv.data(), send.numel(), send.dtype(), sum, 0, false);
    TEST_ASSERT(work != nullptr, "Reduce async returns valid Work object");

    work->wait();
    pg->blockStream();

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(recv);
        // SUM: [5+1, 10+2, 15+3] = [6, 12, 18]
        TEST_ASSERT_NEAR(cpu_data[0],  6.0f, 1e-4, "Reduce async root: [0] == 6.0");
        TEST_ASSERT_NEAR(cpu_data[1], 12.0f, 1e-4, "Reduce async root: [1] == 12.0");
        TEST_ASSERT_NEAR(cpu_data[2], 18.0f, 1e-4, "Reduce async root: [2] == 18.0");
    }
}

void test_reduce_inplace(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({11.0f, 22.0f, 33.0f}));
    else           tensor.set_data(std::vector<float>({1.0f,  2.0f,  3.0f}));

    result_t result = pg->reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), sum, 0, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "Reduce in-place returns pgSuccess");

    if (rank == 0) {
        auto cpu_data = read_tensor_to_cpu(tensor);
        // SUM: [11+1, 22+2, 33+3] = [12, 24, 36]
        TEST_ASSERT_NEAR(cpu_data[0], 12.0f, 1e-4, "Reduce in-place root: [0] == 12.0");
        TEST_ASSERT_NEAR(cpu_data[1], 24.0f, 1e-4, "Reduce in-place root: [1] == 24.0");
        TEST_ASSERT_NEAR(cpu_data[2], 36.0f, 1e-4, "Reduce in-place root: [2] == 36.0");
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_reduce_sum(pg, rank);
    test_reduce_avg(pg, rank);
    test_reduce_max(pg, rank);
    test_reduce_min(pg, rank);
    test_reduce_mul(pg, rank);
    test_reduce_async(pg, rank);
    test_reduce_inplace(pg, rank);

    print_test_summary("Reduce");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
