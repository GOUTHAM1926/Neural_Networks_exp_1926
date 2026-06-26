#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * ReduceScatter tests with 2 ranks.
 * Each rank contributes world_size*recv_count elements.
 * After reduce_scatter, rank i gets the i-th chunk of the reduced result.
 *
 * With 2 ranks and recv_count=2:
 *   rank0 sends [a0, a1, a2, a3], rank1 sends [b0, b1, b2, b3]
 *   Reduce across ranks: [a0+b0, a1+b1, a2+b2, a3+b3]
 *   rank0 receives chunk0: [a0+b0, a1+b1]
 *   rank1 receives chunk1: [a2+b2, a3+b3]
 */

void test_reduce_scatter_sum(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    else           send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter SUM returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0 of SUM: [1+10, 2+20] = [11, 22]
        TEST_ASSERT_NEAR(cpu_data[0], 11.0f, 1e-4, "ReduceScatter SUM rank0: [0] == 11.0");
        TEST_ASSERT_NEAR(cpu_data[1], 22.0f, 1e-4, "ReduceScatter SUM rank0: [1] == 22.0");
    } else {
        // chunk1 of SUM: [3+30, 4+40] = [33, 44]
        TEST_ASSERT_NEAR(cpu_data[0], 33.0f, 1e-4, "ReduceScatter SUM rank1: [0] == 33.0");
        TEST_ASSERT_NEAR(cpu_data[1], 44.0f, 1e-4, "ReduceScatter SUM rank1: [1] == 44.0");
    }
}

void test_reduce_scatter_avg(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}));
    else           send.set_data(std::vector<float>({20.0f, 40.0f, 60.0f, 80.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), avg, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter AVG returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0 of AVG: [(10+20)/2, (20+40)/2] = [15, 30]
        TEST_ASSERT_NEAR(cpu_data[0], 15.0f, 1e-4, "ReduceScatter AVG rank0: [0] == 15.0");
        TEST_ASSERT_NEAR(cpu_data[1], 30.0f, 1e-4, "ReduceScatter AVG rank0: [1] == 30.0");
    } else {
        // chunk1 of AVG: [(30+60)/2, (40+80)/2] = [45, 60]
        TEST_ASSERT_NEAR(cpu_data[0], 45.0f, 1e-4, "ReduceScatter AVG rank1: [0] == 45.0");
        TEST_ASSERT_NEAR(cpu_data[1], 60.0f, 1e-4, "ReduceScatter AVG rank1: [1] == 60.0");
    }
}

void test_reduce_scatter_max(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 100.0f, -5.0f, 50.0f}));
    else           send.set_data(std::vector<float>({99.0f, 2.0f,   10.0f, 3.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), max, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter MAX returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0 of MAX: [max(1,99), max(100,2)] = [99, 100]
        TEST_ASSERT_NEAR(cpu_data[0],  99.0f, 1e-4, "ReduceScatter MAX rank0: [0] == 99.0");
        TEST_ASSERT_NEAR(cpu_data[1], 100.0f, 1e-4, "ReduceScatter MAX rank0: [1] == 100.0");
    } else {
        // chunk1 of MAX: [max(-5,10), max(50,3)] = [10, 50]
        TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "ReduceScatter MAX rank1: [0] == 10.0");
        TEST_ASSERT_NEAR(cpu_data[1], 50.0f, 1e-4, "ReduceScatter MAX rank1: [1] == 50.0");
    }
}

void test_reduce_scatter_min(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({1.0f, 100.0f, -5.0f, 50.0f}));
    else           send.set_data(std::vector<float>({99.0f, 2.0f,   10.0f, 3.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), min, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter MIN returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0 of MIN: [min(1,99), min(100,2)] = [1, 2]
        TEST_ASSERT_NEAR(cpu_data[0], 1.0f, 1e-4, "ReduceScatter MIN rank0: [0] == 1.0");
        TEST_ASSERT_NEAR(cpu_data[1], 2.0f, 1e-4, "ReduceScatter MIN rank0: [1] == 2.0");
    } else {
        // chunk1 of MIN: [min(-5,10), min(50,3)] = [-5, 3]
        TEST_ASSERT_NEAR(cpu_data[0], -5.0f, 1e-4, "ReduceScatter MIN rank1: [0] == -5.0");
        TEST_ASSERT_NEAR(cpu_data[1],  3.0f, 1e-4, "ReduceScatter MIN rank1: [1] == 3.0");
    }
}

void test_reduce_scatter_mul(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({2.0f, 3.0f, 4.0f, 5.0f}));
    else           send.set_data(std::vector<float>({6.0f, 7.0f, 8.0f, 9.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), mul, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter MUL returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0 of MUL: [2*6, 3*7] = [12, 21]
        TEST_ASSERT_NEAR(cpu_data[0], 12.0f, 1e-4, "ReduceScatter MUL rank0: [0] == 12.0");
        TEST_ASSERT_NEAR(cpu_data[1], 21.0f, 1e-4, "ReduceScatter MUL rank0: [1] == 21.0");
    } else {
        // chunk1 of MUL: [4*8, 5*9] = [32, 45]
        TEST_ASSERT_NEAR(cpu_data[0], 32.0f, 1e-4, "ReduceScatter MUL rank1: [0] == 32.0");
        TEST_ASSERT_NEAR(cpu_data[1], 45.0f, 1e-4, "ReduceScatter MUL rank1: [1] == 45.0");
    }
}

void test_reduce_scatter_async(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2;
    auto send = OwnTensor::Tensor({{recv_count * world_size}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({5.0f, 10.0f, 15.0f, 20.0f}));
    else           send.set_data(std::vector<float>({1.0f,  2.0f,  3.0f,  4.0f}));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    auto work = pg->reduce_scatter_async(send.data(), recv.data(), recv_count, send.dtype(), sum, false);
    TEST_ASSERT(work != nullptr, "ReduceScatter async returns valid Work object");

    work->wait();
    pg->blockStream();

    auto cpu_data = read_tensor_to_cpu(recv);
    if (rank == 0) {
        // chunk0: [5+1, 10+2] = [6, 12]
        TEST_ASSERT_NEAR(cpu_data[0],  6.0f, 1e-4, "ReduceScatter async rank0: [0] == 6.0");
        TEST_ASSERT_NEAR(cpu_data[1], 12.0f, 1e-4, "ReduceScatter async rank0: [1] == 12.0");
    } else {
        // chunk1: [15+3, 20+4] = [18, 24]
        TEST_ASSERT_NEAR(cpu_data[0], 18.0f, 1e-4, "ReduceScatter async rank1: [0] == 18.0");
        TEST_ASSERT_NEAR(cpu_data[1], 24.0f, 1e-4, "ReduceScatter async rank1: [1] == 24.0");
    }
}

void test_reduce_scatter_large(std::shared_ptr<ProcessGroupNCCL> pg, int rank, int world_size) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    int recv_count = 2048;
    float fill_val = (rank == 0) ? 1.0f : 3.0f;
    auto send = OwnTensor::Tensor::empty({{recv_count * world_size}}, opts);
    send.set_data(std::vector<float>(recv_count * world_size, fill_val));

    auto recv = OwnTensor::Tensor::empty({{recv_count}}, opts);

    result_t result = pg->reduce_scatter(send.data(), recv.data(), recv_count, send.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "ReduceScatter large (2048) returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(recv);
    // SUM of 1.0+3.0 = 4.0 for every element
    TEST_ASSERT_NEAR(cpu_data[0], 4.0f, 1e-4, "ReduceScatter large: first element == 4.0");
    TEST_ASSERT_NEAR(cpu_data[recv_count - 1], 4.0f, 1e-4, "ReduceScatter large: last element == 4.0");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_reduce_scatter_sum(pg, rank, world_size);
    test_reduce_scatter_avg(pg, rank, world_size);
    test_reduce_scatter_max(pg, rank, world_size);
    test_reduce_scatter_min(pg, rank, world_size);
    test_reduce_scatter_mul(pg, rank, world_size);
    test_reduce_scatter_async(pg, rank, world_size);
    test_reduce_scatter_large(pg, rank, world_size);

    print_test_summary("ReduceScatter");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
