#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include <cmath>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * AllReduce tests with 2 ranks.
 * Rank 0 and Rank 1 each contribute different data.
 * After allreduce, BOTH ranks should see the same result.
 */

void test_allreduce_sum(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({1.0f, 2.0f, 3.0f}));
    else           tensor.set_data(std::vector<float>({4.0f, 5.0f, 6.0f}));

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce SUM returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // SUM: [1+4, 2+5, 3+6] = [5, 7, 9]
    TEST_ASSERT_NEAR(cpu_data[0], 5.0f, 1e-4, "AllReduce SUM: [0] == 5.0");
    TEST_ASSERT_NEAR(cpu_data[1], 7.0f, 1e-4, "AllReduce SUM: [1] == 7.0");
    TEST_ASSERT_NEAR(cpu_data[2], 9.0f, 1e-4, "AllReduce SUM: [2] == 9.0");
}

void test_allreduce_avg(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}));
    else           tensor.set_data(std::vector<float>({20.0f, 40.0f, 60.0f, 80.0f}));

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), avg, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce AVG returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // AVG: [(10+20)/2, (20+40)/2, (30+60)/2, (40+80)/2] = [15, 30, 45, 60]
    TEST_ASSERT_NEAR(cpu_data[0], 15.0f, 1e-4, "AllReduce AVG: [0] == 15.0");
    TEST_ASSERT_NEAR(cpu_data[1], 30.0f, 1e-4, "AllReduce AVG: [1] == 30.0");
    TEST_ASSERT_NEAR(cpu_data[2], 45.0f, 1e-4, "AllReduce AVG: [2] == 45.0");
    TEST_ASSERT_NEAR(cpu_data[3], 60.0f, 1e-4, "AllReduce AVG: [3] == 60.0");
}

void test_allreduce_max(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({1.0f, -5.0f, 100.0f, 0.0f}));
    else           tensor.set_data(std::vector<float>({2.0f,  3.0f,  -1.0f, 0.0f}));

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), max, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce MAX returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // MAX: [max(1,2), max(-5,3), max(100,-1), max(0,0)] = [2, 3, 100, 0]
    TEST_ASSERT_NEAR(cpu_data[0],   2.0f, 1e-4, "AllReduce MAX: [0] == 2.0");
    TEST_ASSERT_NEAR(cpu_data[1],   3.0f, 1e-4, "AllReduce MAX: [1] == 3.0");
    TEST_ASSERT_NEAR(cpu_data[2], 100.0f, 1e-4, "AllReduce MAX: [2] == 100.0");
    TEST_ASSERT_NEAR(cpu_data[3],   0.0f, 1e-4, "AllReduce MAX: [3] == 0.0");
}

void test_allreduce_min(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({1.0f, -5.0f, 100.0f, 0.0f}));
    else           tensor.set_data(std::vector<float>({2.0f,  3.0f,  -1.0f, 0.0f}));

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), min, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce MIN returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // MIN: [min(1,2), min(-5,3), min(100,-1), min(0,0)] = [1, -5, -1, 0]
    TEST_ASSERT_NEAR(cpu_data[0],  1.0f, 1e-4, "AllReduce MIN: [0] == 1.0");
    TEST_ASSERT_NEAR(cpu_data[1], -5.0f, 1e-4, "AllReduce MIN: [1] == -5.0");
    TEST_ASSERT_NEAR(cpu_data[2], -1.0f, 1e-4, "AllReduce MIN: [2] == -1.0");
    TEST_ASSERT_NEAR(cpu_data[3],  0.0f, 1e-4, "AllReduce MIN: [3] == 0.0");
}

void test_allreduce_mul(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({2.0f, 3.0f, 4.0f}));
    else           tensor.set_data(std::vector<float>({5.0f, 6.0f, 7.0f}));

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), mul, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce MUL returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // MUL (product): [2*5, 3*6, 4*7] = [10, 18, 28]
    TEST_ASSERT_NEAR(cpu_data[0], 10.0f, 1e-4, "AllReduce MUL: [0] == 10.0");
    TEST_ASSERT_NEAR(cpu_data[1], 18.0f, 1e-4, "AllReduce MUL: [1] == 18.0");
    TEST_ASSERT_NEAR(cpu_data[2], 28.0f, 1e-4, "AllReduce MUL: [2] == 28.0");
}

void test_allreduce_out_of_place(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto send = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) send.set_data(std::vector<float>({10.0f, 20.0f, 30.0f}));
    else           send.set_data(std::vector<float>({1.0f,  2.0f,  3.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    result_t result = pg->all_reduce(send.data(), recv.data(), send.numel(), send.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce out-of-place returns pgSuccess");

    auto cpu_recv = read_tensor_to_cpu(recv);
    // SUM: [10+1, 20+2, 30+3] = [11, 22, 33]
    TEST_ASSERT_NEAR(cpu_recv[0], 11.0f, 1e-4, "AllReduce out-of-place: [0] == 11.0");
    TEST_ASSERT_NEAR(cpu_recv[1], 22.0f, 1e-4, "AllReduce out-of-place: [1] == 22.0");
    TEST_ASSERT_NEAR(cpu_recv[2], 33.0f, 1e-4, "AllReduce out-of-place: [2] == 33.0");

    // Verify send buffer is untouched
    auto cpu_send = read_tensor_to_cpu(send);
    float expected_first = (rank == 0) ? 10.0f : 1.0f;
    TEST_ASSERT_NEAR(cpu_send[0], expected_first, 1e-4, "AllReduce out-of-place: send buffer untouched");
}

void test_allreduce_async(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{3}}, opts);
    if (rank == 0) tensor.set_data(std::vector<float>({5.0f, 10.0f, 15.0f}));
    else           tensor.set_data(std::vector<float>({1.0f,  2.0f,  3.0f}));

    auto recv = OwnTensor::Tensor::empty({{3}}, opts);

    auto work = pg->all_reduce_async(tensor.data(), recv.data(), tensor.numel(), tensor.dtype(), sum, false);
    TEST_ASSERT(work != nullptr, "AllReduce async returns valid Work object");

    work->wait();
    pg->blockStream();

    auto cpu_data = read_tensor_to_cpu(recv);
    // SUM: [5+1, 10+2, 15+3] = [6, 12, 18]
    TEST_ASSERT_NEAR(cpu_data[0],  6.0f, 1e-4, "AllReduce async SUM: [0] == 6.0");
    TEST_ASSERT_NEAR(cpu_data[1], 12.0f, 1e-4, "AllReduce async SUM: [1] == 12.0");
    TEST_ASSERT_NEAR(cpu_data[2], 18.0f, 1e-4, "AllReduce async SUM: [2] == 18.0");
}

void test_allreduce_flatten_concat(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto t1 = OwnTensor::Tensor({{2}}, opts);
    auto t2 = OwnTensor::Tensor({{2}}, opts);

    if (rank == 0) {
        t1.set_data(std::vector<float>({1.0f, 2.0f}));
        t2.set_data(std::vector<float>({3.0f, 4.0f}));
    } else {
        t1.set_data(std::vector<float>({10.0f, 20.0f}));
        t2.set_data(std::vector<float>({30.0f, 40.0f}));
    }

    auto flattened = OwnTensor::Tensor::flatten_concat({t1, t2});

    result_t result = pg->all_reduce(flattened.data(), flattened.data(), flattened.numel(), flattened.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce flatten_concat returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(flattened);
    // SUM: [1+10, 2+20, 3+30, 4+40] = [11, 22, 33, 44]
    TEST_ASSERT_NEAR(cpu_data[0], 11.0f, 1e-4, "AllReduce flatten_concat: [0] == 11.0");
    TEST_ASSERT_NEAR(cpu_data[1], 22.0f, 1e-4, "AllReduce flatten_concat: [1] == 22.0");
    TEST_ASSERT_NEAR(cpu_data[2], 33.0f, 1e-4, "AllReduce flatten_concat: [2] == 33.0");
    TEST_ASSERT_NEAR(cpu_data[3], 44.0f, 1e-4, "AllReduce flatten_concat: [3] == 44.0");
}

void test_allreduce_large_tensor(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    // Both ranks fill with rank-specific constant value
    int N = 256 * 256;
    auto tensor = OwnTensor::Tensor::empty({{256, 256}}, opts);
    float fill_val = (rank == 0) ? 1.0f : 2.0f;
    std::vector<float> fill_data(N, fill_val);
    tensor.set_data(fill_data);

    result_t result = pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), sum, true);
    pg->blockStream();

    TEST_ASSERT_EQ(result, pgSuccess, "AllReduce large tensor returns pgSuccess");

    auto cpu_data = read_tensor_to_cpu(tensor);
    // SUM: every element should be 1+2=3
    TEST_ASSERT_NEAR(cpu_data[0], 3.0f, 1e-4, "AllReduce large: first element == 3.0");
    TEST_ASSERT_NEAR(cpu_data[N - 1], 3.0f, 1e-4, "AllReduce large: last element == 3.0");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    auto pg = init_process_group(world_size, rank);

    test_allreduce_sum(pg, rank);
    test_allreduce_avg(pg, rank);
    test_allreduce_max(pg, rank);
    test_allreduce_min(pg, rank);
    test_allreduce_mul(pg, rank);
    test_allreduce_out_of_place(pg, rank);
    test_allreduce_async(pg, rank);
    test_allreduce_flatten_concat(pg, rank);
    test_allreduce_large_tensor(pg, rank);

    print_test_summary("AllReduce");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
