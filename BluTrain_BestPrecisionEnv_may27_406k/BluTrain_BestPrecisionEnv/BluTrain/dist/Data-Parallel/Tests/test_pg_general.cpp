#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
#include <string>
#include "TensorLib.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../communication/include/GroupRegister.hpp"
#include "test_utils.hpp"

/*
 * General ProcessGroup tests: init, rank/worldsize, Work object,
 * streams, GroupRegister, error strings, type conversions, timing.
 * All run with 2 ranks via mpirun.
 */


// ---- init_process_group tests ----

void test_init_process_group_default(int rank, int world_size) {
    auto pg = init_process_group(world_size, rank);
    TEST_ASSERT(pg != nullptr, "init_process_group returns non-null shared_ptr");
}

void test_init_with_custom_name(int rank, int world_size) {
    std::string name = "custom_group_rank" + std::to_string(rank);
    auto pg = init_process_group(world_size, rank, 0, name);
    TEST_ASSERT(pg != nullptr, "init_process_group with custom name returns non-null");
}


// ---- Rank / WorldSize / LocalRank tests ----

void test_get_rank(std::shared_ptr<ProcessGroupNCCL> pg, int expected_rank) {
    TEST_ASSERT_EQ(pg->get_rank(), expected_rank, "get_rank() returns correct rank");
}

void test_get_worldsize(std::shared_ptr<ProcessGroupNCCL> pg, int expected_ws) {
    TEST_ASSERT_EQ(pg->get_worldsize(), expected_ws, "get_worldsize() returns correct world_size");
}

void test_get_local_rank(std::shared_ptr<ProcessGroupNCCL> pg, int expected_local) {
    TEST_ASSERT_EQ(pg->get_local_rank(), expected_local, "get_local_rank() returns correct local_rank");
}

void test_ranks_are_unique(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    // Each rank should have a different rank value — verify via allreduce of rank IDs
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{1}}, opts);
    tensor.set_data(std::vector<float>({(float)rank}));

    pg->all_reduce(tensor.data(), tensor.data(), 1, tensor.dtype(), sum, true);
    pg->blockStream();

    auto cpu = read_tensor_to_cpu(tensor);
    // With 2 ranks: sum of rank IDs = 0+1 = 1
    TEST_ASSERT_NEAR(cpu[0], 1.0f, 1e-4, "Sum of all rank IDs == 1 (ranks 0 and 1 are unique)");
}


// ---- Work object tests ----

void test_work_object_lifecycle(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    float fill = (rank == 0) ? 1.0f : 2.0f;
    tensor.set_data(std::vector<float>(4, fill));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    auto work = pg->all_reduce_async(tensor.data(), recv.data(), tensor.numel(), tensor.dtype(), sum, false);

    TEST_ASSERT(work != nullptr, "Work object is non-null after async launch");

    bool wait_result = work->wait();
    pg->blockStream();

    TEST_ASSERT(wait_result, "Work::wait() returns true on success");
    TEST_ASSERT(work->is_completed(), "Work::is_completed() is true after wait");
    TEST_ASSERT(work->is_success(), "Work::is_success() is true after successful op");

    auto cpu = read_tensor_to_cpu(recv);
    TEST_ASSERT_NEAR(cpu[0], 3.0f, 1e-4, "Work lifecycle: allreduce result == 3.0");
}

void test_work_query(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    tensor.set_data(std::vector<float>(4, 1.0f));

    auto recv = OwnTensor::Tensor::empty({{4}}, opts);

    auto work = pg->all_reduce_async(tensor.data(), recv.data(), tensor.numel(), tensor.dtype(), sum, false);

    work->wait();
    pg->blockStream();

    bool query_result = work->query();
    TEST_ASSERT(query_result, "Work::query() returns true after completed op");
}

void test_get_work_obj(std::shared_ptr<ProcessGroupNCCL> pg) {
    auto work = pg->get_work_obj();
    TEST_ASSERT(work != nullptr, "get_work_obj() returns non-null");
}


// ---- Stream tests ----

void test_owns_stream(int rank, int world_size) {
    auto pg = init_process_group(world_size, rank, 0, "stream_test_" + std::to_string(rank));
    TEST_ASSERT(pg->is_owns_stream(), "ProcessGroup owns stream when none provided");
}

void test_get_stream(std::shared_ptr<ProcessGroupNCCL> pg) {
    cudaStream_t stream = pg->get_stream();
    TEST_ASSERT(stream != nullptr, "get_stream() returns non-null stream");
}


// ---- blockStream / blockStreamEvent tests ----

void test_block_stream(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor({{4}}, opts);
    tensor.set_data(std::vector<float>(4, 1.0f));

    pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), sum, false);

    bool result = pg->blockStream();
    TEST_ASSERT(result, "blockStream() returns true");
}


// ---- GroupRegister tests ----

void test_global_counter_increments() {
    int initial = global_counter;
    auto name = get_group_name();
    TEST_ASSERT_EQ(global_counter, initial + 1, "global_counter increments after get_group_name()");
    TEST_ASSERT_EQ(name, std::to_string(initial), "get_group_name() returns counter as string");
}

void test_group_register_and_resolve(int rank, int world_size) {
    std::string gname = "resolve_test_" + std::to_string(rank);
    auto pg = init_process_group(world_size, rank, 0, gname);
    TEST_ASSERT(pg != nullptr, "PG for register/resolve test is non-null");
    TEST_ASSERT(process_registry != nullptr, "process_registry is non-null after init");
}

void test_group_register_duplicate(int rank, int world_size) {
    std::string gname = "dup_test_" + std::to_string(rank);
    auto pg = init_process_group(world_size, rank, 0, gname);

    TEST_ASSERT_THROWS(
        register_group(gname, pg),
        "Registering duplicate group name throws exception"
    );
}

void test_unregister_group(int rank, int world_size) {
    std::string gname = "unreg_test_" + std::to_string(rank);
    auto pg = init_process_group(world_size, rank, 0, gname);

    unregister_group(gname);
    TEST_ASSERT(true, "unregister_group does not throw for valid group");
}


// ---- pgGetError string tests ----

void test_pg_error_strings() {
    TEST_ASSERT_EQ(pgGetError(pgSuccess), std::string("Success"), "pgGetError(pgSuccess) == 'Success'");
    TEST_ASSERT_EQ(pgGetError(pgTimeout), std::string("Process Group timeout"), "pgGetError(pgTimeout) correct");
    TEST_ASSERT_EQ(pgGetError(pgCudaError), std::string("Cuda Error"), "pgGetError(pgCudaError) correct");
    TEST_ASSERT_EQ(pgGetError(pgNcclError), std::string("NCCL Error"), "pgGetError(pgNcclError) correct");
    TEST_ASSERT_EQ(pgGetError(pgCommunicationError), std::string("Internal Communication Error"),
                   "pgGetError(pgCommunicationError) correct");
}


// ---- NCCL type / operation conversion tests ----

void test_nccl_type_conversion() {
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Float32), ncclFloat32, "Float32 -> ncclFloat32");
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Float64), ncclFloat64, "Float64 -> ncclFloat64");
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Float16), ncclFloat16, "Float16 -> ncclFloat16");
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Bfloat16), ncclBfloat16, "Bfloat16 -> ncclBfloat16");
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Int32), ncclInt32, "Int32 -> ncclInt32");
    TEST_ASSERT_EQ(ncclTypeConversion(OwnTensor::Dtype::Int64), ncclInt64, "Int64 -> ncclInt64");
}

void test_nccl_operation_conversion() {
    TEST_ASSERT_EQ(ncclOperationConversion(sum), ncclSum, "sum -> ncclSum");
    TEST_ASSERT_EQ(ncclOperationConversion(avg), ncclAvg, "avg -> ncclAvg");
    TEST_ASSERT_EQ(ncclOperationConversion(max), ncclMax, "max -> ncclMax");
    TEST_ASSERT_EQ(ncclOperationConversion(min), ncclMin, "min -> ncclMin");
    TEST_ASSERT_EQ(ncclOperationConversion(mul), ncclProd, "mul -> ncclProd");
}


// ---- Timing tests ----

void test_start_end_time(std::shared_ptr<ProcessGroupNCCL> pg, int rank) {
    OwnTensor::TensorOptions opts;
    opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);

    auto tensor = OwnTensor::Tensor::empty({{256}}, opts);
    tensor.set_data(std::vector<float>(256, 1.0f));

    pg->start_time();
    pg->all_reduce(tensor.data(), tensor.data(), tensor.numel(), tensor.dtype(), sum, true);
    pg->blockStream();

    float ms = 0.0f;
    pg->end_time(ms);

    TEST_ASSERT(ms >= 0.0f, "Elapsed time is non-negative");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    set_test_rank(rank);

    // Init tests
    test_init_process_group_default(rank, world_size);
    test_init_with_custom_name(rank, world_size);

    auto pg = init_process_group(world_size, rank, 0, "main_pg_" + std::to_string(rank));

    // Rank/size tests
    test_get_rank(pg, rank);
    test_get_worldsize(pg, world_size);
    test_get_local_rank(pg, rank);
    test_ranks_are_unique(pg, rank);

    // Work object tests
    test_work_object_lifecycle(pg, rank);
    test_work_query(pg, rank);
    test_get_work_obj(pg);

    // Stream tests
    test_owns_stream(rank, world_size);
    test_get_stream(pg);

    // Block tests
    test_block_stream(pg, rank);

    // GroupRegister tests
    test_global_counter_increments();
    test_group_register_and_resolve(rank, world_size);
    test_group_register_duplicate(rank, world_size);
    test_unregister_group(rank, world_size);

    // Error string tests
    test_pg_error_strings();

    // Type conversion tests
    test_nccl_type_conversion();
    test_nccl_operation_conversion();

    // Timing tests
    test_start_end_time(pg, rank);

    print_test_summary("ProcessGroup General");

    int exit_code = test_exit_code();
    MPI_Finalize();
    return exit_code;
}
