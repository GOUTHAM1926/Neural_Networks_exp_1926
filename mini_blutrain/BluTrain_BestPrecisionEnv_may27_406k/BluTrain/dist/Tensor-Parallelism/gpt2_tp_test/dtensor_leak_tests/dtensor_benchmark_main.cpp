#include <iostream>
#include <mpi.h>
#include "dtensor_test_utils.h"
#include "communication/include/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"

// Forward declarations of test runners
void run_dtensor_layer_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);
void run_dtensor_activation_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);
void run_dtensor_loss_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);
void run_gpt2_component_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set GPU device based on rank
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int device_id = rank % num_devices;
    cudaSetDevice(device_id);
    
    if (rank == 0) {
        std::cout << "================================================================" << std::endl;
        std::cout << "  DTENSOR OPERATION BENCHMARK & MEMORY LEAK VERIFICATION" << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "World Size: " << world_size << " GPUs" << std::endl;
        std::cout << "Checking for memory leaks via Tensor Active Count..." << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Create ProcessGroup using init_process_group helper
    auto pg = init_process_group(world_size, rank);
    
    // Create DeviceMesh - requires (mesh_shape, device_ids)
    std::vector<int> mesh_shape = {world_size};  // 1D mesh with world_size devices
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % num_devices);
    }
    DeviceMesh mesh(mesh_shape, device_ids);
    
    int64_t initial_count = OwnTensor::Tensor::get_active_tensor_count();
    if (rank == 0) {
        std::cout << "Initial Active Tensors: " << initial_count << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Test Suites
    try {
        run_dtensor_activation_tests(mesh, pg);
        run_dtensor_loss_tests(mesh, pg);
        run_dtensor_layer_tests(mesh, pg);
        run_gpt2_component_tests(mesh, pg);
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] FATAL ERROR: " << e.what() << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n================================================================" << std::endl;
    }
    
    int64_t final_count = OwnTensor::Tensor::get_active_tensor_count();
    
    // Gather counts from all ranks
    int64_t max_final_count;
    MPI_Reduce(&final_count, &max_final_count, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Final Active Tensors (max across ranks): " << max_final_count << std::endl;
        std::cout << "Net Leak: " << (max_final_count - initial_count) << std::endl;
        
        if (max_final_count > initial_count) {
            std::cout << "WARNING: Memory leaks detected! (Or static tensors persisting)" << std::endl;
        } else {
            std::cout << "Memory Efficiency: OK (No dangling tensors)" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
