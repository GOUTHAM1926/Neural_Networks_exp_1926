/**
 * Benchmark LibTorch sharding speed for (batch, time, channels) tensors.
 * Uses raw NCCL for scatter operations (since c10d ProcessGroup is hard to use from C++).
 * 
 * Usage:
 *     mpirun -np 2 ./benchmarks/libtorch_sharding_benchmark
 */

#include <mpi.h>
#include <torch/torch.h>
#include <nccl.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvtx3/nvToolsExt.h>


#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        std::cerr << "NCCL error: " << ncclGetErrorString(r) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

void benchmark_sharding(int64_t b, int64_t t, int64_t c, int shard_dim = 0, int num_iterations = 100) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(rank));
    
    // Initialize NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Create tensor on GPU using LibTorch
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank);
    torch::Tensor tensor = torch::randn({b, t, c}, options);
    
    // Calculate shard shape
    std::vector<int64_t> shard_shape = {b, t, c};
    shard_shape[shard_dim] /= world_size;
    torch::Tensor shard_tensor = torch::empty(shard_shape, options);
    
    int64_t tensor_numel = b * t * c;
    int64_t shard_numel = tensor_numel / world_size;
    float tensor_size_mb = (float)(tensor_numel * sizeof(float)) / (1024.0f * 1024.0f);
    
    // Prepare transposed tensor for dim 2 sharding
    nvtxRangePush("transpose");
    torch::Tensor transposed;
    if (shard_dim == 2) {
        transposed = tensor.transpose(0, 2).contiguous();  // [c, t, b]
    } else if (shard_dim == 1) {
        transposed = tensor.transpose(0, 1).contiguous();  // [t, b, c]
    }
    nvtxRangePop();
    
    // Warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 10; i++) {
        NCCL_CHECK(ncclGroupStart());
        if (shard_dim == 0) {
            // Direct scatter on contiguous chunks
            NCCL_CHECK(ncclScatter(
                tensor.data_ptr<float>(),
                shard_tensor.data_ptr<float>(),
                shard_numel, ncclFloat, 0, nccl_comm, stream
            ));
        } else {
            // Scatter on transposed tensor then transpose back
            NCCL_CHECK(ncclScatter(
                transposed.data_ptr<float>(),
                shard_tensor.data_ptr<float>(),
                shard_numel, ncclFloat, 0, nccl_comm, stream
            ));
        }
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Benchmark
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; i++) {
        nvtxRangePush("nccl operation");
        NCCL_CHECK(ncclGroupStart());
        if (shard_dim == 0) {
            NCCL_CHECK(ncclScatter(
                tensor.data_ptr<float>(),
                shard_tensor.data_ptr<float>(),
                shard_numel, ncclFloat, 0, nccl_comm, stream
            ));
        } else {
            NCCL_CHECK(ncclScatter(
                transposed.data_ptr<float>(),
                shard_tensor.data_ptr<float>(),
                shard_numel, ncclFloat, 0, nccl_comm, stream
            ));
        }
        NCCL_CHECK(ncclGroupEnd());
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_ms = total_time_ms / num_iterations;
    double throughput = tensor_size_mb / (avg_time_ms / 1000.0);
    
    if (rank == 0) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "LibTorch + NCCL Sharding Benchmark" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Shape:       (" << b << ", " << t << ", " << c << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Size:        " << tensor_size_mb << " MB" << std::endl;
        std::cout << "Shard dim:   " << shard_dim << " (" 
                  << (shard_dim == 0 ? "batch" : (shard_dim == 1 ? "time" : "channel")) 
                  << ")" << std::endl;
        std::cout << "World size:  " << world_size << std::endl;
        std::cout << "Iterations:  " << num_iterations << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << std::setprecision(3);
        std::cout << "Avg time:    " << avg_time_ms << " ms" << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "Throughput:  " << throughput << " MB/s" << std::endl;
        std::cout << std::string(50, '=') << "\n" << std::endl;
    }
    
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    // Example: batch=8, time=998, channels=1996 (same as PyTorch benchmark)
    benchmark_sharding(8, 998, 1996, 2, 100);
    
    MPI_Finalize();
    return 0;
}
