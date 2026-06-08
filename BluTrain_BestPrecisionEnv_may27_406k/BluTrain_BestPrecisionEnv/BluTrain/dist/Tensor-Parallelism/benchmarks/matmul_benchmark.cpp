#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>


std::vector<float> randomData(size_t size) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX;
    }
    return data;
}

class MatmulBenchmark {
public:
    MatmulBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        
        
        cudaSetDevice(rank_);
        
        
        device_mesh_ = std::make_shared<DeviceMesh>(std::vector<int>{world_size_});
        
        
        ncclUniqueId nccl_id;
        if (rank_ == 0) ncclGetUniqueId(&nccl_id);
        MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        pg_ = std::make_shared<ProcessGroup>(rank_, world_size_, rank_, nccl_id);
    }
    
    void run() {
        if (rank_ == 0) {

            std::cout << "  Tensor Parallel MatMul Benchmark (" << world_size_ << " GPUs)\n";

        }
        
        // Matrix sizes (Square M=N=K)
        std::vector<int> sizes = {
            128, 256, 512, 1024, 2048, 4096, 8192
        };
        
        benchmarkColumnParallel(sizes);
        benchmarkRowParallel(sizes);
    }

private:
    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroup> pg_;
    
    const int NUM_ITERATIONS = 20;
    const int WARMUP_ITERATIONS = 5;


    // Column Parallel: X(Rep) @ W(Shard Col) -> Y(Shard Col)

    void benchmarkColumnParallel(const std::vector<int>& sizes) {
        if (rank_ == 0) std::cout << "Column-Parallel MatMul \n";
        
        for (int N : sizes) {
            int M = N, K = N; // Square matrix
            
            // Check memory (approx 3 matrices of size N*N)
            size_t total_elements = (size_t)M*K + (size_t)K*N + (size_t)M*N;
            size_t mem_req = total_elements * sizeof(float);
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            if (mem_req > free_mem * 0.8) {
                if (rank_ == 0) std::cout << "Size " << N << ": Skipped (OOM)\n";
                continue;
            }

            try {
                // Setup X (Replicated)
                DTensor X(device_mesh_, pg_);
                Layout X_layout = Layout::replicated(device_mesh_, {M, K});
                X.setData(randomData(M * K), X_layout);

                // Setup W (Sharded on Dim 1)
                DTensor W(device_mesh_, pg_);
                std::vector<std::shared_ptr<Placement>> W_placements = { std::make_shared<Shard>(1) };
                Layout W_layout(device_mesh_, {K, N}, W_placements);
                
                // Local W data size
                int N_local = N / world_size_;
                W.setData(randomData(K * N_local), W_layout);

                // Warmup
                for(int i=0; i<WARMUP_ITERATIONS; ++i) {
                    auto Y = X.matmul(W);
                }
                cudaDeviceSynchronize();

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                for(int i=0; i<NUM_ITERATIONS; ++i) {
                    auto Y = X.matmul(W);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
                double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;

                if (rank_ == 0) printResult(N, avg_ms, tflops);

            } catch (const std::exception& e) {
                if (rank_ == 0) std::cout << "Size " << N << ": Failed (" << e.what() << ")\n";
            }
        }
        if (rank_ == 0) std::cout << "\n";
    }

    // Row Parallel: X(Shard Col) @ W(Shard Row) -> Y(Rep)

    void benchmarkRowParallel(const std::vector<int>& sizes) {
        if (rank_ == 0) std::cout << " Row-Parallel MatMul (With AllReduce) \n";
        
        for (int N : sizes) {
            int M = N, K = N; // Square matrix

            // Check memory
            size_t total_elements = (size_t)M*K + (size_t)K*N + (size_t)M*N;
            size_t mem_req = total_elements * sizeof(float);
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            if (mem_req > free_mem * 0.8) {
                if (rank_ == 0) std::cout << "Size " << N << ": Skipped (OOM)\n";
                continue;
            }

            try {
                // X (Sharded on Dim 1 (Columns))
                DTensor X(device_mesh_, pg_);
                std::vector<std::shared_ptr<Placement>> X_placements = { std::make_shared<Shard>(1) };
                Layout X_layout(device_mesh_, {M, K}, X_placements);
                int K_local = K / world_size_;
                X.setData(randomData(M * K_local), X_layout);

                //  W ( Sharded on Dim 0 (Rows) )
                DTensor W(device_mesh_, pg_);
                std::vector<std::shared_ptr<Placement>> W_placements = { std::make_shared<Shard>(0) };
                Layout W_layout(device_mesh_, {K, N}, W_placements);
                // Local W size is [K_local, N]
                W.setData(randomData(K_local * N), W_layout);

                // Warmup
                for(int i=0; i<WARMUP_ITERATIONS; ++i) {
                    auto Y = X.matmul(W);
                }
                cudaDeviceSynchronize();

                // Benchmark
                auto start = std::chrono::high_resolution_clock::now();
                for(int i=0; i<NUM_ITERATIONS; ++i) {
                    auto Y = X.matmul(W);
                }
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();

                double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / NUM_ITERATIONS;
                double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;

                if (rank_ == 0) printResult(N, avg_ms, tflops);

            } catch (const std::exception& e) {
                if (rank_ == 0) std::cout << "Size " << N << ": Failed (" << e.what() << ")\n";
            }
        }
        if (rank_ == 0) std::cout << "\n";
    }

    void printResult(int N, double ms, double tflops) {
        std::cout << "Size: " << std::setw(5) << N << "x" << N 
                  << " | Latency: " << std::setw(8) << std::fixed << std::setprecision(3) << ms << " ms"
                  << " | TFLOPS: " << std::setw(6) << std::setprecision(3) << tflops << "\n";
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    {
        MatmulBenchmark benchmark;
        benchmark.run();
    }
    MPI_Finalize();
    return 0;
}
