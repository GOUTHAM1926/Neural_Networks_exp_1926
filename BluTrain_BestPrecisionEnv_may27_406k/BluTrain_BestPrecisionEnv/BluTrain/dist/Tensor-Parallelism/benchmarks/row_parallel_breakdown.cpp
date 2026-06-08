#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <cuda_runtime.h>


std::vector<float> randomData(size_t size) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX;
    }
    return data;
}

class RowParallelBreakdown {
public:
    RowParallelBreakdown() {
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
     
            std::cout << "  Row-Parallel MatMul: Compute vs Communication Breakdown\n";
        
            std::cout << "Size    | Column (ms) | Row (ms) | Overhead (ms) | Comm % |\n";
            std::cout << "--------|-------------|----------|---------------|--------|\n";
        }
        
        std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
        
        for (int N : sizes) {
            benchmarkBoth(N);
        }
    }

private:
    int rank_;
    int world_size_;
    std::shared_ptr<DeviceMesh> device_mesh_;
    std::shared_ptr<ProcessGroup> pg_;
    
    const int NUM_ITERATIONS = 20;
    const int WARMUP_ITERATIONS = 5;

    void benchmarkBoth(int N) {
        int M = N, K = N;
        
        // Check memory
        size_t total_elements = (size_t)M*K + (size_t)K*N + (size_t)M*N;
        size_t mem_req = total_elements * sizeof(float);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        if (mem_req > free_mem * 0.8) {
            if (rank_ == 0) std::cout << std::setw(7) << N << " | Skipped (OOM)\n";
            return;
        }

        try {
          
            DTensor X_col(device_mesh_, pg_);
            Layout X_col_layout = Layout::replicated(device_mesh_, {M, K});
            X_col.setData(randomData(M * K), X_col_layout);

            DTensor W_col(device_mesh_, pg_);
            std::vector<std::shared_ptr<Placement>> W_col_placements = { std::make_shared<Shard>(1) };
            Layout W_col_layout(device_mesh_, {K, N}, W_col_placements);
            int N_local = N / world_size_;
            W_col.setData(randomData(K * N_local), W_col_layout);

   
            for(int i=0; i<WARMUP_ITERATIONS; ++i) {
                auto Y_col = X_col.matmul(W_col);
            }
            cudaDeviceSynchronize();
l
            auto col_start = std::chrono::high_resolution_clock::now();
            for(int i=0; i<NUM_ITERATIONS; ++i) {
                auto Y_col = X_col.matmul(W_col);
            }
            cudaDeviceSynchronize();
            auto col_end = std::chrono::high_resolution_clock::now();
            double col_ms = std::chrono::duration<double, std::milli>(col_end - col_start).count() / NUM_ITERATIONS;

            // Row-Parallel (Compute + AllReduce) 
            DTensor X_row(device_mesh_, pg_);
            std::vector<std::shared_ptr<Placement>> X_row_placements = { std::make_shared<Shard>(1) };
            Layout X_row_layout(device_mesh_, {M, K}, X_row_placements);
            int K_local = K / world_size_;
            X_row.setData(randomData(M * K_local), X_row_layout);

            DTensor W_row(device_mesh_, pg_);
            std::vector<std::shared_ptr<Placement>> W_row_placements = { std::make_shared<Shard>(0) };
            Layout W_row_layout(device_mesh_, {K, N}, W_row_placements);
            W_row.setData(randomData(K_local * N), W_row_layout);


            for(int i=0; i<WARMUP_ITERATIONS; ++i) {
                auto Y_row = X_row.matmul(W_row);
            }
            cudaDeviceSynchronize();

        
            auto row_start = std::chrono::high_resolution_clock::now();
            for(int i=0; i<NUM_ITERATIONS; ++i) {
                auto Y_row = X_row.matmul(W_row);
            }
            cudaDeviceSynchronize();
            auto row_end = std::chrono::high_resolution_clock::now();
            double row_ms = std::chrono::duration<double, std::milli>(row_end - row_start).count() / NUM_ITERATIONS;

            double overhead_ms = row_ms - col_ms;
            double comm_percent = (overhead_ms / row_ms) * 100.0;

            if (rank_ == 0) {
                std::cout << std::fixed << std::setprecision(3);
                std::cout << std::setw(7) << N << " | "
                          << std::setw(11) << col_ms << " | "
                          << std::setw(8) << row_ms << " | "
                          << std::setw(13) << overhead_ms << " | "
                          << std::setw(6) << std::setprecision(1) << comm_percent << "% |\n";
            }

        } catch (const std::exception& e) {
            if (rank_ == 0) std::cout << std::setw(7) << N << " | Failed (" << e.what() << ")\n";
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    {
        RowParallelBreakdown benchmark;
        benchmark.run();
    }
    MPI_Finalize();
    return 0;
}
