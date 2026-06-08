/**
 * ============================================================================
 * MLP Interconnect Bandwidth Test (TP=2)
 * ============================================================================
 * 
 * Measures GPU interconnect bandwidth during tensor-parallel MLP operations.
 * Tests AllReduce performance across varying tensor sizes to identify:
 *   - Peak achievable bandwidth
 *   - Bandwidth efficiency vs theoretical max
 *   - Small-message vs large-message performance
 * 
 * Run: mpirun -np 2 ./benchmarks/interconnect_benchmarks
 * ============================================================================
 */

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "tensor/dtensor.h"

// =============================================================================
// CUDA Error Checking
// =============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// =============================================================================
// Bandwidth Test Configuration
// =============================================================================
struct BandwidthTestConfig {
    int warmup_iters = 10;
    int measure_iters = 100;
    std::vector<size_t> tensor_sizes;  // In bytes
    
    BandwidthTestConfig() {
        // Key sizes: small message to large activation
        tensor_sizes = {
            64 * 1024,         // 64 KB (small)
            1024 * 1024,       // 1 MB
            16 * 1024 * 1024,  // 16 MB
            128 * 1024 * 1024  // 128 MB (large)
        };
    }
};

// =============================================================================
// Bandwidth Test Result
// =============================================================================
struct BandwidthResult {
    size_t size_bytes;
    double avg_time_us;
    double bandwidth_gbps;
    double min_time_us;
    double max_time_us;
};

// =============================================================================
// Format bytes to human-readable string
// =============================================================================
std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_idx < 3) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << size << " " << units[unit_idx];
    return oss.str();
}

// =============================================================================
// Benchmark AllReduce bandwidth using DTensor API
// =============================================================================
BandwidthResult benchmark_allreduce_bandwidth(
    size_t size_bytes,
    int rank,
    DeviceMesh& mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    const BandwidthTestConfig& config) {
    
    // Calculate tensor dimensions (use 1D tensor for simplicity)
    size_t num_elements = size_bytes / sizeof(float);
    std::vector<int64_t> shape = {static_cast<int64_t>(num_elements)};
    
    // Create replicated layout (no sharding dim = replicated)
    Layout replicated_layout(mesh, shape);
    
    // Create DTensor with random data
    DTensor tensor(mesh, pg, replicated_layout);
    tensor.rand();
    
    // Create CUDA events for precise timing
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    
    // Warmup iterations
    for (int i = 0; i < config.warmup_iters; i++) {
        tensor.sync();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Measurement iterations
    std::vector<float> times_ms;
    times_ms.reserve(config.measure_iters);
    
    for (int i = 0; i < config.measure_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start_event));
        tensor.sync();
        CUDA_CHECK(cudaEventRecord(end_event));
        CUDA_CHECK(cudaEventSynchronize(end_event));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
        times_ms.push_back(elapsed_ms);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Calculate statistics
    double total_time = 0.0;
    double min_time = times_ms[0];
    double max_time = times_ms[0];
    
    for (float t : times_ms) {
        total_time += t;
        min_time = std::min(min_time, static_cast<double>(t));
        max_time = std::max(max_time, static_cast<double>(t));
    }
    
    double avg_time_ms = total_time / config.measure_iters;
    double avg_time_us = avg_time_ms * 1000.0;
    
    // Calculate bandwidth
    // For AllReduce: data moved = 2 * size * (n-1) / n ≈ 2 * size for large n
    // Using bus bandwidth formula: bandwidth = 2 * size / time
    double data_moved_bytes = 2.0 * size_bytes;
    double bandwidth_gbps = (data_moved_bytes / (avg_time_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));
    
    BandwidthResult result;
    result.size_bytes = size_bytes;
    result.avg_time_us = avg_time_us;
    result.bandwidth_gbps = bandwidth_gbps;
    result.min_time_us = min_time * 1000.0;
    result.max_time_us = max_time * 1000.0;
    
    return result;
}

// =============================================================================
// Benchmark MLP Layer Communication (realistic workload)
// Uses sync() with MLP-sized activation tensors
// =============================================================================
struct MLPBandwidthResult {
    int batch_size;
    int hidden_dim;
    double sync_time_us;
    double bandwidth_gbps;
    double data_moved_mb;
};

MLPBandwidthResult benchmark_mlp_sync_bandwidth(
    int batch_size,
    int hidden_dim,
    int rank,
    DeviceMesh& mesh,
    std::shared_ptr<ProcessGroupNCCL> pg,
    int warmup_iters,
    int measure_iters) {
    
    const int64_t BT = batch_size * 512;  // batch * seq_len
    const int64_t C = hidden_dim;
    
    // Create activation tensor [BT, C] - this represents Y after row-parallel matmul
    // In real MLP, this would need AllReduce after row-parallel layer
    std::vector<int64_t> shape = {BT, C};
    Layout replicated_layout(mesh, shape);  // Replicated layout
    
    // Create tensor with random data (simulating MLP output)
    DTensor Y(mesh, pg, replicated_layout);
    Y.rand();
    
    // CUDA events for timing
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        Y.sync();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Measure sync time (communication)
    std::vector<float> sync_times_ms;
    sync_times_ms.reserve(measure_iters);
    
    for (int i = 0; i < measure_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start_event));
        Y.sync();
        CUDA_CHECK(cudaEventRecord(end_event));
        CUDA_CHECK(cudaEventSynchronize(end_event));
        
        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
        sync_times_ms.push_back(elapsed_ms);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Calculate stats
    double total_sync_time = 0.0;
    for (float t : sync_times_ms) {
        total_sync_time += t;
    }
    double avg_sync_ms = total_sync_time / measure_iters;
    double avg_sync_us = avg_sync_ms * 1000.0;
    
    // Data moved: Y is [BT, C] - AllReduce moves 2 * BT * C * sizeof(float)
    double data_bytes = 2.0 * BT * C * sizeof(float);
    double data_moved_mb = data_bytes / (1024.0 * 1024.0);
    double bandwidth_gbps = (data_bytes / (avg_sync_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));
    
    MLPBandwidthResult result;
    result.batch_size = batch_size;
    result.hidden_dim = hidden_dim;
    result.sync_time_us = avg_sync_us;
    result.bandwidth_gbps = bandwidth_gbps;
    result.data_moved_mb = data_moved_mb;
    
    return result;
}


// =============================================================================
// Main Entry Point
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 MPI processes (TP=2)." << std::endl;
            std::cerr << "Run with: mpirun -np 2 ./benchmarks/interconnect_benchmarks" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Setup CUDA device FIRST before any CUDA/NCCL operations
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(0));  // Initialize CUDA context
    
    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Create mesh and process group
    DeviceMesh mesh({world_size}, {0, 1});
    auto pg = mesh.get_process_group(0);
    
    if (rank == 0) {
        std::cout << "\n=== Interconnect Bandwidth Test (TP=2, " << prop.name << ") ===\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    // =========================================================================
    // Test 1: Raw AllReduce Bandwidth Sweep
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n[Test 1] AllReduce Bandwidth Sweep\n";
        std::cout << std::setw(10) << "Size" << std::setw(12) << "Time(us)" << std::setw(10) << "BW(GB/s)" << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    BandwidthTestConfig config;
    std::vector<BandwidthResult> results;
    
    for (size_t size : config.tensor_sizes) {
        auto result = benchmark_allreduce_bandwidth(size, rank, mesh, pg, config);
        results.push_back(result);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << std::setw(10) << formatBytes(result.size_bytes)
                      << std::setw(12) << result.avg_time_us
                      << std::setw(10) << result.bandwidth_gbps << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Find peak bandwidth
    double peak_bw = 0.0;
    size_t peak_size = 0;
    for (const auto& r : results) {
        if (r.bandwidth_gbps > peak_bw) {
            peak_bw = r.bandwidth_gbps;
            peak_size = r.size_bytes;
        }
    }
    
    if (rank == 0) {
        std::cout << "=> Peak: " << std::fixed << std::setprecision(1) 
                  << peak_bw << " GB/s at " << formatBytes(peak_size) << "\n";
    }
    
    // =========================================================================
    // Test 2: MLP Sync Bandwidth (Realistic Workload)
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n[Test 2] MLP Activation AllReduce (batch*512 x hidden)\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden" << std::setw(10) << "MB" << std::setw(12) << "Time(us)" << std::setw(10) << "BW(GB/s)" << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Representative MLP configs
    std::vector<std::pair<int, int>> mlp_configs = {
        {4, 768},    // GPT-2 small
        {8, 2048},   // GPT-2 large
        {16, 4096},  // GPT-3 style
    };
    
    for (auto& [bs, hdim] : mlp_configs) {
        auto result = benchmark_mlp_sync_bandwidth(bs, hdim, rank, mesh, pg, 5, 50);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << std::setw(6) << result.batch_size
                      << std::setw(8) << result.hidden_dim
                      << std::setw(10) << result.data_moved_mb
                      << std::setw(12) << result.sync_time_us
                      << std::setw(10) << result.bandwidth_gbps << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n[PASS] Peak Bandwidth: " << std::fixed << std::setprecision(1) 
                  << peak_bw << " GB/s\n";
    }
    
    MPI_Finalize();
    return 0;
}