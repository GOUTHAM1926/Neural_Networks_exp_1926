// =============================================================================
// Embedding Kernel Throughput Benchmark
// =============================================================================
// Measures forward and backward kernel throughput in GB/s at GPT-2 scale:
//   vocab_size=50257, embed_dim=768, batch*seq = 4*1024 = 4096
//
// Forward:  measures the vectorized float4 kernel throughput
// Backward: measures the new cooperative (atomicAdd-free) kernel throughput
// =============================================================================

#include "core/Tensor.h"
#include "nn/NN.h"
#include "autograd/operations/ReductionOps.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace OwnTensor;
using namespace OwnTensor::nn;

// GPT-2 scale parameters
static constexpr int VOCAB_SIZE   = 50257;
static constexpr int EMBED_DIM    = 768;
static constexpr int BATCH_SIZE   = 4;
static constexpr int SEQ_LEN      = 1024;
static constexpr int NUM_INDICES  = BATCH_SIZE * SEQ_LEN;  // 4096

// Benchmark parameters
static constexpr int WARMUP_ITERS = 20;
static constexpr int BENCH_ITERS  = 100;

void bench_forward() {
    std::cout << "\n========================================\n";
    std::cout << " FORWARD KERNEL THROUGHPUT\n";
    std::cout << "========================================\n";
    std::cout << "  vocab=" << VOCAB_SIZE << "  embed=" << EMBED_DIM
              << "  N=" << NUM_INDICES << "\n\n";

    // Create embedding layer on CUDA
    Embedding emb(VOCAB_SIZE, EMBED_DIM, -1);
    emb.weight = emb.weight.to(DeviceIndex(Device::CUDA));

    // Create random indices on CUDA
    Tensor indices(Shape{{BATCH_SIZE, SEQ_LEN}}, Dtype::UInt16,
                   DeviceIndex(Device::CUDA));
    std::vector<uint16_t> idx_data(NUM_INDICES);
    std::srand(42);
    for (int i = 0; i < NUM_INDICES; ++i) {
        idx_data[i] = static_cast<uint16_t>(std::rand() % VOCAB_SIZE);
    }
    indices.set_data(idx_data);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        Tensor out = emb.forward(indices);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        Tensor out = emb.forward(indices);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_ms = elapsed_ms / BENCH_ITERS;

    // Bandwidth calculation:
    //   Read:  N * sizeof(uint16_t) [indices] + N * C * sizeof(float) [weight rows]
    //   Write: N * C * sizeof(float) [output]
    double bytes_per_iter = (double)NUM_INDICES * sizeof(uint16_t)
                          + 2.0 * NUM_INDICES * EMBED_DIM * sizeof(float);
    double gbps = (bytes_per_iter / (avg_ms * 1e-3)) / 1e9;

    std::cout << "  Avg latency:   " << avg_ms << " ms\n";
    std::cout << "  Throughput:    " << gbps << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void bench_backward() {
    std::cout << "\n========================================\n";
    std::cout << " BACKWARD KERNEL THROUGHPUT\n";
    std::cout << "========================================\n";
    std::cout << "  vocab=" << VOCAB_SIZE << "  embed=" << EMBED_DIM
              << "  N=" << NUM_INDICES << "\n\n";

    // Create embedding layer on CUDA
    Embedding emb(VOCAB_SIZE, EMBED_DIM, -1);
    emb.weight = emb.weight.to(DeviceIndex(Device::CUDA));

    // Create random indices with duplicates (simulating real NLP data)
    Tensor indices(Shape{{BATCH_SIZE, SEQ_LEN}}, Dtype::UInt16,
                   DeviceIndex(Device::CUDA));
    std::vector<uint16_t> idx_data(NUM_INDICES);
    std::srand(42);
    // Use small vocab subset to create high contention (worst case for atomicAdd)
    int effective_vocab = 1000;  // Only 1000 unique tokens -> lots of duplicates
    for (int i = 0; i < NUM_INDICES; ++i) {
        idx_data[i] = static_cast<uint16_t>(std::rand() % effective_vocab);
    }
    indices.set_data(idx_data);

    // Warmup — full forward + backward
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        emb.weight.zero_grad();
        Tensor out = emb.forward(indices);
        Tensor loss = autograd::mean(out);
        loss.backward();
    }
    cudaDeviceSynchronize();

    // Benchmark backward only (measured via forward+backward - forward)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i) {
        emb.weight.zero_grad();
        Tensor out = emb.forward(indices);
        Tensor loss = autograd::mean(out);
        loss.backward();
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_ms = elapsed_ms / BENCH_ITERS;

    // Bandwidth for backward kernel:
    //   Read:  N * sizeof(uint16_t) [indices] + N * C * sizeof(float) [grad_output]
    //   Write: N * C * sizeof(float) [grad_weight scatter-add]
    double bytes_per_iter = (double)NUM_INDICES * sizeof(uint16_t)
                          + 2.0 * NUM_INDICES * EMBED_DIM * sizeof(float);
    double gbps = (bytes_per_iter / (avg_ms * 1e-3)) / 1e9;

    std::cout << "  Avg latency (fwd+bwd+mean): " << avg_ms << " ms\n";
    std::cout << "  Effective throughput:        " << gbps << " GB/s\n";
    std::cout << "  (includes forward + mean reduction overhead)\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "=== Embedding Kernel Throughput Benchmark ===\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name 
              << " (SM " << prop.major << "." << prop.minor << ")\n";

    bench_forward();
    bench_backward();

    std::cout << "\n✅ Benchmark complete.\n";
    return 0;
}
