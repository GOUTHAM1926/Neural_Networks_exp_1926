// =============================================================================
// Direct Embedding Kernel Benchmark (bypasses autograd overhead)
// =============================================================================
// Calls the CUDA kernels directly to measure raw kernel performance.
// =============================================================================

#include "core/Tensor.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace OwnTensor;

static constexpr int VOCAB_SIZE   = 50257;
static constexpr int EMBED_DIM    = 768;
static constexpr int BATCH_SIZE   = 4;
static constexpr int SEQ_LEN      = 1024;
static constexpr int NUM_INDICES  = BATCH_SIZE * SEQ_LEN;
static constexpr int WARMUP       = 50;
static constexpr int ITERS        = 200;

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name 
              << " (SM " << prop.major << "." << prop.minor << ")\n\n";

    // --- Allocate tensors ---
    Tensor weight(Shape{{VOCAB_SIZE, EMBED_DIM}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor output(Shape{{BATCH_SIZE, SEQ_LEN, EMBED_DIM}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor grad_output(Shape{{BATCH_SIZE, SEQ_LEN, EMBED_DIM}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor grad_weight = Tensor::zeros(Shape{{VOCAB_SIZE, EMBED_DIM}},
        TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA)));

    Tensor indices(Shape{{BATCH_SIZE, SEQ_LEN}}, Dtype::UInt16, DeviceIndex(Device::CUDA));
    std::vector<uint16_t> idx_data(NUM_INDICES);
    std::srand(42);
    // High-contention scenario: 1000 unique tokens in 4096 positions
    for (int i = 0; i < NUM_INDICES; ++i) {
        idx_data[i] = static_cast<uint16_t>(std::rand() % 1000);
    }
    indices.set_data(idx_data);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =============================================
    // BENCHMARK FORWARD KERNEL
    // =============================================
    std::cout << "=== FORWARD KERNEL ===\n";
    for (int i = 0; i < WARMUP; ++i) {
        cuda::embedding_forward_cuda(
            indices.data<uint16_t>(), weight.data<float>(), output.data<float>(),
            NUM_INDICES, EMBED_DIM, VOCAB_SIZE, -1,
            weight.stride().strides[0], weight.stride().strides[1]
        );
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i) {
        cuda::embedding_forward_cuda(
            indices.data<uint16_t>(), weight.data<float>(), output.data<float>(),
            NUM_INDICES, EMBED_DIM, VOCAB_SIZE, -1,
            weight.stride().strides[0], weight.stride().strides[1]
        );
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float fwd_ms = 0;
    cudaEventElapsedTime(&fwd_ms, start, stop);
    float fwd_avg = fwd_ms / ITERS;
    // Bandwidth: read indices + read weight rows + write output
    double fwd_bytes = (double)NUM_INDICES * 2 + 2.0 * NUM_INDICES * EMBED_DIM * 4;
    double fwd_gbps = (fwd_bytes / (fwd_avg * 1e-3)) / 1e9;
    std::cout << "  Avg latency:  " << fwd_avg << " ms\n";
    std::cout << "  Throughput:   " << fwd_gbps << " GB/s\n\n";

    // =============================================
    // BENCHMARK BACKWARD KERNEL (N=4096, scalar path)
    // =============================================
    std::cout << "=== BACKWARD KERNEL (N=" << NUM_INDICES << ", optimized scalar) ===\n";
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        cudaMemset(grad_weight.data<float>(), 0, 
                   (size_t)VOCAB_SIZE * EMBED_DIM * sizeof(float));
        cuda::embedding_backward_cuda(
            indices.data<uint16_t>(), grad_output.data<float>(), grad_weight.data<float>(),
            NUM_INDICES, EMBED_DIM, VOCAB_SIZE, -1,
            grad_weight.stride().strides[0], grad_weight.stride().strides[1]
        );
    }
    cudaDeviceSynchronize();

    // Measure memset cost alone
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i) {
        cudaMemset(grad_weight.data<float>(), 0,
                   (size_t)VOCAB_SIZE * EMBED_DIM * sizeof(float));
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float memset_ms = 0;
    cudaEventElapsedTime(&memset_ms, start, stop);
    float memset_avg = memset_ms / ITERS;
    std::cout << "  Memset overhead: " << memset_avg << " ms\n";

    // Measure memset + kernel
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i) {
        cudaMemset(grad_weight.data<float>(), 0,
                   (size_t)VOCAB_SIZE * EMBED_DIM * sizeof(float));
        cuda::embedding_backward_cuda(
            indices.data<uint16_t>(), grad_output.data<float>(), grad_weight.data<float>(),
            NUM_INDICES, EMBED_DIM, VOCAB_SIZE, -1,
            grad_weight.stride().strides[0], grad_weight.stride().strides[1]
        );
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float bwd_ms = 0;
    cudaEventElapsedTime(&bwd_ms, start, stop);
    float bwd_total_avg = bwd_ms / ITERS;
    float bwd_kernel_avg = bwd_total_avg - memset_avg;
    double bwd_bytes = (double)NUM_INDICES * 2 + 2.0 * NUM_INDICES * EMBED_DIM * 4;
    double bwd_gbps = (bwd_bytes / (bwd_kernel_avg * 1e-3)) / 1e9;
    std::cout << "  Total (memset+kernel): " << bwd_total_avg << " ms\n";
    std::cout << "  Kernel only:           " << bwd_kernel_avg << " ms\n";
    std::cout << "  Kernel throughput:     " << bwd_gbps << " GB/s\n";

    // =============================================
    // BENCHMARK BACKWARD KERNEL (N=2048, cooperative path)
    // =============================================
    int small_N = 2048;
    std::cout << "\n=== BACKWARD KERNEL (N=" << small_N << ", cooperative) ===\n";
    
    Tensor small_indices(Shape{{small_N}}, Dtype::UInt16, DeviceIndex(Device::CUDA));
    std::vector<uint16_t> small_idx_data(small_N);
    for (int i = 0; i < small_N; ++i) {
        small_idx_data[i] = static_cast<uint16_t>(std::rand() % 1000);
    }
    small_indices.set_data(small_idx_data);

    Tensor small_grad_output(Shape{{small_N, EMBED_DIM}}, Dtype::Float32, DeviceIndex(Device::CUDA));

    for (int i = 0; i < WARMUP; ++i) {
        cudaMemset(grad_weight.data<float>(), 0,
                   (size_t)VOCAB_SIZE * EMBED_DIM * sizeof(float));
        cuda::embedding_backward_cuda(
            small_indices.data<uint16_t>(), small_grad_output.data<float>(), grad_weight.data<float>(),
            small_N, EMBED_DIM, VOCAB_SIZE, -1,
            grad_weight.stride().strides[0], grad_weight.stride().strides[1]
        );
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i) {
        cudaMemset(grad_weight.data<float>(), 0,
                   (size_t)VOCAB_SIZE * EMBED_DIM * sizeof(float));
        cuda::embedding_backward_cuda(
            small_indices.data<uint16_t>(), small_grad_output.data<float>(), grad_weight.data<float>(),
            small_N, EMBED_DIM, VOCAB_SIZE, -1,
            grad_weight.stride().strides[0], grad_weight.stride().strides[1]
        );
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float small_ms = 0;
    cudaEventElapsedTime(&small_ms, start, stop);
    float small_total_avg = small_ms / ITERS;
    float small_kernel_avg = small_total_avg - memset_avg;
    double small_bytes = (double)small_N * 2 + 2.0 * small_N * EMBED_DIM * 4;
    double small_gbps = (small_bytes / (small_kernel_avg * 1e-3)) / 1e9;
    std::cout << "  Total (memset+kernel): " << small_total_avg << " ms\n";
    std::cout << "  Kernel only:           " << small_kernel_avg << " ms\n";
    std::cout << "  Kernel throughput:     " << small_gbps << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ Kernel-level benchmark complete.\n";
    return 0;
}
