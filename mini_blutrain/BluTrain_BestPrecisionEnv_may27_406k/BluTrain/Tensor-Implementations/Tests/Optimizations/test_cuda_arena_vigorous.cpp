#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "device/CudaArena.h"
#include "device/CachingCudaAllocator.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include "core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/GraphArena.h"

using namespace OwnTensor::autograd;
using namespace OwnTensor;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      exit(1); \
    } \
  } while (0)
class VigorousNode : public Node {
public:
    uint64_t data[8]; // Some data to verify persistence
    
    VigorousNode(uint64_t val) : Node() {
        for(int i=0; i<8; ++i) data[i] = val + i;
    }
    
    variable_list apply(variable_list&& grads) override { return std::move(grads); }
    const char* name() const override { return "VigorousNode"; }
};
void test_cuda_alignment() {
    std::cout << "[Test] CUDA Alignment Correctness..." << std::endl;
    auto& arena = CudaArena::get_thread_local();
    arena.reset();

    for (size_t align : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
        void* ptr = arena.allocate(17, nullptr, align);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % align != 0) {
            std::cerr << "Alignment FAIL: addr " << std::hex << addr 
                      << " not aligned to " << std::dec << align << std::endl;
            exit(1);
        }
    }
    std::cout << "SUCCESS: All alignments verified on GPU." << std::endl;
}

void test_cuda_memory_reuse() {
    std::cout << "[Test] CUDA Memory Reuse..." << std::endl;
    auto& arena = CudaArena::get_thread_local();
    arena.clear();

    // 1. First allocation
    void* ptr1 = arena.allocate(1024);
    uintptr_t addr1 = reinterpret_cast<uintptr_t>(ptr1);
    std::cout << "First Addr: " << std::hex << addr1 << std::dec << std::endl;

    // 2. Reset
    arena.reset();
    std::cout << "Arena Reset." << std::endl;

    // 3. Second allocation (should be same address)
    void* ptr2 = arena.allocate(1024);
    uintptr_t addr2 = reinterpret_cast<uintptr_t>(ptr2);
    std::cout << "Second Addr: " << std::hex << addr2 << std::dec << std::endl;

    assert(addr1 == addr2);
    std::cout << "SUCCESS: GPU memory reused correctly." << std::endl;
}

void test_cuda_large_and_multi_block() {
    std::cout << "[Test] CUDA Large & Multi-Block Allocation..." << std::endl;
    auto& arena = CudaArena::get_thread_local();
    arena.clear();

    // Default block is 32MB. Allocate 40MB.
    size_t size1 = 40 * 1024 * 1024;
    void* p1 = arena.allocate(size1);
    assert(p1 != nullptr);

    // Allocate another 10MB - should be in a new block
    size_t size2 = 10 * 1024 * 1024;
    void* p2 = arena.allocate(size2);
    assert(p2 != nullptr);
    assert(p2 != p1);

    // Verify both are writable
    std::vector<uint8_t> host_data(1024, 0xAA);
    CUDA_CHECK(cudaMemcpy(p1, host_data.data(), 1024, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(p2, host_data.data(), 1024, cudaMemcpyHostToDevice));
    
    std::cout << "SUCCESS: Large and multi-block GPU allocations verified." << std::endl;
}

void test_cuda_arena_stress() {
    std::cout << "[Test] CUDA Arena Stress Test (1000 iterations)..." << std::endl;
    auto& arena = CudaArena::get_thread_local();
    arena.clear();

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        arena.reset();
        for (int j = 0; j < 100; ++j) {
            void* p = arena.allocate(1024 * (j + 1));
            assert(p != nullptr);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Allocated 100,000 GPU buffers in " << diff.count() << "s" << std::endl;
    std::cout << "SUCCESS: CUDA Arena stress test passed." << std::endl;
}

// 5. Benchmark Performance Difference (Realistic: Nodes persist)
void benchmark_allocation() {
    const int num_nodes = 10000000;
    const int num_runs = 3;
    std::cout << "[Benchmark] Realistic: All nodes stored in vector (" << num_nodes << " nodes)..." << std::endl;

    // 1. Without Arena
    double total_heap = 0;
    for (int r = 0; r < num_runs; ++r) {
        Node::use_arena_ = false;
        std::vector<VigorousNode*> nodes;
        nodes.reserve(num_nodes);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_nodes; ++i) {
            nodes.push_back(new VigorousNode(i));
        }
        for (auto n : nodes) delete n;
        auto end = std::chrono::high_resolution_clock::now();
        
        total_heap += std::chrono::duration<double>(end - start).count();
        nodes.clear();
    }
    std::cout << "  Standard Heap (Avg): " << (total_heap / num_runs) << "s" << std::endl;

    // 2. With Arena
    double total_arena = 0;
    auto& arena = CudaArena::get_thread_local();
    for (int r = 0; r < num_runs; ++r) {
        Node::use_arena_ = true;
        arena.reset();
        std::vector<VigorousNode*> nodes;
        nodes.reserve(num_nodes);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_nodes; ++i) {
            nodes.push_back(new VigorousNode(i));
        }
        // No delete needed for individual nodes
        if (r == num_runs - 1) {
            std::cout << "  Graph Arena Stats:" << std::endl;
            std::cout << "    Reserved:  " << arena.total_reserved_bytes() / (1024 * 1024) << " MB" << std::endl;
            std::cout << "    Allocated: " << arena.total_allocated_bytes() / (1024 * 1024) << " MB" << std::endl;
        }
        arena.reset(); 
        auto end = std::chrono::high_resolution_clock::now();
        
        total_arena += std::chrono::duration<double>(end - start).count();
        nodes.clear();
    }
    std::cout << "  Graph Arena (Avg):   " << (total_arena / num_runs) << "s" << std::endl;

    double speedup = total_heap / total_arena;
    std::cout << "  SPEEDUP:            " << speedup << "x" << std::endl;
}
int main() {
    std::cout << "=== VIGOROUS CUDAARENA TESTING ===" << std::endl;
    
    try {
        test_cuda_alignment();
        test_cuda_memory_reuse();
        test_cuda_large_and_multi_block();
        test_cuda_arena_stress();
        benchmark_allocation();
        
        std::cout << "\nALL CUDAARENA TESTS PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
