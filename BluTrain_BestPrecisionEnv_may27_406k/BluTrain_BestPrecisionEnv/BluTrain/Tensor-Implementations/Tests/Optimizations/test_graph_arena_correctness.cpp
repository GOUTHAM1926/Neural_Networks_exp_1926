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

using namespace OwnTensor;
using namespace OwnTensor::autograd;

/**
 * @brief Vigorous MockNode for stress testing.
 */
class VigorousNode : public Node {
public:
    uint64_t data[8]; // Some data to verify persistence
    
    VigorousNode(uint64_t val) : Node() {
        for(int i=0; i<8; ++i) data[i] = val + i;
    }
    
    variable_list apply(variable_list&& grads) override { return std::move(grads); }
    const char* name() const override { return "VigorousNode"; }
};

// 1. Test Alignment Correctness
void test_alignment() {
    std::cout << "[Test] Alignment Correctness..." << std::endl;
    auto& arena = GraphArena::get_thread_local();
    arena.reset();

    for (size_t align : {1, 2, 4, 8, 16, 32, 64}) {
        void* ptr = arena.allocate(17, align);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % align != 0) {
            std::cerr << "Alignment FAIL: addr " << std::hex << addr 
                      << " not aligned to " << std::dec << align << std::endl;
            exit(1);
        }
    }
    std::cout << "SUCCESS: All alignments verified." << std::endl;
}

// 2. Test Large Allocation spanning multiple blocks
void test_large_allocation() {
    std::cout << "[Test] Large Allocation (Multiple Blocks)..." << std::endl;
    auto& arena = GraphArena::get_thread_local();
    arena.clear();

    // Default block is 64MB. Let's allocate 100MB.
    size_t large_size = 100 * 1024 * 1024;
    void* ptr = arena.allocate(large_size);
    assert(ptr != nullptr);
    
    std::memset(ptr, 0xAA, large_size); // Physical touch
    
    // Allocate another small one - should be in a new block
    void* ptr2 = arena.allocate(1024);
    assert(ptr2 != nullptr);
    assert(ptr2 != ptr);
    
    std::cout << "SUCCESS: Large allocation (100MB) handled." << std::endl;
}

// 3. Test Thread Safety (Thread-Local Isolation)
void test_thread_safety() {
    std::cout << "[Test] Thread-Local Isolation..." << std::endl;
    std::atomic<bool> failed{false};
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &failed]() {
            auto& arena = GraphArena::get_thread_local();
            arena.reset();
            
            // Each thread allocates a unique pattern
            VigorousNode* node = new VigorousNode(t * 1000);
            if (node->data[0] != (uint64_t)(t * 1000)) failed = true;
            
            // Stress it
            for(int i=0; i<10000; ++i) {
                VigorousNode* n = new VigorousNode(i);
                if (n->data[0] != (uint64_t)i) failed = true;
            }
        });
    }

    for (auto& t : threads) t.join();
    assert(!failed);
    std::cout << "SUCCESS: Thread-local isolation verified." << std::endl;
}

// 4. Verification of Reset Stability
void test_reset_stability() {
    std::cout << "[Test] Reset Stability and Data Integrity..." << std::endl;
    auto& arena = GraphArena::get_thread_local();
    arena.reset();

    // Create 1000 nodes with data
    std::vector<uintptr_t> addrs;
    for(int i=0; i<1000; ++i) {
        VigorousNode* n = new VigorousNode(i + 100);
        addrs.push_back(reinterpret_cast<uintptr_t>(n));
    }

    arena.reset();

    // Re-allocate - pointers should match exactly if same pattern used
    for(int i=0; i<1000; ++i) {
        VigorousNode* n = new VigorousNode(i + 999);
        assert(reinterpret_cast<uintptr_t>(n) == addrs[i]);
        assert(n->data[0] == (uint64_t)(i + 999));
    }
    std::cout << "SUCCESS: Reset stability and reuse verified." << std::endl;
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
    auto& arena = GraphArena::get_thread_local();
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
    std::cout << "=== VIGOROUS GRAPHARENA TESTING & BENCHMARKING ===" << std::endl;
    
    // Ensure we start with Arena enabled for correctness tests
    Node::use_arena_ = true;
    
    test_alignment();
    test_large_allocation();
    test_thread_safety();
    test_reset_stability();
    
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    benchmark_allocation();
    
    std::cout << "\nALL TESTS AND BENCHMARKS COMPLETED." << std::endl;
    return 0;
}
