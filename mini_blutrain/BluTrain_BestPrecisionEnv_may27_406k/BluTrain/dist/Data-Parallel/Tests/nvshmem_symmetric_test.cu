#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "../communication/include/nvshmem_include/symmetric_memory.cuh"


/*
 * Test suite for NVSHMEM symmetric memory classes.
 *
 * Build (example):
 *   nvcc -x cu -rdc=true nvshmem_symmetric_test.cu symmetric_memory.cu \
 *        -lnvshmem -lcuda -lcudart -o nvshmem_symmetric_test
 *
 * Run (must be launched with nvshmrun or mpirun for multi-PE):
 *   nvshmrun -np 2 ./nvshmem_symmetric_test
 */


#define TEST_ASSERT(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "[FAIL] PE " << my_pe << " " << msg << std::endl;    \
            test_passed = false;                                              \
        }                                                                     \
    } while(0)

#define TEST_LOG(msg)                                                         \
    do {                                                                      \
        if (my_pe == 0) {                                                     \
            std::cout << "[TEST] " << msg << std::endl;                       \
        }                                                                     \
    } while(0)


static int my_pe = -1;
static int n_pes = -1;
static bool test_passed = true;



// ---- Test: NVSHMEM_ALLOC basic lifecycle ----
void test_nvshmem_alloc_lifecycle() {
    TEST_LOG("test_nvshmem_alloc_lifecycle");

    const size_t buf_size = 1024;
    NVSHMEM_ALLOC alloc(buf_size, my_pe);

    TEST_ASSERT(!alloc.is_allocated(), "should not be allocated before allocate_memory()");
    TEST_ASSERT(alloc.get_buffer_size() == buf_size, "buffer size mismatch");
    TEST_ASSERT(alloc.get_device_idx() == my_pe, "device idx mismatch");

    void* ptr = alloc.allocate_memory();
    TEST_ASSERT(ptr != nullptr, "allocate_memory returned nullptr");
    TEST_ASSERT(alloc.is_allocated(), "should be allocated after allocate_memory()");
    TEST_ASSERT(alloc.get_ptr() == ptr, "get_ptr mismatch");

    // Double-allocate should throw
    bool threw = false;
    try {
        alloc.allocate_memory();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "double allocate should throw");

    alloc.deallocate();
    TEST_ASSERT(!alloc.is_allocated(), "should not be allocated after deallocate()");
    TEST_ASSERT(alloc.get_ptr() == nullptr, "ptr should be nullptr after deallocate");

    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NVSHMEM_ALLOC move semantics ----
void test_nvshmem_alloc_move() {
    TEST_LOG("test_nvshmem_alloc_move");

    const size_t buf_size = 512;
    NVSHMEM_ALLOC alloc1(buf_size, my_pe);
    void* ptr = alloc1.allocate_memory();

    // Move construct
    NVSHMEM_ALLOC alloc2(std::move(alloc1));
    TEST_ASSERT(alloc2.get_ptr() == ptr, "move ctor: ptr not transferred");
    TEST_ASSERT(alloc2.is_allocated(), "move ctor: should be allocated");
    TEST_ASSERT(!alloc1.is_allocated(), "move ctor: source should be invalidated");

    // Move assign
    NVSHMEM_ALLOC alloc3(256, my_pe);
    alloc3 = std::move(alloc2);
    TEST_ASSERT(alloc3.get_ptr() == ptr, "move assign: ptr not transferred");
    TEST_ASSERT(!alloc2.is_allocated(), "move assign: source should be invalidated");

    // alloc3 destructor will free the memory

    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer rank mapping ----
void test_peer_rank_mapping() {
    TEST_LOG("test_peer_rank_mapping");

    NvshmemPeer peer(my_pe, my_pe, n_pes, "test_group");

    // Own mapping should already be registered
    TEST_ASSERT(peer.local_to_global(my_pe) == my_pe, "own mapping failed");
    TEST_ASSERT(peer.global_to_local(my_pe) == my_pe, "reverse own mapping failed");

    // Register all PEs (simulating a 1:1 local-to-global mapping for this test)
    for (int i = 0; i < n_pes; ++i) {
        if (i != my_pe) {
            peer.register_rank_mapping(i, i);
        }
    }

    for (int i = 0; i < n_pes; ++i) {
        TEST_ASSERT(peer.local_to_global(i) == i, "local_to_global mismatch");
        TEST_ASSERT(peer.global_to_local(i) == i, "global_to_local mismatch");
    }

    // Out of range should throw
    bool threw = false;
    try {
        peer.local_to_global(n_pes + 10);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "out-of-range local_to_global should throw");

    threw = false;
    try {
        peer.global_to_local(9999);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "unmapped global_to_local should throw");

    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer allocation create/query/destroy ----
void test_peer_allocation_lifecycle() {
    TEST_LOG("test_peer_allocation_lifecycle");

    NvshmemPeer peer(my_pe, my_pe, n_pes, "alloc_test");
    const size_t buf_size = 2048;

    // Create
    void* ptr = peer.create_allocation("buffer_A", buf_size);
    TEST_ASSERT(ptr != nullptr, "create_allocation returned nullptr");
    TEST_ASSERT(peer.has_allocation("buffer_A"), "has_allocation should be true");
    TEST_ASSERT(peer.get_allocation_size("buffer_A") == buf_size, "size mismatch");
    TEST_ASSERT(peer.get_num_allocations() == 1, "num_allocations should be 1");

    // Local ptr lookup
    void* local_ptr = peer.get_local_ptr("buffer_A");
    TEST_ASSERT(local_ptr == ptr, "get_local_ptr mismatch");

    // Duplicate name should throw
    bool threw = false;
    try {
        peer.create_allocation("buffer_A", 512);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "duplicate alloc name should throw");

    // Non-existent lookup should throw
    threw = false;
    try {
        peer.get_local_ptr("no_such_buffer");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "get_local_ptr for missing alloc should throw");

    // Create a second allocation
    peer.create_allocation("buffer_B", 1024);
    TEST_ASSERT(peer.get_num_allocations() == 2, "should have 2 allocations");

    // Destroy one
    peer.destroy_allocation("buffer_A");
    TEST_ASSERT(!peer.has_allocation("buffer_A"), "buffer_A should be gone");
    TEST_ASSERT(peer.get_num_allocations() == 1, "should have 1 allocation");

    // Destroy all
    peer.destroy_all_allocations();
    TEST_ASSERT(peer.get_num_allocations() == 0, "should have 0 allocations");

    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer remote pointer access ----
void test_peer_remote_ptr() {
    TEST_LOG("test_peer_remote_ptr");

    NvshmemPeer peer(my_pe, my_pe, n_pes, "remote_ptr_test");

    void* ptr = peer.create_allocation("shared_buf", 4096);
    TEST_ASSERT(ptr != nullptr, "create_allocation returned nullptr");

    nvshmem_barrier_all();

    // For intra-node PEs, nvshmem_ptr should return a non-null directly-accessible pointer.
    // For cross-node PEs, it may return nullptr (which is valid behavior).
    for (int target = 0; target < n_pes; ++target) {
        void* remote = peer.get_remote_ptr("shared_buf", target);
        if (target == my_pe) {
            // Pointer to self should always succeed and match local ptr
            TEST_ASSERT(remote == ptr, "remote ptr to self should equal local ptr");
        }
        // For other PEs: remote may be nullptr (cross-node) or a valid ptr (intra-node)
        // We just verify it doesn't crash
    }

    // Out-of-range PE should throw
    bool threw = false;
    try {
        peer.get_remote_ptr("shared_buf", n_pes + 5);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TEST_ASSERT(threw, "out-of-range target_pe should throw");

    peer.destroy_all_allocations();
    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer build_peer_ptr_table ----
void test_peer_ptr_table() {
    TEST_LOG("test_peer_ptr_table");

    NvshmemPeer peer(my_pe, my_pe, n_pes, "ptr_table_test");
    peer.create_allocation("table_buf", 2048);

    nvshmem_barrier_all();

    std::vector<void*> table = peer.build_peer_ptr_table("table_buf");
    TEST_ASSERT((int)table.size() == n_pes, "table size should equal n_pes");

    // Self entry should match local ptr
    void* local = peer.get_local_ptr("table_buf");
    TEST_ASSERT(table[my_pe] == local, "self entry in table should match local ptr");

    peer.destroy_all_allocations();
    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer multiple groups ----
void test_multiple_groups() {
    TEST_LOG("test_multiple_groups");

    NvshmemPeer group1(my_pe, my_pe, n_pes, "group_alpha");
    NvshmemPeer group2(my_pe, my_pe, n_pes, "group_beta");

    void* ptr1 = group1.create_allocation("buf", 1024);
    void* ptr2 = group2.create_allocation("buf", 2048);

    // Same name "buf" in different groups should be independent allocations
    TEST_ASSERT(ptr1 != ptr2, "different groups should have different ptrs");
    TEST_ASSERT(group1.get_allocation_size("buf") == 1024, "group1 size wrong");
    TEST_ASSERT(group2.get_allocation_size("buf") == 2048, "group2 size wrong");
    TEST_ASSERT(group1.get_group_name() == "group_alpha", "group1 name wrong");
    TEST_ASSERT(group2.get_group_name() == "group_beta", "group2 name wrong");

    group1.destroy_all_allocations();
    group2.destroy_all_allocations();

    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}


// ---- Test: NvshmemPeer print_info (smoke test) ----
void test_print_info() {
    TEST_LOG("test_print_info");

    NvshmemPeer peer(my_pe, my_pe, n_pes, "print_test");
    for (int i = 0; i < n_pes; ++i) {
        peer.register_rank_mapping(i, i);
    }
    peer.create_allocation("debug_buf", 512);

    if (my_pe == 0) {
        peer.print_info();
    }

    peer.destroy_all_allocations();
    nvshmem_barrier_all();
    TEST_LOG("  PASSED");
}



int main(int argc, char* argv[]) {
    // Initialize NVSHMEM
    nvshmem_init();
    my_pe = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();

    // Set CUDA device to match PE
    cudaSetDevice(my_pe);

    if (my_pe == 0) {
        std::cout << "======================================" << std::endl;
        std::cout << " NVSHMEM Symmetric Memory Test Suite" << std::endl;
        std::cout << " PEs: " << n_pes << std::endl;
        std::cout << "======================================" << std::endl;
    }

    nvshmem_barrier_all();

    // Run tests
    test_nvshmem_alloc_lifecycle();
    test_nvshmem_alloc_move();
    test_peer_rank_mapping();
    test_peer_allocation_lifecycle();
    test_peer_remote_ptr();
    test_peer_ptr_table();
    test_multiple_groups();
    test_print_info();

    nvshmem_barrier_all();

    if (my_pe == 0) {
        std::cout << "======================================" << std::endl;
        if (test_passed) {
            std::cout << " ALL TESTS PASSED" << std::endl;
        } else {
            std::cout << " SOME TESTS FAILED" << std::endl;
        }
        std::cout << "======================================" << std::endl;
    }

    nvshmem_finalize();
    return test_passed ? 0 : 1;
}
