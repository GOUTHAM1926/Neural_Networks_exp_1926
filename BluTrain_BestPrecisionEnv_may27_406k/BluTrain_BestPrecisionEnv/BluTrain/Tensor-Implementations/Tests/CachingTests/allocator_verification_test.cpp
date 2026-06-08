#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_runtime.h>
#include "device/CachingCudaAllocator.h"
#include "device/AllocationTracker.h"
#include "device/SizeClass.h"

using namespace OwnTensor;

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                  << " " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" << std::endl; \
        std::exit(1); \
    }

#define ASSERT_TRUE(a) \
    if (!(a)) { \
        std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                  << " " << #a << " is false" << std::endl; \
        std::exit(1); \
    }

void print_header(const std::string& title) {
    std::cout << "\n=======================================================\n";
    std::cout << "[TEST] " << title << "\n";
    std::cout << "=======================================================\n";
}

void test_basic_reuse() {
    print_header("Basic Reuse Test");
    auto& allocator = CachingCUDAAllocator::instance();
    
    size_t size = 1024 * 1024; // 1MB (Small Pool boundary is <= 1MB)
    
    std::cout << "1. Allocating Block A (1MB)\n";
    void* ptr_a = allocator.allocate(size);
    
    std::cout << "2. Freeing Block A\n";
    allocator.deallocate(ptr_a);
    
    std::cout << "3. Allocating Block B (1MB)\n";
    void* ptr_b = allocator.allocate(size);
    
    std::cout << "   Ptr A: " << ptr_a << "\n";
    std::cout << "   Ptr B: " << ptr_b << "\n";
    
    ASSERT_EQ(ptr_a, ptr_b);
    std::cout << "PASSED: Pointer reused.\n";
    
    allocator.deallocate(ptr_b);
}

void test_size_rounding() {
    print_header("Size Class Rounding Test");
    auto& allocator = CachingCUDAAllocator::instance();
    
    // 512 bytes -> Rounds to 512
    size_t small_req = 100;
    size_t expected_small = 512;
    
    // 1.5MB -> Rounds to kLargeBuffer (20MB) check logic?
    // Wait, SizeClass logic:
    // < 1MB: Round to 512
    // < 2MB: Round to 2MB (kSmallBuffer)
    // < 20MB: Round to 20MB (kLargeBuffer)
    // >= 20MB: Round to 2MB multiples
    
    std::cout << "1. Allocating 100 bytes (should round to 512)\n";
    void* p1 = allocator.allocate(small_req); // 100 -> 512
    
    // Allocate another 100 bytes
    void* p2 = allocator.allocate(small_req + 10); // 110 -> 512
    
    allocator.deallocate(p1);
    
    // Request 512 exactly
    void* p3 = allocator.allocate(512);
    
    ASSERT_EQ(p1, p3);
    std::cout << "PASSED: 100 bytes and 512 bytes map to same bin.\n";
    
    allocator.deallocate(p2);
    allocator.deallocate(p3);
}

void test_splitting_and_coalescing() {
    print_header("Splitting and Coalescing Test");
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache(); // Start clean
    
    // Test Strategy:
    // 1. Alloc 40MB. Free it.
    // 2. Request 30MB. Should split 40MB -> 30MB + 10MB.
    // 3. Request 1.5MB (rounds to 2MB). Should split 10MB -> 2MB + 8MB.
    
    size_t super_size = 40 * 1024 * 1024; 
    std::cout << "1. Allocating 40MB.\n";
    void* super_ptr = allocator.allocate(super_size);
    std::cout << "   Super Ptr: " << super_ptr << "\n";
    allocator.deallocate(super_ptr);
    
    std::cout << "2. Allocated & Freed 40MB. Now requesting 30MB.\n";
    size_t req_30mb = 30 * 1024 * 1024;
    void* ptr_30mb = allocator.allocate(req_30mb);
    std::cout << "   Ptr 30MB: " << ptr_30mb << "\n";
    
    ASSERT_EQ(super_ptr, ptr_30mb);
    std::cout << "PASSED: Reused start of large block.\n";
    
    std::cout << "3. Requesting 1.5MB (rounds to 2MB). Should take from 10MB remainder.\n";
    // 1.5MB rounds to 2MB. 2MB is > 1MB (Large Pool).
    // The 10MB remainder is in the Large Pool.
    size_t req_1_5mb = 1572864; // 1.5MB
    void* ptr_2mb = allocator.allocate(req_1_5mb);
    
    size_t offset = (char*)ptr_2mb - (char*)ptr_30mb;
    std::cout << "   Ptr 2MB: " << ptr_2mb << " (Offset: " << offset << ")\n";
    
    // We expect the 30MB block to be exactly 30MB? 
    // round_size(30MB) -> 30MB.
    ASSERT_EQ(offset, req_30mb);
    std::cout << "PASSED: 2MB block adjacent to 30MB block.\n";
    
    std::cout << "4. Freeing 2MB block.\n";
    allocator.deallocate(ptr_2mb);
    
    std::cout << "5. Freeing 30MB block. Should coalesce back to 40MB.\n";
    allocator.deallocate(ptr_30mb);
    
    std::cout << "6. Allocating 40MB again. Should get original pointer.\n";
    void* super_ptr_2 = allocator.allocate(super_size);
    
    ASSERT_EQ(super_ptr, super_ptr_2);
    std::cout << "PASSED: Coalesced back to original large block.\n";
    
    allocator.deallocate(super_ptr_2);
}

void test_cross_pool_coalescing() {
    print_header("Cross-Pool Coalescing (Small <-> Large)");
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache();

    // Strategy:
    // 1. Alloc a chunk that bridges the small/large size boundary.
    //    Actually, we can't control adjacency directly. 
    //    But we can split a large block down to a small block.
    
    // Alloc 4MB (Large Pool).
    size_t size_4mb = 4 * 1024 * 1024;
    // ensure it rounds to something specific?
    // 4MB rounds to 20MB (kLargeBuffer) if < 20MB?
    // SizeClass: if (size < kLargeBuffer) return kLargeBuffer;
    // kLargeBuffer = 20MB.
    // So any alloc between 2MB and 20MB becomes 20MB.
    // This makes it hard to test cross-pool split/merge unless we go > 20MB.
    
    // Let's try to split a 22MB block (which is > 20MB).
    // 22MB rounds to 22MB.
    size_t size_22mb = 22 * 1024 * 1024;
    void* ptr = allocator.allocate(size_22mb);
    allocator.deallocate(ptr);
    
    // Now we have 22MB cached.
    // Request 20MB.
    // 2MB remainder.
    // 2MB rounds to 20MB? No.
    // SizeClass::round_size(2MB) -> kLargeBuffer (1048576 * 20 = 20MB)?
    // No, if size < kLargeBuffer (20MB) return kLargeBuffer.
    // So 2MB becomes 20MB.
    // So we can't split 22MB into 20MB + 2MB because the 2MB remainder is too small 
    // to satisfy a "valid" block size for that logic? 
    // Wait, the allocator `try_split` checks `remaining >= SizeClass::kSmallSize`.
    // kSmallSize is 1MB.
    // If we have a 2MB remainder, does it stay 2MB? Yes.
    // But when we try to alloc that 2MB later... `round_size(2MB)` -> 20MB.
    // It won't fit into the 2MB gap.
    
    // This implies `SizeClass` logic is very aggressive about padding up to 20MB.
    // That seems like a potential inefficiency for 3MB, 4MB allocs.
    // But for the purpose of this test, let's use what we have.
    
    // Let's rely on splitting logic:
    // If I request 1MB. `round_size` -> 1MB.
    // If I have a 20MB block. 
    // Split 1MB off. Remainder 19MB.
    // 19MB block goes to Large Pool?
    // 1MB block goes to Small Pool? (Actually allocated blocks are just tracked).
    // When 1MB is freed, it goes to Small Pool.
    // When 19MB is freed, it goes to Large Pool (size > 1MB).
    // They are adjacent. They should coalesce.
    
    // Alloc 20MB.
    void* big = allocator.allocate(20 * 1024 * 1024);
    allocator.deallocate(big); // Cached 20MB in Large Pool.
    
    // Alloc 1MB.
    std::cout << "1. Allocating 1MB. Should split 20MB block.\n";
    void* small = allocator.allocate(1024 * 1024); 
    // rounds to 1048576 (1MB). matches kSmallSize.
    
    // Verify implementation:
    // try_split will check if remaining (19MB) >= kSmallSize (1MB). Yes.
    // New block (19MB) inserted into pool. (Large Pool ? 19MB > 1MB).
    
    ASSERT_EQ(big, small);
    std::cout << "PASSED: 1MB allocation took start of 20MB block.\n";
    
    // Alloc remaining 19MB? 
    // If I alloc 10MB -> rounds to 20MB. Won't fit.
    // I must alloc something that rounds to <= 19MB.
    // But everything between 2MB and 20MB rounds to 20MB.
    // This gap in SizeClass is tricky. 
    // But I can alloc another 1MB.
    
    void* small2 = allocator.allocate(1024 * 1024);
    
    std::cout << "2. Allocated 2nd 1MB.\n";
    size_t off = (char*)small2 - (char*)small;
    std::cout << "   Offset: " << off << "\n";
    
    // Free small 2 (1MB). goes to Small Pool.
    // Remaining 18MB is in Large Pool.
    // Coalesce? 
    // small2 is adjacent to 18MB(free). 
    // small2->next is the 18MB free block.
    // try_coalesce should merge them. 1MB + 18MB -> 19MB.
    // Returns 19MB to Large Pool.
    
    allocator.deallocate(small2);
    std::cout << "3. Freed 2nd 1MB. Should merge with 18MB tail -> 19MB.\n";
    
    // Free small 1 (1MB). goes to Small Pool.
    // Neighbors: Prev=Null. Next=19MB(Free, Large Pool).
    // try_coalesce should merge. 1MB + 19MB -> 20MB.
    // Returns 20MB to Large Pool.
    
    allocator.deallocate(small);
    std::cout << "4. Freed 1st 1MB. Should merge with 19MB tail -> 20MB.\n";
    
    // Verify we have a 20MB block back.
    auto stats = allocator.get_stats();
    // Peak might be higher, but cached should be 20MB.
    std::cout << "   Cached: " << stats.cached << "\n";
    
    ASSERT_TRUE(stats.cached >= 20 * 1024 * 1024);
    
    void* big_again = allocator.allocate(20 * 1024 * 1024);
    ASSERT_EQ(big, big_again);
    
    std::cout << "PASSED: Fully coalesced back to original 20MB block.\n";
    allocator.deallocate(big_again);
}

void test_oom_handling() {
    print_header("OOM and Cache Clearing Test");
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache();
    
    // Alloc 100MB
    void* p = allocator.allocate(100 * 1024 * 1024);
    std::cout << "Allocated 100MB\n";
    allocator.deallocate(p);
    
    auto stats = allocator.get_stats();
    std::cout << "Cached before empty: " << stats.cached << "\n";
    ASSERT_TRUE(stats.cached > 0);
    
    allocator.empty_cache();
    
    stats = allocator.get_stats();
    std::cout << "Cached after empty: " << stats.cached << "\n";
    ASSERT_EQ(stats.cached, 0);
    std::cout << "PASSED: Cache is empty.\n";
}

int main() {
    try {
        AllocationTracker::instance().init("allocator_verification.csv");
        
        test_basic_reuse();
        test_size_rounding();
        test_splitting_and_coalescing();
        test_cross_pool_coalescing();
        test_oom_handling();
        
        AllocationTracker::instance().shutdown();
        
        std::cout << "\nALL TESTS PASSED.\n";
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}