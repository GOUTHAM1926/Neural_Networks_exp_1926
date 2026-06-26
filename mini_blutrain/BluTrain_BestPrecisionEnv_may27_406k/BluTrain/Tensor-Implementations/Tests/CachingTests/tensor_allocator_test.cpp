#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_runtime.h>
#include "core/Tensor.h"
#include "device/CachingCudaAllocator.h"
#include "device/AllocationTracker.h"

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

// Helper to get raw pointer
void* get_ptr(const Tensor& t) {
    if (t.dtype() == Dtype::Float32) return (void*)t.data<float>();
    // Add others if needed
    return nullptr;
}

void test_1gb_allocation() {
    print_header("1GB Tensor Allocation");
    
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache();
    
    // 1GB = 1024 * 1024 * 1024 bytes.
    // Float32 = 4 bytes.
    // Elements = 256 * 1024 * 1024 = 268,435,456.
    // Shape: {256, 1024, 1024}
    
    std::cout << "1. Creating 1GB Tensor...\n";
    {
        Tensor t = Tensor::zeros(Shape{{256, 1024, 1024}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        
        size_t nbytes = t.nbytes();
        std::cout << "   Tensor Size: " << nbytes << " bytes (" << nbytes / 1024.0 / 1024.0 / 1024.0 << " GB)\n";
        ASSERT_EQ(nbytes, 1073741824ULL);
        
        auto stats = allocator.get_stats();
        std::cout << "   Allocator Stats - Allocated: " << stats.allocated << "\n";
        ASSERT_TRUE(stats.allocated >= nbytes);
    }
    std::cout << "2. Tensor out of scope. Should be freed to cache.\n";
    
    auto stats = allocator.get_stats();
    std::cout << "   Allocator Stats - Cached: " << stats.cached << "\n";
    ASSERT_TRUE(stats.cached >= 1073741824ULL);
    
    std::cout << "PASSED: 1GB Tensor allocated and cached.\n";
}

void test_tensor_reuse() {
    print_header("Tensor Memory Reuse");
    
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache();
    
    void* ptr1 = nullptr;
    
    std::cout << "1. Allocating Tensor A (1GB)\n";
    {
        Tensor t1 = Tensor::zeros(Shape{{256, 1024, 1024}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        ptr1 = get_ptr(t1);
        std::cout << "   Ptr A: " << ptr1 << "\n";
    }
    
    std::cout << "2. Allocating Tensor B (1GB) - Should reuse A\n";
    {
        Tensor t2 = Tensor::zeros(Shape{{256, 1024, 1024}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        void* ptr2 = get_ptr(t2);
        std::cout << "   Ptr B: " << ptr2 << "\n";
        
        ASSERT_EQ(ptr1, ptr2);
    }
    
    std::cout << "PASSED: Tensor B reused memory from Tensor A.\n";
}

void test_tensor_splitting() {
    print_header("Tensor Splitting (1GB -> 512MB)");
    
    auto& allocator = CachingCUDAAllocator::instance();
    allocator.empty_cache(); // Clear cache
    
    void* ptr1 = nullptr;
    
    // 1. Alloc 1GB
    std::cout << "1. Allocating 1GB Tensor.\n";
    {
        Tensor t1 = Tensor::zeros(Shape{{256, 1024, 1024}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        ptr1 = get_ptr(t1);
    } // Freed to cache
    
    // 2. Alloc 512MB
    // Shape: {128, 1024, 1024}
    std::cout << "2. Allocating 512MB Tensor. Should take first half of 1GB block.\n";
    {
        Tensor t2 = Tensor::zeros(Shape{{128, 1024, 1024}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        void* ptr2 = get_ptr(t2);
        std::cout << "   1GB Ptr:    " << ptr1 << "\n";
        std::cout << "   512MB Ptr:  " << ptr2 << "\n";
        
        ASSERT_EQ(ptr1, ptr2);
    }
    
    std::cout << "PASSED: 512MB Tensor reused start of 1GB block.\n";
}

int main() {
    try {
        AllocationTracker::instance().init("tensor_test.csv");
        
        test_1gb_allocation();
        test_tensor_reuse();
        test_tensor_splitting();
        
        AllocationTracker::instance().shutdown();
        std::cout << "\nALL TESTS PASSED.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}