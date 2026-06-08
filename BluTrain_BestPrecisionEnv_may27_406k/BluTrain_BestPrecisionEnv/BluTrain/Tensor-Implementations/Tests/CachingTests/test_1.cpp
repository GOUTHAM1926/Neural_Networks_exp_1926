#include "TensorLib.h"
#include "device/CachingCudaAllocator.h"
#include <cassert>

using namespace OwnTensor;
void test()
{
    AllocationTracker::instance().init("test.csv");
    auto& alloc = OwnTensor::CachingCUDAAllocator::instance();

    void* ptr1 = alloc.allocate(1024 * 1024 * 1024);
    alloc.deallocate(ptr1);

    void* ptr2 = alloc.allocate(1024 * 1024 * 1024);
    assert (ptr1 == ptr2 && "Memory should be reused!");
    alloc.deallocate(ptr2);

    auto stats = alloc.get_stats();
    assert(stats.num_cache_hits == 1);

    std::cerr << "Reuse test passed\n";
}

void test_training_memory() {
    auto& allocator = CachingCUDAAllocator::instance();
    AllocationTracker::instance().init("training_test.csv");
    auto opts = TensorOptions().with_device(Device::CUDA).with_dtype(Dtype::Float32);

    std::cout << "Initializing 3-layer MLP on CUDA..." << std::endl;
    nn::Sequential model({
        new nn::Linear(784, 128),
        new nn::ReLU(),
        new nn::Linear(128, 64),
        new nn::ReLU(),
        new nn::Linear(64, 10)
    });
    
    model.to(Device::CUDA);
    
    float learning_rate = 0.01f;
    
    for (int epoch = 0; epoch < 2; epoch++) {
        std::cout << "Epoch " << epoch << "..." << std::endl;
        for (int batch = 0; batch < 20; batch++) {
            Tensor input = Tensor::rand<float>(Shape{{32, 784}}, opts);
            Tensor targets = Tensor::zeros(Shape{{32, 10}}, opts);
            
            // Forward
            Tensor output = model.forward(input);
            Tensor loss = nn::mse_loss(output, targets);
            
            // Backward
            model.zero_grad();
            loss.backward();
            
            // Step (Manual SGD)
            for (auto& param : model.parameters()) {
                if (param.requires_grad() && param.grad() != nullptr) {
                    Tensor g = param.grad_view();
                    // param = param - lr * g
                    // Note: operator-= and operator* (Tensor, float) should be used
                    param -= g * learning_rate;
                }
            }
        }
    }
    
    allocator.print_memory_summary();
    
    auto stats = allocator.get_stats();
    float hit_rate = 100.0f * stats.num_cache_hits / std::max(1UL, stats.num_allocs);
    std::cerr << "Cache hit rate: " << hit_rate << "%\n";
    
    AllocationTracker::instance().shutdown();
    
    assert(hit_rate > 50.0f && "Hit rate should be decent after warmup");
}

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

void test_with_tracker() {
    AllocationTracker::instance().init("caching_test.csv");
    
    auto& allocator = CachingCUDAAllocator::instance();
    
    {
        Tensor a = Tensor::zeros({{1024, 5119}}, {Dtype::Float32, DeviceIndex(Device::CUDA, 0)});
        std::cout << "Allocated: " << a.nbytes() << std::endl;
        Tensor b = a + 1.0f;
        // a.display();
        // b.display();
        // a and b go out of scope - should be "freed" (returned to cache)
    }
    
    // Check tracker shows frees
    AllocationTracker::instance().print_leak_report();
    
    // Cache should have memory
    auto stats = allocator.get_stats();
    assert(stats.cached > 0);
    
    // Empty cache
    allocator.empty_cache();
    stats = allocator.get_stats();
    assert(stats.cached == 0);
    
    AllocationTracker::instance().shutdown();
}

void stress_test() {
    auto& allocator = OwnTensor::CachingCUDAAllocator::instance();
    AllocationTracker::instance().init("stress_test.csv");
    std::vector<void*> ptrs;

    try {
        std::cout << "1. Filling cache..." << std::endl;
        for(int i = 0; i < 10; ++i) {
            ptrs.push_back(allocator.allocate(100 * 1024 * 1024, 0));
        }

        std::cout << "2. Freeing chunks..." << std::endl;
        for(void* p : ptrs) {
            allocator.deallocate(p);
        }
        ptrs.clear();

        std::cout << "3. Requesting massive block..." << std::endl;
        size_t massive = 1ULL * 1024 * 1024 * 1024; 
        void* big_ptr = allocator.allocate(massive, 0);
        allocator.deallocate(big_ptr);

    } catch (const std::exception& e) {
        std::cout << "Caught expected or OOM error: " << e.what() << std::endl;
    }
    cudaDeviceSynchronize();
    AllocationTracker::instance().shutdown();
}

int main()
{
    std::cout << "--- Running test() ---" << std::endl;
    test();
    
    std::cout << "\n--- Running test_training_memory() ---" << std::endl;
    test_training_memory();
    
    std::cout << "\n--- Running test_with_tracker() ---" << std::endl;
    test_with_tracker();
    
    std::cout << "\n--- Running stress_test() ---" << std::endl;
    stress_test();
    
    cudaDeviceSynchronize();
    std::cout << "\nAll tests completed successfully." << std::endl;
    return 0;
}