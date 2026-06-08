#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cassert>
#include "autograd/AutogradContext.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

void test_concurrent_contexts() {
    std::cout << "Testing Concurrent Contexts..." << std::endl;
    
    std::atomic<int> errors{0};
    const int NUM_THREADS = 20;
    const int ITERATIONS = 1000;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            for (int k = 0; k < ITERATIONS; ++k) {
                // Each thread gets its own context
                AutogradContext ctx;
                
                // Save random tensors
                Tensor t1 = Tensor::full(Shape{{2, 2}}, TensorOptions(), static_cast<float>(i));
                ctx.save_for_backward({t1});
                
                // Simulate work
                // std::this_thread::yield();
                
                // Restore
                auto saved = ctx.get_saved_variables();
                if (saved.size() != 1) {
                    errors++;
                    break;
                }
                
                // Verify data integrity
                if (std::abs(saved[0].data<float>()[0] - static_cast<float>(i)) > 1e-4) {
                    errors++;
                    break;
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    if (errors > 0) {
        std::cerr << "FAIL: Concurrent context test failed with " << errors << " errors" << std::endl;
        exit(1);
    }
    
    std::cout << "PASS: Concurrent Contexts (" << NUM_THREADS << " threads, " << ITERATIONS << " iters)" << std::endl;
}

void test_heavy_saving() {
    std::cout << "Testing Heavy Saving (Memory Pressure)..." << std::endl;
    
    AutogradContext ctx;
    const int COUNT = 10000;
    
    // Save many times to same context (should simulate overwrite behavior if logic is standard save_for_backward)
    // The standard PyTorch save_for_backward overwrites previous saves.
    // Our implementation clears existing variables: saved_variables_.clear();
    
    for (int i = 0; i < COUNT; ++i) {
         Tensor t = Tensor::ones(Shape{{100, 100}}); // 10k floats = 40KB
         ctx.save_for_backward({t});
         
         if (ctx.num_saved_variables() != 1) {
             std::cerr << "FAIL: Failed to overwrite saved variables at iter " << i << std::endl;
             exit(1);
         }
    }
    
    // Verify final state
    auto saved = ctx.get_saved_variables();
    if (saved.size() != 1) {
         std::cerr << "FAIL: Heavy saving final size incorrect" << std::endl;
         exit(1);
    }
    
    std::cout << "PASS: Heavy Saving (" << COUNT << " overwrites)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running AutogradContext Stress Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_concurrent_contexts();
    test_heavy_saving();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All AutogradContext stress tests PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
