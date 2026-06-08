#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "device/AllocationTracker.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/TensorOps.h"
#include "device/DeviceCore.h"
#include "core/Tensor.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// A memory-intensive block (lots of large intermediate tensors)
variable_list heavy_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    for (int i = 0; i < 10; ++i) {
        // Each operation here generates a 256KB activation tensor (256*256*4 bytes)
        x = autograd::relu(x);
        Tensor w = Tensor::full(x.shape(), x.opts(), 0.9f);
        x = autograd::mul(x, w);
    }
    return {x};
}

void test_memory_savings(DeviceIndex device) {
    std::string dev_name = device.is_cpu() ? "CPU" : "GPU";
    int dev_id = device.is_cpu() ? -1 : device.index;
    
    std::cout << "Testing Memory Efficiency on " << dev_name << "...\n";

    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    // 256KB tensor
    Tensor x = Tensor::ones(Shape{{256, 256}}, opts);
    
    AllocationTracker& tracker = AllocationTracker::instance();
    tracker.init("memory_test.csv");
    
    // 1. Without Checkpointing
    tracker.reset_peak();
    size_t before_without = tracker.get_current_allocated(dev_id);
    {
        Tensor out = heavy_block({x})[0];
        size_t after_fwd_without = tracker.get_current_allocated(dev_id);
        std::cout << "  [Without CP] Memory after forward: " 
                  << (after_fwd_without - before_without) / 1024 << " KB\n";
        
        autograd::backward(autograd::sum(out));
    }
    tracker.reset_peak();
    x.zero_grad();

    // 2. With Checkpointing
    size_t before_with = tracker.get_current_allocated(dev_id);
    {
        variable_list out = checkpoint(heavy_block, {x});
        size_t after_fwd_with = tracker.get_current_allocated(dev_id);
        std::cout << "  [With CP]    Memory after forward: " 
                  << (after_fwd_with - before_with) / 1024 << " KB\n";
        
        // Detailed check: With CP, memory should only increase by the size of the output tensor.
        // Input x (256KB) is already in before_with. Output (256KB) is new.
        size_t diff_kb = (after_fwd_with - before_with) / 1024;
        if (diff_kb <= 320) { // 256KB + slack
             std::cout << "  - Activation Freeing: PASSED\n";
        } else {
             std::cout << "  - Activation Freeing: FAILED (Expected ~256 KB increase, got " << diff_kb << " KB)\n";
        }

        autograd::backward(autograd::sum(out[0]));
    }
}

int main() {
    try {
        test_memory_savings(DeviceIndex(Device::CPU)); // CPU
        
        #ifdef WITH_CUDA
        if (OwnTensor::device::cuda_available()) {
            test_memory_savings(DeviceIndex(Device::CUDA, 0)); // GPU
        }
        #endif
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
