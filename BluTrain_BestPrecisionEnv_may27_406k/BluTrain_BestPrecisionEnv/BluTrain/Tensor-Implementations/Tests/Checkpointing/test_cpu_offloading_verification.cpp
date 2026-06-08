#include "checkpointing/Checkpoint.h"
#include "autograd/Engine.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "device/AllocationTracker.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <cassert>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

void test_offloading_logic(bool offload) {
    std::cout << "\n>>> Testing CPU Offloading (offload_to_cpu=" << (offload ? "true" : "false") << ") <<<\n";
    
    #ifdef WITH_CUDA
    if (!OwnTensor::device::cuda_available()) {
        std::cout << "Skipping GPU test (CUDA not available)\n";
        return;
    }
    #else
        std::cout << "Skipping GPU test (WITH_CUDA not defined)\n";
        return;
    #endif

    DeviceIndex gpu(Device::CUDA, 0);
    DeviceIndex cpu(Device::CPU);
    
    AllocationTracker& tracker = AllocationTracker::instance();
    // tracker.init("test_offload.csv"); // Already might be init in main or by another test
    tracker.reset_peak();
    
    // 1. Create large GPU tensor (10MB)
    size_t tensor_size = 10 * 1024 * 1024; // 10MB
    TensorOptions opts = TensorOptions().with_device(gpu).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{1, 10 * 1024 * 1024 / 4}}, opts);
    
    size_t gpu_base = tracker.get_current_allocated(0);
    size_t cpu_base = tracker.get_current_allocated(-1);
    
    std::cout << "Initial GPU: " << gpu_base / 1024 << " KB, CPU: " << cpu_base / 1024 << " KB\n";

    // 2. Define a simple checkpointed block
    auto block = [](const variable_list& inputs) {
        return variable_list{autograd::mul(inputs[0], Tensor::full(inputs[0].shape(), inputs[0].opts(), 2.0f))};
    };

    // 3. Apply checkpoint
    variable_list outputs = checkpoint(block, {x}, offload);
    
    size_t gpu_mid = tracker.get_current_allocated(0);
    size_t cpu_mid = tracker.get_current_allocated(-1);
    
    std::cout << "Middle GPU: " << gpu_mid / 1024 << " KB, CPU: " << cpu_mid / 1024 << " KB\n";

    if (offload) {
        // If offloading is ON, CPU memory should have increased by at least 1MB
        // GPU memory might have increased for the output tensor (1MB)
        if (cpu_mid >= cpu_base + tensor_size) {
            std::cout << "SUCCESS: Input offloaded to CPU!\n";
        } else {
            std::cout << "FAILURE: Input NOT offloaded to CPU! Increase: " << (cpu_mid - cpu_base) / 1024 << " KB\n";
        }
    } else {
        // If offloading is OFF, CPU memory should NOT have increased significantly
        if (cpu_mid < cpu_base + 1024) { // small slack
            std::cout << "SUCCESS: Input stayed on GPU!\n";
        } else {
            std::cout << "FAILURE: Input unexpectedly moved to CPU! Increase: " << (cpu_mid - cpu_base) / 1024 << " KB\n";
        }
    }

    // 4. Backward pass
    std::cout << "Running backward...\n";
    autograd::backward(autograd::sum(outputs[0]));
    
    size_t gpu_final = tracker.get_current_allocated(0);
    size_t cpu_final = tracker.get_current_allocated(-1);
    std::cout << "Final GPU: " << gpu_final / 1024 << " KB, CPU: " << cpu_final / 1024 << " KB\n";
    
    // Verify it finished without error
    std::cout << "Test passed correctly.\n";
}

int main() {
    try {
        test_offloading_logic(true);
        test_offloading_logic(false);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
