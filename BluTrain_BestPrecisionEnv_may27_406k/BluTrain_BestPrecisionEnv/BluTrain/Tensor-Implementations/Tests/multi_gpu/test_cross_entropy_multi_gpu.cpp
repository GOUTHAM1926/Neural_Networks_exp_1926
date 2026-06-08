#include "core/Tensor.h"
#include "autograd/operations/LossOps.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;

void test_cross_entropy_multi_gpu() {
    std::cout << "Testing sparse_cross_entropy_loss on GPU 1..." << std::endl;

    DeviceIndex device1(Device::CUDA, 1);
    DeviceIndex device0(Device::CUDA, 0);

    // 1. Create logits on GPU 1
    int64_t B_T = 32, C = 16;
    Tensor logits = Tensor::randn(Shape{{B_T, C}}, TensorOptions().with_device(device1).with_dtype(Dtype::Float32));
    logits.set_requires_grad(true);

    // 2. Create targets on GPU 0 (to simulate mismatch)
    // We'll create on CPU first and move to GPU 0
    Tensor targets_cpu = Tensor::zeros(Shape{{B_T}}, TensorOptions().with_dtype(Dtype::Int32));
    int32_t* ptr = targets_cpu.data<int32_t>();
    for(int i=0; i<B_T; ++i) ptr[i] = i % C;
    
    Tensor targets = targets_cpu.to(device0).as_type(Dtype::UInt16);

    std::cout << "Logits device: " << (logits.device().is_cpu() ? "CPU" : "CUDA") << logits.device().index << std::endl;
    std::cout << "Targets device: " << (targets.device().is_cpu() ? "CPU" : "CUDA") << targets.device().index << std::endl;

    try {
        // 3. Forward pass
        // This should handle moving targets to GPU 1 and setting device 1
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        std::cout << "Forward success. Loss: " << loss.to_cpu().data<float>()[0] << std::endl;

        // 4. Backward pass
        // This should handle grad_output (on GPU 1) and ensure all are consistent
        loss.backward();
        std::cout << "Backward success." << std::endl;

        if (logits.grad_view().is_cuda() && logits.grad_view().device().index == 1) {
            std::cout << "Gradients successfully computed on GPU 1." << std::endl;
        } else {
            std::cout << "FAILED: Gradients not on correct device." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "FAILED with exception: " << e.what() << std::endl;
    }
}

int main() {
    #ifdef WITH_CUDA
    if (device::cuda_device_count() < 2) {
        std::cout << "Skipping test: less than 2 GPUs available." << std::endl;
        return 0;
    }
    test_cross_entropy_multi_gpu();
    #else
    std::cout << "Skipping test: CUDA not enabled." << std::endl;
    #endif
    return 0;
}
