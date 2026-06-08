#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "nn/NN.h"
#include "device/CachingCudaAllocator.h"
#include "device/AllocationTracker.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;

int main() {
    try {
        auto& tracker = AllocationTracker::instance();
        tracker.init("mlp_memory_test.csv");
        // tracker.enable_console_logging(true); // NOT SUPPORTED

        auto& allocator = CachingCUDAAllocator::instance();
        auto opts = TensorOptions().with_device(Device::CUDA).with_req_grad(true);

        std::cout << "=== Detailed MLP Memory Test ===" << std::endl;
        
        // Use 256MB tensors to reach ~1GB+ total with gradients.
        int64_t in_features = 64 * 1024; // 65536
        int64_t out_features = 1024;
        int64_t batch_size = 1024;
        
            {
                TRACK_ALLOC_SCOPE("MODEL_PARAMS");
                std::cout << "\n[DEBUG] Calling nn::Linear constructor (65536 -> 1024)..." << std::endl;
                std::cout << "Initializing Linear Layer (" << in_features << " -> " << out_features << ") on CUDA..." << std::endl;
                nn::Linear linear_layer(in_features, out_features);
                
                std::cout << "[DEBUG] Calling linear_layer.to(Device::CUDA)..." << std::endl;
                linear_layer.to(Device::CUDA);
                
                std::cout << "\n[DEBUG] Calling Tensor::rand<float> (1024 x 65536) for Input X..." << std::endl;
                std::cout << "Creating Input X (" << batch_size << " x " << in_features << ") ~ 256 MB..." << std::endl;
                Tensor x = Tensor::rand<float>(Shape{{batch_size, in_features}}, opts);
                
                {
                    TRACK_ALLOC_SCOPE("FORWARD_PASS");
                    std::cout << "\n[DEBUG] Starting Forward Pass (Linear + ReLU)..." << std::endl;
                    std::cout << "Forward Pass: Linear + ReLU..." << std::endl;
                    Tensor z = linear_layer.forward(x);
                    Tensor a = autograd::relu(z);
                    
                    std::cout << "\n[DEBUG] Calling autograd::mean for Loss..." << std::endl;
                    std::cout << "Loss Calculation (Mean reduction)..." << std::endl;
                    Tensor loss = autograd::mean(a);
                    
                    {
                        TRACK_ALLOC_SCOPE("GRADIENTS");
                        std::cout << "\n[DEBUG] Calling loss.backward()..." << std::endl;
                        std::cout << "Backward Pass (will allocate ~512MB more for gradients)..." << std::endl;
                        loss.backward();
                        
                        cudaDeviceSynchronize();
                        std::cout << "Peak Allocation Snapshot:" << std::endl;
                        tracker.write_leak_report("mlp_memory_test_leaks.txt");
                        std::cout << "Peak allocated (CUDA): " << tracker.get_peak_allocated(0) << " bytes" << std::endl;
                    }
                }
            }
        
        std::cout << "Backward pass completed." << std::endl;
        
        allocator.print_memory_summary();
        
        std::cout << "Test PASSED (Full MLP Memory Flow)" << std::endl;
        
        tracker.shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Test Finished Successfully ===" << std::endl;
    return 0;
}