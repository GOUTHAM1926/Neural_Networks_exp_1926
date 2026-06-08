#include "TensorLib.h"
#include "device/DeviceCore.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace OwnTensor;

int main() {
    try {
        std::cout << "=== Multi-GPU 'as_type' Context Test ===" << std::endl;
        
        int gpu_count = device::cuda_device_count();
        if (gpu_count < 2) {
            std::cout << "Test requires at least 2 GPUs. Found: " << gpu_count << std::endl;
            return 0;
        }

        DeviceIndex dev0(Device::CUDA, 0);
        DeviceIndex dev1(Device::CUDA, 1);

        // --- GPU 0 Test ---
        std::cout << "\n[GPU 0] Step 1: Creating UInt16 tensor..." << std::endl;
        std::vector<uint16_t> h_data0(1024, 42);
        Tensor t0(Shape{{1, 1024}}, TensorOptions().with_dtype(Dtype::UInt16).with_device(dev0));
        t0.set_data(h_data0);

        std::cout << "[GPU 0] Step 2: Casting to Float32 (Triggers convert_type_cuda)..." << std::endl;
        Tensor t0_f32 = t0.as_type(Dtype::Float32);
        
        std::cout << "[GPU 0] Step 3: Verifying results..." << std::endl;
        Tensor t0_cpu = t0_f32.to_cpu();
        const float* d0 = t0_cpu.data<float>();
        if (std::abs(d0[0] - 42.0f) < 1e-5) {
            std::cout << "SUCCESS: GPU 0 cast correct (Value: " << d0[0] << ")" << std::endl;
        } else {
            std::cerr << "FAILURE: GPU 0 cast incorrect (Value: " << d0[0] << ", Expected: 42.0)" << std::endl;
            return 1;
        }

        // --- GPU 1 Test ---
        std::cout << "\n[GPU 1] Step 1: Creating UInt16 tensor..." << std::endl;
        std::vector<uint16_t> h_data1(1024, 13);
        Tensor t1(Shape{{1, 1024}}, TensorOptions().with_dtype(Dtype::UInt16).with_device(dev1));
        t1.set_data(h_data1);

        std::cout << "[GPU 1] Step 2: Casting to Float32 (Triggers convert_type_cuda)..." << std::endl;
        // This is where the gpt2_test.cpp likely fails
        Tensor t1_f32 = t1.as_type(Dtype::Float32);
        
        std::cout << "[GPU 1] Step 3: Verifying results..." << std::endl;
        Tensor t1_cpu = t1_f32.to_cpu();
        const float* d1 = t1_cpu.data<float>();
        if (std::abs(d1[0] - 13.0f) < 1e-5) {
            std::cout << "SUCCESS: GPU 1 cast correct (Value: " << d1[0] << ")" << std::endl;
        } else {
            std::cerr << "FAILURE: GPU 1 cast incorrect (Value: " << d1[0] << ", Expected: 13.0)" << std::endl;
            return 1;
        }

        std::cout << "\n=== All Multi-GPU context tests passed! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL FAILURE] Test threw exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
