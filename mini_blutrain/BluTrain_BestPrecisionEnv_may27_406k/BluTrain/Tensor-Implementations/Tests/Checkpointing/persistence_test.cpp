#include "checkpointing/Checkpointing.h"
#include "nn/NN.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"
#include "autograd/operations/BinaryOps.h"
#include "ops/TensorOps.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Reduction.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <filesystem>
#include <cmath>

using namespace OwnTensor;
using namespace OwnTensor::nn;

// Simple MLP for testing
class SimpleMLP : public Module {
public:
    SimpleMLP() {
        l1 = std::make_shared<Linear>(10, 20);
        l2 = std::make_shared<Linear>(20, 1);
        register_module(l1.get());
        register_module(l2.get());
    }

    Tensor forward(const Tensor& x) override {
        Tensor out = l1->forward(x);
        out = l2->forward(out);
        return out;
    }

private:
    std::shared_ptr<Linear> l1, l2;
};

bool tensors_equal(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.shape() != b.shape()) return false;
    Tensor diff = OwnTensor::abs(a - b); 
    float max_diff = *OwnTensor::reduce_max(diff).to_cpu().data<float>();
    return max_diff < tol;
}

void verify_checkpointing() {
    std::cout << "Starting Model Checkpointing Verification Test..." << std::endl;
    
    std::string ckpt_path = "test_model.ckpt";
    DeviceIndex device(Device::CPU);

#ifdef WITH_CUDA
    if (OwnTensor::device::cuda_available()) {
        device = DeviceIndex(Device::CUDA, 0);
        std::cout << "Using CUDA for test." << std::endl;
    } else {
        std::cout << "CUDA not available, using CPU." << std::endl;
    }
#else
    std::cout << "Built without CUDA, using CPU." << std::endl;
#endif

    // 1. Initialize original model and optimizer
    auto model1 = std::make_shared<SimpleMLP>();
    model1->to(device);
    
    auto params1 = model1->parameters();
    AdamW opt1(params1, 0.001f);

    // 2. Modify state (simulate some training steps)
    Tensor x = Tensor::randn<float>(Shape({4, 10}), TensorOptions().with_device(device));
    Tensor target = Tensor::randn<float>(Shape({4, 1}), TensorOptions().with_device(device));
    
    for (int i = 0; i < 5; ++i) {
        opt1.zero_grad();
        Tensor out = model1->forward(x);
        Tensor loss = mse_loss(out, target);
        loss.backward();
        opt1.step();
    }
    
    int save_epoch = 5;
    float save_loss = 0.1234f;
    
    // 3. Save checkpoint
    save_checkpoint(ckpt_path, *model1, opt1, save_epoch, save_loss);
    std::cout << "Checkpoint saved to " << ckpt_path << std::endl;

    // 4. Initialize second model and optimizer (fresh)
    auto model2 = std::make_shared<SimpleMLP>();
    model2->to(device);
    auto params2 = model2->parameters();
    AdamW opt2(params2, 0.001f);

    int load_epoch = 0;
    float load_loss = 0.0f;

    // 5. Load checkpoint
    load_checkpoint(ckpt_path, *model2, opt2, load_epoch, load_loss);
    std::cout << "Checkpoint loaded." << std::endl;

    // 6. Verify Parity
    assert(load_epoch == save_epoch);
    assert(std::abs(load_loss - save_loss) < 1e-5);

    // Verify parameters
    for (size_t i = 0; i < params1.size(); ++i) {
        if (!tensors_equal(params1[i], params2[i])) {
            std::cerr << "Parameter mismatch at index " << i << std::endl;
            exit(1);
        }
    }
    std::cout << "Model parameters verified." << std::endl;

    // Verify optimizer state (m and v buffers)
    Tensor out1 = model1->forward(x);
    Tensor out2 = model2->forward(x);
    assert(tensors_equal(out1, out2));
    
    opt1.zero_grad();
    Tensor loss1 = mse_loss(out1, target);
    loss1.backward();
    opt1.step();

    opt2.zero_grad();
    Tensor loss2 = mse_loss(out2, target);
    loss2.backward();
    opt2.step();

    // If optimizer states were restored correctly, one more step should lead to identical weights
    for (size_t i = 0; i < params1.size(); ++i) {
        if (!tensors_equal(params1[i], params2[i])) {
            std::cerr << "Optimizer state restoration failed! (Weights diverged after one step)" << std::endl;
            exit(1);
        }
    }
    std::cout << "Optimizer state restoration verified." << std::endl;

    // Cleanup
    if (std::filesystem::exists(ckpt_path)) {
        std::filesystem::remove(ckpt_path);
    }
    std::cout << "Persistence Test PASSED!" << std::endl;
}

int main() {
    try {
        verify_checkpointing();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
