#include <iostream>
#include <cassert>
#include "core/Tensor.h"
#include "nn/NN.h"
#include "device/Device.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;
using namespace OwnTensor::nn;

// Custom module to test nested registration and device transfer
class TestNestedModule : public Module {
public:
    Linear l1;
    Sequential seq;
    Embedding emb;

    TestNestedModule(int in, int hidden, int vocab) 
        : l1(in, hidden), 
          seq({new Linear(hidden, hidden), new ReLU()}),
          emb(vocab, hidden) 
    {
        // Explicitly register submodules
        register_module(l1);
        register_module(seq);
        register_module(emb);
    }

    Tensor forward(const Tensor& input) override {
        auto x = emb.forward(input);
        x = l1.forward(x);
        x = seq.forward(x);
        return x;
    }
};

int main() {
    std::cout << "Starting Module Device Transfer Test..." << std::endl;

    // 1. Check if CUDA is available
    bool cuda_available = OwnTensor::device::cuda_available();
    std::cout << "CUDA Available: " << (cuda_available ? "YES" : "NO") << std::endl;

    // 2. Create the module on CPU
    TestNestedModule model(128, 256, 1000);
    
    // Verify initial state is CPU
    assert(model.l1.weight.is_cpu());
    assert(model.emb.weight.is_cpu());
    
    // Check Sequential parameters
    auto params = model.parameters();
    std::cout << "Total parameters found: " << params.size() << " (Expected: 5 - emb.w, l1.w, l1.b, seq.l.w, seq.l.b)" << std::endl;
    // Note: Sequential contains one Linear (2 params) and one ReLU (0 params). 
    // l1 has 2 params. emb has 1 param. Total = 1 + 2 + 2 = 5.
    assert(params.size() == 5);

    if (cuda_available) {
        std::cout << "Transferring to CUDA:0..." << std::endl;
        model.to(DeviceIndex(Device::CUDA, 0));

        // 3. Verify all components moved to CUDA
        bool all_cuda = true;
        if (!model.l1.weight.is_cuda()) { std::cout << "FAIL: l1.weight not on CUDA" << std::endl; all_cuda = false; }
        if (!model.l1.bias.is_cuda())   { std::cout << "FAIL: l1.bias not on CUDA" << std::endl; all_cuda = false; }
        if (!model.emb.weight.is_cuda()) { std::cout << "FAIL: emb.weight not on CUDA" << std::endl; all_cuda = false; }
        
        // Test Sequential internal components
        // We can't easily access Sequential privates, but we can check if its parameters (from model.parameters()) are on CUDA
        for (auto& p : model.parameters()) {
            if (!p.is_cuda()) {
                std::cout << "FAIL: A parameter is still on CPU" << std::endl;
                all_cuda = false;
            }
        }

        if (all_cuda) {
            std::cout << "SUCCESS: All tensors transferred to CUDA!" << std::endl;
        } else {
            std::cout << "FAILURE: Some tensors remained on CPU!" << std::endl;
            return 1;
        }

        // 4. Test transfer back to CPU
        std::cout << "Transferring back to CPU..." << std::endl;
        model.to(DeviceIndex(Device::CPU));
        
        bool all_cpu = true;
        for (auto& p : model.parameters()) {
            if (!p.is_cpu()) {
                std::cout << "FAIL: A parameter is still on CUDA" << std::endl;
                all_cpu = false;
            }
        }
        
        if (all_cpu) {
            std::cout << "SUCCESS: All tensors transferred back to CPU!" << std::endl;
        } else {
            return 1;
        }
    } else {
        std::cout << "Skipping CUDA test (Device not available)." << std::endl;
    }

    std::cout << "Test Completed Successfully!" << std::endl;
    return 0;
}
