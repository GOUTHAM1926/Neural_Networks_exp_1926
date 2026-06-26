#include <iostream>
#include <vector>
#include "core/Tensor.h"
#include "nn/NN.h"
#include "checkpointing/Checkpointing.h"
#include "nn/optimizer/Optim.h"
#include "utils/Profiler.h"

using namespace OwnTensor;

// Simple module for profiling
class ProfileModule : public nn::Module {
public:
    Tensor w0;
    Tensor w1;

    ProfileModule() {
        w0 = Tensor::randn(Shape{{1024, 1024}}, TensorOptions().with_device(Device::CUDA));
        w1 = Tensor::randn(Shape{{1024, 1024}}, TensorOptions().with_device(Device::CUDA));
        register_parameter(w0);
        register_parameter(w1);
    }

    Tensor forward(const Tensor& input) override {
        return input;
    }
};

int main() {
    try {
        std::cout << "=== Checkpoint Profiling Demo ===" << std::endl;

        // 1. Setup
        ProfileModule model;
        nn::Adam optimizer(model.parameters(), 0.001f);
        
        // 2. Enable Profiler
        autograd::Profiler::instance().set_enabled(true);
        std::cout << "Profiler enabled." << std::endl;

        // 3. Perform a Step to initialize optimizer state
        std::cout << "Performing an optimizer step..." << std::endl;
        optimizer.step();

        // 4. Perform Save
        std::cout << "Saving checkpoint..." << std::endl;
        save_checkpoint("profile_test.ckpt", model, optimizer, 1, 0.1f);

        // 4. Perform Load
        int epoch = 0;
        float loss = 0.0f;
        std::cout << "Loading checkpoint..." << std::endl;
        load_checkpoint("profile_test.ckpt", model, optimizer, epoch, loss);

        // 5. Print Stats
        autograd::Profiler::instance().print_stats();

        std::cout << "Cleanup..." << std::endl;
        std::filesystem::remove("profile_test.ckpt");
        
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
