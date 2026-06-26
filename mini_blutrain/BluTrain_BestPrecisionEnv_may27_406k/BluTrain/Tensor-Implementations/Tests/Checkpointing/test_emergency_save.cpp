#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cassert>
#include "checkpointing/Checkpointing.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;
namespace fs = std::filesystem;

class MockModule : public nn::Module {
public:
    Tensor weight;
    MockModule() {
        weight = Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
        register_parameter(weight);
    }
    Tensor forward(const Tensor& x) override { return x; }
};

int main() {
    std::string test_dir = "emergency_test_checkpoints";
    if (fs::exists(test_dir)) fs::remove_all(test_dir);
    fs::create_directories(test_dir);

    MockModule model;
    auto params = model.parameters();
    nn::Adam optimizer(params, 0.001f);
    CheckpointManager manager(test_dir, "model", 100);

    int current_step = 0;
    float current_loss = 0.5f;

    std::cout << "Testing emergency save on exception..." << std::endl;
    try {
        for (int step = 0; step < 10; ++step) {
            current_step = step;
            if (step == 5) {
                throw std::runtime_error("Simulated Crash");
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << std::endl;
        manager.save(current_step, model, optimizer, current_loss);
    }

    fs::path crash_ckpt = fs::path(test_dir) / "model_step_5.ckpt";
    assert(fs::exists(crash_ckpt));
    std::cout << "Verification successful: Emergency checkpoint 'model_step_5.ckpt' exists." << std::endl;

    fs::remove_all(test_dir);
    return 0;
}
