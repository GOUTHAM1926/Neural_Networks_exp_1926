#include <iostream>
#include <filesystem>
#include <fstream>
#include <cassert>
#include "checkpointing/Checkpointing.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;
namespace fs = std::filesystem;

// Mock classes for testing
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
    try {
        std::string test_dir = "test_checkpoints";
        if (fs::exists(test_dir)) fs::remove_all(test_dir);
        fs::create_directories(test_dir);

        MockModule model;
        auto params = model.parameters();
        nn::Adam optimizer(params, 0.001f);
        
        CheckpointManager manager(test_dir, "test_model", 3);
        
        std::cout << "Testing atomic save..." << std::endl;
        manager.save(1, model, optimizer, 0.5f);
        
        fs::path ckpt_path = fs::path(test_dir) / "test_model_step_1.ckpt";
        fs::path tmp_path = fs::path(test_dir) / "test_model_step_1.tmp";
        
        // Assert final file exists and tmp file does not
        assert(fs::exists(ckpt_path));
        assert(!fs::exists(tmp_path));
        
        std::cout << "Atomic save verification successful: .ckpt exists, .tmp removed." << std::endl;
        
        // Cleanup
        fs::remove_all(test_dir);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
