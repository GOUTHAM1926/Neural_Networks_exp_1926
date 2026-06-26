#include "checkpointing/Checkpointing.h"
#include "nn/NN.h"
#include "core/Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace OwnTensor;
using namespace OwnTensor::nn;

// Simple models for testing mismatches
class SmallModel : public Module {
public:
    SmallModel() {
        l1 = std::make_shared<Linear>(10, 10);
        register_module(l1.get());
    }
    Tensor forward(const Tensor& input) override {
        return l1->forward(input);
    }
private:
    std::shared_ptr<Linear> l1;
};

class LargeModel : public Module {
public:
    LargeModel() {
        l1 = std::make_shared<Linear>(10, 10);
        l2 = std::make_shared<Linear>(10, 10);
        register_module(l1.get());
        register_module(l2.get());
    }
    Tensor forward(const Tensor& input) override {
        return l2->forward(l1->forward(input));
    }
private:
    std::shared_ptr<Linear> l1, l2;
};

class WrongShapeModel : public Module {
public:
    WrongShapeModel() {
        l1 = std::make_shared<Linear>(10, 20); // Different shape than SmallModel
        register_module(l1.get());
    }
    Tensor forward(const Tensor& input) override {
        return l1->forward(input);
    }
private:
    std::shared_ptr<Linear> l1;
};

void test_invalid_magic() {
    std::cout << "Testing invalid magic number..." << std::endl;
    std::string path = "invalid_magic.ckpt";
    {
        std::ofstream os(path, std::ios::binary);
        os.write("WRNG", 4);
    }

    SmallModel model;
    std::vector<Tensor> params = model.parameters();
    AdamW opt(params, 0.001f);
    int epoch;
    float loss;

    try {
        load_checkpoint(path, model, opt, epoch, loss);
        std::cerr << "Expected runtime_error for invalid magic, but none thrown." << std::endl;
        exit(1);
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught expected error: " << e.what() << std::endl;
    }
    std::filesystem::remove(path);
}

void test_missing_file() {
    std::cout << "Testing missing file..." << std::endl;
    SmallModel model;
    std::vector<Tensor> params = model.parameters();
    AdamW opt(params, 0.001f);
    int epoch;
    float loss;

    try {
        load_checkpoint("non_existent.ckpt", model, opt, epoch, loss);
        std::cerr << "Expected runtime_error for missing file, but none thrown." << std::endl;
        exit(1);
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught expected error: " << e.what() << std::endl;
    }
}

void test_parameter_count_mismatch() {
    std::cout << "Testing parameter count mismatch..." << std::endl;
    std::string path = "count_mismatch.ckpt";
    
    // Save with SmallModel (2 params: weight, bias)
    {
        SmallModel model;
        std::vector<Tensor> params = model.parameters();
        AdamW opt(params, 0.001f);
        save_checkpoint(path, model, opt, 1, 0.5f);
    }

    // Try to load into LargeModel (4 params: weight1, bias1, weight2, bias2)
    {
        LargeModel model;
        std::vector<Tensor> params = model.parameters();
        AdamW opt(params, 0.001f);
        int epoch;
        float loss;
        try {
            load_checkpoint(path, model, opt, epoch, loss);
            std::cerr << "Expected error for parameter count mismatch, but none thrown." << std::endl;
            exit(1);
        } catch (const std::runtime_error& e) {
            std::cout << "  Caught expected error: " << e.what() << std::endl;
        }
    }
    std::filesystem::remove(path);
}

void test_parameter_shape_mismatch() {
    std::cout << "Testing parameter shape mismatch..." << std::endl;
    std::string path = "shape_mismatch.ckpt";
    
    // Save with SmallModel [10, 10]
    {
        SmallModel model;
        std::vector<Tensor> params = model.parameters();
        AdamW opt(params, 0.001f);
        save_checkpoint(path, model, opt, 1, 0.5f);
    }

    // Try to load into WrongShapeModel [10, 20]
    {
        WrongShapeModel model;
        std::vector<Tensor> params = model.parameters();
        AdamW opt(params, 0.001f);
        int epoch;
        float loss;
        try {
            load_checkpoint(path, model, opt, epoch, loss);
            std::cerr << "Expected error for parameter shape mismatch, but none thrown." << std::endl;
            exit(1);
        } catch (const std::exception& e) {
            std::cout << "  Caught expected error: " << e.what() << std::endl;
        }
    }
    std::filesystem::remove(path);
}

void test_manager_max_to_keep() {
    std::cout << "Testing CheckpointManager max_to_keep..." << std::endl;
    std::string dir = "test_ckpts";
    if (std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
    
    CheckpointManager manager(dir, "edge", 2); // Keep only 2
    SmallModel model;
    std::vector<Tensor> params = model.parameters();
    AdamW opt(params, 0.001f);

    manager.save(1, model, opt, 0.1f);
    manager.save(2, model, opt, 0.2f);
    manager.save(3, model, opt, 0.3f); // Should delete step 1

    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) count++;
    }

    assert(count == 2);
    assert(!std::filesystem::exists(dir + "/edge_step_1.ckpt"));
    assert(std::filesystem::exists(dir + "/edge_step_2.ckpt"));
    assert(std::filesystem::exists(dir + "/edge_step_3.ckpt"));
    
    std::cout << "  Cleanup working correctly." << std::endl;
    std::filesystem::remove_all(dir);
}

void test_empty_manager() {
    std::cout << "Testing CheckpointManager with no files..." << std::endl;
    std::string dir = "empty_ckpts";
    if (std::filesystem::exists(dir)) std::filesystem::remove_all(dir);
    
    CheckpointManager manager(dir, "edge", 5);
    SmallModel model;
    std::vector<Tensor> params = model.parameters();
    AdamW opt(params, 0.001f);
    int step = -1;
    float loss = -1.0f;

    bool loaded = manager.load_latest(model, opt, step, loss);
    assert(!loaded);
    assert(step == -1);
    
    std::cout << "  Handles empty directory gracefully." << std::endl;
    std::filesystem::remove_all(dir);
}

int main() {
    try {
        test_invalid_magic();
        test_missing_file();
        test_parameter_count_mismatch();
        test_parameter_shape_mismatch();
        test_manager_max_to_keep();
        test_empty_manager();
        
        std::cout << "\nAll Checkpointing Edge Case Tests PASSED!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error in edge case tests: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
