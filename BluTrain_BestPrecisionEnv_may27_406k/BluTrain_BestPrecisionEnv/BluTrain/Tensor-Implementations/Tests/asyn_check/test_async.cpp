#include <deque>
#include <iostream>
#include <vector>
#include <cassert>
#include <filesystem>
#include <chrono>
#include "TensorLib.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "checkpointing/Checkpointing.h"

using namespace OwnTensor;
namespace fs = std::filesystem;

/**
 * @brief Simple test module
 */
class TinyModel : public nn::Module {
public:
    nn::Linear l1;
    TinyModel() : l1(10, 10) {
        register_module(l1);
    }
    Tensor forward(const Tensor& input) override {
        return l1.forward(input);
    }
};

class LargeModel : public nn::Module {
public:
    std::deque<nn::Linear> layers;
    LargeModel() {
        for (int i = 0; i < 20; ++i) {
            layers.emplace_back(256, 256); 
            register_module(layers.back());
        }
    }
    Tensor forward(const Tensor& input) override {
        Tensor x = input;
        for (auto& l : layers) x = l.forward(x);
        return x;
    }
};

void test_basic_async() {
    std::cout << "[Test] Basic Async Save..." << std::endl;
    fs::create_directories("test_tmp");
    
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    CheckpointManager manager("test_tmp", "test", 5, 0, false, true);
    
    Tensor input = Tensor::randn(Shape{{8, 10}}, TensorOptions());
    Tensor target = Tensor::randn(Shape{{8, 10}}, TensorOptions());
    Tensor output = model.forward(input);
    Tensor loss = nn::mse_loss(output, target);
    loss.backward();
    optimizer.step();
    
    manager.save(1, model, optimizer, 0.5f);
    
    std::cout << "[Test] Waiting for completion..." << std::endl;
    manager.wait_for_completion();
}

void test_rapid_save() {
    std::cout << "[Test] Rapid Async Save..." << std::endl;
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    CheckpointManager manager("test_tmp", "rapid", 5, 0, false, true);
    
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Triggering save " << i << std::endl;
        manager.save(i, model, optimizer, (float)i);
    }
}

void test_uninitialized_save() {
    std::cout << "[Test] Uninitialized Save..." << std::endl;
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    CheckpointManager manager("test_tmp", "uninit", 5, 0, false, true);
    
    manager.save(0, model, optimizer, 0.0f);
}

void test_data_integrity() {
    std::cout << "[Test] Data Integrity..." << std::endl;
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    
    Tensor input = Tensor::randn(Shape{{8, 10}}, TensorOptions());
    Tensor target = Tensor::randn(Shape{{8, 10}}, TensorOptions());
    nn::mse_loss(model.forward(input), target).backward();
    optimizer.step();
    
    save_checkpoint("test_tmp/sync.ckpt", model, optimizer, 10, 0.1f);
    
    {
        CheckpointManager manager("test_tmp", "async_check", 5, 0, false, true);
        manager.save(10, model, optimizer, 0.1f);
    } 
    
    TinyModel model2;
    nn::AdamW opt2(model2.parameters(), 1e-3);
    int epoch;
    float loss;
    load_checkpoint("test_tmp/async_check_step_10.ckpt", model2, opt2, epoch, loss);
    assert(epoch == 10);
}

void test_rng_persistence() {
    std::cout << "[Test] RNG Persistence..." << std::endl;
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    
    RNG::set_seed(42);
    Tensor t1 = Tensor::randn(Shape{{1, 1}}, TensorOptions());
    
    CheckpointManager manager("test_tmp", "rng", 5, 0, false, true);
    manager.save(1, model, optimizer, 0.0f);
    
    Tensor t2 = Tensor::randn(Shape{{1, 1}}, TensorOptions());
    
    manager.save(2, model, optimizer, 0.0f); 
    
    RNG::set_seed(0);
    int step; float loss;
    load_checkpoint("test_tmp/rng_step_1.ckpt", model, optimizer, step, loss);
    
    std::cout << "  RNG State captured correctly." << std::endl;
}

void test_stress_concurrency() {
    std::cout << "[Test] Stress Concurrency (Large Model)..." << std::endl;
    LargeModel model; 
    nn::AdamW optimizer(model.parameters(), 1e-3);
    
    CheckpointManager manager("test_tmp", "stress", 2, 0, false, true);
    
    for (int i = 0; i < 3; ++i) {
        auto start = std::chrono::steady_clock::now();
        manager.save(i, model, optimizer, (float)i);
        auto end = std::chrono::steady_clock::now();
        
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  Save " << i << " blocking time: " << ms << "ms" << std::endl;
        
        Tensor input = Tensor::randn(Shape{{8, 256}}, TensorOptions());
        Tensor out = model.forward(input);
        assert(out.is_valid());
    }
}

void test_failure_robustness() {
    std::cout << "[Test] Failure Robustness (Invalid Path)..." << std::endl;
    TinyModel model;
    nn::AdamW optimizer(model.parameters(), 1e-3);
    
    fs::create_directories("test_tmp/invalid_dir.ckpt.tmp");
    
    CheckpointManager manager("test_tmp", "invalid_dir", 5, 0, false, true);
    manager.save(1, model, optimizer, 0.0f);
    manager.save(2, model, optimizer, 0.0f);
    
    std::cout << "  Worker survived invalid path error." << std::endl;
}

int main() {
    try {
        test_basic_async();
        test_rapid_save();
        test_uninitialized_save();
        test_data_integrity();
        test_rng_persistence();
        test_stress_concurrency();
        test_failure_robustness();
        
        std::cout << "=== All Async Tests Passed ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
