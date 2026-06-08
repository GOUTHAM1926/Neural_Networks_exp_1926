#include <iostream>
#include <vector>
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/Hooks.h"
#include "checkpointing/Checkpoint.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"

using namespace OwnTensor;

// Mock DDP Hook Generator
std::unique_ptr<PostAccumulateGradHook> mock_ddp_hook(const std::string& param_name) {
    return make_post_acc_hook([param_name](const Tensor& grad) {
        std::cout << "[DDP Sync Thread] Param '" << param_name << "' gradient ready. Launching All-Reduce (Norm: " 
                  << std::abs(grad.data<float>()[0]) << "...)" << std::endl;
    });
}

// Simple MLP Layer
class MLPLayer : public nn::Module {
public:
    nn::Linear fc;
    std::string name_;

    MLPLayer(int in_features, int out_features, const std::string& name) 
        : fc(in_features, out_features, true), name_(name) {
        register_module(fc);
        // Register DDP Hooks onto the parameters
        auto params = parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            params[i].register_post_acc_hook(mock_ddp_hook(name_ + ".param_" + std::to_string(i)));
        }
    }

    Tensor forward(const Tensor& x) override {
        // std::cout << "  Forward pass: " << name_ << std::endl;
        return autograd::relu(fc.forward(x));
    }
};

int main() {
    try {
        std::cout << "=== Non-Reentrant DDP Training Simulator ===\n\n";
        
        DeviceIndex device(Device::CPU);
        
        // Build Sequential Model: 4 Layers
        auto model = std::make_shared<nn::Sequential>();
        model->add(std::make_shared<MLPLayer>(10, 10, "Layer_0"));
        model->add(std::make_shared<MLPLayer>(10, 10, "Layer_1"));
        model->add(std::make_shared<MLPLayer>(10, 10, "Layer_2"));
        model->add(std::make_shared<MLPLayer>(10, 10, "Layer_3"));
        
        // Optimizer
        auto params = model->parameters();
        nn::AdamW optimizer(params, 0.01f);
        
        // Mock Data
        Tensor input = Tensor::ones(Shape{{4, 10}}, TensorOptions().with_req_grad(false));
        Tensor target = Tensor::full(Shape{{4, 10}}, TensorOptions().with_req_grad(false), 0.5f);
        
        for (int step = 0; step < 3; ++step) {
            std::cout << "\n--- Training Step " << step << " ---" << std::endl;
            optimizer.zero_grad();
            
            // Forward with Sequential Checkpointing. 4 Layers, 2 segments -> 2 layers per segment
            std::cout << "1. Forward Pass (Checkpointed)" << std::endl;
            variable_list out = autograd::checkpoint_sequential(model, 2, {input}, false);
            
            // Dummy loss
            Tensor loss = autograd::sum(autograd::pow(autograd::sub(out[0], target), 2.0f));
            std::cout << "   Loss: " << loss.data<float>()[0] << std::endl;
            
            std::cout << "2. Backward Pass (Watch for synchronous DDP Hooks!)" << std::endl;
            autograd::backward(loss); // Hooks will print automatically as checkpoint backwards progress
            
            std::cout << "3. Optimizer Step" << std::endl;
            optimizer.step();
        }
        
        std::cout << "\nTraining simulation completed flawlessly without global engine re-entrancy!\n";
    } catch (const std::exception& e) {
        std::cerr << "Training Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
