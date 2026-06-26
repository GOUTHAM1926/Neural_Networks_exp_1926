/**
 * @file 05_mlp_examples.cpp
 * @brief GUIDE: Multi-Layer Perceptron Examples
 * 
 * This guide demonstrates:
 * - Building MLP layers with autograd ops
 * - Module base class pattern
 * - Linear, ReLU modules
 * - Training loop with SGD
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/05_mlp_examples.cpp
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/Engine.h"
#include "autograd/Variable.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

using namespace OwnTensor;

// ============================================================================
// MODULE BASE CLASS
// ============================================================================
class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; }
    
    Tensor operator()(const Tensor& input) {
        return forward(input);
    }
};

// ============================================================================
// LINEAR LAYER
// ============================================================================
class Linear : public Module {
private:
    Tensor weight_;
    Tensor bias_;
    
public:
    Linear(int64_t in_features, int64_t out_features) {
        TensorOptions opts = TensorOptions().with_req_grad(true);
        
        // Xavier initialization
        float std = std::sqrt(2.0f / (in_features + out_features));
        weight_ = Tensor::randn<float>(Shape{{out_features, in_features}}, opts);
        
        // Scale weights
        float* w = const_cast<float*>(weight_.data<float>());
        for (size_t i = 0; i < weight_.numel(); i++) {
            w[i] *= std;
        }
        
        bias_ = Tensor::zeros(Shape{{out_features}}, opts);
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor wt = weight_.t();
        Tensor out = autograd::matmul(input, wt);
        return autograd::add(out, bias_);
    }
    
    std::vector<Tensor*> parameters() override {
        return {&weight_, &bias_};
    }
};

// ============================================================================
// RELU ACTIVATION
// ============================================================================
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return autograd::relu(input);
    }
};

// ============================================================================
// SEQUENTIAL CONTAINER
// ============================================================================
class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> modules_;
    
public:
    void add(std::shared_ptr<Module> module) {
        modules_.push_back(module);
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor x = input;
        for (auto& module : modules_) {
            x = module->forward(x);
        }
        return x;
    }
    
    std::vector<Tensor*> parameters() override {
        std::vector<Tensor*> params;
        for (auto& module : modules_) {
            auto module_params = module->parameters();
            params.insert(params.end(), module_params.begin(), module_params.end());
        }
        return params;
    }
    
    size_t size() const { return modules_.size(); }
};

// ============================================================================
// LOSS FUNCTION
// ============================================================================
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    // Compute (pred - target)^2
    size_t n = pred.numel();
    Tensor diff(pred.shape(), TensorOptions().with_req_grad(pred.requires_grad()));
    
    const float* p = pred.data<float>();
    const float* t = target.data<float>();
    float* d = const_cast<float*>(diff.data<float>());
    
    for (size_t i = 0; i < n; i++) {
        d[i] = p[i] - t[i];
    }
    
    Tensor squared = autograd::mul(diff, diff);
    return autograd::mean(squared);
}

// ============================================================================
// SGD UPDATE
// ============================================================================
void sgd_step(std::vector<Tensor*>& params, float lr) {
    for (Tensor* p : params) {
        if (p->owns_grad()) {
            const float* grad = p->grad<float>();
            float* data = const_cast<float*>(p->data<float>());
            
            for (size_t i = 0; i < p->numel(); i++) {
                data[i] -= lr * grad[i];
            }
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║         GUIDE: Multi-Layer Perceptron Examples       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // =========================================================================
    // Example 1: Simple 2-Layer MLP
    // =========================================================================
    std::cout << "=== Example 1: Simple 2-Layer MLP ===\n\n";
    
    {
        Sequential mlp;
        mlp.add(std::make_shared<Linear>(4, 8));
        mlp.add(std::make_shared<ReLU>());
        mlp.add(std::make_shared<Linear>(8, 2));
        
        std::cout << "Architecture: Linear(4->8) -> ReLU -> Linear(8->2)\n";
        std::cout << "Number of modules: " << mlp.size() << "\n";
        
        auto params = mlp.parameters();
        std::cout << "Number of parameter tensors: " << params.size() << "\n\n";
        
        // Forward pass
        Tensor x = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_req_grad(true));
        Tensor y = mlp.forward(x);
        
        std::cout << "Input shape: [2, 4]\n";
        std::cout << "Output numel: " << y.numel() << "\n\n";
    }

    // =========================================================================
    // Example 2: Training Loop
    // =========================================================================
    std::cout << "=== Example 2: Training Loop ===\n\n";
    
    {
        Linear layer1(4, 8);
        Linear layer2(8, 1);
        
        // Synthetic data
        Tensor X = Tensor::randn<float>(Shape{{8, 4}}, TensorOptions());
        Tensor Y = Tensor::randn<float>(Shape{{8, 1}}, TensorOptions());
        
        float lr = 0.01f;
        int epochs = 3;
        
        std::cout << "Training for " << epochs << " epochs...\n";
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Create gradient-enabled copy
            Tensor x = Tensor::randn<float>(Shape{{8, 4}}, TensorOptions().with_req_grad(true));
            const float* X_data = X.data<float>();
            float* x_data = const_cast<float*>(x.data<float>());
            for (size_t i = 0; i < X.numel(); i++) x_data[i] = X_data[i];
            
            // Forward
            Tensor h = layer1.forward(x);
            Tensor h_relu = autograd::relu(h);
            Tensor pred = layer2.forward(h_relu);
            
            Tensor loss = mse_loss(pred, Y);
            
            std::cout << "  Epoch " << epoch << ": loss = " 
                      << loss.data<float>()[0] << "\n";
            
            // Backward
            autograd::backward(loss);
            
            // SGD update
            auto params = layer1.parameters();
            auto params2 = layer2.parameters();
            params.insert(params.end(), params2.begin(), params2.end());
            sgd_step(params, lr);
        }
        std::cout << "\n";
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "=== MLP BUILDING PATTERNS ===\n\n";
    std::cout << "1. Create Module subclasses for each layer type\n";
    std::cout << "2. Use Sequential to chain modules together\n";
    std::cout << "3. Call forward() to compute predictions\n";
    std::cout << "4. Compute loss and call autograd::backward()\n";
    std::cout << "5. Update parameters with SGD\n";
    std::cout << "6. Repeat for multiple epochs\n";

    std::cout << "\n✅ MLP examples guide complete!\n\n";
    return 0;
}
