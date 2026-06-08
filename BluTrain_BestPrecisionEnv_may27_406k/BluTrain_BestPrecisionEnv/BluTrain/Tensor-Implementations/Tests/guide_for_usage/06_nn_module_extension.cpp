/**
 * @file 06_nn_module_extension.cpp
 * @brief GUIDE: Extending the NN Module System
 * 
 * This guide shows advanced patterns:
 * - Creating custom layer types
 * - Train/eval mode handling
 * - Nested modules
 * 
 * Compile: make run-snippet FILE=Tests/guide_for_usage/06_nn_module_extension.cpp
 */

#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/Node.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

using namespace OwnTensor;

// ============================================================================
// MODULE BASE CLASS
// ============================================================================
class Module {
protected:
    bool training_ = true;
    std::string name_ = "Module";
    
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; }
    
    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }
    const std::string& name() const { return name_; }
};

// ============================================================================
// DROPOUT LAYER
// ============================================================================
class Dropout : public Module {
private:
    float p_;
    
public:
    explicit Dropout(float p = 0.5f) : p_(p) { name_ = "Dropout"; }
    
    Tensor forward(const Tensor& input) override {
        if (!training_ || p_ == 0.0f) {
            return input;
        }
        
        size_t n = input.numel();
        Tensor output(input.shape(), TensorOptions());
        
        const float* in = input.data<float>();
        float* out = const_cast<float*>(output.data<float>());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        float scale = 1.0f / (1.0f - p_);
        for (size_t i = 0; i < n; i++) {
            out[i] = (dist(gen) > p_) ? in[i] * scale : 0.0f;
        }
        
        return output;
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
    Linear(int64_t in_f, int64_t out_f) {
        name_ = "Linear";
        TensorOptions opts = TensorOptions().with_req_grad(true);
        weight_ = Tensor::randn<float>(Shape{{out_f, in_f}}, opts);
        bias_ = Tensor::zeros(Shape{{out_f}}, opts);
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
// RESIDUAL BLOCK
// ============================================================================
class ResidualBlock : public Module {
private:
    std::shared_ptr<Linear> layer1_;
    std::shared_ptr<Linear> layer2_;
    
public:
    ResidualBlock(int64_t features) {
        name_ = "ResidualBlock";
        layer1_ = std::make_shared<Linear>(features, features);
        layer2_ = std::make_shared<Linear>(features, features);
    }
    
    Tensor forward(const Tensor& x) override {
        Tensor out = layer1_->forward(x);
        out = autograd::relu(out);
        out = layer2_->forward(out);
        return autograd::add(out, x);  // Residual connection
    }
    
    std::vector<Tensor*> parameters() override {
        auto p1 = layer1_->parameters();
        auto p2 = layer2_->parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║        GUIDE: Extending the NN Module System         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    TensorOptions opts = TensorOptions().with_req_grad(true);

    // =========================================================================
    std::cout << "=== 1. Dropout Layer ===\n\n";
    {
        Dropout dropout(0.3f);
        Tensor x = Tensor::ones(Shape{{4, 4}}, TensorOptions());
        
        dropout.train();
        std::cout << "Training: is_training() = " << (dropout.is_training() ? "true" : "false") << "\n";
        Tensor train_out = dropout.forward(x);
        
        dropout.eval();
        std::cout << "Eval: is_training() = " << (dropout.is_training() ? "true" : "false") << "\n\n";
    }

    // =========================================================================
    std::cout << "=== 2. Linear Layer ===\n\n";
    {
        Linear linear(4, 8);
        std::cout << "Linear(4->8) created\n";
        std::cout << "Parameters: " << linear.parameters().size() << " tensors\n\n";
    }

    // =========================================================================
    std::cout << "=== 3. Residual Block ===\n\n";
    {
        ResidualBlock res(8);
        Tensor x = Tensor::randn<float>(Shape{{2, 8}}, opts);
        Tensor y = res.forward(x);
        std::cout << "ResidualBlock(8) with skip connection\n";
        std::cout << "Output = F(x) + x (residual connection)\n";
        std::cout << "Parameters: " << res.parameters().size() << " tensors\n\n";
    }

    // =========================================================================
    std::cout << "=== PATTERNS SUMMARY ===\n\n";
    std::cout << "1. Inherit from Module, override forward() and parameters()\n";
    std::cout << "2. Use train()/eval() for mode-dependent behavior\n";
    std::cout << "3. Store learnable weights as member Tensors with req_grad\n";
    std::cout << "4. Nest modules by holding shared_ptr to sub-modules\n";
    std::cout << "5. Collect parameters from sub-modules recursively\n";

    std::cout << "\n✅ NN module extension guide complete!\n\n";
    return 0;
}
