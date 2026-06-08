#include "nn/NN.h"
#include "autograd/AutogradOps.h"
#include "ops/ScalarOps.h"  // For operator*(Tensor, float)
#include "ops/TensorOps.h"
#include "core/Serialization.h"
#include "core/Tensor.h"
#include <cmath>
#include <fstream>
#include "device/AllocationTracker.h"
namespace OwnTensor {
namespace nn {

// ============================================================================
// Module
// ============================================================================

std::vector<Tensor> Module::parameters() {
    std::vector<Tensor> all_params = params_;
    for (auto* child : children_) {
        auto child_params = child->parameters();
        all_params.insert(all_params.end(), child_params.begin(), child_params.end());
    }
    return all_params;
}

void Module::to(DeviceIndex dev) {
    for (auto& p : parameters()) {
        if (dev.is_cuda()) {
            p.to_cuda_(dev.index);
        } else if (dev.is_cpu()) {
            p.to_cpu_();
        }
    }
}

void Module::register_module(Module& m) {
    children_.push_back(&m);
}

void Module::register_module(Module* m) {
    if (m) children_.push_back(m);
}

void Module::zero_grad() {
    for (auto& p : parameters()) {
        // Only attempt to zero gradients if they require grad and exist
        if (p.requires_grad()) {
            try {
                p.fill_grad(0.0f);
            } catch (...) {
                // If gradient not allocated, that's fine for zero_grad
            }
        }
    }
}

Tensor Module::operator()(const Tensor& input) {
    return forward(input);
}

void Module::register_parameter(Tensor p) {
    params_.push_back(p);
}

void Module::save_state_dict(const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for save_state_dict: " + path);
    }

    auto params = parameters();
    int count = static_cast<int>(params.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));

    for (const auto& p : params) {
        OwnTensor::save_tensor(p, os);
    }
    os.close();
}

void Module::load_state_dict(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for load_state_dict: " + path);
    }

    auto params = parameters();
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));

    if (count != static_cast<int>(params.size())) {
        throw std::runtime_error("Checkpoint parameter count mismatch! Module has " + 
                                  std::to_string(params.size()) + " but file has " + std::to_string(count));
    }

    for (auto& p : params) {
        Tensor loaded = OwnTensor::load_tensor(is);
        
        // Validation: check shape and dtype match
        if (loaded.shape() != p.shape()) {
             throw std::runtime_error("Parameter shape mismatch during load_state_dict!");
        }
        if (loaded.dtype() != p.dtype()) {
             throw std::runtime_error("Parameter dtype mismatch during load_state_dict!");
        }

        // Copy data into existing parameter (preserves parameter identity/device)
        p.copy_(loaded);
    }
    is.close();
}

// ============================================================================
// Linear
// ============================================================================

Linear::Linear(int in_features, int out_features, bool use_bias) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    TRACK_ALLOC_SCOPE("L118:nn::Linear::ctor");

    // Initialize weights with He/Kaiming initialization basic equivalent
    // scaling by 1/sqrt(fan_in) for uniform or normal
    float stdv = 1.0f / std::sqrt(static_cast<float>(in_features));
    
    weight = Tensor::randn<float>(Shape{{in_features, out_features}}, opts, 1.0f) * stdv;     
    
    if (use_bias) {
        bias = Tensor::zeros(Shape{{out_features}}, opts);
    } // else we should handle no bias case, but for now assuming always bias or zero tensor
      // If no bias, we could use empty tensor? matrix add supports it?
      // For simplicity, if no bias, we just init to zeros with requires_grad=false?
      // Or 0s.
    
    register_parameter(weight);
    if (use_bias) {
        register_parameter(bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // y = x @ W + b
    if (!bias.is_valid()) {
        return autograd::matmul(input, weight);
    }else{
        return autograd::addmm(bias, input, weight);
    }
    // return z;
}

// parameters() and to() are handled by base Module since parameters are registered.

// ============================================================================
// ReLU
// ============================================================================

Tensor ReLU::forward(const Tensor& input) {
    return autograd::relu(input);
}

Tensor GeLU::forward(const Tensor& input) {
    return autograd::gelu(input);
}

// ============================================================================
// Embedding
// ============================================================================

Embedding::Embedding(int num_embeddings, int embedding_dim, int padding_idx) 
    : padding_idx(padding_idx) {
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Normal distribution initialization (small normal)
    weight = Tensor::randn<float>(Shape{{num_embeddings, embedding_dim}}, opts, 0.02f);
    
    // Zero out padding row if requested
    if (padding_idx >= 0 && padding_idx < num_embeddings) {
        float* w_ptr = weight.data<float>();
        std::fill(w_ptr + (size_t)padding_idx * embedding_dim, 
                  w_ptr + (size_t)(padding_idx + 1) * embedding_dim, 0.0f);
    }
    
    register_parameter(weight);
}

Tensor Embedding::forward(const Tensor& input) {
    return autograd::embedding(weight, input, padding_idx);
}

// parameters() and to() are handled by base Module.

// ============================================================================
// Sequential
// ============================================================================

Sequential::Sequential(std::initializer_list<Module*> modules) {
    for (auto* m : modules) {
        add(std::shared_ptr<Module>(m));
    }
}

void Sequential::add(std::shared_ptr<Module> module) {
    modules_.push_back(module);
    register_module(module.get());
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& m : modules_) {
        x = m->forward(x);
    }
    return x;
}

// parameters() and to() are handled by base Module through recursive children_ tracking.

// ============================================================================
// Loss Functions
// ============================================================================

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    // loss = mean((pred - target)^2)
    Tensor neg_target = target * -1.0f;
    Tensor diff = autograd::add(pred, neg_target);
    Tensor sq_diff = autograd::mul(diff, diff);
    return autograd::mean(sq_diff);
}

} // namespace nn
} // namespace OwnTensor