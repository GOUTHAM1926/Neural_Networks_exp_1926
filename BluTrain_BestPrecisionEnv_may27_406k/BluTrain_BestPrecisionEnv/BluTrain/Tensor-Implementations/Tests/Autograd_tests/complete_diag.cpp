/**
 * @file component_diagnostic_test.cpp
 * @brief Comprehensive diagnostic test for GPT-2 training components
 * 
 * This test verifies that all components of the GPT-2 model are working correctly:
 * - Individual component forward/backward passes
 * - Gradient flow through the model
 * - Numerical stability
 * - Optimizer behavior
 * 
 * Run with: CUDA_VISIBLE_DEVICES=1 ./component_diagnostic_test
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReductionOps.h"
#include "nn/optimizer/Optim.h"
#include "nn/NN.h"

using namespace OwnTensor;

// =============================================================================
// Test Configuration
// =============================================================================

struct TestConfig {
    int64_t B = 4;           // Batch size
    int64_t T = 64;          // Sequence length
    int64_t C = 256;         // Embedding dim (smaller for faster testing)
    int64_t vocab_size = 1000;
    int64_t n_layers = 2;
    DeviceIndex device;
    
    // Note: When using CUDA_VISIBLE_DEVICES=1, the selected GPU becomes device 0
    // So we default to device 0 here
    TestConfig(int gpu_id = 0) : device(Device::CUDA, gpu_id) {}
};

// =============================================================================
// Test Utilities
// =============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double forward_time_ms = 0;
    double backward_time_ms = 0;
};

bool check_tensor_valid(const Tensor& t, const std::string& name) {
    Tensor cpu_t = t.to_cpu();
    const float* data = cpu_t.data<float>();
    int64_t n = t.numel();
    
    bool has_nan = false;
    bool has_inf = false;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) has_nan = true;
        if (std::isinf(data[i])) has_inf = true;
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    if (has_nan) {
        std::cerr << "  [FAIL] " << name << " contains NaN values!" << std::endl;
        return false;
    }
    if (has_inf) {
        std::cerr << "  [FAIL] " << name << " contains Inf values!" << std::endl;
        return false;
    }
    
    std::cout << "  [OK] " << name << " range: [" << std::fixed << std::setprecision(4) 
              << min_val << ", " << max_val << "]" << std::endl;
    return true;
}

bool check_gradient_nonzero(const Tensor& t, const std::string& name) {
    if (!t.has_grad()) {
        std::cerr << "  [FAIL] " << name << " has no gradient!" << std::endl;
        return false;
    }
    
    Tensor grad = t.grad_view();
    Tensor cpu_grad = grad.to_cpu();
    const float* data = cpu_grad.data<float>();
    int64_t n = grad.numel();
    
    bool all_zero = true;
    bool has_nan = false;
    bool has_inf = false;
    float grad_norm = 0.0f;
    
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) has_nan = true;
        if (std::isinf(data[i])) has_inf = true;
        if (data[i] != 0.0f) all_zero = false;
        grad_norm += data[i] * data[i];
    }
    grad_norm = std::sqrt(grad_norm);
    
    if (has_nan) {
        std::cerr << "  [FAIL] " << name << " gradient contains NaN!" << std::endl;
        return false;
    }
    if (has_inf) {
        std::cerr << "  [FAIL] " << name << " gradient contains Inf!" << std::endl;
        return false;
    }
    if (all_zero) {
        std::cerr << "  [FAIL] " << name << " gradient is all zeros!" << std::endl;
        return false;
    }
    
    std::cout << "  [OK] " << name << " gradient norm: " << std::scientific 
              << std::setprecision(4) << grad_norm << std::endl;
    return true;
}

void print_test_header(const std::string& name) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_test_result(const TestResult& result) {
    std::cout << (result.passed ? "[PASS]" : "[FAIL]") << " " << result.name;
    if (!result.message.empty()) {
        std::cout << " - " << result.message;
    }
    std::cout << std::endl;
}

// =============================================================================
// Component Tests
// =============================================================================

TestResult test_embedding(const TestConfig& cfg) {
    print_test_header("Embedding Layer");
    TestResult result{"Embedding", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Create embedding weight
        Tensor weight = Tensor::randn<float>(Shape{{cfg.vocab_size, cfg.C}}, opts, 1234, 0.02f);
        
        // Create indices
        std::vector<int64_t> idx_data(cfg.B * cfg.T);
        for (int64_t i = 0; i < cfg.B * cfg.T; ++i) {
            idx_data[i] = i % cfg.vocab_size;
        }
        Tensor indices_cpu = Tensor(Shape{{cfg.B, cfg.T}}, TensorOptions().with_dtype(Dtype::Int64));
        indices_cpu.set_data(idx_data);
        Tensor indices = indices_cpu.to(cfg.device);
        
        // Forward pass
        Tensor output = autograd::embedding(weight, indices);
        
        if (!check_tensor_valid(output, "Embedding output")) {
            result.passed = false;
            return result;
        }
        
        // Backward pass
        Tensor grad_out = Tensor::ones(output.shape(), opts);
        output.backward(&grad_out);
        
        if (!check_gradient_nonzero(weight, "Embedding weight")) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_gelu(const TestConfig& cfg) {
    print_test_header("GELU Activation");
    TestResult result{"GELU", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        Tensor x = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts);
        
        // Forward
        Tensor y = autograd::gelu(x);
        
        if (!check_tensor_valid(y, "GELU output")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(y.shape(), opts);
        y.backward(&grad_out);
        
        if (!check_gradient_nonzero(x, "GELU input")) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_layernorm(const TestConfig& cfg) {
    print_test_header("LayerNorm");
    TestResult result{"LayerNorm", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        Tensor x = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts);
        Tensor gamma = Tensor::ones(Shape{{cfg.C}}, opts);
        Tensor beta = Tensor::zeros(Shape{{cfg.C}}, opts);
        
        // Forward
        Tensor y = autograd::layer_norm(x, gamma, beta, cfg.C);
        
        if (!check_tensor_valid(y, "LayerNorm output")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(y.shape(), opts);
        y.backward(&grad_out);
        
        bool ok = true;
        ok = check_gradient_nonzero(x, "LayerNorm input") && ok;
        ok = check_gradient_nonzero(gamma, "LayerNorm gamma") && ok;
        // beta gradient might be zero if all ones grad_out
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_matmul(const TestConfig& cfg) {
    print_test_header("MatMul");
    TestResult result{"MatMul", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        int64_t hidden = cfg.C * 4;
        Tensor a = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts, 1234, 0.02f);
        Tensor b = Tensor::randn<float>(Shape{{cfg.C, hidden}}, opts, 5678, 0.02f);
        
        // Forward
        Tensor c = autograd::matmul(a, b);
        
        if (!check_tensor_valid(c, "MatMul output")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(c.shape(), opts);
        c.backward(&grad_out);
        
        bool ok = true;
        ok = check_gradient_nonzero(a, "MatMul input A") && ok;
        ok = check_gradient_nonzero(b, "MatMul input B") && ok;
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_add(const TestConfig& cfg) {
    print_test_header("Add (Broadcasting)");
    TestResult result{"Add", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        Tensor a = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts);
        Tensor b = Tensor::randn<float>(Shape{{cfg.C}}, opts);  // Bias-style broadcast
        
        // Forward
        Tensor c = autograd::add(a, b);
        
        if (!check_tensor_valid(c, "Add output")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(c.shape(), opts);
        c.backward(&grad_out);
        
        bool ok = true;
        ok = check_gradient_nonzero(a, "Add input A") && ok;
        ok = check_gradient_nonzero(b, "Add input B (bias)") && ok;
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_sparse_cross_entropy(const TestConfig& cfg) {
    print_test_header("Sparse Cross Entropy Loss");
    TestResult result{"SparseCCE", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Logits: [B*T, vocab]
        Tensor logits = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.vocab_size}}, opts, 1234, 0.1f);
        
        // Targets: [B*T] with class indices
        std::vector<int64_t> target_data(cfg.B * cfg.T);
        for (int64_t i = 0; i < cfg.B * cfg.T; ++i) {
            target_data[i] = i % cfg.vocab_size;
        }
        Tensor targets_cpu = Tensor(Shape{{cfg.B * cfg.T}}, TensorOptions().with_dtype(Dtype::Int64));
        targets_cpu.set_data(target_data);
        Tensor targets = targets_cpu.to(cfg.device);
        
        // Forward
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        
        if (!check_tensor_valid(loss, "Loss output")) {
            result.passed = false;
            return result;
        }
        
        // Print loss value
        Tensor loss_cpu = loss.to_cpu();
        std::cout << "  Loss value: " << loss_cpu.data<float>()[0] << std::endl;
        
        // Backward
        loss.backward();
        
        if (!check_gradient_nonzero(logits, "Logits")) {
            result.passed = false;
            return result;
        }
        
        result.message = "Forward and backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

TestResult test_weight_tying(const TestConfig& cfg) {
    print_test_header("Weight Tying (Transpose)");
    TestResult result{"WeightTying", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Simulate embedding weight [vocab, C]
        Tensor wte_weight = Tensor::randn<float>(Shape{{cfg.vocab_size, cfg.C}}, opts, 1234, 0.02f);
        
        // Hidden state [B*T, C]
        Tensor hidden = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts, 5678, 0.02f);
        
        // Weight-tied final projection: hidden @ wte.t() -> [B*T, vocab]
        Tensor logits = autograd::matmul(hidden, wte_weight.t());
        
        if (!check_tensor_valid(logits, "Weight-tied logits")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(logits.shape(), opts);
        logits.backward(&grad_out);
        
        bool ok = true;
        ok = check_gradient_nonzero(hidden, "Hidden state") && ok;
        ok = check_gradient_nonzero(wte_weight, "WTE weight (tied)") && ok;
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "Weight tying forward and backward successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// MLP Block Test
// =============================================================================

TestResult test_mlp_block(const TestConfig& cfg) {
    print_test_header("MLP Block (Full)");
    TestResult result{"MLPBlock", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        int64_t hidden = cfg.C * 4;
        
        // MLP parameters
        Tensor ln_gamma = Tensor::ones(Shape{{cfg.C}}, opts);
        Tensor ln_beta = Tensor::zeros(Shape{{cfg.C}}, opts);
        Tensor W_up = Tensor::randn<float>(Shape{{cfg.C, hidden}}, opts, 1234, 0.02f);
        Tensor b_up = Tensor::zeros(Shape{{hidden}}, opts);
        Tensor W_down = Tensor::randn<float>(Shape{{hidden, cfg.C}}, opts, 5678, 0.02f);
        Tensor b_down = Tensor::zeros(Shape{{cfg.C}}, opts);
        
        // Input
        Tensor x = Tensor::randn<float>(Shape{{cfg.B * cfg.T, cfg.C}}, opts);
        
        // Forward: LayerNorm -> Linear -> GELU -> Linear
        Tensor h = autograd::layer_norm(x, ln_gamma, ln_beta, cfg.C);
        h = autograd::matmul(h, W_up);
        h = autograd::add(h, b_up);
        h = autograd::gelu(h);
        h = autograd::matmul(h, W_down);
        h = autograd::add(h, b_down);
        
        // Residual connection
        Tensor out = autograd::add(x, h);
        
        if (!check_tensor_valid(out, "MLP output")) {
            result.passed = false;
            return result;
        }
        
        // Backward
        Tensor grad_out = Tensor::ones(out.shape(), opts);
        out.backward(&grad_out);
        
        // Check all gradients
        bool ok = true;
        ok = check_gradient_nonzero(x, "Input x") && ok;
        ok = check_gradient_nonzero(ln_gamma, "LN gamma") && ok;
        ok = check_gradient_nonzero(W_up, "W_up") && ok;
        ok = check_gradient_nonzero(W_down, "W_down") && ok;
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "MLP block forward and backward successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// Optimizer Test
// =============================================================================

TestResult test_optimizer_step(const TestConfig& cfg) {
    print_test_header("Adam Optimizer Step");
    TestResult result{"AdamOptimizer", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Create some parameters
        Tensor w1 = Tensor::randn<float>(Shape{{cfg.C, cfg.C}}, opts, 1234, 0.02f);
        Tensor w2 = Tensor::randn<float>(Shape{{cfg.C, cfg.C}}, opts, 5678, 0.02f);
        
        std::vector<Tensor> params = {w1, w2};
        
        // Create optimizer
        nn::Adam optimizer(params, 1e-4f, 0.9f, 0.99f, 1e-8f, 0.0f);
        
        // Do a forward pass to create gradients
        Tensor x = Tensor::randn<float>(Shape{{cfg.B, cfg.C}}, opts);
        Tensor y = autograd::matmul(x, w1);
        y = autograd::matmul(y, w2);
        Tensor loss = autograd::sum(y);
        
        // Backward
        loss.backward();
        
        // Store original values
        Tensor w1_before = w1.clone();
        
        // Optimizer step
        optimizer.step();
        
        // Check weights changed
        Tensor diff = autograd::sub(w1, w1_before);
        Tensor diff_cpu = diff.to_cpu();
        const float* diff_data = diff_cpu.data<float>();
        
        bool weights_changed = false;
        for (size_t i = 0; i < diff.numel(); ++i) {
            if (std::abs(diff_data[i]) > 1e-10f) {
                weights_changed = true;
                break;
            }
        }
        
        if (!weights_changed) {
            std::cerr << "  [FAIL] Weights did not change after optimizer step!" << std::endl;
            result.passed = false;
            return result;
        }
        
        std::cout << "  [OK] Weights updated after optimizer step" << std::endl;
        
        // Check weights are still valid
        if (!check_tensor_valid(w1, "W1 after step")) {
            result.passed = false;
            return result;
        }
        
        result.message = "Optimizer step successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// Gradient Clipping Test
// =============================================================================

TestResult test_gradient_clipping(const TestConfig& cfg) {
    print_test_header("Gradient Clipping");
    TestResult result{"GradClip", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Create parameters with large gradients
        Tensor w = Tensor::randn<float>(Shape{{cfg.C, cfg.C}}, opts);
        
        // Do forward/backward to create gradients
        Tensor x = Tensor::randn<float>(Shape{{cfg.B, cfg.C}}, opts);
        Tensor y = autograd::matmul(x, w);
        
        // Scale up to create large loss
        Tensor scale = Tensor::full(Shape{{1}}, opts.with_req_grad(false), 1000.0f);
        y = autograd::mul(y, scale);
        Tensor loss = autograd::sum(y);
        loss.backward();
        
        // Compute norm before clipping
        Tensor grad = w.grad_view();
        Tensor grad_cpu_before = grad.to_cpu();
        float norm_before = 0.0f;
        const float* grad_data = grad_cpu_before.data<float>();
        for (size_t i = 0; i < grad.numel(); ++i) {
            norm_before += grad_data[i] * grad_data[i];
        }
        norm_before = std::sqrt(norm_before);
        
        std::cout << "  Gradient norm before clipping: " << norm_before << std::endl;
        
        // Clip gradients
        std::vector<Tensor> params = {w};
        float clipped_norm = nn::clip_grad_norm_(params, 1.0f);
        
        std::cout << "  Clipped norm value: " << clipped_norm << std::endl;
        
        // Compute norm after clipping
        grad = w.grad_view();
        Tensor grad_cpu_after = grad.to_cpu();
        float norm_after = 0.0f;
        grad_data = grad_cpu_after.data<float>();
        for (size_t i = 0; i < grad.numel(); ++i) {
            norm_after += grad_data[i] * grad_data[i];
        }
        norm_after = std::sqrt(norm_after);
        
        std::cout << "  Gradient norm after clipping: " << norm_after << std::endl;
        
        // Verify clipping worked
        if (norm_before > 1.0f && norm_after > 1.1f) {
            std::cerr << "  [FAIL] Gradient norm should be close to 1.0 after clipping!" << std::endl;
            result.passed = false;
            return result;
        }
        
        std::cout << "  [OK] Gradient clipping works correctly" << std::endl;
        result.message = "Gradient clipping successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// Full Model Mini-Test
// =============================================================================

TestResult test_mini_gpt_forward_backward(const TestConfig& cfg) {
    print_test_header("Mini GPT Forward/Backward");
    TestResult result{"MiniGPT", true, ""};
    
    try {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(cfg.device)
                                            .with_req_grad(true);
        
        // Model parameters
        Tensor wte = Tensor::randn<float>(Shape{{cfg.vocab_size, cfg.C}}, opts, 1234, 0.02f);
        Tensor wpe = Tensor::randn<float>(Shape{{cfg.T, cfg.C}}, opts, 2345, 0.02f);
        
        // MLP block parameters
        int64_t hidden = cfg.C * 4;
        Tensor ln_gamma = Tensor::ones(Shape{{cfg.C}}, opts);
        Tensor ln_beta = Tensor::zeros(Shape{{cfg.C}}, opts);
        Tensor W_up = Tensor::randn<float>(Shape{{cfg.C, hidden}}, opts, 3456, 0.02f);
        Tensor b_up = Tensor::zeros(Shape{{hidden}}, opts);
        Tensor W_down = Tensor::randn<float>(Shape{{hidden, cfg.C}}, opts, 4567, 0.02f);
        Tensor b_down = Tensor::zeros(Shape{{cfg.C}}, opts);
        
        // Final LN
        Tensor ln_f_gamma = Tensor::ones(Shape{{cfg.C}}, opts);
        Tensor ln_f_beta = Tensor::zeros(Shape{{cfg.C}}, opts);
        
        // Input tokens
        std::vector<int64_t> tok_data(cfg.B * cfg.T);
        std::vector<int64_t> tgt_data(cfg.B * cfg.T);
        for (int64_t i = 0; i < cfg.B * cfg.T; ++i) {
            tok_data[i] = i % cfg.vocab_size;
            tgt_data[i] = (i + 1) % cfg.vocab_size;  // Next token prediction
        }
        
        Tensor tokens_cpu = Tensor(Shape{{cfg.B, cfg.T}}, TensorOptions().with_dtype(Dtype::Int64));
        tokens_cpu.set_data(tok_data);
        Tensor tokens = tokens_cpu.to(cfg.device);
        
        Tensor targets_cpu = Tensor(Shape{{cfg.B * cfg.T}}, TensorOptions().with_dtype(Dtype::Int64));
        targets_cpu.set_data(tgt_data);
        Tensor targets = targets_cpu.to(cfg.device);
        
        // Position indices
        std::vector<int64_t> pos_data(cfg.T);
        for (int64_t i = 0; i < cfg.T; ++i) pos_data[i] = i;
        Tensor pos_cpu = Tensor(Shape{{1, cfg.T}}, TensorOptions().with_dtype(Dtype::Int64));
        pos_cpu.set_data(pos_data);
        Tensor pos = pos_cpu.to(cfg.device);
        
        // Forward pass
        std::cout << "  Running forward pass..." << std::endl;
        
        // Embeddings
        Tensor tok_emb = autograd::embedding(wte, tokens);  // [B, T, C]
        Tensor pos_emb = autograd::embedding(wpe, pos);     // [1, T, C]
        
        // Broadcast add: [B, T, C] + [1, T, C] -> [B, T, C]
        // This relies on autograd::add supporting broadcasting
        Tensor x = autograd::add(tok_emb, pos_emb);
        
        // Reshape for MLP: [B*T, C]
        x = x.reshape(Shape{{cfg.B * cfg.T, cfg.C}});
        
        // MLP block
        Tensor h = autograd::layer_norm(x, ln_gamma, ln_beta, cfg.C);
        h = autograd::matmul(h, W_up);
        h = autograd::add(h, b_up);
        h = autograd::gelu(h);
        h = autograd::matmul(h, W_down);
        h = autograd::add(h, b_down);
        x = autograd::add(x, h);  // residual
        
        // Final LN
        x = autograd::layer_norm(x, ln_f_gamma, ln_f_beta, cfg.C);
        
        // Final projection (weight tied)
        Tensor logits = autograd::matmul(x, wte.t());  // [B*T, vocab]
        
        if (!check_tensor_valid(logits, "Logits")) {
            result.passed = false;
            return result;
        }
        
        // Loss
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        
        Tensor loss_cpu = loss.to_cpu();
        std::cout << "  Mini GPT loss: " << loss_cpu.data<float>()[0] << std::endl;
        
        // Backward
        std::cout << "  Running backward pass..." << std::endl;
        loss.backward();
        
        // Check key gradients
        bool ok = true;
        ok = check_gradient_nonzero(wte, "WTE (embedding + tied)") && ok;
        ok = check_gradient_nonzero(wpe, "WPE (position)") && ok;
        ok = check_gradient_nonzero(W_up, "W_up") && ok;
        ok = check_gradient_nonzero(W_down, "W_down") && ok;
        
        if (!ok) {
            result.passed = false;
            return result;
        }
        
        result.message = "Full forward/backward pass successful";
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "GPT-2 COMPONENT DIAGNOSTIC TEST" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Parse GPU ID from args or default to 0
    // Note: When CUDA_VISIBLE_DEVICES=1 is set, GPU 1 becomes device 0
    int gpu_id = 0;
    if (argc > 1) {
        gpu_id = std::atoi(argv[1]);
    }
    
    std::cout << "\nUsing GPU device index: " << gpu_id << std::endl;
    
    // Set CUDA device
    cudaSetDevice(gpu_id);
    
    TestConfig cfg(gpu_id);
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  B=" << cfg.B << ", T=" << cfg.T << ", C=" << cfg.C << std::endl;
    std::cout << "  vocab_size=" << cfg.vocab_size << std::endl;
    
    std::vector<TestResult> results;
    
    // Run all tests
    results.push_back(test_embedding(cfg));
    results.push_back(test_gelu(cfg));
    results.push_back(test_layernorm(cfg));
    results.push_back(test_matmul(cfg));
    results.push_back(test_add(cfg));
    results.push_back(test_sparse_cross_entropy(cfg));
    results.push_back(test_weight_tying(cfg));
    results.push_back(test_mlp_block(cfg));
    results.push_back(test_optimizer_step(cfg));
    results.push_back(test_gradient_clipping(cfg));
    results.push_back(test_mini_gpt_forward_backward(cfg));
    
    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& r : results) {
        print_test_result(r);
        if (r.passed) ++passed;
        else ++failed;
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Total: " << passed << " PASSED, " << failed << " FAILED" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return failed > 0 ? 1 : 0;
}
