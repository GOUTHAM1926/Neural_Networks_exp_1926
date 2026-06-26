/**
 * @file test_long_horizon_drift.cpp
 * @brief Long-horizon drift test: 500 steps with non-uniform gradients.
 *
 * Tests Adam, AdamW, and SGD (momentum + weight_decay) over 500 steps.
 * Generates a different random gradient each step using a simple LCG PRNG
 * that is replicated identically in the companion Python script.
 *
 * Outputs CSV files with per-step parameter values for automated comparison.
 *
 * Build & run:
 *   make run-snippet FILE=Tests/OptimizerTests/test_long_horizon_drift.cpp
 *
 * Then run the Python companion:
 *   cd Tests/OptimizerTests && python3 test_long_horizon_drift.py
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#include "TensorLib.h"

using namespace OwnTensor;

// ============================================================================
// Deterministic LCG PRNG — must match Python exactly
// ============================================================================
// Returns a float in [-1.0, 1.0) given a seed.
// Uses the Numerical Recipes LCG: next = (a * seed + c) % m
static float lcg_float(uint64_t seed) {
    // Mix the seed through a few LCG rounds for better distribution
    uint64_t state = seed;
    state = (6364136223846793005ULL * state + 1442695040888963407ULL);
    state = (6364136223846793005ULL * state + 1442695040888963407ULL);
    // Map to [-1.0, 1.0)
    uint32_t bits = static_cast<uint32_t>(state >> 33);  // top 31 bits
    return (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFFU)) * 2.0f - 1.0f;
}


// Generate a gradient tensor with deterministic non-uniform values
static Tensor make_gradient(int step, int numel, bool use_gpu) {
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor grad = Tensor::zeros(Shape{{4, 4}}, opts);
    float* data = grad.data<float>();
    for (int i = 0; i < numel; ++i) {
        uint64_t seed = static_cast<uint64_t>(step) * 1000ULL + static_cast<uint64_t>(i);
        data[i] = lcg_float(seed);
    }
    if (use_gpu) {
        grad = grad.to_cuda(0);
    }
    return grad;
}


// ============================================================================
// Write parameter values to CSV at each step
// ============================================================================
static void write_csv_header(std::ofstream& f, int numel) {
    f << "step";
    for (int i = 0; i < numel; ++i) {
        f << ",p" << i;
    }
    f << "\n";
}

static void write_csv_row(std::ofstream& f, int step, const Tensor& t, int numel) {
    Tensor tc = t.to_cpu();
    const float* d = tc.data<float>();
    f << step;
    for (int i = 0; i < numel; ++i) {
        f << "," << std::fixed << std::setprecision(8) << d[i];
    }
    f << "\n";
}


// ============================================================================
// Test runners
// ============================================================================
static const int NUM_STEPS = 5000;
static const int NUMEL = 16;  // 4x4

// Initial weights: [0.1, 0.2, ..., 1.6]
static Tensor make_initial_weights(bool use_gpu) {
    TensorOptions opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_req_grad(true);
    Tensor W = Tensor::zeros(Shape{{4, 4}}, opts);
    float* data = W.data<float>();
    for (int i = 0; i < NUMEL; ++i) {
        data[i] = 0.1f * (i + 1);
    }
    W.set_requires_grad(true);
    if (use_gpu) {
        W = W.to_cuda(0);
        W.set_requires_grad(true);
    }
    return W;
}


void test_adam(bool use_gpu, const std::string& csv_path) {
    std::string device = use_gpu ? "GPU" : "CPU";
    std::cout << "\n=== Adam (" << device << ") — " << NUM_STEPS << " steps ===" << std::endl;

    Tensor W = make_initial_weights(use_gpu);

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    nn::Adam adam({W}, lr, beta1, beta2, eps, wd);

    std::ofstream csv(csv_path);
    write_csv_header(csv, NUMEL);
    // Write initial state as step 0
    write_csv_row(csv, 0, W, NUMEL);

    for (int step = 1; step <= NUM_STEPS; ++step) {
        Tensor grad = make_gradient(step, NUMEL, use_gpu);
        W.set_grad(grad);
        adam.step();
        write_csv_row(csv, step, W, NUMEL);
    }
    csv.close();
    std::cout << "  Wrote " << csv_path << std::endl;
}


void test_adamw(bool use_gpu, const std::string& csv_path) {
    std::string device = use_gpu ? "GPU" : "CPU";
    std::cout << "\n=== AdamW (" << device << ") — " << NUM_STEPS << " steps ===" << std::endl;

    Tensor W = make_initial_weights(use_gpu);

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    nn::AdamW adamw({W}, lr, beta1, beta2, eps, wd);

    std::ofstream csv(csv_path);
    write_csv_header(csv, NUMEL);
    write_csv_row(csv, 0, W, NUMEL);

    for (int step = 1; step <= NUM_STEPS; ++step) {
        Tensor grad = make_gradient(step, NUMEL, use_gpu);
        W.set_grad(grad);
        adamw.step();
        write_csv_row(csv, step, W, NUMEL);
    }
    csv.close();
    std::cout << "  Wrote " << csv_path << std::endl;
}


void test_sgd(bool use_gpu, const std::string& csv_path) {
    std::string device = use_gpu ? "GPU" : "CPU";
    std::cout << "\n=== SGD+Momentum+WD (" << device << ") — " << NUM_STEPS << " steps ===" << std::endl;

    Tensor W = make_initial_weights(use_gpu);

    float lr = 0.01f, momentum = 0.9f, wd = 0.01f;
    nn::SGDOptimizer sgd({W}, lr, momentum, wd);

    std::ofstream csv(csv_path);
    write_csv_header(csv, NUMEL);
    write_csv_row(csv, 0, W, NUMEL);

    for (int step = 1; step <= NUM_STEPS; ++step) {
        Tensor grad = make_gradient(step, NUMEL, use_gpu);
        W.set_grad(grad);
        sgd.step();
        write_csv_row(csv, step, W, NUMEL);
    }
    csv.close();
    std::cout << "  Wrote " << csv_path << std::endl;
}


int main() {
    std::string base = "Tests/OptimizerTests/";

    std::cout << "================================================================" << std::endl;
    std::cout << "  Long-Horizon Drift Test — 500 steps, non-uniform gradients" << std::endl;
    std::cout << "================================================================" << std::endl;

    // CPU tests
    test_adam(false,  base + "drift_adam_cpu.csv");
    test_adamw(false, base + "drift_adamw_cpu.csv");
    test_sgd(false,   base + "drift_sgd_cpu.csv");

    // GPU tests
    test_adam(true,   base + "drift_adam_gpu.csv");
    test_adamw(true,  base + "drift_adamw_gpu.csv");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Done. Now run:" << std::endl;
    std::cout << "    cd Tests/OptimizerTests" << std::endl;
    std::cout << "    python3 test_long_horizon_drift.py" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
