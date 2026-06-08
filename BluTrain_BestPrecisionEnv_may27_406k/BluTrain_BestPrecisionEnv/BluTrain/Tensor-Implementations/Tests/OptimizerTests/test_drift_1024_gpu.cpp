/**
 * @file test_drift_1024_gpu.cpp
 * @brief GPU-only long-horizon drift test: 500 steps, 1024x1024 tensors, non-uniform gradients.
 *
 * Tests Adam and AdamW on GPU (foreach path) over 500 steps with 1M-element tensors.
 * Uses a simple LCG PRNG for gradient generation, matching the Python companion.
 * Writes sampled parameter values (first 16 + last 16 elements) to CSV.
 *
 * Build & run:
 *   make run-snippet FILE=Tests/OptimizerTests/test_drift_1024_gpu.cpp
 *
 * Then compare:
 *   cd Tests/OptimizerTests && python3 test_drift_1024_gpu.py
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <array>

#include "TensorLib.h"

using namespace OwnTensor;

// ============================================================================
// Configuration
// ============================================================================
static const int NUM_STEPS = 5000;
static const int ROWS = 1024;
static const int COLS = 1024;
static const int NUMEL = ROWS * COLS;  // 1,048,576

// Sample indices: first 16 + last 16 elements for CSV output
static const int NUM_SAMPLES = 32;

static std::array<int, NUM_SAMPLES> get_sample_indices() {
    std::array<int, NUM_SAMPLES> idx;
    for (int i = 0; i < 16; ++i) {
        idx[i] = i;                         // first 16: 0..15
        idx[i + 16] = NUMEL - 16 + i;      // last 16
    }
    return idx;
}

// ============================================================================
// Deterministic LCG PRNG — must match Python exactly
// ============================================================================
static float lcg_float(uint64_t seed) {
    uint64_t state = seed;
    state = (6364136223846793005ULL * state + 1442695040888963407ULL);
    state = (6364136223846793005ULL * state + 1442695040888963407ULL);
    uint32_t bits = static_cast<uint32_t>(state >> 33);
    return (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFFU)) * 2.0f - 1.0f;
}


// Generate a 1024x1024 gradient tensor on CPU, then move to GPU
static Tensor make_gradient(int step) {
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);
    Tensor grad = Tensor::zeros(Shape{{ROWS, COLS}}, opts);
    float* data = grad.data<float>();
    for (int i = 0; i < NUMEL; ++i) {
        uint64_t seed = static_cast<uint64_t>(step) * 2000000ULL + static_cast<uint64_t>(i);
        data[i] = lcg_float(seed);
    }
    return grad.to_cuda(0);
}


// ============================================================================
// CSV utilities — only write sampled elements
// ============================================================================
static void write_csv_header(std::ofstream& f, const std::array<int, NUM_SAMPLES>& indices) {
    f << "step";
    for (int idx : indices) {
        f << ",p" << idx;
    }
    f << "\n";
}

static void write_csv_row(std::ofstream& f, int step, const Tensor& t,
                           const std::array<int, NUM_SAMPLES>& indices) {
    Tensor tc = t.to_cpu();
    const float* d = tc.data<float>();
    f << step;
    for (int idx : indices) {
        f << "," << std::fixed << std::setprecision(8) << d[idx];
    }
    f << "\n";
}


// ============================================================================
// Initial weights
// ============================================================================
static Tensor make_initial_weights() {
    TensorOptions opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_req_grad(true);
    Tensor W = Tensor::zeros(Shape{{ROWS, COLS}}, opts);
    float* data = W.data<float>();
    for (int i = 0; i < NUMEL; ++i) {
        data[i] = 0.1f * static_cast<float>(i + 1) / static_cast<float>(NUMEL);
    }
    W.set_requires_grad(true);
    W = W.to_cuda(0);
    W.set_requires_grad(true);
    return W;
}


// ============================================================================
// Test runners
// ============================================================================
void test_adam_gpu(const std::string& csv_path) {
    std::cout << "\n=== Adam (GPU) — " << NUM_STEPS
              << " steps, [" << ROWS << "x" << COLS << "] ===" << std::endl;

    Tensor W = make_initial_weights();
    auto indices = get_sample_indices();

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    nn::Adam adam({W}, lr, beta1, beta2, eps, wd);

    std::ofstream csv(csv_path);
    write_csv_header(csv, indices);
    write_csv_row(csv, 0, W, indices);

    for (int step = 1; step <= NUM_STEPS; ++step) {
        Tensor grad = make_gradient(step);
        W.set_grad(grad);
        adam.step();
        write_csv_row(csv, step, W, indices);
    }
    csv.close();
    std::cout << "  Wrote " << csv_path << std::endl;
}


void test_adamw_gpu(const std::string& csv_path) {
    std::cout << "\n=== AdamW (GPU) — " << NUM_STEPS
              << " steps, [" << ROWS << "x" << COLS << "] ===" << std::endl;

    Tensor W = make_initial_weights();
    auto indices = get_sample_indices();

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    nn::AdamW adamw({W}, lr, beta1, beta2, eps, wd);

    std::ofstream csv(csv_path);
    write_csv_header(csv, indices);
    write_csv_row(csv, 0, W, indices);

    for (int step = 1; step <= NUM_STEPS; ++step) {
        Tensor grad = make_gradient(step);
        W.set_grad(grad);
        adamw.step();
        write_csv_row(csv, step, W, indices);
    }
    csv.close();
    std::cout << "  Wrote " << csv_path << std::endl;
}


int main() {
    std::string base = "Tests/OptimizerTests/";

    std::cout << "================================================================" << std::endl;
    std::cout << "  GPU Drift Test — 500 steps, 1024x1024 tensors" << std::endl;
    std::cout << "  Non-uniform gradients, sampling 32 elements for CSV" << std::endl;
    std::cout << "================================================================" << std::endl;

    test_adam_gpu(base + "drift_1024_adam_gpu.csv");
    test_adamw_gpu(base + "drift_1024_adamw_gpu.csv");

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Done. Now run:" << std::endl;
    std::cout << "    cd Tests/OptimizerTests" << std::endl;
    std::cout << "    python3 test_drift_1024_gpu.py" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
