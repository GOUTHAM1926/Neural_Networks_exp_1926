#include "core/Tensor.h"
#include "nn/NN.h"
#include "autograd/operations/ReductionOps.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

using namespace OwnTensor;
using namespace OwnTensor::nn;

bool near(float a, float b, float tol = 1e-5) {
    return std::abs(a - b) < tol;
}

void test_cpu() {
    std::cout << "\n--- Testing Embedding Layer Gradients (CPU) ---\n";
    const int num_embeddings = 10;
    const int embedding_dim = 4;
    const int padding_idx = 1;
    
    Embedding emb(num_embeddings, embedding_dim, padding_idx);
    Tensor indices(Shape{{3}}, Dtype::UInt16, DeviceIndex(Device::CPU));
    std::vector<uint16_t> idx_vals = {0, 2, 1};
    indices.set_data(idx_vals);
    
    Tensor output = emb.forward(indices);
    Tensor loss = autograd::mean(output);
    loss.backward();

    assert(emb.weight.grad() != nullptr);
    float* g_ptr = emb.weight.grad<float>();
    float expected_grad = 1.0f / 12.0f;
    for(int i=0; i<num_embeddings; ++i) {
        for(int j=0; j<embedding_dim; ++j) {
            float g = g_ptr[i * embedding_dim + j];
            if (i == 0 || i == 2) assert(near(g, expected_grad));
            else assert(near(g, 0.0f));
        }
    }
    std::cout << "PASS: CPU Gradients verified.\n";
}

void test_cuda() {
    std::cout << "\n--- Testing Embedding Layer Gradients (CUDA) ---\n";
    const int num_embeddings = 10;
    const int embedding_dim = 4;
    const int padding_idx = 1;
    
    Embedding emb(num_embeddings, embedding_dim, padding_idx);
    
    // Move weights to CUDA
    emb.weight = emb.weight.to(DeviceIndex(Device::CUDA));
    
    Tensor indices(Shape{{3}}, Dtype::UInt16, DeviceIndex(Device::CUDA));
    std::vector<uint16_t> idx_vals = {0, 2, 1};
    indices.set_data(idx_vals);
    
    Tensor output = emb.forward(indices);
    Tensor loss = autograd::mean(output);
    loss.backward();

    assert(emb.weight.grad() != nullptr);
    
    Tensor grad_cpu = emb.weight.grad_view().to_cpu();
    float* g_ptr = grad_cpu.data<float>();
    float expected_grad = 1.0f / 12.0f;
    for(int i=0; i<num_embeddings; ++i) {
        for(int j=0; j<embedding_dim; ++j) {
            float g = g_ptr[i * embedding_dim + j];
            if (i == 0 || i == 2) assert(near(g, expected_grad));
            else assert(near(g, 0.0f));
        }
    }
    std::cout << "PASS: CUDA Gradients verified.\n";
}

int main() {
    try {
        test_cpu();
        test_cuda();
        std::cout << "\n✅ All Embedding tests (CPU & CUDA) passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}   