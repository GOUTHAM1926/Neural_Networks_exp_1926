#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "core/Tensor.h"
#include "core/TensorDataManip.h"
#include "device/DeviceCore.h"
#include "ops/Kernels.h"
#include "autograd/AutogradOps.h"
#include "autograd/Engine.h"

using namespace OwnTensor;

bool near(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

void test_cuda_matmul_basic() {
    std::cout << "\n--- Testing Basic CUDA Matmul ---\n";
    
    const int M = 2, N = 3, P = 2;
    // A: [2, 3], B: [3, 2] -> C: [2, 2]
    
    Tensor A(Shape{{M, N}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor B(Shape{{N, P}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    A.set_data(a_data);
    B.set_data(b_data);
    
    // autograd::matmul internally calls OwnTensor::matmul
    Tensor C = autograd::matmul(A, B);
    
    Tensor c_cpu = C.to_cpu();
    float* c_ptr = c_cpu.data<float>();
    
    // Expected:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [7+18+33, 8+20+36] = [58, 64]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [28+45+66, 32+50+72] = [139, 154]
    
    std::cout << "Result matrix C:\n";
    c_cpu.display();
    
    assert(near(c_ptr[0], 58.0f));
    assert(near(c_ptr[1], 64.0f));
    assert(near(c_ptr[2], 139.0f));
    assert(near(c_ptr[3], 154.0f));
    
    std::cout << "PASS: Basic CUDA Matmul verified.\n";
}

void test_cuda_matmul_broadcast() {
    std::cout << "\n--- Testing CUDA Matmul with Broadcasting ---\n";
    
    // A: [2, 2, 3], B: [3, 2] -> C: [2, 2, 2]
    Tensor A(Shape{{2, 2, 3}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor B(Shape{{3, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    
    std::vector<float> a_data(12, 1.0f); // All ones
    std::vector<float> b_data(6, 1.0f);  // All ones
    
    A.set_data(a_data);
    B.set_data(b_data);
    
    Tensor C = autograd::matmul(A, B);
    
    Tensor c_cpu = C.to_cpu();
    float* c_ptr = c_cpu.data<float>();
    
    // Each element should be 1*1 + 1*1 + 1*1 = 3.0
    for(size_t i=0; i<C.numel(); ++i) {
        assert(near(c_ptr[i], 3.0f));
    }
    
    std::cout << "PASS: CUDA Matmul Broadcasting verified.\n";
}

void test_cuda_arithmetic() {
    std::cout << "\n--- Testing CUDA Basic Arithmetic ---\n";
    Tensor A(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor B(Shape{{2, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    
    A.fill(2.0f);
    B.fill(3.0f);
    
    Tensor C = autograd::add(A, B);
    Tensor c_cpu = C.to_cpu();
    assert(near(c_cpu.data<float>()[0], 5.0f));
    std::cout << "PASS: CUDA Add verified.\n";
    
    Tensor D = autograd::mul(A, B);
    Tensor d_cpu = D.to_cpu();
    assert(near(d_cpu.data<float>()[0], 6.0f));
    std::cout << "PASS: CUDA Mul verified.\n";
}

void test_cuda_matmul_bf16_broadcast() {
    std::cout << "\n--- Testing CUDA BFloat16 Matmul with Broadcasting ---\n";
    
    // A: [2, 2, 3], B: [3, 2] -> C: [2, 2, 2]
    Tensor A(Shape{{2, 2, 3}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA));
    Tensor B(Shape{{3, 2}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA));
    
    std::vector<bfloat16_t> a_data(12, bfloat16_t(1.0f));
    std::vector<bfloat16_t> b_data(6, bfloat16_t(1.0f));
    
    A.set_data(a_data);
    B.set_data(b_data);
    
    Tensor C = autograd::matmul(A, B);
    
    Tensor c_cpu = C.to_cpu();
    bfloat16_t* c_ptr = c_cpu.data<bfloat16_t>();
    
    for(size_t i=0; i<C.numel(); ++i) {
        if (!near(static_cast<float>(c_ptr[i]), 3.0f)) {
            std::cout << "FAILURE at index " << i << ": expected 3.0, got " << static_cast<float>(c_ptr[i]) << std::endl;
            assert(false);
        }
    }
    
    std::cout << "PASS: CUDA BFloat16 Matmul Broadcasting verified.\n";
}

void test_cuda_matmul_rank_mismatch() {
    std::cout << "\n--- Testing CUDA Matmul Rank-Mismatch Broadcasting (A:[5,3,2], B:[2,5,2,4]) ---\n";
    
    // A: [5, 3, 2], B: [2, 5, 2, 4] -> C: [2, 5, 3, 4]
    Tensor A(Shape{{5, 3, 2}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    Tensor B(Shape{{2, 5, 2, 4}}, Dtype::Float32, DeviceIndex(Device::CUDA));
    
    A.fill(1.0f);
    B.fill(1.0f);
    
    Tensor C = autograd::matmul(A, B);
    assert((C.shape() == Shape{{2, 5, 3, 4}}));
    
    Tensor c_cpu = C.to_cpu();
    float* c_ptr = c_cpu.data<float>();
    for(size_t i=0; i<C.numel(); ++i) {
        assert(near(c_ptr[i], 2.0f)); // 1*1 + 1*1 = 2
    }
    std::cout << "PASS: CUDA F32 Rank-Mismatch Broadcasting verified.\n";

    std::cout << "--- Testing CUDA BF16 Rank-Mismatch Broadcasting (Sensitive) ---\n";
    // A: [2, 3, 2], B: [2, 2, 2, 2] -> C: [2, 2, 3, 2]
    Tensor Abf(Shape{{2, 3, 2}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA));
    Tensor Bbf(Shape{{2, 2, 2, 2}}, Dtype::Bfloat16, DeviceIndex(Device::CUDA));
    
    std::vector<bfloat16_t> a_vals = {
        bfloat16_t(1.0f), bfloat16_t(1.0f), bfloat16_t(1.0f), bfloat16_t(1.0f), bfloat16_t(1.0f), bfloat16_t(1.0f), // Batch 1
        bfloat16_t(2.0f), bfloat16_t(2.0f), bfloat16_t(2.0f), bfloat16_t(2.0f), bfloat16_t(2.0f), bfloat16_t(2.0f)  // Batch 2
    };
    Abf.set_data(a_vals);
    Bbf.fill(bfloat16_t(1.0f));
    
    Tensor Cbf = autograd::matmul(Abf, Bbf);
    Tensor cbf_cpu = Cbf.to_cpu();
    bfloat16_t* cbf_ptr = cbf_cpu.data<bfloat16_t>();
    
    // Expected result[i, j, k, l] = 2 * A_val[j] * 1 = 2 or 4
    // Batch (0,0) -> 2.0
    // Batch (0,1) -> 4.0
    // Batch (1,0) -> 2.0
    // Batch (1,1) -> 4.0
    
    bool failed = false;
    for(int b=0; b<2; ++b) {
        for(int t=0; t<2; ++t) {
            float expected = (t == 0) ? 2.0f : 4.0f;
            for(int i=0; i<6; ++i) {
                float val = static_cast<float>(cbf_ptr[b*12 + t*6 + i]);
                if (!near(val, expected)) {
                    std::cout << "FAILURE at Batch (" << b << "," << t << "): expected " << expected << ", got " << val << std::endl;
                    failed = true;
                    break;
                }
            }
            if (failed) break;
        }
        if (failed) break;
    }
    
    if (failed) {
        std::cout << "CONFIRMED: BF16 Rank-Mismatch Broadcasting is BROKEN (Left-aligned indexing).\n";
    } else {
        std::cout << "PASS: CUDA BF16 Rank-Mismatch Broadcasting verified.\n";
    }
}

int main() {
    try {
        if (!OwnTensor::device::cuda_available()) {
            std::cout << "CUDA not available, skipping tests.\n";
            return 0;
        }
        test_cuda_matmul_basic();
        test_cuda_matmul_broadcast();
        test_cuda_matmul_bf16_broadcast();
        test_cuda_matmul_rank_mismatch();
        test_cuda_arithmetic();
        std::cout << "\n✅ CUDA All-Ops tests finished.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}