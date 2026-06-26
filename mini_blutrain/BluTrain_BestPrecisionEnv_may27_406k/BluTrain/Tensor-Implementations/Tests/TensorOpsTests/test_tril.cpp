#include <iostream>
#include <vector>
#include <cassert>
#include "TensorLib.h"

using namespace OwnTensor;

void test_tril_cpu() {
    std::cout << "\nRunning test_tril_cpu..." << std::endl;

    // 3x3 matrix
    Shape shape({3, 3});
    Tensor t = Tensor::ones(shape, TensorOptions().with_dtype(Dtype::Float32));
    
    // Set some values to distinguish
    float data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    t.set_data(data, 9);

    // tril with diagonal = 0
    Tensor t0 = tril(t, 0);
    float expected0[] = {
        1.0, 0.0, 0.0,
        4.0, 5.0, 0.0,
        7.0, 8.0, 9.0
    };
    const float* ptr0 = t0.data<float>();
    for (int i = 0; i < 9; ++i) {
        assert(ptr0[i] == expected0[i]);
    }
    std::cout << "\ndiagonal = 0\n";
    t0.display();
    std::cout << "tril(diagonal=0)" << "\033[32m" << " passed" << "\033[0m" << std::endl;

    // tril with diagonal = 1
    Tensor t1 = tril(t, 1);
    float expected1[] = {
        1.0, 2.0, 0.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    const float* ptr1 = t1.data<float>();
    for (int i = 0; i < 9; ++i) {
        assert(ptr1[i] == expected1[i]);
    }
    std::cout << "\ndiagonal = 1\n";
    t1.display();
    std::cout << "tril(diagonal=1)" << "\033[32m" << " passed" << "\033[0m" << std::endl;

    // tril with diagonal = -1
    Tensor tm1 = tril(t, -1);
    float expected_m1[] = {
        0.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
        7.0, 8.0, 0.0
    };
    for (int i = 0; i < 9; ++i) {
        if (tm1.data<float>()[i] != expected_m1[i]) throw std::runtime_error("tril(diagonal=-1) failed");
    }
    std::cout << "\ndiagonal = -1\n";
    tm1.display();
    std::cout << "tril(diagonal=-1) passed" << std::endl;

    // Test custom value
    Tensor t_val = tril(t, 0, 5.0);
    float expected_val[] = {
        1, 5, 5,
        4, 5, 5,
        7, 8, 9
    };
    for (int i = 0; i < 9; ++i) {
        if (t_val.data<float>()[i] != expected_val[i]) throw std::runtime_error("tril(value=5.0) failed");
    }
    std::cout << "\nvalue = 5.0\n";
    t_val.display();
    std::cout << "tril(value=5.0) passed" << std::endl;

    // Test infinity
    double inf = std::numeric_limits<double>::infinity();
    Tensor t_inf = tril(t, 0, -inf);
    for (int i = 0; i < 9; ++i) {
        size_t r = i / 3;
        size_t c = i % 3;
        if (c > r) {
            if (!std::isinf(t_inf.data<float>()[i]) || t_inf.data<float>()[i] > 0) 
                throw std::runtime_error("tril(value=-inf) failed at index " + std::to_string(i));
        }
    }
    std::cout << "\nvalue = inf\n";
    t_inf.display();
    std::cout << "tril(value=-inf) passed" << std::endl;
    std::cout << "check for softmax correctness...\n";
    Tensor softmaxRes = autograd::softmax(t_inf);
    softmaxRes.display();
}

void test_tril_cuda() {
    #ifdef WITH_CUDA
    std::cout << "\nRunning test_tril_cuda..." << std::endl;

    Shape shape({3, 3});
    Tensor t = Tensor::ones(shape, TensorOptions().with_device(DeviceIndex(Device::CUDA)));
    
    float h_data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    t.to_cpu_(); // Copy to CPU temporarily to set data easily, if set_data doesn't handle CUDA directly
    t.set_data(h_data, 9);
    t.to_cuda_();

    Tensor t0 = tril(t, 0);
    Tensor cpu_t0 = t0.to_cpu();
    float expected0[] = {
        1, 0, 0,
        4, 5, 0,
        7, 8, 9
    };
    for (int i = 0; i < 9; ++i) {
        if (cpu_t0.data<float>()[i] != expected0[i]) throw std::runtime_error("CUDA tril(diagonal=0) failed");
    }
    std::cout << "diagonal = 0\n";
    t0.display();
    std::cout << "CUDA tril(diagonal=0) passed" << std::endl;

    // Test infinity on CUDA
    double inf = std::numeric_limits<double>::infinity();
    Tensor t_inf = tril(t, 0, -inf);
    Tensor cpu_inf = t_inf.to_cpu();
    for (int i = 0; i < 9; ++i) {
        size_t r = i / 3;
        size_t c = i % 3;
        if (c > r) {
            if (!std::isinf(cpu_inf.data<float>()[i]) || cpu_inf.data<float>()[i] > 0) 
                throw std::runtime_error("CUDA tril(value=-inf) failed at index " + std::to_string(i));
        }
    }
    std::cout << "\nvalue = inf\n";
    t_inf.display();
    std::cout << "CUDA tril(value=-inf) passed" << std::endl;

    std::cout << "just check for softmax correctness...\n";
    Tensor softmaxRes = autograd::softmax(t_inf);
    softmaxRes.display();
    #endif
}

void test_tril_batch() {
    std::cout << "\nRunning test_tril_batch..." << std::endl;
    Shape shape({2, 2, 2});
    Tensor t = Tensor::empty(shape);
    float data[] = {
        1.0, 2.0,
        3.0, 4.0,
        
        5.0, 6.0,
        7.0, 8.0
    };
    t.set_data(data, 8);

    Tensor t0 = tril(t, 0);
    float expected0[] = {
        1.0, 0.0,
        3.0, 4.0,
        
        5.0, 0.0,
        7.0, 8.0
    };
    const float* ptr0 = t0.data<float>();
    for (int i = 0; i < 8; ++i) {
        assert(ptr0[i] == expected0[i]);
    }
    std::cout << "\nbatch test\n";
    t0.display();
    std::cout << "tril batch" << "\033[32m" << " passed" << "\033[0m" << std::endl;
}

void test_tril_error() {
    std::cout << "\nRunning test_tril_error..." << std::endl;
    Shape shape({5});
    Tensor t = Tensor::ones(shape);
    try {
        tril(t, 0);
        assert(false && "Should have thrown error");
    } catch (const std::runtime_error& e) {
        std::cout << "\033[031m" << "Caught expected error: " << "\033[0m" << e.what() << std::endl;
    }
}

int main() {
    test_tril_cpu();
    test_tril_batch();
    test_tril_error();
    #ifdef WITH_CUDA
    test_tril_cuda();
    #endif
    std::cout << "\033[32m" << "\nAll tril tests passed!" << "\033[0m" << std::endl;
    return 0;
}
