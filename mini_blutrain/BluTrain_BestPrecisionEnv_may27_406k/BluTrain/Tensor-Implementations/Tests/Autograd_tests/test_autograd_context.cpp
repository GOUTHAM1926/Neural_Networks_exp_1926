#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include "autograd/AutogradContext.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

bool near(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

void test_save_restore_cpu() {
    std::cout << "Testing Save/Restore on CPU..." << std::endl;
    
    AutogradContext ctx;
    
    // Create tensors
    Tensor t1 = Tensor::ones(Shape{{2, 2}});
    Tensor t2 = Tensor::full(Shape{{2, 2}}, TensorOptions(), 2.0f);
    
    // Save
    ctx.save_for_backward({t1, t2});
    
    // Verify count
    if (ctx.num_saved_variables() != 2) {
        std::cerr << "FAIL: num_saved_variables() is " << ctx.num_saved_variables() << ", expected 2" << std::endl;
        exit(1);
    }
    
    // Restore
    auto saved = ctx.get_saved_variables();
    
    if (saved.size() != 2) {
        std::cerr << "FAIL: get_saved_variables() returned " << saved.size() << " items, expected 2" << std::endl;
        exit(1);
    }
    
    // Check data equality
    Tensor r1 = saved[0];
    Tensor r2 = saved[1];
    
    if (std::abs(r1.data<float>()[0] - 1.0f) > 1e-4) {
        std::cerr << "FAIL: Restored tensor 1 data mismatch" << std::endl;
        exit(1);
    }
    
    if (std::abs(r2.data<float>()[0] - 2.0f) > 1e-4) {
        std::cerr << "FAIL: Restored tensor 2 data mismatch" << std::endl;
        exit(1);
    }
    std::cout << "PASS: Save/Restore CPU" << std::endl;
}

void test_save_restore_gpu() {
    if (!OwnTensor::device::cuda_available()) {
        std::cout << "SKIP: Save/Restore GPU (CUDA not available)" << std::endl;
        return;
    }
    
    std::cout << "Testing Save/Restore on GPU..." << std::endl;
    AutogradContext ctx;
    
    TensorOptions gpu_opts;
    gpu_opts = gpu_opts.with_device(DeviceIndex(Device::CUDA));
    
    Tensor t1 = Tensor::ones(Shape{{2, 2}}, gpu_opts);
    
    // Save
    ctx.save_for_backward({t1});
    
    // Restore
    auto saved = ctx.get_saved_variables();
    Tensor r1 = saved[0];
    
    // Verify it's still on GPU
    if (!r1.is_cuda()) {
        std::cerr << "FAIL: Restored tensor lost CUDA device assignment" << std::endl;
        exit(1);
    }
    
    // Check value
    Tensor cpu_copy = r1.to_cpu();
    if (std::abs(cpu_copy.data<float>()[0] - 1.0f) > 1e-4) {
        std::cerr << "FAIL: Restored GPU tensor data mismatch" << std::endl;
        exit(1);
    }
    std::cout << "PASS: Save/Restore GPU" << std::endl;
}

void test_dirty_tracking() {
    std::cout << "Testing Dirty Tracking..." << std::endl;
    
    AutogradContext ctx;
    Tensor t1 = Tensor::ones(Shape{{1}});
    Tensor t2 = Tensor::ones(Shape{{1}});
    
    // Initially not dirty
    if (ctx.is_dirty(t1)) {
        std::cerr << "FAIL: New tensor reported as dirty" << std::endl;
        exit(1);
    }
    
    // Mark dirty
    ctx.mark_dirty({t1});
    
    if (!ctx.is_dirty(t1)) {
        std::cerr << "FAIL: Marked tensor not reported as dirty" << std::endl;
        exit(1);
    }
    
    if (ctx.is_dirty(t2)) {
        std::cerr << "FAIL: Unmarked tensor reported as dirty" << std::endl;
        exit(1);
    }
    std::cout << "PASS: Dirty Tracking" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing Edge Cases..." << std::endl;
    AutogradContext ctx;
    
    // 1. Empty save
    ctx.save_for_backward({});
    if (ctx.num_saved_variables() != 0) {
        std::cerr << "FAIL: Saving empty list resulted in non-empty storage" << std::endl;
        exit(1);
    }
    
    // 2. Overwrite save
    Tensor t1 = Tensor::ones(Shape{{1}});
    ctx.save_for_backward({t1});
    
    Tensor t2 = Tensor::ones(Shape{{1}});
    Tensor t3 = Tensor::ones(Shape{{1}});
    ctx.save_for_backward({t2, t3});
    
    if (ctx.num_saved_variables() != 2) {
         std::cerr << "FAIL: Overwrite did not update size correctly" << std::endl;
         exit(1);
    }
    
    // 3. Release
    ctx.release_variables();
    if (ctx.num_saved_variables() != 0) {
        std::cerr << "FAIL: Release did not clear variables" << std::endl;
        exit(1);
    }
    
    std::cout << "PASS: Edge Cases" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running AutogradContext Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_save_restore_cpu();
    test_save_restore_gpu();
    test_dirty_tracking();
    test_edge_cases();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All AutogradContext functional tests PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
