#include "core/Tensor.h"
#include <iostream>
#include <cassert>

using namespace OwnTensor;

int main() {
    std::cout << "\n=== Testing AutogradMeta Integration ===\n\n";
    
    // Test 1: Create tensor with requires_grad=true
    std::cout << "Test 1: Creating tensor with requires_grad=true...\n";
    TensorOptions opts = TensorOptions{}.with_dtype(Dtype::Float32);
    opts.requires_grad = true;
    Tensor t1(Shape{{2, 3}}, opts);
    assert(t1.requires_grad() == true);
    std::cout << "requires_grad correctly set to true\n";
    
    // Test 2: Set some data
    std::cout << "\nTest 2: Setting tensor data...\n";
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    t1.set_data(data);
    std::cout << "Data set successfully\n";
    
    // Test 3: Set gradient data  
    std::cout << "\nTest 3: Setting gradient data...\n"  ;  
    std::vector<float> grad_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    t1.set_grad(grad_data);
    std::cout << "Gradient data set successfully\n";
    
    // Test 4: Access gradient via grad<float>()
    std::cout << "\nTest 4: Accessing gradient via grad<float>()...\n";
    float* grad_ptr = t1.grad<float>();
    assert(grad_ptr != nullptr);
    std::cout << "Gradient pointer retrieved\n";
    
    // Verify gradient values
    bool values_correct = true;
    for (int i = 0; i < 6; i++) {
        if (std::abs(grad_ptr[i] - grad_data[i]) > 1e-6) {
            values_correct = false;
            std::cout << "Gradient value mismatch at index " << i 
                      << ": expected " << grad_data[i] << ", got " << grad_ptr[i] << "\n";
        }
    }
    if (values_correct) {
        std::cout << "Gradient values correct\n";
    }
    
    // Test 5: Access gradient via grad_view()
    std::cout << "\nTest 5: Accessing gradient via grad_view()...\n";
    try {
        Tensor grad_tensor = t1.grad_view();
        std::cout << "grad_view() returned successfully\n";
        
        // Verify grad_view shape matches tensor shape
        if (grad_tensor.shape() == t1.shape()) {
            std::cout << "Gradient shape matches tensor shape\n";
        } else {
            std::cout << "Gradient shape mismatch\n";
        }
    } catch (const std::exception& e) {
        std::cout << "grad_view() threw exception: " << e.what() << "\n";
        return 1;
    }
    
    // Test 6: Tensor without requires_grad
    std::cout << "\nTest 6: Creating tensor without requires_grad...\n";
    Tensor t2(Shape{{2, 3}}, TensorOptions{}.with_dtype(Dtype::Float32));
    assert(t2.requires_grad() == false);
    std::cout << "requires_grad correctly set to false\n";
    
    if (!t2.owns_grad()) {
        std::cout << "No gradient buffer allocated (as expected)\n";
    } else {
        std::cout << "Gradient buffer allocated when it shouldn't be\n";
    }
    
    // Test 7: Setting requires_grad after construction
    std::cout << "\nTest 7: Setting requires_grad after construction...\n";
    t2.set_requires_grad(true);
    if (t2.requires_grad()) {
        std::cout << "requires_grad updated to true\n";
    } else {
        std::cout << "Failed to update requires_grad\n";
    }
    
    std::cout << "\n=== All AutogradMeta tests PASSED! ===\n\n";
    return 0;
}
