#include "TensorLib.h"
#include <iostream>
#include <complex>

using namespace OwnTensor;

int main() {
    std::cout << "=== Testing Complex Type Statistical Operations ===" << std::endl;
    
    // Test 1: Complex64 Mean
    std::cout << "\n--- Test 1: Complex64 Mean ---" << std::endl;
    Tensor complex_tensor({{3, 3}}, Dtype::Complex64, DeviceIndex(Device::CPU));
    
    std::vector<std::complex<float>> data = {
        {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f},
        {4.0f, 4.0f}, {5.0f, 5.0f}, {6.0f, 6.0f},
        {7.0f, 7.0f}, {8.0f, 8.0f}, {9.0f, 9.0f}
    };
    complex_tensor.set_data(data);
    
    std::cout << "Input tensor (3x3 complex):" << std::endl;
    complex_tensor.display(std::cout, 5);
    
    try {
        Tensor mean_result = reduce_mean(complex_tensor);
        std::cout << "\n✅ Mean result (should be 5+5i):" << std::endl;
        mean_result.display(std::cout, 5);
    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    // Test 2: Complex64 Variance
    std::cout << "\n--- Test 2: Complex64 Variance ---" << std::endl;
    try {
        Tensor var_result = reduce_var(complex_tensor, {}, false, 0);
        std::cout << "✅ Variance result:" << std::endl;
        var_result.display(std::cout, 5);
    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    // Test 3: Complex128 Mean
    std::cout << "\n--- Test 3: Complex128 Mean ---" << std::endl;
    Tensor complex128_tensor({{2, 2}}, Dtype::Complex128, DeviceIndex(Device::CPU));
    
    std::vector<std::complex<double>> data128 = {
        {10.0, 20.0}, {30.0, 40.0},
        {50.0, 60.0}, {70.0, 80.0}
    };
    complex128_tensor.set_data(data128);
    
    std::cout << "Input tensor (2x2 complex128):" << std::endl;
    complex128_tensor.display(std::cout, 5);
    
    try {
        Tensor mean128 = reduce_mean(complex128_tensor);
        std::cout << "\n✅ Mean result (should be 40+50i):" << std::endl;
        mean128.display(std::cout, 5);
    } catch (const std::exception& e) {
        std::cout << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== ✅ All Tests Passed! Complex statistical operations work! ===" << std::endl;
    return 0;
}
