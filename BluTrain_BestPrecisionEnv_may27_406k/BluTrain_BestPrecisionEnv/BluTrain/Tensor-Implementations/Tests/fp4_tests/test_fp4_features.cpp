#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include "core/Tensor.h"
#include "dtype/fp4.h"

using namespace OwnTensor;

void test_tensor_creation() {
    std::cout << "Testing FP4 Tensor Creation..." << std::endl;
    
    // Test unpacked FP4
    Shape shape({{2, 2}});
    Tensor t1(shape, Dtype::Float4_e2m1);
    
    // Test packed FP4
    Tensor t2({{3}}, Dtype::Float4_e2m1_2x);
    
    std::cout << "PASS" << std::endl;
}

void test_data_access() {
    std::cout << "Testing FP4 Data Access..." << std::endl;
    
    // Create tensor and set data
    Tensor t(Shape({{4}}), Dtype::Float4_e2m1);
    
    std::vector<float4_e2m1_t> data = {
        float4_e2m1_t(0.0f),
        float4_e2m1_t(1.0f),
        float4_e2m1_t(-2.0f),
        float4_e2m1_t(6.0f)
    };
    
    t.set_data(data);
    
    // Retrieve data
    const float4_e2m1_t* ptr = t.data<float4_e2m1_t>();
    
    // Verify values
    if (static_cast<float>(ptr[0]) != 0.0f) {
        std::cerr << "FAIL: Expected 0.0, got " << static_cast<float>(ptr[0]) << std::endl;
        exit(1);
    }
    if (static_cast<float>(ptr[1]) != 1.0f) {
        std::cerr << "FAIL: Expected 1.0, got " << static_cast<float>(ptr[1]) << std::endl;
        exit(1);
    }
    if (static_cast<float>(ptr[2]) != -2.0f) {
        std::cerr << "FAIL: Expected -2.0, got " << static_cast<float>(ptr[2]) << std::endl;
        exit(1);
    }
    if (static_cast<float>(ptr[3]) != 6.0f) {
        std::cerr << "FAIL: Expected 6.0, got " << static_cast<float>(ptr[3]) << std::endl;
        exit(1);
    }
    
    std::cout << "PASS" << std::endl;
}

void test_fill() {
    std::cout << "Testing FP4 Fill..." << std::endl;
    
    Tensor t(Shape({{3}}), Dtype::Float4_e2m1);
    t.fill(float4_e2m1_t(1.5f));
    
    const float4_e2m1_t* ptr = t.data<float4_e2m1_t>();
    for (int i = 0; i < 3; i++) {
        if (static_cast<float>(ptr[i]) != 1.5f) {
            std::cerr << "FAIL: Expected 1.5, got " << static_cast<float>(ptr[i]) << std::endl;
            exit(1);
        }
    }
    
    std::cout << "PASS" << std::endl;
}

void test_display() {
    std::cout << "Testing FP4 Display..." << std::endl;
    
    Tensor t(Shape({{2, 2}}), Dtype::Float4_e2m1);
    std::vector<float4_e2m1_t> data = {
        float4_e2m1_t(0.0f),
        float4_e2m1_t(1.0f),
        float4_e2m1_t(-2.0f),
        float4_e2m1_t(6.0f)
    };
    t.set_data(data);
    
    std::stringstream ss;
    t.display(ss);
    std::string output = ss.str();
    
    // Verify output contains expected elements
    if (output.find("float4_e2m1") == std::string::npos) {
        std::cerr << "FAIL: Output missing dtype name" << std::endl;
        exit(1);
    }
    
    if (output.find("-2.0000") == std::string::npos) {
        std::cerr << "FAIL: Output missing value -2.0" << std::endl;
        exit(1);
    }
    
    std::cout << "Display Output:\n" << output << std::endl;
    std::cout << "PASS" << std::endl;
}

void test_packed_display() {
    std::cout << "Testing Packed FP4 Display..." << std::endl;
    
    Tensor t(Shape({{2}}), Dtype::Float4_e2m1_2x);
    
    float4_e2m1_2x_t v1(float4_e2m1_t(1.0f), float4_e2m1_t(6.0f));
    float4_e2m1_2x_t v2(float4_e2m1_t(-0.5f), float4_e2m1_t(-6.0f));
    
    std::vector<float4_e2m1_2x_t> data = {v1, v2};
    t.set_data(data);
    
    std::stringstream ss;
    t.display(ss);
    std::string output = ss.str();
    
    if (output.find("float4_e2m1_2x") == std::string::npos) {
        std::cerr << "FAIL: Output missing packed dtype name" << std::endl;
        exit(1);
    }
    
    std::cout << "Packed Display Output:\n" << output << std::endl;
    std::cout << "PASS" << std::endl;
}

void test_view_operations() {
    std::cout << "Testing FP4 View Operations..." << std::endl;
    
    // Create a 2x3 tensor
    Tensor t(Shape({{2, 3}}), Dtype::Float4_e2m1);
    t.fill(float4_e2m1_t(1.0f));
    
    // Test reshape
    Tensor reshaped = t.reshape(Shape({{3, 2}}));
    if (reshaped.shape().dims[0] != 3 || reshaped.shape().dims[1] != 2) {
        std::cerr << "FAIL: Reshape failed" << std::endl;
        exit(1);
    }
    
    // Test flatten
    Tensor flattened = t.flatten();
    if (flattened.shape().dims.size() != 1 || flattened.shape().dims[0] != 6) {
        std::cerr << "FAIL: Flatten failed" << std::endl;
        exit(1);
    }
    
    std::cout << "PASS" << std::endl;
}

int main() {
    try {
        test_tensor_creation();
        test_data_access();
        test_fill();
        test_display();
        test_packed_display();
        test_view_operations();
        
        std::cout << "\n✓ All FP4 feature tests passed!" << std::endl;
        std::cout << "FP4 types support: Creation, Data Access, Fill, Display, and View operations" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Unhandled Exception: " << e.what() << std::endl;
        return 1;
    }
}
