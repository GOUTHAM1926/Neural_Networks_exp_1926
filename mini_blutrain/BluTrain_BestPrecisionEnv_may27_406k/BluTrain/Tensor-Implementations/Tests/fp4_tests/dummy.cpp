#include "TensorLib.h"
#include "dtype/fp4.h"
#include <iostream>

using namespace OwnTensor;

void fp4() {
    std::cout << "Testing FP4 Data Access..." << std::endl;
    
    // Create tensor and set data
    Tensor t(Shape({{2,2}}), Dtype::Float4_e2m1);
    
    std::vector<float4_e2m1_t> data = {
        float4_e2m1_t(0.0f),
        float4_e2m1_t(1.0f),
        float4_e2m1_t(-2.7f),
        float4_e2m1_t(6.0f)
    };
    
    t.set_data(data);
    
    t.display();

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

int main()
{
    Tensor t(Shape({{2,2}}), Dtype::Float4_e2m1);
    
    std::vector<float4_e2m1_t> data = {
        float4_e2m1_t(0.0f),
        float4_e2m1_t(1.0f),
        float4_e2m1_t(-2.7f),
        float4_e2m1_t(6.0f)
    };
    
    t.set_data(data);
    
    t.display();

    Tensor q({{2}}, Dtype::Float4_e2m1_2x);
    
    // float4_e2m1_2x_t v1;
    // v1.set_high(float4_e2m1_t(1.0f));
    // v1.set_low(float4_e2m1_t(6.0f));
    float4_e2m1_2x_t v1(float4_e2m1_t(1.0f), float4_e2m1_t(6.0f));
    // float4_e2m1_2x_t v2(float4_e2m1_t(-0.5f), float4_e2m1_t(-6.0f));
    
    // float4_e2m1_2x_t v2;
    // v2.set_high(float4_e2m1_t(-0.5f));
    // v2.set_low(float4_e2m1_t(-6.0f));
    
    std::vector<float4_e2m1_2x_t> data1 = {v1};

    std::cout << std::endl;
    std::cout << "v1 high part: " << float(v1.get_high()) <<std::endl;
    std::cout << "v1 low part: " << float(v1.get_low()) <<std::endl;

    

    q.set_data(data1);
    
    q.display();
}