#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include "dtype/Types.h"
#include "dtype/FP4Converters.h"

using namespace OwnTensor;

// Expected values mapping
struct FP4Entry {
    uint8_t bits;
    float value;
};

// 0-7: Positive
// 8-15: Negative
std::vector<FP4Entry> expected_table = {
    {0, 0.0f},
    {1, 0.5f},
    {2, 1.0f},
    {3, 1.5f},
    {4, 2.0f},
    {5, 3.0f},
    {6, 4.0f},
    {7, 6.0f},
    {8, -0.0f},
    {9, -0.5f},
    {10, -1.0f},
    {11, -1.5f},
    {12, -2.0f},
    {13, -3.0f},
    {14, -4.0f},
    {15, -6.0f}
};

void test_fp4_to_fp32() {
    std::cout << "Testing FP4 -> FP32..." << std::endl;
    for (const auto& entry : expected_table) {
        float4_e2m1_t val(entry.bits);
        float f = static_cast<float>(val);
        
        // Exact equality check
        if (f != entry.value) {
            // Handle -0.0
            if (entry.value == 0.0f && f == 0.0f) continue; 
            
            std::cerr << "FAIL: Bits " << (int)entry.bits 
                      << " Expected " << entry.value 
                      << " Got " << f << std::endl;
            exit(1);
        }
    }
    std::cout << "PASS" << std::endl;
}

void test_fp32_to_fp4() {
    std::cout << "Testing FP32 -> FP4..." << std::endl;
    for (const auto& entry : expected_table) {
        float4_e2m1_t val(entry.value);
        if (val.raw_bits != entry.bits) {
             std::cerr << "FAIL: Float " << entry.value
                      << " Expected bits " << (int)entry.bits 
                      << " Got " << (int)val.raw_bits << std::endl;
            exit(1);
        }
    }
    // Test rounding
    // 0.2 -> 0 (0)
    // 0.3 -> 0.5 (1)
    if (float4_e2m1_t(0.2f).raw_bits != 0) std::cerr << "FAIL: 0.2f -> 0" << std::endl;
    if (float4_e2m1_t(0.3f).raw_bits != 1) std::cerr << "FAIL: 0.3f -> 0.5" << std::endl;
    
    std::cout << "PASS" << std::endl;
}

void test_fp4_packed() {
    std::cout << "Testing FP4 Packed..." << std::endl;
    // Pack 1.0 (2) and -2.0 (12 -> 0xC)
    // Low: 1.0, High: -2.0
    // Expected byte: 0xC2
    
    float4_e2m1_t low(1.0f);
    float4_e2m1_t high(-2.0f);
    
    float4_e2m1_2x_t packed(low, high);
    
    if (packed.raw_bits != 0xC2) {
        std::cerr << "FAIL: Packed bits expected 0xC2, got " << std::hex << (int)packed.raw_bits << std::dec << std::endl;
        exit(1);
    }
    
    // Unpack
    float4_e2m1_t out_low = packed.get_low();
    float4_e2m1_t out_high = packed.get_high();
    
    if (static_cast<float>(out_low) != 1.0f) std::cerr << "FAIL: Unpacked low != 1.0" << std::endl;
    if (static_cast<float>(out_high) != -2.0f) std::cerr << "FAIL: Unpacked high != -2.0" << std::endl;

    std::cout << "PASS" << std::endl;
}

void test_conversions() {
    std::cout << "Testing Other Conversions..." << std::endl;
    
    float4_e2m1_t v(1.5f); // 3
    
    // To Double
    if (fp4_to_fp64(v) != 1.5) std::cerr << "FAIL: fp4_to_fp64" << std::endl;
    
    // To FP16
    float16_t h = fp4_to_fp16(v);
    if (static_cast<float>(h) != 1.5f) std::cerr << "FAIL: fp4_to_fp16" << std::endl;
    
    // To BF16
    bfloat16_t b = fp4_to_bf16(v);
    if (static_cast<float>(b) != 1.5f) std::cerr << "FAIL: fp4_to_bf16" << std::endl;
    
    std::cout << "PASS" << std::endl;
}

int main() {
    test_fp4_to_fp32();
    test_fp32_to_fp4();
    test_fp4_packed();
    test_conversions();
    
    std::cout << "All FP4 conversion tests passed!" << std::endl;
    return 0;
}
