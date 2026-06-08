#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include "dtype/fp4.h"

// Helper to print bits
void print_bits(uint8_t val) {
    for (int i = 7; i >= 0; --i) {
        std::cout << ((val >> i) & 1);
        if (i == 4) std::cout << " ";
    }
}

void test_packing_unpacking() {
    std::cout << "\n=== Test 1: Packing and Unpacking ===" << std::endl;
    
    // Test values: 1.0 and -2.0
    // 1.0 in E2M1: 
    //   Sign=0, Exp=10(2) -> 2^(2-1)=2. Wait.
    //   Let's check float_to_fp4_e2m1 implementation in fp4.h:
    //   1.0 -> bits=2 (010). Sign=0. -> 0010 = 2.
    //   -2.0 -> abs=2.0 -> bits=4 (100). Sign=1. -> 1100 = 12 (0xC).
    
    float v1 = 1.0f;
    float v2 = -2.0f;
    
    float4_e2m1_t f1(v1);
    float4_e2m1_t f2(v2);
    
    std::cout << "v1: " << v1 << " -> fp4 raw: " << (int)f1.raw_bits << std::endl;
    std::cout << "v2: " << v2 << " -> fp4 raw: " << (int)f2.raw_bits << std::endl;
    
    // Pack: Low=v1, High=v2
    float4_e2m1_2x_t packed(f1, f2); // Low, High
    
    std::cout << "Packed (Low=v1, High=v2) raw bits: ";
    print_bits(packed.raw_bits);
    std::cout << " (" << (int)packed.raw_bits << ")" << std::endl;
    
    // Check bits: High(v2)=1100, Low(v1)=0010 -> 11000010 = 0xC2 = 194
    uint8_t expected_bits = (f2.raw_bits << 4) | (f1.raw_bits & 0xF);
    if (packed.raw_bits != expected_bits) {
        std::cout << "FAIL: Bits mismatch! Expected " << (int)expected_bits << ", got " << (int)packed.raw_bits << std::endl;
    } else {
        std::cout << "PASS: Bits match expected packing." << std::endl;
    }
    
    // Unpack
    float4_e2m1_t unpacked_low = packed.get_low();
    float4_e2m1_t unpacked_high = packed.get_high();
    
    std::cout << "Unpacked Low: " << (float)unpacked_low << " (Expected " << (float)f1 << ")" << std::endl;
    std::cout << "Unpacked High: " << (float)unpacked_high << " (Expected " << (float)f2 << ")" << std::endl;
    
    if ((float)unpacked_low != (float)f1) std::cout << "FAIL: Low value mismatch!" << std::endl;
    if ((float)unpacked_high != (float)f2) std::cout << "FAIL: High value mismatch!" << std::endl;
    
    // Check 2x construct from uint8
    float4_e2m1_2x_t packed_direct(expected_bits);
    if ((float)packed_direct.get_low() != (float)v1) std::cout << "FAIL: Direct Low mismatch" << std::endl;
    if ((float)packed_direct.get_high() != (float)v2) std::cout << "FAIL: Direct High mismatch" << std::endl;
}

void test_arithmetic() {
    std::cout << "\n=== Test 2: Arithmetic Operations ===" << std::endl;
    
    float4_e2m1_2x_t a(float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)); // (1.0, 2.0)
    float4_e2m1_2x_t b(float4_e2m1_t(0.5f), float4_e2m1_t(1.0f)); // (0.5, 1.0)
    
    // Add: (1+0.5=1.5, 2+1=3)
    float4_e2m1_2x_t sum = a + b;
    std::cout << "a + b = (" << (float)sum.get_low() << ", " << (float)sum.get_high() << ")" << std::endl;
    
    // 1.5 is exactly representable (bits 0011 -> 3)
    // 3.0 is exactly representable (bits 0101 -> 5)
    
    if ((float)sum.get_low() != 1.5f) std::cout << "FAIL: Addition Low incorrect. Got " << (float)sum.get_low() << std::endl;
    if ((float)sum.get_high() != 3.0f) std::cout << "FAIL: Addition High incorrect. Got " << (float)sum.get_high() << std::endl;
    
    // Mul: (1*0.5=0.5, 2*1=2)
    float4_e2m1_2x_t prod = a * b;
    std::cout << "a * b = (" << (float)prod.get_low() << ", " << (float)prod.get_high() << ")" << std::endl;
    
    if ((float)prod.get_low() != 0.5f) std::cout << "FAIL: Multiplication Low incorrect." << std::endl;
    if ((float)prod.get_high() != 2.0f) std::cout << "FAIL: Multiplication High incorrect." << std::endl;
}

void test_data_pointer_issues() {
    std::cout << "\n=== Test 3: Data Pointer / Memory Layout Issues ===" << std::endl;
    
    // Issue: Users expect to cast float4_e2m1_2x* to float4_e2m1* and access elements linearly.
    // But float4_e2m1_2x is 1 byte containing 2 elements.
    // float4_e2m1 is also struct { uint8_t } which is 1 byte.
    
    float4_e2m1_2x_t arr[2];
    arr[0] = float4_e2m1_2x_t(float4_e2m1_t(1.0f), float4_e2m1_t(2.0f)); // [Low=1, High=2]
    arr[1] = float4_e2m1_2x_t(float4_e2m1_t(3.0f), float4_e2m1_t(4.0f)); // [Low=3, High=4]
    
    std::cout << "Array Memory Layout:" << std::endl;
    std::cout << "Byte 0: " << (int)arr[0].raw_bits << " (Low=1, High=2)" << std::endl;
    std::cout << "Byte 1: " << (int)arr[1].raw_bits << " (Low=3, High=4)" << std::endl;

    // Simulate "naive" casting
    float4_e2m1_t* cast_ptr = reinterpret_cast<float4_e2m1_t*>(arr);
    
    std::cout << "Accessing via float4_e2m1_t* cast:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        float val = (float)cast_ptr[i];
        // float4_e2m1_t only looks at lowest 4 bits of the byte it points to.
        // i=0: Points to Byte 0. Reads Low bits of Byte 0. -> 1.0. Correct? Yes.
        // i=1: Points to Byte 1. Reads Low bits of Byte 1. -> 3.0. SKIPPING 2.0!
        std::cout << "ptr[" << i << "] = " << val << " (Raw byte: " << (int)cast_ptr[i].raw_bits << ")" << std::endl;
    }
    
    // Verification of the problem
    bool problem_found = false;
    if ((float)cast_ptr[1] == 3.0f) {
        std::cout << "-> Issue Confirmed: ptr[1] accessed the NEXT byte's low part, skipping the HIGH part of the first byte." << std::endl;
        problem_found = true;
    }
    
    if (problem_found) {
        std::cout << "PASS: Successfully demonstrated the pointer stride issue." << std::endl;
    } else {
        std::cout << "FAIL: Could not reproduce the pointer stride issue (unexpected behavior)." << std::endl;
    }
}

int main() {
    std::cout << "Running FP4 Packed Tests..." << std::endl;
    test_packing_unpacking();
    test_arithmetic();
    test_data_pointer_issues();
    std::cout << "\nAll tests completed." << std::endl;
    return 0;
}
