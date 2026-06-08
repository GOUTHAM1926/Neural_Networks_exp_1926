
#pragma once
#include <random>
#include <string>


inline std::string generate_id(size_t length) {
    const std::string charset = "0123456789"
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz";
    
    // 1. Initialize random device and generator
    std::random_device rd; 
    std::mt19937 generator(rd()); // Mersenne Twister engine
    
    // 2. Define the distribution range for charset indices
    std::uniform_int_distribution<int> distribution(0, charset.size() - 1);
    
    std::string id;
    id.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        id += charset[distribution(generator)];
    }
    
    return "tensor_" + id;
}