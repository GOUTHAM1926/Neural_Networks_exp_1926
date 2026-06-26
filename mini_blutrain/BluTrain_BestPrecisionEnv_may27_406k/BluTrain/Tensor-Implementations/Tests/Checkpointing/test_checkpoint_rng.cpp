#include <iostream>
#include <random>
#include <cassert>
#include "checkpointing/RNG.h"
#include "TensorLib.h"

using namespace OwnTensor;

void test_rng_basic() {
    std::cout << "Testing RNG state save/restore..." << std::endl;

    RNG::set_seed(12345);

    // Generate some numbers
    auto& gen = RNG::get_cpu_generator();
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    float v1 = dist(gen);
    float v2 = dist(gen);

    // Save state
    RNGState state = RNG::get_state();

    // Generate more numbers
    float v3 = dist(gen);
    float v4 = dist(gen);

    // Restore state
    RNG::set_state(state);

    // Generate numbers again
    float v3_restore = dist(gen);
    float v4_restore = dist(gen);

    std::cout << "v1: " << v1 << ", v2: " << v2 << std::endl;
    std::cout << "v3: " << v3 << ", v4: " << v4 << std::endl;
    std::cout << "v3_restore: " << v3_restore << ", v4_restore: " << v4_restore << std::endl;

    if (v3 == v3_restore && v4 == v4_restore) {
        std::cout << "RNG restore SUCCESS!" << std::endl;
    } else {
        std::cerr << "RNG restore FAILED!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        test_rng_basic();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
