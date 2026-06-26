#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cstdint>

void write_tensor(std::ofstream& out, const std::vector<int64_t>& dims) {
    uint32_t num_dims = static_cast<uint32_t>(dims.size());
    out.write(reinterpret_cast<const char*>(&num_dims), sizeof(uint32_t));
    
    size_t num_elements = 1;
    for (int64_t d : dims) {
        uint32_t dim_sz = static_cast<uint32_t>(d);
        out.write(reinterpret_cast<const char*>(&dim_sz), sizeof(uint32_t));
        num_elements *= d;
    }
    
    // Generate random float data
    std::vector<float> data(num_elements);
    std::default_random_engine gen;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = dist(gen);
    }
    
    out.write(reinterpret_cast<const char*>(data.data()), num_elements * sizeof(float));
    std::cout << "Written tensor with " << num_elements << " elements." << std::endl;
}

int main() {
    std::string filename = "accuracy_test_data.bin";
    std::ofstream out(filename, std::ios::binary);
    
    if (!out) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    
    // 1. Input: [64, 32]
    write_tensor(out, {64, 32});
    
    // 2. Target: [64, 10]
    write_tensor(out, {64, 10});
    
    // 3. FC1 W: [64, 32]
    write_tensor(out, {64, 32});
    
    // 4. FC1 B: [64]
    write_tensor(out, {64});
    
    // 5. FC2 W: [32, 64]
    write_tensor(out, {32, 64});
    
    // 6. FC2 B: [32]
    write_tensor(out, {32});
    
    // 7. FC3 W: [10, 32]
    write_tensor(out, {10, 32});
    
    // 8. FC3 B: [10]
    write_tensor(out, {10});
    
    out.close();
    std::cout << "Successfully generated " << filename << std::endl;
    return 0;
}
