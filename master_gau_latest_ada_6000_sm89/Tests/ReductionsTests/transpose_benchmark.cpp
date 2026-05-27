#include <iostream>
#include <vector>
#include <chrono>

// Simulate Eigen's GenericDimReducer (Nested loops)
void generic_3d_reduction(const float* data, float* out, int D0, int D1, int D2) {
    // 3D-Y reduction (Reduce D1 (Height), Preserve D0 and D2)
    // Stride for D1 is D2
    int out_idx = 0;
    for (int d0 = 0; d0 < D0; ++d0) {
        int base_d0 = d0 * D1 * D2;
        for (int d2 = 0; d2 < D2; ++d2) {
            float sum = 0;
            for (int d1 = 0; d1 < D1; ++d1) {
                sum += data[base_d0 + d1 * D2 + d2];
            }
            out[out_idx++] = sum;
        }
    }
}

// Simulate TensorFlow's Case 8: Transpose + InnerMostDimReducer
void transpose_then_reduce(const float* data, float* transposed, float* out, int D0, int D1, int D2) {
    // Transpose from [D0, D1, D2] -> [D0, D2, D1]
    int idx = 0;
    for (int d0 = 0; d0 < D0; ++d0) {
        for (int d2 = 0; d2 < D2; ++d2) {
            for (int d1 = 0; d1 < D1; ++d1) {
                transposed[idx++] = data[d0 * D1 * D2 + d1 * D2 + d2];
            }
        }
    }

    // Now reduce the last physical dimension smoothly (InnerMostDimReducer style)
    for (int i = 0; i < D0 * D2; ++i) {
        float sum = 0;
        int base = i * D1;
        // Inner-most SIMD-friendly loop
        for (int d1 = 0; d1 < D1; ++d1) {
            sum += transposed[base + d1];
        }
        out[i] = sum;
    }
}

int main() {
    int D0 = 200, D1 = 200, D2 = 200; // 8 Million elements
    int total_elements = D0 * D1 * D2;
    int out_elements = D0 * D2;

    float* data = new float[total_elements];
    float* transposed = new float[total_elements];
    float* out = new float[out_elements];

    for(int i=0; i<total_elements; i++) data[i] = 1.0f;

    // Test Generic Reduction (No Transpose)
    auto start1 = std::chrono::high_resolution_clock::now();
    generic_3d_reduction(data, out, D0, D1, D2);
    auto end1 = std::chrono::high_resolution_clock::now();
    double generic_time = std::chrono::duration<double>(end1 - start1).count();

    // Test Transpose + Inner Reduction
    auto start2 = std::chrono::high_resolution_clock::now();
    transpose_then_reduce(data, transposed, out, D0, D1, D2);
    auto end2 = std::chrono::high_resolution_clock::now();
    double transpose_time = std::chrono::duration<double>(end2 - start2).count();

    std::cout << "Case 7 Style (Eigen Generic Nested Loops): " << generic_time << " seconds\n";
    std::cout << "Case 8 Style (Physically Transpose + Inner SIMD): " << transpose_time << " seconds\n";
    
    if(generic_time < transpose_time) {
        std::cout << "\nRESULT: Generic Loops are " << transpose_time/generic_time << "X Faster!\n";
    } else {
        std::cout << "\nRESULT: Transpose is Faster!\n";
    }

    delete[] data;
    delete[] transposed;
    delete[] out;
    return 0;
}
