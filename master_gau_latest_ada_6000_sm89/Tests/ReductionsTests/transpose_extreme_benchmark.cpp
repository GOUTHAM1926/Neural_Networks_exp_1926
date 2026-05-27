#include <iostream>
#include <vector>
#include <chrono>

// Simulate Eigen's GenericDimReducer (Nested loops) for a VERY complex reduction
// [D0, D1, D2, D3, D4] -> reduce D1 and D3
void generic_5d_reduction(const float* data, float* out, int D0, int D1, int D2, int D3, int D4) {
    int out_idx = 0;
    for (int d0 = 0; d0 < D0; ++d0) {
        for (int d2 = 0; d2 < D2; ++d2) {
            for (int d4 = 0; d4 < D4; ++d4) {
                float sum = 0;
                for (int d1 = 0; d1 < D1; ++d1) {
                    for (int d3 = 0; d3 < D3; ++d3) {
                        int idx = d0 * (D1*D2*D3*D4) + d1 * (D2*D3*D4) + d2 * (D3*D4) + d3 * D4 + d4;
                        sum += data[idx];
                    }
                }
                out[out_idx++] = sum;
            }
        }
    }
}

// Simulate TensorFlow's Case 8: Transpose + InnerMostDimReducer
void transpose_then_reduce_5d(const float* data, float* transposed, float* out, int D0, int D1, int D2, int D3, int D4) {
    // Transpose from [D0, D1, D2, D3, D4] -> [D0, D2, D4, D1, D3]
    int idx = 0;
    for (int d0 = 0; d0 < D0; ++d0) {
        for (int d2 = 0; d2 < D2; ++d2) {
            for (int d4 = 0; d4 < D4; ++d4) {
                for (int d1 = 0; d1 < D1; ++d1) {
                    for (int d3 = 0; d3 < D3; ++d3) {
                        transposed[idx++] = data[d0*(D1*D2*D3*D4) + d1*(D2*D3*D4) + d2*(D3*D4) + d3*D4 + d4];
                    }
                }
            }
        }
    }

    // Now reduce the last physical dimensions smoothly (InnerMostDimReducer style)
    int inner_elements = D1 * D3;
    int outer_elements = D0 * D2 * D4;
    for (int i = 0; i < outer_elements; ++i) {
        float sum = 0;
        int base = i * inner_elements;
        // Inner-most SIMD-friendly loop
        for (int j = 0; j < inner_elements; ++j) {
            sum += transposed[base + j];
        }
        out[i] = sum;
    }
}

int main() {
    int D0 = 30, D1 = 30, D2 = 30, D3 = 30, D4 = 30; // 24.3 Million elements
    int total_elements = D0 * D1 * D2 * D3 * D4;
    int out_elements = D0 * D2 * D4;

    float* data = new float[total_elements];
    float* transposed = new float[total_elements];
    float* out = new float[out_elements];

    for(int i=0; i<total_elements; i++) data[i] = 1.0f;

    // Test Generic Reduction (No Transpose)
    auto start1 = std::chrono::high_resolution_clock::now();
    generic_5d_reduction(data, out, D0, D1, D2, D3, D4);
    auto end1 = std::chrono::high_resolution_clock::now();
    double generic_time = std::chrono::duration<double>(end1 - start1).count();

    // Test Transpose + Inner Reduction
    auto start2 = std::chrono::high_resolution_clock::now();
    transpose_then_reduce_5d(data, transposed, out, D0, D1, D2, D3, D4);
    auto end2 = std::chrono::high_resolution_clock::now();
    double transpose_time = std::chrono::duration<double>(end2 - start2).count();

    std::cout << "5D Extreme Mess (Generic Loops): " << generic_time << " seconds\n";
    std::cout << "5D Extreme Mess (Transpose + Inner SIMD): " << transpose_time << " seconds\n";
    
    if(generic_time < transpose_time) {
        std::cout << "\nRESULT: Generic Loops are " << transpose_time/generic_time << "X Faster!\n";
    } else {
        std::cout << "\nRESULT: Transpose is " << generic_time/transpose_time << "X Faster!\n";
    }

    delete[] data;
    delete[] transposed;
    delete[] out;
    return 0;
}
