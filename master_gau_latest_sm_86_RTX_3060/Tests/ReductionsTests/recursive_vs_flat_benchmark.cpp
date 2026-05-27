#include <iostream>
#include <vector>
#include <chrono>

int main() {
    int D0 = 100, D1 = 100, D2 = 100; // 1 Million elements
    std::vector<int64_t> strides = {10000, 100, 1};
    std::vector<int64_t> dims = {100, 100, 100};
    int total = D0 * D1 * D2;
    
    float* data = new float[total];
    for(int i=0; i<total; i++) data[i] = 1.0f;

    // YOUR WAY: FLAT LOOP WITH UNRAVEL
    auto start1 = std::chrono::high_resolution_clock::now();
    float sum1 = 0;
    for(int i=0; i<total; i++) {
        int64_t coords[3];
        int64_t temp = i;
        coords[2] = temp % dims[2]; temp /= dims[2];
        coords[1] = temp % dims[1]; temp /= dims[1];
        coords[0] = temp % dims[0];
        
        int64_t idx = (coords[0]*strides[0]) + (coords[1]*strides[1]) + (coords[2]*strides[2]);
        sum1 += data[idx];
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // EIGEN'S WAY: NESTED LOOPS
    auto start2 = std::chrono::high_resolution_clock::now();
    float sum2 = 0;
    for(int d0=0; d0<D0; d0++) {
        int idx0 = d0 * strides[0];
        for(int d1=0; d1<D1; d1++) {
            int idx1 = idx0 + (d1 * strides[1]);
            for(int d2=0; d2<D2; d2++) {
                int idx2 = idx1 + (d2 * strides[2]);
                sum2 += data[idx2];
            }
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::cout << "Your Flat Loop: " << std::chrono::duration<double>(end1 - start1).count() << "s\n";
    std::cout << "Eigen Nested Loops: " << std::chrono::duration<double>(end2 - start2).count() << "s\n";
    
    delete[] data;
    return 0;
}
