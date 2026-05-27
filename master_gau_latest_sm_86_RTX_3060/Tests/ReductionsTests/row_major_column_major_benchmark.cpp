#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
int main() {
    const int64_t D0 = 400, D1 = 400, D2 = 400;
    const size_t numel = D0 * D1 * D2;
    
    // Allocate 256MB of float data
    float* data = new float[numel];
    for (size_t i = 0; i < numel; ++i) data[i] = 1.0f;
    // CASE 1: ROW-MAJOR
    // Last dimension 'k' changes fastest (Sequential memory access)
    {
        auto start = std::chrono::high_resolution_clock::now();
        //volatile float sum = 0; 
         float sum = 0; 
        for (int i = 0; i < D0; ++i)
            for (int j = 0; j < D1; ++j)
                for (int k = 0; k < D2; ++k)
                    sum += data[i * D1 * D2 + j * D2 + k];
        
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Row-Major (Last Dim Fast): " << std::chrono::duration<double>(end - start).count() << "s\n";
    }

    // CASE 2: COLUMN-MAJOR
    // First dimension 'i' changes fastest (Jumping memory access)
    {
        auto start = std::chrono::high_resolution_clock::now();
        //volatile float sum = 0;
         float sum = 0;
        for (int k = 0; k < D2; ++k)
            for (int j = 0; j < D1; ++j)
                for (int i = 0; i < D0; ++i)
                    sum += data[i * D1 * D2 + j * D2 + k];
        //sum += data[i + j * D0 + k * D0 * D1];
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Column-Major (First Dim Fast): " << std::chrono::duration<double>(end - start).count() << "s\n";
    

    delete[] data;
    return 0;
}
}

//Result in Intel i7 - 14700K CPU :
// Row-Major (Last Dim Fast) : 0.0941739 secs ,
// Column-Major (First Dim Fast) : 0.225564 secs . 