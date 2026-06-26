#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
using namespace std;

void run_test(int loop_type) {
    int M = 2000, N = 2000, K = 2000;
    float* A = new float[M * K]; float* B = new float[K * N]; float* C = new float[M * N];
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

    double avg_time = 0.0;
    int runs = 3;
    for (int r = 0; r < runs; ++r) {
        memset(C, 0, M * N * sizeof(float));
        auto start = chrono::high_resolution_clock::now();
        
        if (loop_type == 0) { // i-loop
            #pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < K; ++k) {
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
                    }
                }
            }
        } else { // j-loop
            for (int i = 0; i < M; ++i) {
                #pragma omp parallel for
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < K; ++k) {
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
                    }
                }
            }
        }
        
        auto end = chrono::high_resolution_clock::now();
        avg_time += chrono::duration<double>(end - start).count();
    }
    cout << (loop_type == 0 ? "i-loop" : "j-loop") << " avg time: " << avg_time / runs << " s\n";
    delete[] A; delete[] B; delete[] C;
}

int main(int argc, char** argv) {
    int type = atoi(argv[1]);
    run_test(type);
    return 0;
}
