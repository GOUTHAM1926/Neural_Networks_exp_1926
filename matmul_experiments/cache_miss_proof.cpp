#include <iostream>
#include <vector>
#include <chrono>

const int M = 2000;
const int N = 2000;
const int K = 2000;

void ijk_matmul(float* A, float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void ikj_matmul(float* A, float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    float* A = new float[M * K]();
    float* B = new float[K * N]();
    float* C = new float[M * N]();

    if (argc > 1 && std::string(argv[1]) == "ijk") {
        ijk_matmul(A, B, C);
    } else {
        ikj_matmul(A, B, C);
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
