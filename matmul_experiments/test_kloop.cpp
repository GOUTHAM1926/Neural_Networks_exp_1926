#include <omp.h>
#include <iostream>

int main() {
    int M = 3, N = 4, K = 3;
    float A[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float B[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float C[12] = {0};

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            #pragma omp parallel for
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    return 0;
}
