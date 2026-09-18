#include <omp.h>
void matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
