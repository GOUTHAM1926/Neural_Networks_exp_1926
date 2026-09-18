#include <omp.h>
#include <iostream>
#include <chrono>
#include <cstring>

int main() {
    int M = 2000, N = 2000, K = 2000;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    // Initialize arrays to avoid NaN issues
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

    int num_runs = 3;

    std::cout << "Starting Ablation Tests...\n\n";

    // TEST 0: Baseline (OpenMP on i-loop)
    // No Software Halt (1 barrier total), No Hardware Halt (threads write to different rows)
    double time0 = 0;
    for (int run = 0; run < num_runs; ++run) {
        std::memset(C, 0, M * N * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        time0 += std::chrono::duration<double>(end - start).count();
    }
    std::cout << "Test 0 (i-loop Baseline): " << time0 / num_runs << " seconds\n";

    // TEST 1: OpenMP on j-loop, Default Scheduling
    // Software Halt = MAX (2000 barriers). Hardware Halt = MINIMAL (Default scheduling chunks arrays by ~71 elements, so false sharing only happens at boundaries).
    double time1 = 0;
    for (int run = 0; run < num_runs; ++run) {
        std::memset(C, 0, M * N * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M; ++i) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        time1 += std::chrono::duration<double>(end - start).count();
    }
    std::cout << "Test 1 (j-loop, Default Static): " << time1 / num_runs << " seconds\n";

    // TEST 2: OpenMP on j-loop, Forced False Sharing
    // Software Halt = MAX (2000 barriers). Hardware Halt = MAX (schedule(static, 1) interleaves elements so EVERY single write hits False Sharing).
    double time2 = 0;
    for (int run = 0; run < num_runs; ++run) {
        std::memset(C, 0, M * N * sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M; ++i) {
            #pragma omp parallel for schedule(static, 1)
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        time2 += std::chrono::duration<double>(end - start).count();
    }
    std::cout << "Test 2 (j-loop, schedule(static, 1) - Max False Sharing): " << time2 / num_runs << " seconds\n";

    delete[] A; delete[] B; delete[] C;
    return 0;
}
