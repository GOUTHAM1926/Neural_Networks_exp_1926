#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <functional>
#include <mpi.h>


static int tests_passed = 0;
static int tests_failed = 0;
static int tests_total  = 0;
static int g_rank = -1;

inline void set_test_rank(int rank) { g_rank = rank; }

inline std::string rank_prefix() {
    return (g_rank >= 0) ? "[Rank " + std::to_string(g_rank) + "] " : "";
}

#define TEST_ASSERT(condition, message)                                         \
    do {                                                                        \
        tests_total++;                                                          \
        if (!(condition)) {                                                     \
            tests_failed++;                                                     \
            std::cerr << rank_prefix() << "[FAIL] " << message                  \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"             \
                      << std::endl;                                             \
        } else {                                                                \
            tests_passed++;                                                     \
            std::cout << rank_prefix() << "[PASS] " << message << std::endl;    \
        }                                                                       \
    } while (0)

#define TEST_ASSERT_EQ(a, b, message)                                           \
    do {                                                                         \
        tests_total++;                                                           \
        if ((a) != (b)) {                                                        \
            tests_failed++;                                                      \
            std::cerr << rank_prefix() << "[FAIL] " << message                   \
                      << " (expected: " << (b) << ", got: " << (a) << ")"        \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"              \
                      << std::endl;                                              \
        } else {                                                                 \
            tests_passed++;                                                      \
            std::cout << rank_prefix() << "[PASS] " << message << std::endl;     \
        }                                                                        \
    } while (0)

#define TEST_ASSERT_NEAR(a, b, eps, message)                                    \
    do {                                                                         \
        tests_total++;                                                           \
        if (std::fabs((double)(a) - (double)(b)) > (eps)) {                      \
            tests_failed++;                                                      \
            std::cerr << rank_prefix() << "[FAIL] " << message                   \
                      << " (expected ~" << (b) << ", got: " << (a)               \
                      << ", eps=" << (eps) << ")"                                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"              \
                      << std::endl;                                              \
        } else {                                                                 \
            tests_passed++;                                                      \
            std::cout << rank_prefix() << "[PASS] " << message << std::endl;     \
        }                                                                        \
    } while (0)

#define TEST_ASSERT_THROWS(expr, message)                                       \
    do {                                                                         \
        tests_total++;                                                           \
        bool threw = false;                                                      \
        try { expr; } catch (...) { threw = true; }                              \
        if (!threw) {                                                            \
            tests_failed++;                                                      \
            std::cerr << rank_prefix() << "[FAIL] " << message                   \
                      << " (no exception thrown)"                                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"              \
                      << std::endl;                                              \
        } else {                                                                 \
            tests_passed++;                                                      \
            std::cout << rank_prefix() << "[PASS] " << message << std::endl;     \
        }                                                                        \
    } while (0)


inline void print_test_summary(const std::string& suite_name) {
    // Collect results across all ranks
    int global_total = 0, global_passed = 0, global_failed = 0;
    MPI_Reduce(&tests_total, &global_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tests_passed, &global_passed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tests_failed, &global_failed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::cout << "\n==============================" << std::endl;
        std::cout << "Test Suite: " << suite_name << std::endl;
        std::cout << "  Total:  " << global_total << " (across all ranks)" << std::endl;
        std::cout << "  Passed: " << global_passed << std::endl;
        std::cout << "  Failed: " << global_failed << std::endl;
        std::cout << "==============================\n" << std::endl;
    }
}

inline int test_exit_code() {
    // All ranks must agree: if ANY rank failed, everyone returns 1
    int local_fail = (tests_failed > 0) ? 1 : 0;
    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return (global_fail > 0) ? 1 : 0;
}


inline std::vector<float> read_tensor_to_cpu(OwnTensor::Tensor& tensor) {
    auto cpu_tensor = tensor.to_cpu();
    float* ptr = static_cast<float*>(cpu_tensor.data());
    return std::vector<float>(ptr, ptr + cpu_tensor.numel());
}
