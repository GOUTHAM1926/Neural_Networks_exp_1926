#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/GraphArena.h"

using namespace OwnTensor;

/**
 * @brief Helper to get current Resident Set Size (RSS) in KB.
 */
long get_rss_kb() {
    std::ifstream stat_file("/proc/self/status");
    std::string line;
    while (std::getline(stat_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            return std::stol(line.substr(7, line.length() - 10));
        }
    }
    return 0;
}

/**
 * @brief Benchmark a deep sequence of operations to stress the autograd graph.
 */
void benchmark_autograd_stress(int depth, int iterations) {
    std::cout << "Benchmarking Autograd Stress..." << std::endl;
    std::cout << "Depth: " << depth << ", Iterations: " << iterations << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    long initial_rss = get_rss_kb();
    long peak_rss = initial_rss;

    for (int i = 0; i < iterations; ++i) {
        // 1. Create input tensor
        Tensor x = Tensor::randn<float>(Shape{{64, 64}}, TensorOptions().with_req_grad(true));
        Tensor y = x;

        // 2. Perform many small operations to build a large graph
        for (int d = 0; d < depth; ++d) {
            y = autograd::add(y, x); // Each add creates a Node and Edges
        }

        // 3. Compute scalar loss
        Tensor loss = autograd::sum(y); // Fix: use autograd namespace function

        // 4. Backward pass
        loss.backward();

        // 5. Reset Arena for next iteration
        autograd::GraphArena::get_thread_local().reset();

        // Track peak memory
        peak_rss = std::max(peak_rss, get_rss_kb());

        if ((i + 1) % 10 == 0) {
            std::cout << "\rIteration " << (i + 1) << "/" << iterations 
                      << " | Current RSS: " << get_rss_kb() << " KB" << std::flush;
        }
    }
    std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "Total Time: " << std::fixed << std::setprecision(4) << duration << " s" << std::endl;
    std::cout << "Avg Time/Iter: " << (duration / iterations) * 1000.0 << " ms" << std::endl;
    std::cout << "Initial RSS: " << initial_rss << " KB" << std::endl;
    std::cout << "Peak RSS: " << peak_rss << " KB" << std::endl;
    std::cout << "RSS Growth: " << peak_rss - initial_rss << " KB" << std::endl;
}

int main() {
    // Stress test: 1000 operations deep, 100 iterations
    // This will create ~100,000 Node objects in total across iterations.
    benchmark_autograd_stress(1000, 100);
    return 0;
}
