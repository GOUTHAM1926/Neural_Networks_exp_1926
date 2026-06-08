#include "autograd/AnomalyMode.h"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

using namespace OwnTensor::autograd;

void test_anomaly_mode_toggle() {
    std::cout << "Testing AnomalyMode toggle..." << std::endl;
    
    // Default state should be false (based on AnomalyMode.cpp)
    bool initial_state = AnomalyMode::is_enabled();
    std::cout << "Initial state: " << (initial_state ? "Enabled" : "Disabled") << std::endl;
    
    // Enable
    AnomalyMode::set_enabled(true);
    if (AnomalyMode::is_enabled()) {
        std::cout << "PASS: AnomalyMode enabled successfully." << std::endl;
    } else {
        std::cerr << "FAIL: Failed to enable AnomalyMode." << std::endl;
        exit(1);
    }
    
    // Disable
    AnomalyMode::set_enabled(false);
    if (!AnomalyMode::is_enabled()) {
        std::cout << "PASS: AnomalyMode disabled successfully." << std::endl;
    } else {
        std::cerr << "FAIL: Failed to disable AnomalyMode." << std::endl;
        exit(1);
    }
    
    // Restore initial state (though we likely want it off for other tests unless specified)
    AnomalyMode::set_enabled(initial_state);
}

void test_anomaly_metadata() {
    std::cout << "Testing AnomalyMetadata..." << std::endl;
    
    AnomalyMetadata metadata;
    
    // Test context
    std::string test_ctx = "Created in test_anomaly_metadata";
    metadata.set_context(test_ctx);
    
    if (metadata.context() == test_ctx) {
        std::cout << "PASS: Context storage works." << std::endl;
    } else {
        std::cerr << "FAIL: Context storage mismatch. Expected: " << test_ctx << ", Got: " << metadata.context() << std::endl;
        exit(1);
    }
    
    // Test store_stack mock behavior
    metadata.store_stack();
    std::string expected_stack_msg = "Stack trace capture enabled";
    if (metadata.context() == expected_stack_msg) {
        std::cout << "PASS: store_stack() updates context correctly." << std::endl;
    } else {
        std::cerr << "FAIL: store_stack() did not update context as expected." << std::endl;
        exit(1);
    }
    
    // Test empty stack trace (since it's not fully implemented yet, just checking vector access)
    const auto& stack = metadata.get_stack();
    if (stack.empty()) {
        std::cout << "PASS: Stack trace vector is accessible (and empty as expected)." << std::endl;
    } else {
        std::cout << "WARN: Stack trace vector is not empty (unexpected for current implementation)." << std::endl;
    }
}


#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

void test_stress() {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Running Stress Test (Multi-threaded)..." << std::endl;

    std::atomic<bool> running{true};
    std::atomic<int> ops_count{0};
    const int NUM_THREADS = 10;
    const int DURATION_MS = 1000;

    std::vector<std::thread> threads;

    // Threads that randomly toggle the mode
    for (int i = 0; i < NUM_THREADS / 2; ++i) {
        threads.emplace_back([&]() {
            while (running) {
                bool current = AnomalyMode::is_enabled();
                AnomalyMode::set_enabled(!current);
                ops_count++;
                // Small sleep to yield
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    // Threads that create metadata
    for (int i = 0; i < NUM_THREADS / 2; ++i) {
        threads.emplace_back([&]() {
            while (running) {
                AnomalyMetadata meta;
                if (AnomalyMode::is_enabled()) {
                    meta.store_stack();
                }
                meta.set_context("Stress test context");
                ops_count++;
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    std::cout << "Stress test running for " << DURATION_MS << "ms..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(DURATION_MS));
    running = false;

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "PASS: Stress test completed without crash. Total ops: " << ops_count << std::endl;
    
    // Ensure we leave it in a known state
    AnomalyMode::set_enabled(false);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running AnomalyMode Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_anomaly_mode_toggle();
    std::cout << "----------------------------------------" << std::endl;
    test_anomaly_metadata();
    test_stress();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All AnomalyMode tests PASSED." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
