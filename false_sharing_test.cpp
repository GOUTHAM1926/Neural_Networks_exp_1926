#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>

const int NUM_UPDATES = 100000000;
const int NUM_THREADS = 4;

int main() {
    // 1 float = 4 bytes. A standard CPU cache line is 64 bytes = 16 floats.
    // Using volatile to prevent the compiler from optimizing the memory writes into a register!
    volatile float data[NUM_THREADS * 64] = {0};

    std::cout << "Starting hardware cache tests...\n";

    // TEST 1: FALSE SHARING (Adjacent slots, same cache line)
    auto start1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < NUM_UPDATES; ++i) {
            data[t] += 1.0f; // t=0 writes data[0], t=1 writes data[1], etc.
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1 - start1;
    std::cout << "Time WITH traffic jams (False Sharing - writing to adjacent slots): " << diff1.count() << " seconds\n";

    // TEST 2: NO FALSE SHARING (Distant slots, different cache lines)
    auto start2 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (int i = 0; i < NUM_UPDATES; ++i) {
            data[t * 16] += 1.0f; // t=0 writes data[0], t=1 writes data[16] (64 bytes away!)
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = end2 - start2;
    std::cout << "Time WITHOUT traffic jams (Independent cache lines): " << diff2.count() << " seconds\n";

    return 0;
}
