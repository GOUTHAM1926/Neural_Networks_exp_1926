#include <chrono>
#include <iostream>
#include <vector>

int main() {
  const int N = 1000000;  // 1 Million (Fits in CPU Cache)
  const int ITERS = 1000; // Repeat to get stable measurement

  // --- SETUP DATA ---
  std::vector<float> data_f(N, 123.456f);
  std::vector<float> res_f(N);
  std::vector<double> data_d(N, 123.4567890123456789);
  std::vector<double> res_d(N);

  float div_f = 7.89f;
  float rec_f = 1.0f / div_f;
  double div_d = 7.89;
  double rec_d = 1.0 / div_d;

  std::cout << "Starting Professional Architectural Benchmark...\n";

  // --- STEP 1: WARM-UP ---
  for (int i = 0; i < N; ++i) {
    res_f[i] = data_f[i] * 0.5f;
    res_d[i] = data_d[i] * 0.5;
  }
  std::cout << "Warm-up complete. CPU in High-Performance mode.\n";
  std::cout << "-----------------------------------------------\n";

  // --- STEP 2: FLOAT DIVISION (Standard) ---
  auto s2 = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < ITERS; ++j) {
    for (int i = 0; i < N; ++i)
      res_f[i] = data_f[i] / div_f;
    if (res_f[0] < 0)
      std::cout << " ";
  }
  auto e2 = std::chrono::high_resolution_clock::now();

  // --- STEP 3: FLOAT RECIPROCAL (Optimized) ---
  auto s3 = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < ITERS; ++j) {
    for (int i = 0; i < N; ++i)
      res_f[i] = data_f[i] * rec_f;
    if (res_f[0] < 0)
      std::cout << " ";
  }
  auto e3 = std::chrono::high_resolution_clock::now();

  // --- STEP 4: DOUBLE DIVISION (Standard) ---
  auto s4 = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < ITERS; ++j) {
    for (int i = 0; i < N; ++i)
      res_d[i] = data_d[i] / div_d;
    if (res_d[0] < 0)
      std::cout << " ";
  }
  auto e4 = std::chrono::high_resolution_clock::now();

  // --- STEP 5: DOUBLE RECIPROCAL (Optimized) ---
  auto s5 = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < ITERS; ++j) {
    for (int i = 0; i < N; ++i)
      res_d[i] = data_d[i] * rec_d;
    if (res_d[0] < 0)
      std::cout << " ";
  }
  auto e5 = std::chrono::high_resolution_clock::now();

  // --- RESULTS ---
  double t_f_div = std::chrono::duration<double, std::milli>(e2 - s2).count();
  double t_f_rec = std::chrono::duration<double, std::milli>(e3 - s3).count();
  double t_d_div = std::chrono::duration<double, std::milli>(e4 - s4).count();
  double t_d_rec = std::chrono::duration<double, std::milli>(e5 - s5).count();

  std::cout << "FLOAT Logic (32-bit):\n";
  std::cout << "  Division:    " << t_f_div << " ms\n";
  std::cout << "  Reciprocal:  " << t_f_rec << " ms (" << t_f_div / t_f_rec
            << "x faster)\n\n";

  std::cout << "DOUBLE Logic (64-bit):\n";
  std::cout << "  Division:    " << t_d_div << " ms\n";
  std::cout << "  Reciprocal:  " << t_d_rec << " ms (" << t_d_div / t_d_rec
            << "x faster)\n";
  std::cout << "-----------------------------------------------\n";
  std::cout << "Double-precision division is the biggest legacy bottleneck!\n";

  return 0;
}

/*
 * Execution Results on Local System (Compiled with g++ -O3):
 *
 * FLOAT Logic (32-bit):
 *   Division:    138.083 ms
 *   Reciprocal:  118.081 ms (1.16939x faster)
 *
 * DOUBLE Logic (64-bit):
 *   Division:    367.979 ms
 *   Reciprocal:  240.217 ms (1.53186x faster)
 *
 * -----------------------------------------------
 *
 */
