// Pins down one thing only, which is the order Eigen walks the reduced axes in.
//
// The case is a 3-D row-major [X,Y,Z] reduced over axes {0,2}, keeping Y, and TF
// hands that straight to Eigen with no transpose and no copy on the way --
// reduction_ops_common.h:209-213
//   } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
//       Functor::Reduce(ctx, helper.out<T,1>(&tmp_out), helper.in<T,3>(data),
//                       constants.kZeroTwo, reducer);
//
// Worth knowing before you edit anything, the axis sets go in as
// Eigen::IndexList<type2index<...>> and not Eigen::array, which matters more than
// it looks, since Eigen picks its path using are_inner_most_dims and
// preserve_inner_most_dims, both compile-time traits, and both come out false for
// a runtime Eigen::array, so swapping one in quietly drops every axis set onto the
// generic path and the whole comparison stops saying anything, and TF's CPU path
// uses the IndexList form anyway (Constants<CPUDevice> : ConstantsBase,
// reduction_ops_common.h:61-67), so that's what's matched here.
//
// Three arms, all producing the same numbers:
//   A  Eigen's own in.sum({0,2})        the TF path
//   B  hand loop, Z innermost           stride 1, the way PyTorch would order it
//   C  hand loop, X innermost           stride Y*Z, walking like GenericDimReducer
//
// C is the interesting one, because if A and C land on the same time then Eigen
// isn't slow for being Eigen, it's slow because of the order, and B says what you
// get back by turning the loop around.
//
// Two controls run alongside, other axis sets on the same tensor with the same
// element count, so size isn't the variable:
//   {1,2} keep X  -> are_inner_most_dims true      -> InnerMostDimReducer
//   {0,1} keep Z  -> preserve_inner_most_dims true -> InnerMostDimPreserver
// Both of those are fast Eigen paths, and {0,2} is the only set that's false for
// both and ends up on GenericDimReducer, and the program prints those two flags
// per axis set so you can see the path choice rather than assume it.
//
// Build:  g++ -O3 -march=native -std=c++17 -I<eigen> eigen_3dxz_bench.cpp -o bench
// Single threaded on purpose (Eigen DefaultDevice), same as TF intra_op = 1.

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::steady_clock;
using Tensor3 = Eigen::Tensor<float, 3, Eigen::RowMajor>;   // TF uses RowMajor
using Tensor1 = Eigen::Tensor<float, 1, Eigen::RowMajor>;

// the three axis sets, in TF's own compile-time form
using AxesXZ = Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>>;  // keep Y
using AxesYZ = Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2>>;  // keep X
using AxesXY = Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1>>;  // keep Z

template <typename Axes>
static void print_path(const char* label) {
  const bool inner = Eigen::internal::are_inner_most_dims<Axes, 3, Eigen::RowMajor>::value;
  const bool presv = Eigen::internal::preserve_inner_most_dims<Axes, 3, Eigen::RowMajor>::value;
  printf("  %-22s are_inner_most_dims=%-5s preserve_inner_most_dims=%-5s -> %s\n",
         label, inner ? "true" : "false", presv ? "true" : "false",
         inner ? "InnerMostDimReducer" : (presv ? "InnerMostDimPreserver" : "GenericDimReducer"));
}

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

template <typename F>
static double timed(int reps, F&& f) {
  f();                                   // warm the pages
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) { auto t = Clock::now(); f(); v.push_back(ms_since(t)); }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];                // median
}

static void run_case(int X, int Y, int Z, int reps) {
  Tensor3 in(X, Y, Z);
  float* p = in.data();
  const long long N = (long long)X * Y * Z;
  for (long long i = 0; i < N; ++i) p[i] = float((i * 2654435761u) % 1000) * 0.001f;

  AxesXZ axes_xz; AxesYZ axes_yz; AxesXY axes_xy;

  Tensor1 outA(Y);
  double tA = timed(reps, [&] { outA = in.sum(axes_xz); });

  std::vector<float> outB(Y);                      // Z innermost, stride 1
  double tB = timed(reps, [&] {
    for (int y = 0; y < Y; ++y) {
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      for (int x = 0; x < X; ++x) {
        const float* row = p + (long long)x * Y * Z + (long long)y * Z;
        int z = 0;
        for (; z + 3 < Z; z += 4) { a0 += row[z]; a1 += row[z+1]; a2 += row[z+2]; a3 += row[z+3]; }
        for (; z < Z; ++z) a0 += row[z];
      }
      outB[y] = a0 + a1 + a2 + a3;
    }
  });

  std::vector<float> outC(Y);                      // X innermost, stride Y*Z
  double tC = timed(reps, [&] {
    for (int y = 0; y < Y; ++y) {
      float acc = 0.f;
      for (int z = 0; z < Z; ++z) {
        const float* col = p + (long long)y * Z + z;
        for (int x = 0; x < X; ++x) acc += col[(long long)x * Y * Z];
      }
      outC[y] = acc;
    }
  });

  Tensor1 outD(X); double tD = timed(reps, [&] { outD = in.sum(axes_yz); });   // control
  Tensor1 outE(Z); double tE = timed(reps, [&] { outE = in.sum(axes_xy); });   // control

  double maxrel = 0.0;
  for (int y = 0; y < Y; ++y) {
    double ref = outB[y];
    maxrel = std::max(maxrel,
              std::max(std::fabs(outA(y) - ref) / (std::fabs(ref) + 1e-9),
                       std::fabs(outC[y]  - ref) / (std::fabs(ref) + 1e-9)));
  }

  const double gb = double(N) * sizeof(float) / 1e9;
  printf("%5d %5d %5d | %9.2f %9.2f %9.2f | %6.2fx %6.2fx | %8.2f %8.2f | %5.1f %5.1f | %.0e\n",
         X, Y, Z, tA, tB, tC, tA / tB, tC / tB, tD, tE,
         gb / (tA / 1e3), gb / (tB / 1e3), maxrel);
}

int main(int argc, char** argv) {
  int reps = (argc > 1) ? atoi(argv[1]) : 9;
  printf("Eigen %d.%d.%d | single-threaded DefaultDevice | float32 RowMajor | median of %d\n\n",
         EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION, reps);

  printf("Eigen's compile-time path choice for a rank-3 RowMajor tensor:\n");
  print_path<AxesXZ>("{0,2} keep Y  (TF)");
  print_path<AxesYZ>("{1,2} keep X  (ctrl)");
  print_path<AxesXY>("{0,1} keep Z  (ctrl)");

  printf("\n    X     Y     Z |  A:Eigen   B:Z-in    C:X-in   |   A/B     C/B  | ctl{1,2} ctl{0,1} | GB/s  GB/s | maxrel\n");
  printf("                  |     (ms)     (ms)     (ms)   |                |     (ms)     (ms) |    A     B |\n");
  printf("------------------+------------------------------+----------------+-------------------+------------+-------\n");
  run_case( 512,   8,  512, reps);
  run_case( 512,  16,  512, reps);
  run_case( 256,  64,  256, reps);
  run_case(1024,   8, 1024, reps);
  run_case( 128, 256,  128, reps);
  run_case(  64, 512,   64, reps);
  return 0;
}
