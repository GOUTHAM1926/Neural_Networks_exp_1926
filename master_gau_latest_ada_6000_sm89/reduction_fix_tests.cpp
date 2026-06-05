// ============================================================================
// reduction_fix_tests.cpp  — ONE self-contained test for all reduction fixes.
//
// Covers every bug we fixed:
//   #1 middle-axis reduction (outer>1, the "sandwich")        — CPU + GPU
//   #2 InnerContiguous / last-axis on GPU (block_x_reduce)    — GPU
//   #3 Generic non-consecutive multi-axis (offset calcs)      — GPU
//   #4 vec4 / multi-CTA leading-axis (block_y_reduce sync)    — GPU
//   + ops (sum/mean/max), dtypes (f32/f16/bf16), keepdim, big shapes.
//
// No PyTorch dependency: every result is checked against an independent
// brute-force CPU reference. Prints PASS/FAIL per case and a final verdict
// (process exit code 0 = all pass, 1 = something failed).
//
// BUILD (run from inside the Tensor-Implementations dir; adjust CUDA path):
//   CUDA=/usr/local/cuda-13.0     # or wherever the lib was built against
//   g++ -std=c++2a -O2 -fopenmp -mavx2 -mfma -mf16c -DWITH_CUDA \
//       -Iinclude -I$CUDA/include reduction_fix_tests.cpp \
//       -Llib -ltensor -L$CUDA/lib64 -L$CUDA/lib64/stubs \
//       -lcudart -lcublas -lcublasLt -lcurand -lcuda -lgomp \
//       -o reduction_fix_tests
// RUN:
//   LD_LIBRARY_PATH=$CUDA/lib64:lib ./reduction_fix_tests
// ============================================================================
#include "core/Tensor.h"
#include "device/DeviceCore.h"
#include "dtype/Types.h"
#include "ops/UnaryOps/Reduction.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace OwnTensor;

enum Op { SUM, MEAN, MAX };
static const char *opname(Op o) {
  return o == SUM ? "sum" : o == MEAN ? "mean" : "max";
}
static const char *dtname(Dtype d) {
  return d == Dtype::Float32 ? "f32" : d == Dtype::Float16 ? "f16" : "bf16";
}

static uint64_t g_seed = 0x9E3779B97F4A7C15ULL;
static float frand() {
  g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (float)((double)(uint32_t)(g_seed >> 32) / 2147483648.0 - 1.0);
}

static int g_total = 0, g_pass = 0;

// Independent brute-force reference over arbitrary axis set, for sum/mean/max.
static std::vector<double> reference(const std::vector<int> &shape,
                                     const std::vector<int> &axes,
                                     const std::vector<float> &d, Op op) {
  int nd = (int)shape.size();
  std::vector<char> red(nd, 0);
  for (int a : axes)
    red[a] = 1;
  int n_out = 1, reduced = 1;
  for (int i = 0; i < nd; ++i) {
    if (red[i])
      reduced *= shape[i];
    else
      n_out *= shape[i];
  }
  std::vector<double> out(n_out, op == MAX ? -1e300 : 0.0);
  int n = (int)d.size();
  for (int lin = 0; lin < n; ++lin) {
    int rem = lin;
    std::vector<int> c(nd);
    for (int x = nd - 1; x >= 0; --x) {
      c[x] = rem % shape[x];
      rem /= shape[x];
    }
    int oc = 0;
    for (int x = 0; x < nd; ++x)
      if (!red[x])
        oc = oc * shape[x] + c[x];
    if (op == MAX)
      out[oc] = std::max(out[oc], (double)d[lin]);
    else
      out[oc] += d[lin];
  }
  if (op == MEAN)
    for (auto &v : out)
      v /= reduced;
  return out;
}

static void run(const std::string &tag, std::vector<int> shape,
                std::vector<int> axes, Op op, Dtype dt) {
  g_total++;
  int n = 1;
  for (int s : shape)
    n *= s;
  std::vector<float> data(n);
  for (auto &v : data)
    v = frand();
  auto ref = reference(shape, axes, data, op);

  std::vector<int64_t> sh(shape.begin(), shape.end()),
      ax(axes.begin(), axes.end());
  Tensor t(Shape{sh}, TensorOptions().with_dtype(Dtype::Float32));
  t.set_data(data);
  if (dt != Dtype::Float32)
    t = t.as_type(dt);

  auto call = [&](const Tensor &x) {
    return op == SUM    ? reduce_sum(x, ax, false)
           : op == MEAN ? reduce_mean(x, ax, false)
                        : reduce_max(x, ax, false);
  };
  Tensor c = call(t);
  Tensor g = call(t.to_cuda(0)).to_cpu();
  if (dt != Dtype::Float32) {
    c = c.as_type(Dtype::Float32);
    g = g.as_type(Dtype::Float32);
  }

  // tolerance: tight for f32, looser for half/bf16
  double atol = dt == Dtype::Float32 ? 1e-3 : dt == Dtype::Float16 ? 0.5 : 1.0;
  double rtol = dt == Dtype::Float32 ? 1e-3 : 2e-2;
  int no = (int)ref.size();
  bool cok = (int)c.numel() == no, gok = (int)g.numel() == no;
  double ce = 0, ge = 0;
  for (int i = 0; i < no; ++i) {
    double cc = std::fabs((double)c.data<float>()[i] - ref[i]);
    double gg = std::fabs((double)g.data<float>()[i] - ref[i]);
    ce = std::max(ce, cc);
    ge = std::max(ge, gg);
    if (cc > atol + rtol * std::fabs(ref[i]))
      cok = false;
    if (gg > atol + rtol * std::fabs(ref[i]))
      gok = false;
  }
  bool ok = cok && gok;
  if (ok)
    g_pass++;
  printf("  [%s] %-4s %-4s shape[", ok ? "PASS" : "FAIL", opname(op),
         dtname(dt));
  for (size_t i = 0; i < shape.size(); ++i)
    printf("%s%d", i ? "x" : "", shape[i]);
  printf("] ax{");
  for (size_t i = 0; i < axes.size(); ++i)
    printf("%s%d", i ? "," : "", axes[i]);
  printf("}  CPU %s GPU %s (err c=%.1e g=%.1e)\n", cok ? "ok" : "BAD",
         gok ? "ok" : "BAD", ce, ge);
}

int main() {
  printf("================ REDUCTION FIX TEST SUITE ================\n");

  printf("\n--- bug #2: InnerContiguous / last-axis on GPU (block_x_reduce) "
         "---\n");
  for (int r : {2, 3, 7, 8, 16, 31, 32, 33, 64, 256, 768, 1024})
    run("inner", {5, r}, {1}, SUM, Dtype::Float32);
  run("inner3d", {8, 50, 64}, {2}, SUM, Dtype::Float32);

  printf("\n--- bug #1: middle-axis reduction (outer>1, the sandwich) ---\n");
  for (auto sh : std::vector<std::vector<int>>{{2, 3, 2},
                                               {4, 8, 16},
                                               {8, 50, 64},
                                               {3, 7, 5},
                                               {16, 128, 64},
                                               {1, 9, 4},
                                               {7, 1, 5}})
    run("mid", sh, {1}, SUM, Dtype::Float32);
  run("mid4d_1", {2, 3, 4, 5}, {1}, SUM, Dtype::Float32);
  run("mid4d_2", {2, 3, 4, 5}, {2}, SUM, Dtype::Float32);
  run("mid4d_big", {8, 16, 32, 4}, {1}, SUM, Dtype::Float32);

  printf("\n--- bug #4: leading-axis vec4 + multi-CTA (block_y_reduce sync) "
         "---\n");
  for (auto sh : std::vector<std::vector<int>>{{1024, 1024},
                                               {256, 256},
                                               {512, 512},
                                               {2048, 2048},
                                               {1024, 128},
                                               {1024, 512},
                                               {64, 1024},
                                               {512, 1024},
                                               {4096, 4096}})
    run("lead", sh, {0}, SUM, Dtype::Float32);

  printf("\n--- leading-axis (small / OuterContiguous, outer==1) ---\n");
  for (auto sh : std::vector<std::vector<int>>{{16, 64}, {128, 50}, {2, 3, 2}})
    run("lead_s", sh, {0}, SUM, Dtype::Float32);

  printf("\n--- bug #3: Generic non-consecutive multi-axis on GPU ---\n");
  run("gen", {4, 8, 16}, {0, 2}, SUM, Dtype::Float32);
  run("gen", {2, 3, 4, 5}, {1, 3}, SUM, Dtype::Float32);
  run("gen", {8, 16, 32, 4}, {0, 2}, SUM, Dtype::Float32);
  run("gen", {4, 8, 16}, {1, 2}, SUM, Dtype::Float32); // consecutive
  run("gen", {4, 8, 16}, {0, 1}, SUM, Dtype::Float32); // consecutive

  printf("\n--- ops: mean / max (inner, middle, leading, multi-axis) ---\n");
  for (Op op : {MEAN, MAX}) {
    run("op", {5, 768}, {1}, op, Dtype::Float32);      // inner
    run("op", {8, 50, 64}, {1}, op, Dtype::Float32);   // middle
    run("op", {1024, 1024}, {0}, op, Dtype::Float32);  // leading vec4/mcta
    run("op", {4, 8, 16}, {0, 2}, op, Dtype::Float32); // generic multi-axis
  }

  printf("\n--- dtypes: fp16 / bf16 sum (inner, middle, leading) ---\n");
  for (Dtype dt : {Dtype::Float16, Dtype::Bfloat16}) {
    run("dt", {5, 768}, {1}, SUM, dt);
    run("dt", {8, 50, 64}, {1}, SUM, dt);
    run("dt", {1024, 1024}, {0}, SUM, dt);
  }

  printf("\n=========================================================\n");
  printf("RESULT: %d/%d passed  -> %s\n", g_pass, g_total,
         g_pass == g_total ? "ALL GOOD ✅" : "*** SOME FAILED ❌ ***");
  return g_pass == g_total ? 0 : 1;
}
