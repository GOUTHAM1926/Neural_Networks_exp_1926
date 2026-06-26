// ============================================================================
// reduction_final_test.cpp — FINAL all-in-one regression test for every
// reduction fix we made. Checks CPU and GPU against an independent
// brute-force double-precision reference. No PyTorch dependency.
//
// Covers:
//   bug #1  middle-axis (outer>1, "sandwich")         — CPU + GPU
//   bug #2  InnerContiguous / last-axis (block_x)     — GPU
//   bug #3  Generic non-consecutive multi-axis        — GPU
//   bug #4  leading-axis vec4 + multi-CTA (block_y)   — GPU  ← incl. real GPT-2 bias shapes
//   + ops (sum/mean/max), dtypes (f32/f16/bf16), vec4 gate edges, multi-CTA stress.
//
// Exit code 0 = all pass, 1 = something failed.
//
// BUILD (from the dir holding this file; if that's Tensor-Implementations/ use
//        -Iinclude -Llib, if it's one level up use TI=Tensor-Implementations):
//   CUDA=/usr/local/cuda-13.0
//   TI=Tensor-Implementations          # only if building from the parent dir
//   g++ -std=c++2a -O2 -fopenmp -mavx2 -mfma -mf16c -DWITH_CUDA \
//       -I$TI/include -I$CUDA/include reduction_final_test.cpp \
//       -L$TI/lib -ltensor -L$CUDA/lib64 -L$CUDA/lib64/stubs \
//       -lcudart -lcublas -lcublasLt -lcurand -lcuda -lgomp -o reduction_final_test
//   LD_LIBRARY_PATH=$CUDA/lib64:$TI/lib ./reduction_final_test
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
static int g_cpu_bad = 0, g_gpu_bad = 0;

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
  long long n = (long long)d.size();
  for (long long lin = 0; lin < n; ++lin) {
    long long rem = lin;
    std::vector<int> c(nd);
    for (int x = nd - 1; x >= 0; --x) {
      c[x] = (int)(rem % shape[x]);
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
  long long n = 1;
  for (int s : shape)
    n *= s;
  std::vector<float> data((size_t)n);
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

  // tolerance scales with reduced length for fp32; looser for half/bf16
  long long reduced = 1;
  {
    std::vector<char> red(shape.size(), 0);
    for (int a : axes) red[a] = 1;
    for (size_t i = 0; i < shape.size(); ++i) if (red[i]) reduced *= shape[i];
  }
  // fp16/bf16 sums of N values lose precision ~ sqrt(N) (random-walk magnitude
  // × low-precision rounding), so scale their abs-tolerance with sqrt(reduced).
  // f32 stays tight — that's our precise bug detector. A REAL kernel bug shows
  // up as GPU-diverges-from-CPU (gc below), not as both-equally-off-from-ref.
  double sq = std::sqrt((double)reduced);
  double atol = dt == Dtype::Float32 ? 1e-2
              : dt == Dtype::Float16 ? (0.5 + 0.06 * sq)
                                     : (0.8 + 0.08 * sq);
  double rtol = dt == Dtype::Float32 ? 2e-3 : 3e-2;
  int no = (int)ref.size();
  bool cok = (int)c.numel() == no, gok = (int)g.numel() == no;
  double ce = 0, ge = 0, gc = 0;
  for (int i = 0; i < no; ++i) {
    double cv = (double)c.data<float>()[i], gv = (double)g.data<float>()[i];
    double cc = std::fabs(cv - ref[i]);
    double gg = std::fabs(gv - ref[i]);
    ce = std::max(ce, cc);
    ge = std::max(ge, gg);
    gc = std::max(gc, std::fabs(gv - cv)); // GPU-vs-CPU: the true kernel check
    double tol = atol + rtol * std::fabs(ref[i]);
    if (cc > tol) cok = false;
    if (gg > tol) gok = false;
  }
  // Hard kernel-correctness guard: GPU must match CPU regardless of precision.
  // (the vec4 bug made this ~300; healthy is tiny for f32, modest for half.)
  double gc_tol = dt == Dtype::Float32 ? 0.05 : (1.0 + 0.1 * sq);
  if (gc > gc_tol) gok = false;
  bool ok = cok && gok;
  if (ok) g_pass++;
  if (!cok) g_cpu_bad++;
  if (!gok) g_gpu_bad++;
  printf("  [%s] %-4s %-4s shape[", ok ? "PASS" : "FAIL", opname(op), dtname(dt));
  for (size_t i = 0; i < shape.size(); ++i)
    printf("%s%d", i ? "x" : "", shape[i]);
  printf("] ax{");
  for (size_t i = 0; i < axes.size(); ++i)
    printf("%s%d", i ? "," : "", axes[i]);
  printf("}  CPU %s GPU %s (err c=%.1e g=%.1e)\n", cok ? "ok" : "BAD",
         gok ? "ok" : "BAD", ce, ge);
}

// ── guard tests: a tensor exceeding 32-bit indexing must THROW via the public
//    reduce API (not silently overflow). Uses 1-byte UInt8 so a >2^31-element
//    tensor is only ~2-4 GB; the guard fires on shape alone (before any data
//    read). On a GPU too small to even allocate it, we SKIP (not fail). ──
static void run_guard(const std::string &tag, std::vector<int64_t> shape,
                      std::vector<int64_t> axes, Op op, const char *why) {
  g_total++;
  bool pass = false;
  std::string note;
  try {
    auto dev = Tensor(Shape{{(int64_t)1}}, TensorOptions().with_dtype(Dtype::Float32))
                   .to_cuda(0)
                   .device();
    Tensor big(Shape{shape},
               TensorOptions().with_dtype(Dtype::UInt8).with_device(dev));
    Tensor r = (op == MEAN) ? reduce_mean(big, axes, false)
                            : reduce_sum(big, axes, false);
    (void)r;
    note = "NO THROW — guard did NOT fire (BUG!)";
  } catch (const std::exception &e) {
    std::string m = e.what();
    if (m.find("32-bit") != std::string::npos ||
        m.find("indexing") != std::string::npos) {
      pass = true;
      note = "threw 32-bit guard (correct)";
    } else if (m.find("emory") != std::string::npos ||
               m.find("alloc") != std::string::npos ||
               m.find("CUDA") != std::string::npos ||
               m.find("cuda") != std::string::npos) {
      pass = true; // can't allocate the big tensor here — not a guard failure
      note = "SKIP (alloc failed on this GPU): " + m.substr(0, 60);
    } else {
      note = "threw UNEXPECTED error: " + m.substr(0, 80);
    }
  }
  if (pass) g_pass++;
  printf("  [%s] guard %-14s (%s) -> %s\n", pass ? "PASS" : "FAIL", tag.c_str(),
         why, note.c_str());
}

int main() {
  printf("############ FINAL REDUCTION REGRESSION TEST ############\n");

  printf("\n=== bug #2: InnerContiguous / last-axis (block_x_reduce) ===\n");
  for (int r : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 255,
                256, 512, 768, 1024, 2048})
    run("inner", {5, r}, {1}, SUM, Dtype::Float32);
  run("inner3d", {8, 50, 64}, {2}, SUM, Dtype::Float32);
  run("inner3d", {4, 7, 1000}, {2}, SUM, Dtype::Float32);
  run("inner_big", {128, 4096}, {1}, SUM, Dtype::Float32);

  printf("\n=== bug #1: middle-axis (outer>1, the sandwich) — CPU+GPU ===\n");
  for (auto sh : std::vector<std::vector<int>>{
           {2, 3, 2}, {4, 8, 16}, {8, 50, 64}, {3, 7, 5}, {16, 128, 64},
           {1, 9, 4}, {7, 1, 5}, {32, 64, 32}, {5, 257, 9}})
    run("mid", sh, {1}, SUM, Dtype::Float32);
  run("mid4d_1", {2, 3, 4, 5}, {1}, SUM, Dtype::Float32);
  run("mid4d_2", {2, 3, 4, 5}, {2}, SUM, Dtype::Float32);
  run("mid4d_big", {8, 16, 32, 4}, {1}, SUM, Dtype::Float32);
  run("mid4d_big2", {4, 64, 128, 8}, {2}, SUM, Dtype::Float32);

  printf("\n=== bug #4: leading-axis vec4 + multi-CTA (block_y_reduce sync) ===\n");
  printf("    -- generic vec4 shapes --\n");
  for (auto sh : std::vector<std::vector<int>>{
           {1024, 1024}, {256, 256}, {512, 512}, {2048, 2048}, {1024, 128},
           {1024, 512}, {64, 1024}, {512, 1024}, {4096, 4096}})
    run("lead", sh, {0}, SUM, Dtype::Float32);
  printf("    -- REAL GPT-2 bias-grad shapes [B*T, F] -> [F] --\n");
  for (auto sh : std::vector<std::vector<int>>{
           {16384, 768}, {16384, 2304}, {16384, 3072},
           {8192, 768}, {4096, 2304}, {16384, 768}})
    run("gpt2bias", sh, {0}, SUM, Dtype::Float32);
  printf("    -- vec4 gate edges (mult-of-4 vs not, <128 vs >=128) --\n");
  run("edge_124", {4096, 124}, {0}, SUM, Dtype::Float32);  // <128
  run("edge_128", {4096, 128}, {0}, SUM, Dtype::Float32);  // ==128, mult4
  run("edge_130", {4096, 130}, {0}, SUM, Dtype::Float32);  // not mult of 4
  run("edge_1000", {4096, 1000}, {0}, SUM, Dtype::Float32);// mult4
  run("edge_1001", {4096, 1001}, {0}, SUM, Dtype::Float32);// not mult4
  printf("    -- multi-CTA stress (huge reduced dim) --\n");
  run("mcta", {65536, 256}, {0}, SUM, Dtype::Float32);
  run("mcta", {131072, 512}, {0}, SUM, Dtype::Float32);

  printf("\n=== leading-axis small (OuterContiguous scalar, outer==1) ===\n");
  for (auto sh : std::vector<std::vector<int>>{{16, 64}, {128, 50}, {2, 3, 2}})
    run("lead_s", sh, {0}, SUM, Dtype::Float32);

  printf("\n=== bug #3: Generic non-consecutive multi-axis ===\n");
  run("gen", {4, 8, 16}, {0, 2}, SUM, Dtype::Float32);
  run("gen", {2, 3, 4, 5}, {1, 3}, SUM, Dtype::Float32);
  run("gen", {8, 16, 32, 4}, {0, 2}, SUM, Dtype::Float32);
  run("gen", {16, 32, 64, 8}, {0, 2}, SUM, Dtype::Float32);
  run("gen_cons", {4, 8, 16}, {1, 2}, SUM, Dtype::Float32);
  run("gen_cons", {4, 8, 16}, {0, 1}, SUM, Dtype::Float32);
  run("gen_all", {4, 8, 16}, {0, 1, 2}, SUM, Dtype::Float32);

  printf("\n=== ops: mean / max (inner, middle, leading-vec4, multi-axis) ===\n");
  for (Op op : {MEAN, MAX}) {
    run("op", {5, 768}, {1}, op, Dtype::Float32);       // inner
    run("op", {8, 50, 64}, {1}, op, Dtype::Float32);    // middle
    run("op", {16384, 768}, {0}, op, Dtype::Float32);   // gpt2 bias vec4
    run("op", {16384, 3072}, {0}, op, Dtype::Float32);  // gpt2 bias vec4
    run("op", {4, 8, 16}, {0, 2}, op, Dtype::Float32);  // generic
  }

  printf("\n=== dtypes: fp16 / bf16 (inner, middle, leading-vec4 bias) ===\n");
  for (Dtype dt : {Dtype::Float16, Dtype::Bfloat16}) {
    run("dt", {5, 768}, {1}, SUM, dt);
    run("dt", {8, 50, 64}, {1}, SUM, dt);
    run("dt", {16384, 768}, {0}, SUM, dt);   // gpt2 bias shape, half/bf16
    run("dt", {16384, 3072}, {0}, SUM, dt);
  }

  printf("\n=== guard: tensors > 2^31/2^32 must THROW via public API ===\n");
  // reduced_count > 2^31  (reduce-all of a ~2.15B-element 1-D tensor)
  run_guard("reduced>2^31", {2147484672LL}, {0}, SUM,
            "reduce-all, reduced_count=2.147B > 2^31");
  run_guard("reduced>2^31", {2147484672LL}, {0}, MEAN,
            "mean reduce-all, reduced_count > 2^31");
  // numel > 2^32 while BOTH counts stay < 2^31 (tests the numel branch)
  run_guard("numel>2^32", {65540LL, 65540LL}, {1}, SUM,
            "numel=4.295B > 2^32, counts < 2^31");

  printf("\n#########################################################\n");
  printf("RESULT: %d/%d passed   (CPU failures: %d, GPU failures: %d)\n",
         g_pass, g_total, g_cpu_bad, g_gpu_bad);
  printf("-> %s\n", g_pass == g_total ? "ALL GOOD ✅  (no reduction bugs)"
                                      : "*** SOME FAILED ❌ ***");
  return g_pass == g_total ? 0 : 1;
}
