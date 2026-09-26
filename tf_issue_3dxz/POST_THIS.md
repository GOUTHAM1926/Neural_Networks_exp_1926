Hi @Venkat6871, thanks for looking at it, i ran the benchmarks you asked for and everything is below.

One thing before the numbers, because my own issue title is confusing, TF's code calls this case `Launch3DXZReduction` (reduce axes {0,2}, keep the middle dimension), while `Launch3DYReduction` is a different case that reduces dim 1, so my title saying "3D-Y" meaning "we keep Y" reads like the wrong one, and everything below is `reduction_axes = {0,2}`.

## Setup

```
CPU:      Intel Core i7-14700K, 20 cores / 28 threads (8P + 12E), AVX2, no AVX-512
OS:       Linux 6.8.0-139-generic
Python:   3.12.3
TF:       2.21.0 (tensorflow-cpu wheel) and 2.22.0-dev20260905 (tf-nightly-cpu)
PyTorch:  2.14.0+cpu, reference only
Threads:  intra_op = 1, and the default, both below
dtype:    float32, median of 11 runs
```

I pinned the runs with `taskset` to the P-cores, since this chip mixes fast and slow cores and without pinning the times move around enough to be annoying.

## 1. Benchmark at the TF level

Script is `tf_3dxz_bench.py`, attached, and it times `axis=[0,2]` plus two other axis sets on the same tensor, so the data and the number of elements added up stay the same and only the axis set changes, and every result is checked against numpy.

TF 2.21.0, `intra_op=1`, times in ms:

| shape | `axis=[0,2]` | `axis=[1,2]` | `axis=[0,1]` | torch `[0,2]` |
| --- | ---: | ---: | ---: | ---: |
| (512, 8, 512) | 4.75 | 0.15 | 0.58 | 0.17 |
| (512, 16, 512) | 11.40 | 0.29 | 1.05 | 0.34 |
| (256, 64, 256) | 9.87 | 0.29 | 0.89 | 0.37 |
| (1024, 8, 1024) | 36.96 | 0.74 | 4.37 | 0.80 |
| (128, 256, 128) | 8.73 | 0.29 | 0.97 | 0.42 |
| (64, 512, 64) | 2.88 | 0.15 | 0.46 | 0.29 |

The PyTorch column is 10x to 46x, but i don't think that's the useful comparison, the useful one is inside TF itself, and in the last row reducing {1,2} takes 0.74 ms and reducing {0,2} takes 36.96 ms, same tensor, same binary, same amount of arithmetic, 50x apart, so it isn't my CPU or the wheel's build flags or the tensor size.

tf-nightly 2.22.0-dev20260905, `intra_op=1`, same thing, so it isn't already fixed on master:

| shape | `axis=[0,2]` | `axis=[1,2]` | torch `[0,2]` |
| --- | ---: | ---: | ---: |
| (512, 8, 512) | 4.45 | 0.12 | 0.13 |
| (512, 16, 512) | 8.63 | 0.24 | 0.22 |
| (1024, 8, 1024) | 23.79 | 0.72 | 0.79 |

Default thread count, TF 2.21.0, threads don't help because the other axis sets get faster and {0,2} doesn't:

| shape | `axis=[0,2]` | `axis=[1,2]` | torch `[0,2]` |
| --- | ---: | ---: | ---: |
| (512, 8, 512) | 3.61 | 0.07 | 0.06 |
| (1024, 8, 1024) | 21.51 | 0.22 | 0.25 |

The two smallest shapes move around a bit between runs because of the P/E cores, so the 512 and 1024 rows are the steady ones and those are what i'd trust.

## 2. Same test in Eigen alone

The numbers above show there's a cost but not what causes it, so i ran Eigen directly with no TF involved, file is `eigen_3dxz_bench.cpp`, Eigen 3.5.0, built with `-O3 -march=native`, single threaded, and there are three versions that all compute the same result:

- **A** — `in.sum(axes{0,2})`, what TF calls
- **B** — my own loop, innermost loop over Z, which is the stride-1 axis
- **C** — my own loop, innermost loop over X, stride Y*Z, written to copy what `GenericDimReducer` does

The program also prints which Eigen path each axis set compiles to, so it isn't a guess:

```
{0,2} keep Y  (TF)     are_inner_most_dims=false preserve_inner_most_dims=false -> GenericDimReducer
{1,2} keep X  (ctrl)   are_inner_most_dims=true  preserve_inner_most_dims=false -> InnerMostDimReducer
{0,1} keep Z  (ctrl)   are_inner_most_dims=false preserve_inner_most_dims=true  -> InnerMostDimPreserver

    X     Y     Z |  A:Eigen   B:Z-in    C:X-in   |   A/B     C/B  | ctl{1,2} ctl{0,1} | maxrel
  512     8   512 |     2.09     0.39     2.09    |  5.30x   5.32x |     0.13     0.59 | 2e-07
  512    16   512 |     4.33     0.77     4.37    |  5.61x   5.66x |     0.22     1.23 | 2e-07
  256    64   256 |     2.00     0.75     1.99    |  2.66x   2.66x |     0.21     1.23 | 4e-07
 1024     8  1024 |    20.51     1.95    21.70    | 10.54x  11.16x |     0.75     4.27 | 1e-07
  128   256   128 |     1.84     0.79     1.77    |  2.33x   2.23x |     0.22     0.95 | 6e-07
   64   512    64 |     0.84     0.37     0.87    |  2.31x   2.38x |     0.10     0.47 | 4e-07
```

A and C come out the same in every row, 2.09 and 2.09, 4.33 and 4.37, 20.51 and 21.70, so a plain C loop written to walk in Eigen's order is exactly as slow as Eigen is, which means this isn't template or dispatch overhead, it's the loop order, and turning the same loop around (B) means and it's 2.3x to 11x faster on the same data, and `maxrel` is around 1e-7 which is just float summation order.

If you reproduce this, one thing to watch, the axes have to be passed as `Eigen::IndexList<type2index<...>>` and not `Eigen::array`, because `are_inner_most_dims` and `preserve_inner_most_dims` are compile-time traits and come out false for a runtime `Eigen::array`, so an array sends every axis set to the generic path and the comparison shows nothing, i did exactly that on my first attempt and got a useless result, and TF's CPU path uses the IndexList form anyway (`Constants<CPUDevice> : ConstantsBase`, reduction_ops_common.h:61-67), so that's what i matched.

## 3. Where it comes from

For a row-major `[X,Y,Z]` reducing `{0,2}`, `TensorReduction.h:639-650` fills `m_reducedStrides` in input dimension order:

```
m_reducedStrides[0] = Y*Z    (X)
m_reducedStrides[1] = 1      (Z)
```

`GenericDimReducer` (`TensorReduction.h:131-150`) recurses from `NumReducedDims-1` down to `0`, so slot 0 is the innermost loop, which is X striding Y*Z per element, and Z the stride-1 axis ends up as the outer loop, meaning every element read pulls a new cache line and uses one value out of it.

Two more things on the same path, `ReducingInnerMostDims` and `PreservingInnerMostDims` are both false so `packet()` drops into the last `else` at `TensorReduction.h:849-852` and just calls `coeff()` PacketSize times, meaning no vectorization, and `BlockAccess` is false on the reduction evaluator (`TensorReduction.h:572`) so there's no tiled path either.

For comparison, PyTorch's `TensorIterator::reorder_dimensions` (`TensorIterator.cpp:232-236`) sorts dimensions by stride ascending, so the stride-1 axis is innermost no matter which axes are reduced, which is why its three columns above are flat and TF's aren't.

## 4. GPU is fine

Just to be clear on scope, this is CPU only, `ReduceImpl` in `reduction_gpu_kernels.cu.h` has a branch for `in_rank==3 && out_rank==1 && axes=={0,2}` that calls `Launch3DXZReduction`, and that uses CUB `DeviceSegmentedReduce` with a gather iterator whose index advances along Z first, so reads are coalesced and the order is right, no Eigen, and only the `ReduceEigenImpl` path in `reduction_ops.h:53-58` is affected.

## The benchmarks

Both attached as a zip with the raw output, and inlined here so you don't have to download anything, and the Eigen one only needs the Eigen headers with no TF build.

<details>
<summary><b>eigen_3dxz_bench.cpp</b> — <code>g++ -O3 -march=native -std=c++17 -I&lt;eigen&gt; eigen_3dxz_bench.cpp -o bench</code></summary>

```cpp
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
```

</details>

<details>
<summary><b>tf_3dxz_bench.py</b> — <code>python3 tf_3dxz_bench.py --threads 1</code></summary>

```python
#!/usr/bin/env python3
"""
Times tf.reduce_sum(x, axis=[0,2]) on a 3-D tensor, which is the case that ends up
in Eigen's GenericDimReducer via reduction_ops_common.h:209-213.

Two other axis sets run alongside it on the same tensor with the same number of
elements reduced, [1,2] which Eigen sends to InnerMostDimReducer and [0,1] which
goes to InnerMostDimPreserver, so if [0,2] is the only slow one then the axis set
is the variable and not the size of the problem.

PyTorch gets timed on the same shapes when it's importable, as a reference for what
this costs with stride-ordered traversal.

Run:
  python3 tf_3dxz_bench.py --threads 1
  python3 tf_3dxz_bench.py --threads 0      # 0 leaves the framework default
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1,
                help="intra_op threads; 0 leaves the framework default")
ap.add_argument("--reps", type=int, default=9)
args = ap.parse_args()

# must be set before TF is imported
if args.threads > 0:
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

import numpy as np
import tensorflow as tf

if args.threads > 0:
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    import torch
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

SHAPES = [(512, 8, 512), (512, 16, 512), (256, 64, 256),
          (1024, 8, 1024), (128, 256, 128), (64, 512, 64)]
AXES = [[0, 2], [1, 2], [0, 1]]


def bench(fn, reps):
    for _ in range(3):                     # warm up: trace + thread-pool spin-up
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]                # median


print(f"TF {tf.__version__} | numpy {np.__version__} | "
      f"torch {torch.__version__ if HAVE_TORCH else 'not installed'}")
print(f"intra_op={tf.config.threading.get_intra_op_parallelism_threads()} "
      f"inter_op={tf.config.threading.get_inter_op_parallelism_threads()} "
      f"| devices={[d.device_type for d in tf.config.list_physical_devices()]}")
print(f"float32, median of {args.reps}, times in ms\n")

hdr = f"{'shape':>18} | " + " | ".join(f"TF ax={a}".rjust(13) for a in AXES)
if HAVE_TORCH:
    hdr += " | " + " | ".join(f"pt ax={a}".rjust(13) for a in AXES)
hdr += " | TF[0,2]/pt[0,2]"
print(hdr)
print("-" * len(hdr))

with tf.device("/CPU:0"):
    for shp in SHAPES:
        host = np.arange(np.prod(shp), dtype=np.float32).reshape(shp) % 1000 * 0.001
        x = tf.constant(host)
        row, tf02, pt02 = [], None, None

        for ax in AXES:
            ms = bench(lambda ax=ax: tf.reduce_sum(x, axis=ax).numpy(), args.reps)
            if ax == [0, 2]:
                tf02 = ms
            row.append(f"{ms:13.2f}")

        if HAVE_TORCH:
            xt = torch.from_numpy(host)
            for ax in AXES:
                ms = bench(lambda ax=ax: torch.sum(xt, dim=ax), args.reps)
                if ax == [0, 2]:
                    pt02 = ms
                row.append(f"{ms:13.2f}")

        ratio = f"{tf02 / pt02:15.2f}x" if pt02 else " " * 16
        print(f"{str(shp):>18} | " + " | ".join(row) + " |" + ratio)

        # correctness: all three frameworks/axis-sets must agree
        a = tf.reduce_sum(x, axis=[0, 2]).numpy()
        b = host.sum(axis=(0, 2))
        rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-9))
        assert rel < 1e-4, f"mismatch {rel}"

print("\nAll axis=[0,2] results matched numpy to <1e-4 relative.")
```

</details>

No PR from me yet, since reordering the reduced axes touches `GenericDimReducer`'s contract and there are two ways to go about it, either sorting the axes by stride when the layout gets built in `TensorReductionOp`, or inverting the recursion in `GenericDimReducer` itself, so i'd rather hear which you'd prefer than guess. Happy to test a patch either way, or to rerun this on whatever shapes you care about from real models instead of the synthetic ones i picked.
