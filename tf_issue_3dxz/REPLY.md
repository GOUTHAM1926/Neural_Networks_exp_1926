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
|---|---:|---:|---:|---:|
| (512, 8, 512)   |  4.75 | 0.15 | 0.58 | 0.17 |
| (512, 16, 512)  | 11.40 | 0.29 | 1.05 | 0.34 |
| (256, 64, 256)  |  9.87 | 0.29 | 0.89 | 0.37 |
| (1024, 8, 1024) | 36.96 | 0.74 | 4.37 | 0.80 |
| (128, 256, 128) |  8.73 | 0.29 | 0.97 | 0.42 |
| (64, 512, 64)   |  2.88 | 0.15 | 0.46 | 0.29 |

The PyTorch column is 10x to 46x, but i don't think that's the useful comparison, the useful one is inside TF itself, and in the last row reducing {1,2} takes 0.74 ms and reducing {0,2} takes 36.96 ms, same tensor, same binary, same amount of arithmetic, 50x apart, so it isn't my CPU or the wheel's build flags or the tensor size.

tf-nightly 2.22.0-dev20260905, `intra_op=1`, same thing, so it isn't already fixed on master:

| shape | `axis=[0,2]` | `axis=[1,2]` | torch `[0,2]` |
|---|---:|---:|---:|
| (512, 8, 512)   |  4.45 | 0.12 | 0.13 |
| (512, 16, 512)  |  8.63 | 0.24 | 0.22 |
| (1024, 8, 1024) | 23.79 | 0.72 | 0.79 |

Default thread count, TF 2.21.0, threads don't help because the other axis sets get faster and {0,2} doesn't:

| shape | `axis=[0,2]` | `axis=[1,2]` | torch `[0,2]` |
|---|---:|---:|---:|
| (512, 8, 512)   |  3.61 | 0.07 | 0.06 |
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
