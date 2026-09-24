Hi @Venkat6871, thanks for confirming the path is still live. Here is everything you asked for.

Quick note on naming first, because it tripped me up in my own title: TF's own code calls this case `Launch3DXZReduction` (reduce `{0,2}`, keep the middle dim). `Launch3DYReduction` is the *other* case (reduce dim 1). Everything below is `reduction_axes = {0,2}`.

### Environment

```
CPU:      Intel Core i7-14700K, 20 cores / 28 threads (8P + 12E), AVX2, no AVX-512
OS:       Linux 6.8.0-139-generic
Python:   3.12.3
TF:       2.21.0 (stable, tensorflow-cpu wheel)
          2.22.0-dev20260905 (tf-nightly-cpu)
PyTorch:  2.14.0+cpu   (reference only)
Threads:  intra_op = 1 and framework default, both reported
Pinning:  taskset -c 0-7 (P-cores only; P and E cores clock differently and
          unpinned runs drift enough to matter)
dtype:    float32, median of 11 runs, all results asserted against numpy
```

### 1. Framework level — `tf.reduce_sum(x, axis=[0,2])`

Script: `tf_3dxz_bench.py` (attached below). The two extra axis sets are controls — **same tensor, same number of elements reduced**, only the axis set differs.

**TF 2.21.0 stable, `intra_op=1`**, times in ms:

| shape | `axis=[0,2]` | `[1,2]` ctrl | `[0,1]` ctrl | torch `[0,2]` | TF/torch |
|---|---:|---:|---:|---:|---:|
| (512, 8, 512)    |  4.75 | 0.15 | 0.58 | 0.17 | 27.7x |
| (512, 16, 512)   | 11.40 | 0.29 | 1.05 | 0.34 | 33.6x |
| (256, 64, 256)   |  9.87 | 0.29 | 0.89 | 0.37 | 26.3x |
| (1024, 8, 1024)  | 36.96 | 0.74 | 4.37 | 0.80 | 46.0x |
| (128, 256, 128)  |  8.73 | 0.29 | 0.97 | 0.42 | 21.0x |
| (64, 512, 64)    |  2.88 | 0.15 | 0.46 | 0.29 |  9.9x |

The number I would look at is not the PyTorch column, it is the `[1,2]` control: **same binary, same tensor, same element count, 25-50x apart** purely from which axes are reduced. That rules out wheel build flags, my CPU, and tensor size all at once.

**tf-nightly 2.22.0-dev20260905, `intra_op=1`** — same behaviour, so it is not already fixed on master:

| shape | `axis=[0,2]` | `[1,2]` ctrl | torch `[0,2]` |
|---|---:|---:|---:|
| (512, 8, 512)   |  4.45 | 0.12 | 0.13 |
| (512, 16, 512)  |  8.63 | 0.24 | 0.22 |
| (1024, 8, 1024) | 23.79 | 0.72 | 0.79 |

**Default threads** (stable) — threading does not rescue it, the gap widens because the controls scale and `[0,2]` does not:

| shape | `axis=[0,2]` | `[1,2]` ctrl | torch `[0,2]` |
|---|---:|---:|---:|
| (512, 8, 512)   |  3.61 | 0.07 | 0.06 |
| (1024, 8, 1024) | 21.51 | 0.22 | 0.25 |

Smaller shapes are noisier run to run (hybrid P/E cores); the >=512³ rows are stable and are what I would trust.

### 2. Eigen on its own — this is the part that actually localises it

The above shows a cost but not a cause, so I ran Eigen directly with no TF in the picture (`eigen_3dxz_bench.cpp`, Eigen 3.5.0, `-O3 -march=native`, single threaded). Three arms computing identical numbers:

- **A** — `in.sum(axes{0,2})`, i.e. what TF calls
- **B** — hand-written loop, innermost loop over Z (stride 1)
- **C** — hand-written loop, innermost loop over X (stride Y·Z), imitating `GenericDimReducer`

```
Eigen's compile-time path choice for a rank-3 RowMajor tensor:
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

**A and C land on the same time in every row** (2.09/2.09, 4.33/4.37, 20.51/21.70). A plain C loop written to traverse in Eigen's order is exactly as slow as Eigen. So this is the loop order, not template or dispatch overhead — and reordering the same loop (B) is 2.3-11x faster on identical data. `maxrel` is ~1e-7 throughout, i.e. fp32 summation-order noise only.

One detail worth flagging if you reproduce this: the axes have to be passed as `Eigen::IndexList<type2index<...>>`, not `Eigen::array`. `are_inner_most_dims` / `preserve_inner_most_dims` are compile-time traits and are false for a runtime `Eigen::array`, so using an array pushes *every* axis set onto the generic path and the controls become meaningless. I got that wrong on my first attempt. TF's CPU path uses the IndexList form (`Constants<CPUDevice> : ConstantsBase`, reduction_ops_common.h:61-67), which is what I matched.

### 3. Why

For row-major `[X,Y,Z]` reducing `{0,2}`, `TensorReduction.h:639-650` fills `m_reducedStrides` in input-dimension order:

```
m_reducedStrides[0] = Y*Z   (X)
m_reducedStrides[1] = 1     (Z)
```

`GenericDimReducer` (`TensorReduction.h:131-150`) recurses from `NumReducedDims-1` down to `0`, so **slot 0 is the innermost loop** — X, striding Y·Z per element. The stride-1 axis Z ends up as the outer loop. Every element access takes a fresh cache line and uses one value from it.

Secondary effect on the same path: `ReducingInnerMostDims` and `PreservingInnerMostDims` are both false, so `packet()` falls to the final `else` at `TensorReduction.h:849-852`, which just calls `coeff()` `PacketSize` times. And `BlockAccess = false` on the reduction evaluator. So the case gets no vectorisation and no tiled path either.

For contrast, PyTorch's `TensorIterator::reorder_dimensions` (`TensorIterator.cpp:232-236`) sorts dimensions by stride ascending, so the stride-1 axis is always innermost regardless of which axes are reduced — which is why its three columns above are flat while TF's are not.

### 4. GPU is already fine

Worth saying explicitly so the scope is clear: this is CPU only. `ReduceImpl` in `reduction_gpu_kernels.cu.h` has a dedicated branch for `in_rank==3 && out_rank==1 && axes=={0,2}` that calls `Launch3DXZReduction`, which uses CUB `DeviceSegmentedReduce` with a gather iterator whose index advances with Z first — coalesced, correct order. No Eigen involved. So only the `ReduceEigenImpl` path in `reduction_ops.h:53-58` is affected.

### Repro

Both scripts are attached / in the gist. Nothing in them needs TF to build from source — the Eigen one needs only the Eigen headers.

Happy to test a patch or a different shape set if that is useful. I did not send a PR because reordering the reduced axes by stride touches `GenericDimReducer`'s contract and I would rather you say which way you want it fixed — sorting the axes by stride at layout time in `TensorReductionOp`, or inverting the recursion in `GenericDimReducer`. Also happy to run whatever shapes you care about from real models rather than the synthetic ones above.
