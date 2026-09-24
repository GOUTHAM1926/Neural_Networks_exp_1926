# TF issue: 3-D reduction over axes {0,2} (keep the middle dim)

Repro material for the TensorFlow issue. Two independent benchmarks.

## 1. `eigen_3dxz_bench.cpp` — isolates the traversal order

Eigen only, no TensorFlow. Runs the same reduction three ways and prints which
Eigen path each axis set compiles to.

```
g++ -O3 -march=native -std=c++17 -I<path-to-eigen> eigen_3dxz_bench.cpp -o bench
taskset -c 0-7 ./bench 9
```

Uses `Eigen::IndexList<type2index<...>>` for the axes, not `Eigen::array`.
That is deliberate: Eigen selects its path with the *compile-time* traits
`are_inner_most_dims` / `preserve_inner_most_dims`, which are always false for a
runtime `Eigen::array`, so an array would force every case onto the generic path
and prove nothing. TF's CPU path uses the IndexList form
(`Constants<CPUDevice> : ConstantsBase`, reduction_ops_common.h:61-67).

Arms A (Eigen) and C (manual, big-stride innermost) landing on the same time is
the actual finding — it shows the cost is the loop order, not Eigen overhead.

## 2. `tf_3dxz_bench.py` — what it costs at framework level

```
<venv>/bin/python tf_3dxz_bench.py --threads 1   # intra_op = 1
<venv>/bin/python tf_3dxz_bench.py --threads 0   # framework default
```

Times `axis=[0,2]` against two controls on the *same* tensor (`[1,2]`, `[0,1]`)
and against PyTorch on the same shapes. Asserts every result matches numpy.

## Results captured on this box

| file | run |
|---|---|
| `eigen_result.txt`        | Eigen 3.5.0, single thread |
| `tf_result_t1.txt`        | TF 2.21.0 stable, intra_op=1 |
| `tf_result_default.txt`   | TF 2.21.0 stable, default threads |
| `tf_result_nightly_t1.txt`| tf-nightly 2.22.0-dev20260905, intra_op=1 |

Box: Intel Core i7-14700K (8P+12E, 28 threads), AVX2, no AVX-512,
Linux 6.8.0-139-generic, Python 3.12.3. Timing runs pinned with `taskset` to
P-cores only, because P/E cores run at different clocks and unpinned runs drift.

## Environment used

TF and PyTorch were installed into throwaway venvs outside this repo:
`~/Downloads/tf_issue_venv` (stable) and `~/Downloads/tf_nightly_venv` (nightly).
`python3-venv` is not installed system-wide here, so the venvs were made with
`python3 -m venv --without-pip` and pip bootstrapped via `get-pip.py`.
