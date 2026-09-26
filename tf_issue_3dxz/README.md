# 3-D reduction over axes {0,2}, keeping the middle dim

Repro material for the TF issue, there are two benchmarks in here and they answer
different questions, so it's worth running both.

## eigen_3dxz_bench.cpp

This is the one that actually pins down the cause, and it doesn't need TensorFlow
at all, just the Eigen headers.

```
g++ -O3 -march=native -std=c++17 -I<path-to-eigen> eigen_3dxz_bench.cpp -o bench
taskset -c 0-7 ./bench 9
```

It runs the same sum three ways, Eigen's own `in.sum({0,2})`, then a hand-written
loop with the stride-1 axis innermost, then the same loop walking the way
`GenericDimReducer` walks, and arms A and C coming out at the same time is the
whole point, since it means the cost is the loop order rather than anything to do
with Eigen's machinery.

One thing to watch if you edit it, the axes go in as
`Eigen::IndexList<type2index<...>>` rather than `Eigen::array`, and that matters,
since Eigen picks its path with the compile-time traits `are_inner_most_dims` and
`preserve_inner_most_dims`, both of which come out false for a runtime
`Eigen::array`, so swapping in an array quietly lands every axis set on the
generic path, the controls stop meaning anything and the benchmark proves nothing,
and TF's CPU path uses the IndexList form anyway
(`Constants<CPUDevice> : ConstantsBase`, reduction_ops_common.h:61-67).

The program prints which path each axis set compiled to, so you can see it rather
than take my word for it.

## tf_3dxz_bench.py

Same thing at framework level, i.e. what it costs you in practice.

```
<venv>/bin/python tf_3dxz_bench.py --threads 1   # intra_op = 1
<venv>/bin/python tf_3dxz_bench.py --threads 0   # framework default
```

It times `axis=[0,2]` against `[1,2]` and `[0,1]` on the same tensor, so the data
and the element count stay fixed and only the axis set changes, it times PyTorch
on the same shapes for reference, and every result gets asserted against numpy
before anything is printed.

## What's in here

| file | what it is |
|---|---|
| `eigen_result.txt` | Eigen 3.5.0, single thread |
| `tf_result_t1.txt` | TF 2.21.0 stable, intra_op=1 |
| `tf_result_default.txt` | TF 2.21.0 stable, default threads |
| `tf_result_nightly_t1.txt` | tf-nightly 2.22.0-dev20260905, intra_op=1 |

Box was an i7-14700K (8P+12E, 28 threads), AVX2, no AVX-512, Linux 6.8.0-139,
Python 3.12.3, and everything is pinned with `taskset` to P-cores only, because
the P and E cores run at different clocks and unpinned runs drift enough to be
annoying.

The two smallest shapes wobble run to run for the same reason, so the 512 and 1024
rows are the ones worth trusting.

## Getting TF installed

`python3-venv` wasn't on the box, so the venvs came from
`python3 -m venv --without-pip` with pip bootstrapped via `get-pip.py`, and stable
and nightly went into separate venvs since `tensorflow` and `tf-nightly` conflict.
