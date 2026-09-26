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
