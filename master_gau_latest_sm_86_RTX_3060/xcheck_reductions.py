#!/usr/bin/env python3
# Cross-check our reduce_sum (dumped from xcheck_reductions) against torch.sum.
import struct, sys, numpy as np, torch

f = open("/tmp/xcheck_dump.bin", "rb")
def rd(fmt):
    sz = struct.calcsize(fmt); b = f.read(sz)
    if len(b) < sz: return None
    return struct.unpack(fmt, b)

total = bad = 0
while True:
    h = rd("<i")
    if h is None: break
    ndim = h[0]
    dims = [rd("<i")[0] for _ in range(ndim)]
    axis = rd("<i")[0]
    n_in = rd("<q")[0]
    data = np.frombuffer(f.read(4*n_in), dtype=np.float32).reshape(dims)
    n_out = rd("<q")[0]
    gpu = np.frombuffer(f.read(4*n_out), dtype=np.float32)
    cpu = np.frombuffer(f.read(4*n_out), dtype=np.float32)

    t = torch.from_numpy(data.copy())
    ref = t.sum(dim=axis).reshape(-1).numpy()
    ref_gpu = t.cuda().sum(dim=axis).reshape(-1).cpu().numpy()

    total += 1
    g_ok = np.allclose(gpu, ref, atol=1e-2, rtol=1e-3) and np.allclose(gpu, ref_gpu, atol=1e-2, rtol=1e-3)
    c_ok = np.allclose(cpu, ref, atol=1e-2, rtol=1e-3)
    if not (g_ok and c_ok):
        bad += 1
        i = int(np.argmax(np.abs(gpu-ref)))
        print(f"FAIL dims={dims} axis={axis}: ours_gpu[{i}]={gpu[i]:.4f} torch={ref[i]:.4f} ours_cpu={cpu[i]:.4f}")
    else:
        print(f"ok   dims={dims} axis={axis}  (max|ours_gpu-torch|={np.max(np.abs(gpu-ref)):.2e})")

print(f"\n==== torch cross-check: {total-bad}/{total} match torch.sum (CPU+GPU) ====")
sys.exit(1 if bad else 0)
