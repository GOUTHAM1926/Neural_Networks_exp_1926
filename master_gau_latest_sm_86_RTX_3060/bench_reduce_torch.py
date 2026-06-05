import torch, time
def bench(tag, shape, axis):
    x = torch.ones(*shape, device='cuda', dtype=torch.float32)
    for _ in range(20): torch.sum(x, dim=axis)
    torch.cuda.synchronize()
    it=200; t0=time.perf_counter()
    for _ in range(it): torch.sum(x, dim=axis)
    torch.cuda.synchronize()
    per=(time.perf_counter()-t0)/it*1e3
    n=1
    for s in shape: n*=s
    gbps=n*4/(per*1e-3)/1e9
    print(f"{tag:26s} shape{tuple(shape)} ax{axis}  {per:.3f} ms/call  {gbps:.1f} GB/s")
print("=== torch.sum throughput ===")
bench("last-axis (row-sum)",   [16384,768], 1)
bench("leading-axis (biasgrad)",[16384,768], 0)
bench("middle-axis",           [32,512,768], 1)
bench("large last-axis",       [4096,4096], 1)
bench("large leading-axis",    [4096,4096], 0)
