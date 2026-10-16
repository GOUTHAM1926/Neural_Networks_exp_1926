# 2026-10-16: one rival attention library per process, forward + backward back to back with NO synchronisation between
# calls (as our C++ BENCH runs ours, and as kernels run in training), for a kernel-only, free-clock scoreboard under nsys.
# Why a new runner: bench_step6_libs.py (a) in --dense mode runs fa3, fa3_det, fa4, sdpa_cudnn and sdpa_flash interleaved
# in one loop whatever BENCH_ONLY says, so each kernel runs right after another library's kernels (another power state),
# and (b) synchronises after every call: its ~100 us of Python per call leaves the GPU ~10% idle, so the libraries ran
# cooler (higher clock) than they would back to back. The inputs, packed layouts and library calls are the bench's own
# (imported), so the contenders are configured exactly as in the scoreboards before.
# usage: python rivals_free_1016.py dense  <fa3|fa3_det|fa4|cudnn> [iters]
#        python rivals_free_1016.py packed <fa3|fa3_det|fa4|cudnn> <segments file> <layout name> [iters]
import importlib, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())   # bench_step6_libs.py sits in the working directory (step6) on the box
import bench_step6_libs as S

def dense_contender(impl):
    g = torch.Generator(device=S.dev).manual_seed(4321)
    rnd = lambda *s: torch.randn(*s, device=S.dev, dtype=torch.float32, generator=g).to(S.DT)
    q, k, v, do = rnd(S.B, S.T, S.HQ, S.D), rnd(S.B, S.T, S.HKV, S.D), rnd(S.B, S.T, S.HKV, S.D), rnd(S.B, S.T, S.HQ, S.D)
    if impl in ("fa3", "fa3_det"):
        fa3 = importlib.import_module("flash_attn_interface"); st = {}
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        def f(): st["o"], st["lse"] = fa3._flash_attn_forward(q, k, v, softmax_scale=S.SCALE, causal=True)[:2]
        def b(): fa3._flash_attn_backward(do, q, k, v, st["o"], st["lse"], dq=dq, dk=dk, dv=dv, softmax_scale=S.SCALE,
                                          is_causal=True, deterministic=(impl == "fa3_det"))
        return f, b
    if impl == "fa4":
        fa4 = importlib.import_module("flash_attn.cute.interface"); s4 = {}
        def f():
            r = fa4._flash_attn_fwd(q, k, v, softmax_scale=S.SCALE, causal=True, return_lse=True); s4["o"], s4["lse"] = r[0], r[1]
        def b(): fa4._flash_attn_bwd(q, k, v, s4["o"], do, s4["lse"], softmax_scale=S.SCALE, causal=True)
        return f, b
    if impl == "cudnn":   # PyTorch SDPA, cuDNN backend, as bench_step6_libs.main_dense
        from torch.nn.attention import SDPBackend, sdpa_kernel
        qb, kb, vb, dob = [x.transpose(1, 2).contiguous() for x in (q, k, v, do)]
        leaf = [x.detach().clone().requires_grad_(True) for x in (qb, kb, vb)]; st2 = {}
        def f():
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                st2["o"] = F.scaled_dot_product_attention(*leaf, is_causal=True, scale=S.SCALE, enable_gqa=True)
        def b(): torch.autograd.grad(st2["o"], leaf, dob, retain_graph=True)
        return f, b
    raise SystemExit(f"unknown dense impl {impl}")

def packed_contender(impl, segfile, lname):
    lay = S.load_layouts(segfile, lname)
    names = list(lay)   # load_layouts keeps the layouts whose name CONTAINS lname (real_rows_752 -> real_rows_752-759)
    assert len(names) == 1, f"layout filter {lname} matches {names} in {segfile}"
    segs, cu, mx, fl = S.layout_tensors(lay[names[0]])
    print(f"layout {names[0]}: {len(segs)} segments", flush=True)
    q, k, v, do = S.make_inputs()
    if impl in ("fa3", "fa3_det"):
        fa3 = importlib.import_module("flash_attn_interface"); st = {}
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        def f(): st["o"], st["lse"] = fa3._flash_attn_forward(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx,
                                                              max_seqlen_k=mx, softmax_scale=S.SCALE, causal=True)[:2]
        def b(): fa3._flash_attn_backward(do, q, k, v, st["o"], st["lse"], cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx,
                                          max_seqlen_k=mx, dq=dq, dk=dk, dv=dv, softmax_scale=S.SCALE, is_causal=True,
                                          deterministic=(impl == "fa3_det"))
        return f, b
    if impl == "fa4":
        fa4 = importlib.import_module("flash_attn.cute.interface"); s4 = {}
        def f():
            r = fa4._flash_attn_fwd(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx,
                                    softmax_scale=S.SCALE, causal=True, return_lse=True); s4["o"], s4["lse"] = r[0], r[1]
        def b(): fa4._flash_attn_bwd(q, k, v, s4["o"], do, s4["lse"], softmax_scale=S.SCALE, causal=True, cu_seqlens_q=cu,
                                     cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx)
        return f, b
    if impl == "cudnn":   # cuDNN varlen with the 6 KV heads passed directly (CUDNN_GQA=1 path of the bench)
        c = S.cudnn_gqa_contender(q, k, v, do, cu, mx, torch.zeros((2,), dtype=torch.uint64, device=S.dev))
        return c.fwd, c.bwd
    raise SystemExit(f"unknown packed impl {impl}")

if __name__ == "__main__":
    mode, impl = sys.argv[1], sys.argv[2]
    if mode == "dense":
        f, b = dense_contender(impl); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    else:
        f, b = packed_contender(impl, sys.argv[3], sys.argv[4]); iters = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), "| gpu", torch.cuda.get_device_name(0), flush=True)
    for _ in range(20):   # warm-up (FA4 compiles here)
        f(); b()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):   # back to back, no synchronisation between calls
        f(); b()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"RIVAL mode={mode} impl={impl} iters={iters} wall_s={dt:.2f} per_iter_ms={1000 * dt / iters:.4f}", flush=True)
