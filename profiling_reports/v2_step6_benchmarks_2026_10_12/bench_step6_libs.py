#!/usr/bin/env python3
# Step 6 (2026-10-12): cross-library attention timing on the SAME segment layouts as our harness's
# BENCH mode (test_varlen_attention.cu, VARLEN_ONLY=BENCH VARLEN_BENCH_EXPORT=<file>).
#
# Shapes: B = 8 rows x T = 4096, 18 query heads / 6 KV heads, head_dim 128, bf16, causal.
# Every segment (chat or pad tail) is its own attention block, as in our kernels.
# Contenders (each in its native layout; layout conversion is NOT timed):
#   fa3_varlen        FlashAttention 3 (hopper/), direct _flash_attn_forward/_flash_attn_backward, cu_seqlens
#   fa3_varlen_det    the same with deterministic=True in the backward
#   fa4_varlen        FlashAttention 4 (flash_attn/cute), direct _flash_attn_fwd/_flash_attn_bwd, cu_seqlens
#   cudnn_varlen_exp     cuDNN packed attention (added 2026-10-12, see "cuDNN packed attention" below): direct
#                        aten._cudnn_attention_forward/_backward with cu_seqlens; K/V copied to 18 heads BEFORE
#                        the timed region, so the time is the cuDNN kernels alone
#   cudnn_varlen_expsum  the same with the K/V copy (forward) and the per-group dK/dV sum (backward) inside the
#                        timed region: what a GQA model pays on this route
#   cudnn_varlen_gqa     the 6 KV heads passed straight into the op; only with CUDNN_GQA=1 (run_cudnn_varlen.sh
#                        sets it after --cudnn-probe passed in its own process)
#   torch_varlen      torch.nn.attention.varlen.varlen_attn (autograd backward)
#   flex_docmask      FlexAttention with a same-segment + causal block mask (design B, compiled)
#   sdpa_<b>_fullrow  SDPA over the whole row, plain causal (what the dense path computes), backends
#                     flash / cudnn / efficient
#   sdpa_<b>_mask     SDPA with an explicit block-diagonal causal mask (HF's SDPA packing path)
# Timing: 10 warm-up rounds, then BENCH_ITERS rounds; every round times each contender's forward,
# then each contender's backward (interleaved, so clock drift hits all alike); cudaEvent per call.
# Useful FLOPs (same as our BENCH): fwd = sum over segments 4*Hq*D*L^2/2, bwd = 2.5 x fwd.
# Output: "BENCH layout=.. impl=.. dir=.. med_ms= p10_ms= p90_ms= tflops=" lines, then an agreement
# check: the packed contenders must give the same O and dQ (to bf16 noise) on the same inputs.
# usage: CUDA_VISIBLE_DEVICES=7 python bench_step6_libs.py <segments file> [layout-name-filter]
#        BENCH_ONLY=fa3,fa4,cudnn   keep only contenders whose name starts with one of these prefixes
#        EXPECT_GPU_UUID=GPU-...    abort unless the visible GPU has this UUID (run_cudnn_varlen.sh sets it)
#        BENCH_NVTX=1               NVTX range per timed call (for nsys nvtx_kern_sum)
#        python bench_step6_libs.py --cudnn-probe <segments file>   cuDNN correctness only, no timing
#        CUDA_VISIBLE_DEVICES= python bench_step6_libs.py --selftest <segments file>   CPU-only checks
import importlib, inspect, math, os, sys, traceback

import torch
import torch.nn.functional as F

B, T, HQ, HKV, D = 8, 4096, 18, 6, 128
G = HQ // HKV
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
WARM = int(os.environ.get("BENCH_WARM", "10"))   # 0 for ncu captures
DT = torch.bfloat16
SCALE = 1.0 / math.sqrt(D)
dev = "cuda"


def load_layouts(path, filt):
    lay = {}
    for line in open(path):
        name, b, t0, L = line.split()
        if filt and filt not in name:
            continue
        lay.setdefault(name, []).append((int(b), int(t0), int(L)))
    return lay


def pct(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]


def want(name):
    # BENCH_ONLY used to take one prefix ("fa3" for the ncu capture); a comma list of prefixes now, same result
    # for a single prefix.
    only = [p for p in os.environ.get("BENCH_ONLY", "").split(",") if p]
    return not only or any(name.startswith(p) for p in only)


def check_gpu():
    # Boxes get reassigned: a run must land on the GPU the runner pinned, nothing else.
    # Compared as 32 hex digits; aborts only on a definite mismatch (CUDA_VISIBLE_DEVICES is the pin itself).
    p = torch.cuda.get_device_properties(0)
    got, exp = str(getattr(p, "uuid", "")), os.environ.get("EXPECT_GPU_UUID", "")
    hx = lambda s: "".join(ch for ch in s.lower().removeprefix("gpu-") if ch in "0123456789abcdef")
    print(f"gpu {p.name} uuid {got} | expected {exp or '(not given)'}", flush=True)
    if exp and len(hx(got)) == 32 and len(hx(exp)) == 32 and hx(got) != hx(exp):
        sys.exit("ABORT: the visible GPU is not the expected one")
    if exp and len(hx(got)) != 32:
        print("  (uuid format not recognised: not compared)", flush=True)


def layout_tensors(segs):
    segs = sorted(segs)
    starts = [b * T + t0 for b, t0, _ in segs]
    assert starts[0] == 0 and all(starts[i] + segs[i][2] == (starts[i + 1] if i + 1 < len(segs) else B * T)
                                  for i in range(len(segs))), "segments must tile every row"
    cu = torch.tensor(starts + [B * T], dtype=torch.int32, device=dev)
    mx = max(L for _, _, L in segs)
    fl = sum(4.0 * HQ * D * L * L / 2 for _, _, L in segs)
    return segs, cu, mx, fl


def make_inputs():
    g = torch.Generator(device=dev).manual_seed(1234)
    rnd = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g).to(DT)
    return rnd(B * T, HQ, D), rnd(B * T, HKV, D), rnd(B * T, HKV, D), rnd(B * T, HQ, D)


class Contender:
    def __init__(self, name, fwd, bwd, out_fn=None, dq_fn=None, result=None):
        self.name, self.fwd, self.bwd, self.out_fn, self.dq_fn = name, fwd, bwd, out_fn, dq_fn
        self.result = result   # cuDNN only: one fwd + bwd, returns (O, dQ, dK, dV) with dK/dV over the 6 KV heads
        self.tf, self.tb = [], []


# ---- cuDNN packed attention (added 2026-10-12)
# Before: Step 6 had no packed cuDNN number, because I had said it needs Transformer Engine. Wrong: torch 2.14
# already calls it. How PyTorch itself calls it (torch/nn/attention/varlen.py, torch 2.14.1 in ~/envv):
#   forward  aten._cudnn_attention_forward(query, key, value, attn_bias=None, cum_seq_q, cum_seq_k, max_q, max_k,
#            compute_log_sumexp=True, dropout_p=0.0, is_causal, return_debug_mask=False, scale)   varlen.py:118-134
#   backward aten._cudnn_attention_backward(grad_out, ..., philox_seed=rng, philox_offset=rng, attn_bias=None,
#            ...), rng = zeros(2, uint64) since dropout is 0                              varlen.py:158-160, 559-576
# cudnn_fwd / cudnn_bwd below copy those two calls argument for argument.
# Why direct calls and not varlen_attn: varlen_attn turns causal into window (-1, 0) (varlen.py:336) and sends any
# window, and any GQA, to FA2 (varlen.py:62-65). So through the public API causal packed attention never reaches
# cuDNN; our "torch_varlen" contender was FA2 (its 10-12 kernels: pytorch_flash::flash_fwd_kernel / flash_bwd_*,
# nsys_6b/libs_packed_cuda_gpu_kern_sum.csv).
# Layout: (tokens, heads, 128) + int32 offsets, the same tensors FA3 gets; PyTorch's nested route hands cuDNN this
# same layout (torch/nested/_internal/sdpa.py:578-640).
# GQA: no PyTorch route sends 18/6 heads to the packed op (nested route: sdpa.py:592-598 refuses different head
# counts; varlen_attn: FA2). So the main contender copies K/V to 18 heads (query head h reads KV head h // 3, as in
# FA3 and our kernels) and sums dK/dV per group of 3. Passing the 6 KV heads directly is untested by PyTorch: it
# is probed first in its own process (a crash there cannot take the benchmark down) and timed only if it passed.
# Gate, every layout, before timing: O, dQ, dK, dV against FA3 on the same inputs, relative max error
# max|x - fa3| / max|fa3| <= TOL_REL. On 10-12 every packed contender agreed with FA3 to <= 1.6e-2 absolute on O
# and <= 3.1e-2 on dQ (full_libs_1247.log AGREE lines); a wrong head mapping or a missing mask gives errors of
# order 1. Two controls run with it: FA4 vs FA3 (normal kernel-to-kernel rounding, printed for comparison) and a
# deliberately wrong head mapping (h -> h % 6) that MUST fail, which shows the gate catches the mistake it is for.
TOL_REL = 2e-2
CUDNN_NAMES = ("cudnn_varlen_exp", "cudnn_varlen_expsum", "cudnn_varlen_gqa")


def cudnn_wanted():
    # Build the cuDNN contenders only if BENCH_ONLY keeps one of them. Asked per real name: the first version asked
    # want("cudnn"), which is False for a longer prefix such as "cudnn_varlen_gqa", so a run with that prefix
    # silently timed no cuDNN at all (cudnn_varlen_1012/nsys_libs_gqa.out, first nsys try).
    return any(want(n) for n in CUDNN_NAMES)


def expand_kv(x):   # [N, 6, D] -> [N, 18, D]: query head h reads KV head h // G
    return x.repeat_interleave(G, dim=1)


def sum_kv(dx):     # [N, 18, D] -> [N, 6, D]: the G query heads of each group summed
    return dx.view(dx.shape[0], HKV, G, D).sum(2)


def cudnn_fwd(q, k, v, cu, mx):
    r = torch.ops.aten._cudnn_attention_forward(query=q, key=k, value=v, attn_bias=None, cum_seq_q=cu, cum_seq_k=cu,
                                                max_q=mx, max_k=mx, compute_log_sumexp=True, dropout_p=0.0,
                                                is_causal=True, return_debug_mask=False, scale=SCALE)
    return r[0], r[1]


def cudnn_bwd(do, q, k, v, o, lse, cu, mx, rng):
    return torch.ops.aten._cudnn_attention_backward(grad_out=do, query=q, key=k, value=v, out=o, logsumexp=lse,
                                                    cum_seq_q=cu, cum_seq_k=cu, max_q=mx, max_k=mx, dropout_p=0.0,
                                                    philox_seed=rng, philox_offset=rng, attn_bias=None,
                                                    is_causal=True, scale=SCALE)


def cudnn_contenders(q, k, v, do, cu, mx, gqa_direct):
    rng = torch.zeros((2,), dtype=torch.uint64, device=dev)
    out = []
    k18, v18 = expand_kv(k), expand_kv(v)
    s1 = {}

    def f1():
        s1["o"], s1["lse"] = cudnn_fwd(q, k18, v18, cu, mx)

    def b1():
        s1["g"] = cudnn_bwd(do, q, k18, v18, s1["o"], s1["lse"], cu, mx, rng)

    def r1():
        f1(); b1()
        return s1["o"], s1["g"][0], sum_kv(s1["g"][1]), sum_kv(s1["g"][2])
    out.append(Contender("cudnn_varlen_exp", f1, b1, lambda: s1["o"], lambda: s1["g"][0], r1))
    s2 = {}

    def f2():
        s2["k"], s2["v"] = expand_kv(k), expand_kv(v)
        s2["o"], s2["lse"] = cudnn_fwd(q, s2["k"], s2["v"], cu, mx)

    def b2():
        g = cudnn_bwd(do, q, s2["k"], s2["v"], s2["o"], s2["lse"], cu, mx, rng)
        s2["g"] = (g[0], sum_kv(g[1]), sum_kv(g[2]))

    def r2():
        f2(); b2()
        return (s2["o"],) + s2["g"]
    out.append(Contender("cudnn_varlen_expsum", f2, b2, lambda: s2["o"], lambda: s2["g"][0], r2))
    if gqa_direct:
        out.append(cudnn_gqa_contender(q, k, v, do, cu, mx, rng))
    return out


def cudnn_gqa_contender(q, k, v, do, cu, mx, rng):
    s3 = {}

    def f3():
        s3["o"], s3["lse"] = cudnn_fwd(q, k, v, cu, mx)

    def b3():
        s3["g"] = cudnn_bwd(do, q, k, v, s3["o"], s3["lse"], cu, mx, rng)

    def r3():
        f3(); b3()
        return (s3["o"],) + tuple(s3["g"])
    return Contender("cudnn_varlen_gqa", f3, b3, lambda: s3["o"], lambda: s3["g"][0], r3)


def cudnn_wrong_mapping(q, k, v, do, cu, mx):
    # Negative control: query head h reads KV head h % 6 (.repeat instead of .repeat_interleave), dK/dV summed with
    # the matching wrong grouping. 14 of the 18 heads read the wrong KV head (h = 0, 8, 9, 17 land on the same one
    # either way), so the gate MUST fail on this.
    rng = torch.zeros((2,), dtype=torch.uint64, device=dev)
    kw, vw = k.repeat(1, G, 1), v.repeat(1, G, 1)
    o, lse = cudnn_fwd(q, kw, vw, cu, mx)
    g = cudnn_bwd(do, q, kw, vw, o, lse, cu, mx, rng)
    sw = lambda d: d.view(d.shape[0], G, HKV, D).sum(1)
    return o, g[0], sw(g[1]), sw(g[2])


def fa3_reference(q, k, v, do, cu, mx):
    # The same calls as the fa3_varlen contender, with its own output buffers
    fa3 = importlib.import_module("flash_attn_interface")
    o, lse = fa3._flash_attn_forward(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx,
                                     softmax_scale=SCALE, causal=True)[:2]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    fa3._flash_attn_backward(do, q, k, v, o, lse, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx,
                             dq=dq, dk=dk, dv=dv, softmax_scale=SCALE, is_causal=True, deterministic=False)
    return o, dq, dk, dv


def fa4_outputs(q, k, v, do, cu, mx):
    fa4 = importlib.import_module("flash_attn.cute.interface")
    r = fa4._flash_attn_fwd(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx,
                            softmax_scale=SCALE, causal=True, return_lse=True)
    g = fa4._flash_attn_bwd(q, k, v, r[0], do, r[1], softmax_scale=SCALE, causal=True, cu_seqlens_q=cu,
                            cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx)
    return r[0], g[0], g[1], g[2]


def rel_err(x, ref):
    # max|x - ref| / max|ref|; a non-finite x, or a NaN anywhere, can never pass (NaN <= tol is False)
    x, ref = x.float(), ref.float()
    if not bool(torch.isfinite(x).all()):
        return float("inf")
    return ((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def gate_line(tag, lname, got, ref):
    errs = [rel_err(a, b) for a, b in zip(got, ref)]
    ok = len(errs) == 4 and all(e <= TOL_REL for e in errs)
    print(f"GATE layout={lname} {tag}: rel O {errs[0]:.2e} dQ {errs[1]:.2e} dK {errs[2]:.2e} dV {errs[3]:.2e} "
          f"(tol {TOL_REL:.0e}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def peak_extra_mib(fn):
    # Memory a single fwd + bwd needs above what is already allocated (outputs and workspace included)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    r = fn()
    torch.cuda.synchronize()
    return r, (torch.cuda.max_memory_allocated() - base) / 2**20


def cudnn_gate(lname, cud, q, k, v, do, cu, mx, with_fa4):
    """Every cuDNN contender is checked against FA3 on these inputs; returns the ones that passed."""
    try:
        ref, mib = peak_extra_mib(lambda: fa3_reference(q, k, v, do, cu, mx))
        print(f"MEM layout={lname} fa3_varlen fwd+bwd peak extra {mib:.0f} MiB", flush=True)
    except Exception as e:
        print(f"GATE layout={lname}: FA3 reference unavailable ({type(e).__name__}: {str(e)[:160]}); "
              f"cuDNN contenders UNVERIFIED, dropped", flush=True)
        return []
    if with_fa4:
        try:
            gate_line("CONTROL fa4_varlen vs fa3 (normal rounding)", lname, fa4_outputs(q, k, v, do, cu, mx), ref)
        except Exception as e:
            print(f"GATE layout={lname}: FA4 control unavailable ({type(e).__name__}: {str(e)[:160]})", flush=True)
    try:
        neg_ok = gate_line("NEGATIVE wrong head mapping h%6 (must FAIL)", lname, cudnn_wrong_mapping(q, k, v, do, cu, mx), ref)
    except Exception as e:
        print(f"GATE layout={lname}: negative control raised {type(e).__name__}: {str(e)[:200]}; "
              f"cuDNN contenders dropped", flush=True)
        return []
    if neg_ok:
        print(f"GATE layout={lname}: the wrong head mapping PASSED, so the gate cannot catch it; cuDNN contenders "
              f"UNVERIFIED, dropped", flush=True)
        return []
    passed = []
    for c in cud:
        try:
            got, mib = peak_extra_mib(c.result)
            ok = gate_line(c.name, lname, got, ref)
            print(f"MEM layout={lname} {c.name} fwd+bwd peak extra {mib:.0f} MiB", flush=True)
        except Exception as e:
            print(f"GATE layout={lname} {c.name}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            ok = False
        if ok:
            passed.append(c)
        else:
            print(f"GATE layout={lname} {c.name}: INVALID, not timed", flush=True)
    return passed


def kernel_names(fn):
    # Which kernels one call really launched (proof that a cuDNN contender runs cuDNN kernels)
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA})


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
          "| gpu", torch.cuda.get_device_name(0))
    check_gpu()
    gqa_direct = os.environ.get("CUDNN_GQA", "0") == "1"
    layouts = load_layouts(path, filt)
    q, k, v, do = make_inputs()
    to_bhtd = lambda x, H: x.view(B, T, H, D).transpose(1, 2).contiguous()
    qb, kb, vb, dob = to_bhtd(q, HQ), to_bhtd(k, HKV), to_bhtd(v, HKV), to_bhtd(do, HQ)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # BENCH_NVTX=1 (nsys block of run_cudnn_varlen.sh, added 2026-10-12): every timed call gets an NVTX range
    # "<contender>:<fwd|bwd>", so nsys's nvtx_kern_sum gives kernel time per contender (warm-up and gate calls are
    # outside the ranges). Two cuDNN contenders can launch kernels with the same name, so a plain per-kernel-name
    # summary would average them together. First version also captured only the timed rounds
    # (cudaProfilerStart/Stop + nsys --capture-range=cudaProfilerApi): nsys saw the range start and end but the
    # report held no kernel and no NVTX data (cudnn_varlen_1012/nsys_libs_packed.stats.out), so that part is removed.
    nvtx = os.environ.get("BENCH_NVTX", "0") == "1"

    def timed(f, tag=""):
        if nvtx:
            torch.cuda.nvtx.range_push(tag)
        ev0.record()
        f()
        ev1.record()
        if nvtx:
            torch.cuda.nvtx.range_pop()
        ev1.synchronize()
        return ev0.elapsed_time(ev1)

    for lname, segs in layouts.items():
        segs, cu, mx, fl = layout_tensors(segs)
        seg_id = torch.empty(B, T, dtype=torch.int32, device=dev)
        for i, (b, t0, L) in enumerate(segs):
            seg_id[b, t0:t0 + L] = i
        cons = []

        # ---- FlashAttention 3 (direct kernel calls)
        try:
            fa3 = importlib.import_module("flash_attn_interface")
            st = {}
            dq3, dk3, dv3 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

            def f3():
                st["o"], st["lse"] = fa3._flash_attn_forward(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx,
                                                             max_seqlen_k=mx, softmax_scale=SCALE, causal=True)[:2]
            for det in (False, True):
                def b3(det=det):
                    fa3._flash_attn_backward(do, q, k, v, st["o"], st["lse"], cu_seqlens_q=cu, cu_seqlens_k=cu,
                                             max_seqlen_q=mx, max_seqlen_k=mx, dq=dq3, dk=dk3, dv=dv3,
                                             softmax_scale=SCALE, is_causal=True, deterministic=det)
                cons.append(Contender("fa3_varlen" + ("_det" if det else ""), f3, b3,
                                      (lambda: st["o"]) if not det else None, (lambda: dq3) if not det else None))
        except Exception as e:
            print(f"  [{lname}] fa3 unavailable: {type(e).__name__}: {str(e)[:160]}")

        # ---- FlashAttention 4 (CuTe DSL, direct kernel calls)
        try:
            fa4 = importlib.import_module("flash_attn.cute.interface")
            s4 = {}

            def f4():
                r = fa4._flash_attn_fwd(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx,
                                        softmax_scale=SCALE, causal=True, return_lse=True)
                s4["o"], s4["lse"] = r[0], r[1]

            def b4():
                s4["g"] = fa4._flash_attn_bwd(q, k, v, s4["o"], do, s4["lse"], softmax_scale=SCALE, causal=True,
                                              cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=mx, max_seqlen_k=mx)
            cons.append(Contender("fa4_varlen", f4, b4, lambda: s4["o"], lambda: s4["g"][0]))
        except Exception as e:
            print(f"  [{lname}] fa4 unavailable: {type(e).__name__}: {str(e)[:160]}")

        # ---- cuDNN packed attention (direct op calls; gated against FA3 before timing, see the notes above)
        cud = [c for c in cudnn_contenders(q, k, v, do, cu, mx, gqa_direct) if want(c.name)] if cudnn_wanted() else []
        if cud:
            cons += cudnn_gate(lname, cud, q, k, v, do, cu, mx, with_fa4=want("fa4_varlen"))

        # ---- autograd-based contenders: one forward per timed forward; the backward is timed alone
        def autograd_contender(name, fwd_call, ins, dout, to_varlen=None):
            leaf = [x.detach().clone().requires_grad_(True) for x in ins]
            st = {}

            def f():
                st["o"] = fwd_call(*leaf)

            def bw():
                st["g"] = torch.autograd.grad(st["o"], leaf, dout, retain_graph=True)
            out_fn = (lambda: to_varlen(st["o"])) if to_varlen else None
            dq_fn = (lambda: to_varlen(st["g"][0])) if to_varlen else None
            return Contender(name, f, bw, out_fn, dq_fn)

        from_bhtd = lambda x: x.transpose(1, 2).reshape(B * T, x.shape[1], D)
        if want("torch_varlen"):
            try:
                from torch.nn.attention.varlen import varlen_attn
                print(f"  [{lname}] varlen_attn{inspect.signature(varlen_attn)}"[:300])
                # torch 2.14: causal = window_size (-1, 0); GQA needs enable_gqa (signature printed above)
                cons.append(autograd_contender(
                    "torch_varlen", lambda a, b_, c: varlen_attn(a, b_, c, cu, cu, mx, mx, scale=SCALE, window_size=(-1, 0),
                                                                 enable_gqa=True), (q, k, v), do, lambda x: x))
            except Exception as e:
                print(f"  [{lname}] torch varlen_attn unavailable: {type(e).__name__}: {str(e)[:160]}")
        if want("flex_docmask"):
            try:
                from torch.nn.attention.flex_attention import create_block_mask, flex_attention
                sid = seg_id

                def mask_mod(b, h, qi, ki):
                    return (sid[b, qi] == sid[b, ki]) & (qi >= ki)
                bm = create_block_mask(mask_mod, B, None, T, T, device=dev)
                flex = torch.compile(flex_attention, dynamic=False)
                cons.append(autograd_contender("flex_docmask", lambda a, b_, c: flex(a, b_, c, block_mask=bm, scale=SCALE,
                                                                                     enable_gqa=True),
                                               (qb, kb, vb), dob, from_bhtd))
            except Exception as e:
                print(f"  [{lname}] flex unavailable: {type(e).__name__}: {str(e)[:160]}")
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_list = [(bname, be, kind) for bname, be in (("flash", SDPBackend.FLASH_ATTENTION),
                                                         ("cudnn", SDPBackend.CUDNN_ATTENTION),
                                                         ("efficient", SDPBackend.EFFICIENT_ATTENTION))
                     for kind in ("fullrow", "mask") if want(f"sdpa_{bname}_{kind}")]
        if sdpa_list:
            blk = (seg_id[:, :, None] == seg_id[:, None, :]) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            blk = blk[:, None]   # [B, 1, T, T], True = may attend
        for bname, be, kind in sdpa_list:
            def call(a, b_, c, be=be, kind=kind):
                with sdpa_kernel(be):
                    if kind == "fullrow":
                        return F.scaled_dot_product_attention(a, b_, c, is_causal=True, scale=SCALE, enable_gqa=True)
                    return F.scaled_dot_product_attention(a, b_, c, attn_mask=blk, scale=SCALE, enable_gqa=True)
            c = autograd_contender(f"sdpa_{bname}_{kind}", call, (qb, kb, vb), dob, from_bhtd if kind == "mask" else None)
            try:   # probe once: a backend that cannot run this shape is reported and skipped
                c.fwd(); c.bwd(); torch.cuda.synchronize()
                cons.append(c)
            except Exception as e:
                print(f"  [{lname}] sdpa {bname} {kind} unavailable: {type(e).__name__}: {str(e)[:120]}")

        cons = [c for c in cons if want(c.name)]
        ok = []
        for c in cons:   # warm-up (FA4 / Flex compile on their first call)
            try:
                for _ in range(WARM):
                    c.fwd(); c.bwd()
                torch.cuda.synchronize()
                ok.append(c)
            except Exception:
                print(f"  [{lname}] {c.name} failed in warm-up:\n" + traceback.format_exc()[-600:])
        for _ in range(ITERS):
            for c in ok:
                c.tf.append(timed(c.fwd, c.name + ":fwd"))
            for c in ok:
                c.tb.append(timed(c.bwd, c.name + ":bwd"))
        for c in ok:
            for d, t in (("fwd", c.tf), ("bwd", c.tb)):
                med = pct(t, 0.5)
                f_ = fl * (2.5 if d == "bwd" else 1.0)
                print(f"BENCH layout={lname} impl={c.name} dir={d} med_ms={med:.4f} p10_ms={pct(t, .1):.4f} "
                      f"p90_ms={pct(t, .9):.4f} tflops={f_ / (med * 1e-3) / 1e12:.1f}", flush=True)
        # agreement: every packed contender vs the first one, on O and dQ (bf16 noise ~1e-2)
        ref = [c for c in ok if c.out_fn]
        if len(ref) > 1:
            for c in ok:
                c.fwd(); c.bwd()
            torch.cuda.synchronize()
            o0, g0 = ref[0].out_fn().float(), ref[0].dq_fn().float()
            for c in ref[1:]:
                do_ = (c.out_fn().float() - o0).abs().max().item()
                dg_ = (c.dq_fn().float() - g0).abs().max().item()
                print(f"AGREE layout={lname} {c.name} vs {ref[0].name}: max|dO_out| {do_:.3e}  max|d dQ| {dg_:.3e}")
        del cons, ok
        torch.cuda.empty_cache()


# ---- cuDNN probe (added 2026-10-12): correctness of the cuDNN routes only, in a process of its own, so that a crash
# in the untested 6-KV-head route cannot take the benchmark down. Expanded route first on every layout, then the
# kernel names it launched, then the 6-KV-head route. The last line is "PROBE_SUMMARY exp=.. gqa=..";
# run_cudnn_varlen.sh sets CUDNN_GQA=1 for the timing run only on "gqa=PASS".
def main_probe():
    path = sys.argv[2]
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
          "| gpu", torch.cuda.get_device_name(0))
    check_gpu()
    layouts = load_layouts(path, "")
    q, k, v, do = make_inputs()
    rng = torch.zeros((2,), dtype=torch.uint64, device=dev)
    res = {"exp": True, "gqa": True}
    first = None
    for lname, segs in layouts.items():
        segs, cu, mx, fl = layout_tensors(segs)
        first = first or (lname, cu, mx)
        exp = [c for c in cudnn_contenders(q, k, v, do, cu, mx, False) if c.name == "cudnn_varlen_exp"]
        res["exp"] &= len(cudnn_gate(lname, exp, q, k, v, do, cu, mx, with_fa4=False)) == 1
        torch.cuda.empty_cache()
    print(f"PROBE_STEP expanded route: {'PASS' if res['exp'] else 'FAIL'} on all {len(layouts)} layouts", flush=True)
    lname, cu, mx = first
    try:
        c = [c for c in cudnn_contenders(q, k, v, do, cu, mx, False) if c.name == "cudnn_varlen_exp"][0]
        for n in kernel_names(c.result):
            print(f"KERNEL cudnn_varlen_exp layout={lname}: {n[:160]}", flush=True)
    except Exception as e:
        print(f"KERNEL names unavailable: {type(e).__name__}: {str(e)[:200]}", flush=True)
    for lname, segs in layouts.items():
        segs, cu, mx, fl = layout_tensors(segs)
        try:
            c = cudnn_gqa_contender(q, k, v, do, cu, mx, rng)
            ok = len(cudnn_gate(lname, [c], q, k, v, do, cu, mx, with_fa4=False)) == 1
        except Exception as e:
            print(f"PROBE gqa layout={lname}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            ok = False
        res["gqa"] &= ok
        if not ok:
            print("PROBE_STEP 6-KV-head route failed; remaining layouts skipped", flush=True)
            break
        torch.cuda.empty_cache()
    if res["gqa"]:
        lname, cu, mx = first
        try:
            for n in kernel_names(cudnn_gqa_contender(q, k, v, do, cu, mx, rng).result):
                print(f"KERNEL cudnn_varlen_gqa layout={lname}: {n[:160]}", flush=True)
        except Exception as e:
            print(f"KERNEL names unavailable: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"PROBE_SUMMARY exp={'PASS' if res['exp'] else 'FAIL'} gqa={'PASS' if res['gqa'] else 'FAIL'}", flush=True)


# ---- CPU-only self-test (added 2026-10-12): everything that can be checked without a GPU. Run with
# CUDA_VISIBLE_DEVICES= so it cannot touch one.
def main_selftest():
    n = 0

    def check(cond, what):
        nonlocal n
        n += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {what}", flush=True)
        if not cond:
            sys.exit(f"SELFTEST FAIL: {what}")
    print("torch", torch.__version__, "| cuda visible:", torch.cuda.is_available())
    # 1. the keyword names used in cudnn_fwd / cudnn_bwd exist in this torch's op schemas
    fs = torch.ops.aten._cudnn_attention_forward.default._schema
    bs = torch.ops.aten._cudnn_attention_backward.default._schema
    fa = {a.name for a in fs.arguments}
    ba = {a.name for a in bs.arguments}
    check({"query", "key", "value", "attn_bias", "cum_seq_q", "cum_seq_k", "max_q", "max_k", "compute_log_sumexp",
           "dropout_p", "is_causal", "return_debug_mask", "scale"} <= fa, "forward keyword names in the op schema")
    check({"grad_out", "query", "key", "value", "out", "logsumexp", "cum_seq_q", "cum_seq_k", "max_q", "max_k",
           "dropout_p", "philox_seed", "philox_offset", "attn_bias", "is_causal", "scale"} <= ba,
          "backward keyword names in the op schema")
    check(fs.returns[0].name == "output" and fs.returns[1].name == "logsumexp", "forward returns (output, logsumexp, ..)")
    print("    ", fs)
    print("    ", bs)
    # 2. layouts: tile every row, offsets increasing, longest segment > 128 (PyTorch refuses cuDNN at <= 128,
    #    varlen.py:56 and nested sdpa.py:250-262)
    lay = load_layouts(sys.argv[2], "")
    check(len(lay) == 7, f"7 layouts in the segments file (got {len(lay)})")
    for name, segs in lay.items():
        segs = sorted(segs)
        starts = [b * T + t0 for b, t0, _ in segs]
        tiles = starts[0] == 0 and all(starts[i] + segs[i][2] == (starts[i + 1] if i + 1 < len(segs) else B * T)
                                       for i in range(len(segs)))
        mx = max(L for _, _, L in segs)
        check(tiles and mx > 128 and min(L for _, _, L in segs) >= 1,
              f"{name}: {len(segs)} segments tile 8 x 4096, longest {mx} > 128")
    # 3. GQA mapping: expanded head h holds KV head h // 3; the group sum adds exactly heads 3g..3g+2; the wrong
    #    mapping used by the negative control differs
    gc = torch.Generator().manual_seed(7)
    kc = torch.randn(5, HKV, D, generator=gc)
    ke = expand_kv(kc)
    check(ke.shape == (5, HQ, D) and all(torch.equal(ke[:, h], kc[:, h // G]) for h in range(HQ)),
          "expand_kv: query head h reads KV head h // 3")
    dk18 = torch.randn(5, HQ, D, generator=gc)
    manual = torch.stack([sum(dk18[:, h] for h in range(HQ) if h // G == g) for g in range(HKV)], 1)
    check(torch.allclose(sum_kv(dk18), manual), "sum_kv: dK of KV head g = sum of query heads 3g..3g+2")
    kw = kc.repeat(1, G, 1)
    check([h for h in range(HQ) if torch.equal(kw[:, h], ke[:, h])] == [0, 8, 9, 17],
          "negative control: 14 of 18 heads read the wrong KV head (only 0, 8, 9, 17 coincide)")
    # 4. the gate: same -> 0, bf16-size noise passes, NaN / inf / wrong values fail
    a = torch.randn(1000, generator=gc)
    check(rel_err(a, a) == 0.0, "rel_err(x, x) = 0")
    check(rel_err(a + 1e-3 * a.abs().max(), a) <= TOL_REL, "noise of 1e-3 relative passes")
    bad = a.clone(); bad[3] = float("nan")
    check(rel_err(bad, a) == float("inf"), "a NaN in the result fails")
    check(not rel_err(a, bad) <= TOL_REL, "a NaN in the reference fails")
    check(rel_err(torch.randn(1000, generator=gc), a) > TOL_REL, "unrelated values fail")
    check(not gate_line("selftest wrong", "cpu", [a, a, a, a + 0.5 * a.abs().max()], [a, a, a, a]),
          "gate_line fails when one of the four tensors is off")
    check(gate_line("selftest same", "cpu", [a, a, a, a], [a, a, a, a]), "gate_line passes on four equal tensors")
    # 5. BENCH_ONLY prefixes
    os.environ["BENCH_ONLY"] = "fa3,fa4,cudnn"
    check(want("fa3_varlen_det") and want("cudnn_varlen_gqa") and not want("sdpa_cudnn_mask")
          and not want("torch_varlen"), "BENCH_ONLY=fa3,fa4,cudnn keeps fa3*/fa4*/cudnn* only")
    os.environ["BENCH_ONLY"] = "fa3"
    check(want("fa3_varlen") and not want("fa4_varlen"), "single prefix behaves as before")
    os.environ["BENCH_ONLY"] = "fa3_varlen,cudnn_varlen_exp"
    check(cudnn_wanted() and want("cudnn_varlen_exp") and want("cudnn_varlen_expsum") and not want("cudnn_varlen_gqa")
          and not want("fa4_varlen"), "BENCH_ONLY=fa3_varlen,cudnn_varlen_exp builds cuDNN, keeps exp + expsum only")
    os.environ["BENCH_ONLY"] = "cudnn_varlen_gqa"
    check(cudnn_wanted() and want("cudnn_varlen_gqa") and not want("cudnn_varlen_exp"),
          "BENCH_ONLY=cudnn_varlen_gqa builds cuDNN, keeps gqa only")
    os.environ["BENCH_ONLY"] = "fa3"
    check(not cudnn_wanted(), "BENCH_ONLY=fa3 builds no cuDNN contender")
    src = inspect.getsource(cudnn_contenders) + inspect.getsource(cudnn_gqa_contender)
    check(all(f'"{n}"' in src for n in CUDNN_NAMES), "CUDNN_NAMES are the names the contenders really get")
    del os.environ["BENCH_ONLY"]
    check(want("anything") and cudnn_wanted(), "no BENCH_ONLY keeps everything")
    # 6. the profiler API used for the kernel-name proof exists
    from torch.profiler import ProfilerActivity
    check(hasattr(ProfilerActivity, "CUDA") and hasattr(torch.autograd.DeviceType, "CUDA"),
          "torch.profiler CUDA activity and DeviceType.CUDA exist")
    print(f"SELFTEST PASS ({n} checks)")


# ---- Step 6a: dense causal attention on the colleagues' shapes (same list as the harness's BENCHDENSE)
DENSE_SHAPES = [(2, 12, 4), (4, 12, 4), (8, 12, 4), (2, 16, 4), (4, 16, 4), (8, 16, 4), (2, 24, 8), (4, 24, 8),
                (8, 24, 8), (2, 32, 8), (4, 32, 8), (8, 32, 8), (8, 18, 6)]


def main_dense():
    print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), "| gpu", torch.cuda.get_device_name(0))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(f):
        ev0.record(); f(); ev1.record(); ev1.synchronize()
        return ev0.elapsed_time(ev1)
    g = torch.Generator(device=dev).manual_seed(4321)
    only = sys.argv[2] if len(sys.argv) > 2 else ""   # one shape (profiling)
    for Bn, Hq, Hkv in DENSE_SHAPES:
        name = f"dense_B{Bn}_H{Hq}x{Hkv}"
        if only and name != only:
            continue
        rnd = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g).to(DT)
        q, k, v, do = rnd(Bn, T, Hq, D), rnd(Bn, T, Hkv, D), rnd(Bn, T, Hkv, D), rnd(Bn, T, Hq, D)   # FA layout
        qb, kb, vb, dob = [x.transpose(1, 2).contiguous() for x in (q, k, v, do)]                     # SDPA layout
        fl = 4.0 * Bn * Hq * D * T * T / 2
        cons = []
        try:
            fa3 = importlib.import_module("flash_attn_interface")
            st = {}
            dq3, dk3, dv3 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

            def f3():
                st["o"], st["lse"] = fa3._flash_attn_forward(q, k, v, softmax_scale=SCALE, causal=True)[:2]
            for det in (False, True):
                def b3(det=det):
                    fa3._flash_attn_backward(do, q, k, v, st["o"], st["lse"], dq=dq3, dk=dk3, dv=dv3, softmax_scale=SCALE,
                                             is_causal=True, deterministic=det)
                cons.append(Contender("fa3" + ("_det" if det else ""), f3, b3))
        except Exception as e:
            print(f"  [{name}] fa3 unavailable: {type(e).__name__}: {str(e)[:160]}")
        try:
            fa4 = importlib.import_module("flash_attn.cute.interface")
            s4 = {}

            def f4():
                r = fa4._flash_attn_fwd(q, k, v, softmax_scale=SCALE, causal=True, return_lse=True)
                s4["o"], s4["lse"] = r[0], r[1]

            def b4():
                fa4._flash_attn_bwd(q, k, v, s4["o"], do, s4["lse"], softmax_scale=SCALE, causal=True)
            cons.append(Contender("fa4", f4, b4))
        except Exception as e:
            print(f"  [{name}] fa4 unavailable: {type(e).__name__}: {str(e)[:160]}")
        for bname, be in (("cudnn", SDPBackend.CUDNN_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION)):
            leaf = [x.detach().clone().requires_grad_(True) for x in (qb, kb, vb)]
            st2 = {}

            def fs(be=be, leaf=leaf, st2=st2):
                with sdpa_kernel(be):
                    st2["o"] = F.scaled_dot_product_attention(*leaf, is_causal=True, scale=SCALE, enable_gqa=True)

            def bs(leaf=leaf, st2=st2):
                torch.autograd.grad(st2["o"], leaf, dob, retain_graph=True)
            c = Contender("sdpa_" + bname, fs, bs)
            try:
                c.fwd(); c.bwd(); torch.cuda.synchronize(); cons.append(c)
            except Exception as e:
                print(f"  [{name}] sdpa {bname} unavailable: {type(e).__name__}: {str(e)[:120]}")
        ok = []
        for c in cons:
            try:
                for _ in range(WARM):
                    c.fwd(); c.bwd()
                torch.cuda.synchronize(); ok.append(c)
            except Exception:
                print(f"  [{name}] {c.name} failed in warm-up:\n" + traceback.format_exc()[-400:])
        for _ in range(ITERS):
            for c in ok:
                c.tf.append(timed(c.fwd))
            for c in ok:
                c.tb.append(timed(c.bwd))
        for c in ok:
            for d, t in (("fwd", c.tf), ("bwd", c.tb)):
                med = pct(t, 0.5)
                print(f"BENCH layout={name} impl={c.name} dir={d} med_ms={med:.4f} p10_ms={pct(t, .1):.4f} "
                      f"p90_ms={pct(t, .9):.4f} tflops={fl * (2.5 if d == 'bwd' else 1.0) / (med * 1e-3) / 1e12:.1f}", flush=True)
        del cons, ok
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dense":
        main_dense()
    elif len(sys.argv) > 2 and sys.argv[1] == "--cudnn-probe":
        main_probe()
    elif len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        main_selftest()
    else:
        main()
