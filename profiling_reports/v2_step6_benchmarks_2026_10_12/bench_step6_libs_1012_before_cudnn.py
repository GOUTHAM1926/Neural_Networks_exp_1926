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
import importlib, inspect, math, os, sys, traceback

import torch
import torch.nn.functional as F

B, T, HQ, HKV, D = 8, 4096, 18, 6, 128
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


class Contender:
    def __init__(self, name, fwd, bwd, out_fn=None, dq_fn=None):
        self.name, self.fwd, self.bwd, self.out_fn, self.dq_fn = name, fwd, bwd, out_fn, dq_fn
        self.tf, self.tb = [], []


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
          "| gpu", torch.cuda.get_device_name(0))
    layouts = load_layouts(path, filt)
    g = torch.Generator(device=dev).manual_seed(1234)
    rnd = lambda *s: torch.randn(*s, device=dev, dtype=torch.float32, generator=g).to(DT)
    q, k, v, do = rnd(B * T, HQ, D), rnd(B * T, HKV, D), rnd(B * T, HKV, D), rnd(B * T, HQ, D)
    to_bhtd = lambda x, H: x.view(B, T, H, D).transpose(1, 2).contiguous()
    qb, kb, vb, dob = to_bhtd(q, HQ), to_bhtd(k, HKV), to_bhtd(v, HKV), to_bhtd(do, HQ)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(f):
        ev0.record()
        f()
        ev1.record()
        ev1.synchronize()
        return ev0.elapsed_time(ev1)

    for lname, segs in layouts.items():
        segs = sorted(segs)
        starts = [b * T + t0 for b, t0, _ in segs]
        assert starts[0] == 0 and all(starts[i] + segs[i][2] == (starts[i + 1] if i + 1 < len(segs) else B * T)
                                      for i in range(len(segs))), "segments must tile every row"
        cu = torch.tensor(starts + [B * T], dtype=torch.int32, device=dev)
        mx = max(L for _, _, L in segs)
        fl = sum(4.0 * HQ * D * L * L / 2 for _, _, L in segs)
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
        try:
            from torch.nn.attention.varlen import varlen_attn
            print(f"  [{lname}] varlen_attn{inspect.signature(varlen_attn)}"[:300])
            # torch 2.14: causal = window_size (-1, 0); GQA needs enable_gqa (signature printed above)
            cons.append(autograd_contender(
                "torch_varlen", lambda a, b_, c: varlen_attn(a, b_, c, cu, cu, mx, mx, scale=SCALE, window_size=(-1, 0),
                                                             enable_gqa=True), (q, k, v), do, lambda x: x))
        except Exception as e:
            print(f"  [{lname}] torch varlen_attn unavailable: {type(e).__name__}: {str(e)[:160]}")
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
        blk = (seg_id[:, :, None] == seg_id[:, None, :]) & torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        blk = blk[:, None]   # [B, 1, T, T], True = may attend
        for bname, be in (("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION),
                          ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
            for kind in ("fullrow", "mask"):
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

        only = os.environ.get("BENCH_ONLY", "")   # e.g. "fa3": keep only contenders whose name starts with it
        if only:
            cons = [c for c in cons if c.name.startswith(only)]
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
                c.tf.append(timed(c.fwd))
            for c in ok:
                c.tb.append(timed(c.bwd))
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
    else:
        main()
