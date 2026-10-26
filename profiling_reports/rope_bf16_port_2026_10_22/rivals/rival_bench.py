#!/usr/bin/env python3
"""rival_bench.py (2026-10-22): bf16 QK-RMSNorm + RoPE on q and k, FORWARD and BACKWARD timed separately, ours vs rivals.

Cases
  pretraining : positions 0..T-1 in every row (ours runs its no-positions kernel)
  sft         : packed rows, chats of 1..1500 tokens, positions restart at 0 in every chat and in the pad tail
Input settings: --B (batch rows of T tokens) x --shapes (query heads x KV heads, head dim 128) at --T.

Implementations (each in its own natural setup, as it would be used for training)
  ours       Kruthik's fused bf16 kernel (one launch forward; backward = 1 kernel + 2 dgamma reduces), called through
             libropeours.so. Reads token-major q, k; writes head-major outputs (it fuses the transpose to [B,H,T,hd]).
             fp32 gamma, fp32 cos|sin table [T,128].
  eager      Hugging Face Qwen3/Llama code (RMSNorm, then rotate_half RoPE), bf16 weights, bf16 cos/sin [B,T,128]
  compile    torch.compile of the same eager code
  liger      LigerRMSNormFunction + LigerRopeFunction (Triton), bf16 weights, bf16 cos/sin [B,T,128]
  flashattn  flash_attn Triton rms_norm_fn + apply_rotary_emb (cu_seqlens for packed rows), fp32 weights,
             fp32 cos/sin [T,64]
  quack      Quack rmsnorm (CuTe DSL) + flash_attn Triton RoPE, fp32 weights, fp32 cos/sin [T,64]
Rivals write their outputs in their own layout (token-major or a transposed view); none of them pays for a transpose.

Checks (at --check-B, before any timing): outputs, input gradients and gamma gradients against an fp64 reference.
  bf16 tensors: err = max|x - ref| / max|bf16(ref) - ref|  (1.0 = as accurate as rounding the exact result to bf16)
  gamma grads : rel = max|x - ref| / max|ref|
Timing: event = CUDA-event time per call, `iters` back-to-back calls, median of `rounds`; kernel = summed GPU time of
all device work per call from torch.profiler (no CPU launch gaps), with the number of kernels per call.
usage: python rival_bench.py --lib ./libropeours.so --vendor ./vendor [--B 1,2,4,8] [--shapes 18x6,32x8] [--only ...]
"""
import argparse, ctypes, json, statistics, sys, traceback
import torch

HD, HALF, EPS, THETA = 128, 64, 1e-6, 50000.0
BF16, F32 = torch.bfloat16, torch.float32


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--B", default="1,2,4,8")
    ap.add_argument("--shapes", default="18x6,32x8")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--check-B", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list of implementation names")
    ap.add_argument("--out", default="rival_results.json")
    return ap.parse_args()


# ----------------------------------------------------------------------------------------------- inputs ----
def packed_positions(B, T, gen):
    pos, segs = torch.empty(B, T, dtype=torch.int32), []
    for b in range(B):
        t = 0
        while t < T:
            L = min(int(torch.randint(1, 1501, (1,), generator=gen)), T - t)
            pos[b, t:t + L] = torch.arange(L, dtype=torch.int32)
            segs.append(L)
            t += L
    return pos, segs


class Case:
    def __init__(self, B, T, Hq, Hkv, packed, dev):
        g = torch.Generator().manual_seed(1234 + B * 7 + Hq + (1000 if packed else 0))
        self.B, self.T, self.Hq, self.Hkv, self.packed = B, T, Hq, Hkv, packed
        mk = lambda H: torch.randn(B, T, H, HD, generator=g).to(BF16).to(dev)
        self.q, self.k, self.dzq, self.dzk = mk(Hq), mk(Hkv), mk(Hq), mk(Hkv)
        # gammas exactly representable in bf16, so the fp32 and bf16 weights are the same numbers
        gam = lambda: (1.0 + torch.randint(-16, 17, (HD,), generator=g).float() / 64.0).to(dev)
        self.wq, self.wk = gam(), gam()
        if packed:
            pos, segs = packed_positions(B, T, g)
        else:
            pos, segs = torch.arange(T, dtype=torch.int32).repeat(B, 1), [T] * B
        self.pos = pos.to(dev)
        lens = torch.tensor(segs, dtype=torch.int32)
        self.cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)]).to(dev)
        self.maxlen = int(lens.max())
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
        ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]
        self.cos64, self.sin64 = ang.cos().to(dev), ang.sin().to(dev)                 # [T,64] fp64 (reference)
        self.cos_h, self.sin_h = self.cos64.float(), self.sin64.float()               # [T,64] fp32 (flash_attn)
        self.cs_ours = torch.cat([self.cos64, self.sin64], 1).float().contiguous()    # [T,128] cos|sin (ours)
        p = self.pos.long()                                                            # HF cos/sin [B,T,128] bf16
        self.cos_hf = torch.cat([self.cos64, self.cos64], 1)[p].to(BF16)
        self.sin_hf = torch.cat([self.sin64, self.sin64], 1)[p].to(BF16)


def reference(c):
    """fp64: y = rope(x * rsqrt(mean(x^2) + eps) * w), and the gradients of sum(y * dz)."""
    p = c.pos.long()
    cs, sn = c.cos64[p][:, :, None, :], c.sin64[p][:, :, None, :]
    out = []
    for x, w, dz in ((c.q, c.wq, c.dzq), (c.k, c.wk, c.dzk)):
        x = x.double().requires_grad_(True)
        w = w.double().requires_grad_(True)
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
        a, b = n[..., :HALF], n[..., HALF:]
        y = torch.cat([a * cs - b * sn, a * sn + b * cs], -1)
        gx, gw = torch.autograd.grad(y, (x, w), dz.double())
        out.append((y.detach(), gx, gw))
    (yq, gq, gwq), (yk, gk, gwk) = out
    return yq, yk, gq, gk, gwq, gwk


def err_bf16(x, ref):
    e = (x.double() - ref).abs().max().item()
    r = (ref.to(BF16).double() - ref).abs().max().item()
    return e / r if r > 0 else float("inf")


def rel(x, ref):
    return (x.double() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)


# ------------------------------------------------------------------------------------------ ours (ctypes) ----
class Ours:
    name = "ours"

    def __init__(self, lib):
        self.L = ctypes.CDLL(lib)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        self.L.ours_fwd.argtypes = [P] * 9 + [I] * 5 + [F, P, P]
        self.L.ours_bwd.argtypes = [P] * 13 + [I] * 5 + [P, P]

    def prepare(self, c):
        B, T, Hq, Hkv, L = c.B, c.T, c.Hq, c.Hkv, self.L
        d = lambda t: ctypes.c_void_p(t.data_ptr())
        q, k = c.q.contiguous(), c.k.contiguous()
        qo, ko = torch.empty(B, Hq, T, HD, dtype=BF16, device=q.device), torch.empty(B, Hkv, T, HD, dtype=BF16, device=q.device)
        qr, kr = torch.empty(B, Hq, T, dtype=F32, device=q.device), torch.empty(B, Hkv, T, dtype=F32, device=q.device)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dwq, dwk = torch.empty(HD, dtype=F32, device=q.device), torch.empty(HD, dtype=F32, device=q.device)
        dzq, dzk = c.dzq.permute(0, 2, 1, 3).contiguous(), c.dzk.permute(0, 2, 1, 3).contiguous()   # head-major grads
        pos = d(c.pos) if c.packed else None          # pretraining: the no-positions kernel (positions = t)
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

        def fwd():
            L.ours_fwd(d(q), d(k), d(qo), d(ko), d(c.wq), d(c.wk), d(c.cs_ours), d(qr), d(kr), B, Hq, Hkv, T, HD,
                       ctypes.c_float(EPS), pos, st)

        def bwd():
            L.ours_bwd(d(dzq), d(dzk), d(q), d(k), d(c.wq), d(c.wk), d(c.cs_ours), d(qr), d(kr), d(dq), d(dk),
                       d(dwq), d(dwk), B, Hq, Hkv, T, HD, pos, st)

        def results():
            fwd(); bwd(); torch.cuda.synchronize()
            return qo.permute(0, 2, 1, 3), ko.permute(0, 2, 1, 3), dq, dk, dwq, dwk

        return fwd, bwd, results


# ---------------------------------------------------------------------------------------- autograd rivals ----
def hf_rmsnorm(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(x.dtype)


def rotate_half(x):
    return torch.cat((-x[..., HALF:], x[..., :HALF]), -1)


def hf_qk(q, k, wq, wk, cos, sin):                  # q [B,T,H,hd] -> outputs [B,H,T,hd] (views of token-major data)
    qn, kn = hf_rmsnorm(q, wq).transpose(1, 2), hf_rmsnorm(k, wk).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return qn * c + rotate_half(qn) * s, kn * c + rotate_half(kn) * s


class Autograd:
    """Any rival built from torch autograd functions. fn(c, q, k, wq, wk) -> (yq, yk); `head_major_out` tells whether
    the outputs are [B,H,T,hd] (Hugging Face style) or [B,T,H,hd]."""

    def __init__(self, name, fn, w_dtype, head_major_out):
        self.name, self.fn, self.w_dtype, self.hm = name, fn, w_dtype, head_major_out

    def prepare(self, c):
        q, k = c.q.clone().requires_grad_(True), c.k.clone().requires_grad_(True)
        wq, wk = c.wq.to(self.w_dtype).requires_grad_(True), c.wk.to(self.w_dtype).requires_grad_(True)
        # Every rival gets its OWN copy of the upstream gradients, made fresh again for the accuracy check: Liger's RoPE
        # backward rotates its incoming gradient IN PLACE, and with one shared c.dzq the rivals after it (flashattn, quack)
        # were checked against a corrupted gradient (dx error 1307 instead of 2.15; found 2026-10-23, debug_fa_bwd.py).
        g0 = (c.dzq.transpose(1, 2) if self.hm else c.dzq, c.dzk.transpose(1, 2) if self.hm else c.dzk)
        st = {"g_out": tuple(t.clone() for t in g0)}

        def fwd():
            st["y"] = self.fn(c, q, k, wq, wk)

        def bwd():
            st["g"] = torch.autograd.grad(st["y"], (q, k, wq, wk), st["g_out"], retain_graph=True)

        def results():
            st["g_out"] = tuple(t.clone() for t in g0)
            fwd(); bwd(); torch.cuda.synchronize()
            yq, yk = st["y"]
            if self.hm:
                yq, yk = yq.transpose(1, 2), yk.transpose(1, 2)
            g = st["g"]
            return yq, yk, g[0], g[1], g[2].float(), g[3].float()

        return fwd, bwd, results


def build_rivals(args):
    sys.path.insert(0, args.vendor)
    rivals = [Ours(args.lib)]
    rivals.append(Autograd("eager", lambda c, q, k, wq, wk: hf_qk(q, k, wq, wk, c.cos_hf, c.sin_hf), BF16, True))
    compiled = torch.compile(hf_qk, dynamic=False)
    rivals.append(Autograd("compile", lambda c, q, k, wq, wk: compiled(q, k, wq, wk, c.cos_hf, c.sin_hf), BF16, True))
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        from liger_kernel.ops.rope import LigerRopeFunction

        def liger(c, q, k, wq, wk):
            qn = LigerRMSNormFunction.apply(q, wq, EPS, 0.0, "llama", False)    # in_place=False: dY is reused
            kn = LigerRMSNormFunction.apply(k, wk, EPS, 0.0, "llama", False)
            return LigerRopeFunction.apply(qn.transpose(1, 2), kn.transpose(1, 2), c.cos_hf, c.sin_hf)
        rivals.append(Autograd("liger", liger, BF16, True))
    except Exception as e:
        print(f"[skip] liger: {e!r}")
    try:
        from flash_attn.ops.triton.layer_norm import rms_norm_fn
        from flash_attn.layers.rotary import apply_rotary_emb

        def fa_rope(c, x):
            if not c.packed:
                return apply_rotary_emb(x, c.cos_h, c.sin_h)
            B, T, H, D = x.shape
            y = apply_rotary_emb(x.reshape(B * T, H, D), c.cos_h, c.sin_h, cu_seqlens=c.cu, max_seqlen=c.maxlen)
            return y.view(B, T, H, D)

        def flashattn(c, q, k, wq, wk):
            return fa_rope(c, rms_norm_fn(q, wq, None, eps=EPS)), fa_rope(c, rms_norm_fn(k, wk, None, eps=EPS))
        rivals.append(Autograd("flashattn", flashattn, F32, False))
        try:
            from quack.rmsnorm import rmsnorm as quack_rmsnorm

            def quack(c, q, k, wq, wk):
                qn = quack_rmsnorm(q.reshape(-1, HD), wq, eps=EPS).view(q.shape)
                kn = quack_rmsnorm(k.reshape(-1, HD), wk, eps=EPS).view(k.shape)
                return fa_rope(c, qn), fa_rope(c, kn)
            rivals.append(Autograd("quack", quack, F32, False))
        except Exception as e:
            print(f"[skip] quack: {e!r}")
    except Exception as e:
        print(f"[skip] flashattn (and quack, which uses its RoPE): {e!r}")
    if args.only:
        keep = set(args.only.split(","))
        rivals = [r for r in rivals if r.name in keep]
    return rivals


# ----------------------------------------------------------------------------------------------- timing ----
def event_ms(fn, iters, rounds):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    res = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        res.append(s.elapsed_time(e) / iters)
    return statistics.median(res)


def kernel_us(fn, n=10):
    from torch.autograd import DeviceType
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    dev = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    return sum(e.time_range.elapsed_us() for e in dev) / n, len(dev) / n


# ------------------------------------------------------------------------------------------------- main ----
def main():
    args = parse()
    dev = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name(0)} | torch {torch.__version__} | T {args.T} | B {args.B} | "
          f"shapes {args.shapes} | {args.iters} calls x median of {args.rounds}")
    rivals = build_rivals(args)
    print("implementations:", ", ".join(r.name for r in rivals))
    results = []
    for shape in args.shapes.split(","):
        Hq, Hkv = map(int, shape.split("x"))
        for packed in (False, True):
            case_name = "sft" if packed else "pretraining"
            for B in map(int, args.B.split(",")):
                c = Case(B, args.T, Hq, Hkv, packed, dev)
                ref = reference(c) if B == args.check_B else None
                for r in rivals:
                    row = dict(shape=shape, case=case_name, B=B, impl=r.name)
                    try:
                        fwd, bwd, res = r.prepare(c)
                        out = res()                                   # also warms up / compiles
                        if ref is not None:
                            yq, yk, gq, gk, gwq, gwk = out
                            row.update(err_y=max(err_bf16(yq, ref[0]), err_bf16(yk, ref[1])),
                                       err_dx=max(err_bf16(gq, ref[2]), err_bf16(gk, ref[3])),
                                       rel_dw=max(rel(gwq, ref[4]), rel(gwk, ref[5])))
                        row["fwd_event_us"] = 1e3 * event_ms(fwd, args.iters, args.rounds)
                        row["bwd_event_us"] = 1e3 * event_ms(bwd, args.iters, args.rounds)
                        row["fwd_kernel_us"], row["fwd_kernels"] = kernel_us(fwd)
                        row["bwd_kernel_us"], row["bwd_kernels"] = kernel_us(bwd)
                    except Exception as e:
                        row["error"] = repr(e)[:300]
                        traceback.print_exc(limit=2)
                    results.append(row)
                    del row
                del c, ref
                torch.cuda.empty_cache()
            # ---- table for this shape and case ----
            print(f"\n=== {shape} heads, {case_name}: microseconds per call (kernel = summed GPU time; event = wall) ===")
            print(f"{'B':>2} {'impl':10} {'fwd kern':>9} {'(n)':>4} {'fwd evt':>8} {'bwd kern':>9} {'(n)':>4} {'bwd evt':>8}"
                  f" {'total kern':>10} {'x ours':>7} | {'err y':>6} {'err dx':>6} {'rel dw':>8}")
            for B in map(int, args.B.split(",")):
                rows = [x for x in results if x["shape"] == shape and x["case"] == case_name and x["B"] == B]
                base = next((x for x in rows if x["impl"] == "ours" and "error" not in x), None)
                for x in rows:
                    if "error" in x:
                        print(f"{B:>2} {x['impl']:10} ERROR {x['error'][:110]}")
                        continue
                    tot = x["fwd_kernel_us"] + x["bwd_kernel_us"]
                    ratio = tot / (base["fwd_kernel_us"] + base["bwd_kernel_us"]) if base else float("nan")
                    chk = (f"{x['err_y']:6.2f} {x['err_dx']:6.2f} {x['rel_dw']:8.1e}" if "err_y" in x else "")
                    print(f"{B:>2} {x['impl']:10} {x['fwd_kernel_us']:9.1f} {x['fwd_kernels']:4.0f} {x['fwd_event_us']:8.1f}"
                          f" {x['bwd_kernel_us']:9.1f} {x['bwd_kernels']:4.0f} {x['bwd_event_us']:8.1f} {tot:10.1f}"
                          f" {ratio:7.2f} | {chk}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
