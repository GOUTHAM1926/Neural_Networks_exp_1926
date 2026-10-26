#!/usr/bin/env python3
# bench_rope_rivals.py (2026-10-21): QK-RMSNorm + RoPE of the other libraries at OUR SFT shape, next to bench_rope_sft.cu.
#   q [8, 18, 4096, 128], k [8, 6, 4096, 128] fp32, [B, H, T, hd] contiguous (what our trainer feeds the fused kernel),
#   gammas [128] (the bench's values), eps 1e-6, NeoX rotate-half RoPE, theta 10000.
#   cases  identity = positions t (cos / sin [1, T, hd], broadcast over the rows); packed = positions restart at every chat
#          of the real packed rows 0-7 (cos / sin gathered per row, [B, T, hd]). HF builds cos / sin once per model forward
#          (rotary_emb), so the gather is NOT timed.
#   impls  torch_eager   = HF Qwen3RMSNorm (fp32) on q and k + HF apply_rotary_pos_emb (rotate_half), autograd backward
#          torch_compile = torch.compile of the same function (one compiled function per case, dynamic=False)
#          liger         = liger_rms_norm (in_place) + liger_rotary_pos_emb (Triton)
#   timing F = forward; B = backward alone (after an untimed forward); F+B; CUDA events per call; >= --warm s of F+B first;
#          median and min of --iters calls each. Every backward gets fresh copies of the output gradients, made OUTSIDE the
#          timed region: Liger's backward overwrites its incoming gradients in place.
#   memory kept = what the forward keeps for the backward (allocated after F, minus before F, minus the two outputs);
#          peak = the most allocated during F+B above what was allocated before it
#   check  every impl's q_out, k_out, dq, dk, dgamma_q against an fp64 eager reference: max abs error
# usage: python bench_rope_rivals.py <packed train dir> [--iters 300] [--warm 4] [--impls a,b]
import argparse, statistics, sys, time
import torch

B, HQ, HKV, T, HD = 8, 18, 6, 4096, 128
EPS, THETA = 1e-6, 10000.0


def log(*a):
    print(*a, flush=True)


def packed_positions(d):
    lens = torch.frombuffer(bytearray(open(f"{d}/seq_lengths.bin", "rb").read()), dtype=torch.int32)
    offs = torch.frombuffer(bytearray(open(f"{d}/seq_offsets.bin", "rb").read()), dtype=torch.int64)
    pos = torch.empty(B, T, dtype=torch.int64)
    chats = 0
    for b in range(B):
        t = 0
        for c in range(int(offs[b]), int(offs[b + 1])):
            n = min(int(lens[c]), T - t)
            pos[b, t:t + n] = torch.arange(n)
            t += n
            chats += 1
        pos[b, t:] = torch.arange(T - t)   # pad tail: its own segment, positions 0, 1, 2, ...
    return pos, chats


def cos_sin(pos, dtype):   # pos [b, T] -> cos, sin [b, T, hd] (HF's rotary_emb, NeoX layout)
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64, device=pos.device) / HD))
    f = pos.to(torch.float64)[..., None] * inv
    e = torch.cat((f, f), dim=-1)
    return e.cos().to(dtype), e.sin().to(dtype)


def rms(x, w):   # HF Qwen3RMSNorm on fp32 input
    var = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + EPS))


def rotate_half(x):
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat((-x2, x1), dim=-1)


def eager(q, k, wq, wk, cos, sin):
    q, k = rms(q, wq), rms(k, wk)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


def make_liger():
    from liger_kernel.transformers.functional import liger_rms_norm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    def f(q, k, wq, wk, cos, sin):
        return liger_rotary_pos_emb(liger_rms_norm(q, wq, EPS), liger_rms_norm(k, wk, EPS), cos, sin)
    return f


def ev():
    return torch.cuda.Event(enable_timing=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packed_dir")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warm", type=float, default=4.0)
    ap.add_argument("--impls", default="torch_eager,torch_compile,liger")
    a = ap.parse_args()
    dev = torch.device("cuda")
    log(f"device {torch.cuda.get_device_name()} | torch {torch.__version__} | q [{B},{HQ},{T},{HD}] k [{B},{HKV},{T},{HD}] fp32 "
        f"| theta {THETA:.0f} eps {EPS} | iters {a.iters} warm {a.warm} s")
    g = torch.Generator(device=dev).manual_seed(1234)
    q0 = torch.randn(B, HQ, T, HD, device=dev, generator=g)
    k0 = torch.randn(B, HKV, T, HD, device=dev, generator=g)
    gq = torch.randn(B, HQ, T, HD, device=dev, generator=g)
    gk = torch.randn(B, HKV, T, HD, device=dev, generator=g)
    w = torch.tensor([1.0 + 0.01 * ((i * 37) % 17 - 8) for i in range(HD)], device=dev)
    pos_p, chats = packed_positions(a.packed_dir)
    log(f"packed rows 0-7: {chats} chats ({chats / B:.2f} per row)")
    cases = {"identity": torch.arange(T, device=dev)[None, :], "packed": pos_p.to(dev)}
    out_bytes = (q0.numel() + k0.numel()) * 4

    def grads():   # fresh copies: Liger's backward overwrites the incoming gradients in place
        return gq.clone(), gk.clone()

    for case, pos in cases.items():
        cos, sin = cos_sin(pos, torch.float32)
        # fp64 reference (the eager maths in double)
        qd, kd = q0.double().requires_grad_(), k0.double().requires_grad_()
        wqd, wkd = w.double().requires_grad_(), w.double().requires_grad_()
        cd, sd = cos_sin(pos, torch.float64)
        rq, rk = eager(qd, kd, wqd, wkd, cd, sd)
        torch.autograd.backward((rq, rk), (gq.double(), gk.double()))
        ref = {"q_out": rq.detach(), "k_out": rk.detach(), "dq": qd.grad, "dk": kd.grad, "dwq": wqd.grad}
        del qd, kd, wkd, rq, rk, cd, sd
        torch.cuda.empty_cache()
        for name in a.impls.split(","):
            try:
                if name == "torch_eager": fn = eager
                elif name == "torch_compile": fn = torch.compile(eager, dynamic=False)
                elif name == "liger": fn = make_liger()
                else: log(f"SKIP impl={name} case={case}: unknown"); continue
                q = q0.clone().requires_grad_(); k = k0.clone().requires_grad_()
                wq = w.clone().requires_grad_(); wk = w.clone().requires_grad_()

                def run():
                    return fn(q, k, wq, wk, cos, sin)

                def clear():
                    q.grad = k.grad = wq.grad = wk.grad = None
                # warm (compile, autotune, power): >= --warm s of F+B
                t0 = time.time(); nw = 0
                while time.time() - t0 < a.warm or nw < 3:
                    torch.autograd.backward(run(), grads()); clear(); nw += 1
                torch.cuda.synchronize()
                # correctness: one fresh F+B against the fp64 reference
                o = run(); qo, ko = o[0].detach().clone(), o[1].detach().clone()
                torch.autograd.backward(o, grads()); del o
                got = {"q_out": qo, "k_out": ko, "dq": q.grad, "dk": k.grad, "dwq": wq.grad}
                err = {n2: (v.double() - ref[n2]).abs().max().item() for n2, v in got.items()}
                del qo, ko, got
                clear()
                # memory: kept by the forward; peak of F+B
                torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
                o = run(); torch.cuda.synchronize(); kept = torch.cuda.memory_allocated() - m0 - out_bytes
                del o
                gg = grads(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
                torch.autograd.backward(run(), gg); torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - m0
                del gg; clear()
                # timing
                tf, tb, tfb = [], [], []
                for _ in range(a.iters):
                    e0, e1 = ev(), ev()
                    e0.record(); o = run(); e1.record(); e1.synchronize()
                    tf.append(e0.elapsed_time(e1)); del o
                for _ in range(a.iters):
                    o = run(); gg = grads(); torch.cuda.synchronize()
                    e0, e1 = ev(), ev()
                    e0.record(); torch.autograd.backward(o, gg); e1.record(); e1.synchronize()
                    tb.append(e0.elapsed_time(e1)); del o, gg; clear()
                for _ in range(a.iters):
                    gg = grads(); torch.cuda.synchronize()
                    e0, e1 = ev(), ev()
                    e0.record(); torch.autograd.backward(run(), gg); e1.record(); e1.synchronize()
                    tfb.append(e0.elapsed_time(e1)); del gg; clear()
                med = statistics.median
                log(f"ROPE impl={name} case={case} fwd_med_ms={med(tf):.4f} fwd_min_ms={min(tf):.4f} bwd_med_ms={med(tb):.4f} "
                    f"bwd_min_ms={min(tb):.4f} fb_med_ms={med(tfb):.4f} fb_min_ms={min(tfb):.4f} kept_MiB={kept / 2**20:.1f} "
                    f"peak_MiB={peak / 2**20:.1f} err_q_out={err['q_out']:.2e} err_k_out={err['k_out']:.2e} "
                    f"err_dq={err['dq']:.2e} err_dk={err['dk']:.2e} err_dwq={err['dwq']:.2e}")
                del q, k, wq, wk, fn
                torch.cuda.empty_cache()
            except Exception as e:   # one library failing must not stop the others
                log(f"FAIL impl={name} case={case}: {type(e).__name__}: {str(e)[:300]}")
                torch.cuda.empty_cache()
        del ref
        torch.cuda.empty_cache()
    log("ROPE RIVALS DONE")


if __name__ == "__main__":
    sys.exit(main())
