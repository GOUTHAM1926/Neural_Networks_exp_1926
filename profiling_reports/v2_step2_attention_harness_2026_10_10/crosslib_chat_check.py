#!/usr/bin/env python3
"""crosslib_chat_check.py -- run every available PyTorch attention backend on the exact bytes of
ONE chat exported by test_varlen_attention.cu (VARLEN_EXPORT_DIR), and judge each backend with
the harness's own per-chat rule against the harness's fp64 truth and bf16 floor.

Question it answers (2026-10-10): our Vj1d computes dS with a bf16 P; FA2/FA3/FA4/Flex use an
fp32 P; PyTorch's memory-efficient kernel uses a bf16 P; cuDNN is closed source. On the D10
chat (row 1022, chat@2552, L=1276) the two choices give very different dQ errors on the worst
row (bf16 P ~3.5x the floor, fp32 P ~1.1x the floor), so the backend's dQ error there shows
which choice it makes.

Rule (same as the harness): err = max |x - truth| over the chat;
  bound = 2 * max |floor - truth| + 2 * max |(t + 0.3) - 0.3 - t|  (t in fp32)
  ratio = err / bound, PASS if <= 1.

Usage (box, GPU 7):  CUDA_VISIBLE_DEVICES=7 ~/envv/bin/python crosslib_chat_check.py <export_dir> [--flex]
Only reads the export directory; writes nothing.
"""
import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def load(d, name, dtype, shape):
    n = 1
    for s in shape:
        n *= s
    t = torch.from_file(f"{d}/{name}", shared=False, size=n, dtype=dtype)
    return t.reshape(shape)


def rule(x, truth, floor):
    """Harness rule on one tensor of the chat: (err, floor_err, ratio)."""
    x = x.double().cpu()
    err = (x - truth).abs().max().item()
    ferr = (floor.double() - truth).abs().max().item()
    t32 = truth.float()
    rnd = (((t32 + 0.3) - 0.3) - t32).abs().max().item()
    bound = 2.0 * ferr + 2.0 * rnd
    return err, ferr, (err / bound if bound > 0 else float("inf"))


def main():
    d = sys.argv[1]
    use_flex = "--flex" in sys.argv[2:]
    m = json.load(open(f"{d}/meta.json"))
    L, Hq, Hkv, hd = m["L"], m["Hq"], m["Hkv"], m["hd"]
    G = Hq // Hkv
    wh, wt, wd = m["worst"]["head"], m["worst"]["token"], m["worst"]["dim"]
    q = load(d, "q.bf16", torch.bfloat16, (1, Hq, L, hd))
    k = load(d, "k.bf16", torch.bfloat16, (1, Hkv, L, hd))
    v = load(d, "v.bf16", torch.bfloat16, (1, Hkv, L, hd))
    dO = load(d, "dO.bf16", torch.bfloat16, (1, Hq, L, hd))
    truth = {n: load(d, f"truth_{n}.f64", torch.float64, (1, Hq if n in ("O", "dQ") else Hkv, L, hd))
             for n in ("O", "dQ", "dK", "dV")}
    floor = {n: load(d, f"floor_{n}.f32", torch.float32, (1, Hq if n in ("O", "dQ") else Hkv, L, hd))
             for n in ("O", "dQ", "dK", "dV")}
    ours = {n: load(d, f"ours_{n}.bf16", torch.bfloat16, (1, Hq if n in ("O", "dQ") else Hkv, L, hd))
            for n in ("O", "dQ", "dK", "dV")}

    print(f"chat: D10 row {m['d10_row']}, start {m['t0']}, L={L}, heads {Hq}/{Hkv}, hd {hd}")
    print(f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, GPU {torch.cuda.get_device_name(0)}")
    print(f"worst element (from the harness): head {wh}, chat token {wt}, dim {wd}; "
          f"truth dQ there {truth['dQ'][0, wh, wt, wd].item():.6f}")
    print(f"{'backend':38s} {'O ratio':>8s} {'dQ ratio':>9s} {'dK ratio':>9s} {'dV ratio':>9s}   dQ at worst element")

    def report(name, outs, note=""):
        cells = []
        for n in ("O", "dQ", "dK", "dV"):
            if outs.get(n) is None:
                cells.append(f"{'-':>8s}")
                continue
            _, _, r = rule(outs[n], truth[n], floor[n])
            cells.append(f"{r:8.3f}{'*' if r > 1 else ' '}")
        w = outs["dQ"][0, wh, wt, wd].float().item() if outs.get("dQ") is not None else float("nan")
        print(f"{name:38s} {' '.join(cells)}   {w:.6f}  {note}")

    report("ours: v62 + Vj1d (exported)", ours)

    def run_sdpa(backend):
        """Native GQA first (enable_gqa); if the backend refuses it, repeat K/V to Hq heads.
        dK/dV are only judged for native GQA (repeated heads would add our own group sum)."""
        for native in (True, False):
            qq = q.cuda().requires_grad_()
            if native:
                kk, vv = k.cuda().requires_grad_(), v.cuda().requires_grad_()
                ka, va = kk, vv
            else:
                kk, vv = k.cuda().requires_grad_(), v.cuda().requires_grad_()
                ka = kk.repeat_interleave(G, dim=1)
                va = vv.repeat_interleave(G, dim=1)
            try:
                with sdpa_kernel(backend):
                    o = F.scaled_dot_product_attention(qq, ka, va, is_causal=True, enable_gqa=native)
                o.backward(dO.cuda())
                torch.cuda.synchronize()
            except RuntimeError as e:
                if native:
                    continue
                return None, f"not available: {str(e).splitlines()[0][:90]}"
            outs = {"O": o.detach(), "dQ": qq.grad.detach(),
                    "dK": kk.grad.detach() if native else None, "dV": vv.grad.detach() if native else None}
            return outs, "native GQA" if native else "K/V repeated to Hq heads (dK/dV not judged)"
        return None, "not available"

    for name, be in (("cuDNN (SDPA CUDNN_ATTENTION)", SDPBackend.CUDNN_ATTENTION),
                     ("FA2 in PyTorch (SDPA FLASH_ATTENTION)", SDPBackend.FLASH_ATTENTION),
                     ("memory-efficient (SDPA EFFICIENT)", SDPBackend.EFFICIENT_ATTENTION),
                     ("math, bf16 (SDPA MATH)", SDPBackend.MATH)):
        outs, note = run_sdpa(be)
        if outs is None:
            print(f"{name:38s} {note}")
        else:
            report(name, outs, note)

    if use_flex:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        qq, kk, vv = q.cuda().requires_grad_(), k.cuda().requires_grad_(), v.cuda().requires_grad_()
        bm = create_block_mask(lambda b, h, i, j: j <= i, 1, 1, L, L, device="cuda")
        o = torch.compile(flex_attention)(qq, kk, vv, block_mask=bm, enable_gqa=True)
        o.backward(dO.cuda())
        torch.cuda.synchronize()
        report("Flex (compiled)", {"O": o.detach(), "dQ": qq.grad.detach(), "dK": kk.grad.detach(),
                                   "dV": vv.grad.detach()}, "native GQA")
    print("(* = above the harness limit; ratio = err / (2 x floor err + fp32 rounding term))")


if __name__ == "__main__":
    main()
