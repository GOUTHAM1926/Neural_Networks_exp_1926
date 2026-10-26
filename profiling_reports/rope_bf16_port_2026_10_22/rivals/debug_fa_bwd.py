# Why do flashattn and quack show a wrong backward in rival_bench.py (err dx ~1300, rel dw ~1.4)? (2026-10-23)
# Isolate the two pieces they share or not: FlashAttention's Triton RoPE (apply_rotary_emb, used by both) and the norms
# (flash_attn rms_norm_fn / quack rmsnorm), each against an fp64 reference, forward and backward.
# usage: CUDA_VISIBLE_DEVICES=3 python -u debug_fa_bwd.py ./vendor
import sys
import torch
sys.path.insert(0, sys.argv[1])
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn.ops.triton.layer_norm import rms_norm_fn

torch.manual_seed(0)
B, T, H, D, EPS = 2, 4096, 6, 128, 1e-6
dev = "cuda"
inv = 1.0 / (50000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]
cos64, sin64 = ang.cos().to(dev), ang.sin().to(dev)
x = torch.randn(B, T, H, D, device=dev).to(torch.bfloat16)
dz = torch.randn(B, T, H, D, device=dev).to(torch.bfloat16)


def ref_rope(xx):
    a, b = xx[..., :D // 2], xx[..., D // 2:]
    c, s = cos64[None, :, None, :], sin64[None, :, None, :]
    return torch.cat([a * c - b * s, a * s + b * c], -1)


def report(name, y, gx, yr, gr):
    ulp = lambda r: (r.to(torch.bfloat16).double() - r).abs().max()
    print(f"{name:42s} y err {((y.double() - yr).abs().max() / ulp(yr)).item():8.2f}   dx err {((gx.double() - gr).abs().max() / ulp(gr)).item():8.2f}")


# 1) RoPE only
for cdt in (torch.float32, torch.bfloat16):
    xr = x.double().requires_grad_(True); yr = ref_rope(xr); (gr,) = torch.autograd.grad(yr, xr, dz.double())
    xx = x.clone().requires_grad_(True)
    y = apply_rotary_emb(xx, cos64.to(cdt), sin64.to(cdt))
    (gx,) = torch.autograd.grad(y, xx, dz)
    report(f"FA apply_rotary_emb, cos/sin {str(cdt)[6:]}", y, gx, yr.detach(), gr)

# 2) RMSNorm only (fp32 weight, as in rival_bench)
w = (1.0 + torch.randint(-16, 17, (D,)).float() / 64.0).to(dev)
xr = x.double().requires_grad_(True); wr = w.double().requires_grad_(True)
yr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + EPS) * wr
gr, gwr = torch.autograd.grad(yr, (xr, wr), dz.double())
xx = x.clone().requires_grad_(True); ww = w.clone().requires_grad_(True)
y = rms_norm_fn(xx, ww, None, eps=EPS)
gx, gw = torch.autograd.grad(y, (xx, ww), dz)
report("FA rms_norm_fn (fp32 weight)", y, gx, yr.detach(), gr)
print(f"{'':42s} dw rel {((gw.double() - gwr).abs().max() / gwr.abs().max()).item():.2e}   y dtype {y.dtype}")
