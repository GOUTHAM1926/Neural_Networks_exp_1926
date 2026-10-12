#!/usr/bin/env python3
"""crosslib_bf16P_stats.py -- does a backend compute dS with an fp32 P or a bf16 P?

Statistical test on many rows that have only 2 visible keys (token 1 of each sequence), where a
bf16 P in dS = P (dP - D) leaves a visible error in dQ when dP - D is large. Inputs: 1024
sequences x 64 tokens, 18/6 heads, hd 128, q,k ~ N(0,1), v and dO ~ 4 * N(0,1) (bf16).
Each backend's dQ on the token-1 rows is compared with an fp64 truth, in bf16 steps (ulp of the
truth). Two models give the signatures, both built from the fp64 truth:
  model fp32-P : dS = bf16( P      * (dP - D) )     (FA2/FA3/FA4/Flex arithmetic)
  model bf16-P : dS = bf16( bf16(P)* (dP - D) )     (Vj1d / PyTorch memory-efficient arithmetic)
  with D = dO . bf16(O), dQ = bf16(scale * sum_j dS_j K_j).
Usage (box): CUDA_VISIBLE_DEVICES=7 ~/envv/bin/python crosslib_bf16P_stats.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(1234)
B, L, Hq, Hkv, D = 1024, 64, 18, 6, 128
G = Hq // Hkv
scale = 1.0 / D ** 0.5
dev = "cuda"
q = torch.randn(B, Hq, L, D, device=dev).bfloat16()
k = torch.randn(B, Hkv, L, D, device=dev).bfloat16()
v = (4 * torch.randn(B, Hkv, L, D, device=dev)).bfloat16()
dO = (4 * torch.randn(B, Hq, L, D, device=dev)).bfloat16()

# fp64 truth (causal, GQA by repeating K/V)
qd, kd, vd = q.double().requires_grad_(), k.double(), v.double()
kr, vr = kd.repeat_interleave(G, 1), vd.repeat_interleave(G, 1)
s = (qd @ kr.transpose(-1, -2)) * scale
s = s.masked_fill(torch.ones(L, L, device=dev, dtype=torch.bool).triu(1), float("-inf"))
p = torch.softmax(s, -1)
o = p @ vr
o.backward(dO.double())
dq_t = qd.grad.detach()
row = 1                                  # token 1: exactly 2 visible keys
t = dq_t[:, :, row, :]


def bf16(x):
    return x.bfloat16().double()


def steps(dq):
    """|dq - truth| in bf16 steps of the truth, over the token-1 rows."""
    x = dq.double()[:, :, row, :]
    a = t.abs().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(a)) - 7)
    return ((x - t).abs() / ulp).flatten()


def model(bf16_P):
    with torch.no_grad():
        P = p[:, :, row, : row + 1]                                   # [B, Hq, 2]
        Ob = bf16(o[:, :, row, :])                                    # the kernel stores O in bf16
        dP = torch.einsum("bhd,bhjd->bhj", dO.double()[:, :, row, :], vr[:, :, : row + 1, :])
        Drow = (dO.double()[:, :, row, :] * Ob).sum(-1, keepdim=True)
        Pu = bf16(P) if bf16_P else P
        dS = bf16(Pu * (dP - Drow))
        dq = bf16(scale * torch.einsum("bhj,bhjd->bhd", dS, kr[:, :, : row + 1, :]))
        out = torch.zeros_like(dq_t)
        out[:, :, row, :] = dq
        return out


def row_steps(dq):
    """|dq - truth| in bf16 steps of the ROW's largest |truth| (robust to near-zero elements)."""
    x = dq.double()[:, :, row, :]
    rmax = t.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(rmax)) - 7)
    return ((x - t).abs() / ulp).flatten()


M32 = None
M16 = None


def report(name, dq):
    st = row_steps(dq).float()
    x = dq.double()[:, :, row, :]
    m32 = (x == M32[:, :, row, :]).double().mean().item() * 100
    m16 = (x == M16[:, :, row, :]).double().mean().item() * 100
    print(f"{name:40s} row-steps mean {st.mean():6.4f} p99 {st.quantile(0.99):6.3f} max {st.max():6.3f}"
          f" | equal to fp32-P model {m32:6.2f}%  bf16-P model {m16:6.2f}%")


print(f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, {torch.cuda.get_device_name(0)}")
print(f"{B} seqs x {L} tokens, heads {Hq}/{Hkv}, hd {D}; dQ error on the token-1 rows (2 keys), in bf16 steps")
M32, M16 = model(False), model(True)
report("model fp32-P (FA-style arithmetic)", M32)
report("model bf16-P (Vj1d / mem-eff arithmetic)", M16)
for name, be in (("cuDNN", SDPBackend.CUDNN_ATTENTION), ("FA2 in PyTorch", SDPBackend.FLASH_ATTENTION),
                 ("memory-efficient", SDPBackend.EFFICIENT_ATTENTION), ("math bf16", SDPBackend.MATH)):
    qq = q.clone().requires_grad_()
    kk, vv = k.clone().requires_grad_(), v.clone().requires_grad_()
    try:
        with sdpa_kernel(be):
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=True)
        out.backward(dO)
        torch.cuda.synchronize()
        report(name, qq.grad)
    except RuntimeError as e:
        print(f"{name:38s} not available: {str(e).splitlines()[0][:100]}")
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    qq = q.clone().requires_grad_()
    kk, vv = k.clone().requires_grad_(), v.clone().requires_grad_()
    bm = create_block_mask(lambda b, h, i, j: j <= i, None, None, L, L, device=dev)
    out = torch.compile(flex_attention)(qq, kk, vv, block_mask=bm, enable_gqa=True)
    out.backward(dO)
    torch.cuda.synchronize()
    report("Flex (compiled)", qq.grad)
except Exception as e:
    print(f"{'Flex (compiled)':38s} failed: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
