#!/usr/bin/env python3
# CPU emulation of the attention backward's rounding points (2026-10-12): what the two precision items cost.
# One KV head shared by G = 3 query heads (our 18/6 GQA), causal, head dim 128, bf16 inputs.
# Truth: exact fp64 gradients of the attention function at the same bf16 inputs.
# Kernel emulation (all variants share it):
#   forward outputs as the kernel gets them: LSE fp32, O rounded to bf16, D = rowsum(dO * O_bf16) in fp32
#   GEMMs: bf16 x bf16 products are exact in fp32; the fp32 accumulation is emulated as fp64 then rounded to
#   fp32 (best case, the same for every variant); exp in fp32 (the kernels use ex2.approx: the same for all).
# Variants (the only differences):
#   V0 today     dS = bf16(P) * (dP - D)   [GQA_bwd_vj1d_sm90.cu:338-347]; per-head dK/dV rounded to bf16, then
#                summed in fp32 and rounded again [gqa_dkdv_greduce, :408-427]
#   V1           dS = P_fp32 * (dP - D)    [FA3 mainloop_bwd_sm90_tma_gmma_ws.hpp:868,893; FA2 flash_bwd_kernel.h:591]
#   V2           per-head dK/dV summed in fp32, rounded once [FA3 epilogue_bwd.hpp:480,511 + postprocess]
#   V3           V1 + V2 (= FA3's arithmetic)
# In every variant P and dS are rounded to bf16 before they enter a GEMM (tensor cores take bf16), and the final
# dQ/dK/dV are bf16 tensors. "floor" = the error of rounding the exact answer to bf16, which no kernel can beat.
import sys
import numpy as np

def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)

f32 = lambda x: np.asarray(x, dtype=np.float32)

def run(T, G, qs, ks, vs, seed):
    D_ = 128
    rng = np.random.default_rng(seed)
    Q = bf16(rng.standard_normal((G, T, D_)) * qs); K = bf16(rng.standard_normal((T, D_)) * ks)
    V = bf16(rng.standard_normal((T, D_)) * vs); dO = bf16(rng.standard_normal((G, T, D_)))
    sc = 1.0 / np.sqrt(D_)
    causal = np.tril(np.ones((T, T), dtype=bool))
    Q64, K64, V64, dO64 = (x.astype(np.float64) for x in (Q, K, V, dO))
    # ---- truth (fp64)
    S = np.einsum("gid,jd->gij", Q64, K64) * sc
    S = np.where(causal, S, -np.inf)
    m = S.max(-1, keepdims=True); E = np.exp(S - m); L = E.sum(-1, keepdims=True)
    P = E / L; lse = (m + np.log(L))[..., 0]
    O = np.einsum("gij,jd->gid", P, V64)
    dP = np.einsum("gid,jd->gij", dO64, V64)
    Dt = (dO64 * O).sum(-1, keepdims=True)
    dS = P * (dP - Dt)
    t_dQ = np.einsum("gij,jd->gid", dS, K64) * sc
    t_dK = np.einsum("gij,gid->jd", dS, Q64) * sc
    t_dV = np.einsum("gij,gid->jd", P, dO64)
    # ---- kernel emulation, shared part
    LSE = f32(lse); Ob = bf16(O)
    Dk = f32((dO64 * Ob.astype(np.float64)).sum(-1, keepdims=True))
    S32 = f32(np.einsum("gid,jd->gij", Q64, K64))
    P32 = f32(np.exp(S32.astype(np.float64) * sc - LSE[..., None].astype(np.float64)))
    P32 = np.where(causal, P32, np.float32(0))
    dP32 = f32(dP)
    Pb = bf16(P32)
    out = {}
    for name, ds_from_fp32, head_fp32 in (("V0 today", False, False), ("V1 dS from fp32 P", True, False),
                                          ("V2 fp32 head sum", False, True), ("V3 both (FA3)", True, True)):
        Pin = P32 if ds_from_fp32 else Pb
        dS32 = f32(Pin.astype(np.float64) * (dP32.astype(np.float64) - Dk.astype(np.float64)))
        dSb = bf16(dS32).astype(np.float64)
        dQ = bf16(f32(np.einsum("gij,jd->gid", dSb, K64) * sc))
        dKh = f32(np.einsum("gij,gid->gjd", dSb, Q64) * sc)          # per query head, fp32 accumulators
        dVh = f32(np.einsum("gij,gid->gjd", Pb.astype(np.float64), dO64))
        if head_fp32:
            dK = bf16(f32(dKh.astype(np.float64).sum(0))); dV = bf16(f32(dVh.astype(np.float64).sum(0)))
        else:
            dK = bf16(f32(bf16(dKh).astype(np.float64).sum(0))); dV = bf16(f32(bf16(dVh).astype(np.float64).sum(0)))
        out[name] = (dQ, dK, dV)
    truth = (t_dQ, t_dK, t_dV)
    floor = [np.abs(bf16(t).astype(np.float64) - t).max() for t in truth]
    return out, truth, floor

if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    for label, qs, ks, vs in (("normal inputs N(0,1)", 1, 1, 1), ("large inputs q x5, k x3, v x3 (G13)", 5, 3, 3)):
        for seed in (1, 2):
            out, truth, floor = run(T, 3, qs, ks, vs, seed)
            print(f"\n== {label}, T={T}, 3 query heads on 1 KV head, seed {seed}")
            print(f"   {'variant':22s} {'dQ max err':>12s} {'x floor':>8s} {'dK max err':>12s} {'x floor':>8s} {'dV max err':>12s} {'x floor':>8s}")
            print(f"   {'floor (bf16 of truth)':22s} {floor[0]:12.3e} {1:8.2f} {floor[1]:12.3e} {1:8.2f} {floor[2]:12.3e} {1:8.2f}")
            rfl = [np.sqrt(((bf16(t).astype(np.float64) - t) ** 2).mean()) for t in truth]
            for name, (dQ, dK, dV) in out.items():
                e = [np.abs(x.astype(np.float64) - t).max() for x, t in zip((dQ, dK, dV), truth)]
                r = [np.sqrt(((x.astype(np.float64) - t) ** 2).mean()) for x, t in zip((dQ, dK, dV), truth)]
                print(f"   {name:22s} " + " ".join(f"{e[i]:12.3e} {e[i]/floor[i]:8.2f}" for i in range(3))
                      + "   RMS x floor: " + " ".join(f"{r[i]/rfl[i]:5.2f}" for i in range(3)))
